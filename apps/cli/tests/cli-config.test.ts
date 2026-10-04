import { describe, expect, it } from "vitest";
import { CliConfigError, loadCliConfig, scheduleFireSlots } from "../src/config";
import { buildPaperFilterRequest } from "@arxiv-daily/core";

const minimalToml = `
schema_version = 1
vault_root = "/vault"
cache_dir = "/vault/.cache/arxiv-daily"

[llm]
api_key = "sk-test"
base_url = "https://api.example.com/v1"
model = "m1"

[embedding]
mode = "remote"
base_url = "https://api.openai.com/v1"
api_key = "sk-embed"
model = "text-embedding-3-small"
dimension = 1536

[arxiv]
categories = ["cs.LG"]
timezone = "UTC"

[[arxiv.topics]]
name = "ML"
tag = "ml"
description = "machine learning"
detail = true

[output]
summary_language = "en"
daily_dir = "arxiv-daily/daily"
papers_dir = "arxiv-daily/papers"
link_style = "relative"

[email]
enabled = false
mode = "self"
to = "a@b.com"
api_key = "re_x"

[schedule]
enabled = false
on = "09:30"
interval_hours = 0
until = "18:00"
weekdays_only = true

[advanced]
log_level = "info"
`;

describe("CLI config loader (TOML / XDG)", () => {
  it("errors when config file is missing", async () => {
    await expect(
      loadCliConfig({
        configPath: "/no/such/config.toml",
        readText: async () => {
          const err = new Error("missing") as NodeJS.ErrnoException;
          err.code = "ENOENT";
          throw err;
        },
      }),
    ).rejects.toBeInstanceOf(CliConfigError);
  });

  it("loads TOML and maps snake_case fields", async () => {
    const cfg = await loadCliConfig({
      configPath: "/home/u/.config/arxiv-daily/config.toml",
      readText: async () => minimalToml,
    });

    expect(cfg.configPath).toBe("/home/u/.config/arxiv-daily/config.toml");
    expect(cfg.vaultRoot).toBe("/vault");
    expect(cfg.cacheDir).toBe("/vault/.cache/arxiv-daily");
    expect(cfg.settings.llm.apiKey).toBe("sk-test");
    expect(cfg.settings.llm.baseUrl).toBe("https://api.example.com/v1");
    expect(cfg.settings.embedding).toMatchObject({
      mode: "remote",
      baseUrl: "https://api.openai.com/v1",
      apiKey: "sk-embed",
      model: "text-embedding-3-small",
      dimension: 1536,
    });
    expect(cfg.settings.arxiv.categories).toEqual(["cs.LG"]);
    expect(cfg.settings.arxiv.category).toBe("cs.LG");
    expect(cfg.settings.arxiv.topics[0]?.tag).toBe("ml");
    expect(cfg.settings.output.linkStyle).toBe("relative");
    expect(cfg.settings.output.summaryLanguage).toBe("en");
    expect(cfg.settings.output.maxDailyPapers).toBe(20);
    expect(cfg.settings.email.to).toBe("a@b.com");
    expect(cfg.settings.detailSelection.profile).toBe("balanced");
    expect(cfg.scheduleIntent.on).toBe("09:30");
    expect(cfg.scheduleIntent.intervalHours).toBe(0);
  });

  it("clamps legacy request delays below the safe runtime floor", async () => {
    const cfg = await loadCliConfig({
      configPath: "/cfg.toml",
      readText: async () => `${minimalToml}\nrequest_delay_ms = 1000\n`,
    });

    expect(cfg.settings.advanced.requestDelayMs).toBe(3000);
  });

  it.each([1, 35, Number.MAX_SAFE_INTEGER])("loads output.max_daily_papers = %s", async (limit) => {
    const cfg = await loadCliConfig({
      configPath: "/cfg.toml",
      readText: async () => minimalToml.replace("[output]", `[output]\nmax_daily_papers = ${limit}`),
    });
    expect(cfg.settings.output.maxDailyPapers).toBe(limit);
  });

  it.each(["0", "-1", "1.5", "nan", "inf", '"20"', "true", "[20]", "9.007199254740992e15"])(
    "rejects invalid output.max_daily_papers = %s",
    async (value) => {
      await expect(loadCliConfig({
        configPath: "/cfg.toml",
        readText: async () => minimalToml.replace("[output]", `[output]\nmax_daily_papers = ${value}`),
      })).rejects.toThrow(/invalid output.max_daily_papers/);
    },
  );

  it("rejects invalid request delay configuration", async () => {
    await expect(
      loadCliConfig({
        configPath: "/cfg.toml",
        readText: async () => `${minimalToml}\nrequest_delay_ms = -1\n`,
      }),
    ).rejects.toThrow("invalid advanced.request_delay_ms");
  });

  it("ignores ARXIV_DAILY env for settings", async () => {
    const cfg = await loadCliConfig({
      configPath: "/cfg.toml",
      env: { ARXIV_DAILY_API_KEY: "from-env" },
      readText: async () => minimalToml,
    });
    expect(cfg.settings.llm.apiKey).toBe("sk-test");
  });

  it("expands schedule slots with interval_hours", () => {
    expect(
      scheduleFireSlots({
        enabled: true,
        on: "09:30",
        intervalHours: 4,
        until: "18:00",
        weekdaysOnly: true,
      }),
    ).toEqual(["09:30", "13:30", "17:30"]);
  });

  it("rejects a reversed recurring schedule while loading TOML", async () => {
    const text = minimalToml.replace('on = "09:30"', 'on = "18:00"')
      .replace('until = "18:00"', 'until = "09:00"')
      .replace("interval_hours = 0", "interval_hours = 1");
    await expect(loadCliConfig({ configPath: "/cfg.toml", readText: async () => text }))
      .rejects.toThrow("schedule.until");
  });
});

/**
 * P3: the CLI and the plugin classify through the same core path, so the CLI
 * must reach the filter with directions too. A TOML topic written the legacy
 * way — one `description` line, no `directions` — has to arrive as a topic the
 * filter can judge against (ADR 0003 keeps the two products consistent).
 */
describe("CLI topics reach the direction-driven filter", () => {
  it("turns a legacy description-only TOML topic into a classifiable direction", async () => {
    const cfg = await loadCliConfig({
      configPath: "/cfg.toml",
      readText: async () => minimalToml,
    });

    const request = buildPaperFilterRequest(
      [{ id: "2609.00001", title: "T", authors: "A", abstract: "B" }],
      cfg.settings.arxiv,
    );

    expect(request.messages[0]!.content).toContain("- ml:\n  - ml#1: machine learning");
    expect(request.identity.directions).toMatchObject([
      { ref: "ml#1", tag: "ml", text: "machine learning" },
    ]);
  });

  it("carries an explicit TOML direction list through in order", async () => {
    const toml = minimalToml.replace(
      'description = "machine learning"',
      'directions = [{ id = "d1", text = "graph neural networks" }, { id = "d2", text = "diffusion models" }]',
    );
    const cfg = await loadCliConfig({ configPath: "/cfg.toml", readText: async () => toml });

    const request = buildPaperFilterRequest(
      [{ id: "2609.00001", title: "T", authors: "A", abstract: "B" }],
      cfg.settings.arxiv,
    );

    expect(request.identity.directions.map(({ ref, text }) => ({ ref, text }))).toEqual([
      { ref: "ml#1", text: "graph neural networks" },
      { ref: "ml#2", text: "diffusion models" },
    ]);
  });
});

it("loads custom details, local parser and workbench schedule without changing cron", async () => {
  const config = await loadCliConfig({ readText: async () => `${minimalToml}
[detail_selection]
profile = "custom"
normal_threshold = 80
exceptional_threshold = 94
soft_limit = 2
[pdf_parser_sidecar]
enabled = true
capabilities_url = "http://127.0.0.1:5010/cap"
parse_url = "http://127.0.0.1:5010/parse"
[workbench_schedule]
enabled = true
run_at_local = "08:30"
run_until_local = "17:30"
tick_interval_min = 13
` });
  expect(config.settings.detailSelection).toEqual({ profile: "custom", normalThreshold: 80, exceptionalThreshold: 94, softLimit: 2 });
  expect(config.settings.pdfParserSidecar).toMatchObject({ enabled: true, parseUrl: "http://127.0.0.1:5010/parse" });
  expect(config).toMatchObject({ workbenchSchedule: { enabled: true, tickIntervalMin: 13 } });
  expect(config.settings.schedule.enabled).toBe(false);
});

it('keeps normalized direction identities stable across read-only legacy config loads', async () => {
 const options={configPath:'/cfg.toml',readText:async()=>minimalToml};
 const first=await loadCliConfig(options), second=await loadCliConfig(options);
 expect(first.settings.arxiv.topics).toEqual(second.settings.arxiv.topics);
});
