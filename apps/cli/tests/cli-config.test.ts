import { describe, expect, it } from "vitest";
import { CliConfigError, loadCliConfig, scheduleFireSlots } from "../src/config";

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
});
