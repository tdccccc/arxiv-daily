import { describe, expect, it } from "vitest";
import {
  CliConfigError,
  loadCliConfig,
} from "../src/config";

describe("CLI config loader", () => {
  it("uses defaults when the default config file is missing", async () => {
    const cfg = await loadCliConfig({
      cwd: "/workspace",
      env: { ARXIV_DAILY_API_KEY: "key" },
      readText: async () => {
        const err = new Error("missing") as NodeJS.ErrnoException;
        err.code = "ENOENT";
        throw err;
      },
    });

    expect(cfg.configPath).toBeNull();
    expect(cfg.vaultRoot).toBe("/workspace");
    expect(cfg.cacheDir).toBe("/workspace/.arxiv-daily/cache");
    expect(cfg.linkStyle).toBe("wikilink");
    expect(cfg.settings.output.linkStyle).toBe("wikilink");
    expect(cfg.settings.output.summaryLanguage).toBe("zh");
    expect(cfg.settings.llm.apiKey).toBe("key");
    expect(cfg.settings.arxiv.categories).toEqual(["astro-ph"]);
    expect(cfg.settings.detailSelection).toEqual({
      profile: "balanced",
      normalThreshold: 75,
      exceptionalThreshold: 92,
      softLimit: 3,
    });
  });

  it("loads JSON config and resolves runtime paths from cwd", async () => {
    const cfg = await loadCliConfig({
      cwd: "/workspace",
      configPath: "custom.json",
      env: {},
      readText: async () =>
        JSON.stringify({
          vaultRoot: "vault",
          cacheDir: ".cache/arxiv",
          linkStyle: "relative",
          llm: {
            provider: "openai",
            apiKey: "file-key",
            model: "gpt-test",
          },
          arxiv: {
            category: "cs.LG",
            topics: [
              {
                id: "t1",
                name: "ML",
                tag: "ml",
                description: "machine learning",
                detail: true,
              },
            ],
          },
          output: {
            dailyDir: "daily",
            papersDir: "papers",
            summaryLanguage: "en",
          },
        }),
    });

    expect(cfg.configPath).toBe("/workspace/custom.json");
    expect(cfg.vaultRoot).toBe("/workspace/vault");
    expect(cfg.cacheDir).toBe("/workspace/.cache/arxiv");
    expect(cfg.linkStyle).toBe("relative");
    expect(cfg.settings.output.linkStyle).toBe("relative");
    expect(cfg.settings.llm.provider).toBe("openai");
    expect(cfg.settings.llm.apiKey).toBe("file-key");
    expect(cfg.settings.llm.model).toBe("gpt-test");
    expect(cfg.settings.arxiv.category).toBe("cs.LG");
    expect(cfg.settings.arxiv.categories).toEqual(["cs.LG"]);
    expect(cfg.settings.arxiv.topics).toHaveLength(1);
    expect(cfg.settings.output.dailyDir).toBe("daily");
    expect(cfg.settings.output.summaryLanguage).toBe("en");
  });

  it("supports nested and top-level JSON detail selection settings", async () => {
    const nested = await loadCliConfig({
      cwd: "/workspace",
      configPath: "nested.json",
      env: {},
      readText: async () => JSON.stringify({
        settings: {
          detailSelection: {
            profile: "custom",
            normalThreshold: 71,
            exceptionalThreshold: 91,
            softLimit: 4,
          },
        },
      }),
    });
    expect(nested.settings.detailSelection).toEqual({
      profile: "custom",
      normalThreshold: 71,
      exceptionalThreshold: 91,
      softLimit: 4,
    });

    const topLevel = await loadCliConfig({
      cwd: "/workspace",
      configPath: "top-level.json",
      env: {},
      readText: async () => JSON.stringify({
        settings: { detailSelection: { profile: "conservative" } },
        detailSelection: { profile: "broad" },
      }),
    });
    expect(topLevel.settings.detailSelection).toEqual({
      profile: "broad",
      normalThreshold: 65,
      exceptionalThreshold: 88,
      softLimit: 5,
    });
  });

  it("canonicalizes conflicting numeric values under a named JSON profile", async () => {
    const cfg = await loadCliConfig({
      cwd: "/workspace",
      configPath: "config.json",
      env: {},
      readText: async () => JSON.stringify({
        detailSelection: {
          profile: "conservative",
          normalThreshold: 1,
          exceptionalThreshold: 2,
          softLimit: 20,
        },
      }),
    });
    // Numeric fields in a CLI/config layer are explicit overrides, so they are
    // preserved only under the custom label.
    expect(cfg.settings.detailSelection).toEqual({
      profile: "custom",
      normalThreshold: 1,
      exceptionalThreshold: 2,
      softLimit: 20,
    });
  });

  it("sanitizes malformed JSON detail selection values", async () => {
    const cfg = await loadCliConfig({
      cwd: "/workspace",
      configPath: "config.json",
      env: {},
      readText: async () => JSON.stringify({
        detailSelection: {
          profile: "custom",
          normalThreshold: 110,
          exceptionalThreshold: -1,
          softLimit: 99.4,
        },
      }),
    });
    expect(cfg.settings.detailSelection).toEqual({
      profile: "custom",
      normalThreshold: 100,
      exceptionalThreshold: 100,
      softLimit: 20,
    });
  });

  it("lets env override file settings", async () => {
    const cfg = await loadCliConfig({
      cwd: "/workspace",
      configPath: "config.json",
      env: {
        ARXIV_DAILY_API_KEY: "env-key",
        ARXIV_DAILY_MODEL: "env-model",
        ARXIV_DAILY_CATEGORIES: "astro-ph,cs.LG,astro-ph",
        ARXIV_DAILY_DAILY_DIR: "env-daily",
        ARXIV_DAILY_DETAIL_PROFILE: "custom",
        ARXIV_DAILY_DETAIL_NORMAL_THRESHOLD: "82",
        ARXIV_DAILY_DETAIL_EXCEPTIONAL_THRESHOLD: "70",
        ARXIV_DAILY_DETAIL_SOFT_LIMIT: "22",
        ARXIV_DAILY_LINK_STYLE: "relative",
        ARXIV_DAILY_SUMMARY_LANGUAGE: "en",
        ARXIV_DAILY_VAULT_ROOT: "/vault",
        ARXIV_DAILY_TOPICS_JSON: JSON.stringify([
          {
            id: "env-topic",
            name: "Env Topic",
            tag: "env-topic",
            description: "from env",
            detail: false,
          },
        ]),
      },
      readText: async () =>
        JSON.stringify({
          vaultRoot: "file-vault",
          linkStyle: "wikilink",
          llm: { apiKey: "file-key", model: "file-model" },
          arxiv: { categories: ["cs.CL"] },
          output: { dailyDir: "file-daily" },
        }),
    });

    expect(cfg.vaultRoot).toBe("/vault");
    expect(cfg.linkStyle).toBe("relative");
    expect(cfg.settings.output.linkStyle).toBe("relative");
    expect(cfg.settings.output.summaryLanguage).toBe("en");
    expect(cfg.settings.llm.apiKey).toBe("env-key");
    expect(cfg.settings.llm.model).toBe("env-model");
    expect(cfg.settings.arxiv.category).toBe("astro-ph");
    expect(cfg.settings.arxiv.categories).toEqual(["astro-ph", "cs.LG"]);
    expect(cfg.settings.arxiv.topics[0].id).toBe("env-topic");
    expect(cfg.settings.output.dailyDir).toBe("env-daily");
    expect(cfg.settings.detailSelection).toEqual({
      profile: "custom",
      normalThreshold: 82,
      exceptionalThreshold: 82,
      softLimit: 20,
    });
  });

  it("treats every numeric env override as custom, even when values equal a preset", async () => {
    const cfg = await loadCliConfig({
      cwd: "/workspace",
      env: {
        ARXIV_DAILY_DETAIL_PROFILE: "balanced",
        ARXIV_DAILY_DETAIL_NORMAL_THRESHOLD: "75",
        ARXIV_DAILY_DETAIL_EXCEPTIONAL_THRESHOLD: "92",
        ARXIV_DAILY_DETAIL_SOFT_LIMIT: "3",
      },
      readText: async () => {
        const err = new Error("missing") as NodeJS.ErrnoException;
        err.code = "ENOENT";
        throw err;
      },
    });
    expect(cfg.settings.detailSelection).toEqual({
      profile: "custom",
      normalThreshold: 75,
      exceptionalThreshold: 92,
      softLimit: 3,
    });
  });

  it("lets a profile-only env layer explicitly select its exact preset", async () => {
    const cfg = await loadCliConfig({
      cwd: "/workspace",
      configPath: "config.json",
      env: { ARXIV_DAILY_DETAIL_PROFILE: "broad" },
      readText: async () => JSON.stringify({
        detailSelection: {
          profile: "custom",
          normalThreshold: 81,
          exceptionalThreshold: 96,
          softLimit: 7,
        },
      }),
    });
    expect(cfg.settings.detailSelection).toEqual({
      profile: "broad",
      normalThreshold: 65,
      exceptionalThreshold: 88,
      softLimit: 5,
    });
  });

  it("safely falls back for malformed detail selection env values", async () => {
    const cfg = await loadCliConfig({
      cwd: "/workspace",
      env: {
        ARXIV_DAILY_DETAIL_PROFILE: "invalid",
        ARXIV_DAILY_DETAIL_NORMAL_THRESHOLD: "not-a-number",
        ARXIV_DAILY_DETAIL_EXCEPTIONAL_THRESHOLD: "Infinity",
        ARXIV_DAILY_DETAIL_SOFT_LIMIT: "many",
      },
      readText: async () => {
        const err = new Error("missing") as NodeJS.ErrnoException;
        err.code = "ENOENT";
        throw err;
      },
    });
    expect(cfg.settings.detailSelection).toEqual({
      profile: "balanced",
      normalThreshold: 75,
      exceptionalThreshold: 92,
      softLimit: 3,
    });
  });

  it("canonicalizes output directories", async () => {
    const cfg = await loadCliConfig({
      cwd: "/workspace",
      configPath: "config.json",
      env: {},
      readText: async () => JSON.stringify({
        output: { dailyDir: " 研究\\日报 ", papersDir: " 研究\\论文 " },
      }),
    });
    expect(cfg.settings.output.dailyDir).toBe("研究/日报");
    expect(cfg.settings.output.papersDir).toBe("研究/论文");
  });

  it.each([
    { dailyDir: "../escape", papersDir: "papers" },
    { dailyDir: ".obsidian/cache", papersDir: "papers" },
    { dailyDir: "CON/reports", papersDir: "papers" },
    { dailyDir: "same", papersDir: "same" },
    { dailyDir: "Café/Notes", papersDir: "CAFE\u0301/notes" },
  ])("rejects unsafe output configuration %#", async (output) => {
    await expect(loadCliConfig({
      cwd: "/workspace",
      configPath: "config.json",
      env: {},
      readText: async () => JSON.stringify({ output }),
    })).rejects.toBeInstanceOf(CliConfigError);
  });

  it("throws typed errors for invalid config", async () => {
    await expect(
      loadCliConfig({
        cwd: "/workspace",
        configPath: "bad.json",
        env: {},
        readText: async () => "{bad",
      }),
    ).rejects.toBeInstanceOf(CliConfigError);

    await expect(
      loadCliConfig({
        cwd: "/workspace",
        env: { ARXIV_DAILY_LINK_STYLE: "invalid" },
        readText: async () => {
          const err = new Error("missing") as NodeJS.ErrnoException;
          err.code = "ENOENT";
          throw err;
        },
      }),
    ).rejects.toThrow(/invalid linkStyle/);

    await expect(
      loadCliConfig({
        cwd: "/workspace",
        env: { ARXIV_DAILY_SUMMARY_LANGUAGE: "fr" },
        readText: async () => {
          const err = new Error("missing") as NodeJS.ErrnoException;
          err.code = "ENOENT";
          throw err;
        },
      }),
    ).rejects.toThrow(/invalid summaryLanguage/);
  });
});
