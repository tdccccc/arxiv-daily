import { describe, expect, it } from "vitest";
import {
  CliConfigError,
  loadCliConfig,
} from "../src/cli/config";

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
    expect(cfg.settings.llm.apiKey).toBe("key");
    expect(cfg.settings.arxiv.categories).toEqual(["astro-ph"]);
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
        ARXIV_DAILY_LINK_STYLE: "relative",
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
    expect(cfg.settings.llm.apiKey).toBe("env-key");
    expect(cfg.settings.llm.model).toBe("env-model");
    expect(cfg.settings.arxiv.category).toBe("astro-ph");
    expect(cfg.settings.arxiv.categories).toEqual(["astro-ph", "cs.LG"]);
    expect(cfg.settings.arxiv.topics[0].id).toBe("env-topic");
    expect(cfg.settings.output.dailyDir).toBe("env-daily");
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
  });
});
