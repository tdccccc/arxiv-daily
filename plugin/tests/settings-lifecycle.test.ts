import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import ArxivDailyPlugin, { resolvePluginDir } from "../main.ts";
import { settingsAndStateFromPersistedData } from "../src/settings/load";

const pluginMainSource = readFileSync(resolve(process.cwd(), "main.ts"), "utf8");

const savedCategories = ["cs.CL", "stat.ML", "custom.quantum-ai"];

function persistedData(categories: string[]) {
  return {
    settings: {
      arxiv: {
        category: categories[0] ?? "cs.CL",
        categories,
        topics: [],
        timezone: "UTC",
      },
    },
    runState: {},
  };
}

describe("plugin directory resolution", () => {
  it("prioritizes the manifest directory", () => {
    expect(resolvePluginDir("custom/plugin", ".config", "arxiv-daily")).toBe(
      "custom/plugin",
    );
  });

  it("falls back to the vault configuration directory", () => {
    expect(resolvePluginDir(undefined, ".vault-config", "arxiv-daily")).toBe(
      ".vault-config/plugins/arxiv-daily",
    );
  });
});

describe("plugin settings reload lifecycle", () => {
  it("sanitizes detail selection immediately before saving", () => {
    const saveBody = pluginMainSource.match(
      /async saveSettings\(\): Promise<void> \{[\s\S]*?\n  \}/,
    )?.[0];
    expect(saveBody).toBeDefined();
    expect(saveBody!.indexOf("sanitizeDetailSelection")).toBeLessThan(
      saveBody!.indexOf("persistSettings"),
    );
  });

  it("passes configured detail selection policy to the production pipeline", () => {
    const configuredPolicy = {
      profile: "custom" as const,
      normalThreshold: 82,
      exceptionalThreshold: 97,
      softLimit: 6,
    };
    const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    Object.assign(plugin, {
      settings: {
        ...DEFAULT_SETTINGS,
        detailSelection: configuredPolicy,
      },
      logger: {},
      progress: {},
      host: { markupParser: {} },
      buildSharedDeps: () => ({
        llm: {},
        fetcher: {},
        paperFetcher: {},
        writer: {},
      }),
      buildPaperIndex: () => ({}),
    });

    const pipeline = (plugin as any).buildPipeline();

    expect((pipeline as any).deps.detailSelection).toEqual(configuredPolicy);
    expect((pipeline as any).deps.detailSelection).toBe(
      plugin.settings.detailSelection,
    );
  });

  it("logs persisted sanitation warnings after logger initialization", () => {
    const loggerInit = pluginMainSource.indexOf("this.logger = new Logger(");
    const warningLoop = pluginMainSource.indexOf("for (const warning of settingsWarnings)");
    expect(loggerInit).toBeGreaterThan(-1);
    expect(warningLoop).toBeGreaterThan(loggerInit);
    expect(pluginMainSource).toContain("this.logger.warn(`settings: ${warning}`)");
  });

  it("preserves saved non-default categories exactly across repeated reload migration", () => {
    const saved = persistedData(savedCategories);

    const firstLoad = settingsAndStateFromPersistedData(saved);
    const reloaded = settingsAndStateFromPersistedData({
      settings: firstLoad.settings,
      runState: firstLoad.runState,
    });

    expect(firstLoad.settings.arxiv.categories).toEqual(savedCategories);
    expect(reloaded.settings.arxiv.categories).toEqual(savedCategories);
    expect(saved.settings.arxiv.categories).toEqual(savedCategories);
  });

  it("preserves an explicitly empty category list across reload", () => {
    const firstLoad = settingsAndStateFromPersistedData(persistedData([]));
    const reloaded = settingsAndStateFromPersistedData({ settings: firstLoad.settings });

    expect(firstLoad.settings.arxiv.categories).toEqual([]);
    expect(reloaded.settings.arxiv.categories).toEqual([]);
  });

  it("migrates the legacy singular category only when categories is absent", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        arxiv: {
          category: "hep-th",
          topics: [],
          timezone: "UTC",
        },
      },
    });

    expect(loaded.settings.arxiv.categories).toEqual(["hep-th"]);
  });

  it("adds balanced detail selection settings to old persisted configs", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: { arxiv: persistedData(["cs.CL"]).settings.arxiv },
    });

    expect(loaded.settings.detailSelection).toEqual({
      profile: "balanced",
      normalThreshold: 75,
      exceptionalThreshold: 92,
      softLimit: 3,
    });
  });

  it("canonicalizes conflicting persisted values under a named profile", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        detailSelection: {
          profile: "conservative",
          normalThreshold: 12,
          exceptionalThreshold: 13,
          softLimit: 20,
        },
      },
    });

    expect(loaded.settings.detailSelection).toEqual({
      profile: "conservative",
      normalThreshold: 85,
      exceptionalThreshold: 95,
      softLimit: 1,
    });
  });

  it("sanitizes detail selection settings across save and reload", () => {
    const firstLoad = settingsAndStateFromPersistedData({
      settings: {
        detailSelection: {
          profile: "custom",
          normalThreshold: 101,
          exceptionalThreshold: 20,
          softLimit: 4.6,
        },
      },
    });
    const reloaded = settingsAndStateFromPersistedData({ settings: firstLoad.settings });

    expect(firstLoad.settings.detailSelection).toEqual({
      profile: "custom",
      normalThreshold: 100,
      exceptionalThreshold: 100,
      softLimit: 5,
    });
    expect(reloaded.settings.detailSelection).toEqual(firstLoad.settings.detailSelection);
  });

  it("preserves canonical and arbitrary-minute run-window values on reload", () => {
    const firstLoad = settingsAndStateFromPersistedData({
      settings: {
        schedule: {
          enabled: false,
          runAtLocal: "09:07",
          runUntilLocal: "18:43",
          tickIntervalMin: 20,
        },
      },
    });
    const reloaded = settingsAndStateFromPersistedData({ settings: firstLoad.settings });

    expect(reloaded.settings.schedule.runAtLocal).toBe("09:07");
    expect(reloaded.settings.schedule.runUntilLocal).toBe("18:43");
  });

  it("sanitizes persisted output paths independently with warnings", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        output: {
          dailyDir: " 研究\\日报 ",
          papersDir: "../outside",
        },
      },
    });

    expect(loaded.settings.output.dailyDir).toBe("研究/日报");
    expect(loaded.settings.output.papersDir).toBe("arxiv-daily/papers");
    expect(loaded.warnings).toEqual([
      expect.stringContaining("output.papersDir"),
    ]);
  });

  it("resets both defaults when persisted output directories collide portably", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        output: { dailyDir: "Café/Notes", papersDir: "CAFE\u0301/notes" },
      },
    });
    expect(loaded.settings.output.dailyDir).toBe("arxiv-daily/daily");
    expect(loaded.settings.output.papersDir).toBe("arxiv-daily/papers");
    expect(loaded.warnings.join(" ")).toMatch(/collided.*both defaults/i);
  });

  it("resets both defaults when an invalid field fallback would collide", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        output: {
          dailyDir: "arxiv-daily/papers",
          papersDir: "../outside",
        },
      },
    });
    expect(loaded.settings.output.dailyDir).toBe("arxiv-daily/daily");
    expect(loaded.settings.output.papersDir).toBe("arxiv-daily/papers");
    expect(loaded.warnings).toHaveLength(2);
  });

  it("does not silently rewrite invalid legacy run-window values during load", () => {
    const loaded = settingsAndStateFromPersistedData({
      settings: {
        schedule: {
          enabled: false,
          runAtLocal: "24:00",
          runUntilLocal: "legacy-value",
          tickIntervalMin: 20,
        },
      },
    });

    expect(loaded.settings.schedule.runAtLocal).toBe("24:00");
    expect(loaded.settings.schedule.runUntilLocal).toBe("legacy-value");
  });
});
