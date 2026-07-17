import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
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

describe("plugin settings reload lifecycle", () => {
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
