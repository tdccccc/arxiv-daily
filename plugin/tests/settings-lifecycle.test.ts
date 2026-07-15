import { describe, expect, it } from "vitest";
import { settingsAndStateFromPersistedData } from "../src/settings/load";

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
