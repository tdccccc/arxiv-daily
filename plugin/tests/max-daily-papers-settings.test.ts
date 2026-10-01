import { describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { settingsAndStateFromPersistedData } from "../src/settings/load";
import { SettingsChangeService } from "../src/settings/change-service";

describe("persisted daily paper limits", () => {
  it("loads older settings with the shared default and preserves valid custom limits", () => {
    expect(settingsAndStateFromPersistedData({ settings: { output: {} } }).settings.output.maxDailyPapers).toBe(20);
    expect(settingsAndStateFromPersistedData({ settings: { output: { maxDailyPapers: 35 } } }).settings.output.maxDailyPapers).toBe(35);
  });

  it.each([undefined, null, 0, -1, 1.5, "20", Number.NaN, Infinity].map((value) => ({ value })))(
    "restores 20 when the persisted limit is $value",
    ({ value }) => {
      const loaded = settingsAndStateFromPersistedData({ settings: { output: { maxDailyPapers: value } } });
      expect(loaded.settings.output.maxDailyPapers).toBe(20);
    },
  );
});

describe("daily paper limit changes", () => {
  it("persists a valid limit before committing it to live settings", async () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    const persistSettings = vi.fn(async (candidate: typeof settings) => {
      expect(candidate.output.maxDailyPapers).toBe(35);
      expect(settings.output.maxDailyPapers).toBe(20);
    });
    const service = new SettingsChangeService({ settings, persistSettings });
    await service.changeValue("output.maxDailyPapers", 35);
    expect(persistSettings).toHaveBeenCalledOnce();
    expect(settings.output.maxDailyPapers).toBe(35);
  });

  it.each([undefined, null, 0, -1, 1.5, "20", Number.NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]
    .map((value) => ({ value })))("rejects $value before persistence", async ({ value }) => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    const persistSettings = vi.fn();
    const service = new SettingsChangeService({ settings, persistSettings });
    await expect(service.changeValue("output.maxDailyPapers", value)).rejects.toThrow(/positive safe integer/i);
    expect(persistSettings).not.toHaveBeenCalled();
    expect(settings.output.maxDailyPapers).toBe(20);
  });

  it("refuses to persist an invalid limit through the legacy current-settings path", async () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.output.maxDailyPapers = 0;
    const persistSettings = vi.fn();
    const service = new SettingsChangeService({ settings, persistSettings });
    await expect(service.persistCurrent()).rejects.toThrow(/positive safe integer/i);
    expect(persistSettings).not.toHaveBeenCalled();
  });
});
