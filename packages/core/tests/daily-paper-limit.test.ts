import { describe, expect, it } from "vitest";
import {
  DEFAULT_MAX_DAILY_PAPERS,
  DEFAULT_SETTINGS,
  isValidMaxDailyPapers,
  normalizeMaxDailyPapers,
} from "../src/index";

describe("daily paper limit settings", () => {
  it("defaults the shared output limit to 20 papers", () => {
    expect(DEFAULT_MAX_DAILY_PAPERS).toBe(20);
    expect(DEFAULT_SETTINGS.output.maxDailyPapers).toBe(20);
  });

  it.each([1, 20, 35, 1000, Number.MAX_SAFE_INTEGER])("accepts and preserves the positive safe integer %s", (value) => {
    expect(isValidMaxDailyPapers(value)).toBe(true);
    expect(normalizeMaxDailyPapers(value)).toBe(value);
  });

  it.each([undefined, null, 0, -1, 1.5, Number.NaN, Infinity, -Infinity, Number.MAX_SAFE_INTEGER + 1, "20", true, {}, []]
    .map((value) => ({ value, label: String(value) })))(
    "rejects invalid or missing input $label and restores 20",
    ({ value }) => {
      expect(isValidMaxDailyPapers(value)).toBe(false);
      expect(normalizeMaxDailyPapers(value)).toBe(20);
    },
  );
});
