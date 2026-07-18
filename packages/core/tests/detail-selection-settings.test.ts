import { describe, expect, it } from "vitest";
import {
  DEFAULT_DETAIL_SELECTION,
  DEFAULT_SETTINGS,
  DETAIL_SELECTION_PRESETS,
  detailSelectionPreset,
  sanitizeDetailSelection,
} from "../src/index";

describe("detail selection settings", () => {
  it("defaults to the balanced preset", () => {
    expect(DEFAULT_DETAIL_SELECTION).toEqual({
      profile: "balanced",
      normalThreshold: 75,
      exceptionalThreshold: 92,
      softLimit: 3,
    });
    expect(DEFAULT_SETTINGS.detailSelection).toEqual(DEFAULT_DETAIL_SELECTION);
  });

  it("exports all recommended named presets as fresh settings", () => {
    expect(DETAIL_SELECTION_PRESETS.conservative).toEqual({
      profile: "conservative",
      normalThreshold: 85,
      exceptionalThreshold: 95,
      softLimit: 1,
    });
    expect(DETAIL_SELECTION_PRESETS.broad).toEqual({
      profile: "broad",
      normalThreshold: 65,
      exceptionalThreshold: 88,
      softLimit: 5,
    });
    const first = detailSelectionPreset("balanced");
    const second = detailSelectionPreset("balanced");
    expect(first).toEqual(second);
    expect(first).not.toBe(second);
  });

  it("canonicalizes named profiles even when persisted numbers conflict", () => {
    expect(sanitizeDetailSelection({
      profile: "conservative",
      normalThreshold: 1,
      exceptionalThreshold: 2,
      softLimit: 20,
    })).toEqual(DETAIL_SELECTION_PRESETS.conservative);
    expect(sanitizeDetailSelection({
      profile: "broad",
      normalThreshold: 99,
      exceptionalThreshold: 100,
      softLimit: 0,
    })).toEqual(DETAIL_SELECTION_PRESETS.broad);
  });

  it("falls back safely for legacy and malformed values", () => {
    expect(sanitizeDetailSelection(undefined)).toEqual(DEFAULT_DETAIL_SELECTION);
    expect(sanitizeDetailSelection({
      profile: "unknown",
      normalThreshold: 5,
      exceptionalThreshold: 6,
      softLimit: 20,
    })).toEqual(DEFAULT_DETAIL_SELECTION);
    expect(sanitizeDetailSelection({
      profile: "custom",
      normalThreshold: Number.NaN,
      exceptionalThreshold: Number.POSITIVE_INFINITY,
      softLimit: "many",
    })).toEqual({
      profile: "custom",
      normalThreshold: 75,
      exceptionalThreshold: 92,
      softLimit: 3,
    });
  });

  it("clamps finite values and enforces threshold ordering", () => {
    expect(sanitizeDetailSelection({
      profile: "custom",
      normalThreshold: 110,
      exceptionalThreshold: -5,
      softLimit: 24.7,
    })).toEqual({
      profile: "custom",
      normalThreshold: 100,
      exceptionalThreshold: 100,
      softLimit: 20,
    });
    expect(sanitizeDetailSelection({
      profile: "custom",
      normalThreshold: -1,
      exceptionalThreshold: 33.5,
      softLimit: -2,
    })).toEqual({
      profile: "custom",
      normalThreshold: 0,
      exceptionalThreshold: 33.5,
      softLimit: 0,
    });
  });
});
