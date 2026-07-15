import { describe, expect, it } from "vitest";
import { lookbackDateStrings, todayDateString } from "../../src/services/scheduling/date-selector";

describe("date-selector", () => {
  it("todayDateString formats today in the given timezone", () => {
    // 2026-05-11T05:00:00Z = 13:00 Asia/Shanghai (UTC+8) -> same calendar day 2026-05-11.
    expect(todayDateString("Asia/Shanghai", () => new Date("2026-05-11T05:00:00Z"))).toBe("2026-05-11");
    // 2026-05-11T17:00:00Z = 01:00 next day Shanghai -> 2026-05-12.
    expect(todayDateString("Asia/Shanghai", () => new Date("2026-05-11T17:00:00Z"))).toBe("2026-05-12");
  });

  it("lookbackDateStrings returns N inclusive descending dates ending today", () => {
    const dates = lookbackDateStrings("Asia/Shanghai", 5, () => new Date("2026-05-11T05:00:00Z"));
    expect(dates).toEqual(["2026-05-11", "2026-05-10", "2026-05-09", "2026-05-08", "2026-05-07"]);
  });

  it("lookbackDateStrings with count=1 returns only today", () => {
    expect(lookbackDateStrings("Asia/Shanghai", 1, () => new Date("2026-05-11T05:00:00Z"))).toEqual([
      "2026-05-11",
    ]);
  });
});
