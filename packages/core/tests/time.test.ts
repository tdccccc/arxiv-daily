import { describe, it, expect } from "vitest";
import {
  todayInTz,
  formatDate,
  parseHHMM,
  minutesSinceMidnight,
  daysBefore,
  isTimeWithinLocalWindow,
  isWeekendInTz,
  isWeekendDate,
} from "../src/utils/time";

describe("time utils", () => {
  it("todayInTz returns Asia/Shanghai date for given UTC instant", () => {
    const d = todayInTz(new Date("2026-05-11T18:00:00Z"), "Asia/Shanghai");
    expect(formatDate(d)).toBe("2026-05-12");
  });

  it("todayInTz returns UTC date for UTC tz", () => {
    const d = todayInTz(new Date("2026-05-11T18:00:00Z"), "UTC");
    expect(formatDate(d)).toBe("2026-05-11");
  });

  it("parseHHMM parses HH:MM correctly", () => {
    expect(parseHHMM("09:30")).toEqual({ hour: 9, minute: 30 });
    expect(parseHHMM("23:59")).toEqual({ hour: 23, minute: 59 });
  });

  it("parseHHMM throws on invalid input", () => {
    expect(() => parseHHMM("9:30")).toThrow();
    expect(() => parseHHMM("25:00")).toThrow();
  });

  it("minutesSinceMidnight computes minutes for given tz", () => {
    const d = new Date("2026-05-11T01:30:00Z"); // 09:30 Shanghai
    expect(minutesSinceMidnight(d, "Asia/Shanghai")).toBe(9 * 60 + 30);
  });

  it("isTimeWithinLocalWindow checks an inclusive same-day window", () => {
    expect(
      isTimeWithinLocalWindow(
        new Date("2026-06-23T01:00:00Z"),
        "Asia/Shanghai",
        "09:00",
        "18:00",
      ),
    ).toBe(true);
    expect(
      isTimeWithinLocalWindow(
        new Date("2026-06-23T10:00:00Z"),
        "Asia/Shanghai",
        "09:00",
        "18:00",
      ),
    ).toBe(true);
  });

  it("isTimeWithinLocalWindow rejects times outside same-day windows", () => {
    expect(
      isTimeWithinLocalWindow(
        new Date("2026-06-23T00:30:00Z"),
        "Asia/Shanghai",
        "09:00",
        "18:00",
      ),
    ).toBe(false);
    expect(
      isTimeWithinLocalWindow(
        new Date("2026-06-23T11:00:00Z"),
        "Asia/Shanghai",
        "09:00",
        "18:00",
      ),
    ).toBe(false);
  });

  it("isTimeWithinLocalWindow handles cross-midnight windows", () => {
    expect(
      isTimeWithinLocalWindow(
        new Date("2026-06-23T15:30:00Z"), // 23:30 Shanghai
        "Asia/Shanghai",
        "23:00",
        "02:00",
      ),
    ).toBe(true);
    expect(
      isTimeWithinLocalWindow(
        new Date("2026-06-23T17:30:00Z"), // 01:30 Shanghai next day
        "Asia/Shanghai",
        "23:00",
        "02:00",
      ),
    ).toBe(true);
    expect(
      isTimeWithinLocalWindow(
        new Date("2026-06-23T08:00:00Z"), // 16:00 Shanghai
        "Asia/Shanghai",
        "23:00",
        "02:00",
      ),
    ).toBe(false);
  });

  it("daysBefore subtracts whole days", () => {
    const d = { y: 2026, m: 5, d: 11 };
    expect(daysBefore(d, 1)).toEqual({ y: 2026, m: 5, d: 10 });
    expect(daysBefore(d, 5)).toEqual({ y: 2026, m: 5, d: 6 });
    expect(daysBefore(d, 11)).toEqual({ y: 2026, m: 4, d: 30 });
  });

  it("daysBefore handles leap year Feb 29", () => {
    expect(daysBefore({ y: 2024, m: 3, d: 1 }, 1)).toEqual({
      y: 2024,
      m: 2,
      d: 29,
    });
  });

  it("todayInTz handles DST spring-forward and fall-back dates", () => {
    expect(formatDate(todayInTz(new Date("2026-03-08T07:30:00Z"), "America/New_York")))
      .toBe("2026-03-08");
    expect(formatDate(todayInTz(new Date("2026-11-01T06:30:00Z"), "America/New_York")))
      .toBe("2026-11-01");
  });

  it("daysBefore works across timezone-derived date boundaries", () => {
    const tokyoToday = todayInTz(new Date("2026-01-01T15:30:00Z"), "Asia/Tokyo");
    expect(formatDate(tokyoToday)).toBe("2026-01-02");
    expect(daysBefore(tokyoToday, 1)).toEqual({ y: 2026, m: 1, d: 1 });
  });

  it("daysBefore accepts a timezone for local calendar arithmetic across DST", () => {
    const newYorkToday = todayInTz(
      new Date("2026-03-09T16:00:00Z"),
      "America/New_York",
    );
    expect(formatDate(newYorkToday)).toBe("2026-03-09");
    expect(daysBefore(newYorkToday, 1, "America/New_York")).toEqual({
      y: 2026,
      m: 3,
      d: 8,
    });
  });

  it("isWeekendInTz returns true for Saturday Shanghai", () => {
    const d = new Date("2026-05-09T05:00:00Z"); // 13:00 Shanghai, Sat
    expect(isWeekendInTz(d, "Asia/Shanghai")).toBe(true);
  });

  it("isWeekendInTz returns true for Sunday Shanghai", () => {
    const d = new Date("2026-05-10T05:00:00Z");
    expect(isWeekendInTz(d, "Asia/Shanghai")).toBe(true);
  });

  it("isWeekendInTz returns false for Monday Shanghai", () => {
    const d = new Date("2026-05-11T05:00:00Z");
    expect(isWeekendInTz(d, "Asia/Shanghai")).toBe(false);
  });

  it("isWeekendInTz handles UTC-day-flip", () => {
    // 2026-05-09T18:00Z is 2026-05-10 (Sun) Shanghai
    const d = new Date("2026-05-09T18:00:00Z");
    expect(isWeekendInTz(d, "Asia/Shanghai")).toBe(true);
    // Same instant is still Sat in UTC
    expect(isWeekendInTz(d, "UTC")).toBe(true);
  });

  it("isWeekendInTz respects timezone boundary at local midnight", () => {
    const instant = new Date("2026-05-08T16:30:00Z");
    expect(isWeekendInTz(instant, "Asia/Shanghai")).toBe(true);
    expect(isWeekendInTz(instant, "UTC")).toBe(false);
  });

  it("isWeekendDate checks calendar dates without timezone conversion", () => {
    expect(isWeekendDate({ y: 2026, m: 5, d: 9 })).toBe(true);
    expect(isWeekendDate({ y: 2026, m: 5, d: 10 })).toBe(true);
    expect(isWeekendDate({ y: 2026, m: 5, d: 11 })).toBe(false);
  });

  it("isWeekendDate accepts a timezone for tz-local calendar dates", () => {
    expect(isWeekendDate({ y: 2026, m: 5, d: 9 }, "Asia/Shanghai")).toBe(true);
    expect(isWeekendDate({ y: 2026, m: 5, d: 11 }, "America/New_York")).toBe(false);
  });
});
