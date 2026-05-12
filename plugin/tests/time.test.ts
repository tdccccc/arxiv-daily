import { describe, it, expect } from "vitest";
import {
  todayInTz,
  formatDate,
  parseHHMM,
  minutesSinceMidnight,
  daysBefore,
  isWeekendInTz,
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

  it("daysBefore subtracts whole days", () => {
    const d = { y: 2026, m: 5, d: 11 };
    expect(daysBefore(d, 1)).toEqual({ y: 2026, m: 5, d: 10 });
    expect(daysBefore(d, 5)).toEqual({ y: 2026, m: 5, d: 6 });
    expect(daysBefore(d, 11)).toEqual({ y: 2026, m: 4, d: 30 });
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
});
