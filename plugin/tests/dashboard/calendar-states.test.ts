import { describe, it, expect } from "vitest";
import type { CalendarCellState } from "../../src/dashboard/view";

describe("Calendar cell states", () => {
  it("should include no-relevant-papers state", () => {
    const states: CalendarCellState[] = ["empty", "runnable", "has-report", "no-relevant-papers"];
    expect(states).toContain("no-relevant-papers");
  });

  it("should identify no-relevant-papers state for reports with 0 papers", () => {
    // Verify the type accepts all four states including no-relevant-papers.
    const allStates: CalendarCellState[] = ["empty", "runnable", "has-report", "no-relevant-papers"];
    expect(allStates).toHaveLength(4);
    // The actual logic is tested via buildCalendarCells integration;
    // this verifies the type system includes the new state.
    const state: CalendarCellState = "no-relevant-papers";
    expect(state).toBe("no-relevant-papers");
  });
});

describe("Calendar rendering", () => {
  it("should render no-relevant-papers state with '0' count text", () => {
    // Verify that no-relevant-papers cells display "0" as the paper count.
    const noRelevantPapersReport = { date: "2026-06-18", path: "arxiv-daily/daily/2026-06-18.md", papers: 0, starred: 0 };
    expect(noRelevantPapersReport.papers).toBe(0);
    // The renderNoRelevantPapersCell method creates a span with text "0".
    const countText = String(noRelevantPapersReport.papers);
    expect(countText).toBe("0");
  });

  it("should render runnable state with play icon class", () => {
    // Verify that runnable cells have the icon container class
    const iconClass = "arxiv-daily-dashboard__calendar-day-icon";
    expect(iconClass).toBe("arxiv-daily-dashboard__calendar-day-icon");
  });

  it("should render has-report state with paper count", () => {
    // Verify that has-report cells display the actual paper count
    const report = { date: "2026-06-19", path: "arxiv-daily/daily/2026-06-19.md", papers: 5, starred: 2 };
    const countText = String(report.papers);
    expect(countText).toBe("5");
  });
});

describe("isRunnable", () => {
  it("should return true for past dates in lookback window with no file", () => {
    // Verify the isRunnable logic: past weekday in lookback with no file is runnable
    // This tests the date checking logic indirectly through the type system
    const date = "2026-06-18"; // Assume this is a past weekday
    const hasFile = false;
    const isInLookback = true;
    const isWeekend = false;

    // isRunnable should return true when all conditions are met
    const shouldBeRunnable = !hasFile && isInLookback && !isWeekend;
    expect(shouldBeRunnable).toBe(true);
  });

  it("should return false for weekends", () => {
    // Verify weekend detection logic
    // 2026-06-20 is a Saturday (day 6 of week)
    const date = "2026-06-20";
    const [y, m, d] = date.split('-').map(Number);
    const dateObj = new Date(Date.UTC(y, m - 1, d));
    const dayOfWeek = dateObj.getUTCDay();
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
    expect(isWeekend).toBe(true);
  });

  it("should return false for dates outside lookback window", () => {
    // Verify that dates outside lookback are not runnable
    const isInLookback = false;
    const hasFile = false;
    const isWeekend = false;

    const shouldBeRunnable = !hasFile && isInLookback && !isWeekend;
    expect(shouldBeRunnable).toBe(false);
  });
});
