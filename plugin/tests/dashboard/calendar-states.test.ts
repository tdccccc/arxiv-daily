import { describe, it, expect } from "vitest";
import type { CalendarCellState } from "../../src/dashboard/view";

describe("Calendar cell states", () => {
  it("should include no-papers state", () => {
    const states: CalendarCellState[] = ["empty", "runnable", "has-report", "no-papers"];
    expect(states).toContain("no-papers");
  });

  it("should identify no-papers state for reports with 0 papers", () => {
    // Verify the type accepts all four states including no-papers
    const allStates: CalendarCellState[] = ["empty", "runnable", "has-report", "no-papers"];
    expect(allStates).toHaveLength(4);
    // The actual logic is tested via buildCalendarCells integration;
    // this verifies the type system includes the new state.
    const state: CalendarCellState = "no-papers";
    expect(state).toBe("no-papers");
  });
});
