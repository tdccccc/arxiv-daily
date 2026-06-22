import { describe, it, expect } from "vitest";
import type { CalendarCellState } from "../../src/dashboard/view";

describe("Calendar cell states", () => {
  it("should include no-papers state", () => {
    const states: CalendarCellState[] = ["empty", "runnable", "has-report", "no-papers"];
    expect(states).toContain("no-papers");
  });
});
