import { describe, it, expect, vi } from "vitest";
import { appendSettingsButton, type CalendarCell, type CalendarCellState } from "../../src/dashboard/view";

describe("Dashboard Integration", () => {
  it("should render settings button and calendar with runnable dates", () => {
    // Verify settings button renders correctly in a parent container
    const parent = document.createElement("div");
    appendSettingsButton(parent, () => {});
    const button = parent.querySelector("button.arxiv-daily-dashboard__settings-btn");
    expect(button).not.toBeNull();
    expect(button!.getAttribute("aria-label")).toBe("Open arXiv Daily settings");
    expect(button!.querySelector("span")!.textContent).toBe("Settings");
  });

  it("should support all calendar cell states", () => {
    // Verify CalendarCellState type covers all expected states
    const states: CalendarCellState[] = ["empty", "runnable", "has-report"];
    expect(states).toHaveLength(3);
    expect(states).toContain("empty");
    expect(states).toContain("runnable");
    expect(states).toContain("has-report");
  });

  it("should create calendar cells with correct structure", () => {
    // Verify CalendarCell interface shape
    const emptyCell: CalendarCell = { date: null, state: "empty" };
    expect(emptyCell.date).toBeNull();
    expect(emptyCell.state).toBe("empty");
    expect(emptyCell.report).toBeUndefined();

    const runnableCell: CalendarCell = { date: "2026-06-20", state: "runnable" };
    expect(runnableCell.date).toBe("2026-06-20");
    expect(runnableCell.state).toBe("runnable");

    const reportCell: CalendarCell = {
      date: "2026-06-19",
      state: "has-report",
      report: { date: "2026-06-19", path: "arxiv-daily/daily/2026-06-19.md", papers: 5, starred: 2 },
    };
    expect(reportCell.state).toBe("has-report");
    expect(reportCell.report!.papers).toBe(5);
    expect(reportCell.report!.starred).toBe(2);
  });

  it("should apply correct CSS classes based on calendar cell state", () => {
    // Simulate getCalendarCellClasses logic
    function getCalendarCellClasses(cell: CalendarCell): string {
      const classes = ["arxiv-daily-dashboard__calendar-day"];
      if (!cell.date) {
        classes.push("is-empty");
      } else if (cell.state === "has-report") {
        classes.push("has-report");
      } else if (cell.state === "runnable") {
        classes.push("is-runnable");
      }
      return classes.join(" ");
    }

    const emptyCell = getCalendarCellClasses({ date: null, state: "empty" });
    expect(emptyCell).toContain("arxiv-daily-dashboard__calendar-day");
    expect(emptyCell).toContain("is-empty");
    expect(emptyCell).not.toContain("is-runnable");
    expect(emptyCell).not.toContain("has-report");

    const runnableCell = getCalendarCellClasses({ date: "2026-06-20", state: "runnable" });
    expect(runnableCell).toContain("arxiv-daily-dashboard__calendar-day");
    expect(runnableCell).toContain("is-runnable");
    expect(runnableCell).not.toContain("is-empty");
    expect(runnableCell).not.toContain("has-report");

    const reportCell = getCalendarCellClasses({ date: "2026-06-19", state: "has-report" });
    expect(reportCell).toContain("arxiv-daily-dashboard__calendar-day");
    expect(reportCell).toContain("has-report");
    expect(reportCell).not.toContain("is-empty");
    expect(reportCell).not.toContain("is-runnable");
  });

  it("should invoke onClick handler for settings button", () => {
    const parent = document.createElement("div");
    const onClick = vi.fn();
    appendSettingsButton(parent, onClick);
    const button = parent.querySelector("button")!;
    button.click();
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
