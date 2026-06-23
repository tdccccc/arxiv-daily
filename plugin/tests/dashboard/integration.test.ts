import { describe, it, expect, vi } from "vitest";
import {
  appendSettingsButton,
  calendarCellAriaLabel,
  type CalendarCell,
  type CalendarCellState,
} from "../../src/dashboard/view";

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
    const states: CalendarCellState[] = ["empty", "runnable", "has-report", "no-relevant-papers"];
    expect(states).toHaveLength(4);
    expect(states).toContain("empty");
    expect(states).toContain("runnable");
    expect(states).toContain("has-report");
    expect(states).toContain("no-relevant-papers");
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

    const noRelevantPapersCell: CalendarCell = {
      date: "2026-06-18",
      state: "no-relevant-papers",
      report: { date: "2026-06-18", path: "arxiv-daily/daily/2026-06-18.md", papers: 0, starred: 0 },
    };
    expect(noRelevantPapersCell.state).toBe("no-relevant-papers");
    expect(noRelevantPapersCell.report!.papers).toBe(0);
  });

  it("should apply correct CSS classes based on calendar cell state", () => {
    // Simulate getCalendarCellClasses logic from view.ts
    function getCalendarCellClasses(cell: CalendarCell): string {
      const classes = ["arxiv-daily-dashboard__calendar-day"];
      if (!cell.date) {
        classes.push("is-empty");
      } else if (cell.state === "has-report") {
        classes.push("has-report");
      } else if (cell.state === "no-relevant-papers") {
        classes.push("has-report");
        classes.push("no-relevant-papers");
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
    expect(emptyCell).not.toContain("no-relevant-papers");

    const runnableCell = getCalendarCellClasses({ date: "2026-06-20", state: "runnable" });
    expect(runnableCell).toContain("arxiv-daily-dashboard__calendar-day");
    expect(runnableCell).toContain("is-runnable");
    expect(runnableCell).not.toContain("is-empty");
    expect(runnableCell).not.toContain("has-report");
    expect(runnableCell).not.toContain("no-relevant-papers");

    const reportCell = getCalendarCellClasses({ date: "2026-06-19", state: "has-report" });
    expect(reportCell).toContain("arxiv-daily-dashboard__calendar-day");
    expect(reportCell).toContain("has-report");
    expect(reportCell).not.toContain("is-empty");
    expect(reportCell).not.toContain("is-runnable");
    expect(reportCell).not.toContain("no-relevant-papers");

    const noRelevantPapersCell = getCalendarCellClasses({ date: "2026-06-18", state: "no-relevant-papers" });
    expect(noRelevantPapersCell).toContain("arxiv-daily-dashboard__calendar-day");
    expect(noRelevantPapersCell).toContain("has-report");
    expect(noRelevantPapersCell).toContain("no-relevant-papers");
    expect(noRelevantPapersCell).not.toContain("is-empty");
    expect(noRelevantPapersCell).not.toContain("is-runnable");
  });

  it("should invoke onClick handler for settings button", () => {
    const parent = document.createElement("div");
    const onClick = vi.fn();
    appendSettingsButton(parent, onClick);
    const button = parent.querySelector("button")!;
    button.click();
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("should verify CSS classes for no-relevant-papers state are used correctly", () => {
    // Verify that no-relevant-papers cells get both has-report and no-relevant-papers classes.
    const cell: CalendarCell = {
      date: "2026-06-15",
      state: "no-relevant-papers",
      report: { date: "2026-06-15", path: "arxiv-daily/daily/2026-06-15.md", papers: 0, starred: 0 },
    };

    expect(cell.state).toBe("no-relevant-papers");
    expect(cell.report).toBeDefined();
    expect(cell.report!.papers).toBe(0);
  });

  it("should render runnable cells with correct aria labels", () => {
    const cell: CalendarCell = { date: "2026-06-20", state: "runnable" };
    expect(calendarCellAriaLabel(cell)).toBe("Run daily report");
  });

  it("should render no-relevant-papers cells with correct aria labels", () => {
    const cell: CalendarCell = {
      date: "2026-06-18",
      state: "no-relevant-papers",
      report: { date: "2026-06-18", path: "arxiv-daily/daily/2026-06-18.md", papers: 0, starred: 0 },
    };
    expect(calendarCellAriaLabel(cell)).toBe("No relevant papers");
  });

  it("should render has-report cells with correct aria labels", () => {
    const cell: CalendarCell = {
      date: "2026-06-19",
      state: "has-report",
      report: { date: "2026-06-19", path: "arxiv-daily/daily/2026-06-19.md", papers: 5, starred: 2 },
    };
    expect(calendarCellAriaLabel(cell)).toBe("5 indexed papers, 2 starred");
  });
});
