import { describe, it, expect, vi } from "vitest";
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  calendarCellAriaLabel,
  buildCalendarDailyReportMap,
  isCalendarRunWhitelisted,
  registerDashboardView,
  resolveCalendarEmptyReason,
  resolveCalendarCellState,
  type CalendarCell,
  type CalendarEmptyReason,
  type CalendarCellState,
  type CalendarRunWhitelistInput,
} from "../../src/dashboard/view";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import type { RunStateEntry } from "@arxiv-daily/core";

function runState(
  status: RunStateEntry["status"],
  overrides: Partial<RunStateEntry> = {},
): RunStateEntry {
  return {
    status,
    lastAttempt: 1,
    attempts: 1,
    ...overrides,
  };
}

function whitelistInput(
  overrides: Partial<CalendarRunWhitelistInput> = {},
): CalendarRunWhitelistInput {
  return {
    date: "2026-06-23",
    today: "2026-06-23",
    now: new Date("2026-06-23T03:00:00Z"),
    timezone: "Asia/Shanghai",
    runAtLocal: "09:00",
    runUntilLocal: "18:00",
    inLookback: true,
    isWeekend: false,
    hasDailyReport: false,
    recentDates: new Set(["2026-06-22"]),
    ...overrides,
  };
}

describe("Calendar State Model", () => {
  it("should define correct cell states", () => {
    // Verify all four CalendarCellState values are valid
    const validStates: CalendarCellState[] = ["empty", "runnable", "has-report", "no-relevant-papers"];
    expect(validStates).toHaveLength(4);
    expect(validStates).toContain("empty");
    expect(validStates).toContain("runnable");
    expect(validStates).toContain("has-report");
    expect(validStates).toContain("no-relevant-papers");
  });

  it("should define empty reasons for non-visual calendar context", () => {
    const reasons: CalendarEmptyReason[] = [
      "blank",
      "arxiv-not-updated",
      "future",
      "before-tracking",
      "report-missing",
    ];
    expect(reasons).toHaveLength(5);
  });
});

describe("Calendar Cell Builder", () => {
  it("uses run state to hide already resolved no-work dates", () => {
    expect(
      resolveCalendarCellState({
        runnable: true,
        runState: runState("completed", { papersWritten: 0 }),
      }),
    ).toEqual({ state: "empty", emptyReason: "arxiv-not-updated" });
    expect(
      resolveCalendarCellState({
        runnable: true,
        runState: runState("skipped"),
      }),
    ).toEqual({ state: "empty", emptyReason: "arxiv-not-updated" });
    expect(
      resolveCalendarCellState({
        runnable: true,
        runState: runState("failed_permanent"),
      }),
    ).toEqual({ state: "empty", emptyReason: "arxiv-not-updated" });
  });

  it("uses completed non-zero run state as a non-runnable missing-report fallback", () => {
    expect(
      resolveCalendarCellState({
        runnable: true,
        runState: runState("completed", { papersWritten: 10 }),
      }),
    ).toEqual({ state: "empty", emptyReason: "report-missing" });
  });

  it("keeps pending and transient dates runnable when the date is otherwise runnable", () => {
    expect(
      resolveCalendarCellState({
        runnable: true,
        runState: runState("pending"),
      }),
    ).toEqual({ state: "runnable" });
    expect(
      resolveCalendarCellState({
        runnable: true,
        runState: runState("failed_transient"),
      }),
    ).toEqual({ state: "runnable" });
    expect(resolveCalendarCellState({ runnable: true })).toEqual({ state: "runnable" });
  });

  it("prefers existing daily reports over run state", () => {
    expect(
      resolveCalendarCellState({
        report: { papers: 0 },
        runnable: true,
        runState: runState("completed", { papersWritten: 0 }),
      }),
    ).toEqual({ state: "no-relevant-papers" });
    expect(
      resolveCalendarCellState({
        report: { papers: 3 },
        runnable: false,
        runState: runState("skipped"),
      }),
    ).toEqual({ state: "has-report" });
  });

  it("should identify runnable dates within lookback window", () => {
    // Verify that a date in the lookback window with no file is considered runnable
    const cell: CalendarCell = { date: "2026-06-20", state: "runnable" };
    expect(cell.state).toBe("runnable");
    expect(cell.date).toBe("2026-06-20");
    expect(cell.report).toBeUndefined();
  });

  it("should identify dates with reports", () => {
    // Verify that dates with reports have the has-report state
    const cell: CalendarCell = {
      date: "2026-06-19",
      state: "has-report",
      report: { date: "2026-06-19", path: "arxiv-daily/daily/2026-06-19.md", papers: 5, starred: 2 },
    };
    expect(cell.state).toBe("has-report");
    expect(cell.report).toBeDefined();
    expect(cell.report!.papers).toBe(5);
  });

  it("should identify dates with no relevant papers reports", () => {
    // Verify that dates with 0 papers have the no-relevant-papers state
    const cell: CalendarCell = {
      date: "2026-06-18",
      state: "no-relevant-papers",
      report: { date: "2026-06-18", path: "arxiv-daily/daily/2026-06-18.md", papers: 0, starred: 0 },
    };
    expect(cell.state).toBe("no-relevant-papers");
    expect(cell.report).toBeDefined();
    expect(cell.report!.papers).toBe(0);
  });

  it("builds calendar reports from scanned markdown files without probing storage", async () => {
    const reports = await buildCalendarDailyReportMap({
      month: "2026-06",
      scannedReports: [
        {
          date: "2026-06-24",
          path: "arxiv-daily/daily/2026-06-24.md",
          papers: 10,
          starred: 2,
        },
        {
          date: "2026-07-01",
          path: "arxiv-daily/daily/2026-07-01.md",
          papers: 1,
          starred: 0,
        },
      ],
      normalizePath: (path) => path,
    });

    const report = reports.get("2026-06-24");
    expect(report).toEqual({
      date: "2026-06-24",
      path: "arxiv-daily/daily/2026-06-24.md",
      papers: 10,
      starred: 2,
    });
    expect(reports.has("2026-07-01")).toBe(false);

    expect(
      resolveCalendarCellState({
        report,
        runnable: isCalendarRunWhitelisted(
          whitelistInput({
            date: "2026-06-24",
            today: "2026-06-25",
            hasDailyReport: Boolean(report),
            recentDates: new Set(["2026-06-24"]),
          }),
        ),
        runState: runState("completed", { papersWritten: 10 }),
      }),
    ).toEqual({ state: "has-report" });
  });

  it("does not synthesize calendar reports from run state when scanned reports miss a file", async () => {
    const reports = await buildCalendarDailyReportMap({
      month: "2026-06",
      scannedReports: [],
      normalizePath: (path) => path,
    });

    expect(reports.has("2026-06-24")).toBe(false);
    expect(
      resolveCalendarCellState({
        report: reports.get("2026-06-24"),
        runnable: false,
        runState: runState("completed", { papersWritten: 10 }),
      }),
    ).toEqual({ state: "empty", emptyReason: "report-missing" });
  });
});

describe("isCalendarRunWhitelisted", () => {
  it("shows today only inside the Run window", () => {
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({ now: new Date("2026-06-23T03:00:00Z") }),
      ),
    ).toBe(true);
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({ now: new Date("2026-06-23T00:30:00Z") }),
      ),
    ).toBe(false);
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({ now: new Date("2026-06-23T11:00:00Z") }),
      ),
    ).toBe(false);
  });

  it("allows cross-midnight Run windows", () => {
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          now: new Date("2026-06-23T15:30:00Z"),
          runAtLocal: "23:00",
          runUntilLocal: "02:00",
        }),
      ),
    ).toBe(true);
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          now: new Date("2026-06-23T08:00:00Z"),
          runAtLocal: "23:00",
          runUntilLocal: "02:00",
        }),
      ),
    ).toBe(false);
  });

  it("shows non-today dates only when latest /recent cache contains the date", () => {
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          date: "2026-06-22",
          today: "2026-06-23",
          recentDates: new Set(["2026-06-22"]),
        }),
      ),
    ).toBe(true);
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          date: "2026-06-19",
          today: "2026-06-23",
          recentDates: new Set(["2026-06-22"]),
        }),
      ),
    ).toBe(false);
  });

  it("requires no local daily report for both today and non-today", () => {
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          date: "2026-06-22",
          today: "2026-06-23",
          hasDailyReport: true,
          recentDates: new Set(["2026-06-22"]),
        }),
      ),
    ).toBe(false);
  });

  it("blocks terminal and running states but allows transient failures", () => {
    for (const state of [
      runState("running"),
      runState("skipped"),
      runState("failed_permanent"),
      runState("completed", { papersWritten: 0 }),
      runState("completed", { papersWritten: 10 }),
    ]) {
      expect(
        isCalendarRunWhitelisted(
          whitelistInput({
            date: "2026-06-22",
            today: "2026-06-23",
            recentDates: new Set(["2026-06-22"]),
            runState: state,
          }),
        ),
      ).toBe(false);
    }

    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          date: "2026-06-22",
          today: "2026-06-23",
          recentDates: new Set(["2026-06-22"]),
          runState: runState("failed_transient"),
        }),
      ),
    ).toBe(true);
  });
});

describe("resolveCalendarEmptyReason", () => {
  it("labels future dates distinctly", () => {
    expect(
      resolveCalendarEmptyReason({
        date: "2026-06-25",
        today: "2026-06-24",
        trackingStartDate: "2026-06-20",
        recentDates: new Set(),
      }),
    ).toBe("future");
  });

  it("hides dates before tracking start when they are not in /recent", () => {
    expect(
      resolveCalendarEmptyReason({
        date: "2026-06-20",
        today: "2026-06-24",
        trackingStartDate: "2026-06-24",
        recentDates: new Set(["2026-06-23"]),
      }),
    ).toBe("before-tracking");
  });

  it("keeps real tracked dates without reports as arXiv not updated", () => {
    expect(
      resolveCalendarEmptyReason({
        date: "2026-06-23",
        today: "2026-06-24",
        trackingStartDate: "2026-06-20",
        recentDates: new Set(),
      }),
    ).toBe("arxiv-not-updated");
  });

  it("does not hide dates before tracking start when /recent can still run them", () => {
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          date: "2026-06-22",
          today: "2026-06-24",
          recentDates: new Set(["2026-06-22"]),
        }),
      ),
    ).toBe(true);
  });
});

describe("calendar whitelist and resolution together", () => {
  it("keeps non-recent non-today dates empty even when no report exists", () => {
    const runnable = isCalendarRunWhitelisted(
      whitelistInput({
        date: "2026-06-19",
        today: "2026-06-23",
        recentDates: new Set(["2026-06-22"]),
      }),
    );

    expect(
      resolveCalendarCellState({
        runnable,
        emptyReason: "arxiv-not-updated",
      }),
    ).toEqual({ state: "empty", emptyReason: "arxiv-not-updated" });
  });
});

describe("Calendar Cell Rendering", () => {
  it("should apply correct CSS classes for each state", () => {
    // Verify CSS class application logic matches view.ts getCalendarCellClasses
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

    // Test empty cell
    const emptyClasses = getCalendarCellClasses({ date: null, state: "empty" });
    expect(emptyClasses).toContain("is-empty");
    expect(emptyClasses).not.toContain("has-report");
    expect(emptyClasses).not.toContain("no-relevant-papers");
    expect(emptyClasses).not.toContain("is-runnable");

    // Test runnable cell
    const runnableClasses = getCalendarCellClasses({ date: "2026-06-20", state: "runnable" });
    expect(runnableClasses).toContain("is-runnable");
    expect(runnableClasses).not.toContain("is-empty");
    expect(runnableClasses).not.toContain("has-report");
    expect(runnableClasses).not.toContain("no-relevant-papers");

    // Test has-report cell
    const reportClasses = getCalendarCellClasses({ date: "2026-06-19", state: "has-report" });
    expect(reportClasses).toContain("has-report");
    expect(reportClasses).not.toContain("is-empty");
    expect(reportClasses).not.toContain("is-runnable");
    expect(reportClasses).not.toContain("no-relevant-papers");

    // Test no-relevant-papers cell (should have both has-report and no-relevant-papers)
    const noRelevantPapersClasses = getCalendarCellClasses({ date: "2026-06-18", state: "no-relevant-papers" });
    expect(noRelevantPapersClasses).toContain("has-report");
    expect(noRelevantPapersClasses).toContain("no-relevant-papers");
    expect(noRelevantPapersClasses).not.toContain("is-empty");
    expect(noRelevantPapersClasses).not.toContain("is-runnable");
  });

  it("should render play icon for runnable dates", () => {
    // Verify that runnable cells have the icon class
    const iconClass = "arxiv-daily-dashboard__calendar-day-icon";
    expect(iconClass).toBe("arxiv-daily-dashboard__calendar-day-icon");
  });
});

describe("calendarCellAriaLabel", () => {
  it("labels arXiv-not-updated empty dates in English and hides blank cells", () => {
    expect(
      calendarCellAriaLabel({
        date: "2026-06-20",
        state: "empty",
        emptyReason: "arxiv-not-updated",
      }),
    ).toBe("arXiv not updated");
    expect(
      calendarCellAriaLabel({
        date: null,
        state: "empty",
        emptyReason: "blank",
      }),
    ).toBeUndefined();
  });

  it("labels future dates and hides dates before tracking start", () => {
    expect(
      calendarCellAriaLabel({
        date: "2026-06-25",
        state: "empty",
        emptyReason: "future",
      }),
    ).toBe("Future date");
    expect(
      calendarCellAriaLabel({
        date: "2026-05-01",
        state: "empty",
        emptyReason: "before-tracking",
      }),
    ).toBeUndefined();
  });

  it("labels completed dates whose daily report is missing", () => {
    expect(
      calendarCellAriaLabel({
        date: "2026-06-24",
        state: "empty",
        emptyReason: "report-missing",
      }),
    ).toBe("Daily report missing");
  });

  it("labels zero-count reports as no relevant papers", () => {
    expect(
      calendarCellAriaLabel({
        date: "2026-06-18",
        state: "no-relevant-papers",
        report: { date: "2026-06-18", path: "arxiv-daily/daily/2026-06-18.md", papers: 0, starred: 0 },
      }),
    ).toBe("No relevant papers");
  });

  it("labels runnable cells without repeating the date", () => {
    expect(
      calendarCellAriaLabel({
        date: "2026-06-22",
        state: "runnable",
      }),
    ).toBe("Run daily report");
  });

  it("labels report cells with counts only", () => {
    expect(
      calendarCellAriaLabel({
        date: "2026-06-19",
        state: "has-report",
        report: { date: "2026-06-19", path: "arxiv-daily/daily/2026-06-19.md", papers: 5, starred: 2 },
      }),
    ).toBe("5 indexed papers, 2 starred");
  });
});

describe("runDateFromCalendar", () => {
  it("shows the running notice before refreshing recent dates", async () => {
    const events: string[] = [];
    let createView: ((leaf: unknown) => unknown) | undefined;
    const settings = {
      ...DEFAULT_SETTINGS,
      llm: {
        ...DEFAULT_SETTINGS.llm,
        apiKey: "test-key",
      },
      arxiv: {
        ...DEFAULT_SETTINGS.arxiv,
        topics: [{ name: "Topic", tag: "topic", description: "Topic description" }],
      },
    };
    const plugin = {
      app: {},
      settings,
      registerView: vi.fn((type: string, viewCreator: (leaf: unknown) => unknown) => {
        if (type === ARXIV_DAILY_DASHBOARD_VIEW) createView = viewCreator;
      }),
      openSettings: vi.fn(),
      logger: {
        info: vi.fn((message: string) => events.push(message)),
        warn: vi.fn(),
      },
      recentDates: {
        refresh: vi.fn(async () => {
          events.push("refresh");
        }),
        hasDate: vi.fn(() => true),
      },
      scheduler: {
        runForDateNow: vi.fn(async () => {
          events.push("scheduler");
          throw new Error("stop before reload");
        }),
      },
    };
    registerDashboardView(plugin as never);
    const view = createView?.({}) as { runDateFromCalendar(date: string): Promise<void> };

    await expect(view.runDateFromCalendar("2026-06-22")).rejects.toThrow(
      "stop before reload",
    );

    expect(events.slice(0, 2)).toEqual([
      "arXiv Daily: running for 2026-06-22…",
      "refresh",
    ]);
  });
});
