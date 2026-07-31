import type { RunStateEntry } from "@arxiv-daily/core";
import { isTimeWithinLocalWindow } from "@arxiv-daily/core";
import type { DailyReportDay } from "./types";

export type CalendarCellState =
  | "empty"        // No date or outside lookback
  | "runnable"     // Can generate report
  | "has-report"   // Report exists
  | "no-relevant-papers"; // LLM filtered to 0 relevant papers

export type CalendarEmptyReason =
  | "blank"
  | "arxiv-not-updated"
  | "future"
  | "before-tracking"
  | "report-missing"
  | "permanent-failure";

export interface CalendarCell {
  date: string | null;
  state: CalendarCellState;
  report?: DailyReportDay;
  emptyReason?: CalendarEmptyReason;
  failureReason?: string;
}

export interface CalendarCellResolution {
  state: CalendarCellState;
  emptyReason?: CalendarEmptyReason;
}

export interface CalendarRunWhitelistInput {
  date: string;
  today: string;
  now: Date;
  timezone: string;
  runAtLocal: string;
  runUntilLocal: string;
  inLookback: boolean;
  isWeekend: boolean;
  hasDailyReport: boolean;
  recentDates: ReadonlySet<string>;
  runState?: RunStateEntry;
}

export function resolveCalendarCellState({
  report,
  runnable,
  runState,
  emptyReason,
}: {
  report?: { papers: number };
  runnable: boolean;
  runState?: RunStateEntry;
  emptyReason?: CalendarEmptyReason;
}): CalendarCellResolution {
  if (report) {
    return {
      state: report.papers === 0 ? "no-relevant-papers" : "has-report",
    };
  }

  if (runState?.status === "failed_permanent") {
    return { state: "empty", emptyReason: "permanent-failure" };
  }

  if (isArxivNotUpdatedRunState(runState)) {
    return { state: "empty", emptyReason: "arxiv-not-updated" };
  }

  if (runnable) return { state: "runnable" };

  if (isCompletedRunState(runState)) {
    return { state: "empty", emptyReason: "report-missing" };
  }

  return emptyReason
    ? { state: "empty", emptyReason }
    : { state: "empty" };
}

export function isCalendarRunWhitelisted(
  input: CalendarRunWhitelistInput,
): boolean {
  if (!input.inLookback) return false;
  if (input.isWeekend) return false;
  if (input.hasDailyReport) return false;
  if (isRunStateBlockedForCalendarRun(input.runState)) return false;

  if (input.date === input.today) {
    return isTimeWithinLocalWindow(
      input.now,
      input.timezone,
      input.runAtLocal,
      input.runUntilLocal,
    );
  }

  return input.recentDates.has(input.date);
}

export interface CalendarEmptyReasonInput {
  date: string;
  today: string;
  trackingStartDate: string;
  recentDates: ReadonlySet<string>;
}

export interface CalendarDailyReportMapInput {
  month: string;
  scannedReports: DailyReportDay[];
  normalizePath: (path: string) => string;
}

export function resolveCalendarEmptyReason(
  input: CalendarEmptyReasonInput,
): CalendarEmptyReason {
  if (input.date > input.today) return "future";
  if (
    input.date < input.trackingStartDate &&
    !input.recentDates.has(input.date)
  ) {
    return "before-tracking";
  }
  return "arxiv-not-updated";
}

export function calendarCellAriaLabel(cell: CalendarCell): string | undefined {
  if (!cell.date) return undefined;
  const date = cell.date;

  if (cell.state === "has-report" && cell.report) {
    return `${date}: open daily report, ${cell.report.papers} indexed papers${cell.report.starred ? `, ${cell.report.starred} starred` : ""}`;
  }

  if (cell.state === "no-relevant-papers") {
    return `${date}: open daily report, no relevant papers`;
  }

  if (cell.state === "runnable") {
    return `${date}: run daily report`;
  }

  if (cell.emptyReason === "arxiv-not-updated") {
    return `${date}: arXiv not updated`;
  }

  if (cell.emptyReason === "report-missing") {
    return `${date}: daily report missing`;
  }

  if (cell.emptyReason === "permanent-failure") {
    return `${date}: daily report failed permanently${cell.failureReason ? `, ${cell.failureReason}` : ""}`;
  }

  if (cell.emptyReason === "future") {
    return `${date}: future date`;
  }

  return date;
}

export function applyEmptyCalendarCellA11y(button: HTMLButtonElement): void {
  button.disabled = true;
  button.setAttribute("tabindex", "-1");
  button.setAttribute("aria-hidden", "true");
}

export function isButtonElement(element: HTMLElement): element is HTMLButtonElement {
  return element.tagName === "BUTTON";
}

export function isArxivNotUpdatedRunState(runState?: RunStateEntry): boolean {
  return (
    runState?.status === "skipped" ||
    (runState?.status === "completed" && runState.papersWritten === 0)
  );
}

export function isRunStateBlockedForCalendarRun(runState?: RunStateEntry): boolean {
  return (
    runState?.status === "running" ||
    runState?.status === "skipped" ||
    runState?.status === "failed_permanent" ||
    (runState?.status === "completed" && runState.papersWritten === 0)
  );
}

export function isCompletedRunState(runState?: RunStateEntry): boolean {
  return runState?.status === "completed";
}

export async function buildCalendarDailyReportMap(
  input: CalendarDailyReportMapInput,
): Promise<Map<string, DailyReportDay>> {
  const out = new Map<string, DailyReportDay>();
  const monthPrefix = `${input.month}-`;

  for (const report of input.scannedReports) {
    if (!report.date.startsWith(monthPrefix)) continue;
    out.set(report.date, {
      ...report,
      path: input.normalizePath(report.path),
    });
  }

  return out;
}
