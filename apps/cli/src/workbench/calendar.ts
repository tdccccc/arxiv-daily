import { isWeekendReportDate, isLegacyWeekendAnnouncementGap, WEEKEND_REPORT_MESSAGE } from "./announcement-calendar";
import { calendarCells, createStorageStateStore, formatDate, PaperIndexStore, shiftMonth, todayInTz, type RunStateEntry } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "../config";
import { WorkbenchError, type DocumentEntry, type WorkbenchDocuments } from "./documents";
import type { WorkbenchRun } from "./server";

export interface WorkbenchCalendarDay {
  date: string;
  state: "has-report" | "not-generated" | "running" | "failed" | "no-matches" | "no-updates" | "awaiting-announcement" | "skipped" | "report-missing" | "future";
  reportPath: string | null;
  reportTitle: string | null;
  papers: number | null;
  message: string;
  canGenerate: boolean;
  actionLabel: string | null;
}

export interface WorkbenchCalendar {
  month: string;
  today: string;
  timezone: string;
  previousMonth: string;
  nextMonth: string;
  cells: Array<WorkbenchCalendarDay | null>;
}

/** Read complete persisted date state; browsing never performs scheduler recovery or generation. */
export async function inspectCalendar(
  config: CliRuntimeConfig,
  documents: Pick<WorkbenchDocuments, "list">,
  requestedMonth: string | null,
  now: Date,
  ownedRun: WorkbenchRun | null,
): Promise<WorkbenchCalendar> {
  const timezone = config.settings.arxiv.timezone;
  const today = formatDate(todayInTz(now, timezone));
  const month = requestedMonth ?? today.slice(0, 7);
  // Existing shared arithmetic accepts loose inputs; the HTTP surface accepts real months only.
  if (!/^\d{4}-(?:0[1-9]|1[0-2])$/.test(month) || Number(month.slice(0, 4)) < 100) {
    throw new WorkbenchError(400, "月份无效，请使用 YYYY-MM 格式（年份 0100–9999）。");
  }
  const storage = new NodeStorageAdapter(config.vaultRoot);
  const store = createStorageStateStore(storage, config.settings.output);
  const [entries] = await Promise.all([documents.list(), store.load()]);
  const states = store.snapshot();
  const reports = new Map<string, DocumentEntry>();
  for (const entry of entries) if (entry.kind === "daily" && entry.date.startsWith(`${month}-`) && !reports.has(entry.date)) reports.set(entry.date, entry);
  const indexedCounts = new Map<string, number>();
  const missingCounts = new Set([...reports.values()].filter(report => recordedCount(states[report.date]) === null).map(report => storage.normalizePath(report.path)));
  if (missingCounts.size) {
    try {
      const { inbox } = await new PaperIndexStore(storage, config.settings.output).inspect();
      for (const entry of Object.values(inbox.papers)) {
        for (const reportPath of new Set(entry.dailyReports.map(value => storage.normalizePath(value)))) {
          if (missingCounts.has(reportPath)) indexedCounts.set(reportPath, (indexedCounts.get(reportPath) ?? 0) + 1);
        }
      }
    } catch { /* Optional index counts must not prevent reading saved reports. */ }
  }
  return {
    month, today, timezone,
    previousMonth: month === "0100-01" ? month : shiftMonth(month, -1).padStart(7, "0"),
    nextMonth: month === "9999-12" ? month : shiftMonth(month, 1).padStart(7, "0"),
    cells: calendarCells(month).map(cell => {
      if (!cell.date) return null;
      const report = reports.get(cell.date);
      return resolveDay(cell.date, today, report, states[cell.date], ownedRun, report ? indexedCounts.get(storage.normalizePath(report.path)) : undefined);
    }),
  };
}

function recordedCount(state: RunStateEntry | undefined): number | null {
  return typeof state?.papersWritten === "number" && Number.isSafeInteger(state.papersWritten) && state.papersWritten >= 0 ? state.papersWritten : null;
}

function resolveDay(date: string, today: string, report: DocumentEntry | undefined, state: RunStateEntry | undefined, ownedRun: WorkbenchRun | null, indexedCount?: number): WorkbenchCalendarDay {
  const papers = recordedCount(state) ?? indexedCount ?? null;
  const base = { date, reportPath: report?.path ?? null, reportTitle: report?.title ?? null, papers, canGenerate: false, actionLabel: null };
  if (report) return { ...base, state: "has-report", message: "日报已保存，可以打开阅读。" };
  if ((ownedRun?.status === "running" && ownedRun.date === date) || state?.status === "running") {
    return { ...base, state: "running", message: "该日期的日报任务正在运行。" };
  }
  if (date > today) return { ...base, state: "future", message: "未来日期，暂不可生成日报。" };
  const persistedOutcome = state?.status === "pending" || state?.status === "completed" || state?.status === "skipped" ? state.outcome : undefined;
  const outcome = state ? persistedOutcome
    : ownedRun?.date === date && (ownedRun.status === "pending" || ownedRun.status === "completed") ? ownedRun.outcome : undefined;
  if (outcome === "awaiting_announcement") return { ...base, papers: null, state: "awaiting-announcement", canGenerate: true, actionLabel: "重新检查公告", message: "该日期的公告尚未发布，可以稍后重新检查。" };
  if (outcome === "no_updates") return { ...base, papers: 0, state: "no-updates", message: "已确认该日期所选分类没有新论文。" };
  if (outcome === "no_matches") return { ...base, papers: 0, state: "no-matches", message: "已完成筛选，没有匹配论文，因此没有生成日报文件。" };
  if (state?.status === "completed") {
    return state.papersWritten === 0
      ? { ...base, state: "no-matches", message: "已完成筛选，没有匹配论文，因此没有生成日报文件。" }
      : { ...base, state: "report-missing", message: "运行记录显示已完成，但日报文件已缺失；重复运行会跳过此日期。" };
  }
  if (!state?.outcome && !outcome && isWeekendReportDate(date) && (!state || state.status === "pending" && !state.error || isLegacyWeekendAnnouncementGap(date, state.error))) {
    return { ...base, state: "skipped", message: WEEKEND_REPORT_MESSAGE };
  }
  if (state?.status === "failed_transient") return { ...base, state: "failed", canGenerate: true, actionLabel: "重试生成", message: state.error || "生成暂时失败，可以重试。" };
  if (state?.status === "failed_permanent") return { ...base, state: "failed", canGenerate: true, actionLabel: "重试生成", message: state.error || "生成已停止，修正设置后可以手动重试。" };
  if (state?.status === "skipped") return { ...base, state: "skipped", message: state.error || "该日期已被现有流程跳过。" };
  return { ...base, state: "not-generated", canGenerate: true, actionLabel: "生成日报", message: state?.error || "尚未生成日报；开始生成后，由现有流程检查该日期。" };
}
