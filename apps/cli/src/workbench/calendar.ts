import { calendarCells, createStorageStateStore, formatDate, shiftMonth, todayInTz, type RunStateEntry } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "../config";
import { WorkbenchError, type DocumentEntry, type WorkbenchDocuments } from "./documents";
import type { WorkbenchRun } from "./server";

export interface WorkbenchCalendarDay {
  date: string;
  state: "has-report" | "not-generated" | "running" | "failed" | "no-matches" | "skipped" | "report-missing" | "future";
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
  const store = createStorageStateStore(new NodeStorageAdapter(config.vaultRoot), config.settings.output);
  const [entries] = await Promise.all([documents.list(), store.load()]);
  const states = store.snapshot();
  const reports = new Map<string, DocumentEntry>();
  for (const entry of entries) if (entry.kind === "daily" && !reports.has(entry.date)) reports.set(entry.date, entry);
  return {
    month, today, timezone,
    previousMonth: month === "0100-01" ? month : shiftMonth(month, -1).padStart(7, "0"),
    nextMonth: month === "9999-12" ? month : shiftMonth(month, 1).padStart(7, "0"),
    cells: calendarCells(month).map(cell => cell.date ? resolveDay(cell.date, today, reports.get(cell.date), states[cell.date], ownedRun) : null),
  };
}

function resolveDay(date: string, today: string, report: DocumentEntry | undefined, state: RunStateEntry | undefined, ownedRun: WorkbenchRun | null): WorkbenchCalendarDay {
  const papers = typeof state?.papersWritten === "number" && Number.isSafeInteger(state.papersWritten) && state.papersWritten >= 0 ? state.papersWritten : null;
  const base = { date, reportPath: report?.path ?? null, reportTitle: report?.title ?? null, papers, canGenerate: false, actionLabel: null };
  if (report) return { ...base, state: "has-report", message: "日报已保存，可以打开阅读。" };
  if ((ownedRun?.status === "running" && ownedRun.date === date) || state?.status === "running") {
    return { ...base, state: "running", message: "该日期的日报任务正在运行。" };
  }
  if (date > today) return { ...base, state: "future", message: "未来日期，暂不可生成日报。" };
  if (state?.status === "completed") {
    return state.papersWritten === 0
      ? { ...base, state: "no-matches", message: "已完成筛选，没有匹配论文，因此没有生成日报文件。" }
      : { ...base, state: "report-missing", message: "运行记录显示已完成，但日报文件已缺失；重复运行会跳过此日期。" };
  }
  if (state?.status === "failed_transient") return { ...base, state: "failed", canGenerate: true, actionLabel: "重试生成", message: state.error || "生成暂时失败，可以重试。" };
  if (state?.status === "failed_permanent") return { ...base, state: "failed", message: state.error || "生成已停止，当前流程不会再次生成此日期。" };
  if (state?.status === "skipped") return { ...base, state: "skipped", message: state.error || "该日期已被现有流程跳过。" };
  return { ...base, state: "not-generated", canGenerate: true, actionLabel: "生成日报", message: state?.error || "尚未生成日报；开始生成后，由现有流程检查该日期。" };
}
