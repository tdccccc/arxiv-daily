import { t } from "./i18n";
import type { WorkbenchRun } from "../server";

const statusLabels = { pending: "等待公告发布", running: "正在运行", completed: "已完成", failed: "生成失败", cancelled: "已取消", skipped: "已跳过" } as const;
const outcomeLabels = { awaiting_announcement: "等待公告发布", no_updates: "当日无更新", no_matches: "无匹配论文", papers_written: "日报已保存" } as const;

/** A run in words: what it produced once it has an outcome, otherwise where it stands. */
export function runStateLabel(run: WorkbenchRun): string {
  return t(run.outcome && (run.status === "pending" || run.status === "completed") ? outcomeLabels[run.outcome] : statusLabels[run.status]);
}
