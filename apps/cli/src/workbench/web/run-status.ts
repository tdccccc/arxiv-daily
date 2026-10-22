import { t } from "./i18n";
import { h } from "./dom";
import type { WorkbenchRun } from "../server";

const statusLabels = { pending: "等待公告发布", running: "正在运行", completed: "已完成", failed: "生成失败", cancelled: "已取消", skipped: "已跳过" } as const;
const outcomeLabels = { awaiting_announcement: "等待公告发布", no_updates: "当日无更新", no_matches: "无匹配论文", papers_written: "日报已保存" } as const;

/** A run in words: what it produced once it has an outcome, otherwise where it stands. */
export function runStateLabel(run: WorkbenchRun): string {
  return t(run.outcome && (run.status === "pending" || run.status === "completed") ? outcomeLabels[run.outcome] : statusLabels[run.status]);
}

/** A day whose announcement is awaited, with the newest announced day known: say so and offer it in one click. */
export function announcementOffer(date: string, latestAnnounced: string): HTMLElement {
  return h("div", { class: "announcement-offer" },
    h("p", null, t("{0} 的论文还没公告，最近一次公告是 {1}。", date, latestAnnounced)),
    h("button", { class: "quiet-button", "data-action": "generate-announced", "data-date": latestAnnounced }, t("改为生成 {0} 的日报", latestAnnounced)),
  );
}
