import { renderInlineMarkdown, renderMarkdown } from "../markdown";
import { generationFooter } from "./generation-footer";
import { t } from "./i18n";
import type { WorkbenchPaper, WorkbenchPaperList } from "../papers";
import { calendarStateLabels } from "./calendar";

export const escapeHtml = (value: string): string => value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!);
const e = escapeHtml;
// Titles and snippets are inside controls: format scientific text without nesting links.
export const scientificInline = (source: string): string => renderInlineMarkdown(source, { resolveLink: () => null });
const scientificBody = (source: string): string => `<div class="markdown-body">${renderMarkdown(source).html}</div>`;
export const scopes = { all: "全部论文", inbox: "未标记", to_read: "待读", read: "已读", starred: "收藏" };
const statuses = { inbox: "未标记", to_read: "待读", read: "已读", reading: "阅读中", saved: "已保存", ignored: "已忽略" };
export function marks(paper: WorkbenchPaper, pending = false): string {
  const standard = ["inbox", "to_read", "read"];
  return `<div class="paper-marks" data-key="${e(paper.key)}"><select data-mark="status" aria-label="${e(t("阅读状态 · {0}", paper.title))}" ${pending ? "disabled" : ""}>${Object.entries(statuses).filter(([key]) => standard.includes(key) || key === paper.status).map(([key, label]) => `<option value="${key}" ${paper.status === key ? "selected" : ""}>${e(t(label))}</option>`).join("")}</select><button class="quiet-button star-button" data-mark="star" aria-label="${e(t("收藏 · {0}", paper.title))}" aria-pressed="${paper.starred}" ${pending ? "disabled" : ""}>${paper.starred ? t("★ 已收藏") : t("☆ 收藏")}</button></div>`;
}
export function paperRows(result: WorkbenchPaperList, pending: Set<string>): string {
  return result.papers.map(paper => `<section class="paper-row"><div class="paper-row-meta">${e(paper.published || t("日期未记录"))} · ${e(paper.category)}${paper.arxivId ? ` · ${e(paper.arxivId)}` : ""}</div><button class="paper-title" data-paper="${e(paper.key)}">${scientificInline(paper.title)}</button><div class="paper-authors">${e(paper.authors.join(", "))}</div>${paper.summary?.whyRelevant ? `<p class="paper-reason">${scientificInline(paper.summary.whyRelevant)}</p>` : ""}<div class="paper-row-footer"><span class="paper-topics">${e(paper.topics.join(" · "))}<span class="detail-availability">${paper.detailPath ? t("详细总结已保存") : t("未生成详细总结")}</span></span>${marks(paper, pending.has(paper.key))}</div></section>`).join("");
}
export function dayHeading(day: WorkbenchPaperList["day"]): string {
  if (!day) return "";
  return `<section class="day-reading"><div><span class="day-eyebrow">${e(day.date)} · ${e(t(calendarStateLabels[day.state]))}</span><p>${e(t(day.message))}</p></div>${day.reportPath ? `<button class="primary-button" data-action="read-day">${t("阅读完整日报 ↗")}</button>` : day.canGenerate ? `<button class="primary-button" data-action="generate-date" data-date="${day.date}">${e(t(day.actionLabel || "生成日报"))}</button>` : ""}</section>`;
}
export function sourceLink(url: string | null, label: string, source: string): string {
  if (!url || !(/^(?:https?:\/\/|api\/asset\?)/i.test(url))) return "";
  return `<a data-source="${source}" href="${e(url)}" target="_blank" rel="noopener noreferrer">${e(t(label))} ↗</a>`;
}
export function overview(paper: WorkbenchPaper, pending: boolean): string {
  const fields = { coreProblem: "核心问题", keyMethod: "关键方法", mainResult: "主要结果", whyRelevant: "推荐理由", limitations: "局限与待核对" };
  const summary = Object.entries(fields).flatMap(([key, label]) => {
    const value = paper.summary?.[key as keyof typeof fields];
    return value ? [`<section><h2>${e(t(label))}</h2>${scientificBody(value)}</section>`] : [];
  }).join("");
  return `<div class="reading-toolbar"><button class="quiet-button" data-action="back">${t("← 返回列表")}</button><span class="reading-kind">${t("论文概览")}</span></div><div class="article-wrap paper-overview"><header class="document-header"><div class="document-eyebrow">${e(paper.published)} · ${e(paper.arxivId || paper.category)}</div><h1 class="document-title">${scientificInline(paper.title)}</h1><p class="document-authors">${e(paper.authors.join(", "))}</p>${marks(paper, pending)}<div class="document-links">${sourceLink(paper.originalUrl, "原文", "original")}${sourceLink(paper.pdfUrl, "阅读 PDF", "pdf")}${paper.detailPath ? `<button class="primary-button" data-document="${e(paper.detailPath)}">${t("阅读详细总结")}</button>` : paper.key.startsWith("arxiv:") ? `<button class="primary-button" data-action="generate-paper">${t("生成详细总结")}</button>` : ""}</div></header><div class="overview-content">${summary || `<p class="muted">${t("尚无结构化总结，可阅读原文或已保存的日报。")}</p>`}<div class="reading-appendix"><hr>${paper.summary?.sourceSections ? `<section><h2>${e(t("总结依据"))}</h2>${scientificBody(paper.summary.sourceSections)}</section>` : ""}${paper.provenance ? `<section><h2>${t("发现来源")}</h2><p>${e([...paper.provenance.manualTopics.map(topic => topic.name || topic.tag), ...paper.provenance.directions.map(direction => direction.name)].join(" · "))}</p></section>` : ""}${paper.novelty ? `<section><h2>${t("与已有文献的关系")}</h2>${scientificBody(paper.novelty.explanation)}<small>${t("比较依据：")}${e(paper.novelty.comparisonBasis.map(item => item.paperKey).join(" · "))}</small></section>` : ""}${paper.abstract ? `<section><h2>${t("原始摘要")}</h2>${scientificBody(paper.abstract)}</section>` : ""}${paper.reports.length ? `<section><h2>${t("来源日报")}</h2><div class="report-links">${paper.reports.map(report => report.available ? `<button class="quiet-button" data-document="${e(report.path)}">${e(report.date || report.title)} ↗</button>` : `<span>${e(report.date || report.title)} · ${t("文件已缺失")}</span>`).join("")}</div></section>` : ""}</div></div>${generationFooter(paper.generation?.metrics, paper.generation?.scope)}</div>`;
}
