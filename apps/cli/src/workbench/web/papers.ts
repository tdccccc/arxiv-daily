import type { WorkbenchPaper, WorkbenchPaperList } from "../papers";
import { calendarStateLabels } from "./calendar";

export const escapeHtml = (value: string): string => value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!);
const e = escapeHtml;
export const scopes = { all: "全部论文", inbox: "未标记", to_read: "待读", read: "已读", starred: "收藏" };
const statuses = { inbox: "未标记", to_read: "待读", read: "已读", reading: "阅读中", saved: "已保存", ignored: "已忽略" };
export function marks(paper: WorkbenchPaper, pending = false): string {
  const standard = ["inbox", "to_read", "read"];
  return `<div class="paper-marks" data-key="${e(paper.key)}"><select data-mark="status" aria-label="阅读状态 · ${e(paper.title)}" ${pending ? "disabled" : ""}>${Object.entries(statuses).filter(([key]) => standard.includes(key) || key === paper.status).map(([key, label]) => `<option value="${key}" ${paper.status === key ? "selected" : ""}>${label}</option>`).join("")}</select><button class="quiet-button star-button" data-mark="star" aria-label="收藏 · ${e(paper.title)}" aria-pressed="${paper.starred}" ${pending ? "disabled" : ""}>${paper.starred ? "★ 已收藏" : "☆ 收藏"}</button></div>`;
}
export function paperRows(result: WorkbenchPaperList, pending: Set<string>): string {
  return result.papers.map(paper => `<section class="paper-row"><div class="paper-row-meta">${e(paper.published || "日期未记录")} · ${e(paper.category)}${paper.arxivId ? ` · ${e(paper.arxivId)}` : ""}</div><button class="paper-title" data-paper="${e(paper.key)}">${e(paper.title)}</button><div class="paper-authors">${e(paper.authors.join(", "))}</div>${paper.summary?.whyRelevant ? `<p class="paper-reason">${e(paper.summary.whyRelevant)}</p>` : ""}<div class="paper-row-footer"><span class="paper-topics">${e(paper.topics.join(" · "))}<span class="detail-availability">${paper.detailPath ? "详细总结已保存" : "未生成详细总结"}</span></span>${marks(paper, pending.has(paper.key))}</div></section>`).join("");
}
export function dayHeading(day: WorkbenchPaperList["day"]): string {
  if (!day) return "";
  return `<section class="day-reading"><div><span class="day-eyebrow">${e(day.date)} · ${calendarStateLabels[day.state]}</span><p>${e(day.message)}</p></div>${day.reportPath ? '<button class="primary-button" data-action="read-day">阅读完整日报 ↗</button>' : day.canGenerate ? `<button class="primary-button" data-action="generate-date" data-date="${day.date}">${e(day.actionLabel || "生成日报")}</button>` : ""}</section>`;
}
export function sourceLink(url: string | null, label: string, source: string): string {
  if (!url || !(/^(?:https?:\/\/|api\/asset\?)/i.test(url))) return "";
  return `<a data-source="${source}" href="${e(url)}" target="_blank" rel="noopener noreferrer">${label} ↗</a>`;
}
export function overview(paper: WorkbenchPaper, pending: boolean): string {
  const fields = { coreProblem: "核心问题", keyMethod: "关键方法", mainResult: "主要结果", whyRelevant: "推荐理由", limitations: "局限与待核对", sourceSections: "总结依据" };
  const summary = Object.entries(fields).flatMap(([key, label]) => {
    const value = paper.summary?.[key as keyof typeof fields];
    return value ? [`<section><h2>${label}</h2><p>${e(value)}</p></section>`] : [];
  }).join("");
  return `<div class="reading-toolbar"><button class="quiet-button" data-action="back">← 返回列表</button><span class="reading-kind">论文概览</span></div><div class="article-wrap paper-overview"><header class="document-header"><div class="document-eyebrow">${e(paper.published)} · ${e(paper.arxivId || paper.category)}</div><h1 class="document-title">${e(paper.title)}</h1><p class="document-authors">${e(paper.authors.join(", "))}</p>${marks(paper, pending)}<div class="document-links">${sourceLink(paper.originalUrl, "原文", "original")}${sourceLink(paper.pdfUrl, "阅读 PDF", "pdf")}${paper.detailPath ? `<button class="primary-button" data-document="${e(paper.detailPath)}">阅读详细总结</button>` : paper.key.startsWith("arxiv:") ? '<button class="primary-button" data-action="generate-paper">生成详细总结</button>' : ""}</div></header><div class="overview-content">${summary || '<p class="muted">尚无结构化总结，可阅读原文或已保存的日报。</p>'}${paper.provenance ? `<section><h2>发现来源</h2><p>${e([...paper.provenance.manualTopics.map(topic => topic.name || topic.tag), ...paper.provenance.directions.map(direction => direction.name)].join(" · "))}</p></section>` : ""}${paper.novelty ? `<section><h2>与已有文献的关系</h2><p>${e(paper.novelty.explanation)}</p><small>比较依据：${e(paper.novelty.comparisonBasis.map(item => item.paperKey).join(" · "))}</small></section>` : ""}${paper.abstract ? `<section><h2>原始摘要</h2><p>${e(paper.abstract)}</p></section>` : ""}${paper.reports.length ? `<section><h2>来源日报</h2><div class="report-links">${paper.reports.map(report => report.available ? `<button class="quiet-button" data-document="${e(report.path)}">${e(report.date || report.title)} ↗</button>` : `<span>${e(report.date || report.title)} · 文件已缺失</span>`).join("")}</div></section>` : ""}</div></div>`;
}
