import { renderInlineMarkdownNodes, renderMarkdown } from "../markdown";
import { markdownBody, markdownInline } from "./markdown-dom";
import { generationFooter } from "./generation-footer";
import { h } from "./dom";
import { t } from "./i18n";
import type { WorkbenchPaper, WorkbenchPaperList } from "../papers";
import { calendarStateLabels } from "./calendar";

// Titles and snippets are inside controls: format scientific text without nesting links.
export const scientificInline = (source: string): DocumentFragment => markdownInline(renderInlineMarkdownNodes(source, { resolveLink: () => null }));
const scientificBody = (source: string): HTMLDivElement => markdownBody(renderMarkdown(source).nodes);
export const scopes = { all: "全部论文", inbox: "未标记", to_read: "待读", read: "已读", starred: "收藏" };
const statuses = { inbox: "未标记", to_read: "待读", read: "已读", reading: "阅读中", saved: "已保存", ignored: "已忽略" };
export function marks(paper: WorkbenchPaper, pending = false): HTMLElement {
  const standard = ["inbox", "to_read", "read"];
  return h("div", { class: "paper-marks", "data-key": paper.key },
    h("select", { "data-mark": "status", "aria-label": t("阅读状态 · {0}", paper.title), disabled: pending },
      ...Object.entries(statuses).filter(([key]) => standard.includes(key) || key === paper.status)
        .map(([key, label]) => h("option", { value: key, selected: paper.status === key }, t(label))),
    ),
    h("button", { class: "quiet-button star-button", "data-mark": "star", "aria-label": t("收藏 · {0}", paper.title), "aria-pressed": String(paper.starred), disabled: pending },
      paper.starred ? t("★ 已收藏") : t("☆ 收藏"),
    ),
  );
}
export function paperRows(result: WorkbenchPaperList, pending: Set<string>): HTMLElement[] {
  return result.papers.map(paper => h("section", { class: "paper-row" },
    h("div", { class: "paper-row-meta" }, `${paper.published || t("日期未记录")} · ${paper.category}${paper.arxivId ? ` · ${paper.arxivId}` : ""}`),
    h("button", { class: "paper-title", "data-paper": paper.key }, scientificInline(paper.title)),
    h("div", { class: "paper-authors" }, paper.authors.join(", ")),
    paper.summary?.whyRelevant ? h("p", { class: "paper-reason" }, scientificInline(paper.summary.whyRelevant)) : null,
    h("div", { class: "paper-row-footer" },
      h("span", { class: "paper-topics" },
        paper.topics.join(" · "),
        h("span", { class: "detail-availability" }, paper.detailPath ? t("详细总结已保存") : t("未生成详细总结")),
      ),
      marks(paper, pending.has(paper.key)),
    ),
  ));
}
export function dayHeading(day: WorkbenchPaperList["day"]): HTMLElement | null {
  if (!day) return null;
  return h("section", { class: "day-reading" },
    h("div", null,
      h("span", { class: "day-eyebrow" }, `${day.date} · ${t(calendarStateLabels[day.state])}`),
      h("p", null, t(day.message)),
    ),
    day.reportPath
      ? h("button", { class: "primary-button", "data-action": "read-day" }, t("阅读完整日报 ↗"))
      : day.canGenerate
      ? h("button", { class: "primary-button", "data-action": "generate-date", "data-date": day.date }, t(day.actionLabel || "生成日报"))
      : null,
  );
}
export function sourceLink(url: string | null, label: string, source: string): HTMLAnchorElement | null {
  if (!url || !(/^(?:https?:\/\/|api\/asset\?)/i.test(url))) return null;
  return h("a", { "data-source": source, href: url, target: "_blank", rel: "noopener noreferrer" }, `${t(label)} ↗`);
}
export function overview(paper: WorkbenchPaper, pending: boolean): Node[] {
  const fields = { coreProblem: "核心问题", keyMethod: "关键方法", mainResult: "主要结果", whyRelevant: "推荐理由", limitations: "局限与待核对" };
  const summary = Object.entries(fields).flatMap(([key, label]) => {
    const value = paper.summary?.[key as keyof typeof fields];
    return value ? [h("section", null, h("h2", null, t(label)), scientificBody(value))] : [];
  });
  return [
    h("div", { class: "reading-toolbar" },
      h("button", { class: "quiet-button", "data-action": "back" }, t("← 返回列表")),
      h("span", { class: "reading-kind" }, t("论文概览")),
    ),
    h("div", { class: "article-wrap paper-overview" },
      h("header", { class: "document-header" },
        h("div", { class: "document-eyebrow" }, `${paper.published} · ${paper.arxivId || paper.category}`),
        h("h1", { class: "document-title" }, scientificInline(paper.title)),
        h("p", { class: "document-authors" }, paper.authors.join(", ")),
        marks(paper, pending),
        h("div", { class: "document-links" },
          sourceLink(paper.originalUrl, "原文", "original"),
          sourceLink(paper.pdfUrl, "阅读 PDF", "pdf"),
          paper.detailPath
            ? h("button", { class: "primary-button", "data-document": paper.detailPath }, t("阅读详细总结"))
            : paper.key.startsWith("arxiv:")
            ? h("button", { class: "primary-button", "data-action": "generate-paper" }, t("生成详细总结"))
            : null,
        ),
      ),
      h("div", { class: "overview-content" },
        summary.length ? summary : h("p", { class: "muted" }, t("尚无结构化总结，可阅读原文或已保存的日报。")),
        h("div", { class: "reading-appendix" },
          h("hr"),
          paper.summary?.sourceSections ? h("section", null, h("h2", null, t("总结依据")), scientificBody(paper.summary.sourceSections)) : null,
          paper.provenance
            ? h("section", null, h("h2", null, t("发现来源")), h("p", null, [...paper.provenance.manualTopics.map(topic => topic.name || topic.tag), ...paper.provenance.directions.map(direction => direction.name)].join(" · ")))
            : null,
          paper.novelty
            ? h("section", null, h("h2", null, t("与已有文献的关系")), scientificBody(paper.novelty.explanation), h("small", null, `${t("比较依据：")}${paper.novelty.comparisonBasis.map(item => item.paperKey).join(" · ")}`))
            : null,
          paper.abstract ? h("section", null, h("h2", null, t("原始摘要")), scientificBody(paper.abstract)) : null,
          paper.reports.length
            ? h("section", null, h("h2", null, t("来源日报")),
                h("div", { class: "report-links" },
                  ...paper.reports.map(report => report.available
                    ? h("button", { class: "quiet-button", "data-document": report.path }, `${report.date || report.title} ↗`)
                    : h("span", null, `${report.date || report.title} · ${t("文件已缺失")}`)),
                ))
            : null,
        ),
      ),
      generationFooter(paper.generation?.metrics, paper.generation?.scope),
    ),
  ];
}
