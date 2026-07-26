import { DAILY_SUMMARY_FIELD_LABELS } from "../pipeline/daily-summary-rendering";
import { noCategoryPapersText } from "../settings/summary-language";
import type { SummaryLanguage } from "../settings/types";
import type { DailyDigest, DigestPaper } from "./types";
import { digestLanguage } from "./digest";

export function renderEmailSubject(digest: DailyDigest): string {
  const language = digestLanguage(digest);
  const n = digest.paperCount;
  if (language === "en") {
    return `arXiv Daily ${digest.date} · ${n} ${n === 1 ? "paper" : "papers"}`;
  }
  return `arXiv Daily ${digest.date} · ${n} 篇`;
}

export function renderEmailHtml(digest: DailyDigest): string {
  const language = digestLanguage(digest);
  const parts: string[] = [];
  parts.push(`<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>`);
  parts.push(
    `<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;line-height:1.5;color:#111;max-width:40rem;">`,
  );
  parts.push(`<h1 style="font-size:1.25rem;margin:0 0 0.5rem;">${escapeHtml(renderEmailSubject(digest))}</h1>`);
  parts.push(
    `<p style="margin:0 0 1rem;color:#444;">${escapeHtml(headerMetaLine(digest, language))}</p>`,
  );

  if (digest.paperCount === 0) {
    parts.push(`<p style="margin:0 0 1rem;"><strong>${escapeHtml(zeroDayLead(language))}</strong></p>`);
  }

  for (const topic of digest.topics) {
    parts.push(`<h2 style="font-size:1.1rem;margin:1.25rem 0 0.5rem;">${escapeHtml(topic.name)}</h2>`);
    if (topic.papers.length === 0) {
      parts.push(`<p style="margin:0 0 0.75rem;color:#555;">${escapeHtml(noCategoryPapersText(language))}</p>`);
      continue;
    }
    for (const paper of topic.papers) {
      parts.push(renderPaperHtml(paper, language));
    }
  }

  parts.push(
    `<hr style="border:none;border-top:1px solid #ddd;margin:1.5rem 0 0.75rem;" />`,
    `<p style="margin:0;font-size:0.9rem;color:#666;">${escapeHtml(footerLine(digest, language))}</p>`,
    `<p style="margin:0.25rem 0 0;font-size:0.85rem;color:#888;">arXiv Daily</p>`,
    `</div></body></html>`,
  );
  return parts.join("");
}

export function renderEmailText(digest: DailyDigest): string {
  const language = digestLanguage(digest);
  const lines: string[] = [];
  lines.push(renderEmailSubject(digest));
  lines.push(headerMetaLine(digest, language));
  lines.push("");

  if (digest.paperCount === 0) {
    lines.push(zeroDayLead(language));
    lines.push("");
  }

  for (const topic of digest.topics) {
    lines.push(`## ${topic.name}`);
    if (topic.papers.length === 0) {
      lines.push(noCategoryPapersText(language));
      lines.push("");
      continue;
    }
    for (const paper of topic.papers) {
      lines.push(...renderPaperText(paper, language));
      lines.push("");
    }
  }

  lines.push("---");
  lines.push(footerLine(digest, language));
  lines.push("arXiv Daily");
  return lines.join("\n").trimEnd() + "\n";
}

function headerMetaLine(digest: DailyDigest, language: SummaryLanguage): string {
  if (language === "en") {
    return `Date: ${digest.date} · ${digest.paperCount} papers · Categories: ${digest.categories}`;
  }
  return `日期：${digest.date} · ${digest.paperCount} 篇 · 分类：${digest.categories}`;
}

function zeroDayLead(language: SummaryLanguage): string {
  return language === "en" ? "No relevant papers today." : "今日无相关论文";
}

function footerLine(digest: DailyDigest, language: SummaryLanguage): string {
  return language === "en"
    ? `Full Markdown daily report: ${digest.dailyPath}`
    : `完整 Markdown 日报路径：${digest.dailyPath}`;
}

function renderPaperHtml(paper: DigestPaper, language: SummaryLanguage): string {
  const chunks: string[] = [];
  chunks.push(`<section style="margin:0 0 1.25rem;">`);
  chunks.push(`<h3 style="font-size:1rem;margin:0 0 0.35rem;">${escapeHtml(paper.title)}</h3>`);
  if (paper.sourceSections) {
    const label = language === "en" ? "Source sections:" : "信息来源：";
    chunks.push(
      `<p style="margin:0 0 0.35rem;color:#555;"><em>${escapeHtml(label)} ${escapeHtml(paper.sourceSections)}</em></p>`,
    );
  }
  const authorLabel = language === "en" ? "Authors" : "作者";
  chunks.push(
    `<p style="margin:0 0 0.35rem;">${escapeHtml(authorLabel)}: ${escapeHtml(paper.authors)}</p>`,
  );
  chunks.push(
    `<p style="margin:0 0 0.5rem;">` +
      `<a href="${escapeHtml(paper.absUrl)}">arXiv</a>` +
      ` · ` +
      `<a href="${escapeHtml(paper.pdfUrl)}">PDF</a>` +
      `</p>`,
  );

  if (paper.kind === "structured" && paper.fields) {
    chunks.push(`<ul style="margin:0;padding-left:1.2rem;">`);
    for (const [key, label] of DAILY_SUMMARY_FIELD_LABELS[language]) {
      const value = paper.fields[key];
      chunks.push(
        `<li style="margin:0 0 0.25rem;"><strong>${escapeHtml(label)}</strong>: ${escapeHtml(value)}</li>`,
      );
    }
    chunks.push(`</ul>`);
  } else {
    const warning =
      language === "en"
        ? "Summary unavailable. Read the original paper on arXiv."
        : "自动摘要不可用。请直接阅读 arXiv 原文。";
    chunks.push(`<p style="margin:0 0 0.35rem;"><strong>${escapeHtml(warning)}</strong></p>`);
    const abstractLabel = language === "en" ? "Original abstract" : "原始摘要";
    const abstract =
      paper.abstract?.trim() ||
      (language === "en" ? "Unavailable." : "不可用。");
    chunks.push(
      `<p style="margin:0;"><strong>${escapeHtml(abstractLabel)}</strong>: ${escapeHtml(abstract)}</p>`,
    );
  }

  chunks.push(`</section>`);
  return chunks.join("");
}

function renderPaperText(paper: DigestPaper, language: SummaryLanguage): string[] {
  const lines: string[] = [];
  lines.push(`### ${paper.title}`);
  if (paper.sourceSections) {
    const label = language === "en" ? "Source sections:" : "信息来源：";
    lines.push(`${label} ${paper.sourceSections}`);
  }
  const authorLabel = language === "en" ? "Authors" : "作者";
  lines.push(`${authorLabel}: ${paper.authors}`);
  lines.push(`arXiv: ${paper.absUrl}`);
  lines.push(`PDF: ${paper.pdfUrl}`);
  if (paper.kind === "structured" && paper.fields) {
    for (const [key, label] of DAILY_SUMMARY_FIELD_LABELS[language]) {
      lines.push(`- ${label}: ${paper.fields[key]}`);
    }
  } else {
    const warning =
      language === "en"
        ? "Summary unavailable. Read the original paper on arXiv."
        : "自动摘要不可用。请直接阅读 arXiv 原文。";
    lines.push(warning);
    const abstractLabel = language === "en" ? "Original abstract" : "原始摘要";
    const abstract =
      paper.abstract?.trim() ||
      (language === "en" ? "Unavailable." : "不可用。");
    lines.push(`- ${abstractLabel}: ${abstract}`);
  }
  return lines;
}

/** Escape prose for email HTML; keep TeX `$...$` as literal text. */
export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
