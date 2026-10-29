import { renderPrompt } from "@arxiv-daily/core";
import injectionGuard from "../../packages/core/src/prompts/injection-guard.md";
import judgeDailySubject from "./prompts/summary-judge.daily.md";
import judgeNoteSubject from "./prompts/summary-judge.note.md";
import judgeTemplate from "./prompts/summary-judge.system.md";
import metaTemplate from "./prompts/summary-meta.system.md";
import { explanationHanShare, MIN_EXPLANATION_HAN_SHARE, normalizeQuoteText, parseJudgeOutput } from "./summary-judge-contract";

export const META_VERDICTS = ["agree", "disagree", "unsure"] as const;
export type MetaVerdict = (typeof META_VERDICTS)[number];
export const META_SEVERITY_OPINIONS = ["ok", "too-high", "too-low"] as const;
export type MetaSeverityOpinion = (typeof META_SEVERITY_OPINIONS)[number];

export interface MetaReview {
  index: number;
  verdict: MetaVerdict;
  severity: MetaSeverityOpinion;
  reason: string;
  evidenceQuote: string | null;
}

/** The text between two headings of the judge's own prompt, so the reviewer judges by the same definitions. */
function judgeSection(heading: string, next: string): string {
  const start = judgeTemplate.indexOf(`## ${heading}`);
  const end = judgeTemplate.indexOf(`## ${next}`, start);
  if (start < 0 || end < 0) throw new Error(`the judge prompt no longer has the sections "${heading}" and "${next}", which the reviewer prompt reuses`);
  return judgeTemplate.slice(start + `## ${heading}`.length, end).trim();
}

/** The reviewer's system prompt for a daily entry or a detail note. */
export function buildMetaSystemPrompt(options: { kind: "daily" | "note" }): string {
  return renderPrompt(metaTemplate, {
    injectionGuard: injectionGuard.trim(),
    subject: (options.kind === "daily" ? judgeDailySubject : judgeNoteSubject).trim(),
    dimensionDefinitions: judgeSection("三个评审维度", "严重程度"),
    severityDefinitions: judgeSection("严重程度", "不要评审的内容"),
  }).trim();
}

export interface MetaContext {
  /** How many judge issues were sent; each index 0..count-1 must be answered once. */
  count: number;
  /** The `<paper_data>` text an evidence quote must come from. */
  paperData: string;
}

export type MetaCheck = { ok: true; reviews: MetaReview[] } | { ok: false; reasons: string[] };

const REVIEW_KEYS = ["index", "verdict", "severity", "reason", "evidenceQuote"] as const;
const MAX_REASONS = 12;
const quoted = (values: readonly string[]): string => values.map((value) => `"${value}"`).join("、");
const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

/** Validate a parsed reviewer reply: every issue index once, strict enums, a Chinese reason, evidence from the paper. */
export function validateMetaReviews(value: unknown, context: MetaContext): MetaCheck {
  const reasons: string[] = [];
  if (!isObject(value)) return { ok: false, reasons: ["输出必须是一个 JSON 对象。"] };
  for (const key of Object.keys(value)) if (key !== "reviews") reasons.push(`根对象有多余的键 "${key}"，只允许 "reviews"。`);
  if (!Array.isArray(value.reviews)) {
    reasons.push('根对象缺少 "reviews" 数组。');
    return { ok: false, reasons };
  }
  const paper = normalizeQuoteText(context.paperData);
  const seen = new Set<number>();
  const kept: MetaReview[] = [];
  value.reviews.forEach((entry: unknown, position) => {
    const path = `reviews[${position}]`;
    if (!isObject(entry)) { reasons.push(`${path} 必须是对象。`); return; }
    for (const key of REVIEW_KEYS) if (!(key in entry)) reasons.push(`${path} 缺少键 "${key}"（reason 与 evidenceQuote 都必须出现，没有引用时 evidenceQuote 写 null）。`);
    for (const key of Object.keys(entry)) if (!(REVIEW_KEYS as readonly string[]).includes(key)) reasons.push(`${path} 有多余的键 "${key}"，只允许 ${quoted(REVIEW_KEYS)}。`);
    const index = entry.index;
    if (typeof index !== "number" || !Number.isInteger(index) || index < 0 || index >= context.count) {
      reasons.push(`${path}.index 的值 ${JSON.stringify(index)} 不对：编号必须是 0 到 ${context.count - 1} 之间的整数。`);
    } else if (seen.has(index)) reasons.push(`${path}.index ${index} 重复出现，每个编号必须恰好出现一次。`);
    else seen.add(index);
    if ("verdict" in entry && !META_VERDICTS.includes(entry.verdict as MetaVerdict)) reasons.push(`${path}.verdict 的值 ${JSON.stringify(entry.verdict)} 不合法，只允许 ${quoted(META_VERDICTS)}。`);
    if ("severity" in entry && !META_SEVERITY_OPINIONS.includes(entry.severity as MetaSeverityOpinion)) reasons.push(`${path}.severity 的值 ${JSON.stringify(entry.severity)} 不合法，只允许 ${quoted(META_SEVERITY_OPINIONS)}。`);
    if ("reason" in entry) {
      if (typeof entry.reason !== "string" || !entry.reason.trim()) reasons.push(`${path}.reason 必须是非空字符串。`);
      else {
        const share = explanationHanShare(entry.reason);
        if (share !== null && share < MIN_EXPLANATION_HAN_SHARE) reasons.push(`${path}.reason 必须用中文书写（专有名词、缩写、公式和引文可保留原文），现在几乎全是外文。`);
      }
    }
    if ("evidenceQuote" in entry && entry.evidenceQuote !== null) {
      if (typeof entry.evidenceQuote !== "string" || !entry.evidenceQuote.trim()) reasons.push(`${path}.evidenceQuote 必须是非空字符串或 null。`);
      else if (!paper.includes(normalizeQuoteText(entry.evidenceQuote))) reasons.push(`${path}.evidenceQuote 不是 <paper_data> 里的原文（必须逐字摘自原文）：${JSON.stringify(entry.evidenceQuote)}`);
    }
    kept.push(entry as unknown as MetaReview);
  });
  for (let index = 0; index < context.count; index += 1) if (!seen.has(index)) reasons.push(`缺少 index ${index} 的复核：每个问题都要复核。`);
  if (reasons.length > 0) return { ok: false, reasons: reasons.slice(0, MAX_REASONS) };
  return { ok: true, reviews: kept.sort((a, b) => a.index - b.index) };
}

export function checkMetaOutput(raw: string, context: MetaContext): MetaCheck {
  const parsed = parseJudgeOutput(raw);
  return parsed.ok ? validateMetaReviews(parsed.value, context) : { ok: false, reasons: [parsed.reason] };
}

export function buildMetaCorrection(reasons: readonly string[]): string {
  return [
    "你上一次的输出没有通过程序校验，原因如下：",
    ...reasons.map((reason) => `- ${reason}`),
    "",
    "请重新复核并只输出完整的 JSON 对象，不要输出任何其他文字。每个问题的 index 恰好出现一次；reason 一律用中文书写，专有名词、缩写和公式可以保留原文；evidenceQuote 必须逐字摘自 <paper_data>，没有就写 null。",
  ].join("\n");
}
