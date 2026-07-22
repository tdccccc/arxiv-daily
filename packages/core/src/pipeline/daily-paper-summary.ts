import {
  isLlmTransientExhaustedError,
  type LlmClient,
} from "../llm/client";
import type { MetricsObserver } from "../metrics/generation";
import dailyPaperSummaryTemplateEn from "../prompts/daily-paper-summary.en.system.md";
import dailyPaperSummaryTemplate from "../prompts/daily-paper-summary.system.md";
import injectionGuard from "../prompts/injection-guard.md";
import { renderPrompt } from "../prompts/render";
import {
  isCancellationError,
  throwIfCancelled,
} from "../services/cancellation";
import { normalizeSummaryLanguage } from "../settings/summary-language";
import type { SummaryLanguage } from "../settings/types";
import type {
  DailyPaperResult,
  StructuredPaperSummary,
} from "./daily-summary-assembler";
import { escapePaperDataFence } from "./prompt-safety";

export interface DailyPaperSummaryInput {
  id: string;
  title: string;
  authors: string;
  abstract?: string;
  abstractConclusion: string;
  fullSections: string | null;
}

export interface DailyPaperSummaryDeps {
  llm: LlmClient;
  summaryLanguage?: SummaryLanguage;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
  previousValidationError?: DailyPaperSummaryValidationError;
}

export const DAILY_PAPER_SUMMARY_VALIDATION_ERROR_CODE =
  "ARXIV_DAILY_PAPER_SUMMARY_VALIDATION" as const;

export type DailyPaperSummaryValidationReasonCode =
  | "invalid-json"
  | "invalid-root"
  | "invalid-keys"
  | "invalid-id-type"
  | "mismatched-id"
  | "invalid-field-type"
  | "empty-field";

export class DailyPaperSummaryValidationError extends Error {
  readonly name = "DailyPaperSummaryValidationError";
  readonly code = DAILY_PAPER_SUMMARY_VALIDATION_ERROR_CODE;
  readonly paperId: string;
  readonly reasonCode?: DailyPaperSummaryValidationReasonCode;
  readonly fieldKey?: keyof Omit<StructuredPaperSummary, "id">;

  constructor(
    paperId: string,
    message: string,
    details: {
      reasonCode?: DailyPaperSummaryValidationReasonCode;
      fieldKey?: keyof Omit<StructuredPaperSummary, "id">;
    } = {},
  ) {
    super(message);
    this.paperId = paperId;
    this.reasonCode = details.reasonCode;
    this.fieldKey = details.fieldKey;
  }
}

export function isDailyPaperSummaryValidationError(
  error: unknown,
): error is DailyPaperSummaryValidationError {
  if (error instanceof DailyPaperSummaryValidationError) return true;
  return isErrorLike(error) &&
    error.name === "DailyPaperSummaryValidationError" &&
    error.code === DAILY_PAPER_SUMMARY_VALIDATION_ERROR_CODE &&
    typeof error.paperId === "string" &&
    Boolean(error.paperId);
}

export const DAILY_PAPER_SUMMARY_MAX_ATTEMPTS = 3;

const SUMMARY_KEYS = [
  "id",
  "coreProblem",
  "keyMethod",
  "mainResult",
  "whyRelevant",
  "limitations",
] as const satisfies ReadonlyArray<keyof StructuredPaperSummary>;

const SEMANTIC_KEYS = SUMMARY_KEYS.filter(
  (key): key is Exclude<(typeof SUMMARY_KEYS)[number], "id"> => key !== "id",
);
export async function summarizeDailyPaper(
  paper: DailyPaperSummaryInput,
  deps: DailyPaperSummaryDeps,
): Promise<StructuredPaperSummary> {
  throwIfCancelled(deps.signal);
  const language = normalizeSummaryLanguage(deps.summaryLanguage);
  const systemPrompt = renderPrompt(
    language === "en" ? dailyPaperSummaryTemplateEn : dailyPaperSummaryTemplate,
    { injectionGuard },
  );
  const userContent = [
    buildPaperData(paper, language),
    deps.previousValidationError
      ? buildCorrectionGuidance(deps.previousValidationError, language)
      : "",
  ]
    .filter(Boolean)
    .join("\n\n");
  const raw = await deps.llm.call(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userContent },
    ],
    { temperature: 0, signal: deps.signal, onMetrics: deps.onMetrics },
  );
  throwIfCancelled(deps.signal);
  return parseDailyPaperSummary(raw, paper.id);
}

export async function summarizeDailyPaperWithValidationRetry(
  paper: DailyPaperSummaryInput,
  deps: Omit<DailyPaperSummaryDeps, "previousValidationError">,
): Promise<DailyPaperResult> {
  let previousValidationError: DailyPaperSummaryValidationError | undefined;

  for (let attempt = 1; attempt <= DAILY_PAPER_SUMMARY_MAX_ATTEMPTS; attempt += 1) {
    throwIfCancelled(deps.signal);
    try {
      const summary = await summarizeDailyPaper(paper, {
        ...deps,
        previousValidationError,
      });
      return { kind: "structured", summary };
    } catch (error) {
      if (isCancellationError(error)) throw error;
      if (isDailyPaperSummaryValidationError(error) && error.paperId === paper.id) {
        previousValidationError = error;
        if (attempt < DAILY_PAPER_SUMMARY_MAX_ATTEMPTS) continue;
        return {
          kind: "fallback",
          reasonCode: "validation-exhausted",
          attempts: attempt,
          originalAbstract: resolveTrustedOriginalAbstract(paper),
        };
      }
      if (isLlmTransientExhaustedError(error)) {
        return {
          kind: "fallback",
          reasonCode: "transport-exhausted",
          attempts: attempt,
          originalAbstract: resolveTrustedOriginalAbstract(paper),
        };
      }
      throw error;
    }
  }

  return {
    kind: "fallback",
    reasonCode: "validation-exhausted",
    attempts: DAILY_PAPER_SUMMARY_MAX_ATTEMPTS,
    originalAbstract: resolveTrustedOriginalAbstract(paper),
  };
}

export function derivePaperSourceSections(
  paper: Pick<DailyPaperSummaryInput, "abstractConclusion" | "fullSections">,
): string {
  const titles = [
    ...extractSectionTitles(paper.abstractConclusion),
    ...extractSectionTitles(paper.fullSections),
  ].filter((title, index, values) => values.indexOf(title) === index);
  if (titles.length > 0) return titles.join(", ");
  if (paper.abstractConclusion.startsWith("[获取失败]")) return "获取失败";
  return "正文摘录";
}

export function resolveTrustedOriginalAbstract(
  paper: Pick<DailyPaperSummaryInput, "abstract" | "abstractConclusion">,
): string {
  if (typeof paper.abstract === "string" && paper.abstract.trim()) {
    return paper.abstract.trim();
  }
  return extractAbstractSection(paper.abstractConclusion);
}

function buildPaperData(
  paper: DailyPaperSummaryInput,
  language: SummaryLanguage,
): string {
  const content = [paper.abstractConclusion.trim()];
  if (paper.fullSections?.trim()) {
    content.push(
      language === "en"
        ? `Full-text excerpts:\n${paper.fullSections.trim()}`
        : `正文摘录：\n${paper.fullSections.trim()}`,
    );
  }
  const sourceSections = derivePaperSourceSections(paper);
  return (
    `<paper_data>\n` +
    `ID: ${escapePaperDataFence(paper.id)}\n` +
    `Title: ${escapePaperDataFence(paper.title)}\n` +
    `Authors: ${escapePaperDataFence(paper.authors)}\n` +
    `Source sections: ${escapePaperDataFence(sourceSections)}\n\n` +
    `${escapePaperDataFence(content.filter(Boolean).join("\n\n"))}\n` +
    `</paper_data>`
  );
}

function buildCorrectionGuidance(
  error: DailyPaperSummaryValidationError,
  language: SummaryLanguage,
): string {
  const keys = SUMMARY_KEYS.join(", ");
  const reason = trustedCorrectionReason(error, language);
  if (language === "en") {
    return (
      `Previous response failed validation: ${reason}\n` +
      `Return only strict JSON with exactly these keys: ${keys}.\n` +
      `The id field must equal the ID in paper_data.\n` +
      `Every string field must be non-empty after trimming.\n` +
      `Do not wrap the JSON in Markdown fences or add any other text.`
    );
  }
  return (
    `上一次响应未通过校验：${reason}\n` +
    `只返回严格 JSON，且恰好包含这些键：${keys}。\n` +
    `id 字段必须等于 paper_data 中的 ID。\n` +
    `每个字符串字段在 trim 后都必须非空。\n` +
    `不要用 Markdown 代码块包裹 JSON，也不要附加其他文本。`
  );
}

function trustedCorrectionReason(
  error: DailyPaperSummaryValidationError,
  language: SummaryLanguage,
): string {
  if (!(error instanceof DailyPaperSummaryValidationError)) {
    return language === "en" ? "the response was invalid" : "响应格式或内容无效";
  }
  const field = error.fieldKey && SEMANTIC_KEYS.includes(error.fieldKey)
    ? error.fieldKey
    : undefined;
  if (language === "en") {
    switch (error.reasonCode) {
      case "invalid-json": return "the response was not strict JSON";
      case "invalid-root": return "the JSON root was not an object";
      case "invalid-keys": return "the JSON keys did not match the required schema";
      case "invalid-id-type": return "the id field was not a string";
      case "mismatched-id": return "the id field did not match the requested paper";
      case "invalid-field-type": return field ? `${field} was not a string` : "a semantic field was not a string";
      case "empty-field": return field ? `${field} was empty` : "a semantic field was empty";
      default: return "the response was invalid";
    }
  }
  switch (error.reasonCode) {
    case "invalid-json": return "响应不是严格 JSON";
    case "invalid-root": return "JSON 根值不是对象";
    case "invalid-keys": return "JSON 键与要求的结构不一致";
    case "invalid-id-type": return "id 字段不是字符串";
    case "mismatched-id": return "id 字段与请求的论文不匹配";
    case "invalid-field-type": return field ? `${field} 不是字符串` : "某个语义字段不是字符串";
    case "empty-field": return field ? `${field} 为空` : "某个语义字段为空";
    default: return "响应格式或内容无效";
  }
}

function parseDailyPaperSummary(
  raw: string,
  expectedId: string,
): StructuredPaperSummary {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new DailyPaperSummaryValidationError(
      expectedId,
      `summarizeDailyPaper: response for ${expectedId} is not strict JSON`,
      { reasonCode: "invalid-json" },
    );
  }
  if (!isPlainObject(value)) {
    throw new DailyPaperSummaryValidationError(
      expectedId,
      `summarizeDailyPaper: response for ${expectedId} must be a plain object`,
      { reasonCode: "invalid-root" },
    );
  }
  if (!hasExactKeys(value, SUMMARY_KEYS)) {
    throw new DailyPaperSummaryValidationError(
      expectedId,
      `summarizeDailyPaper: response for ${expectedId} must contain exactly: ${SUMMARY_KEYS.join(", ")}`,
      { reasonCode: "invalid-keys" },
    );
  }
  if (typeof value.id !== "string") {
    throw new DailyPaperSummaryValidationError(
      expectedId,
      `summarizeDailyPaper: id for ${expectedId} must be a string`,
      { reasonCode: "invalid-id-type" },
    );
  }
  if (value.id !== expectedId) {
    throw new DailyPaperSummaryValidationError(
      expectedId,
      `summarizeDailyPaper: response ID does not match ${expectedId}`,
      { reasonCode: "mismatched-id" },
    );
  }

  const summary = { id: expectedId } as StructuredPaperSummary;
  for (const key of SEMANTIC_KEYS) {
    const field = value[key];
    if (typeof field !== "string") {
      throw new DailyPaperSummaryValidationError(
        expectedId,
        `summarizeDailyPaper: ${key} for ${expectedId} must be a string`,
        { reasonCode: "invalid-field-type", fieldKey: key },
      );
    }
    const trimmed = field.trim();
    if (!trimmed) {
      throw new DailyPaperSummaryValidationError(
        expectedId,
        `summarizeDailyPaper: ${key} for ${expectedId} must be non-empty`,
        { reasonCode: "empty-field", fieldKey: key },
      );
    }
    summary[key] = trimmed;
  }
  return summary;
}

function extractSectionTitles(markdown: string | null | undefined): string[] {
  if (!markdown) return [];
  const titles: string[] = [];
  const heading = /^##\s+(.+)$/gm;
  let match: RegExpExecArray | null;
  while ((match = heading.exec(markdown)) !== null) {
    const title = match[1]?.trim() ?? "";
    if (title && !titles.includes(title)) titles.push(title);
  }
  return titles;
}

function extractAbstractSection(markdown: string | null | undefined): string {
  if (!markdown) return "";
  const match =
    /^##\s+(?:Abstract|摘要)\s*\n([\s\S]*?)(?=^##\s+|$)/im.exec(markdown);
  return match?.[1]?.replace(/\s+/g, " ").trim() ?? "";
}

function isErrorLike(value: unknown): value is Error & Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return Object.prototype.toString.call(value) === "[object Error]" &&
    typeof candidate.name === "string" &&
    typeof candidate.message === "string" &&
    typeof candidate.stack === "string";
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index])
  );
}
