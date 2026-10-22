import {
  isLlmContentBlockedError,
  isLlmOutputTruncatedError,
  isLlmTransientExhaustedError,
  type LlmClient,
} from "../llm/client";
import type { MetricsObserver } from "../metrics/generation";
import dailyPaperSummaryTemplateEn from "../prompts/daily-paper-summary.en.system.md";
import dailyPaperSummaryTemplate from "../prompts/daily-paper-summary.system.md";
import injectionGuardEn from "../prompts/injection-guard.en.md";
import injectionGuardZh from "../prompts/injection-guard.md";
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
import { isContentFetchFailure } from "./content-fetch-failure";
import { escapePaperDataFence } from "./prompt-safety";
import {
  canonicalizeScientificMarkdownMath,
  SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES,
  type ScientificMarkdownMathIssue,
} from "./scientific-markdown-math";

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

export interface DailyPaperSummaryRetryDeps
  extends Omit<DailyPaperSummaryDeps, "previousValidationError"> {
  /** Called for every validation failure, including the one that exhausts retries. */
  onValidationFailure?: (
    error: DailyPaperSummaryValidationError,
    attempt: number,
  ) => void;
}

export const DAILY_PAPER_SUMMARY_VALIDATION_ERROR_CODE =
  "ARXIV_DAILY_PAPER_SUMMARY_VALIDATION" as const;

export interface DailyPaperSummaryMathValidationIssue {
  code: ScientificMarkdownMathIssue["code"];
  offset: number;
  length: number;
}

export type DailyPaperSummaryValidationReasonCode =
  | "invalid-json"
  | "invalid-root"
  | "invalid-keys"
  | "invalid-id-type"
  | "mismatched-id"
  | "invalid-field-type"
  | "empty-field"
  | "invalid-math";

export class DailyPaperSummaryValidationError extends Error {
  readonly name = "DailyPaperSummaryValidationError";
  readonly code = DAILY_PAPER_SUMMARY_VALIDATION_ERROR_CODE;
  readonly paperId: string;
  readonly reasonCode?: DailyPaperSummaryValidationReasonCode;
  readonly fieldKey?: keyof Omit<StructuredPaperSummary, "id">;
  readonly issues?: readonly DailyPaperSummaryMathValidationIssue[];
  /**
   * Raw `JSON.parse` message for an invalid-json failure. It can quote model
   * text, so it is for local logs and error messages only, never for prompts.
   */
  readonly detail?: string;
  /** Trusted category (and raw-text offset) derived from `detail`; prompt-safe. */
  readonly jsonCause?: DailyPaperSummaryJsonCause;

  constructor(
    paperId: string,
    message: string,
    details: {
      reasonCode?: DailyPaperSummaryValidationReasonCode;
      fieldKey?: keyof Omit<StructuredPaperSummary, "id">;
      issues?: readonly DailyPaperSummaryMathValidationIssue[];
      detail?: string;
      jsonCause?: DailyPaperSummaryJsonCause;
    } = {},
  ) {
    super(message);
    this.paperId = paperId;
    this.reasonCode = details.reasonCode;
    this.fieldKey = details.fieldKey;
    this.issues = details.issues;
    this.detail = details.detail;
    this.jsonCause = details.jsonCause;
  }
}

export type DailyPaperSummaryJsonCauseCategory =
  | "bad-escape"
  | "truncated"
  | "control-character"
  | "unexpected-token"
  | "other";

export interface DailyPaperSummaryJsonCause {
  category: DailyPaperSummaryJsonCauseCategory;
  /** Character offset (0-based) into the raw response, when V8 reported one. */
  position?: number;
}

/**
 * Map a V8 `JSON.parse` message to a closed set of categories. The message may
 * quote model text, so only the category and the numeric position leave here.
 */
export function classifyJsonParseFailure(message: string): DailyPaperSummaryJsonCause {
  let category: DailyPaperSummaryJsonCauseCategory = "other";
  if (/\bbad (escaped character|unicode escape)/i.test(message)) {
    category = "bad-escape";
  } else if (/unexpected end of json|unterminated string/i.test(message)) {
    category = "truncated";
  } else if (/bad control character/i.test(message)) {
    category = "control-character";
  } else if (
    /unexpected (token|non-whitespace)|^expected\b|is not valid json/i.test(message)
  ) {
    category = "unexpected-token";
  }
  const match = /\bat position (\d+)/.exec(message);
  return match ? { category, position: Number(match[1]) } : { category };
}

const LOG_DETAIL_MAX_CHARS = 240;

/** One-line, bounded description of a validation failure for local warn logs. */
export function describeValidationFailureForLog(
  error: DailyPaperSummaryValidationError,
): string {
  const code = error.reasonCode ?? "unknown";
  if (code === "invalid-json" && error.detail) {
    const flat = error.detail.replace(/\s+/g, " ").trim();
    const bounded = flat.length > LOG_DETAIL_MAX_CHARS
      ? `${flat.slice(0, LOG_DETAIL_MAX_CHARS)}…`
      : flat;
    return `${code} ${JSON.stringify(bounded)}`;
  }
  return error.fieldKey ? `${code} field=${error.fieldKey}` : code;
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
const TRUSTED_MATH_ISSUE_CODES = new Set<string>(
  Object.values(SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES),
);
export async function summarizeDailyPaper(
  paper: DailyPaperSummaryInput,
  deps: DailyPaperSummaryDeps,
): Promise<StructuredPaperSummary> {
  throwIfCancelled(deps.signal);
  const source = resolveSummarySource(paper);
  if (!source) throw new Error(`summarizeDailyPaper: no source text for ${paper.id}`);
  const language = normalizeSummaryLanguage(deps.summaryLanguage);
  const systemPrompt = renderPrompt(
    language === "en" ? dailyPaperSummaryTemplateEn : dailyPaperSummaryTemplate,
    { injectionGuard: language === "en" ? injectionGuardEn : injectionGuardZh },
  );
  const userContent = [
    buildPaperData(paper, source, language),
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
  deps: DailyPaperSummaryRetryDeps,
): Promise<DailyPaperResult> {
  const { onValidationFailure, ...summaryDeps } = deps;
  let previousValidationError: DailyPaperSummaryValidationError | undefined;

  throwIfCancelled(deps.signal);
  if (!resolveSummarySource(paper)) {
    // Neither fetched text nor a listed abstract: a model asked anyway can only invent one.
    return {
      kind: "fallback",
      reasonCode: "content-unavailable",
      attempts: 1,
      originalAbstract: resolveTrustedOriginalAbstract(paper),
    };
  }

  for (let attempt = 1; attempt <= DAILY_PAPER_SUMMARY_MAX_ATTEMPTS; attempt += 1) {
    throwIfCancelled(deps.signal);
    try {
      const summary = await summarizeDailyPaper(paper, {
        ...summaryDeps,
        previousValidationError,
      });
      return { kind: "structured", summary };
    } catch (error) {
      if (isCancellationError(error)) throw error;
      if (isDailyPaperSummaryValidationError(error) && error.paperId === paper.id) {
        previousValidationError = error;
        onValidationFailure?.(error, attempt);
        if (attempt < DAILY_PAPER_SUMMARY_MAX_ATTEMPTS) continue;
        return {
          kind: "fallback",
          reasonCode: "validation-exhausted",
          attempts: attempt,
          originalAbstract: resolveTrustedOriginalAbstract(paper),
        };
      }
      if (isLlmOutputTruncatedError(error)) {
        // The same request would be cut off again, so another attempt cannot help.
        return {
          kind: "fallback",
          reasonCode: "validation-exhausted",
          attempts: attempt,
          originalAbstract: resolveTrustedOriginalAbstract(paper),
        };
      }
      if (isLlmContentBlockedError(error)) {
        // The provider refuses this paper's text every time; the day's other
        // papers are unaffected, so set this one aside instead of failing them.
        return {
          kind: "fallback",
          reasonCode: "content-blocked",
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
  paper: Pick<DailyPaperSummaryInput, "abstract" | "abstractConclusion" | "fullSections">,
): string {
  const source = resolveSummarySource(paper);
  if (!source) return "获取失败";
  const titles = [
    ...extractSectionTitles(source.abstractConclusion),
    ...extractSectionTitles(source.fullSections),
  ].filter((title, index, values) => values.indexOf(title) === index);
  return titles.length > 0 ? titles.join(", ") : "正文摘录";
}

export function resolveTrustedOriginalAbstract(
  paper: Pick<DailyPaperSummaryInput, "abstract" | "abstractConclusion">,
): string {
  if (typeof paper.abstract === "string" && paper.abstract.trim()) {
    return paper.abstract.trim();
  }
  return extractAbstractSection(usableAbstractConclusion(paper.abstractConclusion));
}

interface SummarySource {
  abstractConclusion: string;
  fullSections: string | null;
}

/**
 * What a summary may be written from, or null when there is nothing. A fetch
 * failure placeholder and an abs page without an abstract ("N/A") are not
 * paper text; the abstract the listing already provided stands in for them.
 */
function resolveSummarySource(
  paper: Pick<DailyPaperSummaryInput, "abstract" | "abstractConclusion" | "fullSections">,
): SummarySource | null {
  const fullSections = paper.fullSections?.trim() || null;
  let abstractConclusion = usableAbstractConclusion(paper.abstractConclusion);
  const listed = typeof paper.abstract === "string" ? paper.abstract.trim() : "";
  if (!abstractConclusion && listed) abstractConclusion = `## Abstract\n${listed}`;
  return abstractConclusion || fullSections ? { abstractConclusion, fullSections } : null;
}

function usableAbstractConclusion(abstractConclusion: string): string {
  const text = abstractConclusion.trim();
  if (isContentFetchFailure(text)) return "";
  // The abs-page fallback writes this when the page has no abstract block.
  if (text.length <= 32 && /^##\s*abstract\s+n\/a$/i.test(text)) return "";
  return text;
}

function buildPaperData(
  paper: DailyPaperSummaryInput,
  source: SummarySource,
  language: SummaryLanguage,
): string {
  const content = [source.abstractConclusion];
  if (source.fullSections) {
    content.push(
      language === "en"
        ? `Full-text excerpts:\n${source.fullSections}`
        : `正文摘录：\n${source.fullSections}`,
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

// String.raw keeps the example literal: the raw JSON shows two backslashes and
// the parsed text shows one.
const JSON_ESCAPE_INSTRUCTION_ZH =
  String.raw`JSON 字符串里的每个反斜杠都要写成两个（例如 JSON 原文写 "$28\\pm7$"，解析后是 $28\pm7$）；保留 LaTeX 公式，不要把公式改写成纯文本。`;
const JSON_ESCAPE_INSTRUCTION_EN =
  String.raw`Every backslash inside a JSON string must be written twice (for example, write "$28\\pm7$" in the raw JSON, which parses to $28\pm7$); keep LaTeX formulas as LaTeX and do not rewrite them as plain text.`;

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
      (error.reasonCode === "invalid-json" ? `${JSON_ESCAPE_INSTRUCTION_EN}\n` : "") +
      `Math rules for semantic fields:\n` +
      `- In semantic fields, use Obsidian inline math $...$ only.\n` +
      `- Do not use \\(...\\), \\[...\\], or $$...$$.\n` +
      `- Keep every TeX command inside math delimiters.\n` +
      `- Never split a single formula into multiple adjacent $...$ spans; genuinely separate formulas may use separate spans.\n` +
      `- For ensemble-average or expectation angle brackets use \\langle … \\rangle (or \\left< … \\right>); never write bare angle brackets shaped like <x> in math, as they are treated as HTML tags and will not render. Ordinary comparison operators < and > (e.g. $a<b$, $z<0.5$) remain fine.\n` +
      `Good: $\\langle \\rho \\rangle$, $z<0.5$, $M_{*}=10^{10}\\,M_{\\odot}$, $a<b$.\n` +
      `Bad: $<\\rho>$, bare \\alpha, $M_{*}$ $=10^{10}$ (split formula), \\(\\alpha\\), $$\\alpha$$.\n` +
      `Do not wrap the JSON in Markdown fences or add any other text.`
    );
  }
  return (
    `上一次响应未通过校验：${reason}\n` +
    `只返回严格 JSON，且恰好包含这些键：${keys}。\n` +
    `id 字段必须等于 paper_data 中的 ID。\n` +
    `每个字符串字段在 trim 后都必须非空。\n` +
    (error.reasonCode === "invalid-json" ? `${JSON_ESCAPE_INSTRUCTION_ZH}\n` : "") +
    `语义字段数学规则：\n` +
    `- 语义字段中的数学公式只能使用 Obsidian 行内格式 $...$。\n` +
    `- 禁止使用 \\(...\\)、\\[...\\] 或 $$...$$。\n` +
    `- 所有 TeX 命令都必须位于数学定界符内。\n` +
    `- 绝不能把一个公式拆成多个相邻的 $...$ 片段；真正彼此独立的公式可以分别使用独立片段。\n` +
    `- 表示系综平均或期望的尖括号必须使用 \\langle … \\rangle（或 \\left< … \\right>）；禁止在数学中直接写形如 <x> 的裸尖括号，它会被当作 HTML 标签而无法正确渲染。普通不等号 <、>（如 $a<b$、$z<0.5$）照常使用。\n` +
    `正例：$\\langle \\rho \\rangle$、$z<0.5$、$M_{*}=10^{10}\\,M_{\\odot}$、$a<b$。\n` +
    `反例：$<\\rho>$、裸 \\alpha、$M_{*}$ $=10^{10}$（拆成相邻片段）、\\(\\alpha\\)、$$\\alpha$$。\n` +
    `不要用 Markdown 代码块包裹 JSON，也不要附加其他文本。`
  );
}

function trustedJsonCauseSuffix(
  cause: DailyPaperSummaryJsonCause | undefined,
  language: SummaryLanguage,
): string {
  if (!cause) return "";
  const position = Number.isSafeInteger(cause.position) && cause.position! >= 0
    ? cause.position
    : undefined;
  if (language === "en") {
    const at = position === undefined ? "" : ` at character offset ${position}`;
    switch (cause.category) {
      case "bad-escape": return ` (illegal backslash escape${at})`;
      case "truncated": return ` (the response ended unexpectedly${at}; likely truncated)`;
      case "control-character": return ` (unescaped control character inside a string${at})`;
      case "unexpected-token": return ` (unexpected character${at})`;
      default: return position === undefined ? "" : ` (parse failure${at})`;
    }
  }
  const at = position === undefined ? "" : `字符偏移量 ${position} 处`;
  switch (cause.category) {
    case "bad-escape": return `（${at}有非法反斜杠转义）`;
    case "truncated": return `（响应在${at || "中途"}意外结束，疑似被截断）`;
    case "control-character": return `（字符串内${at}有未转义的控制字符）`;
    case "unexpected-token": return `（${at}有意外字符）`;
    default: return position === undefined ? "" : `（${at}解析失败）`;
  }
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
  const issueCodes = error.issues
    ?.slice(0, 8)
    .map(({ code }) => code)
    .filter((code) => TRUSTED_MATH_ISSUE_CODES.has(code))
    .filter((code, index, values) => values.indexOf(code) === index)
    .join(", ");
  if (language === "en") {
    switch (error.reasonCode) {
      case "invalid-json": return `the response was not strict JSON${trustedJsonCauseSuffix(error.jsonCause, "en")}`;
      case "invalid-root": return "the JSON root was not an object";
      case "invalid-keys": return "the JSON keys did not match the required schema";
      case "invalid-id-type": return "the id field was not a string";
      case "mismatched-id": return "the id field did not match the requested paper";
      case "invalid-field-type": return field ? `${field} was not a string` : "a semantic field was not a string";
      case "empty-field": return field ? `${field} was empty` : "a semantic field was empty";
      case "invalid-math": return field && issueCodes
        ? `${field} contained invalid scientific math (${issueCodes})`
        : "a semantic field contained invalid scientific math";
      default: return "the response was invalid";
    }
  }
  switch (error.reasonCode) {
    case "invalid-json": return `响应不是严格 JSON${trustedJsonCauseSuffix(error.jsonCause, "zh")}`;
    case "invalid-root": return "JSON 根值不是对象";
    case "invalid-keys": return "JSON 键与要求的结构不一致";
    case "invalid-id-type": return "id 字段不是字符串";
    case "mismatched-id": return "id 字段与请求的论文不匹配";
    case "invalid-field-type": return field ? `${field} 不是字符串` : "某个语义字段不是字符串";
    case "empty-field": return field ? `${field} 为空` : "某个语义字段为空";
    case "invalid-math": return field && issueCodes
      ? `${field} 包含无效科学公式（${issueCodes}）`
      : "某个语义字段包含无效科学公式";
    default: return "响应格式或内容无效";
  }
}

const JSON_SINGLE_CHAR_ESCAPES = new Set(['"', "\\", "/"]);
const JSON_CONTROL_ESCAPES = new Set(["b", "f", "n", "r", "t"]);
const ASCII_LETTER = /[A-Za-z]/;
const FOUR_HEX_DIGITS = /^[0-9A-Fa-f]{4}$/;

/**
 * Double backslashes that models leave single when writing LaTeX into JSON
 * strings (`$28\pm7$`), so `JSON.parse` neither throws nor turns `\times` into
 * a tab. Only string literals are touched; valid escapes stay as they are.
 */
export function repairSummaryJsonBackslashes(text: string): string {
  return repairWithInsertions(text).text;
}

/** `inserted` lists the output indexes of the backslashes the repair added. */
function repairWithInsertions(text: string): { text: string; inserted: number[] } {
  const inserted: number[] = [];
  let out = "";
  let inString = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    if (!inString) {
      if (ch === '"') inString = true;
      out += ch;
      continue;
    }
    if (ch === '"') {
      inString = false;
      out += ch;
      continue;
    }
    if (ch !== "\\") {
      out += ch;
      continue;
    }
    const next = text[i + 1];
    let isLatex: boolean;
    if (next === undefined) {
      isLatex = true;
    } else if (JSON_SINGLE_CHAR_ESCAPES.has(next)) {
      isLatex = false;
    } else if (JSON_CONTROL_ESCAPES.has(next)) {
      // Semantic fields never carry control characters (newlines are rejected
      // as multiline math and rendering compacts whitespace), so \nu, \times,
      // \rm, \beta, \frac are LaTeX; \n before a non-letter stays a newline.
      isLatex = ASCII_LETTER.test(text[i + 2] ?? "");
    } else if (next === "u") {
      isLatex = !FOUR_HEX_DIGITS.test(text.slice(i + 2, i + 6));
    } else {
      isLatex = true;
    }
    if (isLatex) {
      inserted.push(out.length);
      out += "\\\\";
    } else {
      out += ch + next;
      i++;
    }
  }
  return { text: out, inserted };
}

function boundedDetail(detail: string): string {
  const flat = detail.replace(/\s+/g, " ").trim();
  return flat.length > 300 ? `${flat.slice(0, 300)}…` : flat;
}

function parseDailyPaperSummary(
  raw: string,
  expectedId: string,
): StructuredPaperSummary {
  let value: unknown;
  const repaired = repairWithInsertions(raw);
  try {
    value = JSON.parse(repaired.text);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    const parsed = classifyJsonParseFailure(detail);
    // V8 reports offsets into the repaired text; map back to the raw response.
    const jsonCause: DailyPaperSummaryJsonCause = parsed.position === undefined
      ? parsed
      : {
          ...parsed,
          position: parsed.position -
            repaired.inserted.filter((index) => index < parsed.position!).length,
        };
    throw new DailyPaperSummaryValidationError(
      expectedId,
      `summarizeDailyPaper: response for ${expectedId} is not strict JSON: ${boundedDetail(detail)}`,
      { reasonCode: "invalid-json", detail: boundedDetail(detail), jsonCause },
    );
  }
  return validateStructuredPaperSummaryValue(value, expectedId);
}

/** Validate and canonicalize a parsed structured summary using generation semantics. */
export function validateStructuredPaperSummaryValue(
  value: unknown,
  expectedId: string,
): StructuredPaperSummary {
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
    const math = canonicalizeScientificMarkdownMath(trimmed);
    if (!math.ok) {
      const issues = math.issues.map(({ code, offset, length }) => ({
        code,
        offset,
        length,
      }));
      const locations = issues
        .map(({ code, offset, length }) => `${code}@${offset}:${length}`)
        .join(", ");
      throw new DailyPaperSummaryValidationError(
        expectedId,
        `summarizeDailyPaper: ${key} for ${expectedId} has invalid scientific math (${locations})`,
        { reasonCode: "invalid-math", fieldKey: key, issues },
      );
    }
    summary[key] = math.value;
  }
  return summary;
}

function extractSectionTitles(markdown: string | null | undefined): string[] {
  if (!markdown) return [];
  const titles: string[] = [];
  // Locate heading markers separately from whitespace and body scanning (same
  // technique as extractAbstractSection below), so long runs of whitespace
  // never participate in overlapping \s+/(.+) searches (CodeQL
  // js/polynomial-redos).
  const headings = /^##/gm;
  let heading: RegExpExecArray | null;
  while ((heading = headings.exec(markdown)) !== null) {
    let cursor = heading.index + 2;
    if (!/\s/u.test(markdown[cursor] ?? "")) continue;
    while (cursor < markdown.length && /\s/u.test(markdown[cursor]!)) cursor += 1;
    let end = cursor;
    while (end < markdown.length && !/[\r\n\u2028\u2029]/u.test(markdown[end]!)) end += 1;
    const title = markdown.slice(cursor, end).trim();
    if (title && !titles.includes(title)) titles.push(title);
  }
  return titles;
}

function extractAbstractSection(markdown: string | null | undefined): string {
  if (!markdown) return "";
  // Locate heading markers separately from whitespace and body scanning, so
  // long runs of whitespace never participate in overlapping regex searches.
  const headings = /^##/gm;
  let heading: RegExpExecArray | null;
  while ((heading = headings.exec(markdown)) !== null) {
    let cursor = heading.index + 2;
    if (!/\s/u.test(markdown[cursor] ?? "")) continue;
    while (cursor < markdown.length && /\s/u.test(markdown[cursor]!)) cursor += 1;
    if (markdown.slice(cursor, cursor + 8).toLowerCase() === "abstract") cursor += 8;
    else if (markdown.slice(cursor, cursor + 2) === "摘要") cursor += 2;
    else continue;

    let bodyStart = -1;
    while (cursor < markdown.length && /\s/u.test(markdown[cursor]!)) {
      if (markdown[cursor] === "\n") bodyStart = cursor + 1;
      cursor += 1;
    }
    if (bodyStart < 0) continue;
    if (markdown.startsWith("##", bodyStart) && /\s/u.test(markdown[bodyStart + 2] ?? "")) return "";
    // The existing multiline-end-anchor contract takes the first content line.
    let bodyEnd = bodyStart;
    while (bodyEnd < markdown.length && !/[\r\n\u2028\u2029]/u.test(markdown[bodyEnd]!)) bodyEnd += 1;
    return markdown.slice(bodyStart, bodyEnd).replace(/\s+/g, " ").trim();
  }
  return "";
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
  const prototype: unknown = Object.getPrototypeOf(value);
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
