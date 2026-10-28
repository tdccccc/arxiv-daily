import { renderPrompt, repairSummaryJsonBackslashes } from "@arxiv-daily/core";
import dailyGenerationPrompt from "../../packages/core/src/prompts/daily-paper-summary.system.md";
import noteGenerationPrompt from "../../packages/core/src/prompts/paper-detail.system.md";
import injectionGuard from "../../packages/core/src/prompts/injection-guard.md";
import judgeDailySubject from "./prompts/summary-judge.daily.md";
import judgeNoteSubject from "./prompts/summary-judge.note.md";
import judgeTemplate from "./prompts/summary-judge.system.md";
import { SUMMARY_FIELDS } from "./summary-checks";

export const JUDGE_DIMENSIONS = ["accuracy", "completeness", "clarity"] as const;
export type JudgeDimension = (typeof JUDGE_DIMENSIONS)[number];
export const JUDGE_VERDICTS = ["ok", "minor", "major"] as const;
export type JudgeVerdict = (typeof JUDGE_VERDICTS)[number];
export type JudgeSeverity = Exclude<JudgeVerdict, "ok">;

/** The sentence an accuracy issue must contain when its claim has no counterpart in the source. */
export const ABSENT_FROM_SOURCE_PHRASE = "原文未提及";

export interface JudgeIssue {
  dimension: JudgeDimension;
  severity: JudgeSeverity;
  location: string;
  summaryQuote: string;
  sourceQuote: string | null;
  explanation: string;
}

export interface Judgment {
  verdicts: Record<JudgeDimension, JudgeVerdict>;
  issues: JudgeIssue[];
}

/** What a judgment is checked against. */
export interface JudgeContext {
  kind: "daily" | "note";
  /** Allowed locations, each with the summary text found there (the rendered field, or the section body). */
  locations: Readonly<Record<string, string>>;
  /** The text of `<paper_data>`: everything the generator could see. */
  sourceText: string;
}

export type JudgeCheck = { ok: true; judgment: Judgment } | { ok: false; reasons: string[] };

const RANK: Record<JudgeVerdict, number> = { ok: 0, minor: 1, major: 2 };
const ROOT_KEYS = ["verdicts", "issues"] as const;
const ISSUE_KEYS = ["dimension", "severity", "location", "summaryQuote", "sourceQuote", "explanation"] as const;
const MAX_REASONS = 12;

export type JudgePromptOptions =
  | { kind: "daily" }
  | { kind: "note"; headings: readonly string[]; /** The topic name the generator was told. */ topic: string };

const quoted = (values: readonly string[]): string => values.map((value) => `"${value}"`).join("、");

/** The judge's system prompt: the contract, the generation prompt as reference, and what is being judged. */
export function buildJudgeSystemPrompt(options: JudgePromptOptions): string {
  if (options.kind === "note" && options.headings.length === 0) throw new Error("buildJudgeSystemPrompt: a note needs its headings");
  // The generator's own guard line is not an instruction for the judge; it is shown without it.
  const generationPrompt = options.kind === "daily"
    ? renderPrompt(dailyGenerationPrompt, { injectionGuard: "" })
    : renderPrompt(noteGenerationPrompt, { injectionGuard: "", topicName: options.topic });
  return renderPrompt(judgeTemplate, {
    injectionGuard: injectionGuard.trim(),
    subject: (options.kind === "daily" ? judgeDailySubject : judgeNoteSubject).trim(),
    generationPrompt: generationPrompt.trim(),
    locations: quoted(options.kind === "daily" ? SUMMARY_FIELDS : options.headings),
  }).trim();
}

/** Whitespace differences never matter for a quote; every other character must match. */
export const normalizeQuoteText = (text: string): string => text.replace(/\s+/gu, " ").trim();

// Some models wrap the whole answer in one ```json fence; exactly that shape is unwrapped, as the filter does.
function unwrapSingleCodeFence(raw: string): string {
  const inner = /^\s*```[A-Za-z]*[ \t]*\r?\n([\s\S]*?)\r?\n```\s*$/.exec(raw)?.[1];
  return inner === undefined || inner.includes("```") ? raw : inner;
}

export function parseJudgeOutput(raw: string): { ok: true; value: unknown } | { ok: false; reason: string } {
  try {
    // Lone backslashes in LaTeX quoted inside the JSON are doubled first, as the summarizer does.
    return { ok: true, value: JSON.parse(repairSummaryJsonBackslashes(unwrapSingleCodeFence(raw))) };
  } catch (error) {
    return { ok: false, reason: `输出不是合法的 JSON（${(error as Error).message}）。` };
  }
}

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function keyProblems(value: Record<string, unknown>, expected: readonly string[], path: string): string[] {
  const problems: string[] = [];
  for (const key of expected) if (!(key in value)) problems.push(`${path} 缺少键 "${key}"。`);
  for (const key of Object.keys(value)) if (!expected.includes(key)) problems.push(`${path} 有多余的键 "${key}"，只允许 ${quoted(expected)}。`);
  return problems;
}

/** Check a parsed judgment against the contract: keys, enums, locations, quotes, and verdict consistency. */
export function validateJudgment(value: unknown, context: JudgeContext): JudgeCheck {
  const reasons: string[] = [];
  if (!isPlainObject(value)) return { ok: false, reasons: ["输出必须是一个 JSON 对象。"] };
  reasons.push(...keyProblems(value, ROOT_KEYS, "根对象"));

  const verdicts = value.verdicts;
  let verdictsValid = false;
  if (!isPlainObject(verdicts)) {
    if ("verdicts" in value) reasons.push("verdicts 必须是对象。");
  } else {
    const problems = keyProblems(verdicts, JUDGE_DIMENSIONS, "verdicts");
    for (const dimension of JUDGE_DIMENSIONS) {
      const verdict = verdicts[dimension];
      if (dimension in verdicts && !JUDGE_VERDICTS.includes(verdict as JudgeVerdict)) {
        problems.push(`verdicts.${dimension} 的值 ${JSON.stringify(verdict)} 不合法，只允许 ${quoted(JUDGE_VERDICTS)}。`);
      }
    }
    reasons.push(...problems);
    verdictsValid = problems.length === 0;
  }

  const issues = value.issues;
  const worst: Record<JudgeDimension, JudgeVerdict> = { accuracy: "ok", completeness: "ok", clarity: "ok" };
  let issuesValid = false;
  if (!Array.isArray(issues)) {
    if ("issues" in value) reasons.push("issues 必须是数组（没有问题时是空数组）。");
  } else {
    issuesValid = true;
    const sourceText = normalizeQuoteText(context.sourceText);
    const allowed = Object.keys(context.locations);
    issues.forEach((issue: unknown, index) => {
      const path = `issues[${index}]`;
      if (!isPlainObject(issue)) {
        issuesValid = false;
        reasons.push(`${path} 必须是对象。`);
        return;
      }
      const before = reasons.length;
      reasons.push(...keyProblems(issue, ISSUE_KEYS, path));
      const dimension = issue.dimension as JudgeDimension;
      const severity = issue.severity as JudgeSeverity;
      const dimensionOk = JUDGE_DIMENSIONS.includes(dimension);
      const severityOk = severity === "minor" || severity === "major";
      if ("dimension" in issue && !dimensionOk) reasons.push(`${path}.dimension 的值 ${JSON.stringify(issue.dimension)} 不合法，只允许 ${quoted(JUDGE_DIMENSIONS)}。`);
      if ("severity" in issue && !severityOk) reasons.push(`${path}.severity 的值 ${JSON.stringify(issue.severity)} 不合法，只允许 "minor"、"major"。`);

      const location = issue.location;
      const summaryText = typeof location === "string" ? context.locations[location] : undefined;
      if ("location" in issue && summaryText === undefined) {
        reasons.push(`${path}.location 的值 ${JSON.stringify(location)} 不是允许的位置，只能是 ${quoted(allowed)}。`);
      }

      const summaryQuote = issue.summaryQuote;
      if ("summaryQuote" in issue) {
        if (typeof summaryQuote !== "string" || !summaryQuote.trim()) {
          reasons.push(`${path}.summaryQuote 必须是非空字符串。`);
        } else if (summaryText !== undefined && !normalizeQuoteText(summaryText).includes(normalizeQuoteText(summaryQuote))) {
          reasons.push(`${path}.summaryQuote 不是“${String(location)}”里的原句（必须逐字摘自该位置的摘要）：${JSON.stringify(summaryQuote)}`);
        }
      }

      const sourceQuote = issue.sourceQuote;
      if ("sourceQuote" in issue) {
        if (sourceQuote === null) {
          if (dimension === "completeness") reasons.push(`${path}（completeness）必须给出 sourceQuote：被漏掉的内容在原文里的原话。`);
          if (dimension === "accuracy" && !(typeof issue.explanation === "string" && issue.explanation.includes(ABSENT_FROM_SOURCE_PHRASE))) {
            reasons.push(`${path}（accuracy）的 sourceQuote 为 null 时，explanation 必须写明“${ABSENT_FROM_SOURCE_PHRASE}”以及摘要的哪个说法没有依据。`);
          }
        } else if (typeof sourceQuote !== "string" || !sourceQuote.trim()) {
          reasons.push(`${path}.sourceQuote 必须是非空字符串或 null。`);
        } else if (!sourceText.includes(normalizeQuoteText(sourceQuote))) {
          reasons.push(`${path}.sourceQuote 不是 <paper_data> 里的原文（必须逐字摘自原文）：${JSON.stringify(sourceQuote)}`);
        }
      }

      if ("explanation" in issue && (typeof issue.explanation !== "string" || !issue.explanation.trim())) {
        reasons.push(`${path}.explanation 必须是非空字符串。`);
      }

      if (reasons.length > before) issuesValid = false;
      if (dimensionOk && severityOk && RANK[severity] > RANK[worst[dimension]]) worst[dimension] = severity;
    });
  }

  if (verdictsValid && issuesValid) {
    for (const dimension of JUDGE_DIMENSIONS) {
      const declared = (verdicts as Record<string, JudgeVerdict>)[dimension]!;
      if (declared !== worst[dimension]) {
        reasons.push(worst[dimension] === "ok"
          ? `verdicts.${dimension} 是 "${declared}"，但 issues 里没有 ${dimension} 问题，应为 "ok"。`
          : `verdicts.${dimension} 是 "${declared}"，但该维度最严重的问题是 "${worst[dimension]}"，判定必须等于最严重的问题。`);
      }
    }
  }

  if (reasons.length > 0) return { ok: false, reasons: reasons.slice(0, MAX_REASONS) };
  return { ok: true, judgment: { verdicts: verdicts as Judgment["verdicts"], issues: issues as JudgeIssue[] } };
}

/** Parse the model's reply and validate it; the reasons are ready to hand back to the model. */
export function checkJudgeOutput(raw: string, context: JudgeContext): JudgeCheck {
  const parsed = parseJudgeOutput(raw);
  return parsed.ok ? validateJudgment(parsed.value, context) : { ok: false, reasons: [parsed.reason] };
}

/** The follow-up message for the one retry: why the last reply was rejected, and the rules again. */
export function buildJudgeCorrection(reasons: readonly string[]): string {
  return [
    "你上一次的输出没有通过程序校验，原因如下：",
    ...reasons.map((reason) => `- ${reason}`),
    "",
    "请重新评审并只输出完整的 JSON 对象，不要输出任何其他文字。修正以上问题：引用必须逐字摘自对应的摘要位置或 <paper_data>，位置只能用允许的值，三项判定必须等于各自问题里最严重的那个（没有问题就是 \"ok\"）。",
  ].join("\n");
}
