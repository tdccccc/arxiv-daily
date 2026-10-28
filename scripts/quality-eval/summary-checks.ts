import { canonicalizeScientificMarkdownMath } from "@arxiv-daily/core";
import type { FallbackEntry, SummaryEntry } from "./summary-corpus";

export const SUMMARY_FIELDS = ["coreProblem", "keyMethod", "mainResult", "whyRelevant", "limitations"] as const;
export type SummaryFieldKey = (typeof SUMMARY_FIELDS)[number];

/** Phrases both prompts forbid as filler; the detail-note checks reuse this list. */
export const BANNED_GENERIC_PHRASES: readonly string[] = [
  "具有重要意义", "提高了理解",
  "important significance", "improves understanding",
];
/** The exact text the prompts require for information the source lacks. */
export const NOT_STATED_PHRASE = "原文未说明";
export const CLAIM_PHRASE = "作者声称";

/** The prompt's usual range for `mainResult` in Chinese characters; a guideline, so a miss is only a soft finding. */
export const MAIN_RESULT_LENGTH_RANGE = { min: 80, max: 180 } as const;
const MIN_CJK_SHARE_FOR_LENGTH = 0.3;

export type SummaryCheckKind = "math" | "length" | "not-stated" | "generic-phrase";

export interface SummaryFinding {
  kind: SummaryCheckKind;
  /** Core issue code or one of ours (`split-fragment`, `paren-delimiter`); for the other kinds, the specific reading. */
  code: string;
  field: SummaryFieldKey;
  snippet: string;
  /** A guideline miss rather than a rule break. */
  soft?: boolean;
  /** Core's math rule rejects text that renders fine (see {@link isCoreFalsePositive}); kept in the record, left out of defect counts. */
  knownFalsePositive?: true;
}

export interface EntryChecks {
  date: string;
  topic: string | null;
  id: string;
  title: string;
  findings: SummaryFinding[];
  /** Per field: characters left after removing `$...$` spans and whitespace. */
  lengths: Record<SummaryFieldKey, number>;
  /** How many times "作者声称" appears across the five fields; information, not a finding. */
  claimCount: number;
}

// A `$...$` span: unescaped single dollars around a non-empty body that has no dollar of its own.
export const MATH_SPAN = /(?<!\\)\$(?!\$)(?:\\.|[^$\\])+?(?<!\\)\$/gu;
const LATEX_PAREN_DELIMITER = /\\[()[\]]/u;
// What may sit between two `$..$` spans when they are one formula split in two. A slash is left out: it lists values.
const SPLIT_GAP = /^(?:\s*|\s*[=+×±∓~∼≈≃<>≤≥·*]\s*|\s+[-−–]\s+)$/u;
const SNIPPET_BEFORE = 12;
const SNIPPET_AFTER = 28;

/**
 * Report fields are rendered text: normalizeMarkdownLine turns `[[` into `\[\[` and `![` into `!\[` outside math.
 * Core reads a bare `\[` as a display-math opener and reports it unclosed, so undo exactly that escaping,
 * and only outside `$...$` where the renderer leaves text alone.
 */
export function unescapeRenderedField(text: string): string {
  let out = "";
  let last = 0;
  for (const span of text.matchAll(MATH_SPAN)) {
    out += undoRendererEscapes(text.slice(last, span.index)) + span[0];
    last = span.index + span[0].length;
  }
  return out + undoRendererEscapes(text.slice(last));
}

function undoRendererEscapes(segment: string): string {
  return segment.replace(/\\\[\\\[/g, "[[").replace(/!\\\[/g, "![");
}

/** Length measure: code points of the field once `$...$` spans and all whitespace are removed. */
export function measureFieldLength(text: string): number {
  return [...text.replace(MATH_SPAN, "").replace(/\s+/gu, "")].length;
}

export function snippetAt(text: string, offset: number, length: number): string {
  return text.slice(Math.max(0, offset - SNIPPET_BEFORE), offset + length + SNIPPET_AFTER).replace(/\s+/gu, " ");
}

export interface MathIssue { code: string; snippet: string; coreFalsePositive?: true }

/**
 * Math problems in one line of text: core's issue codes plus the ones core lacks.
 * Rendered daily fields are unescaped first (see unescapeRenderedField); note lines are not rendered text.
 */
export function findMathIssues(rendered: string, options: { rendered: boolean }): MathIssue[] {
  const text = options.rendered ? unescapeRenderedField(rendered) : rendered;
  const issues: MathIssue[] = [];
  const add = (code: string, offset: number, length: number, coreFalsePositive = false) =>
    issues.push({ code, snippet: snippetAt(text, offset, length), ...(coreFalsePositive ? { coreFalsePositive: true as const } : {}) });

  const result = canonicalizeScientificMarkdownMath(text);
  for (const issue of result.issues) add(issue.code, issue.offset, issue.length, isCoreFalsePositive(text, issue.code, issue.offset));

  // Core quietly rewrites `\(..\)` and `\[..\]` to `$..$` and accepts, so the report must name them itself.
  let last = 0;
  const spans = [...text.matchAll(MATH_SPAN)];
  for (const span of [...spans, null]) {
    const end = span ? span.index : text.length;
    const delimiter = LATEX_PAREN_DELIMITER.exec(text.slice(last, end));
    if (delimiter) add("paren-delimiter", last + delimiter.index, delimiter[0].length);
    if (span) last = span.index + span[0].length;
  }

  for (let index = 1; index < spans.length; index += 1) {
    const before = spans[index - 1]!;
    const after = spans[index]!;
    const gapStart = before.index + before[0].length;
    if (SPLIT_GAP.test(text.slice(gapStart, after.index))) add("split-fragment", before.index, after.index + after[0].length - before.index);
  }
  return issues;
}

// `$6:1$`, `$8:1:1$`: core reads a `$` before a digit as currency, so a digit-leading ratio is not paired as math.
const DIGIT_RATIO_SPAN = /\$\d[\d.]*(?::\d[\d.]*)+\$/u;
const RATIO_FOLLOW_ON_CODES = new Set(["unclosed-delimiter", "nested-delimiter", "unmatched-delimiter"]);
// `(0, 0.15]`, `[0,1)`: a half-open interval is deliberately unbalanced.
const HALF_OPEN_INTERVAL = /\([^()[\]]*\]|\[[^()[\]]*\)/u;

/**
 * Math problems core reports on text that renders fine in Obsidian. Two known patterns: the unclosed and
 * nested follow-ons on a line holding a digit-leading colon ratio, and unbalanced parentheses or brackets inside a
 * span written as a half-open interval. Anything else stays a defect, even on the same line.
 */
function isCoreFalsePositive(text: string, code: string, offset: number): boolean {
  if (RATIO_FOLLOW_ON_CODES.has(code)) return DIGIT_RATIO_SPAN.test(text);
  if (code !== "unbalanced-parentheses" && code !== "unbalanced-optional-argument") return false;
  return [...text.matchAll(MATH_SPAN)].some((span) =>
    offset >= span.index && offset <= span.index + span[0].length && HALF_OPEN_INTERVAL.test(span[0]));
}

function mathFindings(field: SummaryFieldKey, rendered: string): SummaryFinding[] {
  return findMathIssues(rendered, { rendered: true }).map(({ code, snippet, coreFalsePositive }) => ({
    kind: "math" as const, code, field, snippet, ...(coreFalsePositive ? { knownFalsePositive: true as const } : {}),
  }));
}

function cjkShare(text: string): number {
  const letters = text.replace(/\s+/gu, "");
  if (!letters) return 0;
  return (letters.match(/\p{Script=Han}/gu)?.length ?? 0) / [...letters].length;
}

/** Whether the Chinese 80-180 range says anything about this mainResult (it does not for mostly Latin text). */
export function mainResultRangeApplies(mainResult: string): boolean {
  return Boolean(mainResult) && cjkShare(mainResult.replace(MATH_SPAN, "")) >= MIN_CJK_SHARE_FOR_LENGTH;
}

/** Run every per-entry check; pure, so the same entry always gives the same findings. */
export function checkEntry(entry: SummaryEntry): EntryChecks {
  const findings: SummaryFinding[] = [];
  const lengths = {} as Record<SummaryFieldKey, number>;
  let claimCount = 0;
  for (const field of SUMMARY_FIELDS) {
    const text = entry.fields[field];
    lengths[field] = measureFieldLength(text);
    findings.push(...mathFindings(field, text));

    const stated = text.replace(/[\s。.!！]+$/u, "").trim();
    if (stated === NOT_STATED_PHRASE) findings.push({ kind: "not-stated", code: "exact", field, snippet: text });
    else if (text.includes(NOT_STATED_PHRASE)) {
      findings.push({ kind: "not-stated", code: "partial", field, snippet: snippetAt(text, text.indexOf(NOT_STATED_PHRASE), NOT_STATED_PHRASE.length) });
    }
    claimCount += text.split(CLAIM_PHRASE).length - 1;

    const lowered = text.toLowerCase();
    for (const phrase of BANNED_GENERIC_PHRASES) {
      const at = lowered.indexOf(phrase.toLowerCase());
      if (at >= 0) findings.push({ kind: "generic-phrase", code: phrase, field, snippet: snippetAt(text, at, phrase.length) });
    }
  }

  // The 80-180 range is in Chinese characters, so it says nothing about a mostly Latin (English) summary.
  const result = entry.fields.mainResult;
  if (mainResultRangeApplies(result)) {
    const length = lengths.mainResult;
    if (length < MAIN_RESULT_LENGTH_RANGE.min) findings.push({ kind: "length", code: "main-result-short", field: "mainResult", snippet: `${length} chars`, soft: true });
    else if (length > MAIN_RESULT_LENGTH_RANGE.max) findings.push({ kind: "length", code: "main-result-long", field: "mainResult", snippet: `${length} chars`, soft: true });
  }
  return { date: entry.date, topic: entry.topic, id: entry.id, title: entry.title, findings, lengths, claimCount };
}

/** Distribution of a numeric measure; `LengthDistribution` is the same shape for field lengths. */
export interface LengthDistribution { min: number; median: number; p90: number; max: number }
export interface Rate { count: number; of: number; rate: number }

export interface ChecksAggregate {
  fallback: { overall: Rate; byMonth: Record<string, Rate> };
  entries: number;
  /** Per check kind: entries with at least one finding of that kind. */
  entriesWith: Record<SummaryCheckKind, Rate>;
  /** Finding counts per check kind and code, known core false positives left out. */
  codes: Record<SummaryCheckKind, Record<string, number>>;
  /** Math findings on text that renders fine; counted apart and not in `entriesWith` or `codes`. */
  coreFalsePositives: { entries: number; findings: number; codes: Record<string, number> };
  /** Over entries where the field is present; a missing field is not a zero-length one. */
  lengths: Record<SummaryFieldKey, LengthDistribution>;
  /** Entries whose mainResult is outside the soft range, among entries the range applies to. */
  mainResultOutsideRange: Rate;
  notStated: Record<SummaryFieldKey, { exact: number; partial: number }>;
  claims: { occurrences: number; entriesWith: Rate; byField: Record<SummaryFieldKey, number> };
}

export interface SummaryChecksResult { results: EntryChecks[]; aggregate: ChecksAggregate }

export const rate = (count: number, of: number): Rate => ({ count, of, rate: of === 0 ? 0 : count / of });
const KINDS: SummaryCheckKind[] = ["math", "length", "not-stated", "generic-phrase"];

export function distribution(values: number[]): LengthDistribution {
  if (values.length === 0) return { min: 0, median: 0, p90: 0, max: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  const median = sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
  // Nearest rank, so the p90 is always a value that occurs.
  return { min: sorted[0]!, median, p90: sorted[Math.ceil(sorted.length * 0.9) - 1]!, max: sorted.at(-1)! };
}

/** Check every entry and add up rates and distributions; the fallback rate is fallbacks over entries plus fallbacks. */
export function checkSummaries(corpus: { entries: readonly SummaryEntry[]; fallbacks: readonly FallbackEntry[] }): SummaryChecksResult {
  const results = corpus.entries.map(checkEntry);
  const total = results.length;
  const blank = <T,>(make: () => T) => Object.fromEntries(SUMMARY_FIELDS.map((field) => [field, make()])) as Record<SummaryFieldKey, T>;

  const monthTotals = new Map<string, { entries: number; fallbacks: number }>();
  const month = (date: string) => {
    const key = date.slice(0, 7);
    if (!monthTotals.has(key)) monthTotals.set(key, { entries: 0, fallbacks: 0 });
    return monthTotals.get(key)!;
  };
  for (const entry of corpus.entries) month(entry.date).entries += 1;
  for (const fallback of corpus.fallbacks) month(fallback.date).fallbacks += 1;

  const codes = Object.fromEntries(KINDS.map((kind) => [kind, {}])) as Record<SummaryCheckKind, Record<string, number>>;
  const withKind = Object.fromEntries(KINDS.map((kind) => [kind, 0])) as Record<SummaryCheckKind, number>;
  const coreFalsePositives: ChecksAggregate["coreFalsePositives"] = { entries: 0, findings: 0, codes: {} };
  const notStated = blank(() => ({ exact: 0, partial: 0 }));
  const claimsByField = blank(() => 0);
  let claimOccurrences = 0;
  let claimEntries = 0;
  let rangeApplies = 0;
  let outsideRange = 0;
  for (const [index, result] of results.entries()) {
    const kinds = new Set<SummaryCheckKind>();
    let outside = false;
    let falsePositives = 0;
    for (const finding of result.findings) {
      if (finding.knownFalsePositive) {
        falsePositives += 1;
        coreFalsePositives.codes[finding.code] = (coreFalsePositives.codes[finding.code] ?? 0) + 1;
        continue;
      }
      kinds.add(finding.kind);
      codes[finding.kind][finding.code] = (codes[finding.kind][finding.code] ?? 0) + 1;
      if (finding.kind === "not-stated") notStated[finding.field][finding.code === "exact" ? "exact" : "partial"] += 1;
      if (finding.kind === "length") outside = true;
    }
    for (const kind of kinds) withKind[kind] += 1;
    if (falsePositives > 0) {
      coreFalsePositives.entries += 1;
      coreFalsePositives.findings += falsePositives;
    }
    for (const field of SUMMARY_FIELDS) claimsByField[field] += corpus.entries[index]!.fields[field].split(CLAIM_PHRASE).length - 1;
    claimOccurrences += result.claimCount;
    if (result.claimCount > 0) claimEntries += 1;
    if (outside) outsideRange += 1;
    if (cjkShare(corpus.entries[index]!.fields.mainResult.replace(MATH_SPAN, "")) >= MIN_CJK_SHARE_FOR_LENGTH && corpus.entries[index]!.fields.mainResult) rangeApplies += 1;
  }

  return {
    results,
    aggregate: {
      fallback: {
        overall: rate(corpus.fallbacks.length, total + corpus.fallbacks.length),
        byMonth: Object.fromEntries([...monthTotals].sort(([a], [b]) => a.localeCompare(b))
          .map(([key, value]) => [key, rate(value.fallbacks, value.entries + value.fallbacks)])),
      },
      entries: total,
      entriesWith: Object.fromEntries(KINDS.map((kind) => [kind, rate(withKind[kind], total)])) as Record<SummaryCheckKind, Rate>,
      codes,
      coreFalsePositives,
      lengths: Object.fromEntries(SUMMARY_FIELDS.map((field) => [field, distribution(results.filter((_, index) => corpus.entries[index]!.fields[field]).map((result) => result.lengths[field]))])) as Record<SummaryFieldKey, LengthDistribution>,
      mainResultOutsideRange: rate(outsideRange, rangeApplies),
      notStated,
      claims: { occurrences: claimOccurrences, entriesWith: rate(claimEntries, total), byField: claimsByField },
    },
  };
}

export interface MonthlyMainResult { month: string; entries: number; median: number; p90: number; inRange: Rate }

/** Per report month: how long mainResult runs, to see whether a prompt edit moved it. Empty fields are left out. */
export function monthlyMainResult(entries: readonly SummaryEntry[]): MonthlyMainResult[] {
  const months = new Map<string, SummaryEntry[]>();
  for (const entry of entries) {
    if (!entry.fields.mainResult) continue;
    const key = entry.date.slice(0, 7);
    months.set(key, [...(months.get(key) ?? []), entry]);
  }
  return [...months].sort(([a], [b]) => a.localeCompare(b)).map(([month, group]) => {
    const spread = distribution(group.map((entry) => measureFieldLength(entry.fields.mainResult)));
    const applicable = group.filter((entry) => mainResultRangeApplies(entry.fields.mainResult));
    const inside = applicable.filter((entry) => {
      const length = measureFieldLength(entry.fields.mainResult);
      return length >= MAIN_RESULT_LENGTH_RANGE.min && length <= MAIN_RESULT_LENGTH_RANGE.max;
    });
    return { month, entries: group.length, median: spread.median, p90: spread.p90, inRange: rate(inside.length, applicable.length) };
  });
}
