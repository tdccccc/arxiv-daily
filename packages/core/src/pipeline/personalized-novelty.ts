import type { ChatMessage, CallOptions } from "../llm/client";
import type { MetricsObserver } from "../metrics/generation";
import { throwIfCancelled } from "../services/cancellation";
import { paperKeyFromArxivId } from "../services/paper-key";
import injectionGuard from "../prompts/injection-guard.en.md";
import personalNoveltyPromptTemplate from "../prompts/personal-novelty.system.md";
import { renderPrompt } from "../prompts/render";
import { escapePaperDataFence } from "./prompt-safety";

export const PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION = 1 as const;
export const PERSONAL_NOVELTY_RESULT_CONTRACT_VERSION = 1 as const;
export const PERSONAL_NOVELTY_EVIDENCE_DEPTH = "metadata-and-abstract" as const;
export const PERSONAL_NOVELTY_DIFFERENCE_TYPES = [
  "new-task",
  "new-method",
  "new-dataset",
  "new-experiment",
  "efficiency-result",
  "counter-evidence",
] as const;
export type PersonalNoveltyDifferenceType =
  (typeof PERSONAL_NOVELTY_DIFFERENCE_TYPES)[number];

/** Whole-run bound: at most this many library-derived papers receive results. */
export const PERSONAL_NOVELTY_MAX_PAPERS = 400 as const;
/** Whole-run bound: at most this many logical novelty calls. */
export const PERSONAL_NOVELTY_MAX_CALLS = 400 as const;
export const PERSONAL_NOVELTY_MAX_TITLE_CODE_UNITS = 2_000 as const;
export const PERSONAL_NOVELTY_MAX_ABSTRACT_CODE_UNITS = 6_000 as const;
export const PERSONAL_NOVELTY_MAX_AUTHORS = 16 as const;
export const PERSONAL_NOVELTY_MAX_AUTHOR_CODE_UNITS = 120 as const;
export const PERSONAL_NOVELTY_MAX_PUBLISHED_CODE_UNITS = 32 as const;
export const PERSONAL_NOVELTY_MAX_CATEGORIES = 16 as const;
export const PERSONAL_NOVELTY_MAX_CATEGORY_CODE_UNITS = 32 as const;
/** Per-call bound: complete comparison basis at most this many representative papers. */
export const PERSONAL_NOVELTY_MAX_COMPARISON_BASIS = 40 as const;
/** Per-call bound: rendered prompt at most this many UTF-16 code units. */
export const PERSONAL_NOVELTY_MAX_CALL_CODE_UNITS = 60_000 as const;
export const PERSONAL_NOVELTY_MAX_OUTPUT_CODE_UNITS = 16_000 as const;
export const PERSONAL_NOVELTY_MAX_COMPLETION_TOKENS = 4_096 as const;
export const PERSONAL_NOVELTY_MAX_EXPLANATION_CODE_UNITS = 1_000 as const;
export const PERSONAL_NOVELTY_MAX_DIRECTIONS = 256 as const;
export const PERSONAL_NOVELTY_MAX_DIRECTION_IDS_PER_PAPER = 64 as const;
export const PERSONAL_NOVELTY_MAX_REPRESENTATIVES_PER_DIRECTION = 5 as const;
/** DTO bound: representative pool cannot exceed directions × per-direction representatives. */
export const PERSONAL_NOVELTY_MAX_REPRESENTATIVES =
  PERSONAL_NOVELTY_MAX_DIRECTIONS * PERSONAL_NOVELTY_MAX_REPRESENTATIVES_PER_DIRECTION;
export const PERSONAL_NOVELTY_MAX_ID_LENGTH = 128 as const;
export const PERSONAL_NOVELTY_VALIDATION_ATTEMPTS = 3 as const;
export const PERSONAL_NOVELTY_MAX_AGGREGATE_PROMPT_CODE_UNITS = 4_000_000 as const;
/**
 * Whole-run ceiling for worst-case completion tokens across all validation
 * attempts: max calls × attempts × per-call tokens (the filter idiom of an
 * exact full-run worst case; the retry-aware aggregate prompt budget is the
 * binding whole-run limit).
 */
export const PERSONAL_NOVELTY_MAX_AGGREGATE_COMPLETION_TOKENS =
  PERSONAL_NOVELTY_MAX_CALLS
  * PERSONAL_NOVELTY_VALIDATION_ATTEMPTS
  * PERSONAL_NOVELTY_MAX_COMPLETION_TOKENS;

const RETRY_GUIDANCE_PREFIX = "\nPrevious output failed validation: ";
const RETRY_GUIDANCE_SUFFIX = ". Return a fresh result satisfying the contract.";

export const PERSONAL_NOVELTY_VALIDATION_REASONS = [
  "not-json",
  "wrong-shape",
  "difference-type-invalid",
  "basis-invalid",
  "evidence-depth-invalid",
  "explanation-invalid",
] as const;
export type PersonalNoveltyValidationReason =
  (typeof PERSONAL_NOVELTY_VALIDATION_REASONS)[number];

/**
 * Worst-case code units the sanitized retry guidance adds to the system
 * prompt on validation attempts 2+ (longest typed reason). The per-call
 * prompt bound reserves this so retried prompts cannot exceed
 * PERSONAL_NOVELTY_MAX_CALL_CODE_UNITS.
 */
export const PERSONAL_NOVELTY_MAX_RETRY_GUIDANCE_CODE_UNITS = RETRY_GUIDANCE_PREFIX.length
  + Math.max(...PERSONAL_NOVELTY_VALIDATION_REASONS.map((reason) => reason.length))
  + RETRY_GUIDANCE_SUFFIX.length;

/** Strict host-neutral daily paper evidence: identity, title, abstract only. */
export interface NoveltyDailyPaper {
  paperKey: string;
  title: string;
  abstract: string;
}

/** Strict host-neutral representative prior-paper evidence: metadata and abstract only. */
export interface NoveltyRepresentativePaper {
  paperKey: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  categories: string[];
}

/**
 * Trusted host-neutral DTO; it cannot carry paths, PDF bytes, fingerprints,
 * authorization, credentials, or unrelated catalog or file records.
 * Both arrays are bounded, deduplicated, and canonically sorted by paperKey.
 */
export interface PersonalizedNoveltyInput {
  papers: NoveltyDailyPaper[];
  representatives: NoveltyRepresentativePaper[];
}

/** A library-derived daily paper and the direction ids it matched. */
export interface PersonalNoveltyPaperMatch {
  paperKey: string;
  directionIds: string[];
}

/** One confirmed direction and its representative prior-paper paperKeys. */
export interface PersonalNoveltyDirectionRepresentatives {
  directionId: string;
  representativePaperKeys: string[];
}

/**
 * Caller-supplied matched direction→representative mapping. The generator
 * derives each paper's complete comparison basis by deterministic union.
 */
export interface PersonalNoveltyMatchInput {
  paperMatches: PersonalNoveltyPaperMatch[];
  directionRepresentatives: PersonalNoveltyDirectionRepresentatives[];
}

/**
 * Strict validated novelty result DTO: exact keys, known difference-type enum,
 * a non-empty unique subset of the supplied basis paperKeys in code-unit
 * order, the exact metadata-and-abstract evidence depth, and a bounded
 * non-empty trimmed explanation.
 */
export interface PersonalNovelty {
  differenceType: PersonalNoveltyDifferenceType;
  comparisonBasis: string[];
  evidenceDepth: typeof PERSONAL_NOVELTY_EVIDENCE_DEPTH;
  explanation: string;
}

export type PersonalNoveltyNoNoveltyReason =
  | "plan-too-large"
  | "validation-exhausted";

export type PersonalNoveltyPaperOutcome =
  | { paperKey: string; status: "novelty"; novelty: PersonalNovelty }
  | { paperKey: string; status: "no-novelty"; reason: PersonalNoveltyNoNoveltyReason };

export class PersonalNoveltyOutputLimitError extends Error {
  constructor() {
    super("personal novelty output exceeded its code-unit limit");
    this.name = "PersonalNoveltyOutputLimitError";
  }
}

export interface PersonalNoveltyRequest {
  messages: ChatMessage[];
  options: {
    temperature: 0;
    maxOutputCodeUnits: number;
    maxCompletionTokens: number;
  };
  identity: { paperKey: string; basisPaperKeys: string[] };
}

export type PersonalNoveltyPlanEntry =
  | { paperKey: string; kind: "call"; request: PersonalNoveltyRequest }
  | { paperKey: string; kind: "no-novelty"; reason: "plan-too-large" };

export interface PersonalNoveltyCallPlan {
  entries: PersonalNoveltyPlanEntry[];
  totals: {
    /** Matched (library-derived) papers that receive a per-paper outcome. */
    papers: number;
    /** Logical per-paper calls (first attempts); retries may multiply worst-case cost. */
    calls: number;
    /** Worst-case prompt code units across all validation attempts, including retry guidance. */
    aggregatePromptCodeUnits: number;
    /** Worst-case completion tokens: calls × validation attempts × per-call tokens. */
    aggregateCompletionTokens: number;
  };
}

export type PersonalNoveltyPlanResult =
  | { ok: true; value: PersonalNoveltyCallPlan }
  | { ok: false; reason: "plan-too-large" };

export interface PersonalNoveltyLlmPort {
  call(messages: ChatMessage[], options?: CallOptions): Promise<string>;
}

export interface GeneratePersonalNoveltiesOptions {
  input: unknown;
  matches: unknown;
  llm: PersonalNoveltyLlmPort;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
}

const systemPrompt = renderPrompt(personalNoveltyPromptTemplate, { injectionGuard });
const USER_PREFIX = "Compare this new paper against its complete representative basis.\n<paper_data>\n";
const USER_SUFFIX = "\n</paper_data>";

/**
 * Best-effort per-paper novelty evidence contract.
 *
 * - One logical call per library-derived daily paper; the comparison basis is
 *   the complete deterministic union (deduplicated, code-unit sorted) of the
 *   representative papers across the paper's matched directions.
 * - A complete basis that cannot fit the bounded call contract yields a typed
 *   per-paper no-novelty ("plan-too-large"); comparison evidence is never
 *   truncated.
 * - Whole-run bounds (max novelty papers, max calls, aggregate prompt code
 *   units, aggregate completion tokens) are checked by the plan before any
 *   call and produce a typed plan-too-large result. Aggregate budgets are
 *   worst-case across the up-to-3 logical validation attempts per paper,
 *   including sanitized retry guidance, so plan-too-large binds under
 *   worst-case retries.
 * - Exactly 3 logical validation attempts per paper, retried only on invalid
 *   structured output with sanitized retry prompts; raw model output is never
 *   reflected back into prompts.
 * - Transport and output-limit LLM errors are NOT swallowed here: they
 *   propagate to the caller exactly like the personalized filter, and the
 *   future pipeline stage decides degrade/no-novelty.
 * - Per-paper typed outcomes: valid novelty | no-novelty (plan-too-large,
 *   validation-exhausted).
 */
export async function generatePersonalNovelties(
  options: GeneratePersonalNoveltiesOptions,
): Promise<PersonalNoveltyPaperOutcome[]> {
  throwIfCancelled(options.signal);
  const input = preparePersonalizedNoveltyInput(options.input);
  const matches = preparePersonalNoveltyMatches(options.matches);
  if (matches.paperMatches.length === 0) return [];
  const planned = planPersonalNoveltyCalls(input, matches);
  if (!planned.ok) {
    return deepFreeze(matches.paperMatches.map(({ paperKey }) => ({
      paperKey, status: "no-novelty", reason: "plan-too-large",
    })));
  }
  const outcomes: PersonalNoveltyPaperOutcome[] = [];
  for (const entry of planned.value.entries) {
    throwIfCancelled(options.signal);
    if (entry.kind === "no-novelty") {
      outcomes.push({ paperKey: entry.paperKey, status: "no-novelty", reason: entry.reason });
      continue;
    }
    outcomes.push(await callValidatedNovelty(entry, options));
  }
  return deepFreeze(outcomes);
}

export function preparePersonalizedNoveltyInput(value: unknown): PersonalizedNoveltyInput {
  if (!isExactDataObject(value, ["papers", "representatives"])
    || !isOwnDataArray(value.papers, 0, PERSONAL_NOVELTY_MAX_PAPERS)
    || !isOwnDataArray(value.representatives, 0, PERSONAL_NOVELTY_MAX_REPRESENTATIVES)) {
    throw new TypeError("personalized novelty input must be an exact bounded paper list");
  }
  const papers: NoveltyDailyPaper[] = [];
  for (const raw of value.papers) {
    if (!isExactDataObject(raw, ["paperKey", "title", "abstract"])
      || !isCanonicalArxivPaperKey(raw.paperKey)
      || !isBoundedText(raw.title, PERSONAL_NOVELTY_MAX_TITLE_CODE_UNITS)
      || !isBoundedText(raw.abstract, PERSONAL_NOVELTY_MAX_ABSTRACT_CODE_UNITS)) {
      throw new TypeError("personalized novelty daily paper is malformed");
    }
    papers.push({ paperKey: raw.paperKey, title: raw.title, abstract: raw.abstract });
  }
  if (!isStrictlyOrderedUnique(papers.map(({ paperKey }) => paperKey))) {
    throw new TypeError("personalized novelty papers must be code-unit sorted and unique");
  }
  const representatives: NoveltyRepresentativePaper[] = [];
  for (const raw of value.representatives) {
    if (!isExactDataObject(raw, ["paperKey", "title", "authors", "abstract", "published", "categories"])
      || !isCanonicalArxivPaperKey(raw.paperKey)
      || !isBoundedText(raw.title, PERSONAL_NOVELTY_MAX_TITLE_CODE_UNITS)
      || !isOwnDataArray(raw.authors, 1, PERSONAL_NOVELTY_MAX_AUTHORS)
      || !raw.authors.every((author: unknown) =>
        isBoundedText(author, PERSONAL_NOVELTY_MAX_AUTHOR_CODE_UNITS))
      || !isBoundedText(raw.abstract, PERSONAL_NOVELTY_MAX_ABSTRACT_CODE_UNITS)
      || !isBoundedText(raw.published, PERSONAL_NOVELTY_MAX_PUBLISHED_CODE_UNITS)
      || !isOwnDataArray(raw.categories, 1, PERSONAL_NOVELTY_MAX_CATEGORIES)
      || !raw.categories.every((category: unknown) =>
        isBoundedText(category, PERSONAL_NOVELTY_MAX_CATEGORY_CODE_UNITS))) {
      throw new TypeError("personalized novelty representative paper is malformed");
    }
    representatives.push({
      paperKey: raw.paperKey,
      title: raw.title,
      authors: [...raw.authors],
      abstract: raw.abstract,
      published: raw.published,
      categories: [...raw.categories],
    });
  }
  if (!isStrictlyOrderedUnique(representatives.map(({ paperKey }) => paperKey))) {
    throw new TypeError("personalized novelty representatives must be code-unit sorted and unique");
  }
  return deepFreeze({ papers, representatives });
}

export function preparePersonalNoveltyMatches(value: unknown): PersonalNoveltyMatchInput {
  if (!isExactDataObject(value, ["paperMatches", "directionRepresentatives"])
    || !isOwnDataArray(value.paperMatches, 0, PERSONAL_NOVELTY_MAX_PAPERS)
    || !isOwnDataArray(value.directionRepresentatives, 0, PERSONAL_NOVELTY_MAX_DIRECTIONS)) {
    throw new TypeError("personal novelty matches must be an exact bounded mapping");
  }
  const paperMatches: PersonalNoveltyPaperMatch[] = [];
  for (const raw of value.paperMatches) {
    if (!isExactDataObject(raw, ["paperKey", "directionIds"])
      || !isCanonicalArxivPaperKey(raw.paperKey)
      || !isOwnDataArray(raw.directionIds, 1, PERSONAL_NOVELTY_MAX_DIRECTION_IDS_PER_PAPER)
      || !raw.directionIds.every(isOpaqueId)
      || !isStrictlyOrderedUnique(raw.directionIds)) {
      throw new TypeError("personal novelty paper match is malformed");
    }
    paperMatches.push({ paperKey: raw.paperKey, directionIds: [...raw.directionIds] });
  }
  if (!isStrictlyOrderedUnique(paperMatches.map(({ paperKey }) => paperKey))) {
    throw new TypeError("personal novelty paper matches must be code-unit sorted and unique");
  }
  const directionRepresentatives: PersonalNoveltyDirectionRepresentatives[] = [];
  for (const raw of value.directionRepresentatives) {
    if (!isExactDataObject(raw, ["directionId", "representativePaperKeys"])
      || !isOpaqueId(raw.directionId)
      || !isOwnDataArray(
        raw.representativePaperKeys, 1, PERSONAL_NOVELTY_MAX_REPRESENTATIVES_PER_DIRECTION,
      )
      || !raw.representativePaperKeys.every(isCanonicalArxivPaperKey)
      || !isStrictlyOrderedUnique(raw.representativePaperKeys)) {
      throw new TypeError("personal novelty direction representatives are malformed");
    }
    directionRepresentatives.push({
      directionId: raw.directionId,
      representativePaperKeys: [...raw.representativePaperKeys],
    });
  }
  if (!isStrictlyOrderedUnique(directionRepresentatives.map(({ directionId }) => directionId))) {
    throw new TypeError("personal novelty direction representatives must be code-unit sorted and unique");
  }
  return deepFreeze({ paperMatches, directionRepresentatives });
}

/**
 * Deterministically plans one bounded call per matched paper, or a typed
 * per-paper no-novelty when the complete comparison basis cannot fit the
 * bounded call contract (never truncating evidence), or a typed whole-run
 * plan-too-large before any call.
 *
 * Budgets are worst-case across the up-to-PERSONAL_NOVELTY_VALIDATION_ATTEMPTS
 * logical attempts per paper: the per-call prompt bound reserves the longest
 * sanitized retry guidance suffix, aggregatePromptCodeUnits = Σ (promptUnits ×
 * attempts + retryGuidance × (attempts − 1)), and aggregateCompletionTokens =
 * calls × attempts × maxCompletionTokens, so the typed whole-run plan-too-large
 * binds under worst-case retries.
 */
export function planPersonalNoveltyCalls(
  inputValue: unknown,
  matchesValue: unknown,
): PersonalNoveltyPlanResult {
  const input = preparePersonalizedNoveltyInput(inputValue);
  const matches = preparePersonalNoveltyMatches(matchesValue);
  if (matches.paperMatches.length === 0) return emptyPlan();
  const paperByKey = new Map(input.papers.map((paper) => [paper.paperKey, paper]));
  const representativeByKey = new Map(
    input.representatives.map((paper) => [paper.paperKey, paper]),
  );
  const representativesByDirection = new Map(
    matches.directionRepresentatives.map((entry) => [entry.directionId, entry.representativePaperKeys]),
  );
  const entries: PersonalNoveltyPlanEntry[] = [];
  let aggregatePromptCodeUnits = 0;
  let calls = 0;
  for (const match of matches.paperMatches) {
    const paper = paperByKey.get(match.paperKey);
    if (!paper) {
      throw new TypeError(`personal novelty match references unknown daily paper: ${match.paperKey}`);
    }
    const basisKeySet = new Set<string>();
    for (const directionId of match.directionIds) {
      const keys = representativesByDirection.get(directionId);
      if (!keys) {
        throw new TypeError(`personal novelty match references unknown direction: ${directionId}`);
      }
      for (const key of keys) basisKeySet.add(key);
    }
    const basisPaperKeys = [...basisKeySet].sort(codeUnitCompare);
    if (basisPaperKeys.length > PERSONAL_NOVELTY_MAX_COMPARISON_BASIS) {
      entries.push({ paperKey: match.paperKey, kind: "no-novelty", reason: "plan-too-large" });
      continue;
    }
    const basis: NoveltyRepresentativePaper[] = [];
    for (const key of basisPaperKeys) {
      const representative = representativeByKey.get(key);
      if (!representative) {
        throw new TypeError(
          `personal novelty direction references unknown representative paper: ${key}`,
        );
      }
      basis.push(representative);
    }
    const request = buildPersonalNoveltyRequest(paper, basis);
    const promptUnits = request.messages.reduce((sum, message) => sum + message.content.length, 0);
    if (promptUnits + PERSONAL_NOVELTY_MAX_RETRY_GUIDANCE_CODE_UNITS
      > PERSONAL_NOVELTY_MAX_CALL_CODE_UNITS) {
      entries.push({ paperKey: match.paperKey, kind: "no-novelty", reason: "plan-too-large" });
      continue;
    }
    entries.push({ paperKey: match.paperKey, kind: "call", request });
    aggregatePromptCodeUnits += promptUnits * PERSONAL_NOVELTY_VALIDATION_ATTEMPTS
      + PERSONAL_NOVELTY_MAX_RETRY_GUIDANCE_CODE_UNITS * (PERSONAL_NOVELTY_VALIDATION_ATTEMPTS - 1);
    calls += 1;
  }
  const aggregateCompletionTokens = calls * PERSONAL_NOVELTY_VALIDATION_ATTEMPTS
    * PERSONAL_NOVELTY_MAX_COMPLETION_TOKENS;
  if (matches.paperMatches.length > PERSONAL_NOVELTY_MAX_PAPERS
    || calls > PERSONAL_NOVELTY_MAX_CALLS
    || aggregatePromptCodeUnits > PERSONAL_NOVELTY_MAX_AGGREGATE_PROMPT_CODE_UNITS
    || aggregateCompletionTokens > PERSONAL_NOVELTY_MAX_AGGREGATE_COMPLETION_TOKENS) {
    return planTooLarge();
  }
  return {
    ok: true,
    value: deepFreeze({
      entries,
      totals: {
        papers: matches.paperMatches.length,
        calls,
        aggregatePromptCodeUnits,
        aggregateCompletionTokens,
      },
    }),
  };
}

export function buildPersonalNoveltyRequest(
  paper: NoveltyDailyPaper,
  basis: readonly NoveltyRepresentativePaper[],
): PersonalNoveltyRequest {
  const payload = JSON.stringify({ paper, basis });
  const safePayload = escapePaperDataFence(payload);
  return {
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: `${USER_PREFIX}${safePayload}${USER_SUFFIX}` },
    ],
    options: {
      temperature: 0,
      maxOutputCodeUnits: PERSONAL_NOVELTY_MAX_OUTPUT_CODE_UNITS,
      maxCompletionTokens: PERSONAL_NOVELTY_MAX_COMPLETION_TOKENS,
    },
    identity: {
      paperKey: paper.paperKey,
      basisPaperKeys: basis.map(({ paperKey }) => paperKey),
    },
  };
}

/**
 * Strict decode of the model's JSON: exact keys, known enum, non-empty unique
 * subset of the supplied basis paperKeys in code-unit order, exact evidence
 * depth literal, and bounded trimmed explanation. Fails wholly — partial
 * output is never promoted.
 */
export function decodePersonalNovelty(
  raw: string,
  basisPaperKeys: ReadonlySet<string>,
): { ok: true; value: PersonalNovelty } | { ok: false; reason: PersonalNoveltyValidationReason } {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return { ok: false, reason: "not-json" };
  }
  if (!isExactDataObject(value, [
    "differenceType", "comparisonBasis", "evidenceDepth", "explanation",
  ])) {
    return { ok: false, reason: "wrong-shape" };
  }
  if (!PERSONAL_NOVELTY_DIFFERENCE_TYPES.includes(value.differenceType)) {
    return { ok: false, reason: "difference-type-invalid" };
  }
  if (!Array.isArray(value.comparisonBasis) || !hasOwnDataArrayEntries(value.comparisonBasis)
    || value.comparisonBasis.length < 1
    || !value.comparisonBasis.every(
      (key: unknown) => typeof key === "string" && basisPaperKeys.has(key),
    )
    || !isStrictlyOrderedUnique(value.comparisonBasis)) {
    return { ok: false, reason: "basis-invalid" };
  }
  if (value.evidenceDepth !== PERSONAL_NOVELTY_EVIDENCE_DEPTH) {
    return { ok: false, reason: "evidence-depth-invalid" };
  }
  if (!isBoundedText(value.explanation, PERSONAL_NOVELTY_MAX_EXPLANATION_CODE_UNITS)) {
    return { ok: false, reason: "explanation-invalid" };
  }
  return {
    ok: true,
    value: deepFreeze({
      differenceType: value.differenceType,
      comparisonBasis: [...value.comparisonBasis],
      evidenceDepth: PERSONAL_NOVELTY_EVIDENCE_DEPTH,
      explanation: value.explanation,
    }),
  };
}

async function callValidatedNovelty(
  entry: Extract<PersonalNoveltyPlanEntry, { kind: "call" }>,
  options: GeneratePersonalNoveltiesOptions,
): Promise<PersonalNoveltyPaperOutcome> {
  const basisPaperKeys = new Set(entry.request.identity.basisPaperKeys);
  let reason: PersonalNoveltyValidationReason = "wrong-shape";
  for (let attempt = 1; attempt <= PERSONAL_NOVELTY_VALIDATION_ATTEMPTS; attempt += 1) {
    throwIfCancelled(options.signal);
    const guidance = attempt === 1 ? ""
      : `${RETRY_GUIDANCE_PREFIX}${reason}${RETRY_GUIDANCE_SUFFIX}`;
    const raw = await options.llm.call([
      { role: "system", content: `${entry.request.messages[0]!.content}${guidance}` },
      { role: "user", content: entry.request.messages[1]!.content },
    ], {
      temperature: 0,
      signal: options.signal,
      onMetrics: options.onMetrics,
      maxOutputCodeUnits: PERSONAL_NOVELTY_MAX_OUTPUT_CODE_UNITS,
      maxCompletionTokens: PERSONAL_NOVELTY_MAX_COMPLETION_TOKENS,
    });
    throwIfCancelled(options.signal);
    if (raw.length > PERSONAL_NOVELTY_MAX_OUTPUT_CODE_UNITS) {
      throw new PersonalNoveltyOutputLimitError();
    }
    const decoded = decodePersonalNovelty(raw, basisPaperKeys);
    if (decoded.ok) {
      return { paperKey: entry.paperKey, status: "novelty", novelty: decoded.value };
    }
    reason = decoded.reason;
  }
  return { paperKey: entry.paperKey, status: "no-novelty", reason: "validation-exhausted" };
}

function emptyPlan(): PersonalNoveltyPlanResult {
  return {
    ok: true,
    value: deepFreeze({
      entries: [],
      totals: { papers: 0, calls: 0, aggregatePromptCodeUnits: 0, aggregateCompletionTokens: 0 },
    }),
  };
}

function planTooLarge(): PersonalNoveltyPlanResult {
  return { ok: false, reason: "plan-too-large" };
}

function isCanonicalArxivPaperKey(value: unknown): value is string {
  if (typeof value !== "string" || !value.startsWith("arxiv:")) return false;
  try {
    return paperKeyFromArxivId(value.slice("arxiv:".length)) === value;
  } catch {
    return false;
  }
}

function isOpaqueId(value: unknown): value is string {
  return typeof value === "string" && value.length >= 1
    && value.length <= PERSONAL_NOVELTY_MAX_ID_LENGTH && /^[A-Za-z0-9._~-]+$/.test(value);
}

function isBoundedText(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum
    && value.trim() === value;
}

function isOwnDataArray(value: unknown, minimum: number, maximum: number): value is unknown[] {
  return Array.isArray(value) && hasOwnDataArrayEntries(value)
    && value.length >= minimum && value.length <= maximum;
}

function hasOwnDataArrayEntries(value: unknown[]): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !("value" in descriptor)) return false;
  }
  return true;
}

function isStrictlyOrderedUnique(value: readonly string[]): boolean {
  return value.every((entry, index) => index === 0 || codeUnitCompare(value[index - 1]!, entry) < 0);
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function isExactDataObject(value: unknown, keys: readonly string[]): value is Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) return false;
  const actual = Object.keys(value).sort(codeUnitCompare);
  const expected = [...keys].sort(codeUnitCompare);
  if (actual.length !== expected.length
    || !actual.every((key, index) => key === expected[index])) return false;
  return actual.every((key) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return Boolean(descriptor && "value" in descriptor);
  });
}

function deepFreeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}
