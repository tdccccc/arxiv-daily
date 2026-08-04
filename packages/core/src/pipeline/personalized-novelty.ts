import type { ChatMessage, CallOptions } from "../llm/client";
import type { MetricsObserver } from "../metrics/generation";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import { buildCheckpointGenerationIdentity } from "../services/daily-summary-checkpoint-store";
import { paperKeyFromArxivId } from "../services/paper-key";
import type { LlmSettings } from "../settings/types";
import { sha256Hex } from "../utils/digest";
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
  | "validation-exhausted"
  | "input-invalid";

/**
 * Stage-level no-novelty reasons beyond the generator's typed outcomes:
 * "checkpoint" covers prepared-snapshot, lookup, and save failures; "transport"
 * and "output-limit" are per-paper LLM failures the stage degrades instead of
 * propagating; "input-invalid" is a per-paper reference violation (unknown
 * daily paper, unknown matched direction, or unknown representative) that the
 * stage excludes from the plan without degrading valid papers.
 */
export type PersonalNoveltyStageNoNoveltyReason =
  | PersonalNoveltyNoNoveltyReason
  | "checkpoint"
  | "transport"
  | "output-limit";

export type PersonalNoveltyStageOutcome =
  | { paperKey: string; status: "novelty"; novelty: PersonalNovelty }
  | { paperKey: string; status: "no-novelty"; reason: PersonalNoveltyStageNoNoveltyReason };

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

export const NOVELTY_CHECKPOINT_FINGERPRINT_VERSION = 1 as const;

/**
 * One persisted terminal outcome per planned call paper, keyed by canonical
 * paperKey: a validated "novelty" result or a typed "no-novelty" outcome
 * (validation-exhausted for call entries, plan-too-large for plan entries).
 * The persisted result must cover every planned call paper exactly once;
 * partial coverage is treated as a miss and never reused.
 */
export type NoveltyCheckpointRecord =
  | { paperKey: string; status: "novelty"; novelty: PersonalNovelty }
  | { paperKey: string; status: "no-novelty"; reason: "validation-exhausted" | "plan-too-large" };

/**
 * Branded frozen snapshot of the exact per-paper call plan actually consumed by
 * live generation calls, the direction identity (matches), the effective
 * generation identity, and the prompt/result contract versions. The fingerprint
 * is sha256 of the exact JSON of fingerprintInput; load recomputes and compares
 * it before any result record is trusted.
 */
export interface PreparedNoveltyCheckpoint {
  readonly fingerprintInput: {
    fingerprintVersion: typeof NOVELTY_CHECKPOINT_FINGERPRINT_VERSION;
    promptContractVersion: number;
    resultContractVersion: number;
    /** Direction identity: per-paper matched directions and direction representatives. */
    matches: PersonalNoveltyMatchInput;
    /** Exact per-paper call plan (messages, options, identity, totals). */
    plan: PersonalNoveltyCallPlan;
    generation: ReturnType<typeof buildCheckpointGenerationIdentity>;
  };
  readonly fingerprint: string;
}

/** Checkpoint port methods are distinctly named to avoid overload ambiguity with filter ports. */
export interface NoveltyCheckpointPort {
  lookupNoveltyReusable(
    reportDate: string,
    prepared: PreparedNoveltyCheckpoint,
  ): Promise<NoveltyCheckpointRecord[] | null>;
  saveNovelty(
    reportDate: string,
    prepared: PreparedNoveltyCheckpoint,
    result: PersonalNoveltyStageOutcome[],
  ): Promise<unknown>;
}

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
const preparedNoveltyPlans = new WeakSet<object>();
const preparedNoveltySnapshots = new WeakSet<object>();

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
  const plan = deepFreeze({
    entries,
    totals: {
      papers: matches.paperMatches.length,
      calls,
      aggregatePromptCodeUnits,
      aggregateCompletionTokens,
    },
  });
  preparedNoveltyPlans.add(plan);
  return { ok: true, value: plan };
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

/**
 * Builds a branded frozen checkpoint snapshot from the exact plan produced by
 * planPersonalNoveltyCalls (the same planned requests live generation consumes)
 * plus the direction identity, the effective generation identity, and the
 * prompt/result contract versions.
 */
export function prepareNoveltyCheckpoint(input: {
  plan: PersonalNoveltyCallPlan;
  matches: PersonalNoveltyMatchInput;
  llm: LlmSettings;
  promptContractVersion?: number;
  resultContractVersion?: number;
}): PreparedNoveltyCheckpoint {
  if (!preparedNoveltyPlans.has(input.plan) || !isValidNoveltyPlan(input.plan)) {
    throw new TypeError("personal novelty plan must be an exact prepared call plan");
  }
  const matches = preparePersonalNoveltyMatches(input.matches);
  const fingerprintInput = {
    fingerprintVersion: NOVELTY_CHECKPOINT_FINGERPRINT_VERSION,
    promptContractVersion: input.promptContractVersion ?? PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION,
    resultContractVersion: input.resultContractVersion ?? PERSONAL_NOVELTY_RESULT_CONTRACT_VERSION,
    matches: clone(matches),
    plan: clone(input.plan),
    generation: buildCheckpointGenerationIdentity(input.llm, 0),
  };
  const snapshot = {
    fingerprintInput,
    fingerprint: fingerprintNoveltyCheckpointInput(fingerprintInput),
  };
  deepFreeze(snapshot);
  preparedNoveltySnapshots.add(snapshot);
  return snapshot;
}

export function fingerprintNoveltyCheckpointInput(
  input: PreparedNoveltyCheckpoint["fingerprintInput"],
): string {
  return `sha256:${sha256Hex(JSON.stringify(input))}`;
}

export function isPreparedNoveltyCheckpoint(
  value: unknown,
): value is PreparedNoveltyCheckpoint {
  return typeof value === "object" && value !== null && preparedNoveltySnapshots.has(value);
}

/**
 * Strict decode of a persisted fingerprintInput: exact keys, supported contract
 * versions, canonical matches, a valid exact call plan, and a valid generation
 * identity. The caller recomputes the fingerprint over the result.
 */
export function decodeNoveltyFingerprintInput(
  value: unknown,
): PreparedNoveltyCheckpoint["fingerprintInput"] | null {
  if (!isExactDataObject(value, [
    "fingerprintVersion", "promptContractVersion", "resultContractVersion", "matches", "plan",
    "generation",
  ]) || value.fingerprintVersion !== NOVELTY_CHECKPOINT_FINGERPRINT_VERSION
    || value.promptContractVersion !== PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION
    || value.resultContractVersion !== PERSONAL_NOVELTY_RESULT_CONTRACT_VERSION
    || !isValidNoveltyPlan(value.plan)
    || !isCheckpointGenerationIdentity(value.generation)) return null;
  let matches: PersonalNoveltyMatchInput;
  try {
    matches = preparePersonalNoveltyMatches(value.matches);
  } catch {
    return null;
  }
  return clone({
    fingerprintVersion: NOVELTY_CHECKPOINT_FINGERPRINT_VERSION,
    promptContractVersion: PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION,
    resultContractVersion: PERSONAL_NOVELTY_RESULT_CONTRACT_VERSION,
    matches,
    plan: value.plan,
    generation: value.generation,
  });
}

/**
 * Strict structural validation of the exact per-paper call plan: bounded
 * ordered-unique canonical entries, exact request options, message shapes with
 * the exact rendered system prompt and user fence, identity consistent with the
 * entry paperKey and a valid basis, and totals recomputed from the entries.
 * Exact rendered content identity is additionally guaranteed by the fingerprint
 * recompute over fingerprintInput at load.
 */
export function isValidNoveltyPlan(plan: unknown): plan is PersonalNoveltyCallPlan {
  if (!isExactDataObject(plan, ["entries", "totals"])
    || !Array.isArray(plan.entries) || !hasOwnDataArrayEntries(plan.entries)
    || !isExactDataObject(plan.totals, [
      "papers", "calls", "aggregatePromptCodeUnits", "aggregateCompletionTokens",
    ]) || !Object.values(plan.totals).every((entry) => Number.isSafeInteger(entry) && entry >= 0)
    || plan.entries.length > PERSONAL_NOVELTY_MAX_PAPERS) return false;
  const seen = new Set<string>();
  let previousKey = "";
  let callCount = 0;
  let aggregatePromptCodeUnits = 0;
  for (const entry of plan.entries) {
    if (typeof entry !== "object" || entry === null
      || (entry as { kind?: unknown }).kind !== "call"
        && (entry as { kind?: unknown }).kind !== "no-novelty") return false;
    if (entry.kind === "no-novelty") {
      if (!isExactDataObject(entry, ["paperKey", "kind", "reason"])
        || entry.reason !== "plan-too-large") return false;
    } else if (!isExactDataObject(entry, ["paperKey", "kind", "request"])) {
      return false;
    }
    if (!isCanonicalArxivPaperKey(entry.paperKey) || seen.has(entry.paperKey)
      || (previousKey !== "" && codeUnitCompare(previousKey, entry.paperKey) >= 0)) return false;
    seen.add(entry.paperKey);
    previousKey = entry.paperKey;
    if (entry.kind === "no-novelty") continue;
    const request = entry.request;
    if (!isExactDataObject(request, ["messages", "options", "identity"])
      || !Array.isArray(request.messages) || !hasOwnDataArrayEntries(request.messages)
      || request.messages.length !== 2
      || !request.messages.every((message: unknown) => isExactDataObject(message, ["role", "content"])
        && (message.role === "system" || message.role === "user") && typeof message.content === "string")
      || request.messages[0]!.content !== systemPrompt
      || !request.messages[1]!.content.startsWith(USER_PREFIX)
      || !request.messages[1]!.content.endsWith(USER_SUFFIX)
      || !isExactDataObject(request.options, [
        "temperature", "maxOutputCodeUnits", "maxCompletionTokens",
      ]) || request.options.temperature !== 0
      || request.options.maxOutputCodeUnits !== PERSONAL_NOVELTY_MAX_OUTPUT_CODE_UNITS
      || request.options.maxCompletionTokens !== PERSONAL_NOVELTY_MAX_COMPLETION_TOKENS
      || !isExactDataObject(request.identity, ["paperKey", "basisPaperKeys"])
      || request.identity.paperKey !== entry.paperKey
      || !Array.isArray(request.identity.basisPaperKeys)
      || !hasOwnDataArrayEntries(request.identity.basisPaperKeys)
      || request.identity.basisPaperKeys.length < 1
      || request.identity.basisPaperKeys.length > PERSONAL_NOVELTY_MAX_COMPARISON_BASIS
      || !request.identity.basisPaperKeys.every(isCanonicalArxivPaperKey)
      || !isStrictlyOrderedUnique(request.identity.basisPaperKeys)) return false;
    const promptUnits = request.messages.reduce(
      (sum: number, message: ChatMessage) => sum + message.content.length, 0,
    );
    if (promptUnits + PERSONAL_NOVELTY_MAX_RETRY_GUIDANCE_CODE_UNITS
      > PERSONAL_NOVELTY_MAX_CALL_CODE_UNITS) return false;
    callCount += 1;
    aggregatePromptCodeUnits += promptUnits * PERSONAL_NOVELTY_VALIDATION_ATTEMPTS
      + PERSONAL_NOVELTY_MAX_RETRY_GUIDANCE_CODE_UNITS * (PERSONAL_NOVELTY_VALIDATION_ATTEMPTS - 1);
  }
  return plan.totals.papers === plan.entries.length
    && plan.totals.calls === callCount
    && plan.totals.aggregatePromptCodeUnits === aggregatePromptCodeUnits
    && plan.totals.aggregateCompletionTokens === callCount * PERSONAL_NOVELTY_VALIDATION_ATTEMPTS
      * PERSONAL_NOVELTY_MAX_COMPLETION_TOKENS;
}

/**
 * Strict decode of persisted terminal outcomes: a bounded array (never more
 * records than planned call entries), exact {paperKey, status, ...} records
 * ordered unique by canonical paperKey, each paper present in the plan, each
 * novelty strictly validated against that entry's complete basis, each
 * no-novelty reason persistable and consistent with the entry kind (a call
 * entry can only end validation-exhausted; a plan no-novelty entry can only end
 * plan-too-large), and exact coverage of every planned call paper — partial
 * coverage is invalid and treated as a miss.
 */
export function decodeNoveltyCheckpointRecords(
  value: unknown,
  plan: PersonalNoveltyCallPlan,
): { ok: true; value: NoveltyCheckpointRecord[] } | { ok: false; reason: string } {
  if (!Array.isArray(value) || !hasOwnDataArrayEntries(value)) {
    return { ok: false, reason: "result must be an array" };
  }
  if (value.length > plan.entries.length) {
    return { ok: false, reason: "result exceeds the planned call entries" };
  }
  const basisByPaperKey = new Map<string, Set<string>>();
  const expectedKindByKey = new Map<string, "call" | "no-novelty">();
  for (const entry of plan.entries) {
    expectedKindByKey.set(entry.paperKey, entry.kind);
    if (entry.kind === "call") {
      basisByPaperKey.set(entry.paperKey, new Set(entry.request.identity.basisPaperKeys));
    }
  }
  const records: NoveltyCheckpointRecord[] = [];
  const seen = new Set<string>();
  let previousKey = "";
  for (const raw of value) {
    if (typeof raw !== "object" || raw === null
      || ((raw as { status?: unknown }).status !== "novelty"
        && (raw as { status?: unknown }).status !== "no-novelty")) {
      return { ok: false, reason: "novelty record is malformed" };
    }
    if (raw.status === "novelty") {
      if (!isExactDataObject(raw, ["paperKey", "status", "novelty"])) {
        return { ok: false, reason: "novelty record is malformed" };
      }
    } else if (!isExactDataObject(raw, ["paperKey", "status", "reason"])) {
      return { ok: false, reason: "novelty record is malformed" };
    }
    if (!isCanonicalArxivPaperKey(raw.paperKey) || seen.has(raw.paperKey)
      || (previousKey !== "" && codeUnitCompare(previousKey, raw.paperKey) >= 0)) {
      return { ok: false, reason: "novelty records must be ordered and unique" };
    }
    const expectedKind = expectedKindByKey.get(raw.paperKey);
    if (!expectedKind) {
      return { ok: false, reason: "novelty record references an unknown or unplanned paper" };
    }
    seen.add(raw.paperKey);
    previousKey = raw.paperKey;
    if (raw.status === "novelty") {
      if (expectedKind !== "call") {
        return { ok: false, reason: "novelty record does not match the planned entry" };
      }
      const decoded = decodePersonalNovelty(
        JSON.stringify(raw.novelty),
        basisByPaperKey.get(raw.paperKey)!,
      );
      if (!decoded.ok) {
        return { ok: false, reason: `novelty record is invalid: ${decoded.reason}` };
      }
      records.push({ paperKey: raw.paperKey, status: "novelty", novelty: decoded.value });
    } else {
      if (raw.reason !== "validation-exhausted" && raw.reason !== "plan-too-large") {
        return { ok: false, reason: "novelty no-novelty reason is not persistable" };
      }
      if (expectedKind === "call" ? raw.reason !== "validation-exhausted"
        : raw.reason !== "plan-too-large") {
        return { ok: false, reason: "novelty no-novelty record does not match the planned entry" };
      }
      records.push({ paperKey: raw.paperKey, status: "no-novelty", reason: raw.reason });
    }
  }
  if (records.length !== plan.entries.length) {
    return { ok: false, reason: "result must cover every planned paper" };
  }
  return { ok: true, value: records };
}

/**
 * Deterministic best-effort pipeline novelty stage.
 *
 * Degrade-to-no-novelty semantics (documented decision; novelty is additive
 * evidence and must never fail, block, or rewrite the reliable daily run):
 * - Malformed input DTOs (TypeError from the strict prepare contracts) degrade
 *   the whole stage to no-novelty with a sanitized warning.
 * - Reference violations are per-paper, never whole-stage: every library-derived
 *   paper is pre-validated against the supplied daily evidence, matched
 *   directions, and representative evidence; papers with broken references get
 *   a typed per-paper no-novelty ("input-invalid") and valid papers continue.
 * - Whole-run or per-paper plan-too-large yields typed per-paper no-novelty
 *   before any call.
 * - Checkpoint prepare/lookup/save failures degrade the whole stage to
 *   no-novelty with a sanitized warning (the checkpoint is the durable reuse
 *   record; a run that cannot read or write it behaves as if no novelty was
 *   produced rather than failing or persisting unrecoverable state).
 * - Transport and output-limit LLM failures are caught and degrade only the
 *   affected paper, and when any paper degrades this way the checkpoint save is
 *   skipped entirely so degraded papers are regenerated on rerun and never
 *   durably marked no-novelty. Cancellation is never swallowed:
 *   isCancellationError is rethrown.
 * - A checkpoint hit is reused only when its persisted records are strictly
 *   valid and cover every planned call paper; invalid or partial records are
 *   treated as a miss and regenerate all planned papers.
 * - Per-paper generation uses single-paper match slices that render exactly the
 *   same requests as the full plan, so per-paper failure isolation never
 *   changes call identity.
 * - Warnings use fixed text and pass the error object to the redacting logger;
 *   raw error messages are never embedded in warning text.
 */
export async function runPersonalNoveltyStage(
  options: {
    input: PersonalizedNoveltyInput;
    matches: PersonalNoveltyMatchInput;
    llm: PersonalNoveltyLlmPort;
    llmSettings: LlmSettings;
    reportDate: string;
    checkpointStore?: NoveltyCheckpointPort;
    signal?: AbortSignal;
    onMetrics?: MetricsObserver;
    onWarning?: (message: string, error?: unknown) => void;
  },
): Promise<{ outcomes: PersonalNoveltyStageOutcome[]; reusedCheckpoint: boolean }> {
  throwIfCancelled(options.signal);
  let input: PersonalizedNoveltyInput;
  let matches: PersonalNoveltyMatchInput;
  try {
    input = preparePersonalizedNoveltyInput(options.input);
    matches = preparePersonalNoveltyMatches(options.matches);
  } catch (error) {
    if (isCancellationError(error)) throw error;
    throwIfCancelled(options.signal);
    options.onWarning?.("personal novelty input invalid", error);
    return { outcomes: [], reusedCheckpoint: false };
  }
  if (matches.paperMatches.length === 0) return { outcomes: [], reusedCheckpoint: false };

  // Per-paper reference validation: unknown daily papers, unknown matched
  // directions, or unknown representatives produce a typed per-paper no-novelty
  // ("input-invalid") and never join the plan; one broken reference cannot
  // degrade the valid papers.
  const partitioned = partitionValidPapers(input, matches);
  const invalidOutcomes: PersonalNoveltyStageOutcome[] = partitioned.invalid.map(({ paperKey }) => ({
    paperKey, status: "no-novelty", reason: "input-invalid",
  }));
  if (partitioned.valid.length === 0) {
    return { outcomes: invalidOutcomes, reusedCheckpoint: false };
  }
  const validMatches: PersonalNoveltyMatchInput = {
    paperMatches: partitioned.valid,
    directionRepresentatives: matches.directionRepresentatives,
  };

  let planned: PersonalNoveltyCallPlan;
  try {
    const result = planPersonalNoveltyCalls(input, validMatches);
    if (!result.ok) {
      return {
        outcomes: [
          ...invalidOutcomes,
          ...partitioned.valid.map(({ paperKey }) => ({
            paperKey, status: "no-novelty" as const, reason: "plan-too-large" as const,
          })),
        ],
        reusedCheckpoint: false,
      };
    }
    planned = result.value;
  } catch (error) {
    if (isCancellationError(error)) throw error;
    throwIfCancelled(options.signal);
    options.onWarning?.("personal novelty plan invalid", error);
    return {
      outcomes: [
        ...invalidOutcomes,
        ...partitioned.valid.map(({ paperKey }) => ({
          paperKey, status: "no-novelty" as const, reason: "input-invalid" as const,
        })),
      ],
      reusedCheckpoint: false,
    };
  }

  let prepared: PreparedNoveltyCheckpoint | undefined;
  if (options.checkpointStore) {
    try {
      prepared = prepareNoveltyCheckpoint({
        plan: planned, matches: validMatches, llm: options.llmSettings,
      });
    } catch (error) {
      if (isCancellationError(error)) throw error;
      throwIfCancelled(options.signal);
      options.onWarning?.("personal novelty checkpoint prepare failed", error);
      return { outcomes: checkpointNoNovelty(invalidOutcomes, planned), reusedCheckpoint: false };
    }
    let reusable: NoveltyCheckpointRecord[] | null | undefined;
    try {
      reusable = await options.checkpointStore.lookupNoveltyReusable(options.reportDate, prepared);
    } catch (error) {
      if (isCancellationError(error)) throw error;
      throwIfCancelled(options.signal);
      options.onWarning?.("personal novelty checkpoint lookup failed", error);
      return { outcomes: checkpointNoNovelty(invalidOutcomes, planned), reusedCheckpoint: false };
    }
    throwIfCancelled(options.signal);
    if (reusable) {
      const decoded = decodeNoveltyCheckpointRecords(reusable, planned);
      if (decoded.ok) {
        return {
          outcomes: [...invalidOutcomes, ...decoded.value.map(recordAsOutcome)],
          reusedCheckpoint: true,
        };
      }
      options.onWarning?.(`personal novelty checkpoint result invalid: ${decoded.reason}`);
      // Invalid (including partial-coverage) persisted records are not trusted;
      // regenerate every planned paper below.
    }
  }

  const plannedOutcomes: PersonalNoveltyStageOutcome[] = [];
  let degraded = false;
  for (const entry of planned.entries) {
    throwIfCancelled(options.signal);
    if (entry.kind === "no-novelty") {
      plannedOutcomes.push({
        paperKey: entry.paperKey, status: "no-novelty", reason: "plan-too-large",
      });
      continue;
    }
    try {
      const single = await generatePersonalNovelties({
        input,
        matches: singlePaperMatches(validMatches, entry.paperKey),
        llm: options.llm,
        signal: options.signal,
        onMetrics: options.onMetrics,
      });
      plannedOutcomes.push(...single);
    } catch (error) {
      if (isCancellationError(error)) throw error;
      throwIfCancelled(options.signal);
      const reason = error instanceof PersonalNoveltyOutputLimitError
        ? "output-limit"
        : "transport";
      degraded = true;
      options.onWarning?.(
        `personal novelty call degraded for ${entry.paperKey} (${reason})`,
        error,
      );
      plannedOutcomes.push({ paperKey: entry.paperKey, status: "no-novelty", reason });
    }
  }
  if (prepared && options.checkpointStore && !degraded) {
    try {
      await options.checkpointStore.saveNovelty(options.reportDate, prepared, plannedOutcomes);
    } catch (error) {
      if (isCancellationError(error)) throw error;
      throwIfCancelled(options.signal);
      options.onWarning?.("personal novelty checkpoint save failed", error);
      return { outcomes: checkpointNoNovelty(invalidOutcomes, planned), reusedCheckpoint: false };
    }
  }
  throwIfCancelled(options.signal);
  return { outcomes: [...invalidOutcomes, ...plannedOutcomes], reusedCheckpoint: false };
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
  const plan = deepFreeze({
    entries: [],
    totals: { papers: 0, calls: 0, aggregatePromptCodeUnits: 0, aggregateCompletionTokens: 0 },
  });
  preparedNoveltyPlans.add(plan);
  return { ok: true, value: plan };
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

function checkpointNoNovelty(
  invalidOutcomes: PersonalNoveltyStageOutcome[],
  planned: PersonalNoveltyCallPlan,
): PersonalNoveltyStageOutcome[] {
  return [
    ...invalidOutcomes,
    ...planned.entries.map((entry) => ({
      paperKey: entry.paperKey, status: "no-novelty" as const, reason: "checkpoint" as const,
    })),
  ];
}

function recordAsOutcome(record: NoveltyCheckpointRecord): PersonalNoveltyStageOutcome {
  return record.status === "novelty"
    ? { paperKey: record.paperKey, status: "novelty", novelty: record.novelty }
    : { paperKey: record.paperKey, status: "no-novelty", reason: record.reason };
}

/**
 * Per-paper reference validation against the supplied trusted evidence: the
 * daily paper must exist in input.papers, every matched direction must exist in
 * matches.directionRepresentatives, and every representative of those
 * directions must exist in input.representatives. Valid papers keep their
 * match; invalid papers receive a typed per-paper no-novelty ("input-invalid")
 * and never join the plan.
 */
function partitionValidPapers(
  input: PersonalizedNoveltyInput,
  matches: PersonalNoveltyMatchInput,
): { valid: PersonalNoveltyPaperMatch[]; invalid: PersonalNoveltyPaperMatch[] } {
  const paperByKey = new Map(input.papers.map((paper) => [paper.paperKey, paper]));
  const representativeByKey = new Map(
    input.representatives.map((paper) => [paper.paperKey, paper]),
  );
  const representativesByDirection = new Map(
    matches.directionRepresentatives.map((entry) => [entry.directionId, entry.representativePaperKeys]),
  );
  const valid: PersonalNoveltyPaperMatch[] = [];
  const invalid: PersonalNoveltyPaperMatch[] = [];
  for (const match of matches.paperMatches) {
    if (!paperByKey.has(match.paperKey)) {
      invalid.push(match);
      continue;
    }
    let referencesValid = true;
    for (const directionId of match.directionIds) {
      const keys = representativesByDirection.get(directionId);
      if (!keys) {
        referencesValid = false;
        break;
      }
      for (const key of keys) {
        if (!representativeByKey.has(key)) {
          referencesValid = false;
          break;
        }
      }
      if (!referencesValid) break;
    }
    if (referencesValid) valid.push(match);
    else invalid.push(match);
  }
  return { valid, invalid };
}

/** Single-paper match slice rendering exactly the same request as the full plan. */
function singlePaperMatches(
  matches: PersonalNoveltyMatchInput,
  paperKey: string,
): PersonalNoveltyMatchInput {
  const match = matches.paperMatches.find((entry) => entry.paperKey === paperKey)!;
  return { paperMatches: [match], directionRepresentatives: matches.directionRepresentatives };
}

function isCheckpointGenerationIdentity(value: unknown): boolean {
  if (!isExactDataObject(value, ["provider", "endpointDigest", "model", "mode"])
    || typeof value.provider !== "string" || typeof value.model !== "string"
    || typeof value.endpointDigest !== "string"
    || !/^sha256:[0-9a-f]{64}$/.test(value.endpointDigest)
    || !isExactDataObject(value.mode, Object.keys(value.mode ?? {}))) return false;
  if (value.mode.kind === "temperature") {
    return isExactDataObject(value.mode, ["kind", "temperature"])
      && typeof value.mode.temperature === "number" && Number.isFinite(value.mode.temperature);
  }
  if (value.mode.kind === "anthropic-thinking") {
    return value.provider === "anthropic"
      && isExactDataObject(value.mode, ["kind", "budgetTokens"])
      && Number.isSafeInteger(value.mode.budgetTokens) && value.mode.budgetTokens > 0;
  }
  return value.mode.kind === "reasoning-thinking" && value.provider !== "anthropic"
    && isExactDataObject(value.mode, ["kind", "reasoningEffort"])
    && typeof value.mode.reasoningEffort === "string";
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

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function deepFreeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}
