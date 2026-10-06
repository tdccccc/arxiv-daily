import type { ChatMessage } from "../llm/client";
import type { CheckpointGenerationIdentity } from "../services/daily-summary-checkpoint-store";
import { buildCheckpointGenerationIdentity } from "../services/daily-summary-checkpoint-store";
import { formatArxivCategories } from "../settings/categories";
import type { ArxivSettings, LlmSettings, Topic } from "../settings/types";
import filterSystemTemplate from "../prompts/paper-filter.system.md";
import injectionGuardZh from "../prompts/injection-guard.md";
import { renderPrompt } from "../prompts/render";
import type { PaperMeta } from "./arxiv-parser";
import { escapePaperDataFence } from "./prompt-safety";

export const DAILY_FILTER_FINGERPRINT_VERSION = 1 as const;
/**
 * Version 4 resolves overlap between topics by the specificity of the matched
 * direction, then by configured order. Earlier classifications may have a
 * different owner (and therefore a different detail-note policy).
 */
export const DAILY_FILTER_PROMPT_CONTRACT_VERSION = 4 as const;
export const DAILY_FILTER_RESULT_CONTRACT_VERSION = 3 as const;

/**
 * One direction as the classifier sees it. `ref` is the short handle written
 * into the prompt and echoed back by the model — a direction's stored `id` is
 * a UUID, which is both long to repeat and easy to mistranscribe. `tag` makes
 * the reference self-checking: a paper filed under one topic cannot name a
 * direction belonging to another (ADR 0012 §1, one topic holds its own list).
 */
export interface FilterDirectionRef {
  ref: string;
  tag: string;
  id: string;
  text: string;
}

export interface PaperFilterRequest {
  messages: ChatMessage[];
  options: { temperature: 0 };
  identity: {
    knownIds: string[];
    validTags: string[];
    directions: FilterDirectionRef[];
  };
}

export interface FilterRecord {
  id: string;
  category: string;
  /** Direction refs of the chosen topic; empty exactly when `category` is `skip`. */
  directions: string[];
  /** Relevance to the matched directions, on one 0–100 scale across topics. */
  relevanceScore: number;
}

export type FilterRecordDecodeResult =
  | { ok: true; value: FilterRecord[] }
  | { ok: false; reason: string };

export interface DailyFilterCheckpointCompatibilityInput {
  papers: PaperMeta[];
  arxivSettings: ArxivSettings;
  llm: Pick<
    LlmSettings,
    "provider" | "baseUrl" | "model" | "thinkingMode" | "reasoningEffort"
  > & { apiKey?: string };
  promptContractVersion?: number;
  resultContractVersion?: number;
}

export interface DailyFilterCheckpointFingerprintInput {
  fingerprintVersion: typeof DAILY_FILTER_FINGERPRINT_VERSION;
  request: {
    messages: ChatMessage[];
    identity: {
      knownIds: string[];
      validTags: string[];
      directions: FilterDirectionRef[];
    };
  };
  generation: CheckpointGenerationIdentity;
  promptContractVersion: number;
  resultContractVersion: number;
}

export interface PreparedDailyFilterCheckpoint {
  readonly request: PaperFilterRequest;
  readonly fingerprintInput: DailyFilterCheckpointFingerprintInput;
}

const preparedSnapshots = new WeakSet<object>();

/**
 * Capture the one exact immutable request and compatibility identity for a filter run.
 * Store ports accept only snapshots created here, so persisted compatibility cannot
 * drift from the request consumed by the live LLM call.
 */
export function prepareDailyFilterCheckpoint(
  input: DailyFilterCheckpointCompatibilityInput,
): PreparedDailyFilterCheckpoint {
  const request = buildPaperFilterRequest(input.papers, input.arxivSettings);
  const snapshot: PreparedDailyFilterCheckpoint = {
    request: clone(request),
    fingerprintInput: {
      fingerprintVersion: DAILY_FILTER_FINGERPRINT_VERSION,
      request: {
        messages: clone(request.messages),
        identity: clone(request.identity),
      },
      generation: buildCheckpointGenerationIdentity(input.llm, request.options.temperature),
      promptContractVersion:
        input.promptContractVersion ?? DAILY_FILTER_PROMPT_CONTRACT_VERSION,
      resultContractVersion:
        input.resultContractVersion ?? DAILY_FILTER_RESULT_CONTRACT_VERSION,
    },
  };
  deepFreeze(snapshot);
  preparedSnapshots.add(snapshot);
  return snapshot;
}

export function isPreparedDailyFilterCheckpoint(
  value: unknown,
): value is PreparedDailyFilterCheckpoint {
  return typeof value === "object" && value !== null && preparedSnapshots.has(value);
}

/** Construct the exact request consumed by the live paper-filter LLM call. */
export function buildPaperFilterRequest(
  papers: PaperMeta[],
  arxivSettings: ArxivSettings,
): PaperFilterRequest {
  // A topic with no directions has nothing to judge against, so it cannot
  // select a paper. Offering it as a tag anyway invites the model to file a
  // paper under it and then fail the strict decode for naming no direction —
  // a whole run lost to a configuration problem `validateFilterConfig` already
  // reports. `classifiableTopics` is exported so callers can say which topics
  // they left out rather than dropping them silently.
  const topics: Topic[] = classifiableTopics(arxivSettings);
  const directions = buildFilterDirectionRefs(topics);
  const byTag = new Map<string, FilterDirectionRef[]>();
  for (const direction of directions) {
    byTag.set(direction.tag, [...(byTag.get(direction.tag) ?? []), direction]);
  }
  // The topic's `description` is the rollback shadow of its first direction
  // (ADR 0012) and no longer takes part in classification: the judgement is
  // made against the direction lines, one per thread running inside the topic.
  const topicLines = topics
    .map((t) => [
      `- ${t.tag}:`,
      ...(byTag.get(t.tag) ?? []).map((d) => `  - ${d.ref}: ${singleLine(d.text)}`),
    ].join("\n"))
    .join("\n");
  const tagOptions = topics.map((t) => t.tag).join("|") + "|skip";
  const papersText = papers
    .map(
      (p) =>
        `---\nID: ${escapePaperDataFence(p.id)}\n` +
        `Title: ${escapePaperDataFence(p.title)}\n` +
        `Abstract: ${escapePaperDataFence(p.abstract)}\n`,
    )
    .join("");
  return {
    messages: [
      {
        role: "system",
        content: renderPrompt(filterSystemTemplate, {
          topicLines,
          tagOptions,
          injectionGuard: injectionGuardZh,
        }),
      },
      {
        role: "user",
        content: `以下是今日 arXiv ${formatArxivCategories(arxivSettings)} 的所有新论文：\n\n<paper_data>\n${papersText}</paper_data>`,
      },
    ],
    options: { temperature: 0 },
    identity: {
      knownIds: papers.map((paper) => paper.id),
      validTags: topics.map((topic) => topic.tag),
      directions,
    },
  };
}

/** Topics that hold at least one direction, in configured order. */
export function classifiableTopics(arxivSettings: ArxivSettings): Topic[] {
  return (arxivSettings.topics ?? []).filter((topic) => topic.directions.length > 0);
}

/**
 * Number directions within their own topic, so a reference reads as
 * `<tag>#<n>` and carries the topic it belongs to. Splitting on the final `#`
 * recovers the tag even when the tag itself contains one.
 */
export function buildFilterDirectionRefs(topics: readonly Topic[]): FilterDirectionRef[] {
  return topics.flatMap((topic) =>
    topic.directions.map((direction, index) => ({
      ref: `${topic.tag}#${index + 1}`,
      tag: topic.tag,
      id: direction.id,
      text: direction.text,
    })),
  );
}

/** Keep one direction on one prompt line; identity keeps the text as written. */
function singleLine(value: string): string {
  return value.replace(/\s+/gu, " ").trim();
}

/** Strictly decode validated model decisions while preserving record order and omissions. */
export function decodePaperFilterRecords(
  value: unknown,
  knownIds: ReadonlySet<string>,
  validTags: ReadonlySet<string>,
  directions: readonly FilterDirectionRef[],
): FilterRecordDecodeResult {
  if (!isPlainObject(value) || !hasExactKeys(value, ["papers"]) || !Array.isArray(value.papers)) {
    return { ok: false, reason: "root must be exactly {papers:[...]}" };
  }
  const tagByRef = new Map(directions.map((direction) => [direction.ref, direction.tag]));

  const seen = new Set<string>();
  const records: FilterRecord[] = [];
  for (const record of value.papers) {
    if (!isPlainObject(record) || !hasExactKeys(record, ["id", "category", "directions", "relevanceScore"])) {
      return { ok: false, reason: "paper record has an invalid shape" };
    }
    if (typeof record.id !== "string" || !knownIds.has(record.id)) {
      return { ok: false, reason: "paper record has an unknown id" };
    }
    if (seen.has(record.id)) {
      return { ok: false, reason: "paper record has a duplicate id" };
    }
    if (
      typeof record.category !== "string" ||
      (record.category !== "skip" && !validTags.has(record.category))
    ) {
      return { ok: false, reason: `paper ${record.id} has an invalid category` };
    }
    if (!Array.isArray(record.directions)) {
      return { ok: false, reason: "paper record has an invalid shape" };
    }
    if (typeof record.relevanceScore !== "number" || !Number.isFinite(record.relevanceScore)
      || record.relevanceScore < 0 || record.relevanceScore > 100) {
      return { ok: false, reason: `paper ${record.id} has an invalid relevance score` };
    }
    if (record.category === "skip" && record.relevanceScore !== 0) {
      return { ok: false, reason: `paper ${record.id} has relevance while skipped` };
    }
    // The topic is chosen because a direction matched, so a skipped paper has
    // no direction to name and a kept one has no reason to be kept without
    // naming at least one. Both are checked before the references themselves,
    // so the reason reported is the incoherence rather than a stray reference.
    if (record.category === "skip" && record.directions.length > 0) {
      return { ok: false, reason: `paper ${record.id} names a direction while skipped` };
    }
    if (record.category !== "skip" && record.directions.length === 0) {
      return { ok: false, reason: `paper ${record.id} names no direction for its topic` };
    }
    const chosen = new Set<string>();
    const directions: string[] = [];
    for (const ref of record.directions) {
      // A reference is valid only inside the topic the paper was filed under,
      // so a mismatch is a contract violation rather than a silent drop.
      if (typeof ref !== "string" || tagByRef.get(ref) !== record.category) {
        return { ok: false, reason: `paper ${record.id} has an invalid direction` };
      }
      if (chosen.has(ref)) {
        return { ok: false, reason: `paper ${record.id} has a duplicate direction` };
      }
      chosen.add(ref);
      directions.push(ref);
    }
    seen.add(record.id);
    records.push({ id: record.id, category: record.category, directions, relevanceScore: record.relevanceScore });
  }
  return { ok: true, value: records };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index]);
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
