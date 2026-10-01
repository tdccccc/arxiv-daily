import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { ArxivSettings, LlmSettings, Topic } from "../settings/types";
import type { PaperMeta } from "./arxiv-parser";
import type { MetricsObserver } from "../metrics/generation";
import type { PaperDiscoveryProvenance } from "./discovery-provenance-marker";
import {
  buildPaperFilterRequest,
  classifiableTopics,
  decodePaperFilterRecords,
  prepareDailyFilterCheckpoint,
  type FilterRecord,
  type PreparedDailyFilterCheckpoint,
} from "./paper-filter-contract";
import type { PersonalNoveltyWithBasis } from "./personalized-novelty";
import type { TopicDirectionHit } from "./topic-direction-hits";

export {
  buildPaperFilterRequest,
  decodePaperFilterRecords,
  prepareDailyFilterCheckpoint,
  type DailyFilterCheckpointCompatibilityInput,
  type FilterRecord,
  type FilterRecordDecodeResult,
  type PaperFilterRequest,
  type PreparedDailyFilterCheckpoint,
} from "./paper-filter-contract";

export interface FilteredPaper extends PaperMeta {
  category: string;
  isDetail: boolean;
  /** Validated relevance score used to select papers within the daily limit. */
  relevanceScore: number;
  /**
   * Directions of `category`'s topic that selected this paper, in the topic's
   * own order. Every filtered paper now reaches the report by this one route.
   */
  topicDirections?: TopicDirectionHit[];
  /**
   * Read-only legacy field (ADR 0012 / ADR 0014). The library-profile
   * classifier that produced it retired with the profile document, so nothing
   * sets it any more; it stays on the type because reports and index entries
   * already on disk carry the provenance and are still read back.
   */
  discoveryProvenance?: PaperDiscoveryProvenance;
  /**
   * Read-only legacy field. Personal novelty lost its comparison basis when
   * the profile document retired (ADR 0012 §Consequences); choosing a new
   * basis is a decision deliberately left open, so the stage no longer runs
   * and nothing sets this.
   */
  personalNovelty?: PersonalNoveltyWithBasis;
}

export interface DailyFilterCheckpointPort {
  lookupReusable(
    reportDate: string,
    prepared: PreparedDailyFilterCheckpoint,
  ): Promise<FilterRecord[] | null>;
  save(
    reportDate: string,
    prepared: PreparedDailyFilterCheckpoint,
    result: FilterRecord[],
  ): Promise<unknown>;
}

export interface PaperFilterDeps {
  llm: LlmClient;
  logger: Logger;
  arxivSettings: ArxivSettings;
  reportDate: string;
  llmSettings: LlmSettings;
  checkpointStore?: DailyFilterCheckpointPort;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
}

export class PaperFilterCheckpointError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "PaperFilterCheckpointError";
  }
}

export function isPaperFilterCheckpointError(
  error: unknown,
): error is PaperFilterCheckpointError {
  return error instanceof PaperFilterCheckpointError;
}

export const PAPER_FILTER_RESPONSE_VALIDATION_ERROR_CODE =
  "ARXIV_DAILY_PAPER_FILTER_RESPONSE_VALIDATION" as const;

export type PaperFilterResponseValidationReasonCode =
  | "invalid-json"
  | "invalid-contract";

export class PaperFilterResponseValidationError extends Error {
  readonly name = "PaperFilterResponseValidationError";
  readonly code = PAPER_FILTER_RESPONSE_VALIDATION_ERROR_CODE;

  constructor(
    message: string,
    readonly reasonCode: PaperFilterResponseValidationReasonCode,
  ) {
    super(message);
  }
}

export function isPaperFilterResponseValidationError(
  error: unknown,
): error is PaperFilterResponseValidationError {
  if (error instanceof PaperFilterResponseValidationError) return true;
  if (!isErrorLike(error)) return false;
  const candidate = error as Record<string, unknown>;
  return candidate.name === "PaperFilterResponseValidationError" &&
    candidate.code === PAPER_FILTER_RESPONSE_VALIDATION_ERROR_CODE &&
    (candidate.reasonCode === "invalid-json" ||
      candidate.reasonCode === "invalid-contract") &&
    typeof candidate.message === "string";
}

/**
 * The one classifier. A second, profile-driven classifier used to run
 * alongside this one and union its results in; it retired with the profile
 * document (ADR 0012 / ADR 0014), and topic directions (ADR 0012 §1) now carry
 * what it was for. Papers reach the report through topics and nothing else.
 */
export async function filterPapers(
  papers: PaperMeta[],
  deps: PaperFilterDeps,
): Promise<FilteredPaper[]> {
  const { llm, logger, arxivSettings } = deps;
  throwIfCancelled(deps.signal);
  if (papers.length === 0) return [];

  const configured: Topic[] = arxivSettings.topics ?? [];
  if (configured.length === 0) {
    logger.warn("paper-filter: no topics configured, skipping LLM call");
    return [];
  }
  const topics = classifiableTopics(arxivSettings);
  const withoutDirections = configured.filter((topic) => !topics.includes(topic));
  if (withoutDirections.length > 0) {
    logger.warn(
      `paper-filter: skipping topics with no directions: ${
        withoutDirections.map((topic) => topic.tag).join(", ")
      }`,
    );
  }
  if (topics.length === 0) {
    logger.warn("paper-filter: no topic has a direction, skipping LLM call");
    return [];
  }

  let prepared: PreparedDailyFilterCheckpoint | undefined;
  try {
    prepared = deps.checkpointStore
      ? prepareDailyFilterCheckpoint({
          papers,
          arxivSettings,
          llm: deps.llmSettings,
        })
      : undefined;
  } catch (error) {
    throw new PaperFilterCheckpointError(
      `prepare failed for ${deps.reportDate}: ${(error as Error).message}`,
      error,
    );
  }
  const request = prepared?.request ?? buildPaperFilterRequest(papers, arxivSettings);
  let validatedRecords: FilterRecord[];
  let reusable: FilterRecord[] | null | undefined;
  try {
    reusable = await deps.checkpointStore?.lookupReusable(
      deps.reportDate,
      prepared!,
    );
  } catch (error) {
    if (isCancellationError(error)) throw error;
    throwIfCancelled(deps.signal);
    throw new PaperFilterCheckpointError(
      `lookup failed for ${deps.reportDate}: ${(error as Error).message}`,
      error,
    );
  }
  throwIfCancelled(deps.signal);
  if (reusable) {
    validatedRecords = reusable;
    logger.info(
      `paper-filter: checkpoint hit date=${deps.reportDate} count=${validatedRecords.length}`,
    );
  } else {
    if (deps.checkpointStore) {
      logger.info(
        `paper-filter: checkpoint miss date=${deps.reportDate} count=${papers.length}`,
      );
    }
    logger.info(
      `paper-filter: sending ${papers.length} papers to LLM for classification`,
    );

    let raw: string;
    try {
      raw = await llm.call(request.messages, {
        ...request.options,
        signal: deps.signal,
        onMetrics: deps.onMetrics,
      });
    } catch (e) {
      if (isCancellationError(e)) throw e;
      logger.error("paper-filter: LLM call failed", e);
      throw e;
    }
    throwIfCancelled(deps.signal);

    let parsed: unknown;
    try {
      parsed = JSON.parse(unwrapSingleCodeFence(raw));
    } catch {
      throw new PaperFilterResponseValidationError(
        "response is not strict JSON",
        "invalid-json",
      );
    }

    const records = decodePaperFilterRecords(
      parsed,
      new Set(request.identity.knownIds),
      new Set(request.identity.validTags),
      request.identity.directions,
    );
    if (!records.ok) {
      throw new PaperFilterResponseValidationError(
        `response violates the filter contract: ${records.reason}`,
        "invalid-contract",
      );
    }
    validatedRecords = records.value;
    try {
      await deps.checkpointStore?.save(
        deps.reportDate,
        prepared!,
        validatedRecords,
      );
    } catch (error) {
      if (isCancellationError(error)) throw error;
      throwIfCancelled(deps.signal);
      throw new PaperFilterCheckpointError(
        `save failed for ${deps.reportDate}: ${(error as Error).message}`,
        error,
      );
    }
    throwIfCancelled(deps.signal);
    if (deps.checkpointStore) {
      logger.info(
        `paper-filter: checkpoint persisted date=${deps.reportDate} count=${validatedRecords.length}`,
      );
    }
  }

  const idMap = new Map(papers.map((p) => [p.id, p] as const));
  const out: FilteredPaper[] = [];
  for (const item of validatedRecords) {
    if (item.category === "skip") continue;
    const meta = idMap.get(item.id)!;
    // Resolved from the frozen request, never from live settings: those may
    // have been edited while the call was in flight. Walking the request's own
    // direction list keeps the topic's order rather than the model's.
    const chosen = new Set(item.directions);
    const topicDirections = request.identity.directions
      .filter((direction) => chosen.has(direction.ref))
      .map(({ tag, id, text }) => ({ tag, id, text }));
    out.push({ ...meta, category: item.category, isDetail: false, relevanceScore: item.relevanceScore, topicDirections });
  }
  logger.info(`paper-filter: kept ${out.length}/${papers.length} papers`);

  // Log per-tag breakdown
  const tagCounts = new Map<string, number>();
  const detailCounts = new Map<string, number>();
  for (const p of out) {
    tagCounts.set(p.category, (tagCounts.get(p.category) ?? 0) + 1);
    if (p.isDetail) detailCounts.set(p.category, (detailCounts.get(p.category) ?? 0) + 1);
  }
  const breakdown = [...tagCounts.entries()]
    .map(([tag, count]) => {
      const details = detailCounts.get(tag) ?? 0;
      return details > 0 ? `${tag}=${count}(${details} detail)` : `${tag}=${count}`;
    })
    .join(", ");
  const skipped = papers.length - out.length;
  logger.info(`paper-filter: ${breakdown}${skipped > 0 ? `, skipped=${skipped}` : ""}`);
  return out;
}

/**
 * Some models wrap the whole JSON answer in one ```json fence. Unwrap exactly
 * that shape; anything else (prose around it, several fences) is returned
 * unchanged so strict parsing still rejects it.
 */
function unwrapSingleCodeFence(raw: string): string {
  const inner = /^\s*```[A-Za-z]*[ \t]*\r?\n([\s\S]*?)\r?\n```\s*$/.exec(raw)?.[1];
  if (inner === undefined || inner.includes("```")) return raw;
  return inner;
}

function isErrorLike(value: unknown): value is Error & Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return Object.prototype.toString.call(value) === "[object Error]" &&
    typeof candidate.name === "string" &&
    typeof candidate.message === "string" &&
    typeof candidate.stack === "string";
}
