import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { ArxivSettings, LlmSettings, Topic } from "../settings/types";
import type { PaperMeta } from "./arxiv-parser";
import type { MetricsObserver } from "../metrics/generation";
import {
  buildPaperFilterRequest,
  decodePaperFilterRecords,
  prepareDailyFilterCheckpoint,
  type FilterRecord,
  type PreparedDailyFilterCheckpoint,
} from "./paper-filter-contract";

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

export async function filterPapers(
  papers: PaperMeta[],
  deps: PaperFilterDeps,
): Promise<FilteredPaper[]> {
  const { llm, logger, arxivSettings } = deps;
  throwIfCancelled(deps.signal);
  if (papers.length === 0) return [];

  const topics: Topic[] = arxivSettings.topics ?? [];
  if (topics.length === 0) {
    logger.warn("paper-filter: no topics configured, skipping LLM call");
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
      parsed = JSON.parse(raw);
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
    out.push({ ...meta, category: item.category, isDetail: false });
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

function isErrorLike(value: unknown): value is Error & Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return Object.prototype.toString.call(value) === "[object Error]" &&
    typeof candidate.name === "string" &&
    typeof candidate.message === "string" &&
    typeof candidate.stack === "string";
}
