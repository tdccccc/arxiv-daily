import { isLlmContentBlockedError, type ChatMessage, type LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import {
  buildCheckpointGenerationIdentity,
  type CheckpointGenerationIdentity,
} from "../services/daily-summary-checkpoint-store";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { ArxivSettings, LlmSettings, Topic } from "../settings/types";
import type { PaperMeta } from "./arxiv-parser";
import type { MetricsObserver } from "../metrics/generation";
import type { PaperDiscoveryProvenance } from "./discovery-provenance-marker";
import {
  DAILY_FILTER_PROMPT_CONTRACT_VERSION,
  DAILY_FILTER_RESULT_CONTRACT_VERSION,
  buildPaperFilterRequest,
  classifiableTopics,
  decodePaperFilterRecords,
  prepareDailyFilterCheckpoint,
  type FilterDirectionRef,
  type FilterRecord,
  type PaperFilterRequest,
  type PreparedDailyFilterCheckpoint,
} from "./paper-filter-contract";
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
}

/**
 * The validated classification a filter run acted on — every candidate the
 * model returned, `skip` included — whether it was just generated or reused
 * from a checkpoint. `filterPapers` returns only the kept papers, so this is
 * how a caller learns what was decided about the rest.
 */
export interface FilterClassification {
  records: FilterRecord[];
  /** Directions the classifier was offered, from the frozen request. */
  directions: FilterDirectionRef[];
  generation: CheckpointGenerationIdentity;
  promptContractVersion: number;
  resultContractVersion: number;
}

export interface DailyFilterCheckpointPort {
  /** `batch` is 0 for an ordinary one-request day; later batches only exist on very large days. */
  lookupReusable(
    reportDate: string,
    prepared: PreparedDailyFilterCheckpoint,
    batch?: number,
  ): Promise<FilterRecord[] | null>;
  save(
    reportDate: string,
    prepared: PreparedDailyFilterCheckpoint,
    result: FilterRecord[],
    batch?: number,
  ): Promise<unknown>;
}

/**
 * Candidate text (ids, titles, abstracts) one filter request may carry. Sized
 * so an ordinary day, even a busy single category, stays one request while a
 * day that would not fit a ~128k-token context is split.
 */
export const FILTER_BATCH_CHAR_BUDGET = 400_000;

/** Split candidates, in order, into batches whose text stays within `budget` (never empty, never reordered). */
export function splitPapersForFilter(
  papers: readonly PaperMeta[],
  budget = FILTER_BATCH_CHAR_BUDGET,
): PaperMeta[][] {
  const batches: PaperMeta[][] = [];
  let current: PaperMeta[] = [];
  let size = 0;
  for (const paper of papers) {
    const cost = paper.id.length + paper.title.length + paper.abstract.length + 32;
    if (current.length > 0 && size + cost > budget) {
      batches.push(current);
      current = [];
      size = 0;
    }
    current.push(paper);
    size += cost;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

export interface PaperFilterDeps {
  llm: LlmClient;
  logger: Logger;
  arxivSettings: ArxivSettings;
  reportDate: string;
  llmSettings: LlmSettings;
  checkpointStore?: DailyFilterCheckpointPort;
  /** Called once the classification is validated; never affects the result. */
  onClassified?: (classification: FilterClassification) => void;
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
  const { logger, arxivSettings } = deps;
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

  const batches = splitPapersForFilter(papers);
  if (batches.length > 1) {
    logger.info(
      `paper-filter: ${papers.length} papers exceed one request, classifying in ${batches.length} batches`,
    );
  }
  const classified: BatchClassification[] = [];
  for (const [index, batch] of batches.entries()) {
    classified.push(await classifyBatch(batch, index, batches.length, deps));
  }
  const first = classified[0]!;
  const request = first.request;
  const prepared = first.prepared;
  const validatedRecords = classified.flatMap((entry) => entry.records);

  reportClassification(deps, request.options.temperature, prepared, request.identity.directions, validatedRecords);

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

interface BatchClassification {
  request: PaperFilterRequest;
  prepared: PreparedDailyFilterCheckpoint | undefined;
  records: FilterRecord[];
}

/** One validated classification for `batch`: reused from its checkpoint or generated and persisted. */
async function classifyBatch(
  batch: PaperMeta[],
  index: number,
  batchCount: number,
  deps: PaperFilterDeps,
): Promise<BatchClassification> {
  const { logger, arxivSettings } = deps;
  const label = batchCount > 1 ? ` batch=${index + 1}/${batchCount}` : "";
  let prepared: PreparedDailyFilterCheckpoint | undefined;
  try {
    prepared = deps.checkpointStore
      ? prepareDailyFilterCheckpoint({
          papers: batch,
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
  const request = prepared?.request ?? buildPaperFilterRequest(batch, arxivSettings);
  let reusable: FilterRecord[] | null | undefined;
  try {
    reusable = await deps.checkpointStore?.lookupReusable(
      deps.reportDate,
      prepared!,
      ...(index > 0 ? [index] : []),
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
    logger.info(
      `paper-filter: checkpoint hit date=${deps.reportDate} count=${reusable.length}${label}`,
    );
    return { request, prepared, records: reusable };
  }
  if (deps.checkpointStore) {
    logger.info(
      `paper-filter: checkpoint miss date=${deps.reportDate} count=${batch.length}${label}`,
    );
  }
  logger.info(
    `paper-filter: sending ${batch.length} papers to LLM for classification${label}`,
  );

  const refused: RefusedPapers = { papers: [] };
  const records = await classifyRefusingProvider(batch, request, deps, refused);
  if (refused.papers.length === batch.length) {
    // Nothing was classified: the refusal is not about individual papers, and
    // an empty answer would pass for a day without matches.
    logger.error("paper-filter: LLM call failed", refused.first);
    throw refused.first;
  }
  if (refused.papers.length > 0) {
    logger.warn(
      `paper-filter: the model provider refused ${refused.papers.length} paper(s), set aside unclassified: ${
        refused.papers.map(({ id }) => id).join(", ")
      }${label}`,
    );
  }
  try {
    await deps.checkpointStore?.save(
      deps.reportDate,
      prepared!,
      records,
      ...(index > 0 ? [index] : []),
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
      `paper-filter: checkpoint persisted date=${deps.reportDate} count=${records.length}${label}`,
    );
  }
  return { request, prepared, records };
}

/** Single papers the provider's content-safety filter refused while a batch was being narrowed down. */
interface RefusedPapers {
  papers: PaperMeta[];
  /** The refusal of the whole batch; rethrown when no usable answer comes out of the search. */
  first?: Error;
}

/**
 * Most single papers one batch may have refused. More than that means the
 * refusal is not about those papers (a flagged topic text, say), so the search
 * stops and the original error stands.
 */
const FILTER_MAX_REFUSED_PAPERS = 8;

/**
 * How many times one filter request is put to the model before a malformed
 * answer fails the run: the first answer, then one re-ask that names the rule
 * it broke. A re-ask repeats the whole request, so the count stays small.
 */
export const FILTER_ANSWER_ATTEMPTS = 2;

/**
 * The model's validated records for `papers`. When the provider's content-safety
 * filter refuses the request, the papers are halved until the refused ones stand
 * alone; those go into `refused` and are left out of the result like papers the
 * model left out, and the rest are classified as usual. An answer that fails
 * validation is asked for once more with the broken rule spelled out.
 */
async function classifyRefusingProvider(
  papers: PaperMeta[],
  request: PaperFilterRequest,
  deps: PaperFilterDeps,
  refused: RefusedPapers,
): Promise<FilterRecord[]> {
  let messages = request.messages;
  for (let attempt = 1; ; attempt += 1) {
    let raw: string;
    try {
      raw = await deps.llm.call(messages, {
        ...request.options,
        signal: deps.signal,
        onMetrics: deps.onMetrics,
      });
    } catch (e) {
      if (isCancellationError(e)) throw e;
      if (!isLlmContentBlockedError(e)) {
        deps.logger.error("paper-filter: LLM call failed", e);
        throw e;
      }
      refused.first ??= e;
      if (papers.length === 1) {
        refused.papers.push(papers[0]!);
        if (refused.papers.length > FILTER_MAX_REFUSED_PAPERS) {
          deps.logger.error("paper-filter: LLM call failed", refused.first);
          throw refused.first;
        }
        return [];
      }
      deps.logger.warn(
        `paper-filter: the model provider refused a request of ${papers.length} papers; narrowing down the cause`,
      );
      const middle = Math.ceil(papers.length / 2);
      const records: FilterRecord[] = [];
      for (const half of [papers.slice(0, middle), papers.slice(middle)]) {
        records.push(...await classifyRefusingProvider(
          half, buildPaperFilterRequest(half, deps.arxivSettings), deps, refused,
        ));
      }
      return records;
    }
    throwIfCancelled(deps.signal);
    try {
      return decodePaperFilterResponse(raw, request);
    } catch (error) {
      if (!isPaperFilterResponseValidationError(error) || attempt >= FILTER_ANSWER_ATTEMPTS) {
        throw error;
      }
      deps.logger.warn(
        `paper-filter: answer ${attempt} failed validation (${error.reasonCode}); asking once more with the rule it broke`,
      );
      messages = withCorrectionGuidance(request.messages, error);
    }
  }
}

/**
 * The request again, with what was wrong with the last answer appended to its
 * user turn (one turn, so providers that merge or reject consecutive same-role
 * turns see the same request shape). Only this module's own words go in:
 * nothing of the model's answer is echoed back.
 */
function withCorrectionGuidance(
  messages: readonly ChatMessage[],
  error: PaperFilterResponseValidationError,
): ChatMessage[] {
  const guidance = filterCorrectionGuidance(error);
  const last = messages[messages.length - 1];
  if (!last || last.role !== "user") return [...messages, { role: "user", content: guidance }];
  return [...messages.slice(0, -1), { role: "user", content: `${last.content}\n\n${guidance}` }];
}

function filterCorrectionGuidance(error: PaperFilterResponseValidationError): string {
  const reason = error.reasonCode === "invalid-json"
    ? "回复不是严格 JSON"
    : `回复违反了输出格式（${error.message.replace(/^response violates the filter contract: /, "")}）`;
  return [
    `上一次回复未通过校验：${reason}。`,
    "请重新输出，并且只输出一个 JSON 对象：不要用 Markdown 代码块包裹，不要附加任何其他文本；",
    "根对象只有 papers 数组；每条记录恰好包含 id、category、directions、relevanceScore；",
    "id 必须来自输入论文且不重复；relevanceScore 是 0 到 100 的数值；",
    "不相关的论文 category 填 \"skip\"、directions 为空数组、relevanceScore 为 0。",
  ].join("\n");
}

function reportClassification(
  deps: PaperFilterDeps,
  temperature: number,
  prepared: PreparedDailyFilterCheckpoint | undefined,
  directions: FilterDirectionRef[],
  records: FilterRecord[],
): void {
  if (!deps.onClassified) return;
  try {
    deps.onClassified({
      records: records.map((record) => ({ ...record, directions: [...record.directions] })),
      directions: directions.map((direction) => ({ ...direction })),
      generation: prepared?.fingerprintInput.generation
        ?? buildCheckpointGenerationIdentity(deps.llmSettings, temperature),
      promptContractVersion: prepared?.fingerprintInput.promptContractVersion
        ?? DAILY_FILTER_PROMPT_CONTRACT_VERSION,
      resultContractVersion: prepared?.fingerprintInput.resultContractVersion
        ?? DAILY_FILTER_RESULT_CONTRACT_VERSION,
    });
  } catch (error) {
    deps.logger.warn("paper-filter: classification observer failed", error);
  }
}

/**
 * Validate a raw model answer against the request it answers: strict JSON
 * (one surrounding code fence is tolerated), then the filter contract. Papers
 * the model left out are simply absent from the result; by the prompt's
 * contract that means not relevant. Throws `PaperFilterResponseValidationError`.
 */
export function decodePaperFilterResponse(raw: string, request: PaperFilterRequest): FilterRecord[] {
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
  return records.value;
}

/**
 * Some models wrap the whole JSON answer in one ```json fence. Unwrap exactly
 * that shape; anything else (prose around it, several fences) is returned
 * unchanged so strict parsing still rejects it.
 */
export function unwrapSingleCodeFence(raw: string): string {
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
