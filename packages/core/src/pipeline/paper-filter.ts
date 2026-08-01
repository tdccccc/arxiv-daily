import type { ChatMessage, LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { ArxivSettings, Topic } from "../settings/types";
import { formatArxivCategories } from "../settings/categories";
import type { PaperMeta } from "./arxiv-parser";
import { renderPrompt } from "../prompts/render";
import filterSystemTemplate from "../prompts/paper-filter.system.md";
import injectionGuardZh from "../prompts/injection-guard.md";
import { escapePaperDataFence } from "./prompt-safety";
import type { MetricsObserver } from "../metrics/generation";
import type { DailyFilterCheckpointCompatibilityInput } from "../services/daily-filter-checkpoint-store";
import type { LlmSettings } from "../settings/types";

export interface FilteredPaper extends PaperMeta {
  category: string;
  isDetail: boolean;
}

export interface DailyFilterCheckpointPort {
  lookupReusable(
    reportDate: string,
    input: DailyFilterCheckpointCompatibilityInput,
  ): Promise<FilterRecord[] | null>;
  save(
    reportDate: string,
    input: DailyFilterCheckpointCompatibilityInput,
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

export interface PaperFilterRequest {
  messages: ChatMessage[];
  options: { temperature: 0 };
  identity: {
    knownIds: string[];
    validTags: string[];
  };
}

export interface FilterRecord {
  id: string;
  category: string;
}

export type FilterRecordDecodeResult =
  | { ok: true; value: FilterRecord[] }
  | { ok: false; reason: string };

/** Construct the exact request consumed by the live paper-filter LLM call. */
export function buildPaperFilterRequest(
  papers: PaperMeta[],
  arxivSettings: ArxivSettings,
): PaperFilterRequest {
  const topics: Topic[] = arxivSettings.topics ?? [];
  const topicLines = topics.map((t) => `- ${t.tag}: ${t.description}`).join("\n");
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
    },
  };
}

/** Strictly decode validated model decisions while preserving record order and omissions. */
export function decodePaperFilterRecords(
  value: unknown,
  knownIds: ReadonlySet<string>,
  validTags: ReadonlySet<string>,
): FilterRecordDecodeResult {
  if (!isPlainObject(value) || !hasExactKeys(value, ["papers"]) || !Array.isArray(value.papers)) {
    return { ok: false, reason: "root must be exactly {papers:[...]}" };
  }

  const seen = new Set<string>();
  const records: FilterRecord[] = [];
  for (const record of value.papers) {
    if (!isPlainObject(record) || !hasExactKeys(record, ["id", "category"])) {
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
    seen.add(record.id);
    records.push({ id: record.id, category: record.category });
  }
  return { ok: true, value: records };
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

  const request = buildPaperFilterRequest(papers, arxivSettings);
  const checkpointInput: DailyFilterCheckpointCompatibilityInput = {
    papers,
    arxivSettings,
    llm: deps.llmSettings,
  };
  let validatedRecords: FilterRecord[];
  let reusable: FilterRecord[] | null | undefined;
  try {
    reusable = await deps.checkpointStore?.lookupReusable(
      deps.reportDate,
      checkpointInput,
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
    } catch (e) {
      logger.error("paper-filter: response is not strict JSON", e);
      return [];
    }

    const records = decodePaperFilterRecords(
      parsed,
      new Set(request.identity.knownIds),
      new Set(request.identity.validTags),
    );
    if (!records.ok) {
      logger.warn(`paper-filter: invalid LLM response (${records.reason}); keeping no papers`);
      return [];
    }
    validatedRecords = records.value;
    try {
      await deps.checkpointStore?.save(
        deps.reportDate,
        checkpointInput,
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

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index])
  );
}
