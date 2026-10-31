import {
  isLlmContentBlockedError,
  isLlmTransientExhaustedError,
  type LlmClient,
} from "../llm/client";
import type { MetricsObserver } from "../metrics/generation";
import detailSystemTemplateEn from "../prompts/paper-detail.en.system.md";
import detailSystemTemplate from "../prompts/paper-detail.system.md";
import injectionGuardEn from "../prompts/injection-guard.en.md";
import injectionGuardZh from "../prompts/injection-guard.md";
import { renderPrompt } from "../prompts/render";
import { throwIfCancelled } from "../services/cancellation";
import type { PaperIndexEntry, PaperStatus } from "../services/paper-index";
import type { Logger } from "../services/logger";
import type { DailySummaryCheckpointCompatibilityInput } from "../services/daily-summary-checkpoint-store";
import { normalizeSummaryLanguage } from "../settings/summary-language";
import type {
  AdvancedSettings,
  ArxivSettings,
  LinkStyle,
  LlmSettings,
  SummaryLanguage,
} from "../settings/types";
import {
  derivePaperSourceSections,
  describeValidationFailureForLog,
  summarizeDailyPaperWithValidationRetry,
} from "./daily-paper-summary";
import {
  assembleDailySummary,
  assembleEmergencyDailySummary,
  isDailySummaryAssemblyRuntimeError,
  type DailyPaperResult,
  type DailyPaperSlot,
  type DailyTitleOnlyCandidates,
  type DailySummaryAssemblyInput,
  type DailySummaryAssemblyPaper,
  preflightDailySummaryAssembly,
  preflightDailySummaryPapers,
  validateDailyTitleOnlyCandidates,
} from "./daily-summary-assembler";
import {
  isDailySummaryRescueExhaustedError,
  rescueDailySummary,
} from "./daily-summary-rescue";
import type { FilteredPaper } from "./paper-filter";
import { escapePaperDataFence } from "./prompt-safety";

export interface DailyPaperWithContent extends Omit<FilteredPaper, "relevanceScore"> {
  /** Manual paper fetches bypass classification and have no relevance score. */
  relevanceScore?: number;
  abstractConclusion: string;
  fullSections: string | null;
  published?: string;
  updated?: string;
  inboxStatus?: PaperStatus;
  seenBefore?: boolean;
  paperPath?: string | null;
  detailLink?: string;
  indexEntry?: PaperIndexEntry;
}

export interface DailySummaryCheckpointPort {
  lookupReusable(
    reportDate: string,
    input: DailySummaryCheckpointCompatibilityInput,
  ): Promise<DailyPaperResult | null>;
  upsert(
    reportDate: string,
    input: DailySummaryCheckpointCompatibilityInput,
    result: DailyPaperResult,
  ): Promise<unknown>;
}

export interface SummarizerDeps {
  llm: LlmClient;
  llmSettings: LlmSettings;
  checkpointStore?: DailySummaryCheckpointPort;
  logger: Logger;
  arxivSettings: ArxivSettings;
  advanced: AdvancedSettings;
  linkStyle?: LinkStyle;
  summaryLanguage?: SummaryLanguage;
  /** Counts omitted by the report limit, separate from per-paper summaries. */
  omittedByTopic?: Readonly<Record<string, number>>;
  /** Candidates filtered on title only (no abstract); shown in the report header when count > 0. */
  titleOnlyCandidates?: DailyTitleOnlyCandidates;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
  onDailyPaperProgress?: (completed: number, total: number) => void;
  /**
   * Throw {@link DailySummaryIncompleteError} on the first paper whose model
   * call ended as transport-exhausted instead of carrying a fallback into the
   * report. Papers finished so far stay in the checkpoint store.
   */
  failOnTransportExhausted?: boolean;
  /** Papers summarized at once; see {@link DAILY_GENERATION_CONCURRENCY}. */
  concurrency?: number;
  /** Lower-level renderer injected behind the production assembler preflight boundary. */
  dailyRenderer?: (input: DailySummaryAssemblyInput) => string;
  rescueDaily?: typeof rescueDailySummary;
  assembleEmergencyDaily?: (input: DailySummaryAssemblyInput) => string;
}

/** Per-paper model calls (daily summaries, paper notes) in flight at once by default. */
export const DAILY_GENERATION_CONCURRENCY = 3;
/** The most a host may configure; beyond this a provider's rate limit turns into fallbacks. */
export const MAX_GENERATION_CONCURRENCY = 8;

/** The bound to run with for a configured value; unset or invalid means the default. */
export function normalizeGenerationConcurrency(value: number | undefined): number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) return DAILY_GENERATION_CONCURRENCY;
  return Math.min(value as number, MAX_GENERATION_CONCURRENCY);
}

/** The model endpoint was unreachable for a paper; the run should be retried, not committed. */
export class DailySummaryIncompleteError extends Error {
  readonly paperId: string;

  constructor(paperId: string) {
    super(`daily summary incomplete: the model endpoint was unreachable for ${paperId}; finished papers are kept and the run will be retried`);
    this.name = "DailySummaryIncompleteError";
    this.paperId = paperId;
  }
}

export interface SummarizeDailyResult {
  markdown: string;
  slots: DailyPaperSlot[];
}

export async function summarizeDaily(
  papers: DailyPaperWithContent[],
  dateStr: string,
  deps: SummarizerDeps,
): Promise<SummarizeDailyResult> {
  throwIfCancelled(deps.signal);
  const assemblyPapers: DailySummaryAssemblyPaper[] = papers.map((paper) => ({
    id: paper.id,
    title: paper.title,
    authors: paper.authors,
    category: paper.category,
    sourceSections: derivePaperSourceSections(paper),
    isDetail: paper.isDetail,
    paperPath: paper.paperPath,
    detailLink: paper.detailLink,
    discoveryProvenance: paper.discoveryProvenance,
    topicDirections: paper.topicDirections,
  }));
  preflightDailySummaryPapers(assemblyPapers, deps.arxivSettings);
  // Fail before spending model calls on a report that cannot be assembled.
  validateDailyTitleOnlyCandidates(deps);

  const limit = normalizeGenerationConcurrency(deps.concurrency);
  const results = new Array<DailyPaperResult>(papers.length);
  let completed = 0;
  let fallbackCount = 0;
  const summarizeOne = async (index: number, signal: AbortSignal | undefined): Promise<void> => {
    throwIfCancelled(signal);
    const paper = papers[index]!;
    const checkpointInput: DailySummaryCheckpointCompatibilityInput = {
      paper,
      summaryLanguage: deps.summaryLanguage,
      llm: deps.llmSettings,
    };
    let reusable: DailyPaperResult | null | undefined;
    try {
      reusable = await deps.checkpointStore?.lookupReusable(
        dateStr,
        checkpointInput,
      );
    } catch (error) {
      throwIfCancelled(signal);
      throw error;
    }
    throwIfCancelled(signal);

    let result: DailyPaperResult;
    let recovered = false;
    let lastValidationFailure: string | undefined;
    if (reusable) {
      result = reusable;
      recovered = true;
      deps.logger.info(
        `summarizeDaily: checkpoint hit date=${dateStr} paper=${paper.id}`,
      );
    } else {
      if (deps.checkpointStore) {
        deps.logger.info(
          `summarizeDaily: checkpoint miss date=${dateStr} paper=${paper.id}`,
        );
      }
      result = await summarizeDailyPaperWithValidationRetry(paper, {
        llm: deps.llm,
        summaryLanguage: deps.summaryLanguage,
        signal,
        onMetrics: deps.onMetrics,
        onValidationFailure: (error) => {
          lastValidationFailure = describeValidationFailureForLog(error);
        },
      });
      throwIfCancelled(signal);
      try {
        await deps.checkpointStore?.upsert(dateStr, checkpointInput, result);
      } catch (error) {
        throwIfCancelled(signal);
        throw error;
      }
      throwIfCancelled(signal);
      if (deps.checkpointStore) {
        deps.logger.info(
          `summarizeDaily: checkpoint persisted date=${dateStr} paper=${paper.id}`,
        );
      }
    }
    if (
      result.kind === "fallback" &&
      result.reasonCode === "transport-exhausted" &&
      deps.failOnTransportExhausted
    ) {
      deps.logger.warn(
        `summarizeDaily: model endpoint unreachable for ${paper.id}; keeping ${completed} finished papers for a retry`,
      );
      throw new DailySummaryIncompleteError(paper.id);
    }
    if (result.kind === "fallback") {
      fallbackCount += 1;
      deps.logger.warn(
        `summarizeDaily: fallback for ${paper.id} reason=${result.reasonCode} attempts=${result.attempts}` +
          (result.reasonCode === "validation-exhausted" && lastValidationFailure
            ? ` lastFailure=${lastValidationFailure}`
            : "") +
          (recovered ? " recovered=true" : ""),
      );
    }
    results[index] = result;
    completed += 1;
    deps.onDailyPaperProgress?.(completed, papers.length);
  };
  // A few papers at a time; the report keeps their input order regardless of
  // which finishes first.
  await runWithConcurrency(papers.length, limit, summarizeOne, deps.signal);
  throwIfCancelled(deps.signal);
  const slots: DailyPaperSlot[] = papers.map((_, index) => ({
    paper: assemblyPapers[index]!,
    result: results[index]!,
  }));

  throwIfCancelled(deps.signal);
  const assemblyInput: DailySummaryAssemblyInput = {
    slots,
    dateStr,
    arxivSettings: deps.arxivSettings,
    summaryLanguage: deps.summaryLanguage,
    ...(deps.omittedByTopic ? { omittedByTopic: deps.omittedByTopic } : {}),
    ...(deps.titleOnlyCandidates ? { titleOnlyCandidates: deps.titleOnlyCandidates } : {}),
  };
  preflightDailySummaryAssembly(assemblyInput);

  let markdown: string;
  try {
    markdown = assembleDailySummary(
      assemblyInput,
      deps.dailyRenderer ? { renderer: deps.dailyRenderer } : {},
    );
  } catch (error) {
    if (!isDailySummaryAssemblyRuntimeError(error)) throw error;
    deps.logger.warn(
      `summarizeDaily: deterministic assembly runtime failure slots=${slots.length}`,
    );
    try {
      markdown = await (deps.rescueDaily ?? rescueDailySummary)(assemblyInput, {
        llm: deps.llm,
        logger: deps.logger,
        signal: deps.signal,
        onMetrics: deps.onMetrics,
      });
    } catch (rescueError) {
      // The rescue request carries the papers' abstracts, so a provider that
      // refuses one of them refuses the rescue too: fall back like the others.
      const blocked = isLlmContentBlockedError(rescueError);
      if (
        !isDailySummaryRescueExhaustedError(rescueError) &&
        !isLlmTransientExhaustedError(rescueError) &&
        !blocked
      ) {
        throw rescueError;
      }
      const cause = isDailySummaryRescueExhaustedError(rescueError)
        ? "validation-exhausted"
        : blocked ? "content-blocked" : "transport-exhausted";
      deps.logger.warn(
        `summarizeDaily: degraded emergency report cause=${cause} slots=${slots.length} fallback=${fallbackCount}`,
      );
      markdown = (deps.assembleEmergencyDaily ?? assembleEmergencyDailySummary)(
        assemblyInput,
      );
    }
  }
  deps.logger.info(
    `summarizeDaily: assembled ${papers.length} sequential paper summaries` +
      (fallbackCount > 0 ? ` (${fallbackCount} fallback)` : ""),
  );
  return { markdown, slots };
}

export async function summarizePaperDetail(
  paper: DailyPaperWithContent,
  deps: SummarizerDeps,
): Promise<string> {
  throwIfCancelled(deps.signal);
  if (!paper.fullSections) {
    throw new Error(
      `summarizePaperDetail: paper ${paper.id} has no full sections`,
    );
  }

  const topic = deps.arxivSettings.topics.find((t) => t.tag === paper.category);
  const topicName = topic?.name || paper.category;
  const summaryLanguage = normalizeSummaryLanguage(deps.summaryLanguage);
  const systemTemplate =
    summaryLanguage === "en" ? detailSystemTemplateEn : detailSystemTemplate;
  const systemPrompt = renderPrompt(systemTemplate, {
    topicName,
    injectionGuard:
      summaryLanguage === "en" ? injectionGuardEn : injectionGuardZh,
  });

  const userContent =
    `<paper_data>\n` +
    `标题: ${escapePaperDataFence(paper.title)}\n` +
    `arXiv: https://arxiv.org/abs/${escapePaperDataFence(paper.id)}\n` +
    `作者: ${escapePaperDataFence(paper.authors)}\n\n` +
    `以下是论文各章节内容：\n\n${escapePaperDataFence(paper.fullSections)}\n` +
    `</paper_data>`;

  const summary = await deps.llm.call(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userContent },
    ],
    {
      signal: deps.signal,
      onMetrics: deps.onMetrics,
    },
  );
  throwIfCancelled(deps.signal);
  if (!summary.trim()) {
    throw new Error(`summarizePaperDetail: empty LLM response for ${paper.id}`);
  }
  deps.logger.info(
    `summarizePaperDetail: ${paper.id} → ${summary.length} chars`,
  );
  return summary;
}

/**
 * Runs `task` for every index with at most `limit` in flight. The first
 * failure is the run's failure: it stops the remaining workers through the
 * shared signal, and the cancellations that causes in them are not reported.
 */
async function runWithConcurrency(
  count: number,
  limit: number,
  task: (index: number, signal: AbortSignal | undefined) => Promise<void>,
  signal?: AbortSignal,
): Promise<void> {
  if (count === 0) return;
  const pool = linkedAbortController(signal);
  let next = 0;
  let failure: { error: unknown } | undefined;
  const workers = Array.from(
    { length: Math.min(Math.max(1, limit), count) },
    async () => {
      while (next < count && !failure) {
        const index = next;
        next += 1;
        try {
          await task(index, pool.signal);
        } catch (error) {
          failure ??= { error };
          pool.abort();
          return;
        }
      }
    },
  );
  try {
    await Promise.all(workers);
  } finally {
    pool.cleanup();
  }
  if (failure) throw failure.error;
}

/** A signal that follows `signal` and can also be aborted by the pool itself. */
function linkedAbortController(signal?: AbortSignal): {
  signal: AbortSignal;
  abort(): void;
  cleanup(): void;
} {
  const controller = new AbortController();
  const follow = () => controller.abort(signal?.reason);
  signal?.addEventListener("abort", follow, { once: true });
  if (signal?.aborted) follow();
  return {
    signal: controller.signal,
    abort: () => controller.abort(new Error("stopped after another paper failed")),
    cleanup: () => signal?.removeEventListener("abort", follow),
  };
}
