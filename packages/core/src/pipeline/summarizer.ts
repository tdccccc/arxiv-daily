import {
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
  summarizeDailyPaperWithValidationRetry,
} from "./daily-paper-summary";
import {
  assembleDailySummary,
  assembleEmergencyDailySummary,
  isDailySummaryAssemblyRuntimeError,
  type DailyPaperResult,
  type DailyPaperSlot,
  type DailySummaryAssemblyInput,
  type DailySummaryAssemblyPaper,
  preflightDailySummaryAssembly,
  preflightDailySummaryPapers,
} from "./daily-summary-assembler";
import {
  isDailySummaryRescueExhaustedError,
  rescueDailySummary,
} from "./daily-summary-rescue";
import type { FilteredPaper } from "./paper-filter";
import { escapePaperDataFence } from "./prompt-safety";

export interface DailyPaperWithContent extends FilteredPaper {
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
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
  onDailyPaperProgress?: (completed: number, total: number) => void;
  /** Lower-level renderer injected behind the production assembler preflight boundary. */
  dailyRenderer?: (input: DailySummaryAssemblyInput) => string;
  rescueDaily?: typeof rescueDailySummary;
  assembleEmergencyDaily?: (input: DailySummaryAssemblyInput) => string;
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
  }));
  preflightDailySummaryPapers(assemblyPapers, deps.arxivSettings);

  const slots: DailyPaperSlot[] = [];
  let fallbackCount = 0;
  for (let i = 0; i < papers.length; i += 1) {
    throwIfCancelled(deps.signal);
    const paper = papers[i]!;
    const checkpointInput: DailySummaryCheckpointCompatibilityInput = {
      paper,
      summaryLanguage: deps.summaryLanguage,
      llm: deps.llmSettings,
    };
    const reusable = await deps.checkpointStore?.lookupReusable(
      dateStr,
      checkpointInput,
    );
    throwIfCancelled(deps.signal);

    let result: DailyPaperResult;
    let recovered = false;
    if (reusable) {
      result = reusable;
      recovered = true;
    } else {
      result = await summarizeDailyPaperWithValidationRetry(paper, {
        llm: deps.llm,
        summaryLanguage: deps.summaryLanguage,
        signal: deps.signal,
        onMetrics: deps.onMetrics,
      });
      throwIfCancelled(deps.signal);
      await deps.checkpointStore?.upsert(dateStr, checkpointInput, result);
      throwIfCancelled(deps.signal);
    }
    if (result.kind === "fallback") {
      fallbackCount += 1;
      deps.logger.warn(
        `summarizeDaily: fallback for ${paper.id} reason=${result.reasonCode} attempts=${result.attempts}` +
          (recovered ? " recovered=true" : ""),
      );
    }
    slots.push({
      paper: assemblyPapers[i]!,
      result,
    });
    deps.onDailyPaperProgress?.(i + 1, papers.length);
  }

  throwIfCancelled(deps.signal);
  const assemblyInput: DailySummaryAssemblyInput = {
    slots,
    dateStr,
    arxivSettings: deps.arxivSettings,
    summaryLanguage: deps.summaryLanguage,
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
      if (
        !isDailySummaryRescueExhaustedError(rescueError) &&
        !isLlmTransientExhaustedError(rescueError)
      ) {
        throw rescueError;
      }
      const cause = isDailySummaryRescueExhaustedError(rescueError)
        ? "validation-exhausted"
        : "transport-exhausted";
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
