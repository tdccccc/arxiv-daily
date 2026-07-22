import type { LlmClient } from "../llm/client";
import type { MetricsObserver } from "../metrics/generation";
import detailSystemTemplateEn from "../prompts/paper-detail.en.system.md";
import detailSystemTemplate from "../prompts/paper-detail.system.md";
import injectionGuard from "../prompts/injection-guard.md";
import { renderPrompt } from "../prompts/render";
import { throwIfCancelled } from "../services/cancellation";
import type { PaperIndexEntry, PaperStatus } from "../services/paper-index";
import type { Logger } from "../services/logger";
import { normalizeSummaryLanguage } from "../settings/summary-language";
import type {
  AdvancedSettings,
  ArxivSettings,
  LinkStyle,
  SummaryLanguage,
} from "../settings/types";
import {
  derivePaperSourceSections,
  summarizeDailyPaper,
} from "./daily-paper-summary";
import {
  assembleDailySummary,
  type DailySummaryAssemblyPaper,
  type StructuredPaperSummary,
} from "./daily-summary-assembler";
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

export interface SummarizerDeps {
  llm: LlmClient;
  logger: Logger;
  arxivSettings: ArxivSettings;
  advanced: AdvancedSettings;
  linkStyle?: LinkStyle;
  summaryLanguage?: SummaryLanguage;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
  onDailyPaperProgress?: (completed: number, total: number) => void;
}

export async function summarizeDaily(
  papers: DailyPaperWithContent[],
  dateStr: string,
  deps: SummarizerDeps,
): Promise<string> {
  throwIfCancelled(deps.signal);
  const summaries: StructuredPaperSummary[] = [];
  const assemblyPapers: DailySummaryAssemblyPaper[] = [];

  for (let i = 0; i < papers.length; i += 1) {
    throwIfCancelled(deps.signal);
    const paper = papers[i]!;
    const sourceSections = derivePaperSourceSections(paper);
    const summary = await summarizeDailyPaper(paper, {
      llm: deps.llm,
      summaryLanguage: deps.summaryLanguage,
      signal: deps.signal,
      onMetrics: deps.onMetrics,
    });
    throwIfCancelled(deps.signal);
    summaries.push(summary);
    assemblyPapers.push({
      id: paper.id,
      title: paper.title,
      authors: paper.authors,
      category: paper.category,
      sourceSections,
      isDetail: paper.isDetail,
      paperPath: paper.paperPath,
      detailLink: paper.detailLink,
    });
    deps.onDailyPaperProgress?.(i + 1, papers.length);
  }

  throwIfCancelled(deps.signal);
  const markdown = assembleDailySummary({
    papers: assemblyPapers,
    summaries,
    dateStr,
    arxivSettings: deps.arxivSettings,
    summaryLanguage: deps.summaryLanguage,
  });
  deps.logger.info(
    `summarizeDaily: assembled ${papers.length} sequential paper summaries`,
  );
  return markdown;
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
    injectionGuard,
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
