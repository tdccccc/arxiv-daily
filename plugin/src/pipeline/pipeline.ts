import type { Logger } from "../services/logger";
import type { ProgressReporter } from "../services/progress";
import { NoopProgressReporter } from "../services/progress";
import type {
  ArxivSettings,
  AdvancedSettings,
  OutputSettings,
  LlmSettings,
} from "../settings/types";
import type { ArxivFetcher } from "./arxiv-fetcher";
import type { PaperContentFetcher } from "./paper-content";
import type { MarkdownWriter } from "./markdown-writer";
import type { LlmClient } from "../llm/client";
import type { PaperIndexEntry, PaperIndexStore } from "../services/paper-index";
import {
  isCancellationError,
  throwIfCancelled,
} from "../services/cancellation";
import { parseRecent, type DateBucket } from "./arxiv-parser";
import { filterPapers, type FilteredPaper } from "./paper-filter";
import {
  summarizeDaily,
  summarizePaperDetail,
  type DailyPaperWithContent,
} from "./summarizer";

export type PipelineResult =
  | { kind: "completed"; papersWritten: number }
  | { kind: "failed_transient"; reason: string }
  | { kind: "failed_permanent"; reason: string };

export interface PipelineDeps {
  fetcher: ArxivFetcher;
  paperFetcher: PaperContentFetcher;
  writer: MarkdownWriter;
  paperIndex?: PaperIndexStore;
  llm: LlmClient;
  logger: Logger;
  arxiv: ArxivSettings;
  advanced: AdvancedSettings;
  output: OutputSettings;
  llmSettings: LlmSettings;
  progress?: ProgressReporter;
}

export class ArxivPipeline {
  private progress: ProgressReporter;

  constructor(private deps: PipelineDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
  }

  async runForDate(
    dateStr: string,
    signal?: AbortSignal,
  ): Promise<PipelineResult> {
    try {
      return await this.runForDateInner(dateStr, signal);
    } catch (e) {
      if (isCancellationError(e)) {
        return { kind: "failed_transient", reason: (e as Error).message };
      }
      throw e;
    }
  }

  private async runForDateInner(
    dateStr: string,
    signal?: AbortSignal,
  ): Promise<PipelineResult> {
    const { fetcher, logger } = this.deps;
    throwIfCancelled(signal);
    logger.info(`pipeline: start for ${dateStr}`);

    // 0. Skip if daily already exists.
    if (await this.deps.writer.dailyExists(dateStr)) {
      logger.info(`pipeline: daily ${dateStr} already exists, skipping`);
      return { kind: "completed", papersWritten: 0 };
    }
    throwIfCancelled(signal);

    // 1. Fetch /recent
    this.progress.setStage("fetch-recent");
    let recentHtml: string;
    try {
      recentHtml = await fetcher.fetchRecent();
    } catch (e) {
      if (isCancellationError(e)) throw e;
      return {
        kind: "failed_transient",
        reason: `fetch /recent failed: ${(e as Error).message}`,
      };
    }
    throwIfCancelled(signal);

    // 2. Parse
    let buckets: DateBucket[];
    try {
      buckets = parseRecent(recentHtml);
    } catch (e) {
      if (isCancellationError(e)) throw e;
      return {
        kind: "failed_permanent",
        reason: `parse failed: ${(e as Error).message}`,
      };
    }
    throwIfCancelled(signal);
    const bucket = buckets.find((b) => b.announceDate === dateStr);
    if (!bucket) {
      return {
        kind: "failed_transient",
        reason: `date ${dateStr} not in /recent (have: ${buckets
          .map((b) => b.announceDate)
          .join(",")})`,
      };
    }
    logger.info(`pipeline: ${bucket.papers.length} papers for ${dateStr}`);

    // 3. Empty day
    if (bucket.papers.length === 0) {
      throwIfCancelled(signal);
      await this.deps.writer.writeEmptyDaily(dateStr);
      return { kind: "completed", papersWritten: 0 };
    }

    // 4. Enrich abstracts via Atom API (listings no longer include them)
    this.progress.setStage("enrich-abstract");
    try {
      const ids = bucket.papers.map((p) => p.id);
      const absMap = await fetcher.fetchAbstractsByIds(ids);
      for (const p of bucket.papers) {
        const a = absMap.get(p.id);
        if (a) p.abstract = a;
      }
      logger.info(
        `pipeline: enriched ${absMap.size}/${ids.length} abstracts via Atom API`,
      );
    } catch (e) {
      if (isCancellationError(e)) throw e;
      logger.warn(
        `pipeline: abstract enrichment failed, continuing with titles only: ${(e as Error).message}`,
      );
    }
    throwIfCancelled(signal);

    // 5. LLM filter
    this.progress.setStage("filter");
    const filtered = await filterPapers(bucket.papers, {
      llm: this.deps.llm,
      logger,
      arxivSettings: this.deps.arxiv,
      signal,
    });
    throwIfCancelled(signal);
    if (filtered.length === 0) {
      await this.deps.writer.writeEmptyDaily(dateStr);
      return { kind: "completed", papersWritten: 0 };
    }

    throwIfCancelled(signal);
    const indexed = await this.indexFilteredPapers(filtered, dateStr);
    if (indexed.kind !== "ok") return indexed.result;
    const visiblePapers = indexed.papers.filter(
      (p) => p.indexEntry?.status !== "ignored",
    );
    if (visiblePapers.length === 0) {
      throwIfCancelled(signal);
      await this.deps.writer.writeEmptyDaily(dateStr);
      return { kind: "completed", papersWritten: 0 };
    }

    // 6. Fetch content for each filtered paper
    const enriched: DailyPaperWithContent[] = [];
    for (let i = 0; i < visiblePapers.length; i++) {
      throwIfCancelled(signal);
      const p = visiblePapers[i];
      this.progress.setStage("fetch-content", i + 1, visiblePapers.length);
      try {
        const c = await this.deps.paperFetcher.fetch(p.id, {
          isDetail: p.isDetail,
          sectionCharLimit: this.deps.advanced.sectionCharLimit,
          paperCharLimit: this.deps.advanced.paperCharLimit,
          skipSections: this.deps.advanced.skipSections,
          prioritySections: this.deps.advanced.prioritySections,
        });
        enriched.push({
          ...p,
          abstractConclusion: c.abstractConclusion,
          fullSections: c.fullSections,
          inboxStatus: p.indexEntry?.status,
          seenBefore: p.seenBefore,
          paperPath: p.indexEntry?.paperPath ?? null,
        });
      } catch (e) {
        if (isCancellationError(e)) throw e;
        logger.error(`pipeline: content fetch failed for ${p.id}`, e);
        enriched.push({
          ...p,
          abstractConclusion: `[获取失败] arXiv ID: ${p.id}`,
          fullSections: null,
          inboxStatus: p.indexEntry?.status,
          seenBefore: p.seenBefore,
          paperPath: p.indexEntry?.paperPath ?? null,
        });
      }
      throwIfCancelled(signal);
    }

    // 7. Daily summary
    throwIfCancelled(signal);
    this.progress.setStage("summarize-daily");
    let dailySummary: string;
    try {
      dailySummary = await summarizeDaily(enriched, dateStr, {
        llm: this.deps.llm,
        logger,
        arxivSettings: this.deps.arxiv,
        advanced: this.deps.advanced,
        llmTemperature: this.deps.llmSettings.temperature,
        signal,
      });
    } catch (e) {
      if (isCancellationError(e)) throw e;
      return {
        kind: "failed_transient",
        reason: `daily summary LLM failed: ${(e as Error).message}`,
      };
    }
    throwIfCancelled(signal);
    const dailyPath = this.deps.writer.dailyPath(dateStr);
    if (this.deps.paperIndex) {
      try {
        throwIfCancelled(signal);
        await this.deps.paperIndex.addDailyReports(
          visiblePapers.map((p) => p.id),
          dailyPath,
        );
      } catch (e) {
        if (isCancellationError(e)) throw e;
        return {
          kind: "failed_transient",
          reason: `paper index daily report update failed: ${(e as Error).message}`,
        };
      }
    }
    throwIfCancelled(signal);
    await this.deps.writer.writeDaily(dateStr, dailySummary);

    // 8. Detail reports
    const detailPapers = enriched.filter((p) => p.isDetail && p.fullSections);
    for (let i = 0; i < detailPapers.length; i++) {
      throwIfCancelled(signal);
      const p = detailPapers[i];
      if (await this.deps.writer.paperDetailExists(p.id)) {
        logger.info(`pipeline: detail ${p.id} already exists, skipping`);
        if (this.deps.paperIndex) {
          await this.deps.paperIndex.setPaperPath(
            p.id,
            this.deps.writer.paperDetailPath(p.id),
          );
        }
        continue;
      }
      this.progress.setStage("write-detail", i + 1, detailPapers.length);
      logger.info(`pipeline: detail report for ${p.id}`);
      try {
        const detail = await summarizePaperDetail(p, {
          llm: this.deps.llm,
          logger,
          arxivSettings: this.deps.arxiv,
          advanced: this.deps.advanced,
          llmTemperature: this.deps.llmSettings.temperature,
          signal,
        });
        throwIfCancelled(signal);
        const path = await this.deps.writer.writePaperDetail(
          p,
          dateStr,
          detail,
          p.indexEntry,
        );
        if (this.deps.paperIndex) {
          await this.deps.paperIndex.setPaperPath(p.id, path);
        }
      } catch (e) {
        if (isCancellationError(e)) throw e;
        logger.error(`pipeline: detail failed for ${p.id}`, e);
      }
    }

    throwIfCancelled(signal);
    return { kind: "completed", papersWritten: enriched.length };
  }

  private async indexFilteredPapers(
    filtered: FilteredPaper[],
    dateStr: string,
  ): Promise<
    | {
        kind: "ok";
        papers: Array<
          FilteredPaper & {
            indexEntry?: PaperIndexEntry;
            wasNew?: boolean;
            seenBefore?: boolean;
          }
        >;
      }
    | { kind: "error"; result: PipelineResult }
  > {
    const paperIndex = this.deps.paperIndex;
    if (!paperIndex) {
      return {
        kind: "ok",
        papers: filtered.map((p) => ({ ...p, wasNew: true, seenBefore: false })),
      };
    }

    try {
      const results = await paperIndex.upsertManyFromDailyPapers(
        filtered.map((p) => ({
          arxivId: p.id,
          title: p.title,
          authors: p.authors,
          date: dateStr,
          arxivCategory: this.deps.arxiv.category,
          primaryTopic: p.category,
          detail: p.isDetail,
        })),
      );
      return {
        kind: "ok",
        papers: filtered.map((p, i) => ({
          ...p,
          indexEntry: results[i].entry,
          wasNew: results[i].wasNew,
          seenBefore: !results[i].wasNew,
        })),
      };
    } catch (e) {
      return {
        kind: "error",
        result: {
          kind: "failed_permanent",
          reason: `paper index update failed: ${(e as Error).message}`,
        },
      };
    }
  }
}
