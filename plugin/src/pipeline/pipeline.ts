import type { Logger } from "../services/logger";
import type { ProgressReporter } from "../services/progress";
import { NoopProgressReporter } from "../services/progress";
import type {
  ArxivSettings,
  AdvancedSettings,
  OutputSettings,
  LlmSettings,
} from "../settings/types";
import { arxivCategories } from "../settings/categories";
import type { ArxivFetcher } from "./arxiv-fetcher";
import type { PaperContentFetcher } from "./paper-content";
import type { MarkdownWriter } from "./markdown-writer";
import type { LlmClient } from "../llm/client";
import type { PaperIndexEntry, PaperIndexStore } from "../services/paper-index";
import {
  isCancellationError,
  throwIfCancelled,
} from "../services/cancellation";
import { parseRecent, type DateBucket, type PaperMeta } from "./arxiv-parser";
import { filterPapers, type FilteredPaper } from "./paper-filter";
import {
  summarizeDaily,
  summarizePaperDetail,
  type DailyPaperWithContent,
} from "./summarizer";
import { extractPaperSummaries } from "./daily-summary-parser";

interface SourcePaperMeta extends PaperMeta {
  arxivCategories: string[];
  published?: string;
  updated?: string;
}

export type PipelineResult =
  | { kind: "completed"; papersWritten: number }
  | { kind: "pending"; reason: string }
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

    // 1-2. Fetch and parse /recent for all configured categories.
    this.progress.setStage("fetch-recent");
    const fetched = await this.fetchPapersForDate(dateStr, signal);
    if (fetched.kind !== "ok") return fetched.result;
    const sourcePapers = fetched.papers;
    const dateWindowNote = undefined;
    logger.info(
      `pipeline: ${sourcePapers.length} papers for ${dateStr} across ${fetched.categories.join(", ")}`,
    );

    // 3. Empty day
    if (sourcePapers.length === 0) {
      throwIfCancelled(signal);
      await this.deps.writer.writeEmptyDaily(dateStr, { dateWindowNote });
      return { kind: "completed", papersWritten: 0 };
    }

    // 4. Enrich abstracts via Atom API (listings no longer include them)
    this.progress.setStage("enrich-abstract");
    try {
      const ids = sourcePapers.map((p) => p.id);
      const metadataMap = await fetcher.fetchMetadataByIds(ids);
      for (const p of sourcePapers) {
        const meta = metadataMap.get(p.id);
        if (!meta) continue;
        if (meta.abstract) p.abstract = meta.abstract;
        const updated = dateOnly(meta.updated);
        if (updated) p.updated = updated;
        for (const category of sourceCategories(meta)) {
          p.arxivCategories = appendUnique(p.arxivCategories, category);
        }
      }
      logger.info(
        `pipeline: enriched ${metadataMap.size}/${ids.length} papers via Atom API`,
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
    const filtered = await filterPapers(sourcePapers, {
      llm: this.deps.llm,
      logger,
      arxivSettings: this.deps.arxiv,
      signal,
    });
    throwIfCancelled(signal);
    if (filtered.length === 0) {
      await this.deps.writer.writeEmptyDaily(dateStr, { dateWindowNote });
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
      await this.deps.writer.writeEmptyDaily(dateStr, { dateWindowNote });
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
          // Daily summaries should use any high-value sections we can extract.
          // The detail flag still only controls whether a separate paper note is written.
          isDetail: true,
          sectionCharLimit: this.deps.advanced.sectionCharLimit,
          paperCharLimit: this.deps.advanced.paperCharLimit,
        });
        enriched.push({
          ...p,
          abstractConclusion: c.abstractConclusion,
          fullSections: c.fullSections,
          inboxStatus: p.indexEntry?.status,
          seenBefore: p.seenBefore,
          paperPath: p.indexEntry?.paperPath ?? null,
          detailLink: this.deps.writer.paperDetailLink(
            p.id,
            dateStr,
            p.indexEntry?.paperPath,
          ),
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
          detailLink: this.deps.writer.paperDetailLink(
            p.id,
            dateStr,
            p.indexEntry?.paperPath,
          ),
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
        linkStyle: this.deps.output.linkStyle ?? "wikilink",
        summaryLanguage: this.deps.output.summaryLanguage,
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
        await this.deps.paperIndex.setSummaries(
          extractPaperSummaries(dailySummary),
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
    await this.deps.writer.writeDaily(dateStr, dailySummary, { dateWindowNote });

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
          summaryLanguage: this.deps.output.summaryLanguage,
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
          published: dateStr,
          updated: paperUpdatedDate(p),
          arxivCategory: sourceCategories(p)[0] ?? this.deps.arxiv.category,
          arxivCategories: sourceCategories(p),
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

  private async fetchPapersForDate(
    dateStr: string,
    signal?: AbortSignal,
  ): Promise<
    | {
        kind: "ok";
        papers: SourcePaperMeta[];
        categories: string[];
        dateWindow: "recent";
      }
    | { kind: "error"; result: PipelineResult }
  > {
    const { fetcher, logger } = this.deps;
    const categories = arxivCategories(this.deps.arxiv);
    const byId = new Map<string, SourcePaperMeta>();

    for (const category of categories) {
      throwIfCancelled(signal);
      let recentHtml: string;
      try {
        recentHtml = await fetcher.fetchRecent(category);
      } catch (e) {
        if (isCancellationError(e)) throw e;
        return {
          kind: "error",
          result: {
            kind: "failed_transient",
            reason: `fetch /recent failed for ${category}: ${(e as Error).message}`,
          },
        };
      }

      let buckets: DateBucket[];
      try {
        buckets = parseRecent(recentHtml);
      } catch (e) {
        if (isCancellationError(e)) throw e;
        return {
          kind: "error",
          result: {
            kind: "failed_permanent",
            reason: `parse failed for ${category}: ${(e as Error).message}`,
          },
        };
      }

      const bucket = buckets.find((b) => b.announceDate === dateStr);
      if (!bucket) {
        const bounds = recentDateBounds(buckets);
        return {
          kind: "error",
          result: {
            kind: "failed_transient",
            reason: missingRecentDateReason(dateStr, category, buckets, bounds),
          },
        };
      }

      logger.info(
        `pipeline: ${bucket.papers.length} papers for ${dateStr} in ${category}`,
      );
      for (const paper of bucket.papers) {
        addSourcePaper(
          byId,
          { ...paper, published: dateStr } as SourcePaperMeta,
          [category],
        );
      }
    }

    return {
      kind: "ok",
      papers: Array.from(byId.values()),
      categories,
      dateWindow: "recent",
    };
  }
}

function sourceCategories(paper: PaperMeta, fallbackCategory?: string): string[] {
  const categories = (paper as Partial<SourcePaperMeta>).arxivCategories;
  if (Array.isArray(categories)) return categories;
  const atomCategories = (paper as { categories?: unknown }).categories;
  if (Array.isArray(atomCategories)) {
    return atomCategories.filter(
      (value): value is string => typeof value === "string",
    );
  }
  return fallbackCategory ? [fallbackCategory] : [];
}

function paperUpdatedDate(paper: PaperMeta): string | undefined {
  return dateOnly((paper as Partial<SourcePaperMeta>).updated);
}

function dateOnly(value: string | undefined): string | undefined {
  const trimmed = value?.trim() ?? "";
  const match = /^(\d{4}-\d{2}-\d{2})/.exec(trimmed);
  return match?.[1] ?? (trimmed || undefined);
}

function addSourcePaper(
  byId: Map<string, SourcePaperMeta>,
  paper: PaperMeta,
  categories: string[],
): void {
  const existing = byId.get(paper.id);
  if (existing) {
    for (const category of categories) {
      existing.arxivCategories = appendUnique(
        existing.arxivCategories,
        category,
      );
    }
  } else {
    byId.set(paper.id, { ...paper, arxivCategories: categories });
  }
}

function appendUnique(values: string[], value: string): string[] {
  return values.includes(value) ? values : [...values, value];
}

function recentDateBounds(
  buckets: DateBucket[],
): { oldest: string; newest: string } | null {
  const dates = buckets.map((bucket) => bucket.announceDate).sort();
  if (dates.length === 0) return null;
  return { oldest: dates[0], newest: dates[dates.length - 1] };
}

function missingRecentDateReason(
  dateStr: string,
  category: string,
  buckets: DateBucket[],
  bounds: { oldest: string; newest: string } | null,
): string {
  if (bounds && dateStr > bounds.newest) {
    return (
      `date ${dateStr} is newer than newest ${category} /recent bucket ` +
      `${bounds.newest}; arXiv announce page may not be available yet`
    );
  }
  const have = buckets.map((b) => b.announceDate).join(",");
  return `date ${dateStr} is not in ${category} /recent (have: ${have})`;
}
