import type { MarkupParser } from "../core/adapters";
import { buildDailyDigest, emptyDailyDigest } from "../delivery/digest";
import type { DailyDigest } from "../delivery/types";
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
import { isPermanentLlmError, type LlmClient } from "../llm/client";
import type { PaperIndexEntry, PaperIndexStore } from "../services/paper-index";
import {
  isCancellationError,
  throwIfCancelled,
} from "../services/cancellation";
import type { PaperMeta } from "./arxiv-parser";
import {
  filterPapers,
  isPaperFilterCheckpointError,
  type DailyFilterCheckpointPort,
  type FilteredPaper,
} from "./paper-filter";
import type { PersonalizedFilterCheckpointPort } from "./personalized-paper-filter";
import {
  summarizeDaily,
  summarizePaperDetail,
  type DailyPaperWithContent,
  type DailySummaryCheckpointPort,
} from "./summarizer";
import { extractPaperSummaries } from "./daily-summary-parser";
import { dailySelectionMarkerRegExp } from "../services/daily-selection-marker";
import { GenerationMetricsCollector } from "../metrics/generation";
import {
  selectDetailPapers,
  type DetailSelectionPolicy,
} from "./detail-selector";
import {
  ArxivSourceAdapter,
  legacyContentFromNormalized,
  paperMetaFromSourcePaper,
  type SourceAdapter,
} from "../sources";
import { classifyPaperNote } from "../dashboard/paper-note-classifier";

/** Pipeline-local paper meta after source listing (arXiv-compatible shape). */
interface SourcePaperMeta extends PaperMeta {
  arxivCategories: string[];
  published?: string;
  updated?: string;
  paperKey?: string;
}

export type PipelineResult =
  | { kind: "completed"; papersWritten: number; digest?: DailyDigest }
  | { kind: "pending"; reason: string }
  | { kind: "cancelled"; reason: string }
  | { kind: "failed_transient"; reason: string }
  | { kind: "failed_permanent"; reason: string };

type PipelineFailureResult = Extract<
  PipelineResult,
  { kind: "failed_transient" | "failed_permanent" }
>;

const CONTENT_FETCH_CONCURRENCY = 6;

export interface DateScopedCheckpointLifecyclePort {
  removeAll(reportDate: string): Promise<void>;
}

export interface DailySummaryCheckpointLifecyclePort
  extends DailySummaryCheckpointPort,
    DateScopedCheckpointLifecyclePort {}

export interface DailyFilterCheckpointLifecyclePort
  extends DailyFilterCheckpointPort,
    PersonalizedFilterCheckpointPort,
    DateScopedCheckpointLifecyclePort {}

export interface DailyGenerationCheckpointStores {
  filter?: DailyFilterCheckpointLifecyclePort;
  summary?: DailySummaryCheckpointLifecyclePort;
}

export interface PipelineDeps {
  fetcher: ArxivFetcher;
  markupParser: MarkupParser;
  paperFetcher: PaperContentFetcher;
  writer: MarkdownWriter;
  paperIndex?: PaperIndexStore;
  checkpointStores?: DailyGenerationCheckpointStores;
  /** @deprecated Host compatibility until checkpointStores wiring lands. */
  checkpointStore?: DailySummaryCheckpointLifecyclePort;
  llm: LlmClient;
  logger: Logger;
  arxiv: ArxivSettings;
  advanced: AdvancedSettings;
  output: OutputSettings;
  llmSettings: LlmSettings;
  detailSelection: DetailSelectionPolicy;
  progress?: ProgressReporter;
  summarizeDaily?: typeof summarizeDaily;
  /**
   * Optional multi-source discovery/content port. When omitted, pipeline builds
   * an ArxivSourceAdapter from fetcher + paperFetcher (backward compatible).
   */
  sourceAdapter?: SourceAdapter;
}

export class ArxivPipeline {
  private progress: ProgressReporter;
  private sourceAdapter: SourceAdapter;

  constructor(private deps: PipelineDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
    this.sourceAdapter =
      deps.sourceAdapter ??
      new ArxivSourceAdapter({
        fetcher: deps.fetcher,
        paperFetcher: deps.paperFetcher,
        markupParser: deps.markupParser,
        logger: deps.logger,
        defaultCategories: arxivCategories(deps.arxiv),
      });
  }

  async runForDate(
    dateStr: string,
    signal?: AbortSignal,
  ): Promise<PipelineResult> {
    try {
      return await this.runForDateInner(dateStr, signal);
    } catch (e) {
      if (isCancellationError(e)) {
        return { kind: "cancelled", reason: (e as Error).message };
      }
      throw e;
    }
  }

  private async runForDateInner(
    dateStr: string,
    signal?: AbortSignal,
  ): Promise<PipelineResult> {
    const { fetcher, logger } = this.deps;
    const t0 = Date.now();
    const runMetrics = new GenerationMetricsCollector();
    const stageStart = (label: string) => {
      const elapsed = Date.now() - t0;
      logger.info(`pipeline: [${elapsed}ms] enter stage: ${label}`);
    };
    const stageEnd = (label: string, detail = "") => {
      const elapsed = Date.now() - t0;
      logger.info(`pipeline: [${elapsed}ms] done stage: ${label}${detail}`);
    };
    throwIfCancelled(signal);
    logger.info(`pipeline: start for ${dateStr}`);

    // 0. An existing daily note is the durable Markdown commit. Repair the
    // derived Paper Index idempotently before reporting completion; this also
    // supports notes written by older versions without a separate marker.
    if (await this.deps.writer.dailyExists(dateStr)) {
      logger.info(`pipeline: daily ${dateStr} already exists, repairing index`);
      await this.cleanupCommittedCheckpoints(dateStr);
      return await this.repairExistingDaily(dateStr, signal);
    }
    throwIfCancelled(signal);

    // 1-2. Discover papers via SourceAdapter (arXiv /recent + abstract enrich).
    this.progress.setStage("fetch-recent");
    stageStart("fetch-recent");
    const fetched = await this.fetchPapersForDate(dateStr, signal);
    if (fetched.kind !== "ok") return fetched.result;
    const sourcePapers = fetched.papers;
    logger.info(
      `pipeline: ${sourcePapers.length} papers for ${dateStr} across ${fetched.categories.join(", ")}`,
    );
    stageEnd("fetch-recent");

    // 3. Empty day
    if (sourcePapers.length === 0) {
      throwIfCancelled(signal);
      // Don't write empty file - let scheduler retry later
      return { kind: "pending", reason: "no papers from arXiv" };
    }

    // 4. Abstract enrichment is performed inside SourceAdapter.listForDate.
    stageStart("enrich-abstract");
    this.progress.setStage("enrich-abstract");
    stageEnd("enrich-abstract");

    // 5. LLM filter
    stageStart("filter");
    this.progress.setStage("filter");
    let filtered: FilteredPaper[];
    try {
      filtered = await filterPapers(sourcePapers, {
        llm: this.deps.llm,
        logger,
        arxivSettings: this.deps.arxiv,
        reportDate: dateStr,
        llmSettings: this.deps.llmSettings,
        checkpointStore: this.deps.checkpointStores?.filter,
        personalizedCheckpointStore: this.deps.checkpointStores?.filter,
        signal,
        onMetrics: (metrics) => runMetrics.record(metrics),
      });
    } catch (e) {
      if (isCancellationError(e)) throw e;
      if (isPaperFilterCheckpointError(e)) {
        return {
          kind: "failed_transient",
          reason: `paper filter checkpoint failed: ${e.message}`,
        };
      }
      return {
        kind: isPermanentLlmError(e) ? "failed_permanent" : "failed_transient",
        reason: `paper filter LLM failed: ${(e as Error).message}`,
      };
    }
    throwIfCancelled(signal);
    stageEnd("filter", ` (${filtered.length}/${sourcePapers.length} kept)`);
    if (filtered.length === 0) {
      throwIfCancelled(signal);
      // Don't write empty file - show "0" in calendar
      return {
        kind: "completed",
        papersWritten: 0,
        digest: this.buildZeroDigest(dateStr),
      };
    }

    throwIfCancelled(signal);
    const indexed = await this.indexFilteredPapers(filtered, dateStr);
    if (indexed.kind !== "ok") return indexed.result;
    const visiblePapers = indexed.papers.filter(
      (p) => p.indexEntry?.status !== "ignored",
    );
    if (visiblePapers.length === 0) {
      throwIfCancelled(signal);
      // Don't write empty file - show "0" in calendar
      return {
        kind: "completed",
        papersWritten: 0,
        digest: this.buildZeroDigest(dateStr),
      };
    }

    // 6. Fetch content for each filtered paper
    stageStart("fetch-content");
    let completedFetches = 0;
    let enriched = await mapConcurrent(
      visiblePapers,
      CONTENT_FETCH_CONCURRENCY,
      async (p) => {
        throwIfCancelled(signal);
        completedFetches += 1;
        this.progress.setStage("fetch-content", completedFetches, visiblePapers.length);
        try {
          // Daily summaries should use any high-value sections we can extract.
          // The detail flag still only controls whether a separate paper note is written.
          const normalized = await this.sourceAdapter.fetchContent(p.id, {
            wantFullText: true,
            sectionCharLimit: this.deps.advanced.sectionCharLimit,
            paperCharLimit: this.deps.advanced.paperCharLimit,
            signal,
          });
          const c = legacyContentFromNormalized(normalized);
          return {
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
          };
        } catch (e) {
          if (isCancellationError(e)) throw e;
          logger.error(`pipeline: content fetch failed for ${p.id}`, e);
          return {
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
          };
        }
      },
    );
    for (let i = 0; i < enriched.length; i += 1) {
      throwIfCancelled(signal);
    }
    stageEnd("fetch-content", ` (${enriched.length} papers)`);

    // Discover canonical detail notes that exist on disk even when an older or
    // incomplete index entry has no paperPath. This must happen before scoring:
    // existing notes neither reach the selector nor consume its soft quota.
    enriched = await mapConcurrent(
      enriched,
      CONTENT_FETCH_CONCURRENCY,
      async (paper) => {
        throwIfCancelled(signal);
        const path = await this.verifiedDetailPath(paper.id, signal);
        if (!path) return { ...paper, paperPath: null };
        if (paper.paperPath !== path) {
          await this.repairPaperPath(paper.id, path, signal);
        }
        return {
          ...paper,
          paperPath: path,
          detailLink: this.deps.writer.paperDetailLink(paper.id, dateStr, path),
        };
      },
    );

    // Existing detail notes remain linked but are excluded by the selector.
    const selection = await selectDetailPapers(
      enriched,
      this.deps.arxiv.topics ?? [],
      this.deps.detailSelection,
      {
        llm: this.deps.llm,
        logger,
        signal,
        onMetrics: (metrics) => runMetrics.record(metrics),
      },
    );
    const selectedIds = new Set(selection.selected.map(({ id }) => id));

    // 7. Detail reports. Only confirm isDetail/paperPath after a note exists,
    // so a failed selected detail cannot inflate or leave a dangling daily link.
    stageStart("write-detail");
    const detailCandidates = enriched.filter(
      (paper) => selectedIds.has(paper.id) && paper.fullSections,
    );
    for (let i = 0; i < detailCandidates.length; i++) {
      throwIfCancelled(signal);
      const paper = detailCandidates[i];
      if (!paper) continue;
      this.progress.setStage("write-detail", i + 1, detailCandidates.length);
      logger.info(`pipeline: detail report for ${paper.id}`);
      let confirmedPath: string | null = null;
      try {
        const detailMetrics = new GenerationMetricsCollector();
        const detail = await summarizePaperDetail(paper, {
          llm: this.deps.llm,
          llmSettings: this.deps.llmSettings,
          logger,
          arxivSettings: this.deps.arxiv,
          advanced: this.deps.advanced,
          summaryLanguage: this.deps.output.summaryLanguage,
          signal,
          onMetrics: (metrics) => {
            detailMetrics.record(metrics);
            runMetrics.record(metrics);
          },
        });
        throwIfCancelled(signal);
        confirmedPath = await this.deps.writer.writePaperDetail(
          paper,
          dateStr,
          detail,
          paper.indexEntry,
          { metrics: detailMetrics.snapshot() },
        );
      } catch (e) {
        if (isCancellationError(e)) throw e;
        logger.error(`pipeline: detail failed for ${paper.id}`, e);
        // Handle a note created between the initial existence check and write.
        confirmedPath = await this.verifiedDetailPath(paper.id, signal);
      }
      if (!confirmedPath) continue;
      await this.repairPaperPath(paper.id, confirmedPath, signal);
      enriched = enriched.map((candidate) =>
        candidate.id === paper.id
          ? {
              ...candidate,
              isDetail: true,
              paperPath: confirmedPath,
              detailLink: this.deps.writer.paperDetailLink(
                candidate.id,
                dateStr,
                confirmedPath,
              ),
            }
          : candidate,
      );
    }
    stageEnd("write-detail", ` (${detailCandidates.length} selected papers)`);

    // 8. Daily summary, after detail attempts have established which links are real.
    throwIfCancelled(signal);
    stageStart("summarize-daily");
    this.progress.setStage("summarize-daily");
    let dailySummary: string;
    let digestSlots: Awaited<ReturnType<typeof summarizeDaily>>["slots"] = [];
    try {
      const summarized = await (this.deps.summarizeDaily ?? summarizeDaily)(
        enriched,
        dateStr,
        {
          llm: this.deps.llm,
          llmSettings: this.deps.llmSettings,
          checkpointStore: this.deps.checkpointStores?.summary ?? this.deps.checkpointStore,
          logger,
          arxivSettings: this.deps.arxiv,
          advanced: this.deps.advanced,
          linkStyle: this.deps.output.linkStyle ?? "wikilink",
          summaryLanguage: this.deps.output.summaryLanguage,
          signal,
          onMetrics: (metrics) => runMetrics.record(metrics),
          onDailyPaperProgress: (completed, total) =>
            this.progress.setStage("summarize-daily", completed, total),
        },
      );
      dailySummary = summarized.markdown;
      digestSlots = summarized.slots;
    } catch (e) {
      if (isCancellationError(e)) throw e;
      const permanentLlmFailure = isPermanentLlmError(e);
      return {
        kind: permanentLlmFailure ? "failed_permanent" : "failed_transient",
        reason: `${permanentLlmFailure ? "daily summary LLM failed" : "daily summary failed"}: ${(e as Error).message}`,
      };
    }
    throwIfCancelled(signal);
    const dailyPath = this.deps.writer.dailyPath(dateStr);
    stageEnd("summarize-daily", ` (${dailySummary.length} chars)`);

    throwIfCancelled(signal);
    runMetrics.setPipelineElapsedMs(Date.now() - t0);
    await this.deps.writer.writeDaily(dateStr, dailySummary, {
      metrics: runMetrics.snapshot(),
    });
    await this.cleanupCommittedCheckpoints(dateStr);
    if (this.deps.paperIndex) {
      try {
        throwIfCancelled(signal);
        await this.deps.paperIndex.addDailyReports(
          visiblePapers.map((p) => p.id),
          dailyPath,
        );
        throwIfCancelled(signal);
        await this.deps.paperIndex.setSummaries(
          extractPaperSummaries(dailySummary),
        );
        throwIfCancelled(signal);
      } catch (e) {
        if (isCancellationError(e)) throw e;
        return {
          kind: "failed_transient",
          reason: `paper index daily report update failed: ${(e as Error).message}`,
        };
      }
    }
    throwIfCancelled(signal);
    const totalS = ((Date.now() - t0) / 1000).toFixed(1);
    logger.info(
      `pipeline: completed ${dateStr} in ${totalS}s — ` +
      `${enriched.length} papers, ${enriched.filter((paper) => paper.paperPath).length} detail reports`,
    );
    const digest = buildDailyDigest({
      date: dateStr,
      arxiv: this.deps.arxiv,
      output: this.deps.output,
      slots: digestSlots,
      dailyPath,
    });
    return {
      kind: "completed",
      papersWritten: enriched.length,
      digest,
    };
  }

  /** Zero-paper completed digests (filter empty / all ignored). Repair path omits digest. */
  private buildZeroDigest(dateStr: string): DailyDigest {
    return emptyDailyDigest({
      date: dateStr,
      arxiv: this.deps.arxiv,
      output: this.deps.output,
      dailyPath: this.deps.writer.dailyPath(dateStr),
    });
  }

  private async verifiedDetailPath(
    arxivId: string,
    signal?: AbortSignal,
  ): Promise<string | null> {
    if (!(await this.deps.writer.paperDetailExists(arxivId))) return null;
    throwIfCancelled(signal);
    const markdown = await this.deps.writer.readPaperDetail(arxivId);
    throwIfCancelled(signal);
    const classification = classifyPaperNote(markdown, arxivId);
    if (classification.kind !== "verified_detail") {
      this.deps.logger.warn(
        `pipeline: canonical note for ${arxivId} is ${classification.kind}, not verified detail`,
      );
      return null;
    }
    return this.deps.writer.paperDetailPath(arxivId);
  }

  private async repairPaperPath(
    arxivId: string,
    path: string,
    signal?: AbortSignal,
  ): Promise<void> {
    if (!this.deps.paperIndex) return;
    try {
      throwIfCancelled(signal);
      await this.deps.paperIndex.setPaperPath(arxivId, path);
      throwIfCancelled(signal);
    } catch (e) {
      if (isCancellationError(e)) throw e;
      this.deps.logger.error(`pipeline: detail index repair failed for ${arxivId}`, e);
    }
  }

  private async cleanupCommittedCheckpoints(dateStr: string): Promise<void> {
    const stores: Array<[
      keyof DailyGenerationCheckpointStores,
      DateScopedCheckpointLifecyclePort | undefined,
    ]> = [
      ["filter", this.deps.checkpointStores?.filter],
      ["summary", this.deps.checkpointStores?.summary ?? this.deps.checkpointStore],
    ];
    for (const [label, store] of stores) {
      if (!store) continue;
      try {
        await store.removeAll(dateStr);
      } catch (error) {
        this.deps.logger.warn(
          `pipeline: committed daily ${label} checkpoint cleanup failed for ${dateStr}`,
          error,
        );
      }
    }
  }

  private async repairExistingDaily(
    dateStr: string,
    signal?: AbortSignal,
  ): Promise<PipelineResult> {
    if (!this.deps.paperIndex) {
      return { kind: "completed", papersWritten: 0 };
    }
    try {
      throwIfCancelled(signal);
      const markdown = await this.deps.writer.readDaily(dateStr);
      throwIfCancelled(signal);
      const summaries = extractPaperSummaries(markdown);
      const arxivIds = Array.from(
        new Set([...extractDailyArxivIds(markdown), ...Object.keys(summaries)]),
      );
      const dailyPath = this.deps.writer.dailyPath(dateStr);
      const canonicalDetailPaths: Record<string, string | null> = {};
      for (const arxivId of arxivIds) {
        throwIfCancelled(signal);
        canonicalDetailPaths[arxivId] = await this.verifiedDetailPath(arxivId, signal);
      }
      throwIfCancelled(signal);
      await this.deps.paperIndex.reconcilePaperDetails(canonicalDetailPaths);
      throwIfCancelled(signal);
      await this.deps.paperIndex.addDailyReports(arxivIds, dailyPath);
      throwIfCancelled(signal);
      await this.deps.paperIndex.setSummaries(summaries);
      throwIfCancelled(signal);
      this.deps.logger.info(
        `pipeline: repaired daily and detail index for ${dateStr} (${arxivIds.length} papers)`,
      );
      return { kind: "completed", papersWritten: arxivIds.length };
    } catch (e) {
      if (isCancellationError(e)) throw e;
      return {
        kind: "failed_transient",
        reason: `paper index repair failed: ${(e as Error).message}`,
      };
    }
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
          abstract: p.abstract,
          primaryTopic: p.category,
          // Selection is only a candidate decision. The index becomes detail=true
          // after a real paper path is confirmed by setPaperPath.
          detail: false,
        })),
      );
      return {
        kind: "ok",
        papers: filtered.map((p, i) => {
          const result = results[i];
          if (!result) {
            throw new Error(`paper index result missing for ${p.id}`);
          }
          return {
            ...p,
            indexEntry: result.entry,
            wasNew: result.wasNew,
            seenBefore: !result.wasNew,
          };
        }),
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
    const listed = await this.sourceAdapter.listForDate(dateStr, {
      channels: arxivCategories(this.deps.arxiv),
      signal,
    });
    if (listed.kind === "error") {
      return {
        kind: "error",
        result: {
          kind: listed.failureKind,
          reason: listed.reason,
        },
      };
    }

    const papers: SourcePaperMeta[] = listed.papers.map((paper) => {
      const meta = paperMetaFromSourcePaper(paper);
      return {
        ...meta,
        paperKey: paper.paperKey,
      };
    });

    return {
      kind: "ok",
      papers,
      categories: listed.channels,
      dateWindow: "recent",
    };
  }
}

function extractDailyArxivIds(markdown: string): string[] {
  const ids = new Set<string>();
  for (const match of markdown.matchAll(dailySelectionMarkerRegExp("gmi"))) {
    if (match[2]) ids.add(match[2]);
  }
  const arxivBullet =
    /^[ \t]*[-*][ \t]+\*\*arXiv\*\*[:：][^\r\n]*?arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})(?:v\d+)?[^\r\n]*\r?$/gmi;
  for (const match of markdown.matchAll(arxivBullet)) {
    if (match[1]) ids.add(match[1]);
  }
  return Array.from(ids);
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
  const value = (paper as Partial<SourcePaperMeta>).updated?.trim() ?? "";
  const match = /^(\d{4}-\d{2}-\d{2})/.exec(value);
  return match?.[1] ?? (value || undefined);
}

async function mapConcurrent<T, R>(
  items: T[],
  limit: number,
  mapper: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let nextIndex = 0;
  const workers = Array.from(
    { length: Math.min(Math.max(1, limit), items.length) },
    async () => {
      while (nextIndex < items.length) {
        const index = nextIndex;
        nextIndex += 1;
        results[index] = await mapper(items[index]!, index);
      }
    },
  );
  await Promise.all(workers);
  return results;
}
