import { formatArxivHttpError, type ArxivFetcher } from "../pipeline/arxiv-fetcher";
import type { PaperContentFetcher } from "../pipeline/paper-content";
import type { MarkdownWriter } from "../pipeline/markdown-writer";
import type { LlmClient } from "../llm/client";
import type { Logger } from "./logger";
import type { ProgressReporter } from "./progress";
import { NoopProgressReporter } from "./progress";
import type { MarkupParser, StorageAdapter } from "../core/adapters";
import type { AdvancedSettings, ArxivSettings, LlmSettings, OutputSettings } from "../settings/types";
import { summarizePaperDetail, type DailyPaperWithContent } from "../pipeline/summarizer";
import type { PaperIndexEntry, PaperIndexStore } from "./paper-index";
import { GenerationMetricsCollector } from "../metrics/generation";
import { isCancellationError, throwIfCancelled } from "./cancellation";
import { modernArxivResources } from "../utils/arxiv";
import { validateVaultRelativeDirectory } from "../settings/validation";
import {
  classifyPaperNote,
  type VerifiedDetailMetadata,
} from "../dashboard/paper-note-classifier";

export type ManualFetchResult =
  | { kind: "done"; path: string }
  | { kind: "already_exists"; path: string }
  | { kind: "note_conflict"; path: string; reason: string }
  | { kind: "not_found"; reason: string }
  | { kind: "no_html"; reason: string }
  | { kind: "error"; reason: string };

export interface ManualFetchDeps {
  storage: StorageAdapter;
  markupParser: MarkupParser;
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

/** Normalize various user inputs (URL, "arXiv:xxx", plain id with/without version) into a base id. */
export function normalizeArxivId(input: string): string | null {
  return modernArxivResources(input)?.id ?? null;
}

export class ManualFetchService {
  private progress: ProgressReporter;

  constructor(private deps: ManualFetchDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
  }

  async fetchAndSummarize(rawId: string, dateStr: string, signal?: AbortSignal): Promise<ManualFetchResult> {
    const { storage, output, logger } = this.deps;
    throwIfCancelled(signal);

    logger.info(`manual-fetch: requested detail summary for ${rawId} on ${dateStr}`);
    const id = normalizeArxivId(rawId);
    if (!id) {
      logger.warn(`manual-fetch: invalid arXiv id: ${rawId}`);
      return { kind: "error", reason: `invalid arXiv id: ${rawId}` };
    }
    this.progress.setTask("arXiv Daily detail", id);
    this.progress.setStage("fetch-metadata");

    // 1. Duplicate check
    const papersDir = validateVaultRelativeDirectory(output.papersDir);
    if (!papersDir.ok || !papersDir.value) {
      return { kind: "error", reason: `invalid papersDir: ${papersDir.reason}` };
    }
    const targetPath = storage.normalizePath(`${papersDir.value}/${id}.md`);
    let replaceableExistingContent: string | null = null;
    if (await storage.exists(targetPath)) {
      let existing: string;
      try {
        existing = await storage.readText(targetPath);
      } catch (e) {
        if (isCancellationError(e)) throw e;
        const reason = `failed to read existing note ${targetPath}: ${errorMessage(e)}`;
        logger.error(`manual-fetch: ${reason}`, e);
        this.progress.setError(`Could not inspect existing note: ${id}`);
        return { kind: "error", reason };
      }
      const classification = classifyPaperNote(existing, id);
      if (classification.kind === "replaceable") {
        replaceableExistingContent = existing;
        logger.warn(
          `manual-fetch: replacing safe ${classification.form} note for ${id} at ${targetPath}`,
        );
      } else if (classification.kind === "verified_detail") {
        logger.info(`manual-fetch: verified detail summary already exists at ${targetPath}`);
        const reconciled = await this.reconcileVerifiedDetail(
          id,
          dateStr,
          targetPath,
          classification.metadata,
        );
        if (reconciled) return reconciled;
        this.progress.setComplete(`Detail note already exists: ${id}`);
        return { kind: "already_exists", path: targetPath };
      } else {
        const reason = conflictReason(classification.reason, id);
        logger.warn(`manual-fetch: protected existing note at ${targetPath}: ${reason}`);
        this.progress.setError(`Existing note conflict: ${id}`);
        return { kind: "note_conflict", path: targetPath, reason };
      }
    }

    // 2. Pull metadata + abstract from Atom API
    let title = "";
    let authors = "Unknown";
    let category = "other";
    let published = "";
    let updated = "";
    let categories: string[] = [];
    let abstract = "";
    try {
      const meta = (await this.deps.fetcher.fetchMetadataByIds([id], signal)).get(id) ?? null;
      if (!meta) {
        logger.warn(`manual-fetch: arXiv has no entry for ${id}`);
        this.progress.setError(`arXiv has no entry for ${id}`);
        return { kind: "not_found", reason: `arXiv has no entry for ${id}` };
      }
      logger.info(`manual-fetch: fetched Atom metadata for ${id}`);
      title = meta.title;
      authors = meta.authors;
      category = meta.primaryCategory || meta.categories[0] || "other";
      published = meta.published;
      updated = meta.updated;
      categories = meta.categories;
      abstract = meta.abstract;
    } catch (e) {
      if (isCancellationError(e)) throw e;
      const message = formatArxivHttpError(e);
      logger.error(`manual-fetch: atom metadata failed for ${id}`, e);
      this.progress.setError(`Metadata failed: ${message}`);
      return { kind: "error", reason: `atom metadata: ${message}` };
    }

    // 3. Pull /html and extract full sections
    this.progress.setStage("fetch-content");
    let content: {
      abstractConclusion: string;
      fullSections: string | null;
      fullTextFailure?: string;
    };
    try {
      content = await this.deps.paperFetcher.fetch(id, {
        isDetail: true,
        sectionCharLimit: this.deps.advanced.sectionCharLimit,
        paperCharLimit: this.deps.advanced.paperCharLimit,
      }, signal);
    } catch (e) {
      if (isCancellationError(e)) throw e;
      const message = formatArxivHttpError(e);
      logger.error(`manual-fetch: content fetch failed for ${id}`, e);
      this.progress.setError(`Content fetch failed: ${message}`);
      return { kind: "error", reason: `content fetch: ${message}` };
    }
    if (!content.fullSections) {
      logger.warn(
        `manual-fetch: no full text for ${id}: ${
          content.fullTextFailure ?? "no rendered HTML or extractable source"
        }`,
      );
      this.progress.setError(`No full text for ${id}`);
      return {
        kind: "no_html",
        reason:
          content.fullTextFailure ??
          `no rendered HTML or extractable arXiv source for ${id}; cannot produce a detail summary`,
      };
    }

    // 4. Summarize as detail
    const paper: DailyPaperWithContent = {
      id,
      title,
      authors,
      abstract,
      category,
      isDetail: true,
      abstractConclusion: content.abstractConclusion,
      fullSections: content.fullSections,
      published,
      updated,
    };
    let summary: string;
    const detailMetrics = new GenerationMetricsCollector();
    try {
      this.progress.setStage("summarize-detail");
      summary = await summarizePaperDetail(paper, {
        llm: this.deps.llm,
        logger,
        arxivSettings: this.deps.arxiv,
        advanced: this.deps.advanced,
        summaryLanguage: this.deps.output.summaryLanguage,
        signal,
        onMetrics: (metrics) => detailMetrics.record(metrics),
      });
    } catch (e) {
      if (isCancellationError(e)) throw e;
      logger.error(`manual-fetch: LLM summary failed for ${id}`, e);
      this.progress.setError(`Summary failed: ${(e as Error).message}`);
      return { kind: "error", reason: `LLM summary: ${(e as Error).message}` };
    }

    // 5. Write, then commit index state. A preexisting entry is safe input for
    // frontmatter generation, but no index mutation may precede the note write.
    throwIfCancelled(signal);
    this.progress.setStage("write-detail");
    let existingIndexEntry: PaperIndexEntry | null = null;
    if (this.deps.paperIndex) {
      try {
        existingIndexEntry = await this.deps.paperIndex.get(id);
      } catch (e) {
        logger.error(`manual-fetch: paper index read failed for ${id}`, e);
        this.progress.setError(`Index read failed: ${(e as Error).message}`);
        return {
          kind: "error",
          reason: `paper index: ${(e as Error).message}`,
        };
      }
    }

    // Final cancellation boundary. Revalidate a note approved for replacement
    // immediately before entering the non-interruptible write/index commit.
    throwIfCancelled(signal);
    if (replaceableExistingContent !== null) {
      let current: string;
      try {
        current = await storage.readText(targetPath);
      } catch (e) {
        if (isCancellationError(e)) throw e;
        const reason = `existing note changed or could not be re-read; protected ${id} note from overwrite`;
        logger.warn(`manual-fetch: ${reason} at ${targetPath}`, e);
        this.progress.setError(`Existing note conflict: ${id}`);
        return { kind: "note_conflict", path: targetPath, reason };
      }
      const classification = classifyPaperNote(current, id);
      if (current !== replaceableExistingContent || classification.kind !== "replaceable") {
        const reason = classification.kind === "conflict"
          ? conflictReason(classification.reason, id)
          : `existing note changed while the detail summary was prepared; protected ${id} note from overwrite`;
        logger.warn(`manual-fetch: protected changed note at ${targetPath}: ${reason}`);
        this.progress.setError(`Existing note conflict: ${id}`);
        return { kind: "note_conflict", path: targetPath, reason };
      }
    }

    logger.info(
      `manual-fetch: ${replaceableExistingContent !== null ? "replacing" : "writing"} detail note for ${id}`,
    );
    const path = await this.deps.writer.writePaperDetail(
      paper,
      dateStr,
      summary,
      existingIndexEntry ?? undefined,
      {
        metrics: detailMetrics.snapshot(),
        replaceExisting: replaceableExistingContent !== null,
      },
    );
    if (this.deps.paperIndex) {
      try {
        const displayDate = displayDateFromIndexEntry(existingIndexEntry);
        const indexed = await this.deps.paperIndex.reconcileManualDetail({
          arxivId: id,
          title,
          authors,
          date: dateStr,
          published: displayDate ?? published,
          updated,
          arxivCategory: category,
          arxivCategories: categories,
          abstract,
          primaryTopic: category,
          detail: true,
        }, path, "saved");
        logger.info(
          `manual-fetch: paper index ${indexed.wasNew ? "created as saved" : "updated"} ` +
          `with detail path for ${id}`,
        );
        await this.deps.writer.refreshPaperNoteFrontmatter(indexed.entry, path);
      } catch (e) {
        logger.error(`manual-fetch: paper index update failed for ${id}`, e);
        this.progress.setError(`Index update failed: ${(e as Error).message}`);
        return {
          kind: "error",
          reason: `paper index: ${(e as Error).message}`,
        };
      }
    }
    logger.info(`manual-fetch: wrote ${path}`);
    this.progress.setComplete(`Detail note ready: ${id}`);
    return { kind: "done", path };
  }

  private async reconcileVerifiedDetail(
    id: string,
    dateStr: string,
    path: string,
    metadata: VerifiedDetailMetadata,
  ): Promise<ManualFetchResult | null> {
    const paperIndex = this.deps.paperIndex;
    if (!paperIndex) return null;
    try {
      const existing = await paperIndex.get(id);
      if (!existing &&
        (!metadata.title || !metadata.authors || !metadata.primaryTopic || !metadata.published)) {
        const reason =
          `verified detail ${path} cannot safely recreate its missing index entry: ` +
          "frontmatter requires title, authors, primary_topic, and published";
        this.deps.logger.warn(`manual-fetch: ${reason}`);
        this.progress.setError(`Detail index needs repair metadata: ${id}`);
        return { kind: "error", reason };
      }
      const entry = (await paperIndex.reconcileManualDetail({
        arxivId: id,
        title: metadata.title ?? existing?.title ?? id,
        authors: metadata.authors ?? existing?.authors ?? [],
        date: metadata.published ?? dateStr,
        published: metadata.published ?? existing?.published,
        updated: existing?.updated ?? metadata.published,
        arxivCategory: metadata.primaryTopic ?? existing?.category ?? "",
        arxivCategories: existing?.categories ?? (
          metadata.primaryTopic ? [metadata.primaryTopic] : []
        ),
        abstract: existing?.abstract,
        primaryTopic: metadata.primaryTopic ?? existing?.primaryTopic ?? "",
        detail: true,
      }, path, "saved")).entry;
      await this.deps.writer.refreshPaperNoteFrontmatter(entry, path);
      this.progress.setComplete(`Detail note index repaired: ${id}`);
      return { kind: "already_exists", path };
    } catch (e) {
      const reason = `paper index reconciliation failed: ${errorMessage(e)}`;
      this.deps.logger.error(`manual-fetch: ${reason} for ${id}`, e);
      this.progress.setError(`Index repair failed: ${id}`);
      return { kind: "error", reason };
    }
  }
}

function displayDateFromIndexEntry(
  entry: Pick<PaperIndexEntry, "dailyReports" | "published"> | null | undefined,
): string | undefined {
  if (!entry) return undefined;
  return firstDailyReportDate(entry.dailyReports) ?? entry.published;
}

function firstDailyReportDate(paths: string[]): string | undefined {
  const dates = paths
    .map((path) => /(\d{4}-\d{2}-\d{2})\.md$/i.exec(path.trim())?.[1])
    .filter((date): date is string => Boolean(date))
    .sort();
  return dates[0];
}

function conflictReason(
  reason: "identity_mismatch" | "identity_invalid" | "user_content",
  id: string,
): string {
  if (reason === "identity_mismatch") {
    return `existing note has a different arXiv ID; protected ${id} note from overwrite`;
  }
  if (reason === "identity_invalid") {
    return `existing note has ambiguous or invalid arXiv identity; protected ${id} note from overwrite`;
  }
  return `existing note contains user-authored or unverified content; protected ${id} note from overwrite`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error && error.message ? error.message : String(error);
}
