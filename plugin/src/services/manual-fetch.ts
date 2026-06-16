import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import type { PaperContentFetcher } from "../pipeline/paper-content";
import type { MarkdownWriter } from "../pipeline/markdown-writer";
import type { LlmClient } from "../llm/client";
import type { Logger } from "./logger";
import type { ProgressReporter } from "./progress";
import { NoopProgressReporter } from "./progress";
import type { StorageAdapter } from "../core/adapters";
import type { AdvancedSettings, ArxivSettings, LlmSettings, OutputSettings } from "../settings/types";
import { summarizePaperDetail, type DailyPaperWithContent } from "../pipeline/summarizer";
import type { PaperIndexEntry, PaperIndexStore } from "./paper-index";
import { parseAtomPapers, type AtomPaperMeta } from "../pipeline/atom-parser";

export type ManualFetchResult =
  | { kind: "done"; path: string }
  | { kind: "already_exists"; path: string }
  | { kind: "not_found"; reason: string }
  | { kind: "no_html"; reason: string }
  | { kind: "error"; reason: string };

export interface ManualFetchDeps {
  storage: StorageAdapter;
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

const ID_RE = /^(\d{4}\.\d{4,5})(?:v\d+)?$/;

/** Normalize various user inputs (URL, "arXiv:xxx", plain id with/without version) into a base id. */
export function normalizeArxivId(input: string): string | null {
  const trimmed = input.trim();
  if (!trimmed) return null;
  // Strip URL prefix
  const stripped = trimmed
    .replace(/^https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf|html)\//i, "")
    .replace(/^arxiv:\s*/i, "")
    .replace(/\.pdf$/i, "")
    .trim();
  const m = ID_RE.exec(stripped);
  return m ? m[1] : null;
}

export class ManualFetchService {
  private progress: ProgressReporter;

  constructor(private deps: ManualFetchDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
  }

  async fetchAndSummarize(rawId: string, dateStr: string): Promise<ManualFetchResult> {
    const { storage, output, logger } = this.deps;

    const id = normalizeArxivId(rawId);
    if (!id) {
      return { kind: "error", reason: `invalid arXiv id: ${rawId}` };
    }
    this.progress.setTask("arXiv Daily detail", id);
    this.progress.setStage("fetch-metadata");

    // 1. Duplicate check
    const targetPath = storage.normalizePath(`${output.papersDir}/${id}.md`);
    let replaceEmptyExistingNote = false;
    if (await storage.exists(targetPath)) {
      const existing = await storage.readText(targetPath).catch(() => undefined);
      if (typeof existing === "string" && isFrontmatterOnlyNote(existing)) {
        replaceEmptyExistingNote = true;
        logger.warn(
          `manual-fetch: ${id} exists at ${targetPath} but has no markdown body; regenerating`,
        );
      } else {
        logger.info(`manual-fetch: ${id} already exists at ${targetPath}`);
        const entry = await this.syncExistingIndexEntry(id, dateStr, targetPath);
        if (entry) {
          await this.deps.writer.refreshPaperNoteFrontmatter(entry, targetPath);
        }
        this.progress.setComplete(`Detail note already exists: ${id}`);
        return { kind: "already_exists", path: targetPath };
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
      const meta = await this.fetchAtomMetadata(id);
      if (!meta) {
        this.progress.setError(`arXiv has no entry for ${id}`);
        return { kind: "not_found", reason: `arXiv has no entry for ${id}` };
      }
      title = meta.title;
      authors = meta.authors;
      category = meta.primaryCategory || meta.categories[0] || "other";
      published = meta.published;
      updated = meta.updated;
      categories = meta.categories;
      abstract = meta.abstract;
    } catch (e) {
      logger.error(`manual-fetch: atom metadata failed for ${id}`, e);
      this.progress.setError(`Metadata failed: ${(e as Error).message}`);
      return { kind: "error", reason: `atom metadata: ${(e as Error).message}` };
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
        skipSections: this.deps.advanced.skipSections,
        prioritySections: this.deps.advanced.prioritySections,
      });
    } catch (e) {
      logger.error(`manual-fetch: content fetch failed for ${id}`, e);
      this.progress.setError(`Content fetch failed: ${(e as Error).message}`);
      return { kind: "error", reason: `content fetch: ${(e as Error).message}` };
    }
    if (!content.fullSections) {
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
    try {
      this.progress.setStage("summarize-detail");
      summary = await summarizePaperDetail(paper, {
        llm: this.deps.llm,
        logger,
        arxivSettings: this.deps.arxiv,
        advanced: this.deps.advanced,
        llmTemperature: this.deps.llmSettings.temperature,
      });
    } catch (e) {
      logger.error(`manual-fetch: LLM summary failed for ${id}`, e);
      this.progress.setError(`Summary failed: ${(e as Error).message}`);
      return { kind: "error", reason: `LLM summary: ${(e as Error).message}` };
    }

    // 5. Index + write
    this.progress.setStage("write-detail");
    let indexEntry: PaperIndexEntry | undefined;
    if (this.deps.paperIndex) {
      try {
        const existing = await this.deps.paperIndex.get(id);
        const displayDate = displayDateFromIndexEntry(existing);
        const indexed = await this.deps.paperIndex.upsertFromDailyPaper({
          arxivId: id,
          title,
          authors,
          date: dateStr,
          published: displayDate ?? published,
          updated,
          arxivCategory: category,
          arxivCategories: categories,
          primaryTopic: category,
          detail: true,
        });
        indexEntry = indexed.entry;
        if (indexed.wasNew) {
          const saved = await this.deps.paperIndex.setStatus(id, "saved");
          indexEntry = saved ?? indexEntry;
        }
      } catch (e) {
        logger.error(`manual-fetch: paper index update failed for ${id}`, e);
        this.progress.setError(`Index update failed: ${(e as Error).message}`);
        return {
          kind: "error",
          reason: `paper index: ${(e as Error).message}`,
        };
      }
    }

    if (replaceEmptyExistingNote) {
      await storage.remove(targetPath);
    }
    const path = await this.deps.writer.writePaperDetail(
      paper,
      dateStr,
      summary,
      indexEntry,
    );
    if (this.deps.paperIndex) {
      try {
        await this.deps.paperIndex.setPaperPath(id, path);
      } catch (e) {
        logger.error(`manual-fetch: failed to store paperPath for ${id}`, e);
      }
    }
    logger.info(`manual-fetch: wrote ${path}`);
    this.progress.setComplete(`Detail note ready: ${id}`);
    return { kind: "done", path };
  }

  private async syncExistingIndexEntry(
    id: string,
    dateStr: string,
    targetPath: string,
  ): Promise<PaperIndexEntry | undefined> {
    const { paperIndex, logger } = this.deps;
    if (!paperIndex) return undefined;
    try {
      const meta = await this.fetchAtomMetadata(id);
      if (!meta) return undefined;
      const existing = await paperIndex.get(id);
      const displayDate = displayDateFromIndexEntry(existing);
      const indexed = await paperIndex.upsertFromDailyPaper({
        arxivId: id,
        title: meta.title,
        authors: meta.authors,
        date: dateStr,
        published: displayDate ?? meta.published,
        updated: meta.updated,
        arxivCategory: meta.primaryCategory || meta.categories[0] || "other",
        arxivCategories: meta.categories,
        primaryTopic: meta.primaryCategory || meta.categories[0] || "other",
        detail: true,
        paperPath: targetPath,
      });
      if (indexed.wasNew) {
        return (await paperIndex.setStatus(id, "saved")) ?? indexed.entry;
      }
      return indexed.entry;
    } catch (e) {
      logger.warn(
        `manual-fetch: failed to refresh existing index entry for ${id}: ${(e as Error).message}`,
      );
      return undefined;
    }
  }

  /** Returns Atom metadata for one id, or null if not found. */
  private async fetchAtomMetadata(id: string): Promise<AtomPaperMeta | null> {
    const xml = await this.deps.fetcher.fetchAtomEntry(id);
    const paper = parseAtomPapers(xml).find((candidate) => candidate.id === id);
    if (!paper || !paper.title || !paper.abstract) return null;
    const primaryCategory = paper.primaryCategory || paper.categories[0] || "other";
    return {
      ...paper,
      primaryCategory,
      categories: appendUnique(paper.categories, primaryCategory),
    };
  }
}

function appendUnique(values: string[], value: string): string[] {
  return value && !values.includes(value) ? [...values, value] : values;
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

function isFrontmatterOnlyNote(markdown: string): boolean {
  const trimmedStart = markdown.trimStart();
  if (!trimmedStart.startsWith("---")) return markdown.trim().length === 0;
  const match = /^---\s*\n[\s\S]*?\n---\s*(?:\n|$)([\s\S]*)$/.exec(trimmedStart);
  return match ? match[1].trim().length === 0 : markdown.trim().length === 0;
}
