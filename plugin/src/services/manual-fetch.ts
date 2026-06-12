import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import type { PaperContentFetcher } from "../pipeline/paper-content";
import type { MarkdownWriter } from "../pipeline/markdown-writer";
import type { LlmClient } from "../llm/client";
import type { Logger } from "./logger";
import type { Vault } from "obsidian";
import { normalizePath } from "obsidian";
import type { AdvancedSettings, ArxivSettings, LlmSettings, OutputSettings } from "../settings/types";
import { summarizePaperDetail, type DailyPaperWithContent } from "../pipeline/summarizer";
import type { PaperIndexEntry, PaperIndexStore } from "./paper-index";

export type ManualFetchResult =
  | { kind: "done"; path: string }
  | { kind: "already_exists"; path: string }
  | { kind: "not_found"; reason: string }
  | { kind: "no_html"; reason: string }
  | { kind: "error"; reason: string };

export interface ManualFetchDeps {
  vault: Vault;
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
  constructor(private deps: ManualFetchDeps) {}

  async fetchAndSummarize(rawId: string, dateStr: string): Promise<ManualFetchResult> {
    const { vault, output, logger } = this.deps;

    const id = normalizeArxivId(rawId);
    if (!id) {
      return { kind: "error", reason: `invalid arXiv id: ${rawId}` };
    }

    // 1. Duplicate check
    const targetPath = normalizePath(`${output.papersDir}/${id}.md`);
    if (await vault.adapter.exists(targetPath)) {
      logger.info(`manual-fetch: ${id} already exists at ${targetPath}`);
      return { kind: "already_exists", path: targetPath };
    }

    // 2. Pull metadata + abstract from Atom API
    let title = "";
    let authors = "Unknown";
    let category = "other";
    let abstract = "";
    try {
      const meta = await this.fetchAtomMetadata(id);
      if (!meta) {
        return { kind: "not_found", reason: `arXiv has no entry for ${id}` };
      }
      title = meta.title;
      authors = meta.authors;
      category = meta.primaryCategory;
      abstract = meta.abstract;
    } catch (e) {
      logger.error(`manual-fetch: atom metadata failed for ${id}`, e);
      return { kind: "error", reason: `atom metadata: ${(e as Error).message}` };
    }

    // 3. Pull /html and extract full sections
    let content: { abstractConclusion: string; fullSections: string | null };
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
      return { kind: "error", reason: `content fetch: ${(e as Error).message}` };
    }
    if (!content.fullSections) {
      return {
        kind: "no_html",
        reason: `no rendered HTML for ${id}; cannot produce a detail summary`,
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
    };
    let summary: string;
    try {
      summary = await summarizePaperDetail(paper, {
        llm: this.deps.llm,
        logger,
        arxivSettings: this.deps.arxiv,
        advanced: this.deps.advanced,
        llmTemperature: this.deps.llmSettings.temperature,
      });
    } catch (e) {
      logger.error(`manual-fetch: LLM summary failed for ${id}`, e);
      return { kind: "error", reason: `LLM summary: ${(e as Error).message}` };
    }

    // 5. Index + write
    let indexEntry: PaperIndexEntry | undefined;
    if (this.deps.paperIndex) {
      try {
        const indexed = await this.deps.paperIndex.upsertFromDailyPaper({
          arxivId: id,
          title,
          authors,
          date: dateStr,
          arxivCategory: category,
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
        return {
          kind: "error",
          reason: `paper index: ${(e as Error).message}`,
        };
      }
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
    return { kind: "done", path };
  }

  /** Returns title/authors/primary_category/abstract for one id, or null if not found. */
  private async fetchAtomMetadata(id: string): Promise<{
    title: string;
    authors: string;
    primaryCategory: string;
    abstract: string;
  } | null> {
    const xml = await this.deps.fetcher.fetchAtomEntry(id);
    const doc = new DOMParser().parseFromString(xml, "application/xml");
    const entry = doc.querySelector("entry");
    if (!entry) return null;
    const titleEl = entry.querySelector("title");
    const summaryEl = entry.querySelector("summary");
    const authorEls = Array.from(entry.querySelectorAll("author > name"));
    const primaryEl = entry.querySelector("primary_category, *|primary_category");
    const titleText = (titleEl?.textContent ?? "").replace(/\s+/g, " ").trim();
    const abstract = (summaryEl?.textContent ?? "").replace(/\s+/g, " ").trim();
    if (!titleText || !abstract) return null;
    const authorNames = authorEls.map((n) => (n.textContent ?? "").trim()).filter(Boolean);
    const authors =
      authorNames.length === 0
        ? "Unknown"
        : authorNames.length === 1
        ? authorNames[0]
        : `${authorNames[0]} et al.`;
    const primaryCategory =
      primaryEl?.getAttribute("term") ?? "other";
    return { title: titleText, authors, primaryCategory, abstract };
  }
}
