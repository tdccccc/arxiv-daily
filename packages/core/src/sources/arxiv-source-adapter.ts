import type { MarkupParser } from "../core/adapters";
import type { Logger } from "../services/logger";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import { formatPaperKey } from "../services/paper-key";
import { modernArxivResources } from "../utils/arxiv";
import {
  formatArxivHttpError,
  isArxivHttpError,
  isArxivRetryDeferredError,
  isRetryableArxivError,
  type ArxivFetcher,
} from "../pipeline/arxiv-fetcher";
import {
  parseRecent,
  type DateBucket,
  type PaperMeta,
} from "../pipeline/arxiv-parser";
import type {
  PaperContentFetcher,
} from "../pipeline/paper-content";
import { isContentFetchFailure } from "../pipeline/content-fetch-failure";
import type {
  NormalizedPaperContent,
  PaperContentQuality,
  PaperContentSection,
  SourceAdapter,
  SourceFetchContentOptions,
  SourceListForDateOptions,
  SourceListForDateResult,
  SourcePaperMeta,
} from "./types";

export interface ArxivSourceAdapterDeps {
  fetcher: ArxivFetcher;
  paperFetcher: PaperContentFetcher;
  markupParser: MarkupParser;
  logger: Logger;
  /** Default arXiv categories when list options omit channels. */
  defaultCategories: string[];
  /** Per-run cap on abs-page network fetches (tests may lower it). */
  absFallbackNetworkLimit?: number;
}

/** Max arxiv.org/abs network fetches per run when the query API left gaps. */
const ABS_FALLBACK_NETWORK_LIMIT = 500;
/** The abs-page fallback gives up after this many failures in a row. */
const ABS_FALLBACK_FAILURE_LIMIT = 3;

type FailureKind = "failed_transient" | "failed_permanent";

/**
 * First real SourceAdapter: arXiv /recent discovery + existing PaperContentFetcher body path.
 */
export class ArxivSourceAdapter implements SourceAdapter {
  readonly sourceId = "arxiv";

  constructor(private deps: ArxivSourceAdapterDeps) {}

  async listForDate(
    dateStr: string,
    options: SourceListForDateOptions = {},
  ): Promise<SourceListForDateResult> {
    const { fetcher, logger } = this.deps;
    const signal = options.signal;
    const categories =
      options.channels && options.channels.length > 0
        ? options.channels
        : this.deps.defaultCategories;
    const byId = new Map<string, SourcePaperMeta>();
    const succeededCategories: string[] = [];
    const awaiting: Array<{ channel: string; reason: string }> = [];
    const failures: Array<{ kind: FailureKind; reason: string }> = [];

    for (const category of categories) {
      throwIfCancelled(signal);
      let recentHtml: string;
      try {
        recentHtml = signal
          ? await fetcher.fetchRecent(category, signal)
          : await fetcher.fetchRecent(category);
      } catch (e) {
        if (isCancellationError(e)) throw e;
        const message = formatArxivHttpError(e);
        failures.push({
          kind: classifyArxivSourceFailure(e),
          reason: `fetch /recent failed for ${category}: ${message}`,
        });
        logger.warn(
          `arxiv-source: fetch /recent failed for ${category}, continuing: ${message}`,
        );
        continue;
      }

      let buckets: DateBucket[];
      try {
        buckets = parseRecent(recentHtml, this.deps.markupParser);
      } catch (e) {
        if (isCancellationError(e)) throw e;
        failures.push({
          kind: "failed_permanent",
          reason: `parse failed for ${category}: ${(e as Error).message}`,
        });
        logger.error(
          `arxiv-source: parse failed for ${category}: ${(e as Error).message}`,
        );
        continue;
      }

      const bucket = buckets.find((b) => b.announceDate === dateStr);
      if (!bucket) {
        const bounds = recentDateBounds(buckets);
        if (bounds && dateStr > bounds.newest) {
          awaiting.push({ channel: category, reason: missingRecentDateReason(dateStr, category, buckets, bounds) });
          continue;
        }
        failures.push({
          // /recent only ever moves forward: a date older than its oldest
          // day can never appear again, so retrying it cannot succeed.
          kind: bounds && dateStr < bounds.oldest ? "failed_permanent" : "failed_transient",
          reason: missingRecentDateReason(dateStr, category, buckets, bounds),
        });
        logger.warn(
          `arxiv-source: ${dateStr} missing in ${category} /recent, continuing`,
        );
        continue;
      }

      succeededCategories.push(category);
      logger.info(
        `arxiv-source: ${bucket.papers.length} papers for ${dateStr} in ${category}`,
      );
      for (const paper of bucket.papers) {
        addSourcePaper(byId, paper, [category], dateStr);
      }
    }

    if (failures.length > 0) {
      logger.warn(
        `arxiv-source: rejecting partial discovery; ${succeededCategories.length}/${categories.length} categories succeeded, ${failures.length} failed`,
      );
      return {
        kind: "error",
        failureKind: collapseFailureKind(failures),
        reason: collapseFailureReason(failures, categories.length),
      };
    }

    if (awaiting.length > 0) {
      return {
        kind: "pending",
        outcome: "awaiting_announcement",
        channels: awaiting.map((entry) => entry.channel),
        reason: awaiting.map((entry) => entry.reason).join("; "),
      };
    }

    const papers = Array.from(byId.values());
    await this.enrichAbstracts(papers, signal);

    return {
      kind: "ok",
      outcome: papers.length === 0 ? "no_updates" : "available",
      papers,
      channels: succeededCategories,
      dateWindow: "recent",
    };
  }

  async fetchContent(
    externalId: string,
    options: SourceFetchContentOptions,
  ): Promise<NormalizedPaperContent> {
    throwIfCancelled(options.signal);
    const resources = modernArxivResources(externalId);
    if (!resources) {
      return {
        abstract: "",
        sections: [],
        quality: "unavailable",
        canonicalUrl: "",
        fullTextFailure: `invalid arXiv id: ${externalId}`,
      };
    }

    const raw = options.signal
      ? await this.deps.paperFetcher.fetch(
          resources.id,
          {
            isDetail: options.wantFullText,
            sectionCharLimit: options.sectionCharLimit,
            paperCharLimit: options.paperCharLimit,
          },
          options.signal,
        )
      : await this.deps.paperFetcher.fetch(resources.id, {
          isDetail: options.wantFullText,
          sectionCharLimit: options.sectionCharLimit,
          paperCharLimit: options.paperCharLimit,
        });

    return mapLegacyPaperContent(raw, resources.absUrl, resources.id);
  }

  private async enrichAbstracts(
    papers: SourcePaperMeta[],
    signal?: AbortSignal,
  ): Promise<void> {
    if (papers.length === 0) return;
    const { fetcher, logger } = this.deps;
    try {
      const ids = papers.map((p) => p.externalId);
      const metadataMap = signal
        ? await fetcher.fetchMetadataByIds(ids, signal)
        : await fetcher.fetchMetadataByIds(ids);
      for (const p of papers) {
        const meta = metadataMap.get(p.externalId);
        if (!meta) continue;
        if (meta.abstract) p.abstract = meta.abstract;
        const updated = dateOnly(meta.updated);
        if (updated) p.updated = updated;
        for (const category of sourceCategories(meta)) {
          if (!p.categories.includes(category)) p.categories.push(category);
        }
      }
      const missing = papers.filter(lacksAbstract).length;
      const summary = `arxiv-source: enriched ${metadataMap.size}/${ids.length} papers via Atom API`;
      if (missing === 0) {
        logger.info(summary);
      } else {
        logger.warn(
          `${summary}; ${missing} still without an abstract, trying arxiv.org/abs pages`,
        );
      }
    } catch (e) {
      if (isCancellationError(e)) throw e;
      logger.warn(
        `arxiv-source: abstract enrichment failed: ${formatArxivHttpError(e)}; trying arxiv.org/abs pages`,
      );
    }
    await this.fillMissingAbstractsFromAbsPages(papers, signal);
  }

  /**
   * Sequential fallback for papers the query API left without an abstract:
   * read each arxiv.org/abs page (cache first). Network fetches are capped per
   * run (cache hits are free) and the loop stops after a few failures in a
   * row, so a blocked or throttled host cannot stall the run. Never throws
   * except for cancellation; whatever stays empty is filtered on title only.
   */
  private async fillMissingAbstractsFromAbsPages(
    papers: SourcePaperMeta[],
    signal?: AbortSignal,
  ): Promise<void> {
    const pending = papers.filter(lacksAbstract);
    if (pending.length === 0) return;
    const { paperFetcher, logger } = this.deps;
    const networkLimit = this.deps.absFallbackNetworkLimit ?? ABS_FALLBACK_NETWORK_LIMIT;
    let networkFetches = 0;
    let consecutiveFailures = 0;
    let lastFailure = "";
    let filled = 0;
    let stop: "cap" | "failures" | undefined;

    for (const paper of pending) {
      throwIfCancelled(signal);
      if (networkFetches >= networkLimit) {
        stop = "cap";
        break;
      }
      try {
        const result = await paperFetcher.fetchListingAbstract(paper.externalId, signal);
        if (result.fromNetwork) networkFetches += 1;
        // A page without an abstract block is a successful fetch, not a failure.
        consecutiveFailures = 0;
        if (result.abstract.trim()) {
          paper.abstract = result.abstract;
          filled += 1;
        }
      } catch (e) {
        if (isCancellationError(e)) throw e;
        // A cooldown refusal never reached arxiv.org, so it spends no budget.
        if (!isArxivRetryDeferredError(e)) networkFetches += 1;
        consecutiveFailures += 1;
        lastFailure = formatArxivHttpError(e);
        if (consecutiveFailures >= ABS_FALLBACK_FAILURE_LIMIT) {
          stop = "failures";
          break;
        }
      }
    }

    const remaining = pending.length - filled;
    const summary = `arxiv-source: abs-page fallback filled ${filled}/${pending.length} missing abstracts`;
    if (remaining === 0) {
      logger.info(summary);
      return;
    }
    const why =
      stop === "cap"
        ? `network fetch cap of ${networkLimit} reached`
        : stop === "failures"
          ? `stopped after ${ABS_FALLBACK_FAILURE_LIMIT} consecutive failures (last: ${lastFailure})`
          : "their abs pages carry no abstract";
    logger.warn(`${summary}; ${remaining} papers remain without an abstract (${why})`);
  }
}

function lacksAbstract(paper: SourcePaperMeta): boolean {
  return paper.abstract.trim().length === 0;
}

/** Map legacy arXiv extractor DTO → normalized content contract. */
export function mapLegacyPaperContent(
  raw: {
    abstractConclusion: string;
    fullSections: string | null;
    fullTextSource?: string;
    fullTextFailure?: string;
  },
  canonicalUrl: string,
  externalId: string,
): NormalizedPaperContent {
  const abstract = extractAbstractText(raw.abstractConclusion);
  const sections = parseSectionsMarkdown(raw.fullSections);
  const quality = deriveQuality(raw, abstract, sections);
  return {
    abstract,
    sections,
    fullTextFallback: raw.fullSections ?? undefined,
    quality,
    canonicalUrl: canonicalUrl || `https://arxiv.org/abs/${externalId}`,
    fullTextSource: raw.fullTextSource,
    fullTextFailure: raw.fullTextFailure,
  };
}

/**
 * Bridge normalized content back to the strings summarizer/pipeline still use.
 * Keeps P2 from rewriting DailyPaperWithContent.
 */
export function legacyContentFromNormalized(
  content: NormalizedPaperContent,
): {
  abstractConclusion: string;
  fullSections: string | null;
  fullTextSource?: "arxiv-html" | "arxiv-source";
  fullTextFailure?: string;
} {
  // The failure placeholder is not an abstract and the failure reasons are not
  // paper text: neither may reach a prompt dressed up as one.
  const abstractConclusion =
    content.abstract.trim().length > 0
      ? content.abstract.startsWith("##") || isContentFetchFailure(content.abstract)
        ? content.abstract
        : `## Abstract\n${content.abstract}`
      : content.fullTextFallback?.slice(0, 2000) || "";
  const fullSections =
    content.fullTextFallback ??
    (content.sections.length > 0
      ? content.sections
          .map((s) => `## ${s.heading}\n${s.text}`.trim())
          .join("\n\n")
      : null);
  const fullTextSource =
    content.fullTextSource === "arxiv-html" ||
    content.fullTextSource === "arxiv-source"
      ? content.fullTextSource
      : undefined;
  return {
    abstractConclusion,
    fullSections,
    fullTextSource,
    fullTextFailure: content.fullTextFailure,
  };
}

/** Convert SourcePaperMeta to the PaperMeta shape filterPapers expects. */
export function paperMetaFromSourcePaper(paper: SourcePaperMeta): PaperMeta & {
  arxivCategories: string[];
  published?: string;
  updated?: string;
} {
  return {
    id: paper.externalId,
    title: paper.title,
    authors: paper.authors,
    abstract: paper.abstract,
    arxivCategories: [...paper.categories],
    published: paper.published,
    updated: paper.updated,
  };
}

function deriveQuality(
  raw: {
    fullSections: string | null;
    fullTextFailure?: string;
    abstractConclusion: string;
  },
  abstract: string,
  sections: PaperContentSection[],
): PaperContentQuality {
  const hasAbstract = abstract.trim() !== "" && !isContentFetchFailure(abstract);
  if (sections.length > 0 || (raw.fullSections && raw.fullSections.trim())) {
    return hasAbstract ? "full" : "partial";
  }
  if (hasAbstract || (raw.abstractConclusion.trim() && !isContentFetchFailure(raw.abstractConclusion))) {
    return raw.fullTextFailure ? "abstract_only" : "partial";
  }
  return "unavailable";
}

function extractAbstractText(abstractConclusion: string): string {
  const text = abstractConclusion.trim();
  if (!text) return "";
  // Common extractor shape: "## Abstract\n...". Scan positions instead of
  // retrying a whitespace-to-end assertion at every character of the body.
  if (!text.startsWith("##")) return text;
  let cursor = 2;
  while (cursor < text.length && /\s/u.test(text[cursor]!)) cursor += 1;
  if (text.slice(cursor, cursor + 8).toLowerCase() !== "abstract") return text;
  cursor += 8;
  let bodyStart = -1;
  while (cursor < text.length && /\s/u.test(text[cursor]!)) {
    if (text[cursor] === "\n") bodyStart = cursor + 1;
    cursor += 1;
  }
  if (bodyStart >= 0) {
    let bodyEnd = text.length;
    let nextHeading = text.indexOf("\n##", bodyStart - 1);
    while (nextHeading >= 0) {
      if (/\s/u.test(text[nextHeading + 3] ?? "")) {
        bodyEnd = Math.max(bodyStart, nextHeading);
        break;
      }
      nextHeading = text.indexOf("\n##", nextHeading + 3);
    }
    const abstract = text.slice(bodyStart, bodyEnd).trim();
    if (abstract) return abstract;
  }
  // Preserve the legacy fallback for an inline heading or an empty section.
  return text.slice(cursor).trim();
}

function parseSectionsMarkdown(
  fullSections: string | null,
): PaperContentSection[] {
  if (!fullSections?.trim()) return [];
  const parts = fullSections.split(/\n(?=##\s)/);
  const sections: PaperContentSection[] = [];
  for (const part of parts) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const match = /^##\s+([^\n]+)\n?([\s\S]*)$/.exec(trimmed);
    if (match) {
      sections.push({
        heading: match[1]!.trim(),
        text: (match[2] ?? "").trim(),
      });
    } else {
      sections.push({ heading: "", text: trimmed });
    }
  }
  return sections;
}

function addSourcePaper(
  byId: Map<string, SourcePaperMeta>,
  paper: PaperMeta,
  categories: string[],
  published: string,
): void {
  const resources = modernArxivResources(paper.id);
  if (!resources) return;
  const existing = byId.get(resources.id);
  if (existing) {
    for (const c of categories) {
      if (!existing.categories.includes(c)) existing.categories.push(c);
    }
    if (!existing.abstract && paper.abstract) existing.abstract = paper.abstract;
    if (!existing.title && paper.title) existing.title = paper.title;
    if (!existing.authors && paper.authors) existing.authors = paper.authors;
    return;
  }
  byId.set(resources.id, {
    paperKey: formatPaperKey("arxiv", resources.id),
    source: "arxiv",
    externalId: resources.id,
    title: paper.title,
    authors: paper.authors,
    abstract: paper.abstract ?? "",
    canonicalUrl: resources.absUrl,
    pdfUrl: resources.pdfUrl,
    categories: [...categories],
    published,
  });
}

function sourceCategories(meta: {
  category?: string;
  categories?: string[];
}): string[] {
  const out: string[] = [];
  for (const c of meta.categories ?? []) {
    const t = c.trim();
    if (t && !out.includes(t)) out.push(t);
  }
  const primary = meta.category?.trim();
  if (primary && !out.includes(primary)) out.unshift(primary);
  return out;
}

function dateOnly(value: string | undefined): string {
  if (!value) return "";
  const m = /^(\d{4}-\d{2}-\d{2})/.exec(value.trim());
  return m?.[1] ?? "";
}

function recentDateBounds(
  buckets: DateBucket[],
): { oldest: string; newest: string } | null {
  const dates = buckets.map((b) => b.announceDate).sort();
  if (dates.length === 0) return null;
  return { oldest: dates[0]!, newest: dates[dates.length - 1]! };
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
  if (bounds && dateStr < bounds.oldest) {
    return (
      `date ${dateStr} is older than the oldest ${category} /recent bucket ` +
      `${bounds.oldest}; arXiv no longer lists it there (have: ${have})`
    );
  }
  return `date ${dateStr} is not in ${category} /recent (have: ${have})`;
}

function classifyArxivSourceFailure(error: unknown): FailureKind {
  if (isArxivRetryDeferredError(error)) return "failed_transient";
  if (!isArxivHttpError(error)) return "failed_transient";
  return isRetryableArxivError(error) ? "failed_transient" : "failed_permanent";
}

function collapseFailureKind(
  failures: Array<{ kind: FailureKind; reason: string }>,
): FailureKind {
  if (failures.length === 0) return "failed_transient";
  if (failures.every((f) => f.kind === "failed_permanent")) {
    return "failed_permanent";
  }
  return "failed_transient";
}

function collapseFailureReason(
  failures: Array<{ kind: FailureKind; reason: string }>,
  categoryCount: number,
): string {
  if (failures.length === 0) {
    return "fetch /recent failed: no arXiv categories succeeded";
  }
  if (failures.length === 1) return failures[0]!.reason;
  const scope = failures.length === categoryCount ? "all" : "some";
  return `${scope} arXiv categories failed: ${failures.map((f) => f.reason).join("; ")}`;
}
