/**
 * Multi-source paper identity and content contracts (initiative P2).
 *
 * Disk note/PDF paths still use short `externalId` stems; map keys use paperKey.
 */

export type PaperContentQuality =
  | "full"
  | "partial"
  | "abstract_only"
  | "unavailable";

/** One logical section of paper body text (normalized across sources). */
export interface PaperContentSection {
  heading: string;
  text: string;
}

/**
 * Normalized full-text (or best-effort) payload returned by SourceAdapter.fetchContent.
 * Distinct from the legacy arXiv extractor DTO in pipeline/paper-content.ts
 * (`abstractConclusion` / `fullSections` strings).
 */
export interface NormalizedPaperContent {
  abstract: string;
  sections: PaperContentSection[];
  /** Unstructured body when sections are unavailable. */
  fullTextFallback?: string;
  quality: PaperContentQuality;
  canonicalUrl: string;
  /** Optional provenance label (e.g. arxiv-html, arxiv-source). */
  fullTextSource?: string;
  fullTextFailure?: string;
}

/** Discovery-time paper metadata (before filter / full-text fetch). */
export interface SourcePaperMeta {
  paperKey: string;
  source: string;
  externalId: string;
  title: string;
  authors: string;
  abstract: string;
  canonicalUrl: string;
  pdfUrl?: string;
  categories: string[];
  published?: string;
  updated?: string;
}

export interface SourceListForDateOptions {
  /** arXiv categories or other source-specific channel ids. */
  channels?: string[];
  signal?: AbortSignal;
}

export type SourceListForDateResult =
  | {
      kind: "ok";
      papers: SourcePaperMeta[];
      /** Channels/categories that contributed papers or empty success. */
      channels: string[];
      dateWindow?: string;
    }
  | {
      kind: "error";
      failureKind: "failed_transient" | "failed_permanent";
      reason: string;
    };

export interface SourceFetchContentOptions {
  /** When true, prefer full-text sections (detail / deep dive). */
  wantFullText: boolean;
  sectionCharLimit: number;
  paperCharLimit: number;
  signal?: AbortSignal;
}

export interface SourceAdapter {
  /** Lowercase source id, e.g. `arxiv`. */
  readonly sourceId: string;

  /**
   * List papers announced/available for a calendar date in the adapter's semantics.
   * For arXiv this is the /recent announce date window.
   */
  listForDate(
    dateStr: string,
    options?: SourceListForDateOptions,
  ): Promise<SourceListForDateResult>;

  /**
   * Fetch normalized body content for a paper in this source.
   * `externalId` is the source-local id (not paperKey).
   */
  fetchContent(
    externalId: string,
    options: SourceFetchContentOptions,
  ): Promise<NormalizedPaperContent>;
}
