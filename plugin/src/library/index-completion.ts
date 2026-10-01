import type { FullTextIndexRunSummary } from "@arxiv-daily/core";
import { isEmbeddingModelDownloadNetworkError } from "../hosts/obsidian/embedding-model";

/**
 * What the catalog that fed a full-text index run looked like, so the
 * completion text can explain a zero-paper result instead of just counting
 * it. Captured by `indexPersonalLibraryFullText` (main.ts) from the catalog it
 * indexed — scanned moments earlier when the library had never been scanned,
 * or read as-is when it had.
 */
export interface FullTextIndexLibraryContext {
  /** Every file the catalog knows about (any status), i.e. what the folder held. */
  totalFiles: number;
  /** Files recognized as arXiv papers; local indexed PDFs can also support directions. */
  readyPapers: number;
  /** Unrecognized or metadata-failed files still indexed as full-text-search-only fallback units. */
  unresolvedFallbackFiles: number;
  /**
   * Files whose arXiv id was known but fetching its metadata failed — the one
   * count a scan can attribute to a network problem rather than "not on
   * arXiv" (unresolved files don't distinguish the two: a failed title search
   * is silently indistinguishable from a paper that genuinely has none).
   */
  metadataFetchFailures: number;
  /** Whether this run scanned the folder itself before indexing (first run, or a stale/missing catalog). */
  scannedBeforeIndexing: boolean;
}

/**
 * "What happened" text for a finished full-text index run, shared by the
 * command palette entry and the settings row's own button (commands.ts,
 * settings/tab.ts) so the two surfaces agree on what a run reports.
 *
 * Per-paper failures never fail the whole run (core's index-orchestration
 * keeps one bad paper from taking the rest down), so a model download that
 * fails on the very first embed call still completes as an ordinary summary
 * with every paper marked `failed` — the generic "N failed" count then hides
 * the one thing a reader can act on. This recognizes that specific failure
 * reason (by the message `embeddingModelDownloadNetworkError` always
 * produces) and reports it plainly instead of the count.
 */
export function describeFullTextIndexCompletion(
  summary: FullTextIndexRunSummary,
  options?: {
    /** Appended only when the run is not the network-download failure (e.g. "Search from the Dashboard."). */
    onCompletionSuffix?: string;
    /** Catalog shape that produced this run; enables the zero-paper and zero-arXiv-paper explanations. */
    libraryContext?: FullTextIndexLibraryContext;
  },
): string {
  const networkFailure = summary.failed > 0
    && summary.outcomes.some(
      (outcome) => outcome.status === "failed" && isEmbeddingModelDownloadNetworkError(outcome.error),
    );
  if (networkFailure) {
    return "couldn't download the embedding model (network problem). Check your internet "
      + "connection and try indexing again; see the developer console for details.";
  }
  const context = options?.libraryContext;
  // Zero indexable units: nothing was even attempted, so the generic counts
  // ("0 indexed, 0 reused, 0 failed") would just look broken. Say why instead.
  if (summary.outcomes.length === 0 && context) {
    if (context.totalFiles === 0) {
      return "no PDFs found in the personal library folder. Add PDFs to the folder, "
        + "then use Scan library before building the index again.";
    }
    return "found files in the personal library folder, but none of them could be read as paper "
      + "text (unsupported file types or unreadable PDFs). Nothing is searchable yet.";
  }
  if (summary.searchablePapers === 0 && summary.failed > 0) {
    return `library indexing failed for ${summary.failed} PDF(s). Nothing is searchable yet. `
      + "The PDF title or abstract could not be read or embedded; see the developer console for details.";
  }
  const refreshed = summary.titlesRefreshed > 0 ? `, ${summary.titlesRefreshed} titles refreshed` : "";
  const suffix = options?.onCompletionSuffix ? ` ${options.onCompletionSuffix}` : "";
  let base = `library index (titles and abstracts) — ${summary.indexed} indexed, ${summary.reused} reused, `
    + `${summary.failed} failed, ${summary.pruned} pruned${refreshed}.${suffix}`;
  // Recognition is optional: local title/abstract evidence can support both
  // retrieval and proposed research directions after authorization.
  if (context?.scannedBeforeIndexing && context.readyPapers === 0 && context.unresolvedFallbackFiles > 0) {
    base += " Search and research directions use the titles and abstracts extracted from these PDFs, "
      + "even when arXiv metadata is unavailable.";
    if (context.metadataFetchFailures > 0) {
      base += " Some arXiv lookups failed, possibly because this device was offline during the scan.";
    }
  }
  return base;
}
