import type { FullTextIndexRunSummary } from "@arxiv-daily/core";
import { isEmbeddingModelDownloadNetworkError } from "../hosts/obsidian/embedding-model";

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
  const refreshed = summary.titlesRefreshed > 0 ? `, ${summary.titlesRefreshed} titles refreshed` : "";
  const suffix = options?.onCompletionSuffix ? ` ${options.onCompletionSuffix}` : "";
  return `full-text index — ${summary.indexed} indexed, ${summary.reused} reused, `
    + `${summary.failed} failed, ${summary.pruned} pruned${refreshed}.${suffix}`;
}
