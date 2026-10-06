import type { PipelineResult } from "./pipeline/pipeline";
import type { ManualFetchResult } from "./services/manual-fetch";

export type RunResult = PipelineResult | { kind: "skipped"; reason: string };

export function describeResult(result: RunResult | null | undefined): string {
  if (!result) return "no result";
  if (result.kind === "completed") {
    if (result.outcome === "no_updates") return "no arXiv updates";
    if (result.outcome === "no_matches") return "no matching papers";
    return `done (${result.papersWritten} papers)`;
  }
  if (result.kind === "pending") {
    return `${result.outcome === "awaiting_announcement" ? "awaiting announcement" : "pending"}: ${result.reason}`;
  }
  if (result.kind === "cancelled") return `cancelled: ${result.reason}`;
  if (result.kind === "failed_transient") {
    return `transient: ${result.reason}`;
  }
  if (result.kind === "failed_permanent") {
    return `permanent: ${result.reason}`;
  }
  if (result.kind === "skipped") return `skipped: ${result.reason}`;
  return JSON.stringify(result);
}

export function describeManualResult(
  result: ManualFetchResult | null | undefined,
): string {
  if (!result) return "no result";
  if (result.kind === "done") return `done → ${result.path}`;
  if (result.kind === "already_exists") {
    return `verified detail already exists at ${result.path}`;
  }
  if (result.kind === "note_conflict") {
    return `note conflict at ${result.path}: ${result.reason}`;
  }
  if (result.kind === "not_found") return `not found: ${result.reason}`;
  if (result.kind === "no_html") return `no full text: ${result.reason}`;
  if (result.kind === "error") return `error: ${result.reason}`;
  return JSON.stringify(result);
}

export function describeRunResults(
  results: Array<{ date: string; result: RunResult }>,
): string {
  return results
    .map((entry) => `${entry.date}: ${describeResult(entry.result)}`)
    .join("\n");
}

/** A no-announcement day has not exercised the discovery configuration. */
export function isCompletedDiscovery(entry: { status: string; outcome?: string }): boolean {
  return entry.status === "completed" && entry.outcome !== "no_updates";
}
