import type { PipelineResult } from "./pipeline/pipeline";
import type { ManualFetchResult } from "./services/manual-fetch";

export type RunResult = PipelineResult | { kind: "skipped"; reason: string };

export function describeResult(result: RunResult | null | undefined): string {
  if (!result) return "no result";
  if (result.kind === "completed") {
    return `done (${result.papersWritten} papers)`;
  }
  if (result.kind === "pending") return `pending: ${result.reason}`;
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
    return `already exists at ${result.path}`;
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
