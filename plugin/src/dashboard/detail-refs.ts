import {
  classifyPaperNote,
  normalizeArxivId,
  validateVaultRelativeDirectory,
  type DashboardRow,
} from "@arxiv-daily/core";
import { normalizeVaultPath } from "./files";

/** Indexed detail-summary reference helpers (moved out of view.ts). */
export function collectIndexedDetailSummaryRefs(
  entries: DashboardRow["entry"][],
): { ids: Set<string>; paths: Map<string, string> } {
  const ids = new Set<string>();
  const paths = new Map<string, string>();
  for (const entry of entries) {
    const path = normalizeVaultPath(entry.paperPath ?? "");
    if (!entry.detail || !path) continue;
    ids.add(entry.arxivId);
    paths.set(entry.arxivId, path);
  }
  return { ids, paths };
}

export function expectedDetailSummaryPath(
  papersDir: string,
  rawArxivId: string,
): string | null {
  const canonicalId = normalizeArxivId(rawArxivId);
  const directory = validateVaultRelativeDirectory(papersDir);
  if (!canonicalId || !directory.ok || !directory.value) return null;
  return `${directory.value}/${canonicalId}.md`;
}

export function isExpectedGeneratedDetailSummary(
  markdown: string,
  canonicalArxivId: string,
): boolean {
  return classifyPaperNote(markdown, canonicalArxivId).kind === "verified_detail";
}

export function shouldForceDashboardHistorySyncAfterDetailDeletion(
  trashedFiles: number,
  updatedEntries: number,
): boolean {
  return trashedFiles > 0 || updatedEntries > 0;
}
