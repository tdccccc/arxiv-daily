import { comparePaperKeysByRecency, identifyingTitleKey } from "./clustering/paper-vector";
import type { FullTextKnowledgeBaseManifest } from "./fulltext/knowledge-base";
import type { PersonalLibraryCatalog } from "./personal-library-catalog";
import {
  PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS,
  accountedPaperKeysOf,
  type PersonalLibraryDirectionProposal,
} from "./personal-library-interest-profile";

/**
 * Unaccounted library papers needed before the researcher is nudged to update
 * direction suggestions (ADR 0015 section 3). Carried over from the buffer-pool
 * trigger of ADR 0007: a count chosen so that a single added paper does not
 * prompt, not a measured similarity threshold.
 */
export const LIBRARY_CHANGE_NUDGE_COUNT = 3 as const;

export function shouldNudgeLibraryChange(unaccountedCount: number): boolean {
  return unaccountedCount >= LIBRARY_CHANGE_NUDGE_COUNT;
}

/**
 * A knowledge-base paper in the order the first scan considers papers, with
 * the papers a scan would have merged away already removed. Only manifest
 * records are read: no paper file, no vector, no model.
 */
export interface LibraryScanCandidate {
  paperKey: string;
  /** Other files that share this paper's identifying title (merged by a scan). */
  mergedPaperKeys: string[];
}

/**
 * The papers the first scan's input selection would consider today: ready
 * papers newest first, same-title files collapsed to the first, at most
 * `PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS` kept. This mirrors
 * `buildClusteringInput` from manifest facts alone; a ready paper with no
 * chunks is skipped there and here.
 */
export function selectLibraryScanCandidates(manifest: FullTextKnowledgeBaseManifest): LibraryScanCandidate[] {
  const candidates: LibraryScanCandidate[] = [];
  const byTitle = new Map<string, LibraryScanCandidate>();
  for (const paperKey of Object.keys(manifest.papers).sort(comparePaperKeysByRecency)) {
    if (candidates.length >= PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS) break;
    const record = manifest.papers[paperKey];
    if (!record || record.status !== "ready" || !(record.chunkCount > 0)) continue;
    const title = identifyingTitleKey(record.title);
    if (title !== null) {
      const kept = byTitle.get(title);
      if (kept) {
        kept.mergedPaperKeys.push(paperKey);
        continue;
      }
      const candidate = { paperKey, mergedPaperKeys: [] };
      byTitle.set(title, candidate);
      candidates.push(candidate);
      continue;
    }
    candidates.push({ paperKey, mergedPaperKeys: [] });
  }
  return candidates;
}

/**
 * Library papers that no scan has accounted for, newest first; null when there
 * is no proposal (the first scan is the first-run flow, not a change). A
 * same-title file counts as accounted when any file of that title is, so a
 * re-download that now sorts ahead of the copy the scan kept does not look new.
 *
 * With `catalog`, papers a scan could not use are left out: a scan needs each
 * paper's metadata from the catalog or, for a file the catalog does not know,
 * an indexed title. Without it such a paper would be counted and then fail the
 * update the count points to. Callers that can read the catalog should pass it.
 */
export function listUnaccountedLibraryPapers(input: {
  manifest: FullTextKnowledgeBaseManifest;
  proposal: Pick<PersonalLibraryDirectionProposal, "catalogInputPapers" | "accountedPaperKeys"> | null;
  catalog?: Pick<PersonalLibraryCatalog, "papers">;
}): string[] | null {
  if (!input.proposal) return null;
  const accounted = new Set(accountedPaperKeysOf(input.proposal));
  const { catalog, manifest } = input;
  return selectLibraryScanCandidates(manifest)
    .filter(({ paperKey, mergedPaperKeys }) =>
      !accounted.has(paperKey) && !mergedPaperKeys.some((key) => accounted.has(key)))
    .filter(({ paperKey }) => !catalog || catalog.papers[paperKey] !== undefined || Boolean(manifest.papers[paperKey]?.title))
    .map(({ paperKey }) => paperKey);
}
