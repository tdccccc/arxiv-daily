import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { Logger } from "../services/logger";
import {
  LibraryPriorWorkUnavailableError,
  PERSONAL_NOVELTY_MAX_RETRIEVED_BASIS,
  isNoveltyFileBasisKey,
  preparePersonalizedNoveltyInput,
  type LibraryPriorWorkPort,
  type NoveltyRepresentativePaper,
} from "../pipeline/personalized-novelty";
import { limitBm25QueryTerms } from "./fulltext/generation-bm25-index";
import type { FullTextPaperKnowledgeRecord } from "./fulltext/knowledge-base";
import type { PersonalLibraryPaperRecord } from "./personal-library-catalog";

export type LibraryPriorWorkMode = "lexical" | "hybrid";

/** One library lookup as the host's search closure receives it. */
export interface LibraryPriorWorkSearchRequest {
  /** Title and full abstract: what hybrid retrieval embeds. */
  queryText: string;
  /**
   * The same text cut to the lexical index's unique-term limit (title terms
   * first). Pass it as the search's lexicalQueryText so BM25 never throws on a
   * real-length abstract.
   */
  lexicalQueryText: string;
  limit: number;
  mode: LibraryPriorWorkMode;
  signal?: AbortSignal;
}

/**
 * Candidates requested per basis paper wanted. Hits that cannot serve as a
 * basis (not in the catalog, not an arXiv record, outside the DTO bounds) are
 * dropped after the search, so asking for exactly the wanted count would
 * routinely leave the basis short; a real run had file: documents fill more
 * than half of the first six hits. Three times the need, plus one for the
 * query paper itself, keeps the fetch small and bounded (16 for a basis of 5).
 */
const CANDIDATES_PER_BASIS_PAPER = 3;

export interface LibraryPriorWorkDeps {
  /** Host closure over the catalog store and its scope fingerprints. */
  catalog: { load(): Promise<{ papers: Readonly<Record<string, PersonalLibraryPaperRecord>> }> };
  /**
   * Host closure over searchFullTextKnowledgeBase: ranked library papers for a
   * text query, best first. An absent or empty index yields [].
   */
  search(request: LibraryPriorWorkSearchRequest): Promise<ReadonlyArray<{ paperKey: string }>>;
  /**
   * Host closure over the knowledge-base manifest, the only place a `file:`
   * document's extracted title and abstract live. Read lazily, at most once
   * per lookup, and only when a `file:` document is actually considered.
   * Without it `file:` hits are dropped (arXiv-only basis).
   */
  knowledgeBase?: {
    loadManifest(): Promise<{ papers: Readonly<Record<string, FullTextPaperKnowledgeRecord>> }>;
  };
  /** Throws while library processing consent is not valid for the model endpoint. */
  assertAuthorized(): void | Promise<void>;
  mode: LibraryPriorWorkMode;
  logger?: Logger;
}

/**
 * Builds the daily pipeline's library port: the library papers most similar to
 * a digest paper, with the title and abstract the catalog holds for them.
 *
 * No similarity floor is applied (ADR 0014, ADR 0015 §1): the ranked top-N is
 * returned and the model judges comparability. Catalog papers whose metadata
 * cannot satisfy the novelty DTO bounds are dropped, never truncated.
 */
export function createLibraryPriorWork(deps: LibraryPriorWorkDeps): LibraryPriorWorkPort {
  return {
    async findSimilar(query, limit, signal) {
      throwIfCancelled(signal);
      try {
        await deps.assertAuthorized();
      } catch (error) {
        if (isCancellationError(error)) throw error;
        throw new LibraryPriorWorkUnavailableError("library processing is not authorized");
      }
      const wanted = Math.min(limit, PERSONAL_NOVELTY_MAX_RETRIEVED_BASIS);
      if (wanted < 1) return [];
      // The extra one covers the query paper appearing in its own results.
      const matches = await deps.search({
        queryText: `${query.title}\n${query.abstract}`,
        lexicalQueryText: limitBm25QueryTerms([query.title, query.abstract]),
        limit: wanted * CANDIDATES_PER_BASIS_PAPER + 1,
        mode: deps.mode,
        signal,
      });
      throwIfCancelled(signal);
      if (matches.length === 0) return [];
      const { papers } = await deps.catalog.load();
      throwIfCancelled(signal);
      const basis: NoveltyRepresentativePaper[] = [];
      const seen = new Set<string>([query.paperKey]);
      const queryTitle = normalizeTitle(query.title);
      let manifest: ReturnType<NonNullable<LibraryPriorWorkDeps["knowledgeBase"]>["loadManifest"]> | undefined;
      for (const { paperKey } of matches) {
        if (basis.length === wanted) break;
        if (seen.has(paperKey)) continue;
        seen.add(paperKey);
        let candidate: NoveltyRepresentativePaper | null;
        if (isNoveltyFileBasisKey(paperKey)) {
          if (!deps.knowledgeBase) {
            candidate = null;
          } else {
            manifest ??= deps.knowledgeBase.loadManifest();
            const loaded = await manifest;
            throwIfCancelled(signal);
            const record = Object.prototype.hasOwnProperty.call(loaded.papers, paperKey)
              ? loaded.papers[paperKey]
              : undefined;
            candidate = record ? fileBasisPaper(record) : null;
            // A digest paper whose PDF is in the library under an unrecognised
            // filename is a file: document and would be its own top hit.
            if (candidate && normalizeTitle(candidate.title) === queryTitle) candidate = null;
          }
        } else {
          const catalogPaper = Object.prototype.hasOwnProperty.call(papers, paperKey)
            ? papers[paperKey]
            : undefined;
          candidate = catalogPaper ? basisPaper(catalogPaper) : null;
        }
        if (candidate) {
          basis.push(candidate);
        } else {
          deps.logger?.debug(`library prior work: ${paperKey} has no usable title and abstract`);
        }
      }
      return basis;
    },
  };
}

/** arXiv catalog paper as a novelty basis, or null when it does not fit the bounded contract. */
function basisPaper(paper: PersonalLibraryPaperRecord): NoveltyRepresentativePaper | null {
  if (paper.source !== "arxiv") return null;
  const candidate: NoveltyRepresentativePaper = {
    paperKey: paper.paperKey,
    title: paper.title,
    authors: [...paper.authors],
    abstract: paper.abstract,
    published: paper.published,
    categories: [...paper.categories],
  };
  try {
    preparePersonalizedNoveltyInput({ papers: [], representatives: [candidate] });
  } catch {
    return null;
  }
  return candidate;
}

/**
 * A ready knowledge-base file document as a novelty basis (title and abstract
 * only), or null when the extraction is missing or outside the DTO bounds.
 */
function fileBasisPaper(record: FullTextPaperKnowledgeRecord): NoveltyRepresentativePaper | null {
  if (record.status !== "ready" || !record.title || !record.abstract) return null;
  const candidate: NoveltyRepresentativePaper = {
    paperKey: record.paperKey,
    title: record.title,
    abstract: record.abstract,
  };
  try {
    preparePersonalizedNoveltyInput({ papers: [], representatives: [candidate] });
  } catch {
    return null;
  }
  return candidate;
}

/** Case, punctuation and spacing insensitive title identity. */
function normalizeTitle(title: string): string {
  return title.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}
