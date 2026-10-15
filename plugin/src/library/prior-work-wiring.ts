import {
  createLibraryPriorWork,
  searchFullTextKnowledgeBase,
  type EmbeddingModel,
  type Logger,
  type LibraryConnectionStatus,
  type LibraryPriorWorkMode,
  type LibraryPriorWorkPort,
  type PersonalLibraryPaperRecord,
  type SearchFullTextKnowledgeBaseInput,
} from "@arxiv-daily/core";

/** What the plugin's stores provide for one library search; hosts build these per lookup. */
export type PriorWorkSearchSource = Pick<
  SearchFullTextKnowledgeBaseInput,
  "store" | "sourceManifest" | "generationStore" | "embedding"
>;

export interface LibraryPriorWorkWiringDeps {
  /** Live connection status; read at wiring time and again before every lookup. */
  getStatus(): LibraryConnectionStatus;
  /** Embedding mode when the daily run starts; local searches lexically, remote hybrid. */
  embeddingMode: "local" | "remote";
  /** Throws unless the remote embedding configuration is complete and authorized. */
  assertRemoteEmbeddingReady(): void;
  loadCatalog(): Promise<{ papers: Readonly<Record<string, PersonalLibraryPaperRecord>> }>;
  openSearchSource(): Promise<PriorWorkSearchSource>;
  /** Defaults to the core search; injectable for tests. */
  searchCore?: typeof searchFullTextKnowledgeBase;
  logger?: Logger;
}

/**
 * The daily pipeline's library port, or null when the daily run must not use
 * the library at all (no library connected, or processing not authorized for
 * the model endpoint the run is using).
 *
 * Consent is re-checked before every lookup, so revoking it or changing the
 * endpoint mid-run stops the remaining lookups. A remote embedding setup that
 * is incomplete makes every lookup unavailable instead of quietly falling back
 * to another basis.
 */
export function wireLibraryPriorWork(deps: LibraryPriorWorkWiringDeps): LibraryPriorWorkPort | null {
  if (deps.getStatus().kind !== "authorized") return null;
  const mode: LibraryPriorWorkMode = deps.embeddingMode === "remote" ? "hybrid" : "lexical";
  const searchCore = deps.searchCore ?? searchFullTextKnowledgeBase;

  // One run, one catalog: loaded lazily, shared by title ranking and the basis join.
  let catalogPromise: ReturnType<LibraryPriorWorkWiringDeps["loadCatalog"]> | null = null;
  const catalog = {
    load() {
      catalogPromise ??= deps.loadCatalog().catch((error: unknown) => {
        catalogPromise = null;
        throw error;
      });
      return catalogPromise;
    },
  };

  // A file: document's title and abstract exist only in the knowledge-base
  // manifest. Reuse the manifest the search that just ran was bound to, so the
  // hits and their records come from one snapshot and nothing is read twice.
  let lastSource: PriorWorkSearchSource | undefined;
  const knowledgeBase = {
    async loadManifest() {
      if (!lastSource) throw new Error("Library search source is not open");
      return lastSource.sourceManifest ?? await lastSource.store.loadManifest();
    },
  };

  return createLibraryPriorWork({
    mode,
    logger: deps.logger,
    catalog,
    knowledgeBase,
    assertAuthorized() {
      if (deps.getStatus().kind !== "authorized") {
        throw new Error("Library processing is not authorized");
      }
      if (mode === "hybrid") deps.assertRemoteEmbeddingReady();
    },
    async search({ queryText, lexicalQueryText, limit, mode: searchMode, signal }) {
      const [source, { papers }] = await Promise.all([deps.openSearchSource(), catalog.load()]);
      lastSource = source;
      const titles = new Map<string, string>();
      for (const [paperKey, paper] of Object.entries(papers)) {
        if (paper.title) titles.set(paperKey, paper.title);
      }
      return await searchCore({
        ...source,
        embedding: searchMode === "lexical" ? embeddingThatNeverEmbeds(source.embedding) : source.embedding,
        queryText,
        lexicalQueryText,
        mode: searchMode,
        limit,
        titles,
        logger: deps.logger,
        signal,
      });
    },
  });
}

/**
 * Lexical search only needs the model's identity (the index was built with
 * it); embedding anything would load a ~130 MB local model and stall the
 * renderer, so make that impossible rather than merely unlikely.
 */
function embeddingThatNeverEmbeds(model: EmbeddingModel): EmbeddingModel {
  return {
    modelId: model.modelId,
    dimension: model.dimension,
    prefixPolicy: model.prefixPolicy,
    embed: () => Promise.reject(new Error("lexical library search does not embed")),
  };
}
