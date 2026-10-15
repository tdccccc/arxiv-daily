import {
  FullTextGenerationIndexStore,
  FullTextKnowledgeBaseFileStore,
  PersonalLibraryCatalogStore,
  createLibraryPriorWork,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
  createRemoteEmbeddingModel,
  searchFullTextKnowledgeBase,
  validateEmbeddingConfig,
  type EmbeddingModel,
  type HttpClient,
  type Logger,
  type LibraryPriorWorkPort,
  type StorageAdapter,
} from "@arxiv-daily/core";
import { createNodeLibraryEmbeddingModel } from "@arxiv-daily/node-runtime";
import type { CliRuntimeConfig } from "./config";
import { inspectCliLibraryConnection } from "./library-connection-cmd";

export interface CliLibraryPriorWorkDeps {
  config: CliRuntimeConfig;
  storage: StorageAdapter;
  http: HttpClient;
  logger: Logger;
}

/**
 * The daily report's library port, or undefined when library processing is not
 * authorized for the model endpoint the report uses (no connection, no consent,
 * or consent invalidated). Stores are built directly: the library workflow, its
 * directory handle and a second model client are not needed for read-only lookup.
 */
export function createCliLibraryPriorWork(deps: CliLibraryPriorWorkDeps): LibraryPriorWorkPort | undefined {
  const { config, storage, http, logger } = deps;
  const connection = config.libraryConnection;
  if (!connection || inspectCliLibraryConnection(config).status.kind !== "authorized") return undefined;

  const remote = config.settings.embedding.mode === "remote";
  const scope = createPersonalLibraryScopeFingerprint(connection);
  const identity = createPersonalLibraryIdentificationFingerprint(connection.eligibleExtensions);
  const knowledgeBase = new FullTextKnowledgeBaseFileStore(storage, config.settings.output, scope, identity);
  const generationStore = new FullTextGenerationIndexStore(storage, config.settings.output, scope, identity);
  const catalogStore = new PersonalLibraryCatalogStore(storage, config.settings.output, {
    onWarning: (message, error) => logger.warn(message, error),
  });
  const catalog = { load: () => catalogStore.load(scope, identity) };

  // The local model object only loads its weights on the first embed(), which
  // lexical retrieval never calls; remote hybrid embeds the query text.
  let embedding: EmbeddingModel | undefined;
  const embeddingModel = (): EmbeddingModel => {
    if (embedding) return embedding;
    if (remote) {
      const { baseUrl, apiKey, model, dimension } = config.settings.embedding;
      return embedding = createRemoteEmbeddingModel({ baseUrl, apiKey, model, dimension, http });
    }
    return embedding = createNodeLibraryEmbeddingModel(config.cacheDir);
  };

  return createLibraryPriorWork({
    catalog,
    knowledgeBase,
    mode: remote ? "hybrid" : "lexical",
    logger,
    assertAuthorized() {
      // Re-checked per lookup so consent withdrawn or a changed endpoint stops
      // the run's remaining lookups.
      if (inspectCliLibraryConnection(config).status.kind !== "authorized") {
        throw new Error("Library model processing requires current authorization");
      }
      if (remote && !validateEmbeddingConfig(config.settings).ok) {
        throw new Error("Remote embedding configuration is incomplete");
      }
    },
    async search({ queryText, lexicalQueryText, limit, mode, signal }) {
      const { papers } = await catalog.load();
      return searchFullTextKnowledgeBase({
        store: knowledgeBase, generationStore, embedding: embeddingModel(),
        queryText, lexicalQueryText, mode, limit, maxHitsPerPaper: 1, signal, logger,
        titles: new Map(Object.values(papers).map((paper) => [paper.paperKey, paper.title])),
      });
    },
  });
}
