/**
 * Full-text knowledge base indexing orchestration (core, host-neutral).
 *
 * Hosts (Obsidian plugin today, CLI later) provide the PDF text extractor and
 * the embedding model; core decides what to index, in what order, and how to
 * persist it. The knowledge base is a bypass store: per-paper documents are
 * derived data saved before the CAS-protected manifest, which is the
 * authoritative index.
 *
 * Incremental policy: a paper is reused when its catalog observation
 * fingerprints (path + size + mtime, per file) and the embedding model are
 * unchanged. Any change re-extracts, re-chunks, re-embeds and rewrites the
 * paper document; records for papers that left the catalog are pruned.
 * Failures are recorded per paper (status `failed`) and retried on the next
 * run; one bad paper never fails the whole index run.
 */

import { throwIfCancelled } from "../../services/cancellation";
import type { Logger } from "../../services/logger";
import { sha256Hex } from "../../utils/digest";
import type { PersonalLibraryCatalog } from "../personal-library-catalog";
import type { ScopedLibrarySource } from "../scoped-library-source";
import { chunkFullText } from "./chunking";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  type FullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBaseStore,
  type FullTextPaperDocument,
  type FullTextPaperKnowledgeRecord,
} from "./knowledge-base";
import type { EmbeddingModel, PdfTextExtractor } from "./ports";
import { applyEmbeddingPrefix } from "./ports";
import { searchKnowledgeBase, type KnowledgeBasePaperMatch } from "./retrieval";
import { lexicalTitleSimilarity } from "./title-similarity";
import { extractTitleFromFirstPage } from "./title-extraction";
import {
  significantQueryTokens,
  computeTokenHitScores,
  isKeywordQuery,
} from "./lexical-search";

export interface FullTextIndexPaperOutcome {
  paperKey: string;
  status: "indexed" | "reused" | "failed";
  chunkCount?: number;
  error?: string;
}

export interface FullTextIndexRunSummary {
  indexed: number;
  reused: number;
  /** Fallback papers whose extracted title was refreshed on reuse (no re-embedding). */
  titlesRefreshed: number;
  failed: number;
  pruned: number;
  outcomes: readonly FullTextIndexPaperOutcome[];
  manifestRevision: number;
}

interface IndexUnit {
  paperKey: string;
  label: string;
  filePaths: string[];
  /** Missing entries surface as undefined and fail the unit in the main loop. */
  observationFingerprints: Array<string | undefined>;
  fallback: boolean;
  contentHash?: string;
  migrationSourceKeys: string[];
  preparationError?: string;
}

export interface IndexPersonalLibraryFullTextInput {
  /** Ready papers with per-file observation fingerprints (the change signal). */
  catalog: PersonalLibraryCatalog;
  /** Host file access for PDF bytes. */
  source: ScopedLibrarySource;
  extractor: PdfTextExtractor;
  embedding: EmbeddingModel;
  store: FullTextKnowledgeBaseStore;
  logger?: Logger;
  /** Called with a short detail string before each paper. */
  onProgress?: (detail: string) => void;
  now?: () => Date;
  signal?: AbortSignal;
}

export async function indexPersonalLibraryFullText(
  input: IndexPersonalLibraryFullTextInput,
): Promise<FullTextIndexRunSummary> {
  throwIfCancelled(input.signal);
  const { catalog, source, extractor, embedding, store } = input;
  const log = input.logger;
  const nowIso = (input.now ?? (() => new Date()))().toISOString();

  const loaded = await store.loadManifest();
  if (catalog.scopeFingerprint !== loaded.scopeFingerprint
    || catalog.identificationFingerprint !== loaded.identificationFingerprint) {
    throw new Error(
      "full-text knowledge base store fingerprints do not match the catalog; "
      + "the store was bound to a different library scope",
    );
  }
  const hasPapers = Object.keys(loaded.papers).length > 0;
  if (hasPapers && loaded.modelId !== embedding.modelId) {
    throw new Error(
      `full-text knowledge base was built with model ${loaded.modelId || "(unknown)"} but the current `
      + `model is ${embedding.modelId}; delete the knowledge base and re-index (rebuild) before switching models`,
    );
  }

  const papers: Record<string, FullTextPaperKnowledgeRecord> = { ...loaded.papers };
  const next: FullTextKnowledgeBaseManifest = {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    revision: loaded.revision,
    scopeFingerprint: loaded.scopeFingerprint,
    identificationFingerprint: loaded.identificationFingerprint,
    modelId: embedding.modelId,
    dimension: embedding.dimension,
    updatedAt: loaded.updatedAt,
    papers,
  };

  const outcomes: FullTextIndexPaperOutcome[] = [];
  let titlesRefreshed = 0;
  const units = await collectIndexUnits({
    catalog,
    source,
    loaded,
    onProgress: input.onProgress,
    signal: input.signal,
  });
  const migrationSourceKeys = new Set(units.flatMap((unit) => unit.migrationSourceKeys));
  const completedMigrationKeys = new Set<string>();
  const total = units.length;
  const progressStartedAt = Date.now();

  for (let position = 0; position < total; position += 1) {
    throwIfCancelled(input.signal);
    const unit = units[position]!;
    const { paperKey } = unit;
    input.onProgress?.(indexProgressDetail(unit.label, position + 1, total, progressStartedAt));

    const fingerprints = unit.observationFingerprints;
    if (fingerprints.some((fingerprint) => fingerprint === undefined)) {
      outcomes.push(recordFailed(
        paperKey,
        "catalog file record missing",
        nowIso,
        papers,
        embedding.modelId,
        embedding.dimension,
        unit,
      ));
      continue;
    }
    const observationFingerprints = fingerprints as string[];
    const previous = papers[paperKey];
    const exactReady = previous
      && previous.status === "ready"
      && previous.modelId === embedding.modelId
      && sameFingerprints(previous, observationFingerprints);

    if (exactReady && (!unit.fallback || previous.titleVersion === TITLE_EXTRACTION_VERSION)) {
      outcomes.push({ paperKey, status: "reused" });
      continue;
    }

    // Content-addressed fallback papers can retain their chunks/vectors when a
    // path observation changes (for example, a rename) or when a legacy
    // observation-key document is migrated to its PDF-byte hash key.
    if (unit.fallback) {
      const reuseSourceKey = exactReady
        ? paperKey
        : previous?.status === "ready"
          && previous.modelId === embedding.modelId
          && unit.contentHash !== undefined
          && previous.contentHash === unit.contentHash
          ? paperKey
          : unit.migrationSourceKeys.find((candidate) => {
            const record = papers[candidate];
            return record?.status === "ready" && record.modelId === embedding.modelId;
          });
      if (reuseSourceKey) {
        const sourceRecord = papers[reuseSourceKey]!;
        const existing = await store.loadPaper(reuseSourceKey);
        if (existing) {
          const refreshTitle = sourceRecord.titleVersion !== TITLE_EXTRACTION_VERSION;
          let rebound: FullTextPaperDocument;
          let titleRefreshed = false;
          try {
            rebound = await rebindFallbackDocument({
              document: existing,
              paperKey,
              filePaths: unit.filePaths,
              observationFingerprints,
              contentHash: unit.contentHash,
              refreshTitle,
              source,
              extractor,
              nowIso,
              signal: input.signal,
            });
            titleRefreshed = refreshTitle;
          } catch (caught) {
            if (isCancellationErrorLike(caught, input.signal)) throw caught;
            log?.warn(
              `fulltext: fallback title refresh failed for ${paperKey}, keeping previous title: ${describeRefreshError(caught)}`,
            );
            rebound = await rebindFallbackDocument({
              document: existing,
              paperKey,
              filePaths: unit.filePaths,
              observationFingerprints,
              contentHash: unit.contentHash,
              refreshTitle: false,
              source,
              extractor,
              nowIso,
              signal: input.signal,
            });
          }
          await store.savePaper(rebound);
          papers[paperKey] = recordFromDocument(rebound, nowIso);
          for (const sourceKey of unit.migrationSourceKeys) {
            completedMigrationKeys.add(sourceKey);
          }
          outcomes.push({ paperKey, status: "reused" });
          if (titleRefreshed) {
            log?.info(`fulltext: refreshed fallback title for ${paperKey}`);
            titlesRefreshed += 1;
          }
          continue;
        }
      }
    }

    if (unit.preparationError) {
      outcomes.push(recordFailed(
        paperKey,
        unit.preparationError,
        nowIso,
        papers,
        embedding.modelId,
        embedding.dimension,
        unit,
      ));
      continue;
    }

    try {
      const document = await buildPaperDocument({
        paperKey,
        filePaths: unit.filePaths,
        observationFingerprints,
        contentHash: unit.contentHash,
        extractTitle: unit.fallback,
        source,
        extractor,
        embedding,
        nowIso,
        signal: input.signal,
      });
      await store.savePaper(document);
      papers[paperKey] = recordFromDocument(document, nowIso);
      for (const sourceKey of unit.migrationSourceKeys) completedMigrationKeys.add(sourceKey);
      outcomes.push({ paperKey, status: "indexed", chunkCount: document.chunks.length });
      log?.info(`fulltext: indexed ${paperKey} (${document.chunks.length} chunks)`);
    } catch (caught) {
      if (isCancellationErrorLike(caught, input.signal)) throw caught;
      const message = caught instanceof Error ? caught.message : String(caught);
      log?.warn(`fulltext: indexing failed for ${paperKey}: ${message}`);
      outcomes.push(recordFailed(
        paperKey,
        message,
        nowIso,
        papers,
        embedding.modelId,
        embedding.dimension,
        unit,
      ));
    }
    // Yield between papers so a long index run does not freeze host UIs (the
    // Obsidian renderer processes queued events between papers); harmless on
    // Node hosts.
    await yieldToEventLoop();
  }

  // Prune papers that left the catalog. A legacy source document is retained
  // if its migration failed, so a transient read/write error cannot destroy a
  // previously usable index entry.
  const validKeys = new Set(units.map((unit) => unit.paperKey));
  let pruned = 0;
  for (const paperKey of Object.keys(papers)) {
    throwIfCancelled(input.signal);
    if (validKeys.has(paperKey)) continue;
    if (migrationSourceKeys.has(paperKey) && !completedMigrationKeys.has(paperKey)) continue;
    await store.removePaper(paperKey);
    delete papers[paperKey];
    if (!completedMigrationKeys.has(paperKey)) pruned += 1;
  }

  const saved = await store.replaceManifest(next, loaded.revision);
  log?.info(
    `fulltext: index run complete (revision ${saved.revision}, indexed ${outcomes.filter((o) => o.status === "indexed").length}, `
    + `reused ${outcomes.filter((o) => o.status === "reused").length}, failed ${outcomes.filter((o) => o.status === "failed").length}, pruned ${pruned})`,
  );
  return {
    indexed: outcomes.filter((o) => o.status === "indexed").length,
    reused: outcomes.filter((o) => o.status === "reused").length,
    titlesRefreshed,
    failed: outcomes.filter((o) => o.status === "failed").length,
    pruned,
    outcomes,
    manifestRevision: saved.revision,
  };
}

/**
 * Version of the first-page title extraction rules. Bumped when the rules
 * change so previously indexed fallback papers refresh their titles on the
 * next index run (reuse detects `titleVersion` mismatch; the refresh re-reads
 * the first page and updates the title without re-embedding).
 */
export const TITLE_EXTRACTION_VERSION = 8 as const;

/**
 * Index units: catalog papers plus unresolved files keyed by SHA-256 of the
 * source PDF bytes. Existing content-addressed records reuse their stored hash
 * while the catalog observation is unchanged; a rename/change reads the file
 * once to recover the stable key. Legacy observation-key records are exposed
 * as migration sources so their vectors can be re-keyed without re-embedding.
 */
async function collectIndexUnits(input: {
  catalog: PersonalLibraryCatalog;
  source: ScopedLibrarySource;
  loaded: FullTextKnowledgeBaseManifest;
  onProgress?: (detail: string) => void;
  signal?: AbortSignal;
}): Promise<IndexUnit[]> {
  const units: IndexUnit[] = [];
  for (const paperKey of Object.keys(input.catalog.papers).sort()) {
    const paper = input.catalog.papers[paperKey]!;
    units.push({
      paperKey,
      label: paperKey,
      filePaths: [...paper.filePaths],
      observationFingerprints: paper.filePaths.map(
        (path) => input.catalog.files[path]?.observationFingerprint,
      ),
      fallback: false,
      migrationSourceKeys: [],
    });
  }

  const fallbackRecords = Object.entries(input.loaded.papers)
    .filter(([paperKey]) => paperKey.startsWith("file:"));
  const unresolved = Object.entries(input.catalog.files)
    .filter(([, record]) => record.status === "unresolved")
    .sort(([left], [right]) => left.localeCompare(right));
  const byPaperKey = new Map<string, IndexUnit>();

  for (let position = 0; position < unresolved.length; position += 1) {
    throwIfCancelled(input.signal);
    const [path, record] = unresolved[position]!;
    const observationFingerprint = record.observationFingerprint;
    const legacyKey = `file:${observationFingerprint}`;
    const unchanged = fallbackRecords.find(([, candidate]) => {
      if (candidate.status !== "ready" || candidate.contentHash === undefined) return false;
      const pathIndex = candidate.filePaths.indexOf(path);
      return pathIndex >= 0
        && candidate.observationFingerprints[pathIndex] === observationFingerprint;
    });

    let contentHash = unchanged?.[1].contentHash;
    let preparationError: string | undefined;
    if (!contentHash) {
      input.onProgress?.(
        `Preparing local document ${position + 1}/${unresolved.length}: ${path}`,
      );
      try {
        const buffer = await input.source.readBinary(path, { signal: input.signal });
        contentHash = `sha256:${sha256Hex(new Uint8Array(buffer))}`;
      } catch (caught) {
        if (isCancellationErrorLike(caught, input.signal)) throw caught;
        preparationError = caught instanceof Error ? caught.message : String(caught);
      }
    }

    const paperKey = contentHash ? `file:${contentHash}` : unchanged?.[0] ?? legacyKey;
    const migrationSourceKeys = fallbackRecords
      .filter(([candidateKey, candidate]) => (
        candidateKey !== paperKey
        && (candidateKey === legacyKey
          || (contentHash !== undefined && candidate.contentHash === contentHash))
      ))
      .map(([candidateKey]) => candidateKey);
    const current = byPaperKey.get(paperKey);
    if (current) {
      current.filePaths.push(path);
      current.observationFingerprints.push(observationFingerprint);
      current.migrationSourceKeys.push(...migrationSourceKeys);
      current.preparationError ??= preparationError;
      if (!unchanged) await yieldToEventLoop();
      continue;
    }
    byPaperKey.set(paperKey, {
      paperKey,
      label: path,
      filePaths: [path],
      observationFingerprints: [observationFingerprint],
      fallback: true,
      contentHash,
      migrationSourceKeys,
      preparationError,
    });
    if (!unchanged) await yieldToEventLoop();
  }

  for (const unit of byPaperKey.values()) {
    const paired = unit.filePaths.map((path, index) => ({
      path,
      fingerprint: unit.observationFingerprints[index],
    })).sort((left, right) => left.path.localeCompare(right.path));
    unit.filePaths = paired.map(({ path }) => path);
    unit.observationFingerprints = paired.map(({ fingerprint }) => fingerprint);
    unit.migrationSourceKeys = [...new Set(unit.migrationSourceKeys)].sort();
    units.push(unit);
  }
  return units.sort((left, right) => left.paperKey.localeCompare(right.paperKey));
}

async function buildPaperDocument(input: {
  paperKey: string;
  filePaths: readonly string[];
  observationFingerprints: readonly string[];
  contentHash?: string;
  /** Extract a title from the first page (fallback papers have no catalog metadata). */
  extractTitle?: boolean;
  source: ScopedLibrarySource;
  extractor: PdfTextExtractor;
  embedding: EmbeddingModel;
  nowIso: string;
  signal?: AbortSignal;
}): Promise<FullTextPaperDocument> {
  const { paperKey, filePaths, observationFingerprints } = input;
  if (filePaths.length === 0) {
    throw new Error("paper has no file paths to index");
  }
  const bytes = await input.source.readBinary(filePaths[0]!, { signal: input.signal });
  const extraction = await input.extractor.extractPdfText(
    new Uint8Array(bytes),
    { signal: input.signal },
  );
  const pages = extraction.pages;
  const textHash = `sha256:${sha256Hex(pages.join("\n"))}`;
  const chunks = chunkFullText(pages);
  const vectors = await input.embedding.embed(
    chunks.map((chunk) => prefixFor("passage", input.embedding, chunk.text)),
    { signal: input.signal },
  );
  if (vectors.length !== chunks.length) {
    throw new Error(
      `embedding model returned ${vectors.length} vectors for ${chunks.length} chunks`,
    );
  }
  const dimension = input.embedding.dimension;
  if (vectors.some((vector) => vector.length !== dimension)) {
    throw new Error(`embedding model returned a vector of unexpected dimension`);
  }
  const flat = new Float32Array(chunks.length * dimension);
  vectors.forEach((vector, index) => flat.set(vector, index * dimension));
  const title = input.extractTitle
    ? extractTitleFromFirstPage(pages, extraction.layout, extraction.metadataTitle) ?? undefined
    : undefined;
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey,
    modelId: input.embedding.modelId,
    dimension,
    textHash,
    contentHash: input.contentHash,
    title,
    titleVersion: input.extractTitle ? TITLE_EXTRACTION_VERSION : undefined,
    filePaths: [...filePaths],
    observationFingerprints: [...observationFingerprints],
    chunks,
    vectors: flat,
    updatedAt: input.nowIso,
  };
}

/** Re-key/rebind a fallback document while preserving chunks and vectors. */
async function rebindFallbackDocument(input: {
  document: FullTextPaperDocument;
  paperKey: string;
  filePaths: readonly string[];
  observationFingerprints: readonly string[];
  contentHash?: string;
  refreshTitle: boolean;
  source: ScopedLibrarySource;
  extractor: PdfTextExtractor;
  nowIso: string;
  signal?: AbortSignal;
}): Promise<FullTextPaperDocument> {
  let title = input.document.title;
  if (input.refreshTitle) {
    const path = input.filePaths[0];
    if (!path) throw new Error("fallback paper has no file path for title refresh");
    const bytes = await input.source.readBinary(path, { signal: input.signal });
    const extraction = await input.extractor.extractPdfText(
      new Uint8Array(bytes),
      { signal: input.signal },
    );
    title = extractTitleFromFirstPage(extraction.pages, extraction.layout, extraction.metadataTitle) ?? undefined;
  }
  return {
    ...input.document,
    paperKey: input.paperKey,
    contentHash: input.contentHash ?? input.document.contentHash,
    title,
    titleVersion: input.refreshTitle ? TITLE_EXTRACTION_VERSION : input.document.titleVersion,
    filePaths: [...input.filePaths],
    observationFingerprints: [...input.observationFingerprints],
    updatedAt: input.nowIso,
  };
}

function recordFromDocument(
  document: FullTextPaperDocument,
  nowIso: string,
): FullTextPaperKnowledgeRecord {
  return {
    paperKey: document.paperKey,
    status: "ready",
    modelId: document.modelId,
    dimension: document.dimension,
    textHash: document.textHash,
    contentHash: document.contentHash,
    title: document.title,
    titleVersion: document.titleVersion,
    filePaths: [...document.filePaths],
    observationFingerprints: [...document.observationFingerprints],
    chunkCount: document.chunks.length,
    updatedAt: nowIso,
  };
}

function recordFailed(
  paperKey: string,
  error: string,
  nowIso: string,
  papers: Record<string, FullTextPaperKnowledgeRecord>,
  modelId: string,
  dimension: number,
  unit?: Pick<IndexUnit, "filePaths" | "observationFingerprints">,
): FullTextIndexPaperOutcome {
  const previous = papers[paperKey];
  const fingerprints = unit?.observationFingerprints.filter(
    (value): value is string => value !== undefined,
  );
  const completeUnit = unit && fingerprints?.length === unit.filePaths.length;
  papers[paperKey] = {
    paperKey,
    status: "failed",
    modelId,
    dimension,
    filePaths: completeUnit ? unit.filePaths : previous?.filePaths ?? [],
    observationFingerprints: completeUnit ? fingerprints : previous?.observationFingerprints ?? [],
    chunkCount: 0,
    error: error.slice(0, 500),
    updatedAt: nowIso,
  };
  return { paperKey, status: "failed", error };
}

function sameFingerprints(
  record: FullTextPaperKnowledgeRecord,
  fingerprints: readonly string[],
): boolean {
  return record.observationFingerprints.length === fingerprints.length
    && record.observationFingerprints.every((value, index) => value === fingerprints[index]);
}

function isCancellationErrorLike(error: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true;
  return error instanceof Error && error.name === "AbortError";
}

function describeRefreshError(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error);
}

/**
 * Query-time orchestration: embed the query (with the e5 `query:` prefix) and
 * brute-force search all ready papers. The embedding is the only model call at
 * query time; chunk vectors were precomputed at index time. Title fusion is
 * lexical (no model calls): provided titles are scored against the query with
 * `lexicalTitleSimilarity` and passed to the retriever as `titleScores`; see
 * `retrieval.ts` / `title-similarity.ts` for the short-query rationale.
 */
export interface SearchFullTextKnowledgeBaseInput {
  store: FullTextKnowledgeBaseStore;
  embedding: EmbeddingModel;
  queryText: string;
  /**
   * Optional per-paper titles (paperKey → title). When present they are
   * scored lexically against the query; titles without an entry are ignored.
   */
  titles?: ReadonlyMap<string, string>;
  limit?: number;
  maxHitsPerPaper?: number;
  signal?: AbortSignal;
}

export async function searchFullTextKnowledgeBase(
  input: SearchFullTextKnowledgeBaseInput,
): Promise<KnowledgeBasePaperMatch[]> {
  throwIfCancelled(input.signal);
  const manifest = await input.store.loadManifest();
  if (Object.keys(manifest.papers).length === 0) return [];
  const queryVectors = await input.embedding.embed(
    [prefixFor("query", input.embedding, input.queryText)],
    { signal: input.signal },
  );
  const queryVector = queryVectors[0];
  if (!queryVector) throw new Error("embedding model returned no query vector");
  const papers: FullTextPaperDocument[] = [];
  for (const [paperKey, record] of Object.entries(manifest.papers)) {
    throwIfCancelled(input.signal);
    if (record.status !== "ready") continue;
    const document = await input.store.loadPaper(paperKey);
    if (document) papers.push(document);
  }
  // Title fusion uses the first paragraph only. Find-similar builds
  // `title\n\nabstract`; scoring the whole blob against short titles collapses
  // Jaccard and never lifts the matching library paper.
  const titleQuery = firstQueryParagraph(input.queryText);
  let titleScores: Map<string, number> | undefined;
  if (input.titles && input.titles.size > 0) {
    titleScores = new Map();
    for (const [paperKey, title] of input.titles) {
      const score = lexicalTitleSimilarity(titleQuery, title);
      if (score > 0) titleScores.set(paperKey, score);
    }
    if (titleScores.size === 0) titleScores = undefined;
  }
  // Lexical token hits: keyword queries embed into the collapsed region and
  // need a literal signal; see `lexical-search.ts`. Long title+abstract queries
  // keep vector ranking only — body-token fusion would reward incidental terms.
  let tokenScores: Map<string, number> | undefined;
  if (isKeywordQuery(input.queryText)) {
    const tokens = significantQueryTokens(input.queryText);
    if (tokens.length > 0) {
      tokenScores = computeTokenHitScores(
        papers.map((paper) => ({
          paperKey: paper.paperKey,
          text: paper.chunks.map((chunk) => chunk.text).join(" "),
          title: input.titles?.get(paper.paperKey),
        })),
        tokens,
      );
      if (tokenScores.size === 0) tokenScores = undefined;
    }
  }
  return searchKnowledgeBase({
    papers,
    queryVector,
    titleScores,
    tokenScores,
    limit: input.limit,
    maxHitsPerPaper: input.maxHitsPerPaper,
  });
}

/** First non-empty paragraph of a query (title line of a title+abstract blob). */
function firstQueryParagraph(queryText: string): string {
  const paragraphs = queryText.split(/\n\s*\n/);
  for (const paragraph of paragraphs) {
    const trimmed = paragraph.trim();
    if (trimmed) return trimmed;
  }
  return queryText.trim();
}

/**
 * Let queued host events run (timers, clicks, renders) before continuing a
 * long index run, so host UIs (the Obsidian renderer) stay responsive while
 * many papers are extracted and embedded. Harmless on Node hosts.
 */
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof setTimeout === "function") setTimeout(resolve, 0);
    else resolve();
  });
}

/**
 * Progress line with wall-clock rate and remaining-time estimate, so a long
 * local index run reads as "working, N remaining" rather than "stuck".
 */
function indexProgressDetail(
  paperKey: string,
  position: number,
  total: number,
  startedAt: number,
): string {
  const elapsedMs = Date.now() - startedAt;
  const done = position - 1;
  let eta = "";
  if (done > 0 && elapsedMs > 0) {
    const perPaperMs = elapsedMs / done;
    const remainingSeconds = Math.round((total - done) * perPaperMs / 1000);
    if (remainingSeconds > 0) eta = `, ~${formatDuration(remainingSeconds)} remaining`;
  }
  return `indexing ${paperKey} (${position}/${total})${eta}`;
}

function formatDuration(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

/**
 * Apply the model's prefix policy: e5-family models get the query/passage
 * prefixes, remote models embed plain text.
 */
function prefixFor(
  kind: "query" | "passage",
  embedding: { readonly prefixPolicy: "e5" | "none" },
  text: string,
): string {
  return embedding.prefixPolicy === "e5" ? applyEmbeddingPrefix(kind, text) : text;
}
