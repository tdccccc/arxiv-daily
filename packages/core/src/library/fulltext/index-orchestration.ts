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

export interface FullTextIndexPaperOutcome {
  paperKey: string;
  status: "indexed" | "reused" | "failed";
  chunkCount?: number;
  error?: string;
}

export interface FullTextIndexRunSummary {
  indexed: number;
  reused: number;
  failed: number;
  pruned: number;
  outcomes: readonly FullTextIndexPaperOutcome[];
  manifestRevision: number;
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
  const paperKeys = Object.keys(catalog.papers).sort();
  const total = paperKeys.length;
  const progressStartedAt = Date.now();

  for (let position = 0; position < total; position += 1) {
    throwIfCancelled(input.signal);
    const paperKey = paperKeys[position]!;
    const paper = catalog.papers[paperKey]!;
    input.onProgress?.(indexProgressDetail(paperKey, position + 1, total, progressStartedAt));

    const fingerprints = paper.filePaths.map((path) => catalog.files[path]?.observationFingerprint);
    if (fingerprints.some((fingerprint) => fingerprint === undefined)) {
      outcomes.push(recordFailed(paperKey, "catalog file record missing", nowIso, papers, embedding.modelId, embedding.dimension));
      continue;
    }
    const observationFingerprints = fingerprints as string[];

    const previous = papers[paperKey];
    if (previous
      && previous.status === "ready"
      && previous.modelId === embedding.modelId
      && sameFingerprints(previous, observationFingerprints)) {
      outcomes.push({ paperKey, status: "reused" });
      continue;
    }

    try {
      const document = await buildPaperDocument({
        paperKey,
        filePaths: paper.filePaths,
        observationFingerprints,
        source,
        extractor,
        embedding,
        nowIso,
        signal: input.signal,
      });
      await store.savePaper(document);
      papers[paperKey] = recordFromDocument(document, nowIso);
      outcomes.push({ paperKey, status: "indexed", chunkCount: document.chunks.length });
      log?.info(`fulltext: indexed ${paperKey} (${document.chunks.length} chunks)`);
    } catch (caught) {
      if (isCancellationErrorLike(caught, input.signal)) throw caught;
      const message = caught instanceof Error ? caught.message : String(caught);
      log?.warn(`fulltext: indexing failed for ${paperKey}: ${message}`);
      outcomes.push(recordFailed(paperKey, message, nowIso, papers, embedding.modelId, embedding.dimension));
    }
    // Yield between papers so a long index run does not freeze host UIs (the
    // Obsidian renderer processes queued events between papers); harmless on
    // Node hosts.
    await yieldToEventLoop();
  }

  // Prune papers that left the catalog; their derived documents are deleted too.
  let pruned = 0;
  for (const paperKey of Object.keys(papers)) {
    throwIfCancelled(input.signal);
    if (catalog.papers[paperKey]) continue;
    await store.removePaper(paperKey);
    delete papers[paperKey];
    pruned += 1;
  }

  const saved = await store.replaceManifest(next, loaded.revision);
  log?.info(
    `fulltext: index run complete (revision ${saved.revision}, indexed ${outcomes.filter((o) => o.status === "indexed").length}, `
    + `reused ${outcomes.filter((o) => o.status === "reused").length}, failed ${outcomes.filter((o) => o.status === "failed").length}, pruned ${pruned})`,
  );
  return {
    indexed: outcomes.filter((o) => o.status === "indexed").length,
    reused: outcomes.filter((o) => o.status === "reused").length,
    failed: outcomes.filter((o) => o.status === "failed").length,
    pruned,
    outcomes,
    manifestRevision: saved.revision,
  };
}

async function buildPaperDocument(input: {
  paperKey: string;
  filePaths: readonly string[];
  observationFingerprints: readonly string[];
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
  const pages = (await input.extractor.extractPdfText(new Uint8Array(bytes), { signal: input.signal })).pages;
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
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey,
    modelId: input.embedding.modelId,
    dimension,
    textHash,
    filePaths: [...filePaths],
    observationFingerprints: [...observationFingerprints],
    chunks,
    vectors: flat,
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
): FullTextIndexPaperOutcome {
  const previous = papers[paperKey];
  papers[paperKey] = {
    paperKey,
    status: "failed",
    modelId,
    dimension,
    filePaths: previous?.filePaths ?? [],
    observationFingerprints: previous?.observationFingerprints ?? [],
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

/**
 * Query-time orchestration: embed the query (with the e5 `query:` prefix) and
 * brute-force search all ready papers. The embedding is the only model call at
 * query time; chunk vectors were precomputed at index time.
 */
export interface SearchFullTextKnowledgeBaseInput {
  store: FullTextKnowledgeBaseStore;
  embedding: EmbeddingModel;
  queryText: string;
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
  return searchKnowledgeBase({
    papers,
    queryVector,
    limit: input.limit,
    maxHitsPerPaper: input.maxHitsPerPaper,
  });
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
