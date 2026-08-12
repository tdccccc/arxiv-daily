/**
 * Full-text knowledge base document model, paths, and store contract.
 *
 * The knowledge base is a bypass store (side catalog): it never lives inside
 * papers.json, can be deleted and rebuilt, and its paths are sharded by the
 * same scope / identification fingerprints that bind the personal library
 * catalog. Concurrency follows the interest-profile store pattern
 * (expectedRevision CAS), not the catalog's whole-document replace.
 */

import type { OutputSettings } from "../../settings/types";
import { derivePaperInboxPaths } from "../../services/paper-index";
import { sha256Hex } from "../../utils/digest";

export const FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION = 1 as const;

/** One extracted, embedded chunk of a paper's full text. */
export interface FullTextChunk {
  /** Zero-based chunk position within the paper. */
  index: number;
  /** One-based source page of the chunk (first page of the chunk span). */
  page: number;
  text: string;
}

/**
 * Per-paper full-text document: chunks plus their row-major vectors.
 * Derived, content-addressed data — safe to rewrite idempotently; the
 * manifest is the authoritative index.
 */
export interface FullTextPaperDocument {
  schemaVersion: typeof FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION;
  paperKey: string;
  modelId: string;
  dimension: number;
  /** SHA-256 of the normalized extracted full text. */
  textHash: string;
  /** SHA-256 of source PDF bytes; present for content-addressed fallback papers. */
  contentHash?: string;
  /**
   * Title extracted from the first page at index time. Present for papers
   * without catalog metadata (fallback-indexed unresolved files); arXiv
   * papers get their title from the catalog at query time.
   */
  title?: string;
  /**
   * Version of the extraction rules that produced `title`; reuse refreshes
   * fallback titles when the rules advance (`TITLE_EXTRACTION_VERSION`).
   */
  titleVersion?: number;
  /** Catalog file paths observed at index time (change detection). */
  filePaths: readonly string[];
  /** Per-path observation fingerprints, same order as `filePaths`. */
  observationFingerprints: readonly string[];
  chunks: readonly FullTextChunk[];
  /** Row-major vectors; length === chunks.length * dimension. */
  vectors: Float32Array;
  updatedAt: string;
}

/** Manifest record per paper — the authoritative, CAS-protected index. */
export interface FullTextPaperKnowledgeRecord {
  paperKey: string;
  status: "ready" | "failed";
  modelId: string;
  dimension: number;
  /** Present when status is ready. */
  textHash?: string;
  /** SHA-256 of source PDF bytes for content-addressed fallback papers. */
  contentHash?: string;
  /** Extracted title for fallback-indexed files; arXiv papers use the catalog. */
  title?: string;
  /** Extraction-rule version of `title`; see `FullTextPaperDocument`. */
  titleVersion?: number;
  filePaths: readonly string[];
  observationFingerprints: readonly string[];
  chunkCount: number;
  error?: string;
  updatedAt: string;
}

export interface FullTextKnowledgeBaseManifest {
  schemaVersion: typeof FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION;
  revision: number;
  scopeFingerprint: string;
  identificationFingerprint: string;
  modelId: string;
  dimension: number;
  updatedAt: string;
  papers: Readonly<Record<string, FullTextPaperKnowledgeRecord>>;
}

export interface FullTextKnowledgeBaseDocumentPaths {
  directory: string;
  documentPath: string;
  backupPath: string;
}

export interface FullTextKnowledgeBasePaths {
  directory: string;
  manifest: FullTextKnowledgeBaseDocumentPaths;
  papersDirectory: string;
}

export interface FullTextKnowledgeBaseStorePathsOptions {
  now?: () => Date;
  onWarning?: (message: string, error?: unknown) => void;
}

export interface FullTextKnowledgeBaseStore {
  readonly paths: FullTextKnowledgeBasePaths;
  loadManifest(): Promise<FullTextKnowledgeBaseManifest>;
  replaceManifest(next: FullTextKnowledgeBaseManifest, expectedRevision: number): Promise<FullTextKnowledgeBaseManifest>;
  loadPaper(paperKey: string): Promise<FullTextPaperDocument | null>;
  savePaper(document: FullTextPaperDocument): Promise<void>;
  removePaper(paperKey: string): Promise<void>;
  /** Delete the whole knowledge base for this scope/identification (rebuild path). */
  removeAll(): Promise<void>;
}

export function createFullTextKnowledgeBasePaperPath(
  storage: Pick<StorageAdapterLike, "normalizePath">,
  paths: FullTextKnowledgeBasePaths,
  paperKey: string,
): string {
  return storage.normalizePath(`${paths.papersDirectory}/${sha256Hex(paperKey)}.json`);
}

/**
 * Derive knowledge base paths sharded by scope/identification fingerprints,
 * mirroring the interest-profile layout:
 * `<indexDir>/personal-library-knowledge-base/<scopeHex>/<idHex>/`.
 */
export function deriveFullTextKnowledgeBasePaths(
  storage: Pick<StorageAdapterLike, "normalizePath">,
  output: OutputSettings,
  scopeFingerprint: string,
  identificationFingerprint: string,
): FullTextKnowledgeBasePaths {
  const scopeHex = fingerprintHex("scopeFingerprint", scopeFingerprint);
  const identificationHex = fingerprintHex("identificationFingerprint", identificationFingerprint);
  const { indexDir } = derivePaperInboxPaths(output, (path) => storage.normalizePath(path));
  const directory = storage.normalizePath(
    `${indexDir}/personal-library-knowledge-base/${scopeHex}/${identificationHex}`,
  );
  return {
    directory,
    manifest: {
      directory,
      documentPath: storage.normalizePath(`${directory}/manifest.json`),
      backupPath: storage.normalizePath(`${directory}/manifest.json.backup`),
    },
    papersDirectory: storage.normalizePath(`${directory}/papers`),
  };
}

export interface FullTextPaperDocumentJson {
  schemaVersion: typeof FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION;
  paperKey: string;
  modelId: string;
  dimension: number;
  textHash: string;
  contentHash?: string;
  title?: string;
  titleVersion?: number;
  filePaths: string[];
  observationFingerprints: string[];
  chunks: FullTextChunk[];
  vectors: { encoding: "base64-float32-le"; data: string };
  updatedAt: string;
}

/** Serialize a paper document to JSON (vectors as base64 LE float32). */
export function serializeFullTextPaperDocument(document: FullTextPaperDocument): string {
  return `${JSON.stringify(toJsonDocument(document), null, 2)}\n`;
}

export function decodeFullTextPaperDocument(value: unknown): FullTextPaperDocument | null {
  if (!isPlainObject(value)) return null;
  if (value.schemaVersion !== FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION) return null;
  if (typeof value.paperKey !== "string" || !isNonEmptyString(value.paperKey)) return null;
  if (typeof value.modelId !== "string" || !isNonEmptyString(value.modelId)) return null;
  if (!isPositiveInteger(value.dimension)) return null;
  if (typeof value.textHash !== "string" || !isFingerprint(value.textHash)) return null;
  if (value.contentHash !== undefined && !isFingerprint(value.contentHash)) return null;
  if (!isLogicalPathArray(value.filePaths)) return null;
  if (!isFingerprintArray(value.observationFingerprints, value.filePaths.length)) return null;
  if (!isIsoDate(value.updatedAt)) return null;
  if (!Array.isArray(value.chunks)) return null;
  const chunks: FullTextChunk[] = [];
  let expectedIndex = 0;
  for (const chunk of value.chunks) {
    if (!isPlainObject(chunk)) return null;
    if (!isNonNegativeInteger(chunk.index) || chunk.index !== expectedIndex) return null;
    if (!isPositiveInteger(chunk.page)) return null;
    if (typeof chunk.text !== "string") return null;
    chunks.push({ index: chunk.index, page: chunk.page, text: chunk.text });
    expectedIndex += 1;
  }
  const vectors = decodeVectors(value.vectors, chunks.length * value.dimension);
  if (!vectors) return null;
  if (value.title !== undefined && (typeof value.title !== "string" || value.title.length === 0)) {
    return null;
  }
  if (value.titleVersion !== undefined && !isPositiveInteger(value.titleVersion)) {
    return null;
  }
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey: value.paperKey,
    modelId: value.modelId,
    dimension: value.dimension,
    textHash: value.textHash,
    contentHash: value.contentHash,
    title: value.title,
    titleVersion: value.titleVersion,
    filePaths: [...value.filePaths],
    observationFingerprints: [...value.observationFingerprints],
    chunks,
    vectors,
    updatedAt: value.updatedAt,
  };
}

export function decodeFullTextKnowledgeBaseManifest(value: unknown): FullTextKnowledgeBaseManifest | null {
  if (!isPlainObject(value)) return null;
  if (value.schemaVersion !== FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION) return null;
  if (!isNonNegativeSafeInteger(value.revision)) return null;
  if (!isFingerprint(value.scopeFingerprint) || !isFingerprint(value.identificationFingerprint)) return null;
  if (typeof value.modelId !== "string" || !isNonEmptyString(value.modelId)) return null;
  if (!isPositiveInteger(value.dimension)) return null;
  if (!isIsoDate(value.updatedAt)) return null;
  if (!isPlainObject(value.papers)) return null;
  const papers: Record<string, FullTextPaperKnowledgeRecord> = Object.create(null);
  for (const [paperKey, record] of Object.entries(value.papers)) {
    if (typeof paperKey !== "string" || paperKey.length === 0) return null;
    if (!isPlainObject(record)) return null;
    if (record.paperKey !== paperKey) return null;
    if (record.status !== "ready" && record.status !== "failed") return null;
    if (typeof record.modelId !== "string" || !isNonEmptyString(record.modelId)) return null;
    if (!isPositiveInteger(record.dimension)) return null;
    if (record.status === "ready") {
      if (typeof record.textHash !== "string" || !isFingerprint(record.textHash)) return null;
      if (record.contentHash !== undefined && !isFingerprint(record.contentHash)) return null;
      // A ready record must name the files it was indexed from.
      if (!isLogicalPathArray(record.filePaths)) return null;
    } else if (record.textHash !== undefined || record.contentHash !== undefined) {
      return null;
    } else if (!isOptionalLogicalPathArray(record.filePaths)) {
      // A failed record may carry no indexed files (first failure).
      return null;
    }
    if (record.status === "ready" && !isFingerprintArray(record.observationFingerprints, record.filePaths.length)) {
      return null;
    }
    if (record.status === "failed" && !isOptionalFingerprintArray(record.observationFingerprints, record.filePaths.length)) {
      return null;
    }
    if (record.title !== undefined && (typeof record.title !== "string" || record.title.length === 0)) {
      return null;
    }
    if (record.titleVersion !== undefined && !isPositiveInteger(record.titleVersion)) {
      return null;
    }
    if (!isNonNegativeInteger(record.chunkCount)) return null;
    if (record.error !== undefined && typeof record.error !== "string") return null;
    if (!isIsoDate(record.updatedAt)) return null;
    papers[paperKey] = {
      paperKey: record.paperKey,
      status: record.status,
      modelId: record.modelId,
      dimension: record.dimension,
      ...(record.textHash === undefined ? {} : { textHash: record.textHash }),
      ...(record.contentHash === undefined ? {} : { contentHash: record.contentHash }),
      ...(record.title === undefined ? {} : { title: record.title }),
      ...(record.titleVersion === undefined ? {} : { titleVersion: record.titleVersion }),
      filePaths: [...record.filePaths],
      observationFingerprints: [...record.observationFingerprints],
      chunkCount: record.chunkCount,
      ...(record.error === undefined ? {} : { error: record.error }),
      updatedAt: record.updatedAt,
    };
  }
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    revision: value.revision,
    scopeFingerprint: value.scopeFingerprint,
    identificationFingerprint: value.identificationFingerprint,
    modelId: value.modelId,
    dimension: value.dimension,
    updatedAt: value.updatedAt,
    papers,
  };
}

function toJsonDocument(document: FullTextPaperDocument): FullTextPaperDocumentJson {
  const bytes = new Uint8Array(document.vectors.buffer, document.vectors.byteOffset, document.vectors.byteLength);
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]!);
  }
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey: document.paperKey,
    modelId: document.modelId,
    dimension: document.dimension,
    textHash: document.textHash,
    contentHash: document.contentHash,
    title: document.title,
    titleVersion: document.titleVersion,
    filePaths: [...document.filePaths],
    observationFingerprints: [...document.observationFingerprints],
    chunks: document.chunks.map((chunk) => ({ ...chunk })),
    vectors: { encoding: "base64-float32-le", data: btoa(binary) },
    updatedAt: document.updatedAt,
  };
}

function decodeVectors(value: unknown, expectedLength: number): Float32Array | null {
  if (!isPlainObject(value)) return null;
  if (value.encoding !== "base64-float32-le" || typeof value.data !== "string") return null;
  let binary: string;
  try {
    binary = atob(value.data);
  } catch {
    return null;
  }
  if (binary.length % 4 !== 0 || binary.length / 4 !== expectedLength) return null;
  const words = new Uint32Array(expectedLength);
  for (let index = 0; index < expectedLength; index += 1) {
    const code = binary.charCodeAt(index * 4)
      | (binary.charCodeAt(index * 4 + 1) << 8)
      | (binary.charCodeAt(index * 4 + 2) << 16)
      | (binary.charCodeAt(index * 4 + 3) << 24);
    words[index] = code >>> 0;
  }
  const view = new DataView(words.buffer);
  const out = new Float32Array(expectedLength);
  for (let index = 0; index < expectedLength; index += 1) {
    out[index] = view.getFloat32(index * 4, true);
  }
  return out;
}

type StorageAdapterLike = { normalizePath(path: string): string };

function fingerprintHex(name: string, value: string): string {
  if (!isFingerprint(value)) {
    throw new Error(`full-text knowledge base ${name} must be a SHA-256 fingerprint`);
  }
  return value.slice("sha256:".length);
}

function isFingerprint(value: unknown): value is string {
  return typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
}

function isFingerprintArray(value: unknown, expectedLength: number): value is string[] {
  return Array.isArray(value)
    && value.length === expectedLength
    && value.every(isFingerprint);
}

function isOptionalFingerprintArray(value: unknown, expectedLength: number): value is string[] {
  return Array.isArray(value)
    && (value.length === 0 || value.length === expectedLength)
    && value.every(isFingerprint);
}

function isLogicalRelativePath(value: unknown): value is string {
  return typeof value === "string"
    && value.length > 0
    && !value.includes("\\")
    && !value.includes("\0")
    && !value.startsWith("/")
    && !/^[A-Za-z]:/.test(value)
    && value.split("/").every((segment) => segment.length > 0 && segment !== "." && segment !== "..");
}

function isLogicalPathArray(value: unknown): value is string[] {
  return Array.isArray(value)
    && value.length > 0
    && value.every(isLogicalRelativePath)
    && new Set(value).size === value.length
    && [...value].sort().every((path, index) => path === value[index]);
}

function isOptionalLogicalPathArray(value: unknown): value is string[] {
  return Array.isArray(value)
    && (value.length === 0 || value.every(isLogicalRelativePath))
    && new Set(value).size === value.length
    && [...value].sort().every((path, index) => path === value[index]);
}

function isIsoDate(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return isNonNegativeInteger(value);
}

function isPlainObject(value: unknown): value is Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}
