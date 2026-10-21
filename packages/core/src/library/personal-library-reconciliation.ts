import type { LibraryInventory, LibrarySourceEntry } from "./scoped-library-source";
import {
  createEmptyPersonalLibraryCatalog,
  type PersonalLibraryCatalog,
  type PersonalLibraryFileRecord,
  type PersonalLibraryPaperRecord,
} from "./personal-library-catalog";
import { paperKeyFromArxivId } from "../services/paper-key";
import { modernArxivResources } from "../utils/arxiv";
import { sha256Hex } from "../utils/digest";
import { throwIfCancelled } from "../services/cancellation";

export interface PersonalLibraryResolvedMetadata {
  arxivId: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  updated: string;
  primaryCategory: string;
  categories: string[];
}

export interface PersonalLibraryMetadataResolver {
  resolve(
    arxivIds: string[],
    signal?: AbortSignal,
  ): Promise<Map<string, PersonalLibraryResolvedMetadata>>;
}

/**
 * Content-based file identification (identification strategy v2). The host
 * supplies evidence (e.g. PDF text extraction via the scoped source) for
 * files whose filenames carry no arXiv ID; the identifier returns a
 * canonical arXiv ID or null. Returned values are normalized and rejected
 * unless the catalog supports them, then the resolver still fetches canonical
 * metadata before a file becomes ready.
 */
export interface PersonalLibraryFileIdentifier {
  /** Bump when content-evidence rules change so unchanged files are re-evaluated. */
  readonly version: number;
  /**
   * `null` is a judgement: the content names no arXiv paper. It is remembered
   * under `version` and not asked again while the file is unchanged. A throw
   * means the file could not be judged this time (arXiv unreachable, rate
   * limited): the file stays unresolved but nothing is remembered, so the next
   * scan asks again.
   */
  identify(
    logicalPath: string,
    signal?: AbortSignal,
    /** Observed file size in bytes, when the inventory provides it. */
    size?: number,
  ): Promise<string | null>;
}

export interface ReconcilePersonalLibraryCatalogInput {
  current: PersonalLibraryCatalog;
  inventory: LibraryInventory;
  eligibleExtensions: readonly string[];
  resolver: PersonalLibraryMetadataResolver;
  /** Optional content-based identification for files with unrecognized names. */
  identifyFile?: PersonalLibraryFileIdentifier;
  now?: Date;
  signal?: AbortSignal;
}

export interface ReconcilePersonalLibraryCatalogResult {
  catalog: PersonalLibraryCatalog;
  resolvedArxivIds: string[];
  reusedFileCount: number;
}

interface IdentifiedFile {
  entry: LibrarySourceEntry;
  observationFingerprint: string;
  arxivId: string;
  contentIdentificationVersion?: number;
}

export function identifyModernArxivIdFromFilename(logicalPath: string): string | null {
  const filename = logicalPath.split("/").at(-1) ?? "";
  const stem = filename.replace(/\.[^.]+$/, "");
  const matches = Array.from(stem.matchAll(/(?:^|[^0-9A-Za-z])(\d{4}\.\d{4,5}(?:v\d+)?)(?=$|[^0-9A-Za-z])/gi));
  const canonicalIds = new Set(matches.flatMap((match) => {
    const id = modernArxivResources(match[1] ?? "")?.id;
    return id ? [id] : [];
  }));
  return canonicalIds.size === 1 ? [...canonicalIds][0]! : null;
}

export function createLibraryFileObservationFingerprint(
  entry: Pick<LibrarySourceEntry, "path" | "type" | "size" | "mtimeMs">,
  identificationFingerprint: string,
): string {
  return `sha256:${sha256Hex(JSON.stringify({
    identificationFingerprint,
    path: entry.path,
    type: entry.type,
    size: entry.size ?? null,
    mtimeMs: entry.mtimeMs ?? null,
  }))}`;
}

export interface PersonalLibraryFolderChange {
  changed: boolean;
  /** The catalog has no completed scan, so there is nothing to compare with. */
  neverScanned: boolean;
  /** Eligible files the last scan did not see. */
  added: string[];
  /** Eligible files the last scan saw that are no longer listed (a complete listing only). */
  removed: string[];
  /** Eligible files whose size or modification time differs from the last scan. */
  modified: string[];
}

/**
 * Whether a scan would find anything new in the folder, from a local listing
 * alone (no file reads, no network). It applies the same rules as
 * `reconcilePersonalLibraryCatalog`: only eligible regular files matter, a file
 * is unchanged when its observation fingerprint (path, size, modification time
 * under the catalog's identification strategy) matches the stored record, and a
 * truncated listing cannot show removals. A renamed or moved file is a removal
 * plus an addition. Files a scan could not resolve or fetch metadata for are
 * "unchanged" while the folder is: retrying them is an explicit scan's job.
 */
export function detectPersonalLibraryFolderChange(input: {
  catalog: PersonalLibraryCatalog;
  inventory: LibraryInventory;
  eligibleExtensions: readonly string[];
}): PersonalLibraryFolderChange {
  if (input.catalog.lastScan === null) {
    return { changed: true, neverScanned: true, added: [], removed: [], modified: [] };
  }
  const eligibleExtensions = normalizeExtensions(input.eligibleExtensions);
  const added: string[] = [];
  const modified: string[] = [];
  const listed = new Set<string>();
  for (const entry of input.inventory.entries) {
    if (entry.type !== "file" || !eligibleExtensions.has(extensionOf(entry.path))) continue;
    listed.add(entry.path);
    const previous = Object.hasOwn(input.catalog.files, entry.path)
      ? input.catalog.files[entry.path]
      : undefined;
    if (!previous || previous.status === "unrelated") added.push(entry.path);
    else if (previous.observationFingerprint
      !== createLibraryFileObservationFingerprint(entry, input.catalog.identificationFingerprint)) {
      modified.push(entry.path);
    }
  }
  const removed = input.inventory.truncated
    ? []
    : Object.values(input.catalog.files)
      .filter((record) => record.status !== "unrelated" && !listed.has(record.path))
      .map((record) => record.path);
  added.sort(); modified.sort(); removed.sort();
  return {
    changed: added.length + modified.length + removed.length > 0,
    neverScanned: false, added, removed, modified,
  };
}

export async function reconcilePersonalLibraryCatalog(
  input: ReconcilePersonalLibraryCatalogInput,
): Promise<ReconcilePersonalLibraryCatalogResult> {
  throwIfCancelled(input.signal);
  const now = input.now ?? new Date();
  const nowIso = now.toISOString();
  const eligibleExtensions = normalizeExtensions(input.eligibleExtensions);
  const next = createEmptyPersonalLibraryCatalog(
    input.current.scopeFingerprint,
    input.current.identificationFingerprint,
    now,
  );
  const files: Record<string, PersonalLibraryFileRecord> = Object.create(null) as Record<
    string,
    PersonalLibraryFileRecord
  >;
  const identified: IdentifiedFile[] = [];
  let reusedFileCount = 0;

  for (const entry of input.inventory.entries) {
    throwIfCancelled(input.signal);
    if (entry.type !== "file") continue;
    const eligible = eligibleExtensions.has(extensionOf(entry.path));
    const filenameArxivId = eligible ? identifyModernArxivIdFromFilename(entry.path) : null;
    const contentIdentificationVersion = eligible && !filenameArxivId && input.identifyFile
      ? requireIdentifierVersion(input.identifyFile.version)
      : undefined;
    const observationFingerprint = createLibraryFileObservationFingerprint(
      entry,
      input.current.identificationFingerprint,
    );
    const previous = input.current.files[entry.path];
    if (
      previous
      && previous.observationFingerprint === observationFingerprint
      && previous.status !== "failed"
      && recordContentIdentificationVersion(previous) === contentIdentificationVersion
    ) {
      defineRecordEntry(files, entry.path, clone(previous));
      reusedFileCount += 1;
      continue;
    }

    if (!eligible) {
      defineRecordEntry(files, entry.path, {
        path: entry.path,
        status: "unrelated",
        observationFingerprint,
        reason: "unsupported-file-type",
        updatedAt: nowIso,
      });
      continue;
    }

    let arxivId = filenameArxivId;
    let judged = true;
    if (!arxivId && input.identifyFile) {
      try {
        const identified = await input.identifyFile.identify(
          entry.path,
          input.signal,
          entry.size,
        );
        arxivId = identified ? modernArxivResources(identified)?.id ?? null : null;
      } catch {
        // Content identification is best-effort: any failure or unsupported ID
        // keeps the file unresolved instead of failing or blocking the scan. A
        // failure is not a judgement, though: it is left without the rule
        // version so that the next scan identifies the file again.
        throwIfCancelled(input.signal);
        arxivId = null;
        judged = false;
      }
    }
    if (!arxivId) {
      defineRecordEntry(files, entry.path, {
        path: entry.path,
        status: "unresolved",
        observationFingerprint,
        reason: "unrecognized-filename",
        ...(contentIdentificationVersion === undefined || !judged ? {} : { contentIdentificationVersion }),
        updatedAt: nowIso,
      });
      continue;
    }
    identified.push({
      entry,
      observationFingerprint,
      arxivId,
      ...(contentIdentificationVersion === undefined ? {} : { contentIdentificationVersion }),
    });
  }

  if (input.inventory.truncated) {
    for (const [path, record] of Object.entries(input.current.files)) {
      if (!Object.hasOwn(files, path) && !identified.some((candidate) => candidate.entry.path === path)) {
        defineRecordEntry(files, path, clone(record));
      }
    }
  }

  // Metadata is looked up on arXiv only for papers the catalog does not hold
  // yet. A paper it already holds (kept by an unchanged file, or reached by a
  // renamed, moved or touched file) keeps its stored record: the arXiv id is
  // what identifies the paper, so the lookup would only ask the same question.
  const reusedPaperIds = new Set<string>();
  for (const record of Object.values(files)) {
    if (record.status === "ready") reusedPaperIds.add(record.arxivId);
  }
  for (const paper of Object.values(input.current.papers)) reusedPaperIds.add(paper.externalId);
  const idsToResolve = Array.from(new Set(
    identified
      .map(({ arxivId }) => arxivId)
      .filter((arxivId) => !reusedPaperIds.has(arxivId)),
  )).sort();
  let resolved = new Map<string, PersonalLibraryResolvedMetadata>();
  let resolverFailed = false;
  if (idsToResolve.length > 0) {
    try {
      resolved = await input.resolver.resolve(idsToResolve, input.signal);
      throwIfCancelled(input.signal);
    } catch {
      throwIfCancelled(input.signal);
      resolverFailed = true;
    }
  }

  for (const candidate of identified) {
    throwIfCancelled(input.signal);
    const previousPaper = input.current.papers[paperKeyFromArxivId(candidate.arxivId)];
    const metadata = resolved.get(candidate.arxivId);
    if (metadata && isResolvedMetadataFor(metadata, candidate.arxivId)) {
      defineRecordEntry(files, candidate.entry.path, readyFile(candidate, nowIso));
      continue;
    }
    if (previousPaper) {
      defineRecordEntry(files, candidate.entry.path, readyFile(candidate, nowIso));
      continue;
    }
    defineRecordEntry(files, candidate.entry.path, {
      path: candidate.entry.path,
      status: "failed",
      observationFingerprint: candidate.observationFingerprint,
      reason: resolverFailed ? "metadata-fetch-failed" : "metadata-unavailable",
      arxivId: candidate.arxivId,
      updatedAt: nowIso,
    });
  }

  const paperPaths = new Map<string, string[]>();
  for (const record of Object.values(files)) {
    if (record.status !== "ready") continue;
    const paths = paperPaths.get(record.arxivId) ?? [];
    paths.push(record.path);
    paperPaths.set(record.arxivId, paths);
  }

  const papers: Record<string, PersonalLibraryPaperRecord> = {};
  for (const [arxivId, paths] of paperPaths) {
    const paperKey = paperKeyFromArxivId(arxivId);
    const metadata = resolved.get(arxivId);
    const previous = input.current.papers[paperKey];
    const paper = metadata && isResolvedMetadataFor(metadata, arxivId)
      ? paperFromMetadata(metadata, paths.sort())
      : previous
        ? { ...clone(previous), filePaths: paths.sort() }
        : null;
    if (!paper) continue;
    papers[paperKey] = paper;
  }

  next.files = sortRecord(files);
  next.papers = sortRecord(papers);
  next.lastScan = summarize(next, input.inventory.truncated);
  return {
    catalog: next,
    resolvedArxivIds: idsToResolve,
    reusedFileCount,
  };
}

function readyFile(candidate: IdentifiedFile, updatedAt: string): PersonalLibraryFileRecord {
  return {
    path: candidate.entry.path,
    status: "ready",
    observationFingerprint: candidate.observationFingerprint,
    paperKey: paperKeyFromArxivId(candidate.arxivId),
    arxivId: candidate.arxivId,
    ...(candidate.contentIdentificationVersion === undefined
      ? {}
      : { contentIdentificationVersion: candidate.contentIdentificationVersion }),
    updatedAt,
  };
}

function paperFromMetadata(
  metadata: PersonalLibraryResolvedMetadata,
  filePaths: string[],
): PersonalLibraryPaperRecord {
  return {
    paperKey: paperKeyFromArxivId(metadata.arxivId),
    source: "arxiv",
    externalId: metadata.arxivId,
    title: metadata.title,
    authors: [...metadata.authors],
    abstract: metadata.abstract,
    published: metadata.published,
    updated: metadata.updated,
    primaryCategory: metadata.primaryCategory,
    categories: [...metadata.categories],
    evidenceDepth: "metadata-and-abstract",
    filePaths,
  };
}

function isResolvedMetadataFor(
  metadata: PersonalLibraryResolvedMetadata,
  arxivId: string,
): boolean {
  const canonical = modernArxivResources(metadata.arxivId)?.id;
  return canonical === arxivId
    && metadata.arxivId === canonical
    && metadata.title.trim().length > 0
    && metadata.authors.length > 0
    && metadata.authors.every((author) => author.trim().length > 0)
    && isCanonicalIsoDate(metadata.published)
    && isCanonicalIsoDate(metadata.updated)
    && metadata.primaryCategory.trim().length > 0
    && metadata.categories.length > 0
    && metadata.categories.every((category) => category.trim().length > 0)
    && new Set(metadata.categories).size === metadata.categories.length
    && metadata.categories.includes(metadata.primaryCategory);
}

function summarize(
  catalog: Pick<PersonalLibraryCatalog, "files" | "papers">,
  truncated: boolean,
) {
  const summary = {
    ready: 0,
    unresolved: 0,
    unrelated: 0,
    failed: 0,
    papers: Object.keys(catalog.papers).length,
    truncated,
  };
  for (const record of Object.values(catalog.files)) summary[record.status] += 1;
  return summary;
}

function recordContentIdentificationVersion(
  record: PersonalLibraryFileRecord,
): number | undefined {
  return record.status === "ready" || record.status === "unresolved"
    ? record.contentIdentificationVersion
    : undefined;
}

function requireIdentifierVersion(value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error("identifyFile.version must be a positive safe integer");
  }
  return value;
}

function normalizeExtensions(extensions: readonly string[]): Set<string> {
  const normalized = extensions.map((extension) => extension.trim().toLowerCase());
  if (normalized.length === 0 || normalized.some((extension) => !/^\.[a-z0-9]+$/.test(extension))) {
    throw new Error("eligibleExtensions must contain valid extensions");
  }
  return new Set(normalized);
}

function extensionOf(logicalPath: string): string {
  const filename = logicalPath.split("/").at(-1) ?? "";
  const index = filename.lastIndexOf(".");
  return index < 0 ? "" : filename.slice(index).toLowerCase();
}

function sortRecord<T>(record: Record<string, T>): Record<string, T> {
  return Object.fromEntries(Object.entries(record).sort(([left], [right]) => (
    left < right ? -1 : left > right ? 1 : 0
  )));
}

function defineRecordEntry<T>(record: Record<string, T>, key: string, value: T): void {
  Object.defineProperty(record, key, {
    value,
    enumerable: true,
    configurable: true,
    writable: true,
  });
}

function isCanonicalIsoDate(value: string): boolean {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
