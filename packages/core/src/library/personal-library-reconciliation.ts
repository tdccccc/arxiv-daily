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
 * canonical arXiv ID or null. Identity remains evidence-checked downstream:
 * the resolver still fetches canonical metadata before a file becomes ready.
 */
export interface PersonalLibraryFileIdentifier {
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
    const observationFingerprint = createLibraryFileObservationFingerprint(
      entry,
      input.current.identificationFingerprint,
    );
    const previous = input.current.files[entry.path];
    if (
      previous
      && previous.observationFingerprint === observationFingerprint
      && previous.status !== "failed"
    ) {
      defineRecordEntry(files, entry.path, clone(previous));
      reusedFileCount += 1;
      continue;
    }

    if (!eligibleExtensions.has(extensionOf(entry.path))) {
      defineRecordEntry(files, entry.path, {
        path: entry.path,
        status: "unrelated",
        observationFingerprint,
        reason: "unsupported-file-type",
        updatedAt: nowIso,
      });
      continue;
    }

    let arxivId = identifyModernArxivIdFromFilename(entry.path);
    if (!arxivId && input.identifyFile) {
      try {
        arxivId = await input.identifyFile.identify(entry.path, input.signal, entry.size) ?? null;
      } catch {
        // Content identification is best-effort: any failure keeps the file
        // unresolved instead of failing or blocking the scan.
        arxivId = null;
      }
    }
    if (!arxivId) {
      defineRecordEntry(files, entry.path, {
        path: entry.path,
        status: "unresolved",
        observationFingerprint,
        reason: "unrecognized-filename",
        updatedAt: nowIso,
      });
      continue;
    }
    identified.push({ entry, observationFingerprint, arxivId });
  }

  if (input.inventory.truncated) {
    for (const [path, record] of Object.entries(input.current.files)) {
      if (!Object.hasOwn(files, path) && !identified.some((candidate) => candidate.entry.path === path)) {
        defineRecordEntry(files, path, clone(record));
      }
    }
  }

  const reusedPaperIds = new Set<string>();
  for (const record of Object.values(files)) {
    if (record.status === "ready") reusedPaperIds.add(record.arxivId);
  }
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
    } catch (error) {
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
