import { constants, type Stats } from "node:fs";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import {
  isCancellationError,
  LibrarySourceError,
  throwIfCancelled,
  type LibraryInventory,
  type LibraryInventoryOptions,
  type LibraryReadOptions,
  type LibrarySourceEntry,
  type ScopedLibrarySource,
} from "@arxiv-daily/core";

const DEFAULT_MAX_ENTRIES = 10_000;
const DEFAULT_MAX_DEPTH = 16;
const DEFAULT_MAX_READ_BYTES = 25 * 1024 * 1024;

export interface OpenScopedLibrarySourceOptions {
  maxEntries?: number;
  maxDepth?: number;
  maxReadBytes?: number;
}

export interface OpenedScopedLibrarySource extends ScopedLibrarySource {
  readonly canonicalRoot: string;
  readonly rootIdentity: string;
}

export async function openScopedLibrarySource(
  selectedRoot: string,
  options: OpenScopedLibrarySourceOptions = {},
): Promise<OpenedScopedLibrarySource> {
  const canonicalRoot = await canonicalizeRoot(selectedRoot);
  const rootInfo = await fs.stat(canonicalRoot);
  return new NodeScopedLibrarySource(
    canonicalRoot,
    `${rootInfo.dev}:${rootInfo.ino}`,
    options,
  );
}

class NodeScopedLibrarySource implements OpenedScopedLibrarySource {
  private readonly maxEntries: number;
  private readonly maxDepth: number;
  private readonly maxReadBytes: number;

  constructor(
    readonly canonicalRoot: string,
    readonly rootIdentity: string,
    options: OpenScopedLibrarySourceOptions,
  ) {
    this.maxEntries = positiveInteger(options.maxEntries, DEFAULT_MAX_ENTRIES);
    this.maxDepth = nonNegativeInteger(options.maxDepth, DEFAULT_MAX_DEPTH);
    this.maxReadBytes = positiveInteger(options.maxReadBytes, DEFAULT_MAX_READ_BYTES);
  }

  async inventory(options: LibraryInventoryOptions = {}): Promise<LibraryInventory> {
    await this.assertRootIdentity();
    const maxEntries = boundedPositiveInteger(options.maxEntries, this.maxEntries);
    const maxDepth = boundedNonNegativeInteger(options.maxDepth, this.maxDepth);
    const entries: LibrarySourceEntry[] = [];
    let truncated = false;

    const visit = async (relativeDir: string, depth: number): Promise<void> => {
      throwIfCancelled(options.signal);
      const absoluteDir = relativeDir
        ? path.join(this.canonicalRoot, ...relativeDir.split("/"))
        : this.canonicalRoot;
      let directory: Awaited<ReturnType<typeof fs.opendir>>;
      try {
        await assertCanonicalDirectory(this.canonicalRoot, absoluteDir);
        directory = await fs.opendir(absoluteDir);
        await assertCanonicalDirectory(this.canonicalRoot, absoluteDir);
      } catch (error) {
        if (error instanceof LibrarySourceError) throw error;
        throw mapFsError(error, "Unable to list the selected library");
      }

      try {
        for await (const dirEntry of directory) {
          throwIfCancelled(options.signal);
          if (entries.length >= maxEntries) {
            truncated = true;
            return;
          }
          const relativePath = relativeDir
            ? `${relativeDir}/${dirEntry.name}`
            : dirEntry.name;

          const absoluteEntry = path.join(absoluteDir, dirEntry.name);
          const entryInfo = await inspectInventoryEntry(absoluteEntry, dirEntry);
          throwIfCancelled(options.signal);
          if (entryInfo.kind === "symbolic-link") {
            entries.push({
              path: relativePath,
              type: "ignored",
              ignoredReason: "symbolic-link",
            });
            continue;
          }
          if (entryInfo.kind === "directory") {
            entries.push({ path: relativePath, type: "folder" });
            if (depth < maxDepth) {
              await visit(relativePath, depth + 1);
              if (truncated) return;
            } else {
              truncated = true;
            }
            continue;
          }
          if (entryInfo.kind === "file") {
            entries.push({
              path: relativePath,
              type: "file",
              size: entryInfo.info.size,
              mtimeMs: entryInfo.info.mtimeMs,
            });
            continue;
          }
          entries.push({
            path: relativePath,
            type: "ignored",
            ignoredReason: "unsupported-entry",
          });
        }
      } catch (error) {
        if (isCancellationError(error) || error instanceof LibrarySourceError) throw error;
        throw mapFsError(error, "Unable to list the selected library");
      } finally {
        await directory.close().catch(() => undefined);
      }
      await assertCanonicalDirectory(this.canonicalRoot, absoluteDir);
    };

    await visit("", 0);
    await this.assertRootIdentity();
    return { entries, truncated };
  }

  async readBinary(
    logicalPath: string,
    options: LibraryReadOptions = {},
  ): Promise<ArrayBuffer> {
    throwIfCancelled(options.signal);
    await this.assertRootIdentity();
    const segments = validateLogicalPath(logicalPath);
    const absolutePath = path.join(this.canonicalRoot, ...segments);
    await this.assertNoSymbolicLink(segments);

    let handle: Awaited<ReturnType<typeof fs.open>>;
    try {
      handle = await fs.open(
        absolutePath,
        constants.O_RDONLY | noFollowFlag(),
      );
    } catch (error) {
      throw mapFsError(error, "Unable to inspect the requested library file");
    }

    try {
      await this.assertNoSymbolicLink(segments);
      const [canonicalTarget, pathInfo, pathLinkInfo, handleInfo] = await Promise.all([
        fs.realpath(absolutePath),
        fs.stat(absolutePath),
        fs.lstat(absolutePath),
        handle.stat(),
      ]);
      assertContained(this.canonicalRoot, canonicalTarget);
      if (pathLinkInfo.isSymbolicLink()) {
        throw new LibrarySourceError(
          "unsafe-path",
          "Symbolic links are not readable through the library boundary",
        );
      }
      if (!handleInfo.isFile() || !pathInfo.isFile()) {
        throw new LibrarySourceError(
          "not-file",
          "The requested library entry is not a file",
        );
      }
      if (!sameFileIdentity(pathInfo, handleInfo)) {
        throw new LibrarySourceError(
          "unsafe-path",
          "The requested library entry changed while it was being opened",
        );
      }

      const maxBytes = boundedPositiveInteger(options.maxBytes, this.maxReadBytes);
      if (handleInfo.size > maxBytes) {
        throw new LibrarySourceError(
          "limit-exceeded",
          "The requested library file exceeds the configured read limit",
        );
      }

      throwIfCancelled(options.signal);
      const buffer = await handle.readFile({ signal: options.signal });
      throwIfCancelled(options.signal);
      if (buffer.byteLength > maxBytes) {
        throw new LibrarySourceError(
          "limit-exceeded",
          "The requested library file exceeds the configured read limit",
        );
      }
      return Uint8Array.from(buffer).buffer;
    } catch (error) {
      if (isCancellationError(error) || error instanceof LibrarySourceError) throw error;
      throw mapFsError(error, "Unable to read the requested library file");
    } finally {
      await handle.close().catch(() => undefined);
    }
  }

  private async assertRootIdentity(): Promise<void> {
    try {
      const info = await fs.stat(this.canonicalRoot);
      if (
        !info.isDirectory()
        || `${info.dev}:${info.ino}` !== this.rootIdentity
      ) {
        throw new LibrarySourceError(
          "unsafe-path",
          "The selected library root changed after access was granted",
        );
      }
    } catch (error) {
      if (error instanceof LibrarySourceError) throw error;
      throw mapFsError(error, "Unable to verify the selected library root", "invalid-root");
    }
  }

  private async assertNoSymbolicLink(segments: string[]): Promise<void> {
    let current = this.canonicalRoot;
    for (const segment of segments) {
      current = path.join(current, segment);
      let info: Stats;
      try {
        info = await fs.lstat(current);
      } catch (error) {
        throw mapFsError(error, "Unable to inspect the requested library entry");
      }
      if (info.isSymbolicLink()) {
        throw new LibrarySourceError(
          "unsafe-path",
          "Symbolic links are not readable through the library boundary",
        );
      }
    }
  }
}

type InventoryEntryInfo =
  | { kind: "symbolic-link" }
  | { kind: "directory" }
  | { kind: "file"; info: Stats }
  | { kind: "unsupported" };

async function inspectInventoryEntry(
  absoluteEntry: string,
  dirEntry: { isSymbolicLink(): boolean },
): Promise<InventoryEntryInfo> {
  if (dirEntry.isSymbolicLink()) return { kind: "symbolic-link" };
  try {
    // lstat both supplies file observations and classifies DT_UNKNOWN entries.
    // Rechecking known entries also catches replacement by a symbolic link after
    // the directory entry was read.
    const info = await fs.lstat(absoluteEntry);
    if (info.isSymbolicLink()) return { kind: "symbolic-link" };
    if (info.isDirectory()) return { kind: "directory" };
    if (info.isFile()) return { kind: "file", info };
    return { kind: "unsupported" };
  } catch (error) {
    throw mapFsError(error, "Unable to inspect a selected library entry");
  }
}

async function canonicalizeRoot(selectedRoot: string): Promise<string> {
  if (!selectedRoot.trim()) {
    throw new LibrarySourceError("invalid-root", "A library root is required");
  }
  try {
    const canonicalRoot = await fs.realpath(selectedRoot);
    const info = await fs.stat(canonicalRoot);
    if (!info.isDirectory()) {
      throw new LibrarySourceError("invalid-root", "The selected library root is not a directory");
    }
    return canonicalRoot;
  } catch (error) {
    if (error instanceof LibrarySourceError) throw error;
    throw mapFsError(error, "Unable to open the selected library root", "invalid-root");
  }
}

function validateLogicalPath(logicalPath: string): string[] {
  if (
    !logicalPath
    || logicalPath.includes("\\")
    || logicalPath.includes("\0")
    || logicalPath.startsWith("/")
    || /^[A-Za-z]:/.test(logicalPath)
  ) {
    throw new LibrarySourceError("unsafe-path", "The library path is not a safe relative path");
  }
  const segments = logicalPath.split("/");
  if (segments.some((segment) => !segment || segment === "." || segment === "..")) {
    throw new LibrarySourceError("unsafe-path", "The library path is not a safe relative path");
  }
  return segments;
}

async function assertCanonicalDirectory(
  canonicalRoot: string,
  absoluteDir: string,
): Promise<void> {
  const canonicalDir = await fs.realpath(absoluteDir);
  assertContained(canonicalRoot, canonicalDir);
  const relative = path.relative(canonicalRoot, canonicalDir);
  if (relative !== path.relative(canonicalRoot, absoluteDir)) {
    throw new LibrarySourceError(
      "unsafe-path",
      "Symbolic links are not traversable through the library boundary",
    );
  }
  const info = await fs.stat(canonicalDir);
  if (!info.isDirectory()) {
    throw new LibrarySourceError("not-file", "The requested library entry is not a directory");
  }
}

function noFollowFlag(): number {
  return typeof constants.O_NOFOLLOW === "number" ? constants.O_NOFOLLOW : 0;
}

function sameFileIdentity(left: Stats, right: Stats): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

function assertContained(root: string, target: string): void {
  const relative = path.relative(root, target);
  if (
    relative === ".."
    || relative.startsWith(`..${path.sep}`)
    || path.isAbsolute(relative)
  ) {
    throw new LibrarySourceError("unsafe-path", "The requested library path escapes its root");
  }
}

function positiveInteger(value: number | undefined, fallback: number): number {
  return Number.isInteger(value) && value! > 0 ? value! : fallback;
}

function boundedPositiveInteger(value: number | undefined, upperBound: number): number {
  return Math.min(positiveInteger(value, upperBound), upperBound);
}

function nonNegativeInteger(value: number | undefined, fallback: number): number {
  return Number.isInteger(value) && value! >= 0 ? value! : fallback;
}

function boundedNonNegativeInteger(value: number | undefined, upperBound: number): number {
  return Math.min(nonNegativeInteger(value, upperBound), upperBound);
}

function mapFsError(
  error: unknown,
  message: string,
  fallback: "invalid-root" | "io" = "io",
): LibrarySourceError {
  if (error instanceof LibrarySourceError) return error;
  const code = (error as NodeJS.ErrnoException | undefined)?.code;
  if (code === "ENOENT") return new LibrarySourceError("not-found", message);
  if (code === "EACCES" || code === "EPERM") {
    return new LibrarySourceError("permission-denied", message);
  }
  return new LibrarySourceError(fallback, message);
}
