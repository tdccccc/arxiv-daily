import type { Stats } from "node:fs";
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

export async function openScopedLibrarySource(
  selectedRoot: string,
  options: OpenScopedLibrarySourceOptions = {},
): Promise<ScopedLibrarySource> {
  const canonicalRoot = await canonicalizeRoot(selectedRoot);
  return new NodeScopedLibrarySource(canonicalRoot, options);
}

class NodeScopedLibrarySource implements ScopedLibrarySource {
  private readonly maxEntries: number;
  private readonly maxDepth: number;
  private readonly maxReadBytes: number;

  constructor(
    private readonly canonicalRoot: string,
    options: OpenScopedLibrarySourceOptions,
  ) {
    this.maxEntries = positiveInteger(options.maxEntries, DEFAULT_MAX_ENTRIES);
    this.maxDepth = nonNegativeInteger(options.maxDepth, DEFAULT_MAX_DEPTH);
    this.maxReadBytes = positiveInteger(options.maxReadBytes, DEFAULT_MAX_READ_BYTES);
  }

  async inventory(options: LibraryInventoryOptions = {}): Promise<LibraryInventory> {
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
        directory = await fs.opendir(absoluteDir);
      } catch (error) {
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
          const absolutePath = path.join(this.canonicalRoot, ...relativePath.split("/"));
          let info: Stats;
          try {
            info = await fs.lstat(absolutePath);
          } catch (error) {
            throw mapFsError(error, "Unable to inspect a library entry");
          }

          if (info.isSymbolicLink()) {
            entries.push({
              path: relativePath,
              type: "ignored",
              ignoredReason: "symbolic-link",
            });
            continue;
          }
          if (info.isDirectory()) {
            entries.push({ path: relativePath, type: "folder" });
            if (depth < maxDepth) {
              await visit(relativePath, depth + 1);
              if (truncated) return;
            } else {
              truncated = true;
            }
            continue;
          }
          if (info.isFile()) {
            entries.push({ path: relativePath, type: "file", size: info.size });
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
    };

    await visit("", 0);
    return { entries, truncated };
  }

  async readBinary(
    logicalPath: string,
    options: LibraryReadOptions = {},
  ): Promise<ArrayBuffer> {
    throwIfCancelled(options.signal);
    const segments = validateLogicalPath(logicalPath);
    const absolutePath = path.join(this.canonicalRoot, ...segments);
    await this.assertNoSymbolicLink(segments);

    let canonicalTarget: string;
    let info: Stats;
    try {
      canonicalTarget = await fs.realpath(absolutePath);
      assertContained(this.canonicalRoot, canonicalTarget);
      info = await fs.stat(canonicalTarget);
    } catch (error) {
      if (error instanceof LibrarySourceError) throw error;
      throw mapFsError(error, "Unable to inspect the requested library file");
    }
    if (!info.isFile()) {
      throw new LibrarySourceError("not-file", "The requested library entry is not a file");
    }

    const maxBytes = boundedPositiveInteger(options.maxBytes, this.maxReadBytes);
    if (info.size > maxBytes) {
      throw new LibrarySourceError(
        "limit-exceeded",
        "The requested library file exceeds the configured read limit",
      );
    }

    throwIfCancelled(options.signal);
    let buffer: Buffer;
    try {
      buffer = await fs.readFile(canonicalTarget, { signal: options.signal });
    } catch (error) {
      if (isCancellationError(error)) throw error;
      throw mapFsError(error, "Unable to read the requested library file");
    }
    throwIfCancelled(options.signal);
    return Uint8Array.from(buffer).buffer;
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
