export type LibraryEntryType = "file" | "folder" | "ignored";

export interface LibrarySourceEntry {
  path: string;
  type: LibraryEntryType;
  size?: number;
  ignoredReason?: "symbolic-link" | "unsupported-entry";
}

export interface LibraryInventory {
  entries: LibrarySourceEntry[];
  truncated: boolean;
}

export interface LibraryInventoryOptions {
  maxEntries?: number;
  maxDepth?: number;
  signal?: AbortSignal;
}

export interface LibraryReadOptions {
  maxBytes?: number;
  signal?: AbortSignal;
}

export interface ScopedLibrarySource {
  inventory(options?: LibraryInventoryOptions): Promise<LibraryInventory>;
  readBinary(path: string, options?: LibraryReadOptions): Promise<ArrayBuffer>;
}

export type LibrarySourceErrorKind =
  | "invalid-root"
  | "unsafe-path"
  | "not-found"
  | "not-file"
  | "limit-exceeded"
  | "permission-denied"
  | "io";

export class LibrarySourceError extends Error {
  constructor(
    public readonly kind: LibrarySourceErrorKind,
    message: string,
  ) {
    super(message);
    this.name = "LibrarySourceError";
  }
}
