import type { StorageAdapter } from "../../core/adapters";
import type { OutputSettings } from "../../settings/types";
import {
  BINARY_BLOCK_HEADER_BYTES,
  MAX_BINARY_OBJECT_BYTES,
  MAX_GENERATION_CHUNKS,
  blockObjectChecksum,
  deriveFullTextGenerationIndexPaths,
  type GenerationObjectKind,
  type GenerationObjectReference,
} from "./generation-index-format";
import type { GenerationObjectSpool } from "./generation-index-builder";

export interface StorageGenerationObjectSpoolOptions {
  readonly generationId: string;
  /** High-entropy writer identity; isolates concurrent builds of the same generation. */
  readonly writerToken: string;
}

export interface StorageGenerationObjectSpoolPaths {
  readonly rootDirectory: string;
  readonly generationDirectory: string;
  readonly directory: string;
  readonly objectsDirectory: string;
}

export type StorageGenerationObjectSpoolErrorCode =
  | "invalid"
  | "capability-unsupported"
  | "write-failed"
  | "corrupt-or-unreadable"
  | "cleanup-failed";

export class StorageGenerationObjectSpoolError extends Error {
  constructor(
    message: string,
    readonly code: StorageGenerationObjectSpoolErrorCode,
    options: ErrorOptions = {},
  ) {
    super(message, options);
    this.name = "StorageGenerationObjectSpoolError";
  }
}

/**
 * Temporary, generation-scoped binary spool used by the multi-pass builder.
 *
 * The generation id groups related attempts while writerToken gives every
 * concurrent attempt a private namespace. Only paths accepted by the
 * generation descriptor format can be written. A path is registered for
 * cleanup before writeBinary is invoked, so a thrown write with an uncertain
 * outcome remains recoverable by removeAll().
 */
export class StorageGenerationObjectSpool implements GenerationObjectSpool {
  readonly paths: StorageGenerationObjectSpoolPaths;

  private readonly references = new Map<string, GenerationObjectReference>();
  private readonly pendingFiles = new Set<string>();
  private readonly pendingDirectories = new Set<string>();
  private disposed = false;
  private closing = false;
  private activeOperations = 0;
  private operationsSettled: Promise<void> | undefined;
  private settleOperations: (() => void) | undefined;
  private cleanupInFlight: Promise<void> | undefined;

  constructor(
    private readonly storage: StorageAdapter,
    output: OutputSettings,
    scopeFingerprint: string,
    identificationFingerprint: string,
    options: StorageGenerationObjectSpoolOptions,
  ) {
    try {
      validateGenerationId(options?.generationId);
      validateWriterToken(options?.writerToken);
      const index = deriveFullTextGenerationIndexPaths(
        storage,
        output,
        scopeFingerprint,
        identificationFingerprint,
      );
      const rootDirectory = storage.normalizePath(`${index.directory}/spool`);
      const generationDirectory = storage.normalizePath(`${rootDirectory}/${options.generationId}`);
      const directory = storage.normalizePath(`${generationDirectory}/${options.writerToken}`);
      const objectsDirectory = storage.normalizePath(`${directory}/objects`);
      this.paths = { rootDirectory, generationDirectory, directory, objectsDirectory };
    } catch (caught) {
      throw spoolError("invalid", "invalid storage generation spool identity", caught);
    }
  }

  async put(
    seed: Omit<GenerationObjectReference, "byteLength" | "checksum">,
    bytes: Uint8Array,
  ): Promise<GenerationObjectReference> {
    const release = this.beginOperation();
    try {
      requireBinaryCapabilities(this.storage);
      validateSeed(seed);
      if (!(bytes instanceof Uint8Array)
        || bytes.byteLength < BINARY_BLOCK_HEADER_BYTES
        || bytes.byteLength > MAX_BINARY_OBJECT_BYTES) {
        throw spoolError("invalid", "generation spool object bytes violate the binary object bounds");
      }

      const path = this.objectPath(seed.path);
      if (this.pendingFiles.has(path) || this.references.has(seed.path)) {
        throw spoolError("invalid", `generation spool object path was already used: ${seed.path}`);
      }
      const copy = bytes.slice();
      const reference: GenerationObjectReference = {
        ...seed,
        byteLength: copy.byteLength,
        checksum: blockObjectChecksum(copy),
      };

      // Register before creating directories or writing so uncertain partial
      // effects remain owned by this instance's cleanup contract.
      this.pendingFiles.add(path);
      this.pendingDirectories.add(this.paths.directory);
      this.pendingDirectories.add(this.paths.objectsDirectory);
      try {
        await ensureDirDeep(this.storage, this.paths.objectsDirectory);
        await this.storage.writeBinary!(path, exactArrayBuffer(copy));
        this.references.set(seed.path, reference);
        return { ...reference };
      } catch (caught) {
        throw spoolError("write-failed", `failed to persist generation spool object: ${seed.path}`, caught);
      }
    } finally {
      release();
    }
  }

  async read(reference: GenerationObjectReference): Promise<Uint8Array> {
    const release = this.beginOperation();
    try {
      requireBinaryCapabilities(this.storage);
      validateReference(reference);
      const expected = this.references.get(reference.path);
      if (!expected || !sameReference(expected, reference)) {
        throw spoolError("invalid", `generation spool reference is not owned by this build: ${reference.path}`);
      }
      const path = this.objectPath(reference.path);
      try {
        const buffer = await this.storage.readBinary!(path);
        if (!(buffer instanceof ArrayBuffer)) throw new Error("storage returned a non-ArrayBuffer binary value");
        const bytes = new Uint8Array(buffer);
        if (bytes.byteLength !== reference.byteLength
          || bytes.byteLength > MAX_BINARY_OBJECT_BYTES
          || blockObjectChecksum(bytes) !== reference.checksum) {
          throw new Error("persisted generation spool object failed reference verification");
        }
        return bytes.slice();
      } catch (caught) {
        if (caught instanceof StorageGenerationObjectSpoolError) throw caught;
        throw spoolError(
          "corrupt-or-unreadable",
          `failed to read or verify generation spool object: ${reference.path}`,
          caught,
        );
      }
    } finally {
      release();
    }
  }

  removeAll(): Promise<void> {
    if (this.disposed) return Promise.resolve();
    if (this.cleanupInFlight) return this.cleanupInFlight;
    this.closing = true;
    const attempt = (async () => {
      await this.waitForOperations();
      await this.removeAllOnce();
    })();
    this.cleanupInFlight = attempt;
    void attempt.then(
      () => {
        this.disposed = true;
        this.references.clear();
        this.cleanupInFlight = undefined;
      },
      () => {
        this.cleanupInFlight = undefined;
      },
    );
    return attempt;
  }

  private async removeAllOnce(): Promise<void> {
    let firstFailure: unknown;
    for (const path of [...this.pendingFiles].sort().reverse()) {
      try {
        if (await this.storage.exists(path)) await this.storage.remove(path);
        this.pendingFiles.delete(path);
        const logicalPrefix = `${this.paths.directory}/`;
        if (path.startsWith(logicalPrefix)) this.references.delete(path.slice(logicalPrefix.length));
      } catch (caught) {
        firstFailure ??= caught;
      }
    }
    // Do not invoke a host's potentially recursive directory removal while an
    // object remains uncertain; a retry must retain an exact file target.
    if (this.pendingFiles.size > 0) {
      throw spoolError("cleanup-failed", "failed to remove every generation spool object", firstFailure);
    }

    const directories = [...this.pendingDirectories]
      .sort((left, right) => depth(right) - depth(left) || right.localeCompare(left));
    for (const path of directories) {
      try {
        if (await this.storage.exists(path)) await this.storage.remove(path);
      } catch (caught) {
        firstFailure ??= caught;
      }
      // StorageAdapter does not guarantee directory deletion (the Obsidian
      // adapter exposes file removal only). Empty spool directories are not
      // authoritative data; deleting them is best-effort and must not make a
      // successful object cleanup fail or poison builder promotion.
      this.pendingDirectories.delete(path);
    }
  }

  private objectPath(logicalPath: string): string {
    const path = this.storage.normalizePath(`${this.paths.directory}/${logicalPath}`);
    if (!path.startsWith(`${this.paths.objectsDirectory}/`)) {
      throw spoolError("invalid", `generation spool object escapes its objects directory: ${logicalPath}`);
    }
    return path;
  }

  private assertAvailable(): void {
    if (this.disposed || this.closing || this.cleanupInFlight) {
      throw spoolError("invalid", "generation spool is cleaning up or already disposed");
    }
  }

  private beginOperation(): () => void {
    this.assertAvailable();
    this.activeOperations += 1;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.activeOperations -= 1;
      if (this.activeOperations === 0) {
        this.settleOperations?.();
        this.settleOperations = undefined;
        this.operationsSettled = undefined;
      }
    };
  }

  private waitForOperations(): Promise<void> {
    if (this.activeOperations === 0) return Promise.resolve();
    if (!this.operationsSettled) {
      this.operationsSettled = new Promise<void>((resolve) => { this.settleOperations = resolve; });
    }
    return this.operationsSettled;
  }
}

function requireBinaryCapabilities(storage: StorageAdapter): void {
  if (typeof storage.readBinary !== "function" || typeof storage.writeBinary !== "function") {
    throw spoolError(
      "capability-unsupported",
      "generation spool requires StorageAdapter readBinary and writeBinary capabilities",
    );
  }
}

function validateSeed(seed: Omit<GenerationObjectReference, "byteLength" | "checksum">): void {
  if (!seed || typeof seed !== "object" || !isObjectKind(seed.kind)
    || typeof seed.path !== "string" || !/^objects\/[a-z0-9][a-z0-9._-]{0,127}$/.test(seed.path)
    || !Number.isSafeInteger(seed.recordStart) || seed.recordStart < 0
    || !Number.isSafeInteger(seed.recordCount) || seed.recordCount < 1
    || seed.recordStart + seed.recordCount > MAX_GENERATION_CHUNKS) {
    throw spoolError("invalid", "invalid generation spool object reference seed");
  }
}

function validateReference(reference: GenerationObjectReference): void {
  validateSeed(reference);
  if (!Number.isSafeInteger(reference.byteLength)
    || reference.byteLength < BINARY_BLOCK_HEADER_BYTES
    || reference.byteLength > MAX_BINARY_OBJECT_BYTES
    || typeof reference.checksum !== "string"
    || !/^sha256:[a-f0-9]{64}$/.test(reference.checksum)) {
    throw spoolError("invalid", "invalid generation spool object reference");
  }
}

function isObjectKind(value: unknown): value is GenerationObjectKind {
  return value === "vector" || value === "evidence" || value === "paper-metadata"
    || value === "lexical-postings" || value === "lexical-dictionary";
}

function sameReference(left: GenerationObjectReference, right: GenerationObjectReference): boolean {
  return left.kind === right.kind && left.path === right.path
    && left.byteLength === right.byteLength && left.recordStart === right.recordStart
    && left.recordCount === right.recordCount && left.checksum === right.checksum;
}

function validateGenerationId(value: unknown): asserts value is string {
  if (typeof value !== "string" || !/^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/.test(value)) {
    throw new Error("generationId must be a bounded lowercase path segment");
  }
}

function validateWriterToken(value: unknown): asserts value is string {
  if (typeof value !== "string" || !/^[a-z0-9](?:[a-z0-9-]{30,126}[a-z0-9])$/.test(value)) {
    throw new Error("writerToken must contain 32-128 lowercase alphanumeric/hyphen characters");
  }
}

async function ensureDirDeep(storage: StorageAdapter, directory: string): Promise<void> {
  const parts = storage.normalizePath(directory).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (await storage.exists(current)) continue;
    try { await storage.mkdir(current); }
    catch (caught) { if (!(await storage.exists(current))) throw caught; }
  }
}

function exactArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

function depth(path: string): number {
  return path.split("/").filter(Boolean).length;
}

function spoolError(
  code: StorageGenerationObjectSpoolErrorCode,
  message: string,
  cause?: unknown,
): StorageGenerationObjectSpoolError {
  return new StorageGenerationObjectSpoolError(message, code, cause === undefined ? {} : { cause });
}
