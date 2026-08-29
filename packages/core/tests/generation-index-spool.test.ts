import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  StorageGenerationObjectSpool,
  blockObjectChecksum,
  encodeVectorBlock,
  type GenerationObjectReference,
} from "../src/index";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const WRITER = `writer-${"c".repeat(32)}`;

function vectorBytes(value = 1): Uint8Array {
  return encodeVectorBlock({
    rowStart: 0,
    dimension: 2,
    paperOrdinals: new Uint32Array([0]),
    vectors: new Float32Array([value, value + 1]),
  });
}

function seed(path = "objects/vector-0000000000.bin"): Omit<GenerationObjectReference, "byteLength" | "checksum"> {
  return { kind: "vector", path, recordStart: 0, recordCount: 1 };
}

function memoryStorage() {
  const text = new Map<string, string>();
  const binary = new Map<string, Uint8Array>();
  const dirs = new Set<string>();
  let failRemovePath: string | undefined;
  let failRemoveCount = 0;
  let failWriteAfterPersist = false;
  let failDirectoryRemovals = false;
  let writeGate: ReturnType<typeof deferred> | undefined;
  let writeStarted: (() => void) | undefined;
  const normalizePath = (path: string) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
  const storage: StorageAdapter = {
    normalizePath,
    readText: async (path) => {
      const value = text.get(normalizePath(path));
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    },
    writeText: async (path, value) => { text.set(normalizePath(path), value); },
    exists: async (path) => {
      const normalized = normalizePath(path);
      return text.has(normalized) || binary.has(normalized) || dirs.has(normalized);
    },
    mkdir: async (path) => { dirs.add(normalizePath(path)); },
    remove: vi.fn(async (path) => {
      const normalized = normalizePath(path);
      if (normalized === failRemovePath && failRemoveCount > 0) {
        failRemoveCount -= 1;
        throw new Error("injected remove failure");
      }
      const prefix = `${normalized}/`;
      if ([...text.keys(), ...binary.keys(), ...dirs].some((entry) => entry.startsWith(prefix))) {
        throw new Error(`directory not empty: ${normalized}`);
      }
      if (failDirectoryRemovals && dirs.has(normalized)) {
        throw new Error("host remove only supports files");
      }
      text.delete(normalized);
      binary.delete(normalized);
      dirs.delete(normalized);
    }),
    rename: async () => undefined,
    writeBinary: vi.fn(async (path, value) => {
      const normalized = normalizePath(path);
      if (writeGate) {
        const gate = writeGate;
        writeGate = undefined;
        writeStarted?.();
        writeStarted = undefined;
        await gate.promise;
      }
      binary.set(normalized, new Uint8Array(value).slice());
      if (failWriteAfterPersist) {
        failWriteAfterPersist = false;
        throw new Error("injected uncertain write");
      }
    }),
    readBinary: vi.fn(async (path) => {
      const value = binary.get(normalizePath(path));
      if (!value) throw new Error(`missing ${path}`);
      return value.slice().buffer;
    }),
  };
  return {
    storage,
    binary,
    dirs,
    failNextRemove(path: string) { failRemovePath = normalizePath(path); failRemoveCount = 1; },
    failNextWriteAfterPersist() { failWriteAfterPersist = true; },
    rejectDirectoryRemovals() { failDirectoryRemovals = true; },
    blockNextWrite() {
      writeGate = deferred();
      const started = deferred();
      writeStarted = started.resolve;
      return { started: started.promise, release: writeGate.resolve };
    },
  };
}

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((settle) => { resolve = settle; });
  return { promise, resolve };
}

function spool(storage: StorageAdapter, writerToken = WRITER) {
  return new StorageGenerationObjectSpool(
    storage,
    DEFAULT_SETTINGS.output,
    SCOPE,
    IDENTIFICATION,
    { generationId: "gen-sync", writerToken },
  );
}

describe("StorageGenerationObjectSpool", () => {
  it("persists verified objects in a writer-isolated generation spool and removes them", async () => {
    const memory = memoryStorage();
    const target = spool(memory.storage);
    const bytes = vectorBytes();

    const reference = await target.put(seed(), bytes);

    expect(reference).toEqual({
      ...seed(),
      byteLength: bytes.byteLength,
      checksum: blockObjectChecksum(bytes),
    });
    expect(target.paths.directory).toContain("/personal-library-search-index/");
    expect(target.paths.directory).toContain("/spool/gen-sync/");
    expect(target.paths.directory).toContain(WRITER);
    expect(await target.read(reference)).toEqual(bytes);
    expect(memory.binary.get(`${target.paths.directory}/${reference.path}`)).toEqual(bytes);

    await target.removeAll();
    expect([...memory.binary.keys()].filter((path) => path.startsWith(target.paths.directory))).toEqual([]);
    expect([...memory.dirs].filter((path) => path.startsWith(target.paths.directory))).toEqual([]);
    await expect(target.removeAll()).resolves.toBeUndefined();
  });

  it("rejects unsafe or duplicate object paths without overwriting the first object", async () => {
    const memory = memoryStorage();
    const target = spool(memory.storage);
    const bytes = vectorBytes();
    const first = await target.put(seed(), bytes);

    await expect(target.put(seed(), vectorBytes(9))).rejects.toMatchObject({ code: "invalid" });
    await expect(target.put(seed("../escape.bin"), bytes)).rejects.toMatchObject({ code: "invalid" });
    await expect(target.read(first)).resolves.toEqual(bytes);
  });

  it("fails closed when durable binary capabilities are absent or persisted bytes change", async () => {
    const unsupported = memoryStorage();
    delete unsupported.storage.readBinary;
    await expect(spool(unsupported.storage).put(seed(), vectorBytes())).rejects.toMatchObject({
      code: "capability-unsupported",
    });

    const memory = memoryStorage();
    const target = spool(memory.storage);
    const reference = await target.put(seed(), vectorBytes());
    memory.binary.set(`${target.paths.directory}/${reference.path}`, vectorBytes(12));
    await expect(target.read(reference)).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });

  it("tracks uncertain writes for cleanup and retries only the paths that remain", async () => {
    const memory = memoryStorage();
    const target = spool(memory.storage);
    const sibling = spool(memory.storage, `writer-${"d".repeat(32)}`);
    const siblingReference = await sibling.put(seed(), vectorBytes(20));
    memory.failNextWriteAfterPersist();

    await expect(target.put(seed(), vectorBytes())).rejects.toMatchObject({ code: "write-failed" });
    const uncertainPath = `${target.paths.directory}/${seed().path}`;
    expect(memory.binary.has(uncertainPath)).toBe(true);
    memory.failNextRemove(uncertainPath);
    await expect(target.removeAll()).rejects.toMatchObject({ code: "cleanup-failed" });
    expect(memory.binary.has(uncertainPath)).toBe(true);

    await expect(target.removeAll()).resolves.toBeUndefined();
    expect(memory.binary.has(uncertainPath)).toBe(false);
    expect(memory.binary.has(`${sibling.paths.directory}/${siblingReference.path}`)).toBe(true);
  });

  it("stays closed after partial cleanup failure while allowing removeAll to retry", async () => {
    const memory = memoryStorage();
    const target = spool(memory.storage);
    const reference = await target.put(seed(), vectorBytes());
    const path = `${target.paths.directory}/${reference.path}`;
    memory.failNextRemove(path);

    await expect(target.removeAll()).rejects.toMatchObject({ code: "cleanup-failed" });
    await expect(target.read(reference)).rejects.toMatchObject({ code: "invalid" });
    await expect(target.put(seed("objects/vector-0000000001.bin"), vectorBytes(5)))
      .rejects.toMatchObject({ code: "invalid" });

    await expect(target.removeAll()).resolves.toBeUndefined();
    expect(memory.binary.has(path)).toBe(false);
  });

  it("treats object deletion as authoritative when the host cannot remove empty directories", async () => {
    const memory = memoryStorage();
    memory.rejectDirectoryRemovals();
    const target = spool(memory.storage);
    const reference = await target.put(seed(), vectorBytes());

    await expect(target.removeAll()).resolves.toBeUndefined();
    expect(memory.binary.has(`${target.paths.directory}/${reference.path}`)).toBe(false);
    expect([...memory.dirs].some((path) => path.startsWith(target.paths.directory))).toBe(true);
  });

  it("waits for an in-flight put before cleanup so a late write cannot revive an orphan", async () => {
    const memory = memoryStorage();
    const target = spool(memory.storage);
    const gate = memory.blockNextWrite();
    const writing = target.put(seed(), vectorBytes());
    await gate.started;

    const cleanup = target.removeAll();
    await Promise.resolve();
    expect(memory.storage.remove).not.toHaveBeenCalled();
    gate.release();
    const reference = await writing;
    await cleanup;

    expect(memory.binary.has(`${target.paths.directory}/${reference.path}`)).toBe(false);
  });
});
