import { beforeEach, describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  AtomMetadataCache,
  resetAtomMetadataCacheForTests,
} from "../src/pipeline/atom-metadata-cache";

const paper = {
  id: "2605.08080",
  title: "A paper",
  authors: "A. Author",
  abstract: "An abstract.",
  published: "2026-05-01T00:00:00Z",
  updated: "2026-05-02T00:00:00Z",
  primaryCategory: "astro-ph.CO",
  categories: ["astro-ph.CO"],
};

function makeStorage(atomic = false) {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const writeText = vi.fn(async (path: string, content: string) => { files[path] = content; });
  const writeTextAtomic = vi.fn(async (path: string, content: string) => { files[path] = content; });
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""),
    readText: async (path) => {
      if (!(path in files)) throw new Error("missing");
      return files[path]!;
    },
    writeText,
    ...(atomic ? { writeTextAtomic } : {}),
    exists: async (path) => path in files || dirs.has(path),
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => { delete files[path]; dirs.delete(path); },
    rename: async (from, to) => { files[to] = files[from]!; delete files[from]; },
    list: async (dir) => Object.keys(files)
      .filter((path) => path.startsWith(`${dir}/`) && !path.slice(dir.length + 1).includes("/"))
      .map((path) => ({ path, type: "file" as const })),
  };
  return { files, dirs, storage, writeText, writeTextAtomic };
}

describe("AtomMetadataCache", () => {
  beforeEach(async () => {
    await resetAtomMetadataCacheForTests();
  });
  it("persists a versioned positive entry without refreshing cachedAt on read", async () => {
    const { files, storage } = makeStorage();
    let now = new Date("2026-07-01T00:00:00Z");
    const cache = new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage, now: () => now });
    await cache.set(paper.id, paper);
    const raw = files["cache/atom-metadata/2605.08080.json"]!;
    expect(JSON.parse(raw)).toMatchObject({ schemaVersion: 1, cachedAt: now.toISOString(), paper });

    now = new Date("2026-07-02T00:00:00Z");
    expect(await cache.get("2605.08080v4")).toEqual(paper);
    expect(files["cache/atom-metadata/2605.08080.json"]).toBe(raw);
  });

  it("uses atomic writes when supported and normal writes otherwise", async () => {
    const atomic = makeStorage(true);
    await new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage: atomic.storage }).set(paper.id, paper);
    expect(atomic.writeTextAtomic).toHaveBeenCalledOnce();
    expect(atomic.writeText).not.toHaveBeenCalled();

    const normal = makeStorage();
    await new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage: normal.storage }).set(paper.id, paper);
    expect(normal.writeText).toHaveBeenCalledOnce();
  });

  it.each([
    ["malformed", "not json"],
    ["mismatched", JSON.stringify({ schemaVersion: 1, cachedAt: "2026-07-01T00:00:00Z", paper: { ...paper, id: "2605.99999" } })],
    ["incomplete", JSON.stringify({ schemaVersion: 1, cachedAt: "2026-07-01T00:00:00Z", paper: { ...paper, abstract: "" } })],
    ["expired", JSON.stringify({ schemaVersion: 1, cachedAt: "2026-06-01T00:00:00Z", paper })],
  ])("treats %s entries as misses and removes them", async (_label, raw) => {
    const { files, dirs, storage } = makeStorage();
    dirs.add("cache/atom-metadata");
    files["cache/atom-metadata/2605.08080.json"] = raw;
    const cache = new AtomMetadataCache({
      rootDir: "cache", expiryDays: 7, storage, now: () => new Date("2026-07-02T00:00:00Z"),
    });
    expect(await cache.get(paper.id)).toBeNull();
    expect(files["cache/atom-metadata/2605.08080.json"]).toBeUndefined();
  });

  it("cleans malformed, expired, and filename-mismatched entries", async () => {
    const { files, dirs, storage } = makeStorage();
    dirs.add("cache/atom-metadata");
    files["cache/atom-metadata/bad.json"] = "bad";
    files["cache/atom-metadata/2605.99999.json"] = JSON.stringify({ schemaVersion: 1, cachedAt: "2026-07-01T00:00:00Z", paper });
    files["cache/atom-metadata/2605.08080.json"] = JSON.stringify({ schemaVersion: 1, cachedAt: "2026-06-01T00:00:00Z", paper });
    const cache = new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage, now: () => new Date("2026-07-02T00:00:00Z") });
    expect(await cache.cleanupExpired()).toBe(3);
    expect(files).toEqual({});
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, -1])(
    "fails closed for invalid expiryDays %s in get and cleanup",
    async (expiryDays) => {
      const { files, dirs, storage } = makeStorage();
      dirs.add("cache/atom-metadata");
      const path = "cache/atom-metadata/2605.08080.json";
      const raw = JSON.stringify({ schemaVersion: 1, cachedAt: "2026-07-01T00:00:00Z", paper });
      files[path] = raw;
      const cache = new AtomMetadataCache({
        rootDir: "cache", expiryDays, storage, now: () => new Date("2026-07-02T00:00:00Z"),
      });
      expect(await cache.get(paper.id)).toBeNull();
      expect(files[path]).toBeUndefined();

      files[path] = raw;
      expect(await cache.cleanupExpired()).toBe(1);
      expect(files[path]).toBeUndefined();
    },
  );

  it("rejects future cachedAt consistently in get and cleanup", async () => {
    const { files, dirs, storage } = makeStorage();
    dirs.add("cache/atom-metadata");
    const path = "cache/atom-metadata/2605.08080.json";
    const future = JSON.stringify({ schemaVersion: 1, cachedAt: "2026-07-03T00:00:00Z", paper });
    const cache = new AtomMetadataCache({
      rootDir: "cache", expiryDays: 7, storage, now: () => new Date("2026-07-02T00:00:00Z"),
    });
    files[path] = future;
    expect(await cache.get(paper.id)).toBeNull();
    expect(files[path]).toBeUndefined();

    files[path] = future;
    expect(await cache.cleanupExpired()).toBe(1);
    expect(files[path]).toBeUndefined();
  });

  it("serializes instances so a stale read cannot remove a fresh write", async () => {
    const { files, dirs, storage } = makeStorage();
    dirs.add("cache/atom-metadata");
    const path = "cache/atom-metadata/2605.08080.json";
    files[path] = "malformed";
    let releaseRead!: () => void;
    const readStarted = new Promise<void>((resolve) => {
      storage.readText = vi.fn(async (readPath) => {
        const snapshot = files[readPath];
        resolve();
        await new Promise<void>((release) => { releaseRead = release; });
        return snapshot!;
      });
    });
    const first = new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage });
    const second = new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage });

    const staleGet = first.get(paper.id);
    await readStarted;
    const freshSet = second.set(paper.id, paper);
    releaseRead();
    await expect(staleGet).resolves.toBeNull();
    await freshSet;

    expect(JSON.parse(files[path]!)).toMatchObject({ paper });
  });

  it("serializes cleanup with directory creation and non-atomic writes", async () => {
    const { files, storage } = makeStorage();
    let releaseWrite!: () => void;
    const writeStarted = new Promise<void>((resolve) => {
      storage.writeText = vi.fn(async (path, content) => {
        resolve();
        await new Promise<void>((release) => { releaseWrite = release; });
        files[path] = content;
      });
    });
    const writer = new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage });
    const cleaner = new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage });

    const set = writer.set(paper.id, paper);
    await writeStarted;
    const cleanup = cleaner.cleanupExpired();
    releaseWrite();

    await set;
    await expect(cleanup).resolves.toBe(0);
    await expect(cleaner.get(paper.id)).resolves.toEqual(paper);
  });

  it("does not let a never-settling operation block another root or storage", async () => {
    const first = makeStorage();
    const second = makeStorage();
    first.storage.writeText = vi.fn((path, content) => {
      if (path.startsWith("stalled/")) return new Promise<void>(() => {});
      first.files[path] = content;
      return Promise.resolve();
    });
    const stalled = new AtomMetadataCache({
      rootDir: "stalled",
      expiryDays: 7,
      storage: first.storage,
      operationLeaseMs: 10,
    }).set(paper.id, paper);
    await vi.waitFor(() => expect(first.storage.writeText).toHaveBeenCalledOnce());

    const otherRoot = new AtomMetadataCache({ rootDir: "other", expiryDays: 7, storage: first.storage });
    const otherStorage = new AtomMetadataCache({ rootDir: "stalled", expiryDays: 7, storage: second.storage });
    await expect(otherRoot.set(paper.id, paper)).resolves.toBeUndefined();
    await expect(otherStorage.set(paper.id, paper)).resolves.toBeUndefined();
    await expect(stalled).rejects.toThrow("operation lease expired");
  });

  it("recovers the same namespace after a never-settling operation lease expires", async () => {
    const { storage, writeText } = makeStorage();
    storage.writeText = vi.fn()
      .mockImplementationOnce(() => new Promise<void>(() => {}))
      .mockImplementation(writeText);
    const cache = new AtomMetadataCache({
      rootDir: "cache",
      expiryDays: 7,
      storage,
      operationLeaseMs: 10,
    });

    await expect(cache.set(paper.id, paper)).rejects.toThrow("operation lease expired");
    await expect(cache.set(paper.id, paper)).resolves.toBeUndefined();
    await expect(cache.get(paper.id)).resolves.toEqual(paper);
  });

  it("cancels a queued cache wait without starting its physical operation", async () => {
    const { storage } = makeStorage();
    let release!: () => void;
    storage.writeText = vi.fn()
      .mockImplementationOnce(() => new Promise<void>((resolve) => { release = resolve; }))
      .mockResolvedValue(undefined);
    const cache = new AtomMetadataCache({ rootDir: "cache", expiryDays: 7, storage });
    const first = cache.set(paper.id, paper);
    await vi.waitFor(() => expect(storage.writeText).toHaveBeenCalledOnce());
    const controller = new AbortController();
    const queued = cache.set(paper.id, paper, controller.signal);
    controller.abort("stop cache wait");

    await expect(queued).rejects.toThrow("stop cache wait");
    expect(storage.writeText).toHaveBeenCalledOnce();
    release();
    await first;
  });
});
