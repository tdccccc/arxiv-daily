import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { AtomMetadataCache } from "../src/pipeline/atom-metadata-cache";

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
});
