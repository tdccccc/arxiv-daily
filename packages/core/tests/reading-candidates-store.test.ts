import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  emptyReadingCandidatesDocument,
  upsertReadingCandidate,
  type ReadingCandidateRecord,
  type ReadingCandidatesDocument,
} from "../src/library/reading-candidates/reading-candidates";
import {
  ReadingCandidatesStore,
  ReadingCandidatesStoreError,
  deriveReadingCandidatesPaths,
} from "../src/library/reading-candidates/reading-candidates-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const otherScope = `sha256:${"c".repeat(64)}`;
const firstTime = new Date("2026-08-13T09:00:00.000Z");
const secondTime = new Date("2026-08-13T10:00:00.000Z");
const directory = `arxiv-daily/.index/personal-library-reading-candidates/${"a".repeat(64)}/${"b".repeat(64)}`;
const documentPath = `${directory}/reading-candidates.json`;
const backupPath = `${documentPath}.backup`;

function candidate(index: number): ReadingCandidateRecord {
  return {
    paperKey: `arxiv:2608.${String(index).padStart(5, "0")}`,
    arxivId: `2608.${String(index).padStart(5, "0")}`,
    title: `Candidate ${index}`,
    authors: "A. Author",
    topic: "astrophysics",
    source: {
      kind: "library",
      manualTopics: [],
      directions: [{ id: "direction-1", name: "Cosmology" }],
      reportPath: "arxiv-daily/daily/2026-08-12.md",
      reportDate: "2026-08-12",
    },
    relatedPriorWorks: [],
    savedAt: "2026-08-12T00:00:00.000Z",
    updatedAt: "2026-08-12T00:00:00.000Z",
  };
}

function documentWith(entries: ReadingCandidateRecord[]): ReadingCandidatesDocument {
  const doc = emptyReadingCandidatesDocument(scope, identification, "2026-08-13T00:00:00.000Z");
  for (const entry of entries) doc.candidates[entry.paperKey] = entry;
  return doc;
}

function makeStorage(atomic = true) {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  let atomicImplementation: ((path: string, content: string) => Promise<void>) | null = null;
  const normalizePath = vi.fn((path: string) => path.replace(/\\/g, "/")
    .replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""));
  const writeTextAtomic = vi.fn(async (path: string, content: string) => {
    if (atomicImplementation) return await atomicImplementation(path, content);
    files[path] = content;
  });
  const storage: StorageAdapter = {
    normalizePath,
    readText: vi.fn(async (path) => {
      if (!(path in files)) throw new Error(`unreadable ${path}`);
      return files[path]!;
    }),
    writeText: vi.fn(async (path, content) => { files[path] = content; }),
    ...(atomic ? { writeTextAtomic } : {}),
    exists: vi.fn(async (path) => path in files || dirs.has(path)),
    mkdir: vi.fn(async (path) => { dirs.add(path); }),
    remove: vi.fn(async (path) => { delete files[path]; dirs.delete(path); }),
    rename: vi.fn(async (from, to) => { files[to] = files[from]!; delete files[from]; }),
  };
  return {
    files, storage, writeTextAtomic,
    setAtomicImplementation(value: typeof atomicImplementation) { atomicImplementation = value; },
  };
}

function store(storage: StorageAdapter, now = () => firstTime) {
  return new ReadingCandidatesStore(storage, DEFAULT_SETTINGS.output, scope, identification, { now });
}

describe("ReadingCandidatesStore", () => {
  it("derives sharded paths and loads an empty document when nothing is stored", async () => {
    const { storage } = makeStorage();
    const instance = store(storage);
    expect(deriveReadingCandidatesPaths(storage, DEFAULT_SETTINGS.output, scope, identification)).toEqual({
      directory,
      documentPath,
      backupPath,
    });
    await expect(instance.load()).resolves.toMatchObject({
      schemaVersion: 1,
      revision: 0,
      candidates: {},
    });
  });

  it("persists documents with CAS revision bumps and reloads them", async () => {
    const { storage } = makeStorage();
    const instance = store(storage);
    const saved = await instance.replace(documentWith([candidate(1)]), 0);
    expect(saved.revision).toBe(1);
    const loaded = await instance.load();
    expect(loaded.revision).toBe(1);
    expect(loaded.candidates["arxiv:2608.00001"]?.title).toBe("Candidate 1");
  });

  it("rejects stale revisions", async () => {
    const { storage } = makeStorage();
    const instance = store(storage);
    await instance.replace(documentWith([candidate(1)]), 0);
    await expect(instance.replace(documentWith([candidate(2)]), 0)).rejects.toMatchObject({
      name: "ReadingCandidatesStoreError",
      code: "stale",
      expectedRevision: 0,
      currentRevision: 1,
    });
  });

  it("replays an identical semantic state idempotently without writing", async () => {
    const { storage, writeTextAtomic } = makeStorage();
    const instance = store(storage);
    const document = documentWith([candidate(1)]);
    await instance.replace(document, 0);
    writeTextAtomic.mockClear();
    const replayed = await instance.replace(documentWith([candidate(1)]), 1);
    expect(replayed.revision).toBe(1);
    expect(writeTextAtomic).not.toHaveBeenCalled();
  });

  it("recovers from a valid backup when the primary is corrupt and repairs it", async () => {
    const { files, storage } = makeStorage();
    const instance = store(storage);
    await instance.replace(documentWith([candidate(1)]), 0);
    files[documentPath] = "{corrupt";
    const loaded = await instance.load();
    expect(loaded.candidates["arxiv:2608.00001"]?.title).toBe("Candidate 1");
    expect(files[documentPath]).not.toBe("{corrupt");
  });

  it("fails closed when both primary and backup are corrupt", async () => {
    const { files, storage } = makeStorage();
    const instance = store(storage);
    await instance.replace(documentWith([candidate(1)]), 0);
    files[documentPath] = "{corrupt";
    files[backupPath] = "{corrupt-too";
    await expect(instance.load()).rejects.toMatchObject({
      name: "ReadingCandidatesStoreError",
      code: "corrupt-or-unreadable",
    });
  });

  it("rejects documents bound to a different library identity", async () => {
    const { files, storage } = makeStorage();
    const instance = store(storage);
    const foreign = emptyReadingCandidatesDocument(otherScope, identification, "2026-08-13T00:00:00.000Z");
    await expect(instance.replace(foreign, 0)).rejects.toMatchObject({
      name: "ReadingCandidatesStoreError",
      code: "invalid",
    });
    // A persisted foreign-identity document is rejected with a typed incompatible error on load.
    files[documentPath] = `${JSON.stringify(foreign, null, 2)}\n`;
    await expect(instance.load()).rejects.toMatchObject({
      name: "ReadingCandidatesStoreError",
      code: "incompatible",
    });
  });

  it("rejects invalid documents", async () => {
    const { storage } = makeStorage();
    const instance = store(storage);
    await expect(instance.replace({ ...documentWith([]), schemaVersion: 99 }, 0)).rejects.toMatchObject({
      name: "ReadingCandidatesStoreError",
      code: "invalid",
    });
  });

  it("rejects persistence when atomic writes are unavailable", async () => {
    const { storage } = makeStorage(false);
    const instance = store(storage);
    await expect(instance.replace(documentWith([candidate(1)]), 0)).rejects.toMatchObject({
      name: "ReadingCandidatesStoreError",
      code: "atomic-write-unsupported",
    });
  });

  it("preserves backups with the previous committed generation on each save", async () => {
    const { files, storage } = makeStorage();
    const instance = store(storage);
    await instance.replace(documentWith([candidate(1)]), 0);
    const firstPrimary = files[documentPath]!;
    await instance.replace(
      upsertReadingCandidate(documentWith([candidate(1)]), candidate(2), "2026-08-13T11:00:00.000Z").document,
      1,
    );
    expect(files[backupPath]).toBe(firstPrimary);
    expect(files[documentPath]).not.toBe(firstPrimary);
  });

  it("throws a typed error when a replacement save fails mid-write", async () => {
    const { storage, setAtomicImplementation } = makeStorage();
    const instance = store(storage);
    setAtomicImplementation(async () => {
      throw new Error("disk full");
    });
    await expect(instance.replace(documentWith([candidate(1)]), 0)).rejects.toMatchObject({
      name: "ReadingCandidatesStoreError",
      code: "save-failed",
    });
  });

  it("returns the committed document on clock rollback without shrinking timestamps", async () => {
    const { storage } = makeStorage();
    let now = secondTime;
    const instance = new ReadingCandidatesStore(
      storage, DEFAULT_SETTINGS.output, scope, identification, { now: () => now },
    );
    await instance.replace(documentWith([candidate(1)]), 0);
    now = firstTime;
    const saved = await instance.replace(documentWith([candidate(2)]), 1);
    expect(saved.updatedAt).toBe(secondTime.toISOString());
  });

  it("uses ReadingCandidatesStoreError for unreadable primary reads", async () => {
    const { files, storage } = makeStorage();
    files[documentPath] = "{}";
    const instance = store(storage);
    await expect(instance.load()).rejects.toBeInstanceOf(ReadingCandidatesStoreError);
  });
});
