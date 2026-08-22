import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  DEFAULT_FULL_TEXT_GENERATION_INDEX_DERIVATION,
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  FullTextGenerationIndexStore,
  FullTextGenerationIndexStoreError,
  createEvidenceChunkId,
  preflightFullTextGenerationSynchronization,
  searchFullTextKnowledgeBase,
  synchronizeFullTextGenerationIndex,
  type EmbeddingModel,
  type FullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBaseStore,
  type FullTextPaperDocument,
} from "../src/index";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const TEXT_HASH = `sha256:${"c".repeat(64)}`;
const OBSERVATION = `sha256:${"d".repeat(64)}`;
const DERIVATION = {
  parser: { id: "fixture-parser", version: "1" },
  chunkerVersion: 2,
  embeddingInputVersion: 1,
} as const;

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((settle) => { resolve = settle; });
  return { promise, resolve };
}

function paper(paperKey: string, text: string, vector: readonly number[]): FullTextPaperDocument {
  const identity = {
    text,
    headings: ["Methods"],
    locator: { pageStart: 2 },
    derivation: DERIVATION,
  };
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey,
    modelId: "model-a",
    dimension: 2,
    textHash: TEXT_HASH,
    title: `Title ${paperKey}`,
    filePaths: [`${paperKey.replaceAll(":", "-")}.pdf`],
    observationFingerprints: [OBSERVATION],
    derivation: DERIVATION,
    chunks: [{
      id: createEvidenceChunkId(identity),
      index: 0,
      page: 2,
      ...identity,
    }],
    vectors: new Float32Array(vector),
    updatedAt: "2026-08-18T00:00:00.000Z",
  };
}

function manifest(revision: number, documents: readonly FullTextPaperDocument[]): FullTextKnowledgeBaseManifest {
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    revision,
    scopeFingerprint: SCOPE,
    identificationFingerprint: IDENTIFICATION,
    modelId: "model-a",
    dimension: 2,
    updatedAt: "2026-08-18T00:00:00.000Z",
    papers: Object.fromEntries(documents.map((document) => [document.paperKey, {
      paperKey: document.paperKey,
      status: "ready" as const,
      modelId: document.modelId,
      dimension: document.dimension,
      textHash: document.textHash,
      title: document.title,
      filePaths: [...document.filePaths],
      observationFingerprints: [...document.observationFingerprints],
      derivation: document.derivation,
      chunkCount: document.chunks.length,
      updatedAt: document.updatedAt,
    }])),
  };
}

function mutableSource(documents: readonly FullTextPaperDocument[], revision = 7) {
  const byKey = new Map(documents.map((document) => [document.paperKey, document]));
  const state = { manifest: manifest(revision, documents) };
  const loadManifest = vi.fn(async () => state.manifest);
  const loadPaper = vi.fn(async (paperKey: string) => byKey.get(paperKey) ?? null);
  const store: FullTextKnowledgeBaseStore = {
    paths: {
      directory: "legacy",
      manifest: { directory: "legacy", documentPath: "legacy/manifest.json", backupPath: "legacy/manifest.json.backup" },
      papersDirectory: "legacy/papers",
    },
    loadManifest,
    loadPaper,
    replaceManifest: vi.fn(async (next) => next),
    savePaper: vi.fn(async (document) => { byKey.set(document.paperKey, document); }),
    removePaper: vi.fn(async (paperKey) => { byKey.delete(paperKey); }),
    removeAll: vi.fn(async () => { byKey.clear(); }),
  };
  return {
    state,
    store,
    loadManifest,
    loadPaper,
    replace(revision: number, next: readonly FullTextPaperDocument[]) {
      byKey.clear();
      next.forEach((document) => byKey.set(document.paperKey, document));
      state.manifest = manifest(revision, next);
    },
  };
}

function memoryStorage() {
  const text = new Map<string, string>();
  const binary = new Map<string, Uint8Array>();
  const dirs = new Set<string>();
  const normalizePath = (path: string) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
  const storage: StorageAdapter = {
    normalizePath,
    readText: async (path) => {
      const value = text.get(normalizePath(path));
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    },
    writeText: async (path, value) => { text.set(normalizePath(path), value); },
    writeTextAtomic: async (path, value) => { text.set(normalizePath(path), value); },
    createTextExclusive: async (path, value) => {
      const normalized = normalizePath(path);
      if (text.has(normalized) || binary.has(normalized) || dirs.has(normalized)) return false;
      text.set(normalized, value);
      return true;
    },
    exists: async (path) => {
      const normalized = normalizePath(path);
      return text.has(normalized) || binary.has(normalized) || dirs.has(normalized);
    },
    mkdir: async (path) => { dirs.add(normalizePath(path)); },
    remove: async (path) => {
      const normalized = normalizePath(path);
      const prefix = `${normalized}/`;
      for (const key of [...text.keys()]) if (key === normalized || key.startsWith(prefix)) text.delete(key);
      for (const key of [...binary.keys()]) if (key === normalized || key.startsWith(prefix)) binary.delete(key);
      for (const key of [...dirs]) if (key === normalized || key.startsWith(prefix)) dirs.delete(key);
    },
    rename: async (from, to) => {
      const source = normalizePath(from);
      const target = normalizePath(to);
      const value = text.get(source);
      if (value === undefined) throw new Error(`missing ${from}`);
      text.delete(source);
      text.set(target, value);
    },
    writeBinary: vi.fn(async (path, value) => { binary.set(normalizePath(path), new Uint8Array(value).slice()); }),
    readBinary: vi.fn(async (path) => {
      const value = binary.get(normalizePath(path));
      if (!value) throw new Error(`missing ${path}`);
      return value.slice().buffer;
    }),
    list: async (directory) => {
      const normalized = normalizePath(directory);
      const prefix = `${normalized}/`;
      const entries = new Map<string, "file" | "folder">();
      for (const path of [...text.keys(), ...binary.keys(), ...dirs]) {
        if (!path.startsWith(prefix)) continue;
        const suffix = path.slice(prefix.length);
        if (!suffix) continue;
        const [head, ...rest] = suffix.split("/");
        entries.set(`${prefix}${head}`, rest.length > 0 || dirs.has(`${prefix}${head}`) ? "folder" : "file");
      }
      return [...entries].map(([path, type]) => ({ path, type }));
    },
  };
  return { storage, text, binary, dirs };
}

function embedding(queryVector: readonly number[] = [1, 0]): EmbeddingModel {
  return {
    modelId: "model-a",
    dimension: 2,
    prefixPolicy: "none",
    embed: vi.fn(async (texts) => texts.map(() => new Float32Array(queryVector))),
  };
}

function writer(suffix: string): string {
  return `writer-${suffix}-${"f".repeat(40)}`.slice(0, 64);
}

function syncInput(
  sourceStore: FullTextKnowledgeBaseStore,
  generationStore: FullTextGenerationIndexStore,
  storage: StorageAdapter,
  writerToken: string,
) {
  return {
    sourceStore,
    generationStore,
    storage,
    output: DEFAULT_SETTINGS.output,
    scopeFingerprint: SCOPE,
    identificationFingerprint: IDENTIFICATION,
    writerToken,
  };
}

describe("full-text generation production orchestration", () => {
  it("allows unsupported hosts only in the never-cut-over migration window", async () => {
    const source = mutableSource([paper("paper:alpha", "alpha", [1, 0])]);
    const memory = memoryStorage();
    const limitedStorage = { ...memory.storage };
    delete limitedStorage.createTextExclusive;
    const limitedGenerations = new FullTextGenerationIndexStore(
      limitedStorage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    await expect(preflightFullTextGenerationSynchronization({
      storage: limitedStorage,
      generationStore: limitedGenerations,
    })).resolves.toBe("migration-fallback");

    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("preflight-cutover")),
    );
    await expect(preflightFullTextGenerationSynchronization({
      storage: limitedStorage,
      generationStore: limitedGenerations,
    })).rejects.toMatchObject({ code: "capability-unsupported" });
  });

  it("treats missing list as fallback only before cutover", async () => {
    const source = mutableSource([paper("paper:alpha", "alpha", [1, 0])]);
    const memory = memoryStorage();
    const noList = { ...memory.storage };
    delete noList.list;
    const limitedGenerations = new FullTextGenerationIndexStore(
      noList, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    await expect(preflightFullTextGenerationSynchronization({
      storage: noList,
      generationStore: limitedGenerations,
    })).resolves.toBe("migration-fallback");

    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("list-cutover")),
    );
    await expect(preflightFullTextGenerationSynchronization({
      storage: noList,
      generationStore: limitedGenerations,
    })).rejects.toMatchObject({ code: "capability-unsupported" });
  });

  it("builds the committed revision, reuses it exactly, and searches without legacy paper loads", async () => {
    const alpha = paper("paper:alpha", "alpha telescope survey", [1, 0]);
    const beta = paper("paper:beta", "unrelated chemistry", [0, 1]);
    const source = mutableSource([alpha, beta]);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );

    const first = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("first")),
    );
    expect(first).toMatchObject({ kind: "rebuilt", sourceRevision: 7, indexedPaperCount: 2 });
    expect(source.loadPaper).toHaveBeenCalledTimes(2);
    expect([...memory.binary.keys()].some((path) => path.includes("/spool/"))).toBe(false);

    source.loadPaper.mockClear();
    const writes = vi.mocked(memory.storage.writeBinary!).mock.calls.length;
    const second = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("reuse")),
    );
    expect(second).toMatchObject({ kind: "reused", generationId: first.generationId, sourceRevision: 7 });
    expect(source.loadPaper).not.toHaveBeenCalled();
    expect(vi.mocked(memory.storage.writeBinary!).mock.calls).toHaveLength(writes);

    const model = embedding();
    const sourceManifest = await source.store.loadManifest();
    source.loadManifest.mockClear();
    const matches = await searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      sourceManifest,
      embedding: model,
      queryText: "alpha telescope",
      limit: 2,
    });
    expect(matches[0]).toMatchObject({ paperKey: "paper:alpha", rankingScoreKind: "rrf" });
    expect(source.loadPaper).not.toHaveBeenCalled();
    expect(source.loadManifest).not.toHaveBeenCalled();
    expect(model.embed).toHaveBeenCalledTimes(1);

    const titleMatches = await searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      sourceManifest,
      embedding: model,
      queryText: "catalog override",
      mode: "lexical",
      titles: new Map([["paper:beta", "Catalog Override"]]),
    });
    expect(titleMatches[0]?.paperKey).toBe("paper:beta");
    expect(model.embed).toHaveBeenCalledTimes(1);
  });

  it("rebuilds for a derivation change even at the same source revision", async () => {
    const source = mutableSource([paper("paper:alpha", "alpha", [1, 0])]);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const first = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("initial")),
    );
    source.loadPaper.mockClear();

    const second = await synchronizeFullTextGenerationIndex({
      ...syncInput(source.store, generations, memory.storage, writer("derive")),
      indexDerivation: {
        ...DEFAULT_FULL_TEXT_GENERATION_INDEX_DERIVATION,
        builderVersion: DEFAULT_FULL_TEXT_GENERATION_INDEX_DERIVATION.builderVersion + 1,
      },
    });

    expect(second).toMatchObject({ kind: "rebuilt", sourceRevision: 7 });
    expect(second.generationId).not.toBe(first.generationId);
    expect(source.loadPaper).toHaveBeenCalledTimes(1);
  });

  it("rebuilds a changed committed revision and preserves the prior current on build failure", async () => {
    const alpha = paper("paper:alpha", "alpha", [1, 0]);
    const source = mutableSource([alpha], 7);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const first = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("revision-first")),
    );

    const beta = paper("paper:beta", "beta", [0, 1]);
    source.replace(8, [beta]);
    const second = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("revision-second")),
    );
    expect(second).toMatchObject({ kind: "rebuilt", sourceRevision: 8, indexedPaperCount: 1 });
    expect(second.generationId).not.toBe(first.generationId);

    source.replace(9, [alpha]);
    source.loadPaper.mockRejectedValueOnce(new Error("source read failed"));
    await expect(synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("revision-failure")),
    )).rejects.toBeTruthy();
    await expect(generations.openCurrent()).resolves.toMatchObject({
      descriptor: { generationId: second.generationId, sourceRevision: 8 },
    });
  });

  it("replaces a corrupt current generation with a complete newer candidate", async () => {
    const alpha = paper("paper:alpha", "alpha", [1, 0]);
    const source = mutableSource([alpha], 7);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const first = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("repair-first")),
    );
    source.replace(8, [paper("paper:beta", "beta", [0, 1])]);
    const second = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("repair-second")),
    );
    const corruptPath = [...memory.binary.keys()].find((path) => path.includes(
      `/generations/${second.generationId}/objects/`,
    ));
    expect(corruptPath).toBeDefined();
    memory.binary.get(corruptPath!)![60]! ^= 1;
    source.replace(9, [paper("paper:gamma", "gamma", [0.5, 0.5])]);

    const repaired = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("repair-third")),
    );

    expect(repaired).toMatchObject({ kind: "rebuilt", sourceRevision: 9 });
    await expect(generations.openCurrent()).resolves.toMatchObject({
      descriptor: { generationId: repaired.generationId, sourceRevision: 9 },
    });
    const backup = memory.text.get(generations.paths.backupPath);
    expect(backup).toContain(first.generationId);
  });

  it("retries synchronization after a definite first CURRENT write failure", async () => {
    const source = mutableSource([paper("paper:alpha", "alpha", [1, 0])]);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const writeTextAtomic = memory.storage.writeTextAtomic!;
    let failCurrent = true;
    memory.storage.writeTextAtomic = vi.fn(async (path, value) => {
      if (path === generations.paths.currentPath && failCurrent) {
        failCurrent = false;
        throw new Error("first CURRENT write failed");
      }
      await writeTextAtomic(path, value);
    });

    await expect(synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("first-failure")),
    )).rejects.toMatchObject({ code: "write-failed" });
    source.loadPaper.mockClear();
    await expect(searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      embedding: embedding(),
      queryText: "alpha",
    })).resolves.toMatchObject([{ paperKey: "paper:alpha" }]);
    expect(source.loadPaper).toHaveBeenCalledTimes(1);

    const retried = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("retry")),
    );
    expect(retried).toMatchObject({ kind: "rebuilt", sourceRevision: 7 });
    await expect(generations.openCurrent()).resolves.toMatchObject({
      descriptor: { generationId: retried.generationId, sourceRevision: 7 },
    });
  });

  it("uses a fresh generation namespace when failed cleanup leaves an earlier attempt", async () => {
    const source = mutableSource([paper("paper:alpha", "alpha", [1, 0])]);
    const memory = memoryStorage();
    const failing = new FullTextGenerationIndexStore(
      memory.storage,
      DEFAULT_SETTINGS.output,
      SCOPE,
      IDENTIFICATION,
      { beforePointerPromotion: () => { throw new Error("promotion rejected"); } },
    );
    const generationPrefix = `${failing.paths.generationsDirectory}/`;
    const remove = memory.storage.remove.bind(memory.storage);
    let refuseGenerationCleanup = true;
    memory.storage.remove = vi.fn(async (path) => {
      const normalized = memory.storage.normalizePath(path);
      if (refuseGenerationCleanup && normalized.startsWith(generationPrefix)) {
        throw new Error("generation cleanup unavailable");
      }
      await remove(path);
    });

    await expect(synchronizeFullTextGenerationIndex(
      syncInput(source.store, failing, memory.storage, writer("orphan-first")),
    )).rejects.toMatchObject({ code: "write-failed" });
    const staleClaim = [...memory.text.keys()].find((path) => path.endsWith("/.staging-claim.json"));
    expect(staleClaim).toBeDefined();
    const staleGenerationId = staleClaim!.split("/").at(-2);

    refuseGenerationCleanup = false;
    const retrying = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const retried = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, retrying, memory.storage, writer("orphan-second")),
    );

    expect(retried).toMatchObject({ kind: "rebuilt", sourceRevision: 7 });
    expect(retried.generationId).not.toBe(staleGenerationId);
    await expect(retrying.openCurrent()).resolves.toMatchObject({
      descriptor: { generationId: retried.generationId, sourceRevision: 7 },
    });
  });

  it("does not report exact reuse when the source advances during generation validation", async () => {
    const original = paper("paper:alpha", "alpha", [1, 0]);
    const source = mutableSource([original], 7);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("reuse-race-first")),
    );
    const readBinary = memory.storage.readBinary!;
    let advanced = false;
    memory.storage.readBinary = vi.fn(async (path) => {
      if (!advanced) {
        advanced = true;
        source.replace(8, [paper("paper:alpha", "alpha revised", [0.9, 0.1])]);
      }
      return readBinary(path);
    });

    await expect(synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("reuse-race-second")),
    )).rejects.toMatchObject({ code: "stale-source", expectedRevision: 7, currentRevision: 8 });
  });

  it("returns an empty result for every search mode on an empty committed generation", async () => {
    const source = mutableSource([], 1);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("empty")),
    );

    for (const mode of ["dense", "lexical", "hybrid"] as const) {
      const model = embedding();
      await expect(searchFullTextKnowledgeBase({
        store: source.store,
        generationStore: generations,
        embedding: model,
        queryText: "anything",
        mode,
      })).resolves.toEqual([]);
      expect(model.embed).not.toHaveBeenCalled();
    }
    expect(source.loadPaper).not.toHaveBeenCalled();
    const maintenance = await generations.beginHostAuthorizedMaintenance();
    await expect(maintenance.run()).resolves.toMatchObject({ removedGenerationIds: [] });
    maintenance.release();
  });

  it("keeps one pinned generation while a concurrent promotion changes CURRENT", async () => {
    const alpha = paper("paper:alpha", "alpha telescope", [1, 0]);
    const source = mutableSource([alpha], 7);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const first = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("pin-first")),
    );
    const readStarted = deferred();
    const releaseRead = deferred();
    const readBinary = memory.storage.readBinary!;
    let paused = false;
    memory.storage.readBinary = vi.fn(async (path) => {
      if (!paused && path.includes(`/generations/${first.generationId}/`)) {
        paused = true;
        readStarted.resolve();
        await releaseRead.promise;
      }
      return readBinary(path);
    });

    const pinnedSearch = searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      embedding: embedding([1, 0]),
      queryText: "alpha",
      mode: "dense",
    });
    await readStarted.promise;
    const beta = paper("paper:beta", "beta chemistry", [0, 1]);
    source.replace(8, [beta]);
    const promoted = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("pin-second")),
    );
    releaseRead.resolve();

    await expect(pinnedSearch).resolves.toMatchObject([{ paperKey: "paper:alpha" }]);
    await expect(searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      embedding: embedding([0, 1]),
      queryText: "beta",
      mode: "dense",
    })).resolves.toMatchObject([{ paperKey: "paper:beta" }]);
    expect(promoted).toMatchObject({ kind: "rebuilt", sourceRevision: 8 });
  });

  it("fails closed instead of serving a generation behind the committed source manifest", async () => {
    const alpha = paper("paper:alpha", "alpha", [1, 0]);
    const source = mutableSource([alpha], 7);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("stale-search")),
    );
    source.replace(8, [paper("paper:beta", "beta", [0, 1])]);
    source.loadPaper.mockClear();

    await expect(searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      embedding: embedding(),
      queryText: "alpha",
    })).rejects.toMatchObject({ code: "stale-source", expectedRevision: 7, currentRevision: 8 });
    expect(source.loadPaper).not.toHaveBeenCalled();
  });

  it("keeps the prior current generation when the committed source changes during promotion", async () => {
    const alpha = paper("paper:alpha", "alpha", [1, 0]);
    const source = mutableSource([alpha], 7);
    const memory = memoryStorage();
    const firstStore = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const first = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, firstStore, memory.storage, writer("stable")),
    );
    source.replace(8, [paper("paper:alpha", "alpha revised", [0.9, 0.1])]);
    const racingStore = new FullTextGenerationIndexStore(
      memory.storage,
      DEFAULT_SETTINGS.output,
      SCOPE,
      IDENTIFICATION,
      { beforePointerPromotion: () => source.replace(9, [alpha]) },
    );

    await expect(synchronizeFullTextGenerationIndex(
      syncInput(source.store, racingStore, memory.storage, writer("racing")),
    )).rejects.toMatchObject({ code: "stale-source" });
    await expect(firstStore.openCurrent()).resolves.toMatchObject({
      descriptor: { generationId: first.generationId, sourceRevision: 7 },
    });
  });

  it("rebuilds again when the source advances while CURRENT commit is in flight", async () => {
    const alpha = paper("paper:alpha", "alpha", [1, 0]);
    const beta = paper("paper:beta", "beta", [0, 1]);
    const source = mutableSource([alpha], 7);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const writeTextAtomic = memory.storage.writeTextAtomic!;
    let advanced = false;
    memory.storage.writeTextAtomic = vi.fn(async (path, value) => {
      if (!advanced && path === generations.paths.currentPath) {
        advanced = true;
        source.replace(8, [beta]);
      }
      await writeTextAtomic(path, value);
    });

    const synchronized = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("commit-race")),
    );

    expect(synchronized).toMatchObject({ kind: "rebuilt", sourceRevision: 8 });
    await expect(generations.openCurrent()).resolves.toMatchObject({
      descriptor: { generationId: synchronized.generationId, sourceRevision: 8 },
    });
  });

  it("rebuilds the latest revision after a stale CURRENT commit is acknowledged by postcheck", async () => {
    const alpha = paper("paper:alpha", "alpha", [1, 0]);
    const beta = paper("paper:beta", "beta", [0, 1]);
    const source = mutableSource([alpha], 7);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const writeTextAtomic = memory.storage.writeTextAtomic!;
    let lostResponse = true;
    memory.storage.writeTextAtomic = vi.fn(async (path, value) => {
      if (lostResponse && path === generations.paths.currentPath) {
        lostResponse = false;
        source.replace(8, [beta]);
        await writeTextAtomic(path, value);
        throw new Error("CURRENT response lost after commit");
      }
      await writeTextAtomic(path, value);
    });

    const synchronized = await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("commit-ack-race")),
    );

    expect(synchronized).toMatchObject({ kind: "rebuilt", sourceRevision: 8 });
    await expect(generations.openCurrent()).resolves.toMatchObject({
      descriptor: { generationId: synchronized.generationId, sourceRevision: 8 },
    });
  });

  it("stops with typed stale-source after three continuously advancing commits", async () => {
    const source = mutableSource([paper("paper:revision-7", "revision 7", [1, 0])], 7);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    const promotionTokens: string[] = [];
    const promote = generations.stageAndPromote.bind(generations);
    vi.spyOn(generations, "stageAndPromote").mockImplementation((input) => {
      promotionTokens.push(input.writerToken);
      return promote(input);
    });
    const writeTextAtomic = memory.storage.writeTextAtomic!;
    let currentCommits = 0;
    memory.storage.writeTextAtomic = vi.fn(async (path, value) => {
      await writeTextAtomic(path, value);
      if (path !== generations.paths.currentPath) return;
      currentCommits += 1;
      const revision = 7 + currentCommits;
      source.replace(revision, [paper(`paper:revision-${revision}`, `revision ${revision}`, [1, 0])]);
    });

    await expect(synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("continuous-source")),
    )).rejects.toMatchObject({
      code: "stale-source",
      expectedRevision: 9,
      currentRevision: 10,
    });
    expect(currentCommits).toBe(3);
    expect(promotionTokens).toHaveLength(3);
    expect(new Set(promotionTokens).size).toBe(3);
    await expect(generations.openCurrent()).resolves.toMatchObject({
      descriptor: { sourceRevision: 9 },
    });
  });

  it("uses legacy search only when CURRENT has never existed and fails closed for a corrupt pointer", async () => {
    const source = mutableSource([paper("paper:alpha", "alpha", [1, 0])]);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );

    await expect(searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      embedding: embedding(),
      queryText: "alpha",
    })).resolves.toMatchObject([{ paperKey: "paper:alpha" }]);
    expect(source.loadPaper).toHaveBeenCalledTimes(1);

    source.loadPaper.mockClear();
    memory.text.set(generations.paths.currentPath, "{not-json");
    await expect(searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      embedding: embedding(),
      queryText: "alpha",
    })).rejects.toBeInstanceOf(FullTextGenerationIndexStoreError);
    expect(source.loadPaper).not.toHaveBeenCalled();

    memory.text.set(generations.paths.currentPath, JSON.stringify({ formatVersion: 2, schemaVersion: 2 }));
    await expect(searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      embedding: embedding(),
      queryText: "alpha",
    })).rejects.toMatchObject({ code: "incompatible" });
    expect(source.loadPaper).not.toHaveBeenCalled();
  });

  it("does not return to legacy fallback after a committed cutover loses both pointers", async () => {
    const source = mutableSource([paper("paper:alpha", "alpha", [1, 0])]);
    const memory = memoryStorage();
    const generations = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION,
    );
    await synchronizeFullTextGenerationIndex(
      syncInput(source.store, generations, memory.storage, writer("cutover")),
    );
    await memory.storage.remove(generations.paths.currentPath);
    await memory.storage.remove(generations.paths.backupPath);
    source.loadPaper.mockClear();

    await expect(searchFullTextKnowledgeBase({
      store: source.store,
      generationStore: generations,
      embedding: embedding(),
      queryText: "alpha",
    })).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    expect(source.loadPaper).not.toHaveBeenCalled();
  });
});
