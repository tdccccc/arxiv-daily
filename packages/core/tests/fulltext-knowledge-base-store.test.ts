import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter, StorageEntry } from "../src/core/adapters";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  createFullTextKnowledgeBasePaperPath,
  type FullTextKnowledgeBaseManifest,
  type FullTextPaperDocument,
  type FullTextPaperKnowledgeRecord,
} from "../src/library/fulltext/knowledge-base";
import {
  FullTextKnowledgeBaseFileStore,
  FullTextKnowledgeBaseStoreError,
} from "../src/library/fulltext/knowledge-base-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const otherScope = `sha256:${"c".repeat(64)}`;
const textHash = `sha256:${"e".repeat(64)}`;
const firstTime = new Date("2026-08-05T09:00:00.000Z");
const secondTime = new Date("2026-08-05T10:00:00.000Z");
const directory = `arxiv-daily/.index/personal-library-knowledge-base/${"a".repeat(64)}/${"b".repeat(64)}`;
const manifestPath = `${directory}/manifest.json`;
const manifestBackupPath = `${manifestPath}.backup`;
const papersDirectory = `${directory}/papers`;

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
  const list = vi.fn(async (dir: string): Promise<StorageEntry[]> => {
    const prefix = dir.endsWith("/") ? dir : `${dir}/`;
    const entries = new Map<string, StorageEntry>();
    for (const path of dirs) {
      if (path.startsWith(prefix)) {
        const segment = path.slice(prefix.length).split("/")[0];
        if (segment) entries.set(`${dir}/${segment}`, { path: `${dir}/${segment}`, type: "folder" });
      }
    }
    for (const path of Object.keys(files)) {
      if (path.startsWith(prefix)) {
        const rest = path.slice(prefix.length);
        if (rest.includes("/")) {
          const segment = rest.split("/")[0]!;
          entries.set(`${dir}/${segment}`, { path: `${dir}/${segment}`, type: "folder" });
        } else {
          entries.set(path, { path, type: "file" });
        }
      }
    }
    return [...entries.values()];
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
    list,
  };
  return {
    files, storage, writeTextAtomic,
    setAtomicImplementation(value: typeof atomicImplementation) { atomicImplementation = value; },
  };
}

function store(
  storage: StorageAdapter,
  now = () => secondTime,
  onWarning?: (message: string, error?: unknown) => void,
) {
  return new FullTextKnowledgeBaseFileStore(
    storage, DEFAULT_SETTINGS.output, scope, identification, { now, onWarning },
  );
}

function readyRecord(paperKey: string): FullTextPaperKnowledgeRecord {
  return {
    paperKey,
    status: "ready",
    modelId: "multilingual-e5-small-q8",
    dimension: 384,
    textHash,
    filePaths: ["library/paper.pdf"],
    observationFingerprints: [scope],
    chunkCount: 2,
    updatedAt: firstTime.toISOString(),
  };
}

function manifest(overrides: Partial<FullTextKnowledgeBaseManifest> = {}): FullTextKnowledgeBaseManifest {
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    revision: 99,
    scopeFingerprint: scope,
    identificationFingerprint: identification,
    modelId: "multilingual-e5-small-q8",
    dimension: 384,
    updatedAt: firstTime.toISOString(),
    papers: {},
    ...overrides,
  };
}

function paperDocument(overrides: Partial<FullTextPaperDocument> = {}): FullTextPaperDocument {
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey: "arxiv:2403.19236",
    modelId: "multilingual-e5-small-q8",
    dimension: 4,
    textHash,
    filePaths: ["library/paper.pdf"],
    observationFingerprints: [scope],
    chunks: [
      { index: 0, page: 1, text: "First chunk" },
      { index: 1, page: 1, text: "Second chunk" },
    ],
    vectors: new Float32Array([0.1, -0.2, 0.3, 0.4, 1.0, 2.0, -3.0, 4.5]),
    updatedAt: firstTime.toISOString(),
    ...overrides,
  };
}

function parse<T>(raw: string | undefined): T {
  if (!raw) throw new Error("missing document");
  return JSON.parse(raw) as T;
}

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

describe("construction and identity binding", () => {
  it("validates bound fingerprints before path normalization or I/O", () => {
    const memory = makeStorage();
    expect(() => new FullTextKnowledgeBaseFileStore(
      memory.storage, DEFAULT_SETTINGS.output, "bad", identification,
    )).toThrow(expect.objectContaining({ code: "invalid" }));
    expect(() => new FullTextKnowledgeBaseFileStore(
      memory.storage, DEFAULT_SETTINGS.output, scope, "sha256:not-hex",
    )).toThrow(expect.objectContaining({ code: "invalid", name: "FullTextKnowledgeBaseStoreError" }));
    expect(memory.storage.normalizePath).not.toHaveBeenCalled();
  });

  it("exposes the sharded paths bound to the constructed identity", () => {
    const memory = makeStorage();
    expect(store(memory.storage).paths).toEqual({
      directory,
      manifest: { directory, documentPath: manifestPath, backupPath: manifestBackupPath },
      papersDirectory,
    });
  });

  it("rejects a manifest bound to a different identity as invalid", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    await expect(kb.replaceManifest({ ...manifest(), scopeFingerprint: otherScope }, 0))
      .rejects.toMatchObject({ code: "invalid" });
    await expect(kb.replaceManifest({ ...manifest(), identificationFingerprint: otherScope }, 0))
      .rejects.toMatchObject({ code: "invalid" });
    expect(memory.files).toEqual({});
  });
});

describe("manifest lifecycle", () => {
  it("returns an unpersisted empty manifest and first replace writes revision one", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    const empty = await kb.loadManifest();
    expect(empty).toEqual({
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      revision: 0,
      scopeFingerprint: scope,
      identificationFingerprint: identification,
      modelId: "",
      dimension: 0,
      updatedAt: secondTime.toISOString(),
      papers: {},
    });
    expect(memory.files).toEqual({});
    const saved = await kb.replaceManifest(
      { ...empty, modelId: "multilingual-e5-small-q8", dimension: 384 }, 0,
    );
    expect(saved).toMatchObject({
      revision: 1,
      modelId: "multilingual-e5-small-q8",
      dimension: 384,
      papers: {},
    });
    expect(memory.writeTextAtomic.mock.calls.map(([path]) => path)).toEqual([
      manifestBackupPath, manifestPath,
    ]);
    await expect(kb.loadManifest()).resolves.toEqual(saved);
  });

  it("rejects an empty modelId or dimension in the first replace", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    await expect(kb.replaceManifest(manifest({ modelId: "", dimension: 0 }), 0))
      .rejects.toMatchObject({ code: "invalid" });
    await expect(kb.replaceManifest(manifest({ modelId: "model-x" }), -1))
      .rejects.toMatchObject({ code: "invalid" });
    expect(memory.files).toEqual({});
  });

  it("rejects stale replacements with currentRevision and succeeds after correction", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    await kb.replaceManifest(manifest(), 0);
    const next = manifest({ papers: { "arxiv:2403.19236": readyRecord("arxiv:2403.19236") } });
    const caught = await kb.replaceManifest(next, 0).catch((caught) => caught);
    expect(caught).toBeInstanceOf(FullTextKnowledgeBaseStoreError);
    expect(caught).toMatchObject({
      code: "stale", expectedRevision: 0, currentRevision: 1, name: "FullTextKnowledgeBaseStoreError",
    });
    const saved = await kb.replaceManifest(next, 1);
    expect(saved).toMatchObject({ revision: 2 });
    await expect(kb.loadManifest()).resolves.toEqual(saved);
  });

  it("accepts stale equal replay idempotently without incrementing or writing", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    const first = await kb.replaceManifest(manifest(), 0);
    memory.writeTextAtomic.mockClear();
    const replayed = await kb.replaceManifest(
      { ...first, revision: 0, updatedAt: "2000-01-01T00:00:00.000Z" }, 0,
    );
    expect(replayed).toEqual(first);
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
    await expect(kb.loadManifest()).resolves.toEqual(first);
  });

  it("makes a committed-then-thrown first replace retry idempotent", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path === manifestPath) throw new Error("response lost");
    });
    await expect(kb.replaceManifest(manifest(), 0)).rejects.toMatchObject({ code: "save-failed" });
    memory.setAtomicImplementation(null);
    memory.writeTextAtomic.mockClear();
    await expect(kb.replaceManifest(manifest(), 0)).resolves.toMatchObject({ revision: 1 });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("makes a committed-then-thrown second replace retry idempotent", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    const first = await kb.replaceManifest(manifest(), 0);
    const requested = manifest({ papers: { "arxiv:2403.19236": readyRecord("arxiv:2403.19236") } });
    memory.setAtomicImplementation(async (path, content) => {
      memory.files[path] = content;
      if (path === manifestPath) throw new Error("response lost");
    });
    await expect(kb.replaceManifest(requested, first.revision)).rejects.toMatchObject({ code: "save-failed" });
    memory.setAtomicImplementation(null);
    memory.writeTextAtomic.mockClear();
    await expect(kb.replaceManifest(requested, first.revision)).resolves.toMatchObject({ revision: 2 });
    expect(memory.writeTextAtomic).not.toHaveBeenCalled();
  });

  it("rotates the prior primary into the backup and never resurrects it as primary", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    const first = await kb.replaceManifest(manifest(), 0);
    const second = await kb.replaceManifest(
      manifest({ papers: { "arxiv:2403.19236": readyRecord("arxiv:2403.19236") } }), 1,
    );
    expect(parse<FullTextKnowledgeBaseManifest>(memory.files[manifestBackupPath])).toEqual(first);
    expect(parse<FullTextKnowledgeBaseManifest>(memory.files[manifestPath])).toEqual(second);
  });

  it("keeps updatedAt monotonic across a backward clock", async () => {
    const memory = makeStorage();
    let now = secondTime;
    const kb = store(memory.storage, () => now);
    const first = await kb.replaceManifest(manifest(), 0);
    now = new Date("2020-01-01T00:00:00.000Z");
    const second = await kb.replaceManifest(
      manifest({ papers: { "arxiv:2403.19236": readyRecord("arxiv:2403.19236") } }), 1,
    );
    expect(second.updatedAt).toBe(first.updatedAt);
  });

  it("rejects a model switch while papers exist and allows it after removeAll", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    await kb.replaceManifest(
      manifest({ papers: { "arxiv:2403.19236": readyRecord("arxiv:2403.19236") } }), 0,
    );
    const switched = manifest({
      modelId: "other-model-q8",
      papers: { "arxiv:2403.19236": { ...readyRecord("arxiv:2403.19236"), modelId: "other-model-q8" } },
    });
    await expect(kb.replaceManifest(switched, 1)).rejects.toMatchObject({ code: "invalid" });
    await kb.removeAll();
    const rebuilt = await kb.replaceManifest({ ...manifest(), modelId: "other-model-q8" }, 0);
    expect(rebuilt).toMatchObject({ revision: 1, modelId: "other-model-q8" });
  });

  it("rejects a persisted manifest with a different identity as incompatible", async () => {
    const memory = makeStorage();
    memory.files[manifestPath] = `${JSON.stringify({ ...manifest(), scopeFingerprint: otherScope }, null, 2)}\n`;
    await expect(store(memory.storage).loadManifest())
      .rejects.toMatchObject({ code: "incompatible" });
  });
});

describe("manifest durability and recovery", () => {
  it("repairs a corrupt primary from a valid backup and warns", async () => {
    const memory = makeStorage();
    const saved = { ...manifest({ revision: 3 }),
      papers: { "arxiv:2403.19236": readyRecord("arxiv:2403.19236") } };
    memory.files[manifestPath] = "corrupt";
    memory.files[manifestBackupPath] = `${JSON.stringify(saved, null, 2)}\n`;
    const onWarning = vi.fn();
    const loaded = await store(memory.storage, () => secondTime, onWarning).loadManifest();
    expect(loaded).toEqual(saved);
    expect(parse<FullTextKnowledgeBaseManifest>(memory.files[manifestPath])).toEqual(saved);
    expect(onWarning).toHaveBeenCalledTimes(1);
    expect(onWarning.mock.calls[0]![0]).toContain("recovered from backup");
  });

  it("treats corrupt primary and backup as corrupt-or-unreadable", async () => {
    const memory = makeStorage();
    memory.files[manifestPath] = "bad";
    memory.files[manifestBackupPath] = "also bad";
    await expect(store(memory.storage).loadManifest())
      .rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });

  it("rejects a valid incompatible backup and fails repairs with repair-failed", async () => {
    const incompatible = makeStorage();
    incompatible.files[manifestPath] = "corrupt";
    incompatible.files[manifestBackupPath] = `${JSON.stringify({ ...manifest(), scopeFingerprint: otherScope }, null, 2)}\n`;
    await expect(store(incompatible.storage).loadManifest())
      .rejects.toMatchObject({ code: "incompatible" });

    const repair = makeStorage();
    repair.files[manifestBackupPath] = `${JSON.stringify(manifest({ revision: 1 }), null, 2)}\n`;
    repair.setAtomicImplementation(async () => { throw new Error("repair failed"); });
    await expect(store(repair.storage).loadManifest())
      .rejects.toMatchObject({ code: "repair-failed" });
  });

  it("fails closed without atomic write support", async () => {
    const memory = makeStorage(false);
    const kb = store(memory.storage);
    await expect(kb.replaceManifest(manifest(), 0))
      .rejects.toMatchObject({ code: "atomic-write-unsupported" });
    await expect(kb.savePaper(paperDocument()))
      .rejects.toMatchObject({ code: "atomic-write-unsupported" });
  });
});

describe("per-paper documents", () => {
  it("round-trips paper documents including float vectors", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    const document = paperDocument();
    await kb.savePaper(document);
    const path = createFullTextKnowledgeBasePaperPath(memory.storage, kb.paths, document.paperKey);
    expect(path).toContain(`${papersDirectory}/`);
    expect(memory.files[path]).toBeDefined();
    const loaded = await kb.loadPaper(document.paperKey);
    expect(loaded).not.toBeNull();
    expect(loaded!.paperKey).toBe(document.paperKey);
    expect(loaded!.chunks).toEqual(document.chunks);
    expect(loaded!.vectors).toBeInstanceOf(Float32Array);
    expect(Array.from(loaded!.vectors)).toEqual(Array.from(document.vectors));
    expect(loaded!.updatedAt).toBe(document.updatedAt);
  });

  it("writes idempotently without touching the manifest", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    await kb.savePaper(paperDocument());
    memory.writeTextAtomic.mockClear();
    await kb.savePaper(paperDocument({ updatedAt: secondTime.toISOString() }));
    expect(memory.writeTextAtomic).toHaveBeenCalledTimes(1);
    expect(memory.files[manifestPath]).toBeUndefined();
    expect(memory.files[manifestBackupPath]).toBeUndefined();
  });

  it("rejects invalid paper documents without writing", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    const wrongLength = paperDocument();
    wrongLength.vectors = new Float32Array(3);
    await expect(kb.savePaper(wrongLength)).rejects.toMatchObject({ code: "invalid" });
    const reordered = paperDocument();
    reordered.chunks = [{ index: 1, page: 1, text: "Second first" },
      { index: 0, page: 1, text: "First second" }];
    await expect(kb.savePaper(reordered)).rejects.toMatchObject({ code: "invalid" });
    expect(memory.files).toEqual({});
  });

  it("throws a rebuildable corrupt error for an unreadable paper document", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    memory.files[createFullTextKnowledgeBasePaperPath(memory.storage, kb.paths, "arxiv:2403.19236")] = "corrupt";
    const caught = await kb.loadPaper("arxiv:2403.19236").catch((caught) => caught);
    expect(caught).toBeInstanceOf(FullTextKnowledgeBaseStoreError);
    expect(caught).toMatchObject({ code: "corrupt-or-unreadable" });
    expect(String(caught.message)).toContain("rebuild");
  });

  it("returns null for missing papers and removes idempotently", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    await expect(kb.loadPaper("arxiv:9999.99999")).resolves.toBeNull();
    const document = paperDocument();
    await kb.savePaper(document);
    await kb.removePaper(document.paperKey);
    await expect(kb.loadPaper(document.paperKey)).resolves.toBeNull();
    await expect(kb.removePaper(document.paperKey)).resolves.toBeUndefined();
    await expect(kb.removePaper("arxiv:9999.99999")).resolves.toBeUndefined();
  });
});

describe("removeAll", () => {
  it("clears papers and manifest so the store returns to its empty state", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    await kb.replaceManifest(
      manifest({ papers: { "arxiv:2403.19236": readyRecord("arxiv:2403.19236") } }), 0,
    );
    await kb.savePaper(paperDocument());
    await kb.savePaper(paperDocument({ paperKey: "arxiv:2309.11425" }));
    expect(memory.files[manifestPath]).toBeDefined();
    expect(memory.files[manifestBackupPath]).toBeDefined();
    await kb.removeAll();
    expect(memory.files).toEqual({});
    const empty = await kb.loadManifest();
    expect(empty).toMatchObject({ revision: 0, papers: {}, modelId: "", dimension: 0 });
    await expect(kb.loadPaper("arxiv:2403.19236")).resolves.toBeNull();
  });

  it("clears paper files through the manifest fallback when list is unsupported", async () => {
    const memory = makeStorage();
    delete memory.storage.list;
    const kb = store(memory.storage);
    await kb.replaceManifest(
      manifest({ papers: { "arxiv:2403.19236": readyRecord("arxiv:2403.19236") } }), 0,
    );
    await kb.savePaper(paperDocument());
    await kb.removeAll();
    expect(memory.files).toEqual({});
    await expect(kb.loadPaper("arxiv:2403.19236")).resolves.toBeNull();
  });

  it("tolerates directory removal failure while still removing every file", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    await kb.replaceManifest(manifest(), 0);
    await kb.savePaper(paperDocument());
    memory.storage.remove = vi.fn(async (path) => {
      if (path === papersDirectory || path === directory) return; // host refuses dir removal
      delete memory.files[path];
    });
    await kb.removeAll();
    expect(memory.files).toEqual({});
  });
});

describe("concurrency", () => {
  it("serializes concurrent replacements on the same manifest path without losing updates", async () => {
    const memory = makeStorage();
    const gate = deferred();
    let block = true;
    memory.setAtomicImplementation(async (path, content) => {
      if (path === manifestPath && block) { block = false; await gate.promise; }
      memory.files[path] = content;
    });
    const first = store(memory.storage);
    const second = store(memory.storage);
    const updateA = manifest({ papers: { "arxiv:2403.19236": readyRecord("arxiv:2403.19236") } });
    const updateB = manifest({ papers: { "arxiv:2309.11425": readyRecord("arxiv:2309.11425") } });
    const saving = first.replaceManifest(updateA, 0);
    const concurrent = second.replaceManifest(updateB, 0);
    await Promise.resolve();
    gate.resolve();
    const results = await Promise.allSettled([saving, concurrent]);
    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    expect(results.filter((result) => result.status === "rejected")).toHaveLength(1);
    const winner = results.find((result) => result.status === "fulfilled")!;
    expect(winner.value).toMatchObject({ revision: 1 });
    const loser = results.find((result) => result.status === "rejected")!;
    expect(loser.reason).toMatchObject({ code: "stale", expectedRevision: 0, currentRevision: 1 });
    const loaded = await first.loadManifest();
    expect(loaded).toEqual(winner.value);
    expect(Object.keys(loaded.papers)).toEqual(Object.keys(winner.value.papers));
  });

  it("keeps queued operations isolated per path", async () => {
    const memory = makeStorage();
    const kb = store(memory.storage);
    await kb.savePaper(paperDocument());
    await kb.savePaper(paperDocument({ paperKey: "arxiv:2309.11425" }));
    expect(Object.keys(memory.files).filter((path) => path.startsWith(papersDirectory))).toHaveLength(2);
    const a = await kb.loadPaper("arxiv:2403.19236");
    const b = await kb.loadPaper("arxiv:2309.11425");
    expect(a!.paperKey).toBe("arxiv:2403.19236");
    expect(b!.paperKey).toBe("arxiv:2309.11425");
  });
});
