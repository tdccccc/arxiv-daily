import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  FullTextGenerationIndexStore,
  createEvidenceChunkId,
  fusePaperRankingsRrf,
  searchGenerationBm25,
  searchGenerationDense,
  searchKnowledgeBase,
  searchKnowledgeBaseBm25,
  decodeEvidenceBlock,
  deriveLexicalChunk,
  decodeGenerationDescriptor,
  MAX_BINARY_OBJECT_BYTES,
  decodeLexicalDictionaryBlock,
  decodeLexicalPostingsBlock,
  decodePaperMetadataBlock,
  decodeVectorBlock,
  type FullTextKnowledgeBaseManifest,
  type FullTextPaperDocument,
  type GenerationObjectReference,
  type GenerationObjectWrite,
  type GenerationIndexBuildInstrumentation,
  type GenerationIndexBuildOperation,
  type GenerationObjectSpool,
} from "../src/index";
import { buildFullTextGeneration, GenerationIndexBuildError } from "../src/library/fulltext/generation-index-builder";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const HASH = `sha256:${"c".repeat(64)}`;
const OBS = `sha256:${"d".repeat(64)}`;
const DERIVATION = { parser: { id: "fixture", version: "1" }, chunkerVersion: 2, embeddingInputVersion: 1 } as const;
const INDEX_DERIVATION = { builderVersion: 1, denseCenteringVersion: 1, tokenizerVersion: 1, postingsVersion: 1 } as const;

function paper(paperKey: string, texts: readonly string[], values?: readonly number[], title = `Title ${paperKey}`): FullTextPaperDocument {
  const chunks = texts.map((text, index) => {
    const identity = { text, headings: ["Section"], locator: { pageStart: index + 1 }, derivation: DERIVATION };
    return { id: createEvidenceChunkId(identity), index, page: index + 1, ...identity };
  });
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey, modelId: "model-a", dimension: 2, textHash: HASH, title,
    filePaths: [`${paperKey.replaceAll(":", "-")}.pdf`], observationFingerprints: [OBS],
    derivation: DERIVATION, chunks,
    vectors: new Float32Array(values ?? texts.flatMap((_, index) => [index + 1, index + 2])),
    updatedAt: "2026-08-17T00:00:00.000Z",
  };
}

function manifest(documents: readonly FullTextPaperDocument[], failedKeys: readonly string[] = []): FullTextKnowledgeBaseManifest {
  const papers: Record<string, any> = {};
  for (const document of documents) papers[document.paperKey] = {
    paperKey: document.paperKey, status: "ready", modelId: document.modelId, dimension: document.dimension,
    textHash: document.textHash, title: document.title, filePaths: [...document.filePaths],
    observationFingerprints: [...document.observationFingerprints], derivation: document.derivation,
    chunkCount: document.chunks.length, updatedAt: document.updatedAt,
  };
  for (const paperKey of failedKeys) papers[paperKey] = {
    paperKey, status: "failed", modelId: "model-a", dimension: 2, filePaths: [], observationFingerprints: [],
    chunkCount: 0, error: "failed", updatedAt: "2026-08-17T00:00:00.000Z",
  };
  return { schemaVersion: 2, revision: 7, scopeFingerprint: SCOPE, identificationFingerprint: IDENTIFICATION,
    modelId: "model-a", dimension: 2, updatedAt: "2026-08-17T00:00:00.000Z", papers };
}

class MemorySpool implements GenerationObjectSpool {
  readonly data = new Map<string, Uint8Array>();
  reads = 0;
  removes = 0;
  async put(seed: Omit<GenerationObjectReference, "byteLength" | "checksum">, bytes: Uint8Array) {
    const { blockObjectChecksum } = await import("../src/library/fulltext/generation-index-format");
    const reference = { ...seed, byteLength: bytes.byteLength, checksum: blockObjectChecksum(bytes) };
    this.data.set(seed.path, bytes.slice());
    return reference;
  }
  async read(reference: GenerationObjectReference) { this.reads += 1; const bytes = this.data.get(reference.path); if (!bytes) throw new Error("missing spool object"); return bytes.slice(); }
  async removeAll() { this.removes += 1; this.data.clear(); }
}

async function build(documents: readonly FullTextPaperDocument[], options: Record<string, unknown> = {}) {
  const source = manifest(documents, ["paper:failed"]);
  const byKey = new Map(documents.map((document) => [document.paperKey, document]));
  const spool = new MemorySpool();
  const loads: string[] = [];
  const result = await buildFullTextGeneration({
    manifest: source, generationId: "gen-builder", indexDerivation: INDEX_DERIVATION, spool,
    loadPaper: async (key) => { loads.push(key); return byKey.get(key) ?? null; }, ...options,
  });
  return { result, spool, loads };
}

async function collect(iterable: AsyncIterable<GenerationObjectWrite>) { const result = []; for await (const value of iterable) result.push(value); return result; }

function decode(write: GenerationObjectWrite, kind: GenerationObjectReference["kind"]) {
  return kind === "vector" ? decodeVectorBlock(write.bytes)
    : kind === "evidence" ? decodeEvidenceBlock(write.bytes)
      : kind === "paper-metadata" ? decodePaperMetadataBlock(write.bytes)
        : kind === "lexical-postings" ? decodeLexicalPostingsBlock(write.bytes)
          : decodeLexicalDictionaryBlock(write.bytes);
}

function memoryStorage(): StorageAdapter {
  const text = new Map<string, string>(); const binary = new Map<string, Uint8Array>(); const dirs = new Set<string>();
  return {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""),
    readText: async (path) => { const value = text.get(path); if (value === undefined) throw new Error(`missing ${path}`); return value; },
    writeText: async (path, value) => { text.set(path, value); },
    writeTextAtomic: async (path, value) => { text.set(path, value); },
    createTextExclusive: async (path, value) => { if (text.has(path) || binary.has(path) || dirs.has(path)) return false; text.set(path, value); return true; },
    exists: async (path) => text.has(path) || binary.has(path) || dirs.has(path),
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => { const prefix = `${path}/`; for (const key of [...text.keys()]) if (key === path || key.startsWith(prefix)) text.delete(key); for (const key of [...binary.keys()]) if (key === path || key.startsWith(prefix)) binary.delete(key); },
    rename: async () => undefined,
    list: async (dir) => {
      const prefix = `${dir}/`; const entries = new Map<string, "file" | "folder">();
      for (const path of [...text.keys(), ...binary.keys(), ...dirs]) {
        if (!path.startsWith(prefix)) continue;
        const suffix = path.slice(prefix.length); if (!suffix) continue;
        const child = suffix.split("/")[0]!; const childPath = `${dir}/${child}`;
        entries.set(childPath, suffix.includes("/") || dirs.has(childPath) ? "folder" : "file");
      }
      return [...entries].map(([path, type]) => ({ path, type }));
    },
    writeBinary: vi.fn(async (path, value) => { binary.set(path, new Uint8Array(value).slice()); }),
    readBinary: vi.fn(async (path) => { const value = binary.get(path); if (!value) throw new Error(`missing ${path}`); return value.slice().buffer; }),
  };
}

describe("full-text generation builder", () => {
  it("snapshots, loads ready papers once in UTF-16 order, packs paired blocks, and replays canonical objects", async () => {
    const docs = [paper("paper:😀", ["alpha beta", "中文检索"], [1, 2, 3, 4]), paper("#paper:a", ["gamma"], [5, 6])];
    const { result, spool, loads } = await build(docs, { blockTargetRows: 1 });
    expect(loads).toEqual(["#paper:a", "paper:😀"]);
    expect(result.descriptor.sourceRevision).toBe(7);
    expect(result.descriptor.objects.map((entry) => entry.kind)).toEqual([
      "vector", "vector", "vector", "evidence", "evidence", "evidence", "paper-metadata", "paper-metadata",
      "lexical-postings", "lexical-postings", "lexical-postings", "lexical-dictionary",
    ]);
    const first = await collect(result.objects());
    expect(first.map((entry) => entry.path)).toEqual(result.descriptor.objects.map((entry) => entry.path));
    await expect(collect(result.objects())).rejects.toMatchObject({ code: "invalid-source" });
    expect(spool.reads).toBeGreaterThanOrEqual(first.length + 3);
    expect(spool.removes).toBe(1);
    result.descriptor.objects.forEach((ref, index) => expect(decode(first[index]!, ref.kind)).toBeTruthy());
    expect(result.diagnostics).toMatchObject({ peakLoadedPapers: 1, sourcePaperLoads: 2, objectRefs: result.descriptor.objects.length });
    expect(() => decodeGenerationDescriptor(JSON.stringify(result.descriptor))).not.toThrow();
  });

  it("binds every ready source field and never loads failed records", async () => {
    const document = paper("paper:a", ["alpha"]); const fields = [
      "paperKey", "modelId", "dimension", "textHash", "contentHash", "title", "titleVersion", "filePaths", "observationFingerprints", "derivation", "chunkCount", "updatedAt",
    ] as const;
    for (const field of fields) {
      const source = manifest([document], ["paper:failed"]); const record = source.papers["paper:a"] as any;
      if (field === "paperKey") record.paperKey = "paper:other";
      else if (field === "dimension") record.dimension = 3;
      else if (field === "chunkCount") record.chunkCount = 2;
      else if (field === "filePaths") record.filePaths = ["other.pdf"];
      else if (field === "observationFingerprints") record.observationFingerprints = [`sha256:${"e".repeat(64)}`];
      else if (field === "derivation") record.derivation = { ...DERIVATION, chunkerVersion: 3 };
      else record[field] = `${String(record[field])}-changed`;
      const load = vi.fn(async (key: string) => key === "paper:a" ? document : null);
      await expect(buildFullTextGeneration({ manifest: source, loadPaper: load, generationId: "gen-invalid", indexDerivation: INDEX_DERIVATION, spool: new MemorySpool() }))
        .rejects.toMatchObject({ name: "GenerationIndexBuildError", code: "invalid-source" });
      expect(load).not.toHaveBeenCalledWith("paper:failed");
    }
  });

  it("builds empty, failed-only, and tokenless generations", async () => {
    for (const source of [manifest([]), manifest([], ["paper:failed"])]) {
      const built = await buildFullTextGeneration({ manifest: source, loadPaper: vi.fn(), generationId: "gen-empty", indexDerivation: INDEX_DERIVATION, spool: new MemorySpool() });
      expect(built.descriptor).toMatchObject({ lexicalCapability: "none", objects: [], corpusMean: [0, 0] });
    }
    const tokenless = await build([paper("paper:a", ["--- !!!"])]);
    expect(tokenless.result.descriptor.lexicalCapability).toBe("bm25-v1");
    expect(tokenless.result.descriptor.objects.some((entry) => entry.kind === "lexical-postings")).toBe(false);
    expect(tokenless.result.descriptor.objects.map((entry) => entry.kind)).toContain("paper-metadata");
  });

  it("computes the exact float32-row mean and produces deterministic bytes", async () => {
    const docs = [paper("paper:a", ["alpha", "beta"], [0.1, 0.2, 0.3, 0.4])];
    const one = await build(docs); const two = await build(docs);
    expect(one.result.descriptor.corpusMean).toEqual([(new Float32Array([0.1]))[0]! / 2 + (new Float32Array([0.3]))[0]! / 2, (new Float32Array([0.2]))[0]! / 2 + (new Float32Array([0.4]))[0]! / 2]);
    expect(one.result.descriptor).toEqual(two.result.descriptor);
    expect(await collect(one.result.objects())).toEqual(await collect(two.result.objects()));
  });

  it("keeps actual lexical derivation and codec work linear as corpus size doubles", async () => {
    const make = (count: number) => Array.from({ length: count }, (_, index) => paper(`paper:${String(index).padStart(4, "0")}`, [`term${index}`]));
    const observe = () => {
      const calls: GenerationIndexBuildOperation[] = [];
      const instrumentation: GenerationIndexBuildInstrumentation = { onOperation: (event) => calls.push(event) };
      return { calls, instrumentation };
    };
    const smallObserved = observe(); const largeObserved = observe();
    const small = await build(make(32), { blockTargetRows: 64, instrumentation: smallObserved.instrumentation });
    const large = await build(make(64), { blockTargetRows: 64, instrumentation: largeObserved.instrumentation });
    const count = (calls: typeof smallObserved.calls, operation: (typeof calls)[number]["operation"]) => calls.filter((event) => event.operation === operation).length;
    expect(count(smallObserved.calls, "derive-lexical-chunk")).toBe(32);
    expect(count(largeObserved.calls, "derive-lexical-chunk")).toBe(64);
    expect(count(largeObserved.calls, "encode")).toBeLessThanOrEqual(count(smallObserved.calls, "encode") * 2 + 8);
    expect(small.result.diagnostics.derivedChunks).toBe(count(smallObserved.calls, "derive-lexical-chunk"));
    expect(large.result.diagnostics.encodeAttempts).toBe(count(largeObserved.calls, "encode"));
    await small.result.dispose(); await large.result.dispose();
  });

  it("owns spool cleanup across explicit dispose, partial replay, and build failures", async () => {
    const explicit = await build([paper("paper:a", ["alpha"])]); await explicit.result.dispose(); await explicit.result.dispose(); expect(explicit.spool.removes).toBe(1);
    await expect(collect(explicit.result.objects())).rejects.toMatchObject({ code: "invalid-source" });

    const partial = await build([paper("paper:a", ["alpha", "beta"])], { blockTargetRows: 1 });
    const iterator = partial.result.objects()[Symbol.asyncIterator](); await iterator.next(); await iterator.return?.();
    expect(partial.spool.removes).toBe(1); expect(partial.spool.data.size).toBe(0);

    const concurrent = await build([paper("paper:a", ["alpha"])]); const first = concurrent.result.objects()[Symbol.asyncIterator](); const second = concurrent.result.objects()[Symbol.asyncIterator]();
    await first.next(); await expect(second.next()).rejects.toMatchObject({ code: "invalid-source" }); await first.return?.(); expect(concurrent.spool.removes).toBe(1);

    const source = manifest([paper("paper:a", ["alpha"])]); const failing = new MemorySpool(); failing.put = async () => { throw new Error("put failed"); };
    await expect(buildFullTextGeneration({ manifest: source, loadPaper: async () => paper("paper:a", ["alpha"]), generationId: "gen-fail-clean", indexDerivation: INDEX_DERIVATION, spool: failing }))
      .rejects.toMatchObject({ code: "spool-failed" });
    expect(failing.removes).toBe(1);

    const retry = await build([paper("paper:retry", ["alpha"])]);
    const cleanupError = new Error("cleanup denied");
    retry.spool.removeAll = vi.fn()
      .mockRejectedValueOnce(cleanupError)
      .mockImplementationOnce(async () => { retry.spool.data.clear(); });
    await expect(retry.result.dispose()).rejects.toMatchObject({ code: "spool-failed", cause: cleanupError });
    expect(retry.spool.data.size).toBeGreaterThan(0);
    await expect(retry.result.dispose()).resolves.toBeUndefined();
    expect(retry.spool.removeAll).toHaveBeenCalledTimes(2);
    expect(retry.spool.data.size).toBe(0);

    const primary = await build([paper("paper:primary", ["alpha"])]);
    const primaryError = new Error("read failed");
    primary.spool.read = async () => { throw primaryError; };
    primary.spool.removeAll = vi.fn(async () => { throw new Error("cleanup also failed"); });
    const rejection = await collect(primary.result.objects()).catch((caught) => caught);
    expect(rejection).toMatchObject({ code: "spool-failed", cause: primaryError });
    expect(rejection).not.toMatchObject({ message: "failed to clean generation object spool" });
    await expect(primary.result.dispose()).rejects.toMatchObject({ code: "spool-failed" });
  });

  it("retries disposal after removeAll throws synchronously", async () => {
    const built = await build([paper("paper:sync-cleanup", ["alpha"])]);
    const cleanupError = new Error("synchronous cleanup denied");
    built.spool.removeAll = vi.fn()
      .mockImplementationOnce(() => { throw cleanupError; })
      .mockImplementationOnce(async () => { built.spool.data.clear(); });

    await expect(built.result.dispose()).rejects.toMatchObject({ code: "spool-failed", cause: cleanupError });
    await expect(built.result.dispose()).resolves.toBeUndefined();
    expect(built.spool.removeAll).toHaveBeenCalledTimes(2);
    expect(built.spool.data.size).toBe(0);
  });

  it("retries disposal after an immediately rejected removeAll promise", async () => {
    const built = await build([paper("paper:rejected-cleanup", ["alpha"])]);
    const cleanupError = new Error("rejected cleanup denied");
    built.spool.removeAll = vi.fn()
      .mockImplementationOnce(() => Promise.reject(cleanupError))
      .mockImplementationOnce(async () => { built.spool.data.clear(); });

    await expect(built.result.dispose()).rejects.toMatchObject({ code: "spool-failed", cause: cleanupError });
    await expect(built.result.dispose()).resolves.toBeUndefined();
    expect(built.spool.removeAll).toHaveBeenCalledTimes(2);
    expect(built.spool.data.size).toBe(0);
  });

  it("shares one in-flight disposal attempt across concurrent callers", async () => {
    const built = await build([paper("paper:concurrent-cleanup", ["alpha"])]);
    const gate = Promise.withResolvers<void>();
    built.spool.removeAll = vi.fn(() => gate.promise);

    const first = built.result.dispose();
    const second = built.result.dispose();
    const third = built.result.dispose();
    expect(second).toBe(first);
    expect(third).toBe(first);
    expect(built.spool.removeAll).toHaveBeenCalledTimes(1);

    gate.resolve();
    await expect(Promise.all([first, second, third])).resolves.toEqual([undefined, undefined, undefined]);
    await expect(built.result.dispose()).resolves.toBeUndefined();
    expect(built.spool.removeAll).toHaveBeenCalledTimes(1);
  });

  it("snapshots before async loading and validates schema, optional binding fields, vectors, and title fallback", async () => {
    const document = { ...paper("paper:a", ["alpha"]), contentHash: `sha256:${"e".repeat(64)}`, titleVersion: 3 };
    const source = manifest([document]); const record = source.papers[document.paperKey] as any; record.contentHash = document.contentHash; record.titleVersion = 3;
    const gate = Promise.withResolvers<void>();
    const pending = buildFullTextGeneration({ manifest: source, loadPaper: async () => { await gate.promise; return document; }, generationId: "gen-snapshot", indexDerivation: INDEX_DERIVATION, spool: new MemorySpool(), titles: new Map([[document.paperKey, "   "]]) });
    record.textHash = `sha256:${"f".repeat(64)}`; gate.resolve();
    const built = await pending; const metadataRef = built.descriptor.objects.find((entry) => entry.kind === "paper-metadata")!;
    const metadataWrite = (await collect(built.objects())).find((entry) => entry.path === metadataRef.path)!;
    expect(decodePaperMetadataBlock(metadataWrite.bytes).records[0]!.title).toBe(document.title);

    for (const changed of [
      { schemaVersion: 1 }, { vectors: [1, 2] }, { vectors: new Float32Array([1]) }, { vectors: new Float32Array([Number.NaN, 2]) },
    ]) {
      const invalid = Object.assign({ ...document }, changed) as FullTextPaperDocument;
      await expect(buildFullTextGeneration({ manifest: source, loadPaper: async () => invalid, generationId: "gen-binding", indexDerivation: INDEX_DERIVATION, spool: new MemorySpool() }))
        .rejects.toMatchObject({ code: "invalid-source" });
    }
  });

  it("releases unconsumed builder spools on exact replay and precondition rejection", async () => {
    const documents = [paper("paper:a", ["alpha beta"])];
    const first = await build(documents);
    const storage = memoryStorage();
    const writeBinary = storage.writeBinary!;
    storage.writeBinary = vi.fn(async (path, bytes) => {
      expect(first.spool.data.size).toBeGreaterThan(0);
      await writeBinary(path, bytes);
    });
    const index = new FullTextGenerationIndexStore(storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION);
    await index.stageAndPromote({ descriptor: first.result.descriptor, objects: first.result.objects(), writerToken: `writer-first-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 7 });
    expect(first.spool.removes).toBe(1);

    const replay = await build(documents);
    const writesBefore = vi.mocked(storage.writeBinary!).mock.calls.length;
    await index.stageAndPromote({ descriptor: replay.result.descriptor, objects: replay.result.objects(), writerToken: `writer-replay-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 7 });
    expect(replay.spool.removes).toBe(1);
    expect(replay.spool.data.size).toBe(0);
    expect(vi.mocked(storage.writeBinary!).mock.calls.length).toBe(writesBefore);

    const rejected = await build([paper("paper:b", ["beta"])]);
    await expect(index.stageAndPromote({ descriptor: rejected.result.descriptor, objects: rejected.result.objects(), writerToken: "weak", expectedCurrent: null, sourceCurrentRevision: () => 7 }))
      .rejects.toMatchObject({ code: "invalid" });
    expect(rejected.spool.removes).toBe(1);
    expect(rejected.spool.data.size).toBe(0);
  });

  it("passes real store open and matches P3 dense, BM25, and RRF fields exactly", async () => {
    const papers = [
      paper("paper:a", ["Pan-STARRS alpha", "中文检索"], [1, 0, 0.8, 0.2], "Survey Alpha"),
      paper("paper:b", ["beta unrelated"], [0, 1], "Exact title only"),
      paper("paper:c", ["哈哈 alpha alpha"], [0.7, 0.3], "Mixed Han"),
    ];
    const built = await build(papers, { blockTargetRows: 1 });
    const storage = memoryStorage(); const store = new FullTextGenerationIndexStore(storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION);
    await store.stageAndPromote({ descriptor: built.result.descriptor, objects: built.result.objects(), writerToken: `writer-builder-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 7 });
    const opened = (await store.openCurrent())!; await expect(opened.validateClosure()).resolves.toBeUndefined();
    const titles = new Map(papers.map((entry) => [entry.paperKey, entry.title!]));
    const queryVector = new Float32Array([1, 0]);
    const dense = await searchGenerationDense({ generation: opened, queryVector, centerCorpus: false, limit: 3 });
    const denseP3 = searchKnowledgeBase({ papers, queryVector, centerCorpus: false, limit: 3 });
    expect(dense).toEqual(denseP3); dense.forEach((entry, index) => expect(Object.is(entry.score, denseP3[index]!.score)).toBe(true));
    const bm25 = await searchGenerationBm25({ generation: opened, queryText: "panstarrs 哈哈 alpha", limit: 3 });
    const bm25P3 = searchKnowledgeBaseBm25({ papers, titles, queryText: "panstarrs 哈哈 alpha", limit: 3 });
    expect(bm25).toEqual(bm25P3); bm25.forEach((entry, index) => expect(Object.is(entry.score, bm25P3[index]!.score)).toBe(true));
    const rrf = fusePaperRankingsRrf({ rankings: [dense, bm25], candidateLimit: 3, limit: 3 });
    const rrfP3 = fusePaperRankingsRrf({ rankings: [denseP3, bm25P3], candidateLimit: 3, limit: 3 });
    expect(rrf).toEqual(rrfP3); rrf.forEach((entry, index) => expect(Object.is(entry.score, rrfP3[index]!.score)).toBe(true));
  });

  it("rejects a derived chunk as soon as its occurrence count crosses 65536", async () => {
    const uniqueHan = (count: number) => Array.from({ length: count }, (_, index) => String.fromCodePoint(0x20000 + index)).join("");
    let low = 8_000; let high = 20_000;
    while (low + 1 < high) { const middle = Math.floor((low + high) / 2); try { deriveLexicalChunk(uniqueHan(middle), 0); low = middle; } catch { high = middle; } }
    expect(deriveLexicalChunk(uniqueHan(low), 0).occurrences.length).toBeLessThanOrEqual(65_536);
    expect(() => deriveLexicalChunk(uniqueHan(high), 0)).toThrow(/65536/);
    await expect(build([paper("paper:over", [uniqueHan(high)])])).rejects.toMatchObject({ code: "object-too-large" });
  });

  it("splits because of the 4 MiB cap and proves all coverage through real store closure", async () => {
    const largeText = "alpha ".repeat(116_000);
    const papers = [paper("paper:large", Array.from({ length: 7 }, () => largeText))];
    const built = await build(papers);
    const references = built.result.descriptor.objects;
    expect(references.every((entry) => entry.byteLength <= MAX_BINARY_OBJECT_BYTES)).toBe(true);
    const evidence = references.filter((entry) => entry.kind === "evidence");
    expect(evidence.length).toBeGreaterThan(1);
    expect(Math.max(...evidence.map((entry) => entry.byteLength))).toBeGreaterThan(MAX_BINARY_OBJECT_BYTES * 0.9);
    expect(evidence.reduce((total, entry) => total + entry.recordCount, 0)).toBe(papers[0]!.chunks.length);
    const vectors = references.filter((entry) => entry.kind === "vector");
    expect(vectors.map(({ recordStart, recordCount }) => [recordStart, recordCount]))
      .toEqual(evidence.map(({ recordStart, recordCount }) => [recordStart, recordCount]));

    const storage = memoryStorage();
    const opened = await new FullTextGenerationIndexStore(storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION).stageAndPromote({
      descriptor: built.result.descriptor, objects: built.result.objects(), writerToken: `writer-cap-${"f".repeat(32)}`,
      expectedCurrent: null, sourceCurrentRevision: () => 7,
    });
    await expect(opened.validateClosure()).resolves.toBeUndefined();
    expect(opened.descriptor.corpusStats.chunkCount).toBe(papers[0]!.chunks.length);

    const limitedSpool = new MemorySpool(); const document = paper("paper:a", ["alpha"]);
    await expect(buildFullTextGeneration({ manifest: manifest([document]), loadPaper: async () => document, generationId: "gen-limit", indexDerivation: INDEX_DERIVATION, spool: limitedSpool, maxObjects: 1 }))
      .rejects.toMatchObject({ code: "object-limit" });
    expect(limitedSpool.removes).toBe(1); expect(limitedSpool.data.size).toBe(0);
  }, 30_000);

  it("normalizes non-object spool results and throwing getters as spool-failed", async () => {
    const document = paper("paper:a", ["alpha"]); const source = manifest([document]);
    const invalidPut = new MemorySpool();
    invalidPut.put = async () => null as unknown as GenerationObjectReference;
    await expect(buildFullTextGeneration({ manifest: source, loadPaper: async () => document, generationId: "gen-null-put", indexDerivation: INDEX_DERIVATION, spool: invalidPut }))
      .rejects.toMatchObject({ code: "spool-failed" });

    const getterPut = new MemorySpool();
    getterPut.put = async () => Object.defineProperty({}, "kind", { get() { throw new Error("put getter failed"); } }) as GenerationObjectReference;
    await expect(buildFullTextGeneration({ manifest: source, loadPaper: async () => document, generationId: "gen-getter-put", indexDerivation: INDEX_DERIVATION, spool: getterPut }))
      .rejects.toMatchObject({ code: "spool-failed" });

    class ThrowingBytes extends Uint8Array { override get byteLength(): number { throw new Error("read getter failed"); } }
    for (const returned of [null, new ThrowingBytes(1)]) {
      const built = await build([document]);
      built.spool.read = async () => returned as Uint8Array;
      await expect(collect(built.result.objects())).rejects.toMatchObject({ code: "spool-failed" });
      expect(built.spool.removes).toBe(1);
    }
  });

  it("types missing/corrupt source, spool failures, replay corruption, and preserves AbortError", async () => {
    const document = paper("paper:a", ["alpha"]); const source = manifest([document]);
    await expect(buildFullTextGeneration({ manifest: source, loadPaper: async () => null, generationId: "gen-missing", indexDerivation: INDEX_DERIVATION, spool: new MemorySpool() }))
      .rejects.toMatchObject({ code: "invalid-source" });
    const badSpool = new MemorySpool(); badSpool.put = async () => { throw new Error("put failed"); };
    await expect(buildFullTextGeneration({ manifest: source, loadPaper: async () => document, generationId: "gen-spool", indexDerivation: INDEX_DERIVATION, spool: badSpool }))
      .rejects.toMatchObject({ code: "spool-failed" });
    const built = await build([document]); const ref = built.result.descriptor.objects[0]!; built.spool.data.get(ref.path)![0] ^= 1;
    await expect(collect(built.result.objects())).rejects.toMatchObject({ code: "spool-failed" });
    expect(built.spool.removes).toBe(1); expect(built.spool.data.size).toBe(0);
    const loadError = new Error("load failed");
    await expect(buildFullTextGeneration({ manifest: source, loadPaper: async () => { throw loadError; }, generationId: "gen-load", indexDerivation: INDEX_DERIVATION, spool: new MemorySpool() }))
      .rejects.toMatchObject({ code: "invalid-source", cause: loadError });
    const controller = new AbortController(); controller.abort();
    await expect(build([document], { signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
    expect(GenerationIndexBuildError).toBeDefined();
  });
});
