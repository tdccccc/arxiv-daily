import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { createEvidenceChunkId, type EvidenceChunk } from "../src/library/fulltext/evidence-chunk";
import {
  GENERATION_DESCRIPTOR_FORMAT_VERSION,
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
  blockObjectChecksum,
  encodeEvidenceBlock,
  encodeGenerationDescriptor,
  encodeVectorBlock,
  type GenerationDescriptor,
} from "../src/library/fulltext/generation-index-format";
import {
  CURRENT_GENERATION_POINTER_FORMAT_VERSION,
  CURRENT_GENERATION_POINTER_SCHEMA_VERSION,
  FullTextGenerationIndexStore,
  FullTextGenerationIndexStoreError,
  computeCanonicalVectorMean,
  decodeCurrentGenerationPointer,
  encodeCurrentGenerationPointer,
  type CurrentGenerationPointer,
  type GenerationObjectWrite,
} from "../src/library/fulltext/generation-index-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const OTHER = `sha256:${"c".repeat(64)}`;
const BASE = `arxiv-daily/.index/personal-library-search-index/${"a".repeat(64)}/${"b".repeat(64)}`;
const CURRENT = `${BASE}/current.json`;
const BACKUP = `${CURRENT}.backup`;
const PROMOTION_CLAIM = `${BASE}/.current-promotion-claim.json`;

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((settle) => { resolve = settle; });
  return { promise, resolve };
}

function chunk(index: number, text = `chunk ${index}`): EvidenceChunk {
  const identity = {
    text,
    headings: ["Methods"],
    locator: { pageStart: 1 },
    derivation: { parser: { id: "fixture", version: "1" }, chunkerVersion: 2, embeddingInputVersion: 1 },
  };
  return { id: createEvidenceChunkId(identity), index, page: 1, ...identity };
}

function emptyFixture(generationId: string, sourceRevision: number) {
  const descriptor: GenerationDescriptor = {
    formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION,
    schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION,
    generationId,
    sourceRevision,
    scopeFingerprint: SCOPE,
    identificationFingerprint: IDENTIFICATION,
    modelId: "model-a",
    dimension: 2,
    corpusMean: [0, 0],
    corpusStats: { indexedPaperCount: 0, chunkCount: 0 },
    indexDerivation: { builderVersion: 1, denseCenteringVersion: 1, tokenizerVersion: 1, postingsVersion: 1 },
    objects: [],
  };
  return { descriptor, objects: [] as GenerationObjectWrite[] };
}

function fixture(generationId: string, sourceRevision: number, values = [1, 2, 3, 4]) {
  const vector = encodeVectorBlock({ rowStart: 0, dimension: 2, vectors: new Float32Array(values) });
  const evidence = encodeEvidenceBlock({ rowStart: 0, records: [
    { paperIndex: 0, paperKey: "paper:a", vectorRow: 0, chunk: chunk(0) },
    { paperIndex: 1, paperKey: "paper:b", vectorRow: 1, chunk: chunk(0) },
  ] });
  const objects: GenerationObjectWrite[] = [
    { path: "objects/000.vector.bin", bytes: vector },
    { path: "objects/000.evidence.bin", bytes: evidence },
  ];
  const descriptor: GenerationDescriptor = {
    formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION,
    schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION,
    generationId,
    sourceRevision,
    scopeFingerprint: SCOPE,
    identificationFingerprint: IDENTIFICATION,
    modelId: "model-a",
    dimension: 2,
    corpusMean: [(values[0]! + values[2]!) / 2, (values[1]! + values[3]!) / 2],
    corpusStats: { indexedPaperCount: 2, chunkCount: 2 },
    indexDerivation: { builderVersion: 1, denseCenteringVersion: 1, tokenizerVersion: 1, postingsVersion: 1 },
    objects: [
      { kind: "vector", path: objects[0]!.path, byteLength: vector.byteLength, recordStart: 0, recordCount: 2, checksum: blockObjectChecksum(vector) },
      { kind: "evidence", path: objects[1]!.path, byteLength: evidence.byteLength, recordStart: 0, recordCount: 2, checksum: blockObjectChecksum(evidence) },
    ],
  };
  return { descriptor, objects };
}

interface MemoryBackend {
  text: Map<string, string>;
  binary: Map<string, Uint8Array>;
  dirs: Set<string>;
}

function memoryStorage(capabilities = true, backend: MemoryBackend = {
  text: new Map(), binary: new Map(), dirs: new Set(),
}) {
  const { text, binary, dirs } = backend;
  let atomicHook: ((path: string, content: string) => Promise<void>) | undefined;
  let textReadHook: ((path: string, value: string | undefined) => Promise<string>) | undefined;
  let binaryWriteHook: ((path: string, bytes: Uint8Array) => Promise<void>) | undefined;
  let binaryReadHook: ((path: string, bytes: Uint8Array) => Promise<ArrayBuffer>) | undefined;
  let exclusiveHook: ((path: string, content: string) => Promise<boolean>) | undefined;
  let removeHook: ((path: string) => Promise<void>) | undefined;
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""),
    readText: vi.fn(async (path) => {
      const value = text.get(path);
      if (textReadHook) return textReadHook(path, value);
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    }),
    writeText: vi.fn(async (path, value) => { text.set(path, value); }),
    ...(capabilities ? {
      writeTextAtomic: vi.fn(async (path: string, value: string) => { if (atomicHook) return atomicHook(path, value); text.set(path, value); }),
      createTextExclusive: vi.fn(async (path: string, value: string) => {
        if (exclusiveHook) return exclusiveHook(path, value);
        if (text.has(path) || binary.has(path) || dirs.has(path)) return false;
        text.set(path, value);
        return true;
      }),
    } : {}),
    exists: vi.fn(async (path) => text.has(path) || binary.has(path) || dirs.has(path)),
    mkdir: vi.fn(async (path) => { dirs.add(path); }),
    remove: vi.fn(async (path) => {
      if (removeHook) await removeHook(path);
      const prefix = `${path}/`;
      for (const key of [...text.keys()]) if (key === path || key.startsWith(prefix)) text.delete(key);
      for (const key of [...binary.keys()]) if (key === path || key.startsWith(prefix)) binary.delete(key);
      for (const key of [...dirs]) if (key === path || key.startsWith(prefix)) dirs.delete(key);
    }),
    rename: vi.fn(async () => undefined),
    ...(capabilities ? {
      writeBinary: vi.fn(async (path: string, buffer: ArrayBuffer) => {
        const bytes = new Uint8Array(buffer).slice();
        if (binaryWriteHook) await binaryWriteHook(path, bytes);
        else binary.set(path, bytes);
      }),
      readBinary: vi.fn(async (path: string) => {
        const bytes = binary.get(path); if (!bytes) throw new Error(`missing ${path}`);
        return binaryReadHook ? binaryReadHook(path, bytes) : bytes.slice().buffer;
      }),
    } : {}),
  };
  return {
    storage, text, binary, dirs,
    setAtomicHook(hook?: typeof atomicHook) { atomicHook = hook; },
    setTextReadHook(hook?: typeof textReadHook) { textReadHook = hook; },
    setBinaryWriteHook(hook?: typeof binaryWriteHook) { binaryWriteHook = hook; },
    setBinaryReadHook(hook?: typeof binaryReadHook) { binaryReadHook = hook; },
    setExclusiveHook(hook?: typeof exclusiveHook) { exclusiveHook = hook; },
    setRemoveHook(hook?: typeof removeHook) { removeHook = hook; },
  };
}

function store(storage: StorageAdapter, options: ConstructorParameters<typeof FullTextGenerationIndexStore>[4] = {}) {
  return new FullTextGenerationIndexStore(storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION, options);
}

async function publish(target: FullTextGenerationIndexStore, generationId: string, revision: number, expectedCurrent: null | { generationId: string; sourceRevision: number } = null) {
  const built = fixture(generationId, revision);
  return target.stageAndPromote({
    ...built,
    writerToken: `writer-${generationId}-${"f".repeat(32)}`,
    expectedCurrent,
    sourceCurrentRevision: async () => revision,
  });
}

function generationPath(id: string, child: string) { return `${BASE}/generations/${id}/${child}`; }

function pointerFor(descriptor: GenerationDescriptor): CurrentGenerationPointer {
  return decodeCurrentGenerationPointer(encodeCurrentGenerationPointer({
    formatVersion: CURRENT_GENERATION_POINTER_FORMAT_VERSION,
    schemaVersion: CURRENT_GENERATION_POINTER_SCHEMA_VERSION,
    generationId: descriptor.generationId,
    sourceRevision: descriptor.sourceRevision,
    scopeFingerprint: descriptor.scopeFingerprint,
    identificationFingerprint: descriptor.identificationFingerprint,
    descriptorChecksum: `sha256:${"d".repeat(64)}`,
    checksum: `sha256:${"0".repeat(64)}`,
  }));
}

describe("current generation pointer", () => {
  it("round-trips a strict checksummed identity-bound pointer and rejects tampering/future schema", () => {
    const pointer = pointerFor(fixture("gen-a", 1).descriptor);
    expect(decodeCurrentGenerationPointer(encodeCurrentGenerationPointer(pointer))).toEqual(pointer);
    const raw = JSON.parse(encodeCurrentGenerationPointer(pointer));
    raw.generationId = "gen-b";
    expect(() => decodeCurrentGenerationPointer(JSON.stringify(raw))).toThrow(/checksum/i);
    raw.generationId = "gen-a"; raw.schemaVersion += 1;
    expect(() => decodeCurrentGenerationPointer(JSON.stringify(raw))).toThrow(/schema version/i);
    raw.schemaVersion -= 1; raw.extra = true;
    expect(() => decodeCurrentGenerationPointer(JSON.stringify(raw))).toThrow(/unknown field/i);
  });
});

describe("FullTextGenerationIndexStore promotion", () => {
  it("fails closed when binary or atomic capabilities are absent", async () => {
    const memory = memoryStorage(false);
    await expect(publish(store(memory.storage), "gen-a", 1)).rejects.toMatchObject({ code: "capability-unsupported" });
    expect(memory.text.size + memory.binary.size).toBe(0);
  });

  it("rejects a weak writer token before storage I/O", async () => {
    const memory = memoryStorage();
    const built = fixture("gen-weak-token", 1);
    await expect(store(memory.storage).stageAndPromote({
      ...built, writerToken: "weak", expectedCurrent: null, sourceCurrentRevision: () => 1,
    })).rejects.toMatchObject({ code: "invalid" });
    expect(memory.text.size + memory.binary.size + memory.dirs.size).toBe(0);
  });

  it("fails closed without createTextExclusive before writing objects", async () => {
    const memory = memoryStorage();
    delete memory.storage.createTextExclusive;
    await expect(publish(store(memory.storage), "gen-no-exclusive", 1))
      .rejects.toMatchObject({ code: "capability-unsupported" });
    expect(memory.binary.size).toBe(0);
  });

  it("binds a strict staging claim before any object write and rejects claim conflict without cleanup", async () => {
    const memory = memoryStorage();
    memory.setExclusiveHook(async (path, content) => {
      memory.text.set(path, content.replace("writer-gen-claim", "writer-other"));
      return false;
    });
    await expect(publish(store(memory.storage), "gen-claim", 1))
      .rejects.toMatchObject({ code: "concurrent" });
    expect(memory.binary.size).toBe(0);
    expect(memory.text.has(generationPath("gen-claim", ".staging-claim.json"))).toBe(true);
    expect(memory.storage.remove).not.toHaveBeenCalledWith(generationPath("gen-claim", "").replace(/\/$/, ""));
  });

  it("treats a staging claim create exception as uncertain without inferring ownership", async () => {
    const memory = memoryStorage();
    memory.setExclusiveHook(async (path, content) => {
      memory.text.set(path, content);
      throw new Error("staging claim EIO after possible create");
    });
    await expect(publish(store(memory.storage), "gen-claim-uncertain", 1))
      .rejects.toMatchObject({ code: "claim-uncertain" });
    const directory = generationPath("gen-claim-uncertain", "").replace(/\/$/, "");
    expect(memory.binary.size).toBe(0);
    expect(memory.text.has(`${directory}/.staging-claim.json`)).toBe(true);
    expect(memory.storage.remove).not.toHaveBeenCalledWith(directory);
    expect(memory.storage.remove).not.toHaveBeenCalledWith(`${directory}/.staging-claim.json`);
    expect(memory.text.has(CURRENT)).toBe(false);
  });

  it("does not pass a pre-existing identical staging claim when exclusive create throws", async () => {
    const memory = memoryStorage();
    memory.setExclusiveHook(async (path, content) => {
      memory.text.set(path, content);
      throw new Error("first exclusive create outcome unknown");
    });
    await expect(publish(store(memory.storage), "gen-identical-claim", 1))
      .rejects.toMatchObject({ code: "claim-uncertain" });
    memory.setExclusiveHook(async (path, content) => {
      expect(memory.text.get(path)).toBe(content);
      throw new Error("EIO with identical claim already present");
    });
    await expect(publish(store(memory.storage), "gen-identical-claim", 1))
      .rejects.toMatchObject({ code: "claim-uncertain" });
    expect(memory.binary.size).toBe(0);
    expect(memory.text.has(CURRENT)).toBe(false);
  });

  it("treats promotion claim create exceptions as uncertain without cleanup or pointer writes", async () => {
    const memory = memoryStorage();
    memory.setExclusiveHook(async (path, content) => {
      if (path === PROMOTION_CLAIM) {
        memory.text.set(path, content);
        throw new Error("promotion claim EIO after possible create");
      }
      if (memory.text.has(path)) return false;
      memory.text.set(path, content);
      return true;
    });
    await expect(publish(store(memory.storage), "gen-promotion-uncertain", 1))
      .rejects.toMatchObject({ code: "claim-uncertain" });
    const directory = generationPath("gen-promotion-uncertain", "").replace(/\/$/, "");
    expect(memory.binary.has(`${directory}/objects/000.vector.bin`)).toBe(true);
    expect(memory.text.has(`${directory}/descriptor.json`)).toBe(true);
    expect(memory.text.has(PROMOTION_CLAIM)).toBe(true);
    expect(memory.text.has(CURRENT)).toBe(false);
    expect(memory.storage.remove).not.toHaveBeenCalledWith(directory);
    expect(memory.storage.remove).not.toHaveBeenCalledWith(PROMOTION_CLAIM);
  });

  it("arbitrates different-generation promotion across adapters with one root claim", async () => {
    const backend: MemoryBackend = { text: new Map(), binary: new Map(), dirs: new Set() };
    const first = memoryStorage(true, backend);
    const second = memoryStorage(true, backend);
    await publish(store(first.storage), "gen-old", 1);
    const firstStaged = deferred();
    const secondStaged = deferred();
    const releaseFirstPromotion = deferred();
    const releaseSecondPromotion = deferred();
    const firstOwnsPromotion = deferred();
    const releaseFirstOwner = deferred();
    const firstRun = store(first.storage, {
      beforePointerPromotion: async () => { firstStaged.resolve(); await releaseFirstPromotion.promise; },
      afterPromotionClaimAcquired: async () => {
        expect(JSON.parse(backend.text.get(PROMOTION_CLAIM)!)).toMatchObject({
          formatVersion: 1,
          schemaVersion: 1,
          operation: "promote",
          writerToken: `writer-gen-first-${"f".repeat(32)}`,
          candidateGenerationId: "gen-first",
          sourceRevision: 2,
          expectedCurrent: { generationId: "gen-old", sourceRevision: 1 },
          scopeFingerprint: SCOPE,
          identificationFingerprint: IDENTIFICATION,
        });
        firstOwnsPromotion.resolve();
        await releaseFirstOwner.promise;
      },
    });
    const secondRun = store(second.storage, {
      beforePointerPromotion: async () => { secondStaged.resolve(); await releaseSecondPromotion.promise; },
    });
    const expected = { generationId: "gen-old", sourceRevision: 1 };
    const firstPromise = publish(firstRun, "gen-first", 2, expected);
    const secondPromise = publish(secondRun, "gen-second", 2, expected);
    await Promise.all([firstStaged.promise, secondStaged.promise]);
    releaseFirstPromotion.resolve();
    await firstOwnsPromotion.promise;
    releaseSecondPromotion.resolve();
    const secondResult = await secondPromise.then(
      (value) => ({ status: "fulfilled" as const, value }),
      (reason) => ({ status: "rejected" as const, reason }),
    );
    releaseFirstOwner.resolve();
    const firstResult = await firstPromise.then(
      (value) => ({ status: "fulfilled" as const, value }),
      (reason) => ({ status: "rejected" as const, reason }),
    );
    const results = [firstResult, secondResult];
    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    expect(results.filter((result) => result.status === "rejected" && result.reason.code === "concurrent")).toHaveLength(1);
    const winner = decodeCurrentGenerationPointer(backend.text.get(CURRENT)!);
    expect(["gen-first", "gen-second"]).toContain(winner.generationId);
    const loser = winner.generationId === "gen-first" ? "gen-second" : "gen-first";
    expect([...backend.binary.keys()].some((path) => path.startsWith(`${generationPath(loser, "").replace(/\/$/, "")}/`))).toBe(false);
    expect(backend.text.has(PROMOTION_CLAIM)).toBe(false);
    expect(decodeCurrentGenerationPointer(backend.text.get(BACKUP)!)).toMatchObject({ generationId: "gen-old" });
  });

  it("arbitrates same-generation writers across adapters sharing one backend", async () => {
    const backend: MemoryBackend = { text: new Map(), binary: new Map(), dirs: new Set() };
    const first = memoryStorage(true, backend);
    const second = memoryStorage(true, backend);
    const built = fixture("gen-shared", 1);
    const results = await Promise.allSettled([
      firstStore().stageAndPromote({ ...built, writerToken: `writer-first-${"a".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1 }),
      secondStore().stageAndPromote({ ...built, writerToken: `writer-second-${"b".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1 }),
    ]);
    function firstStore() { return store(first.storage); }
    function secondStore() { return store(second.storage); }
    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    expect(results.filter((result) => result.status === "rejected" && result.reason.code === "concurrent")).toHaveLength(1);
    expect(backend.binary.size).toBe(2);
    await expect(store(first.storage).openCurrent()).resolves.toMatchObject({ descriptor: { generationId: "gen-shared" } });
  });

  it("writes only a unique generation, verifies each object and descriptor, then promotes backup and primary", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const first = await publish(index, "gen-a", 1);
    expect(first.descriptor.generationId).toBe("gen-a");
    expect(memory.text.has(CURRENT)).toBe(true);
    expect(memory.text.has(BACKUP)).toBe(false);
    expect(memory.binary.has(generationPath("gen-a", "objects/000.vector.bin"))).toBe(true);
    expect(JSON.parse(memory.text.get(generationPath("gen-a", ".staging-claim.json"))!)).toMatchObject({
      formatVersion: 1,
      schemaVersion: 1,
      generationId: "gen-a",
      sourceRevision: 1,
      scopeFingerprint: SCOPE,
      identificationFingerprint: IDENTIFICATION,
      descriptorChecksum: expect.stringMatching(/^sha256:[a-f0-9]{64}$/),
      writerToken: `writer-gen-a-${"f".repeat(32)}`,
    });
    expect(memory.storage.readBinary).toHaveBeenCalledTimes(4); // write verification plus full pre-promotion closure validation
    expect(memory.storage.readText).toHaveBeenCalledWith(generationPath("gen-a", "descriptor.json"));
    const second = await publish(index, "gen-b", 2, { generationId: "gen-a", sourceRevision: 1 });
    expect(memory.text.has(BACKUP)).toBe(true);
    expect(second.descriptor.generationId).toBe("gen-b");
    expect(decodeCurrentGenerationPointer(memory.text.get(CURRENT)!)).toMatchObject({ generationId: "gen-b" });
    expect(decodeCurrentGenerationPointer(memory.text.get(BACKUP)!)).toMatchObject({ generationId: "gen-a" });
    expect(memory.binary.has(generationPath("gen-a", "objects/000.vector.bin"))).toBe(true);
  });

  it("rejects descriptor-valid but mismatched same-ordinal vector/evidence coverage", async () => {
    const memory = memoryStorage();
    const built = fixture("gen-misaligned", 1);
    const descriptor: GenerationDescriptor = {
      ...built.descriptor,
      objects: [
        { ...built.descriptor.objects[0]!, path: "objects/vector-a.bin", recordCount: 1 },
        { ...built.descriptor.objects[0]!, path: "objects/vector-b.bin", recordStart: 1, recordCount: 1 },
        built.descriptor.objects[1]!,
      ],
    };
    expect(() => encodeGenerationDescriptor(descriptor)).not.toThrow();
    await expect(store(memory.storage).stageAndPromote({
      descriptor, objects: [], writerToken: `writer-misaligned-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1,
    })).rejects.toMatchObject({ code: "invalid" });
    expect(memory.text.size + memory.binary.size).toBe(0);
  });

  it("rejects complete or partial generation collisions without overwrite", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-a", 1);
    const before = memory.binary.get(generationPath("gen-a", "objects/000.vector.bin"))!.slice();
    await expect(publish(index, "gen-a", 1)).resolves.toMatchObject({ descriptor: { generationId: "gen-a" } });
    expect(memory.binary.get(generationPath("gen-a", "objects/000.vector.bin"))).toEqual(before);
    memory.dirs.add(generationPath("partial", "" ).replace(/\/$/, ""));
    memory.text.set(generationPath("partial", ".staging-claim.json"), "existing-writer-claim");
    await expect(publish(index, "partial", 2, { generationId: "gen-a", sourceRevision: 1 }))
      .rejects.toMatchObject({ code: "concurrent" });
  });

  it("keeps old current on object write/read/checksum and descriptor write/read failures", async () => {
    for (const seam of ["object-write", "object-read", "object-checksum", "descriptor-write", "descriptor-read"] as const) {
      const memory = memoryStorage();
      const index = store(memory.storage);
      await publish(index, "gen-old", 1);
      const old = memory.text.get(CURRENT);
      const vectorPath = generationPath(`gen-${seam}`, "objects/000.vector.bin");
      if (seam === "object-write") memory.setBinaryWriteHook(async () => { throw new Error("write injected"); });
      if (seam === "object-read") memory.setBinaryReadHook(async () => { throw new Error("read injected"); });
      if (seam === "object-checksum") memory.setBinaryWriteHook(async (path, bytes) => { const copy = bytes.slice(); copy[0] = 0; memory.binary.set(path, copy); });
      if (seam === "descriptor-write") memory.setAtomicHook(async (path, value) => { if (path.endsWith("descriptor.json")) throw new Error("descriptor write injected"); memory.text.set(path, value); });
      if (seam === "descriptor-read") {
        const original = memory.storage.readText.bind(memory.storage);
        memory.storage.readText = vi.fn(async (path) => { if (path.endsWith("descriptor.json")) throw new Error("descriptor read injected"); return original(path); });
      }
      await expect(publish(index, `gen-${seam}`, 2, { generationId: "gen-old", sourceRevision: 1 })).rejects.toBeInstanceOf(FullTextGenerationIndexStoreError);
      expect(memory.text.get(CURRENT)).toBe(old);
    }
  });

  it("rejects an embedded block decode failure even when the outer reference checksum is correct", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-old", 1);
    const old = memory.text.get(CURRENT);
    const built = fixture("gen-decode", 2);
    const invalid = built.objects[0]!.bytes.slice();
    invalid[60]! ^= 1; // breaks the block's embedded checksum
    const descriptor: GenerationDescriptor = {
      ...built.descriptor,
      objects: [{ ...built.descriptor.objects[0]!, checksum: blockObjectChecksum(invalid) }, built.descriptor.objects[1]!],
    };
    await expect(index.stageAndPromote({
      descriptor,
      writerToken: `writer-decode-${"f".repeat(32)}`,
      objects: [{ path: built.objects[0]!.path, bytes: invalid }, built.objects[1]!],
      expectedCurrent: { generationId: "gen-old", sourceRevision: 1 },
      sourceCurrentRevision: () => 2,
    })).rejects.toMatchObject({ code: "write-failed" });
    expect(memory.text.get(CURRENT)).toBe(old);
  });

  it("re-reads the staging claim before promotion and rejects lost ownership", async () => {
    const memory = memoryStorage();
    const claimPath = generationPath("gen-claim-tamper", ".staging-claim.json");
    const index = store(memory.storage, {
      beforePointerPromotion: () => {
        const claim = JSON.parse(memory.text.get(claimPath)!);
        memory.text.set(claimPath, JSON.stringify({ ...claim, writerToken: `writer-other-${"e".repeat(32)}` }));
      },
    });
    await expect(publish(index, "gen-claim-tamper", 1)).rejects.toMatchObject({ code: "generation-conflict" });
    expect(memory.text.has(CURRENT)).toBe(false);
    // Ownership was lost, so this writer must not delete the directory.
    expect(memory.storage.remove).not.toHaveBeenCalledWith(generationPath("gen-claim-tamper", "").replace(/\/$/, ""));
  });

  it("does not release a replaced promotion claim or delete a generation after staging ownership changes", async () => {
    const memory = memoryStorage();
    const promotionEntered = deferred();
    const releasePromotion = deferred();
    const index = store(memory.storage, {
      afterPromotionClaimAcquired: async () => {
        promotionEntered.resolve();
        await releasePromotion.promise;
      },
    });
    const publishing = publish(index, "gen-replaced-claims", 1);
    await promotionEntered.promise;
    memory.text.set(PROMOTION_CLAIM, JSON.stringify({ writerToken: `writer-other-${"e".repeat(32)}` }));
    const stagingPath = generationPath("gen-replaced-claims", ".staging-claim.json");
    memory.text.set(stagingPath, JSON.stringify({ writerToken: `writer-other-${"e".repeat(32)}` }));
    releasePromotion.resolve();
    await expect(publishing).rejects.toMatchObject({ code: "stale-claim" });
    expect(memory.text.has(PROMOTION_CLAIM)).toBe(true);
    expect(memory.storage.remove).not.toHaveBeenCalledWith(PROMOTION_CLAIM);
    expect(memory.storage.remove).not.toHaveBeenCalledWith(generationPath("gen-replaced-claims", "").replace(/\/$/, ""));
  });

  it("rechecks staging ownership and current reachability in the cleanup window", async () => {
    const memory = memoryStorage();
    const built = fixture("gen-cleanup-window", 1);
    const pointer = pointerFor(built.descriptor);
    memory.setBinaryWriteHook(async () => {
      memory.text.set(CURRENT, encodeCurrentGenerationPointer(pointer));
      const claimPath = generationPath("gen-cleanup-window", ".staging-claim.json");
      const claim = JSON.parse(memory.text.get(claimPath)!);
      memory.text.set(claimPath, JSON.stringify({ ...claim, writerToken: `writer-other-${"e".repeat(32)}` }));
      throw new Error("early object failure");
    });
    await expect(store(memory.storage).stageAndPromote({
      ...built,
      writerToken: `writer-cleanup-window-${"f".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 1,
    })).rejects.toMatchObject({ code: "write-failed" });
    expect(memory.storage.remove).not.toHaveBeenCalledWith(generationPath("gen-cleanup-window", "").replace(/\/$/, ""));
  });

  it("best-effort removes an owned uncommitted generation after failure without masking the error", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-old", 1);
    memory.setBinaryWriteHook(async () => { throw new Error("object write failed"); });
    await expect(publish(index, "gen-clean", 2, { generationId: "gen-old", sourceRevision: 1 }))
      .rejects.toMatchObject({ code: "write-failed" });
    const directory = generationPath("gen-clean", "").replace(/\/$/, "");
    expect(memory.storage.remove).toHaveBeenCalledWith(directory);
    expect([...memory.text.keys(), ...memory.binary.keys()].some((path) => path.startsWith(`${directory}/`))).toBe(false);

    const cleanupFailure = memoryStorage();
    await publish(store(cleanupFailure.storage), "gen-old", 1);
    cleanupFailure.setBinaryWriteHook(async () => { throw new Error("original failure"); });
    cleanupFailure.storage.remove = vi.fn(async () => { throw new Error("cleanup failure"); });
    await expect(publish(store(cleanupFailure.storage), "gen-cleanup-fails", 2, { generationId: "gen-old", sourceRevision: 1 }))
      .rejects.toMatchObject({ code: "write-failed", cause: expect.objectContaining({ message: "original failure" }) });
  });

  it("preserves a possibly committed generation when CURRENT verification is temporarily unreadable", async () => {
    const memory = memoryStorage();
    let currentReadFailures = 0;
    let commitAttempted = false;
    memory.setAtomicHook(async (path, value) => {
      memory.text.set(path, value);
      if (path === CURRENT) {
        commitAttempted = true;
        throw new Error("commit response lost");
      }
    });
    memory.setTextReadHook(async (path, value) => {
      if (path === CURRENT && commitAttempted && currentReadFailures < 2) {
        currentReadFailures += 1;
        throw new Error("CURRENT temporarily unreadable");
      }
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    });
    await expect(publish(store(memory.storage), "gen-uncertain", 1))
      .rejects.toMatchObject({ code: "commit-uncertain" });
    const directory = generationPath("gen-uncertain", "").replace(/\/$/, "");
    expect(memory.storage.remove).not.toHaveBeenCalledWith(directory);
    expect(memory.binary.has(`${directory}/objects/000.vector.bin`)).toBe(true);
    expect(memory.text.has(`${directory}/descriptor.json`)).toBe(true);
    memory.setAtomicHook();
    memory.setTextReadHook();
    await expect(store(memory.storage).openCurrent()).resolves.toMatchObject({ descriptor: { generationId: "gen-uncertain" } });
  });

  it("never cleans a commit-uncertain candidate after a successor makes it backup", async () => {
    const backend: MemoryBackend = { text: new Map(), binary: new Map(), dirs: new Set() };
    const memory = memoryStorage(true, backend);
    const successorMemory = memoryStorage(true, backend);
    let firstCurrentWrite = true;
    let failCommitRead = false;
    memory.setAtomicHook(async (path, value) => {
      memory.text.set(path, value);
      if (path === CURRENT && firstCurrentWrite) {
        firstCurrentWrite = false;
        failCommitRead = true;
        throw new Error("first CURRENT response lost");
      }
    });
    memory.setTextReadHook(async (path, value) => {
      if (path === CURRENT && failCommitRead) {
        failCommitRead = false;
        throw new Error("commit verification temporarily failed");
      }
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    });
    let successor: Promise<unknown> | undefined;
    memory.setRemoveHook(async (path) => {
      if (path !== PROMOTION_CLAIM || successor) return;
      memory.text.delete(path);
      memory.setRemoveHook();
      successor = publish(store(successorMemory.storage), "gen-successor", 2, {
        generationId: "gen-uncertain-backup",
        sourceRevision: 1,
      });
      await successor;
    });
    await expect(publish(store(memory.storage), "gen-uncertain-backup", 1))
      .rejects.toMatchObject({ code: "commit-uncertain" });
    await successor;
    const candidateDirectory = generationPath("gen-uncertain-backup", "").replace(/\/$/, "");
    expect(memory.binary.has(`${candidateDirectory}/objects/000.vector.bin`)).toBe(true);
    expect(memory.text.has(`${candidateDirectory}/descriptor.json`)).toBe(true);
    expect(decodeCurrentGenerationPointer(memory.text.get(BACKUP)!)).toMatchObject({ generationId: "gen-uncertain-backup" });
    memory.text.set(CURRENT, "corrupt successor pointer");
    await expect(store(memory.storage).openCurrent()).resolves.toMatchObject({ descriptor: { generationId: "gen-uncertain-backup" } });
  });

  it("never removes a generation after commit-wins confirms current", async () => {
    const memory = memoryStorage();
    memory.setAtomicHook(async (path, value) => {
      memory.text.set(path, value);
      if (path === CURRENT) throw new Error("response lost");
    });
    await expect(publish(store(memory.storage), "gen-commit-kept", 1)).resolves.toBeTruthy();
    const directory = generationPath("gen-commit-kept", "").replace(/\/$/, "");
    expect(memory.storage.remove).not.toHaveBeenCalledWith(directory);
    expect(memory.binary.has(`${directory}/objects/000.vector.bin`)).toBe(true);
  });

  it("checks source revision and expected-current optimistic guard before promotion", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-a", 1);
    const old = memory.text.get(CURRENT);
    await expect(index.stageAndPromote({ ...fixture("gen-stale-source", 2), writerToken: `writer-stale-${"f".repeat(32)}`, expectedCurrent: { generationId: "gen-a", sourceRevision: 1 }, sourceCurrentRevision: async () => 3 }))
      .rejects.toMatchObject({ code: "stale-source", expectedRevision: 2, currentRevision: 3 });
    await expect(publish(index, "gen-stale-pointer", 2, { generationId: "other", sourceRevision: 1 }))
      .rejects.toMatchObject({ code: "stale-current" });
    expect(memory.text.get(CURRENT)).toBe(old);
  });

  it("serializes writers for the same adapter/path and prevents lost updates", async () => {
    const memory = memoryStorage();
    const first = store(memory.storage);
    const second = store(memory.storage);
    const results = await Promise.allSettled([
      publish(first, "gen-a", 1),
      publish(second, "gen-b", 1),
    ]);
    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    expect(results.filter((result) => result.status === "rejected" && result.reason.code === "stale-current")).toHaveLength(1);
  });

  it("consumes async object streams in descriptor order and isolates scope/id paths", async () => {
    const memory = memoryStorage();
    const built = fixture("gen-stream", 1);
    async function* objects() { for (const object of built.objects) yield object; }
    await store(memory.storage).stageAndPromote({ ...built, objects: objects(), writerToken: `writer-stream-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1 });

    const isolated = new FullTextGenerationIndexStore(
      memory.storage, DEFAULT_SETTINGS.output, OTHER, IDENTIFICATION,
    );
    expect(isolated.paths.currentPath).not.toBe(CURRENT);
    expect(isolated.paths.currentPath).toContain(`${"c".repeat(64)}/${"b".repeat(64)}`);
    expect([...memory.text.keys()].every((path) => path.startsWith(BASE))).toBe(true);
  });

  it("does not create a backup on first promotion or resurrect a candidate when current write fails", async () => {
    const memory = memoryStorage();
    memory.setAtomicHook(async (path, value) => {
      if (path === CURRENT) throw new Error("current write failed");
      memory.text.set(path, value);
    });
    await expect(publish(store(memory.storage), "gen-first", 1)).rejects.toMatchObject({ code: "write-failed" });
    expect(memory.text.has(BACKUP)).toBe(false);
    memory.setAtomicHook();
    await expect(store(memory.storage).openCurrent()).resolves.toBeNull();
  });

  it("treats current-write committed-then-thrown and exact complete replay as success", async () => {
    const memory = memoryStorage();
    memory.setAtomicHook(async (path, value) => {
      memory.text.set(path, value);
      if (path === CURRENT) throw new Error("response lost");
    });
    await expect(publish(store(memory.storage), "gen-committed", 1)).resolves.toMatchObject({ descriptor: { generationId: "gen-committed" } });
    memory.setAtomicHook();
    const writesBefore = vi.mocked(memory.storage.writeBinary!).mock.calls.length;
    await expect(publish(store(memory.storage), "gen-committed", 1)).resolves.toMatchObject({ descriptor: { generationId: "gen-committed" } });
    expect(vi.mocked(memory.storage.writeBinary!).mock.calls.length).toBe(writesBefore);
  });

  it("keeps before/after promotion seams commit-aware and checks source revision immediately before write", async () => {
    const beforeMemory = memoryStorage();
    await publish(store(beforeMemory.storage), "gen-old", 1);
    const old = beforeMemory.text.get(CURRENT);
    await expect(publish(store(beforeMemory.storage, { beforePointerPromotion: () => { throw new Error("before"); } }), "gen-before", 2, { generationId: "gen-old", sourceRevision: 1 })).rejects.toBeTruthy();
    expect(beforeMemory.text.get(CURRENT)).toBe(old);

    const afterMemory = memoryStorage();
    await publish(store(afterMemory.storage), "gen-old", 1);
    await expect(publish(store(afterMemory.storage, { afterPointerPromotion: () => { throw new Error("post-commit observer failed"); } }), "gen-after", 2, { generationId: "gen-old", sourceRevision: 1 }))
      .resolves.toMatchObject({ descriptor: { generationId: "gen-after" } });
    await expect(store(afterMemory.storage).openCurrent()).resolves.toMatchObject({ descriptor: { generationId: "gen-after" } });

    const revisionMemory = memoryStorage();
    await publish(store(revisionMemory.storage), "gen-old", 1);
    let revision = 2;
    let currentReads = 0;
    revisionMemory.setTextReadHook(async (path, value) => {
      if (path === CURRENT && ++currentReads === 3) revision = 3;
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    });
    await expect(store(revisionMemory.storage).stageAndPromote({
      ...fixture("gen-revision-race", 2),
      writerToken: `writer-revision-${"f".repeat(32)}`,
      expectedCurrent: { generationId: "gen-old", sourceRevision: 1 },
      sourceCurrentRevision: () => revision,
    })).rejects.toMatchObject({ code: "stale-source", expectedRevision: 2, currentRevision: 3 });
    expect(decodeCurrentGenerationPointer(revisionMemory.text.get(CURRENT)!)).toMatchObject({ generationId: "gen-old" });
  });
});

describe("open and bounded reads", () => {
  it("opens a healthy generation with readText only and defers stronger capability gates", async () => {
    const memory = memoryStorage();
    await publish(store(memory.storage), "gen-readonly", 1);
    const readonly = { ...memory.storage };
    delete readonly.writeTextAtomic;
    delete readonly.createTextExclusive;
    delete readonly.readBinary;
    delete readonly.writeBinary;
    const opened = await store(readonly).openCurrent();
    expect(opened).toMatchObject({ descriptor: { generationId: "gen-readonly" } });
    await expect(opened!.readObject(opened!.descriptor.objects[0]!))
      .rejects.toMatchObject({ code: "capability-unsupported" });

    await publish(store(memory.storage), "gen-next", 2, { generationId: "gen-readonly", sourceRevision: 1 });
    memory.text.set(CURRENT, "corrupt");
    await expect(store(readonly).openCurrent()).rejects.toMatchObject({ code: "capability-unsupported" });
  });

  it("supports an empty generation through promotion, open, iteration, and closure validation", async () => {
    const memory = memoryStorage();
    const built = emptyFixture("gen-empty", 1);
    const opened = await store(memory.storage).stageAndPromote({
      ...built,
      writerToken: `writer-empty-${"f".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 1,
    });
    await expect(opened.validateClosure()).resolves.toBeUndefined();
    const seen = [];
    for await (const object of opened.iterateObjects()) seen.push(object);
    expect(seen).toEqual([]);
    await expect(store(memory.storage).openCurrent()).resolves.toMatchObject({
      descriptor: { generationId: "gen-empty", objects: [], corpusStats: { chunkCount: 0, indexedPaperCount: 0 } },
    });
  });

  it("opens from pointer+descriptor without object scans, freezes its private snapshot, and reads bounded objects on demand", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-a", 1);
    vi.mocked(memory.storage.readBinary!).mockClear();
    const opened = await index.openCurrent();
    expect(opened.descriptor.generationId).toBe("gen-a");
    expect(memory.storage.readBinary).not.toHaveBeenCalled();
    expect(() => { (opened.descriptor.objects as any[]).length = 0; }).toThrow();
    const publicRef = { ...opened.descriptor.objects[0]!, path: "objects/../escape.bin" };
    await expect(opened.readObject(publicRef)).rejects.toMatchObject({ code: "invalid" });
    const seen: string[] = [];
    for await (const object of opened.iterateObjects()) seen.push(object.reference.kind);
    expect(seen).toEqual(["vector", "evidence"]);
    expect(opened.diagnostics.maxObjectBytes).toBeGreaterThan(0);
    expect(opened.diagnostics.objectReads).toBe(2);
    memory.binary.get(generationPath("gen-a", "objects/000.vector.bin"))![60]! ^= 1;
    await expect(opened.readRawObject(opened.descriptor.objects[0]!))
      .rejects.toMatchObject({ code: "corrupt-or-unreadable", cause: expect.anything() });
    await expect(opened.readObject(opened.descriptor.objects[0]!))
      .rejects.toMatchObject({ code: "corrupt-or-unreadable", cause: expect.anything() });
    await expect(opened.validateClosure())
      .rejects.toMatchObject({ code: "corrupt-or-unreadable", cause: expect.anything() });
  });

  it("recovers corrupt primary only after backup pointer and generation validate", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-a", 1);
    await publish(index, "gen-b", 2, { generationId: "gen-a", sourceRevision: 1 });
    memory.text.set(CURRENT, "corrupt");
    await expect(index.openCurrent()).resolves.toMatchObject({ descriptor: { generationId: "gen-a" } });
    expect(decodeCurrentGenerationPointer(memory.text.get(CURRENT)!)).toMatchObject({ generationId: "gen-a" });
  });

  it("does not repair current from a backup whose generation object or mean is corrupt", async () => {
    for (const corruption of ["object", "mean"] as const) {
      const memory = memoryStorage();
      const index = store(memory.storage);
      await publish(index, "gen-backup", 1);
      await publish(index, "gen-current", 2, { generationId: "gen-backup", sourceRevision: 1 });
      if (corruption === "object") {
        memory.binary.get(generationPath("gen-backup", "objects/000.vector.bin"))![60]! ^= 1;
      } else {
        const descriptorPath = generationPath("gen-backup", "descriptor.json");
        const descriptor = JSON.parse(memory.text.get(descriptorPath)!);
        descriptor.corpusMean = [999, 999];
        const raw = encodeGenerationDescriptor(descriptor);
        memory.text.set(descriptorPath, raw);
        const backup = decodeCurrentGenerationPointer(memory.text.get(BACKUP)!);
        memory.text.set(BACKUP, encodeCurrentGenerationPointer({
          ...backup,
          descriptorChecksum: blockObjectChecksum(new TextEncoder().encode(raw)),
          checksum: `sha256:${"0".repeat(64)}`,
        }));
      }
      memory.text.set(CURRENT, "corrupt-primary");
      await expect(index.openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
      expect(memory.text.get(CURRENT)).toBe("corrupt-primary");
    }
  });

  it("waits behind a real queued writer and returns its newer current without recovery overwrite", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-a", 1);
    await publish(index, "gen-b", 2, { generationId: "gen-a", sourceRevision: 1 });

    const writerAtPromotion = deferred();
    const releaseWriter = deferred();
    const writer = store(memory.storage, {
      afterPromotionClaimAcquired: async () => {
        writerAtPromotion.resolve();
        await releaseWriter.promise;
      },
    });
    const writing = publish(writer, "gen-c", 3, { generationId: "gen-b", sourceRevision: 2 });
    await writerAtPromotion.promise;
    const descriptorPath = generationPath("gen-b", "descriptor.json");
    const validDescriptor = memory.text.get(descriptorPath)!;
    memory.text.set(descriptorPath, "corrupt descriptor");
    const recoveryStarted = deferred();
    const recovering = store(memory.storage, { beforeRecoveryQueue: () => recoveryStarted.resolve() }).openCurrent();
    await recoveryStarted.promise;
    let recoverySettled = false;
    void recovering.finally(() => { recoverySettled = true; });
    await Promise.resolve();
    expect(recoverySettled).toBe(false);
    memory.text.set(descriptorPath, validDescriptor);
    releaseWriter.resolve();
    await expect(writing).resolves.toMatchObject({ descriptor: { generationId: "gen-c" } });
    await expect(recovering).resolves.toMatchObject({ descriptor: { generationId: "gen-c" } });
    expect(decodeCurrentGenerationPointer(memory.text.get(CURRENT)!)).toMatchObject({ generationId: "gen-c" });
  });

  it("fails closed on a fixed residual promotion claim without time-based stealing", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-a", 1);
    await publish(index, "gen-b", 2, { generationId: "gen-a", sourceRevision: 1 });
    memory.text.set(CURRENT, "corrupt");
    memory.text.set(PROMOTION_CLAIM, JSON.stringify({ writerToken: `writer-crashed-${"e".repeat(32)}` }));
    await expect(store(memory.storage).openCurrent()).rejects.toMatchObject({ code: expect.stringMatching(/concurrent|stale-claim/) });
    expect(memory.text.has(PROMOTION_CLAIM)).toBe(true);
    expect(memory.text.get(CURRENT)).toBe("corrupt");
  });

  it("fails closed for both bad pointers, future primary schema, or invalid backup generation", async () => {
    const both = memoryStorage(); both.text.set(CURRENT, "bad"); both.text.set(BACKUP, "bad too");
    await expect(store(both.storage).openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });

    const future = memoryStorage();
    const raw = JSON.parse(encodeCurrentGenerationPointer(pointerFor(fixture("gen-a", 1).descriptor)));
    raw.schemaVersion += 1; future.text.set(CURRENT, JSON.stringify(raw));
    await expect(store(future.storage).openCurrent()).rejects.toMatchObject({ code: "incompatible" });

    const incomplete = memoryStorage();
    await publish(store(incomplete.storage), "gen-a", 1);
    incomplete.text.set(CURRENT, "bad");
    incomplete.binary.delete(generationPath("gen-a", "objects/000.vector.bin"));
    await expect(store(incomplete.storage).openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    expect(incomplete.text.get(CURRENT)).toBe("bad");
  });

  it("rejects missing/tampered/wrong kind/count/dimension/mean/evidence order and identity", async () => {
    const mutations: Array<(memory: ReturnType<typeof memoryStorage>, id: string) => void> = [
      (m, id) => { m.binary.delete(generationPath(id, "objects/000.vector.bin")); },
      // Outer reference checksum mismatch.
      (m, id) => { m.binary.get(generationPath(id, "objects/000.vector.bin"))![60]! ^= 1; },
      // Outer reference checksum is correct, but the block's embedded checksum is invalid.
      (m, id) => { const bytes = m.binary.get(generationPath(id, "objects/000.vector.bin"))!.slice(); bytes[60]! ^= 1; m.binary.set(generationPath(id, "objects/000.vector.bin"), bytes); const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.objects[0].checksum = blockObjectChecksum(bytes); m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
      (m, id) => { const bytes = m.binary.get(generationPath(id, "objects/000.evidence.bin"))!.slice(); m.binary.set(generationPath(id, "objects/000.vector.bin"), bytes); const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.objects[0].byteLength = bytes.byteLength; d.objects[0].checksum = blockObjectChecksum(bytes); m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
      (m, id) => { const bytes = encodeVectorBlock({ rowStart: 0, dimension: 2, vectors: new Float32Array([1, 2]) }); m.binary.set(generationPath(id, "objects/000.vector.bin"), bytes); const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.objects[0].byteLength = bytes.byteLength; d.objects[0].checksum = blockObjectChecksum(bytes); m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
      (m, id) => { const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.dimension = 3; d.corpusMean = [2, 3, 0]; m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
      (m, id) => { const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.corpusMean = [2.1, 3]; m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
      (m, id) => { const bad = fixture(id, 1); const records = [{ paperIndex: 1, paperKey: "paper:b", vectorRow: 0, chunk: chunk(0) }, { paperIndex: 1, paperKey: "paper:b", vectorRow: 1, chunk: chunk(1) }]; const bytes = encodeEvidenceBlock({ rowStart: 0, records }); m.binary.set(generationPath(id, bad.objects[1]!.path), bytes); const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.objects[1].byteLength = bytes.byteLength; d.objects[1].checksum = blockObjectChecksum(bytes); m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
      (m, id) => { const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.corpusStats.indexedPaperCount = 1; m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
      (m, id) => { const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.scopeFingerprint = OTHER; m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
    ];
    for (let index = 0; index < mutations.length; index += 1) {
      const memory = memoryStorage(); const id = `gen-${index}`; await publish(store(memory.storage), id, 1);
      mutations[index]!(memory, id);
      const opening = store(memory.storage).openCurrent();
      if (index === mutations.length - 1) {
        await expect(opening).rejects.toMatchObject({ code: "incompatible" });
      } else {
        const opened = await opening;
        await expect(opened!.validateClosure()).rejects.toBeTruthy();
      }
    }
  });
});

describe("canonical generation mean", () => {
  it("uses canonical ref/row float64 accumulation order and survives exact JSON roundtrip under cancellation", () => {
    const blocks = [
      encodeVectorBlock({ rowStart: 0, dimension: 1, vectors: new Float32Array([1e20, 1]) }),
      encodeVectorBlock({ rowStart: 2, dimension: 1, vectors: new Float32Array([-1e20, 3]) }),
    ].map((bytes) => ({ dimension: 1, vectors: new Float32Array([
      new DataView(bytes.buffer).getFloat32(60, true),
      new DataView(bytes.buffer).getFloat32(64, true),
    ]) }));
    const mean = computeCanonicalVectorMean(blocks, 1);
    expect(mean).toEqual([0.75]);
    expect(JSON.parse(JSON.stringify(mean))).toEqual(mean);
    expect(computeCanonicalVectorMean([...blocks].reverse(), 1)).toEqual([0.25]);
  });
});

function resealPointer(memory: ReturnType<typeof memoryStorage>, descriptor: GenerationDescriptor) {
  const current = decodeCurrentGenerationPointer(memory.text.get(CURRENT)!);
  memory.text.set(CURRENT, encodeCurrentGenerationPointer({ ...current, descriptorChecksum: blockObjectChecksum(new TextEncoder().encode(encodeGenerationDescriptor(descriptor))), checksum: `sha256:${"0".repeat(64)}` }));
}
