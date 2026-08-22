import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { createEvidenceChunkId, type EvidenceChunk } from "../src/library/fulltext/evidence-chunk";
import { tokenizeUnicode, tokenizeUnicodeWithHanSingles } from "../src/library/fulltext/bm25-retrieval";
import {
  GENERATION_DESCRIPTOR_FORMAT_VERSION,
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
  blockObjectChecksum,
  decodeEvidenceBlock,
  decodeLexicalDictionaryBlock,
  decodeLexicalPostingsBlock,
  decodeVectorBlock,
  encodeEvidenceBlock,
  encodeGenerationDescriptor,
  encodeLexicalDictionaryBlock,
  encodeLexicalPostingsBlock,
  encodePaperMetadataBlock,
  encodeVectorBlock,
  lexicalTermBucket,
  type GenerationDescriptor,
  type GenerationObjectReference,
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
import { sha256Hex } from "../src/utils/digest";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const OTHER = `sha256:${"c".repeat(64)}`;
const BASE = `arxiv-daily/.index/personal-library-search-index/${"a".repeat(64)}/${"b".repeat(64)}`;
const CURRENT = `${BASE}/current.json`;
const BACKUP = `${CURRENT}.backup`;
const PROMOTION_CLAIM = `${BASE}/.current-promotion-claim.json`;

const V2_VECTOR_HEX = "41444749010002000100000014000000010000001ad0a1687cfe2e882fa9b0a1bf16d579795fa2627cd1782cb8f8431f0fe536b60200000000000000000000000000c03f000000c0";
const V2_EVIDENCE_HEX = "4144474901000200020000006e01000001000000976008a44bb6a088e42207007a7cc2079905b3eff60a625c7694c4d7271b895e7b22726f775374617274223a302c227265636f726473223a5b7b227061706572496e646578223a302c2270617065724b6579223a2270617065723a61222c22766563746f72526f77223a302c226368756e6b223a7b226964223a227368613235363a31613337646138643535336130336237383065396539663230376233366335626165306661326634656333346637323363626365633539663661366464393837222c22696e646578223a302c2270616765223a312c2274657874223a226c65676163792064656e73652065766964656e6365222c2268656164696e6773223a5b224d6574686f6473225d2c226c6f6361746f72223a7b22706167655374617274223a317d2c2264657269766174696f6e223a7b22706172736572223a7b226964223a2266697874757265222c2276657273696f6e223a2231227d2c226368756e6b657256657273696f6e223a322c22656d62656464696e67496e70757456657273696f6e223a317d7d7d5d7d";
const V2_DESCRIPTOR = "{\"formatVersion\":1,\"schemaVersion\":2,\"generationId\":\"legacy-v2\",\"sourceRevision\":7,\"scopeFingerprint\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"identificationFingerprint\":\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\",\"modelId\":\"legacy-model\",\"dimension\":2,\"corpusMean\":[1.5,-2],\"corpusStats\":{\"indexedPaperCount\":1,\"chunkCount\":1},\"indexDerivation\":{\"builderVersion\":1,\"denseCenteringVersion\":1,\"tokenizerVersion\":1,\"postingsVersion\":1},\"objects\":[{\"kind\":\"vector\",\"path\":\"objects/vector-v2.bin\",\"byteLength\":72,\"recordStart\":0,\"recordCount\":1,\"checksum\":\"sha256:210554f1efccfebd1ea4ddfa40c680f24ab5a815d3e5736173de0ff55c46e2bf\"},{\"kind\":\"evidence\",\"path\":\"objects/evidence-v2.bin\",\"byteLength\":418,\"recordStart\":0,\"recordCount\":1,\"checksum\":\"sha256:d127617e404d4e1c38033b3cb75ac8c3a3dae7e2fc581c8e74b1c50fe324ab6a\"}]}";
const V2_POINTER = "{\"formatVersion\":1,\"schemaVersion\":1,\"generationId\":\"legacy-v2\",\"sourceRevision\":7,\"scopeFingerprint\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"identificationFingerprint\":\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\",\"descriptorChecksum\":\"sha256:71a8e7cc0593076dcb9c2da2d9a976dde2d5bc82eeeeefa3837197847ed82e2e\",\"checksum\":\"sha256:cbf5b5fec57318218d5242beaf8ec9cd23133f09bdb850884c09b161fbdc144c\"}";
function bytesFromHex(hex: string): Uint8Array { const bytes = new Uint8Array(hex.length / 2); for (let index = 0; index < bytes.length; index += 1) bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16); return bytes; }

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((settle) => { resolve = settle; });
  return { promise, resolve };
}

function pointerObservationChecksum(raw: string | null): string {
  if (raw === null) return `sha256:${sha256Hex(new Uint8Array([0]))}`;
  const encoded = new TextEncoder().encode(raw);
  const bytes = new Uint8Array(encoded.length + 1);
  bytes[0] = 1;
  bytes.set(encoded, 1);
  return `sha256:${sha256Hex(bytes)}`;
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
    corpusStats: { indexedPaperCount: 0, chunkCount: 0, totalLexicalTokenCount: 0, avgdl: 0, totalLexicalTokenCountWithHanSingles: 0, avgdlWithHanSingles: 0 },
    lexicalCapability: "none",
    lexicalRouting: Array.from({ length: 256 }, () => [] as string[]),
    indexDerivation: { builderVersion: 1, denseCenteringVersion: 1, tokenizerVersion: 1, postingsVersion: 1 },
    objects: [],
  };
  return { descriptor, objects: [] as GenerationObjectWrite[] };
}

function fixture(generationId: string, sourceRevision: number, values = [1, 2, 3, 4]) {
  const vector = encodeVectorBlock({ rowStart: 0, dimension: 2, paperOrdinals: new Uint32Array([0, 1]), vectors: new Float32Array(values) });
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
    corpusStats: { indexedPaperCount: 2, chunkCount: 2, totalLexicalTokenCount: 0, avgdl: 0, totalLexicalTokenCountWithHanSingles: 0, avgdlWithHanSingles: 0 },
    lexicalCapability: "none",
    lexicalRouting: Array.from({ length: 256 }, () => [] as string[]),
    indexDerivation: { builderVersion: 1, denseCenteringVersion: 1, tokenizerVersion: 1, postingsVersion: 1 },
    objects: [
      { kind: "vector", path: objects[0]!.path, byteLength: vector.byteLength, recordStart: 0, recordCount: 2, checksum: blockObjectChecksum(vector) },
      { kind: "evidence", path: objects[1]!.path, byteLength: evidence.byteLength, recordStart: 0, recordCount: 2, checksum: blockObjectChecksum(evidence) },
    ],
  };
  return { descriptor, objects };
}

function compareUtf8(left: string, right: string): number {
  const a = new TextEncoder().encode(left); const b = new TextEncoder().encode(right);
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) if (a[index] !== b[index]) return a[index]! - b[index]!;
  return a.length - b.length;
}
function compareNamespace(left: string, right: string): number { return ["alias", "base", "expanded"].indexOf(left) - ["alias", "base", "expanded"].indexOf(right); }
function maskHex(buckets: ReadonlySet<number>): string { const bytes = new Uint8Array(32); for (const bucket of buckets) bytes[bucket >>> 3]! |= 1 << (bucket & 7); return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(""); }
function lexicalFixture(generationId: string, sourceRevision: number, input: string | readonly string[] = "哈哈 alpha") {
  const texts = typeof input === "string" ? [input] : [...input];
  const chunks = texts.map((text, index) => chunk(index, text));
  const vector = encodeVectorBlock({ rowStart: 0, dimension: 2, paperOrdinals: new Uint32Array(texts.length), vectors: new Float32Array(texts.flatMap((_, index) => [index + 1, index + 2])) });
  const evidence = encodeEvidenceBlock({ rowStart: 0, records: chunks.map((entry, vectorRow) => ({ paperIndex: 0, paperKey: "paper:a", vectorRow, chunk: entry })) });
  const metadata = encodePaperMetadataBlock({ paperStart: 0, records: [{ paperOrdinal: 0, paperKey: "paper:a", chunkStart: 0, chunkCount: texts.length, title: "A" }] });
  const occurrences: Array<{ chunkOrdinal: number; namespace: "alias" | "base" | "expanded"; term: string; tf: number }> = [];
  const chunkRecords = texts.map((text, chunkOrdinal) => {
    const base = tokenizeUnicode(text); const expanded = tokenizeUnicodeWithHanSingles(text);
    const frequencies = (tokens: readonly string[]) => { const map = new Map<string, number>(); for (const token of tokens) map.set(token, (map.get(token) ?? 0) + 1); return map; };
    for (const [term, tf] of frequencies(base)) occurrences.push({ chunkOrdinal, namespace: "base", term, tf });
    for (const [term, tf] of frequencies(expanded)) occurrences.push({ chunkOrdinal, namespace: "expanded", term, tf });
    const compactText = text.normalize("NFKC").toLocaleLowerCase("und").replace(/[^\p{L}\p{N}]+/gu, ""); const chars = Array.from(compactText); const grams = new Set<string>();
    for (const size of [1, 2, 3]) for (let offset = 0; offset + size <= chars.length; offset += 1) grams.add(chars.slice(offset, offset + size).join(""));
    for (const term of grams) occurrences.push({ chunkOrdinal, namespace: "alias", term, tf: 1 });
    return { paperOrdinal: 0, chunkIndex: chunkOrdinal, baseLength: base.length, expandedLength: expanded.length, compactText };
  });
  occurrences.sort((a, b) => a.chunkOrdinal - b.chunkOrdinal || compareNamespace(a.namespace, b.namespace) || compareUtf8(a.term, b.term));
  const termCatalog = occurrences.map((_, index) => index).sort((a, b) => compareNamespace(occurrences[a]!.namespace, occurrences[b]!.namespace) || compareUtf8(occurrences[a]!.term, occurrences[b]!.term) || occurrences[a]!.chunkOrdinal - occurrences[b]!.chunkOrdinal);
  const postings = encodeLexicalPostingsBlock({ postingOrdinal: 0, chunkStart: 0, chunks: chunkRecords, occurrences, termCatalog });
  const aggregate = new Map<string, { postingOrdinal: number; namespace: "alias" | "base" | "expanded"; term: string; chunkDf: number; totalTf: number }>();
  for (const occurrence of occurrences) { const key = `${occurrence.namespace}\0${occurrence.term}`; const old = aggregate.get(key); aggregate.set(key, old ? { ...old, chunkDf: old.chunkDf + 1, totalTf: old.totalTf + occurrence.tf } : { postingOrdinal: 0, namespace: occurrence.namespace, term: occurrence.term, chunkDf: 1, totalTf: occurrence.tf }); }
  const entries = [...aggregate.values()].sort((a, b) => compareNamespace(a.namespace, b.namespace) || compareUtf8(a.term, b.term));
  const queryCatalog = entries.map((_, index) => index).sort((a, b) => lexicalTermBucket(entries[a]!.namespace, entries[a]!.term) - lexicalTermBucket(entries[b]!.namespace, entries[b]!.term) || compareNamespace(entries[a]!.namespace, entries[b]!.namespace) || compareUtf8(entries[a]!.term, entries[b]!.term));
  const buckets = new Set(entries.map((entry) => lexicalTermBucket(entry.namespace, entry.term)));
  const dictionary = encodeLexicalDictionaryBlock({ dictionaryOrdinal: 0, postingStart: 0, postingCount: 1, entries, queryCatalog, bucketMask: maskHex(buckets) });
  const postingPath = "objects/postings.bin", dictionaryPath = "objects/dictionary.bin";
  const routing = Array.from({ length: 256 }, () => [] as string[]); for (const bucket of buckets) routing[bucket] = [dictionaryPath];
  const objects: GenerationObjectWrite[] = [{ path: "objects/vector.bin", bytes: vector }, { path: "objects/evidence.bin", bytes: evidence }, { path: "objects/metadata.bin", bytes: metadata }, { path: postingPath, bytes: postings }, { path: dictionaryPath, bytes: dictionary }];
  const ref = (kind: GenerationObjectReference["kind"], path: string, bytes: Uint8Array, recordStart: number, recordCount: number): GenerationObjectReference => ({ kind, path, byteLength: bytes.length, recordStart, recordCount, checksum: blockObjectChecksum(bytes) });
  const total = chunkRecords.reduce((sum, row) => sum + row.baseLength, 0), expandedTotal = chunkRecords.reduce((sum, row) => sum + row.expandedLength, 0);
  const descriptor: GenerationDescriptor = { formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION, schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION, generationId, sourceRevision, scopeFingerprint: SCOPE, identificationFingerprint: IDENTIFICATION, modelId: "model-a", dimension: 2, corpusMean: [(texts.length + 1) / 2, (texts.length + 3) / 2], corpusStats: { indexedPaperCount: 1, chunkCount: texts.length, totalLexicalTokenCount: total, avgdl: total / texts.length, totalLexicalTokenCountWithHanSingles: expandedTotal, avgdlWithHanSingles: expandedTotal / texts.length }, lexicalCapability: "bm25-v1", lexicalRouting: routing, indexDerivation: { builderVersion: 1, denseCenteringVersion: 1, tokenizerVersion: 1, postingsVersion: 1 }, objects: [ref("vector", objects[0]!.path, vector, 0, texts.length), ref("evidence", objects[1]!.path, evidence, 0, texts.length), ref("paper-metadata", objects[2]!.path, metadata, 0, 1), ref("lexical-postings", postingPath, postings, 0, texts.length), ref("lexical-dictionary", dictionaryPath, dictionary, 0, 1)] };
  return { descriptor, objects, postings, dictionary, buckets };
}

function multiBlockLexicalFixture(generationId: string, sourceRevision: number) {
  const base = lexicalFixture(generationId, sourceRevision, ["哈哈 alpha", "beta", "gamma gamma", "delta", "epsilon", "zeta"]);
  const decodedVector = decodeVectorBlock(base.objects[0]!.bytes);
  const decodedEvidence = decodeEvidenceBlock(base.objects[1]!.bytes);
  const decodedPostings = decodeLexicalPostingsBlock(base.postings);
  const paperForChunk = (ordinal: number) => ordinal < 3 ? 0 : 1;
  const indexForChunk = (ordinal: number) => ordinal < 3 ? ordinal : ordinal - 3;
  const writes: GenerationObjectWrite[] = [];
  const refs: GenerationObjectReference[] = [];
  const add = (kind: GenerationObjectReference["kind"], path: string, bytes: Uint8Array, start: number, count: number) => {
    writes.push({ path, bytes });
    refs.push({ kind, path, byteLength: bytes.length, recordStart: start, recordCount: count, checksum: blockObjectChecksum(bytes) });
  };
  for (const [block, start] of [[2, 0], [2, 2], [2, 4]] as const) {
    const count = block;
    add("vector", `objects/vector-${start}.bin`, encodeVectorBlock({
      rowStart: start,
      dimension: 2,
      paperOrdinals: new Uint32Array(Array.from({ length: count }, (_, offset) => paperForChunk(start + offset))),
      vectors: decodedVector.vectors.slice(start * 2, (start + count) * 2),
    }), start, count);
  }
  for (const [count, start] of [[2, 0], [2, 2], [2, 4]] as const) {
    add("evidence", `objects/evidence-${start}.bin`, encodeEvidenceBlock({
      rowStart: start,
      records: decodedEvidence.records.slice(start, start + count).map((record, offset) => ({
        ...record,
        paperIndex: paperForChunk(start + offset),
        paperKey: paperForChunk(start + offset) === 0 ? "paper:a" : "paper:b",
        chunk: { ...record.chunk, index: indexForChunk(start + offset) },
      })),
    }), start, count);
  }
  add("paper-metadata", "objects/metadata-0.bin", encodePaperMetadataBlock({ paperStart: 0, records: [{ paperOrdinal: 0, paperKey: "paper:a", chunkStart: 0, chunkCount: 3, title: "A" }] }), 0, 1);
  add("paper-metadata", "objects/metadata-1.bin", encodePaperMetadataBlock({ paperStart: 1, records: [{ paperOrdinal: 1, paperKey: "paper:b", chunkStart: 3, chunkCount: 3, title: "B" }] }), 1, 1);
  const postingCuts = [[1, 0], [2, 1], [2, 3], [1, 5]] as const;
  const postingBlocks = postingCuts.map(([count, start], postingOrdinal) => {
    const occurrences = decodedPostings.occurrences.filter((entry) => entry.chunkOrdinal >= start && entry.chunkOrdinal < start + count);
    const termCatalog = occurrences.map((_, index) => index).sort((a, b) => compareNamespace(occurrences[a]!.namespace, occurrences[b]!.namespace) || compareUtf8(occurrences[a]!.term, occurrences[b]!.term) || occurrences[a]!.chunkOrdinal - occurrences[b]!.chunkOrdinal);
    const bytes = encodeLexicalPostingsBlock({
      postingOrdinal,
      chunkStart: start,
      chunks: decodedPostings.chunks.slice(start, start + count).map((chunkRecord, offset) => ({
        ...chunkRecord,
        paperOrdinal: paperForChunk(start + offset),
        chunkIndex: indexForChunk(start + offset),
      })),
      occurrences,
      termCatalog,
    });
    add("lexical-postings", `objects/postings-${postingOrdinal}.bin`, bytes, start, count);
    return decodeLexicalPostingsBlock(bytes);
  });
  const dictionaryPaths: string[] = [];
  const routing = Array.from({ length: 256 }, () => [] as string[]);
  for (const [postingCount, postingStart] of [[3, 0], [1, 3]] as const) {
    const entries = postingBlocks.slice(postingStart, postingStart + postingCount).flatMap((postings) => {
      const result: Array<{ postingOrdinal: number; namespace: "alias" | "base" | "expanded"; term: string; chunkDf: number; totalTf: number }> = [];
      let catalog = 0;
      while (catalog < postings.termCatalog.length) {
        const first = postings.occurrences[postings.termCatalog[catalog]!]!;
        let chunkDf = 0; let totalTf = 0;
        do { const occurrence = postings.occurrences[postings.termCatalog[catalog]!]!; chunkDf += 1; totalTf += occurrence.tf; catalog += 1; }
        while (catalog < postings.termCatalog.length && postings.occurrences[postings.termCatalog[catalog]!]!.namespace === first.namespace && postings.occurrences[postings.termCatalog[catalog]!]!.term === first.term);
        result.push({ postingOrdinal: postings.postingOrdinal, namespace: first.namespace, term: first.term, chunkDf, totalTf });
      }
      return result;
    });
    const queryCatalog = entries.map((_, index) => index).sort((a, b) => lexicalTermBucket(entries[a]!.namespace, entries[a]!.term) - lexicalTermBucket(entries[b]!.namespace, entries[b]!.term) || compareNamespace(entries[a]!.namespace, entries[b]!.namespace) || compareUtf8(entries[a]!.term, entries[b]!.term) || entries[a]!.postingOrdinal - entries[b]!.postingOrdinal);
    const buckets = new Set(entries.map((entry) => lexicalTermBucket(entry.namespace, entry.term)));
    const path = `objects/dictionary-${postingStart}.bin`;
    dictionaryPaths.push(path);
    add("lexical-dictionary", path, encodeLexicalDictionaryBlock({ dictionaryOrdinal: dictionaryPaths.length - 1, postingStart, postingCount, entries, queryCatalog, bucketMask: maskHex(buckets) }), postingStart, postingCount);
    for (const bucket of buckets) routing[bucket]!.push(path);
  }
  for (const route of routing) route.sort();
  const descriptor: GenerationDescriptor = {
    ...base.descriptor,
    corpusStats: { ...base.descriptor.corpusStats, indexedPaperCount: 2 },
    lexicalRouting: routing,
    objects: refs,
  };
  return { descriptor, objects: writes };
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
  let listHook: ((dir: string) => Promise<void>) | undefined;
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
    list: vi.fn(async (dir) => {
      if (listHook) await listHook(dir);
      if (!dirs.has(dir)) throw new Error(`missing ${dir}`);
      const prefix = `${dir}/`;
      const entries = new Map<string, "file" | "folder">();
      for (const path of [...text.keys(), ...binary.keys(), ...dirs]) {
        if (!path.startsWith(prefix)) continue;
        const remainder = path.slice(prefix.length);
        if (!remainder) continue;
        const child = remainder.split("/")[0]!;
        const childPath = `${dir}/${child}`;
        entries.set(childPath, remainder.includes("/") || dirs.has(childPath) ? "folder" : "file");
      }
      return [...entries].map(([path, type]) => ({ path, type }));
    }),
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
    setListHook(hook?: typeof listHook) { listHook = hook; },
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
  it("tracks concurrent legacy fallback writers as an active set", async () => {
    const backend: MemoryBackend = { text: new Map(), binary: new Map(), dirs: new Set() };
    const firstMemory = memoryStorage(true, backend);
    const secondMemory = memoryStorage(true, backend);
    const first = store(firstMemory.storage);
    const second = store(secondMemory.storage);
    const firstLease = await first.acquireLegacyMigrationLease(
      `writer-legacy-first-${"a".repeat(32)}`,
    );
    const secondLease = await second.acquireLegacyMigrationLease(
      `writer-legacy-second-${"b".repeat(32)}`,
    );

    const active = await firstMemory.storage.list!(first.paths.legacyMigrationLeasesDirectory);
    expect(active).toHaveLength(2);

    await secondLease.release();
    await expect(publish(store(memoryStorage(true, backend).storage), "gen-first-still-active", 1))
      .rejects.toMatchObject({ code: "concurrent" });

    await firstLease.release();
    await expect(publish(store(memoryStorage(true, backend).storage), "gen-after-all-legacy", 1))
      .resolves.toMatchObject({ descriptor: { generationId: "gen-after-all-legacy" } });
  });

  it("rejects sequential reuse of one legacy lease token", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const token = `writer-legacy-reused-${"a".repeat(32)}`;
    const lease = await index.acquireLegacyMigrationLease(token);

    await expect(store(memory.storage).acquireLegacyMigrationLease(token))
      .rejects.toMatchObject({ code: "concurrent" });

    await lease.release();
  });

  it("recovers a legacy lease write that commits before its response is lost", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    let responseLost = true;
    memory.setAtomicHook(async (path, value) => {
      memory.text.set(path, value);
      if (responseLost && path.startsWith(`${index.paths.legacyMigrationLeasesDirectory}/`)) {
        responseLost = false;
        throw new Error("lease write response lost");
      }
    });

    const lease = await index.acquireLegacyMigrationLease(
      `writer-legacy-committed-${"b".repeat(32)}`,
    );
    await expect(lease.assertOwned()).resolves.toBeUndefined();
    await expect(lease.release()).resolves.toBeUndefined();
    await expect(memory.storage.list!(index.paths.legacyMigrationLeasesDirectory)).resolves.toEqual([]);
  });

  it("rejects a legacy lease after generation cutover is established", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-before-legacy-admission", 1);

    await expect(index.acquireLegacyMigrationLease(
      `writer-legacy-after-cutover-${"c".repeat(32)}`,
    )).rejects.toMatchObject({ code: "capability-unsupported" });
  });

  it("arbitrates an active legacy fallback lease against first generation cutover", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const lease = await index.acquireLegacyMigrationLease(
      `writer-legacy-fallback-${"f".repeat(32)}`,
    );

    await expect(publish(index, "gen-blocked-by-legacy", 1))
      .rejects.toMatchObject({ code: "concurrent" });
    expect(memory.text.has(index.paths.cutoverMarkerPath)).toBe(false);
    expect(memory.text.has(CURRENT)).toBe(false);

    await lease.release();
    await expect(publish(index, "gen-after-legacy", 1)).resolves.toMatchObject({
      descriptor: { generationId: "gen-after-legacy" },
    });
  });

  it("rolls back a new cutover marker when a legacy lease appears after the first scan", async () => {
    const backend: MemoryBackend = { text: new Map(), binary: new Map(), dirs: new Set() };
    const cutoverMemory = memoryStorage(true, backend);
    const fallbackMemory = memoryStorage(true, backend);
    const cutover = store(cutoverMemory.storage);
    let fallbackLease: Awaited<ReturnType<typeof cutover.acquireLegacyMigrationLease>> | undefined;
    cutoverMemory.setExclusiveHook(async (path, value) => {
      if (path === cutover.paths.cutoverMarkerPath) {
        fallbackLease = await store(fallbackMemory.storage).acquireLegacyMigrationLease(
          `writer-legacy-racing-${"c".repeat(32)}`,
        );
      }
      if (backend.text.has(path) || backend.binary.has(path) || backend.dirs.has(path)) return false;
      backend.text.set(path, value);
      return true;
    });

    await expect(publish(cutover, "gen-raced-by-legacy", 1))
      .rejects.toMatchObject({ code: "concurrent" });
    expect(backend.text.has(cutover.paths.cutoverMarkerPath)).toBe(false);
    expect(backend.text.has(CURRENT)).toBe(false);

    await fallbackLease?.release();
  });

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
    let observedPrimaryChecksum: string | undefined;
    const firstRun = store(first.storage, {
      beforePointerPromotion: async () => { firstStaged.resolve(); await releaseFirstPromotion.promise; },
      afterPromotionClaimAcquired: async () => {
        const parsed = JSON.parse(backend.text.get(PROMOTION_CLAIM)!);
        observedPrimaryChecksum = parsed.observedPrimaryChecksum;
        expect(parsed).toMatchObject({
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
    expect(observedPrimaryChecksum).toBe(pointerObservationChecksum(backend.text.get(BACKUP)!));
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
    expect(memory.text.has(generationPath("gen-a", ".staging-claim.json"))).toBe(false);
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

  it("rolls back its first cutover marker after a definitely failed CURRENT write", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    let rollbackHeldPromotionClaim = false;
    memory.setAtomicHook(async (path, value) => {
      if (path === CURRENT) throw new Error("current write failed");
      memory.text.set(path, value);
    });
    memory.setRemoveHook(async (path) => {
      if (path !== index.paths.cutoverMarkerPath) return;
      rollbackHeldPromotionClaim = memory.text.has(PROMOTION_CLAIM);
      if (!rollbackHeldPromotionClaim) throw new Error("promotion claim released before marker rollback");
    });
    await expect(publish(index, "gen-first", 1)).rejects.toMatchObject({ code: "write-failed" });
    expect(rollbackHeldPromotionClaim).toBe(true);
    expect(memory.text.has(BACKUP)).toBe(false);
    memory.setAtomicHook();
    memory.setRemoveHook();
    await expect(store(memory.storage).openCurrent()).resolves.toBeNull();
  });

  it("keeps fallback closed and preserves the candidate when marker rollback is uncertain", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    memory.setAtomicHook(async (path, value) => {
      if (path === CURRENT) throw new Error("current write failed");
      memory.text.set(path, value);
    });
    memory.setRemoveHook(async (path) => {
      if (path === index.paths.cutoverMarkerPath) throw new Error("marker remove failed");
    });

    await expect(publish(index, "gen-marker-rollback-uncertain", 1))
      .rejects.toMatchObject({ code: "commit-uncertain" });
    expect(memory.text.has(index.paths.cutoverMarkerPath)).toBe(true);
    expect(memory.text.has(generationPath("gen-marker-rollback-uncertain", "descriptor.json"))).toBe(true);
    memory.setAtomicHook();
    memory.setRemoveHook();
    await expect(store(memory.storage).openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });

  it("does not roll back the cutover marker after promotion claim ownership changes", async () => {
    const memory = memoryStorage();
    const foreignClaim = JSON.stringify({ writerToken: `writer-foreign-${"e".repeat(32)}` });
    const index = store(memory.storage, {
      afterCutoverMarkerEstablished: () => { memory.text.set(PROMOTION_CLAIM, foreignClaim); },
    });

    await expect(publish(index, "gen-marker-foreign-claim", 1))
      .rejects.toMatchObject({ code: "commit-uncertain" });
    expect(memory.text.has(index.paths.cutoverMarkerPath)).toBe(true);
    expect(memory.text.has(generationPath("gen-marker-foreign-claim", "descriptor.json"))).toBe(true);
    expect(memory.text.get(PROMOTION_CLAIM)).toBe(foreignClaim);
    await expect(store(memory.storage).openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });

  it("fails closed for oversized and future cutover markers before legacy fallback", async () => {
    const oversized = memoryStorage();
    const oversizedStore = store(oversized.storage);
    oversized.text.set(oversizedStore.paths.cutoverMarkerPath, "x".repeat(16 * 1024 + 1));
    await expect(oversizedStore.openCurrent()).rejects.toMatchObject({
      code: "corrupt-or-unreadable",
      cause: { message: expect.stringMatching(/marker exceeds its byte limit/i) },
    });

    const future = memoryStorage();
    const futureStore = store(future.storage);
    future.text.set(futureStore.paths.cutoverMarkerPath, JSON.stringify({ formatVersion: 2, schemaVersion: 2 }));
    await expect(futureStore.openCurrent()).rejects.toMatchObject({ code: "incompatible" });
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

  it("rejects exact committed replay when the source revision advances during final validation", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const built = fixture("gen-replay-source-race", 1);
    await index.stageAndPromote({
      ...built,
      writerToken: `writer-replay-first-${"f".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 1,
    });

    let sourceRevision = 1;
    const replaying = store(memory.storage, {
      afterCutoverMarkerEstablished: () => { sourceRevision = 2; },
    });
    await expect(replaying.stageAndPromote({
      ...built,
      writerToken: `writer-replay-stale-${"f".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => sourceRevision,
    })).rejects.toMatchObject({ code: "stale-source", expectedRevision: 1, currentRevision: 2 });
  });

  it.each([
    {
      change: "is replaced",
      mutate: (memory: ReturnType<typeof memoryStorage>) => {
        memory.text.set(CURRENT, encodeCurrentGenerationPointer(pointerFor(
          fixture("gen-replay-current-replacement", 9).descriptor,
        )));
      },
    },
    {
      change: "is deleted",
      mutate: (memory: ReturnType<typeof memoryStorage>) => { memory.text.delete(CURRENT); },
    },
  ])("fails closed when CURRENT $change during exact replay", async ({ mutate }) => {
    const memory = memoryStorage();
    const built = fixture("gen-replay-current-race", 1);
    await publish(store(memory.storage), built.descriptor.generationId, built.descriptor.sourceRevision);
    const replaying = store(memory.storage, {
      afterCutoverMarkerEstablished: () => { mutate(memory); },
    });

    await expect(replaying.stageAndPromote({
      ...built,
      writerToken: `writer-replay-current-${"f".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 1,
    })).rejects.toMatchObject({ code: "concurrent" });
  });

  it.each([
    {
      change: "is replaced",
      mutate: (memory: ReturnType<typeof memoryStorage>, markerPath: string) => {
        memory.text.set(markerPath, JSON.stringify({ formatVersion: 2, schemaVersion: 2 }));
      },
    },
    {
      change: "is deleted",
      mutate: (memory: ReturnType<typeof memoryStorage>, markerPath: string) => {
        memory.text.delete(markerPath);
      },
    },
  ])("fails closed when the cutover marker $change during exact replay", async ({ mutate }) => {
    const memory = memoryStorage();
    const built = fixture("gen-replay-marker-race", 1);
    const initial = store(memory.storage);
    await publish(initial, built.descriptor.generationId, built.descriptor.sourceRevision);
    const replaying = store(memory.storage, {
      afterCutoverMarkerEstablished: () => { mutate(memory, initial.paths.cutoverMarkerPath); },
    });

    await expect(replaying.stageAndPromote({
      ...built,
      writerToken: `writer-replay-marker-${"f".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 1,
    })).rejects.toMatchObject({ code: "stale-claim" });
  });

  it("rechecks source and CURRENT after cutover marker I/O before promotion", async () => {
    const sourceMemory = memoryStorage();
    const sourceStore = store(sourceMemory.storage);
    await publish(sourceStore, "gen-marker-source-old", 1);
    const sourceOld = sourceMemory.text.get(CURRENT);
    await sourceMemory.storage.remove(sourceStore.paths.cutoverMarkerPath);
    let revision = 2;
    const sourceRacingStore = store(sourceMemory.storage, {
      afterCutoverMarkerEstablished: () => { revision = 3; },
    });
    await expect(sourceRacingStore.stageAndPromote({
      ...fixture("gen-marker-source-new", 2),
      writerToken: `writer-marker-source-${"f".repeat(32)}`,
      expectedCurrent: { generationId: "gen-marker-source-old", sourceRevision: 1 },
      sourceCurrentRevision: () => revision,
    })).rejects.toMatchObject({ code: "stale-source", expectedRevision: 2, currentRevision: 3 });
    expect(sourceMemory.text.get(CURRENT)).toBe(sourceOld);

    const currentMemory = memoryStorage();
    const currentStore = store(currentMemory.storage);
    await publish(currentStore, "gen-marker-current-old", 1);
    await currentMemory.storage.remove(currentStore.paths.cutoverMarkerPath);
    const changedCurrent = encodeCurrentGenerationPointer(pointerFor(
      fixture("gen-marker-current-concurrent", 9).descriptor,
    ));
    const currentRacingStore = store(currentMemory.storage, {
      afterCutoverMarkerEstablished: () => { currentMemory.text.set(CURRENT, changedCurrent); },
    });
    await expect(currentRacingStore.stageAndPromote({
      ...fixture("gen-marker-current-new", 2),
      writerToken: `writer-marker-current-${"f".repeat(32)}`,
      expectedCurrent: { generationId: "gen-marker-current-old", sourceRevision: 1 },
      sourceCurrentRevision: () => 2,
    })).rejects.toMatchObject({ code: "concurrent" });
    expect(currentMemory.text.get(CURRENT)).toBe(changedCurrent);
  });

  it("rechecks the exact cutover marker before committing CURRENT", async () => {
    const memory = memoryStorage();
    const initial = store(memory.storage);
    await publish(initial, "gen-marker-exact-old", 1);
    const old = memory.text.get(CURRENT);
    await memory.storage.remove(initial.paths.cutoverMarkerPath);
    const racing = store(memory.storage, {
      afterCutoverMarkerEstablished: async () => {
        await memory.storage.remove(initial.paths.cutoverMarkerPath);
      },
    });

    await expect(racing.stageAndPromote({
      ...fixture("gen-marker-exact-new", 2),
      writerToken: `writer-marker-exact-${"f".repeat(32)}`,
      expectedCurrent: { generationId: "gen-marker-exact-old", sourceRevision: 1 },
      sourceCurrentRevision: () => 2,
    })).rejects.toMatchObject({ code: "stale-claim" });
    expect(memory.text.get(CURRENT)).toBe(old);
  });

  it("rechecks the exact cutover marker after the final legacy lease scan", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await memory.storage.mkdir(index.paths.legacyMigrationLeasesDirectory);
    let leaseScans = 0;
    memory.setListHook(async (dir) => {
      if (dir !== index.paths.legacyMigrationLeasesDirectory || ++leaseScans !== 2) return;
      memory.text.delete(index.paths.cutoverMarkerPath);
    });

    await expect(publish(index, "gen-marker-final-scan", 1))
      .rejects.toMatchObject({ code: "stale-claim" });
    expect(memory.text.has(CURRENT)).toBe(false);
  });

  it("rechecks CURRENT after cutover marker I/O before recovery", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-marker-recovery-old", 1);
    await publish(index, "gen-marker-recovery-new", 2, {
      generationId: "gen-marker-recovery-old",
      sourceRevision: 1,
    });
    memory.text.set(CURRENT, "{original-corruption");
    await memory.storage.remove(index.paths.cutoverMarkerPath);
    const changedCurrent = "{concurrently-changed";
    const recoveryStore = store(memory.storage, {
      afterCutoverMarkerEstablished: () => { memory.text.set(CURRENT, changedCurrent); },
    });

    await expect(recoveryStore.openCurrent()).rejects.toMatchObject({ code: "concurrent" });
    expect(memory.text.get(CURRENT)).toBe(changedCurrent);
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

describe("accepted schema-v2 generation compatibility", () => {
  it("opens fixed P4b.3 bytes, reads dense objects, and validates closure without rewriting source schema", async () => {
    const memory = memoryStorage();
    memory.text.set(CURRENT, V2_POINTER);
    memory.text.set(generationPath("legacy-v2", "descriptor.json"), V2_DESCRIPTOR);
    memory.binary.set(generationPath("legacy-v2", "objects/vector-v2.bin"), bytesFromHex(V2_VECTOR_HEX));
    memory.binary.set(generationPath("legacy-v2", "objects/evidence-v2.bin"), bytesFromHex(V2_EVIDENCE_HEX));
    const opened = await store(memory.storage).openCurrent();
    expect(opened!.descriptor).toMatchObject({ schemaVersion: 2, lexicalCapability: "none", corpusStats: {
      indexedPaperCount: 1, chunkCount: 1, totalLexicalTokenCount: 0, avgdl: 0,
      totalLexicalTokenCountWithHanSingles: 0, avgdlWithHanSingles: 0,
    } });
    expect(opened!.descriptor.lexicalRouting).toEqual(Array.from({ length: 256 }, () => []));
    await expect(opened!.readObject(opened!.descriptor.objects[0]!)).resolves.toMatchObject({
      reference: { kind: "vector" }, block: { rowStart: 0, rowCount: 1, dimension: 2 },
    });
    await expect(opened!.readObject(opened!.descriptor.objects[1]!)).resolves.toMatchObject({
      reference: { kind: "evidence" }, block: { rowStart: 0, records: [{ paperKey: "paper:a" }] },
    });
    await expect(opened!.validateClosure()).resolves.toBeUndefined();
    expect(() => encodeGenerationDescriptor(opened!.descriptor)).toThrow(/only.*v4/i);
    expect(memory.text.get(generationPath("legacy-v2", "descriptor.json"))).toBe(V2_DESCRIPTOR);
  });

  it("rejects unaccepted schema v3 and future v5 descriptors", async () => {
    for (const schemaVersion of [3, 5]) {
      const memory = memoryStorage(); const raw = V2_DESCRIPTOR.replace('"schemaVersion":2', `"schemaVersion":${schemaVersion}`);
      const descriptorChecksum = blockObjectChecksum(new TextEncoder().encode(raw));
      const pointer = encodeCurrentGenerationPointer({ ...JSON.parse(V2_POINTER), descriptorChecksum, checksum: `sha256:${"0".repeat(64)}` });
      memory.text.set(CURRENT, pointer); memory.text.set(generationPath("legacy-v2", "descriptor.json"), raw);
      await expect(store(memory.storage).openCurrent()).rejects.toMatchObject({ code: "incompatible" });
    }
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

  it("validates and decodes linear lexical objects through typed reads", async () => {
    const memory = memoryStorage(); const built = lexicalFixture("gen-lexical-healthy", 1);
    const opened = await store(memory.storage).stageAndPromote({ ...built, writerToken: `writer-lexical-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1 });
    const postingRef = opened.descriptor.objects.find((ref): ref is GenerationObjectReference & { kind: "lexical-postings" } => ref.kind === "lexical-postings")!;
    const dictionaryRef = opened.descriptor.objects.find((ref): ref is GenerationObjectReference & { kind: "lexical-dictionary" } => ref.kind === "lexical-dictionary")!;
    await expect(opened.readLexicalPostings(postingRef)).resolves.toMatchObject({ block: { postingOrdinal: 0, chunkStart: 0 } });
    await expect(opened.readLexicalDictionary(dictionaryRef)).resolves.toMatchObject({ block: { dictionaryOrdinal: 0, postingStart: 0, postingCount: 1 } });
    await expect(opened.validateClosure()).resolves.toBeUndefined();
  });

  it("checks postings reference metadata while staging typed blocks", async () => {
    const memory = memoryStorage(); const built = lexicalFixture("gen-lexical-ref-mismatch", 1);
    const badPosting = encodeLexicalPostingsBlock({ ...decodeLexicalPostingsBlock(built.postings), postingOrdinal: 1 });
    const postingWrite = built.objects.find((object) => object.path === "objects/postings.bin")!;
    (postingWrite as { bytes: Uint8Array }).bytes = badPosting;
    const postingRef = built.descriptor.objects.find((ref) => ref.kind === "lexical-postings")!;
    Object.assign(postingRef, { byteLength: badPosting.length, checksum: blockObjectChecksum(badPosting) });
    await expect(store(memory.storage).stageAndPromote({ ...built, writerToken: `writer-lexical-ref-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1 }))
      .rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });

  it("rejects dictionary routing membership that is not exact", async () => {
    const memory = memoryStorage(); const built = lexicalFixture("gen-routing-mismatch", 1);
    const extraBucket = [...Array(256).keys()].find((bucket) => !built.buckets.has(bucket))!;
    (built.descriptor.lexicalRouting[extraBucket] as string[]).push("objects/dictionary.bin");
    await expect(store(memory.storage).stageAndPromote({ ...built, writerToken: `writer-routing-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1 }))
      .rejects.toMatchObject({ code: "corrupt-or-unreadable" });
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

  it("never repairs an unreadable CURRENT whose actual value is unknown", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    await publish(index, "gen-unreadable-backup", 1);
    await publish(index, "gen-unreadable-current", 2, {
      generationId: "gen-unreadable-backup",
      sourceRevision: 1,
    });
    const originalCurrent = memory.text.get(CURRENT);
    vi.mocked(memory.storage.writeTextAtomic!).mockClear();
    memory.setTextReadHook(async (path, value) => {
      if (path === CURRENT) throw new Error("CURRENT read unavailable");
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    });

    await expect(index.openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    expect(memory.text.get(CURRENT)).toBe(originalCurrent);
    expect(vi.mocked(memory.storage.writeTextAtomic!).mock.calls
      .some(([path]) => path === CURRENT)).toBe(false);
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

  it("rechecks the cutover marker when both pointers disappear in the recovery queue", async () => {
    const memory = memoryStorage();
    const initial = store(memory.storage);
    await publish(initial, "gen-recovery-pointer-loss", 1);
    await memory.storage.remove(CURRENT);
    memory.text.set(BACKUP, "{corrupt-before-recovery");
    const recovering = store(memory.storage, {
      beforeRecoveryQueue: async () => {
        await memory.storage.remove(CURRENT);
        await memory.storage.remove(BACKUP);
      },
    });

    await expect(recovering.openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
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

    const parse = vi.spyOn(JSON, "parse");
    try {
      const oversizedPrimary = memoryStorage();
      oversizedPrimary.text.set(CURRENT, JSON.stringify({ formatVersion: 2, schemaVersion: 2, padding: "x".repeat(17_000) }));
      await expect(store(oversizedPrimary.storage).openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
      expect(parse).not.toHaveBeenCalled();

      const oversizedBackup = memoryStorage();
      oversizedBackup.text.set(BACKUP, JSON.stringify({ formatVersion: 2, schemaVersion: 2, padding: "界".repeat(9_000) }));
      await expect(store(oversizedBackup.storage).openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
      expect(parse).not.toHaveBeenCalled();
    } finally {
      parse.mockRestore();
    }

    const incomplete = memoryStorage();
    await publish(store(incomplete.storage), "gen-a", 1);
    incomplete.text.set(CURRENT, "bad");
    incomplete.binary.delete(generationPath("gen-a", "objects/000.vector.bin"));
    await expect(store(incomplete.storage).openCurrent()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    expect(incomplete.text.get(CURRENT)).toBe("bad");
  });

  it("rejects a locally valid paired ordinal mismatch for two papers in one block", async () => {
    const memory = memoryStorage();
    const id = "gen-paired-ordinal-mismatch";
    await publish(store(memory.storage), id, 1);
    const vector = encodeVectorBlock({
      rowStart: 0,
      dimension: 2,
      paperOrdinals: new Uint32Array([0, 0]),
      vectors: new Float32Array([1, 2, 3, 4]),
    });
    expect(() => decodeVectorBlock(vector)).not.toThrow();
    const evidenceBytes = memory.binary.get(generationPath(id, "objects/000.evidence.bin"))!;
    expect(decodeEvidenceBlock(evidenceBytes).records.map((record) => record.paperIndex)).toEqual([0, 1]);
    memory.binary.set(generationPath(id, "objects/000.vector.bin"), vector);
    const descriptor = JSON.parse(memory.text.get(generationPath(id, "descriptor.json"))!);
    descriptor.objects[0].byteLength = vector.byteLength;
    descriptor.objects[0].checksum = blockObjectChecksum(vector);
    memory.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(descriptor));
    resealPointer(memory, descriptor);
    const opened = await store(memory.storage).openCurrent();
    await expect(opened!.validateClosure()).rejects.toMatchObject({
      code: "corrupt-or-unreadable",
      cause: expect.objectContaining({ message: expect.stringMatching(/ordinal.*paperIndex/i) }),
    });
  });

  it("rejects missing/tampered/wrong kind/count/dimension/mean/evidence order and identity", async () => {
    const mutations: Array<(memory: ReturnType<typeof memoryStorage>, id: string) => void> = [
      (m, id) => { m.binary.delete(generationPath(id, "objects/000.vector.bin")); },
      // Outer reference checksum mismatch.
      (m, id) => { m.binary.get(generationPath(id, "objects/000.vector.bin"))![60]! ^= 1; },
      // Outer reference checksum is correct, but the block's embedded checksum is invalid.
      (m, id) => { const bytes = m.binary.get(generationPath(id, "objects/000.vector.bin"))!.slice(); bytes[60]! ^= 1; m.binary.set(generationPath(id, "objects/000.vector.bin"), bytes); const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.objects[0].checksum = blockObjectChecksum(bytes); m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
      (m, id) => { const bytes = m.binary.get(generationPath(id, "objects/000.evidence.bin"))!.slice(); m.binary.set(generationPath(id, "objects/000.vector.bin"), bytes); const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.objects[0].byteLength = bytes.byteLength; d.objects[0].checksum = blockObjectChecksum(bytes); m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
      (m, id) => { const bytes = encodeVectorBlock({ rowStart: 0, dimension: 2, paperOrdinals: new Uint32Array([0]), vectors: new Float32Array([1, 2]) }); m.binary.set(generationPath(id, "objects/000.vector.bin"), bytes); const d = JSON.parse(m.text.get(generationPath(id, "descriptor.json"))!); d.objects[0].byteLength = bytes.byteLength; d.objects[0].checksum = blockObjectChecksum(bytes); m.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(d)); resealPointer(m, d); },
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

describe("schema-v4 lexical semantic closure", () => {
  async function openedLexical(id: string, texts: readonly string[] = ["哈哈 alpha", "beta beta"]) {
    const memory = memoryStorage();
    const built = lexicalFixture(id, 1, texts);
    const opened = await store(memory.storage).stageAndPromote({
      ...built,
      writerToken: `writer-${id}-${"f".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 1,
    });
    return { memory, built, opened };
  }

  function replaceGenerationObject(
    memory: ReturnType<typeof memoryStorage>,
    id: string,
    descriptor: GenerationDescriptor,
    kind: GenerationObjectReference["kind"],
    bytes: Uint8Array,
  ) {
    const reference = descriptor.objects.find((candidate) => candidate.kind === kind)!;
    memory.binary.set(generationPath(id, reference.path), bytes);
    Object.assign(reference, { byteLength: bytes.byteLength, checksum: blockObjectChecksum(bytes) });
    memory.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(descriptor));
    resealPointer(memory, descriptor);
  }

  it("rejects metadata identity/coverage and exact lexical statistics mismatches", async () => {
    for (const mutation of ["metadata", "total", "expanded-total"] as const) {
      const id = `gen-lexical-${mutation}`;
      const { memory, built } = await openedLexical(id);
      if (mutation === "metadata") {
        const bytes = encodePaperMetadataBlock({ paperStart: 0, records: [{ paperOrdinal: 0, paperKey: "paper:wrong", chunkStart: 0, chunkCount: 2 }] });
        replaceGenerationObject(memory, id, built.descriptor, "paper-metadata", bytes);
      } else {
        const stats = built.descriptor.corpusStats as { totalLexicalTokenCount: number; avgdl: number; totalLexicalTokenCountWithHanSingles: number; avgdlWithHanSingles: number };
        if (mutation === "total") { stats.totalLexicalTokenCount += 1; stats.avgdl = stats.totalLexicalTokenCount / 2; }
        else { stats.totalLexicalTokenCountWithHanSingles += 1; stats.avgdlWithHanSingles = stats.totalLexicalTokenCountWithHanSingles / 2; }
        memory.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(built.descriptor));
        resealPointer(memory, built.descriptor);
      }
      const opened = await store(memory.storage).openCurrent();
      await expect(opened!.validateClosure()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    }
  });

  it("rejects wrong chunk metadata and missing, extra, or wrong lexical occurrences", async () => {
    for (const mutation of ["compact", "base", "expanded", "identity", "missing", "extra", "tf"] as const) {
      const id = `gen-postings-${mutation}`;
      const { memory, built } = await openedLexical(id);
      const decoded = decodeLexicalPostingsBlock(built.postings);
      const chunks = decoded.chunks.map((entry) => ({ ...entry }));
      const occurrences = decoded.occurrences.map((entry) => ({ ...entry }));
      if (mutation === "compact") chunks[0]!.compactText += "x";
      if (mutation === "base") chunks[0]!.baseLength += 1;
      if (mutation === "expanded") chunks[0]!.expandedLength += 1;
      if (mutation === "identity") for (const chunkRecord of chunks) chunkRecord.paperOrdinal = 1;
      if (mutation === "missing") occurrences.splice(0, 1);
      if (mutation === "extra") occurrences.push({ chunkOrdinal: 1, namespace: "expanded", term: "zzz", tf: 1 });
      if (mutation === "tf") occurrences.find((entry) => entry.namespace === "base")!.tf += 1;
      occurrences.sort((a, b) => a.chunkOrdinal - b.chunkOrdinal || compareNamespace(a.namespace, b.namespace) || compareUtf8(a.term, b.term));
      const termCatalog = occurrences.map((_, index) => index).sort((a, b) => compareNamespace(occurrences[a]!.namespace, occurrences[b]!.namespace) || compareUtf8(occurrences[a]!.term, occurrences[b]!.term) || occurrences[a]!.chunkOrdinal - occurrences[b]!.chunkOrdinal);
      const bytes = encodeLexicalPostingsBlock({ ...decoded, chunks, occurrences, termCatalog });
      replaceGenerationObject(memory, id, built.descriptor, "lexical-postings", bytes);
      const opened = await store(memory.storage).openCurrent();
      await expect(opened!.validateClosure()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    }
  });

  it("rejects locally synchronized deletion of a complete term or one chunk posting", async () => {
    for (const deletion of ["term", "chunk"] as const) {
      const id = `gen-synchronized-${deletion}`;
      const { memory, built } = await openedLexical(id, ["alpha alpha", "alpha beta"]);
      const postings = decodeLexicalPostingsBlock(built.postings);
      const occurrences = postings.occurrences.filter((entry) => deletion === "term"
        ? !(entry.namespace === "base" && entry.term === "alpha")
        : !(entry.namespace === "base" && entry.term === "alpha" && entry.chunkOrdinal === 1));
      const termCatalog = occurrences.map((_, index) => index).sort((a, b) => compareNamespace(occurrences[a]!.namespace, occurrences[b]!.namespace) || compareUtf8(occurrences[a]!.term, occurrences[b]!.term) || occurrences[a]!.chunkOrdinal - occurrences[b]!.chunkOrdinal);
      const postingBytes = encodeLexicalPostingsBlock({ ...postings, occurrences, termCatalog });
      replaceGenerationObject(memory, id, built.descriptor, "lexical-postings", postingBytes);

      const dictionary = decodeLexicalDictionaryBlock(built.dictionary);
      const entries = dictionary.entries.flatMap((entry) => {
        if (entry.namespace !== "base" || entry.term !== "alpha") return [{ ...entry }];
        if (deletion === "term") return [];
        return [{ ...entry, chunkDf: entry.chunkDf - 1, totalTf: entry.totalTf - 1 }];
      });
      const queryCatalog = entries.map((_, index) => index).sort((a, b) => lexicalTermBucket(entries[a]!.namespace, entries[a]!.term) - lexicalTermBucket(entries[b]!.namespace, entries[b]!.term) || compareNamespace(entries[a]!.namespace, entries[b]!.namespace) || compareUtf8(entries[a]!.term, entries[b]!.term));
      const buckets = new Set(entries.map((entry) => lexicalTermBucket(entry.namespace, entry.term)));
      const dictionaryBytes = encodeLexicalDictionaryBlock({ ...dictionary, entries, queryCatalog, bucketMask: maskHex(buckets) });
      replaceGenerationObject(memory, id, built.descriptor, "lexical-dictionary", dictionaryBytes);
      const dictionaryPath = built.descriptor.objects.find((entry) => entry.kind === "lexical-dictionary")!.path;
      for (const route of built.descriptor.lexicalRouting as string[][]) route.splice(0);
      for (const bucket of buckets) (built.descriptor.lexicalRouting[bucket] as string[]).push(dictionaryPath);
      memory.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(built.descriptor));
      resealPointer(memory, built.descriptor);
      const opened = await store(memory.storage).openCurrent();
      await expect(opened!.validateClosure()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    }
  });

  it("rejects missing/extra dictionary routes and wrong df or totalTf", async () => {
    for (const mutation of ["missing", "extra", "df", "tf"] as const) {
      const id = `gen-dictionary-${mutation}`;
      const { memory, built } = await openedLexical(id);
      const decoded = decodeLexicalDictionaryBlock(built.dictionary);
      const entries = decoded.entries.map((entry) => ({ ...entry }));
      if (mutation === "missing") entries.splice(0, 1);
      if (mutation === "extra") entries.push({ postingOrdinal: 0, namespace: "expanded", term: "zzz", chunkDf: 1, totalTf: 1 });
      if (mutation === "df") { entries[0]!.chunkDf += 1; entries[0]!.totalTf += 1; }
      if (mutation === "tf") entries[0]!.totalTf += 1;
      entries.sort((a, b) => a.postingOrdinal - b.postingOrdinal || compareNamespace(a.namespace, b.namespace) || compareUtf8(a.term, b.term));
      const queryCatalog = entries.map((_, index) => index).sort((a, b) => lexicalTermBucket(entries[a]!.namespace, entries[a]!.term) - lexicalTermBucket(entries[b]!.namespace, entries[b]!.term) || compareNamespace(entries[a]!.namespace, entries[b]!.namespace) || compareUtf8(entries[a]!.term, entries[b]!.term));
      const buckets = new Set(entries.map((entry) => lexicalTermBucket(entry.namespace, entry.term)));
      const bytes = encodeLexicalDictionaryBlock({ ...decoded, entries, queryCatalog, bucketMask: maskHex(buckets) });
      replaceGenerationObject(memory, id, built.descriptor, "lexical-dictionary", bytes);
      const dictionaryPath = built.descriptor.objects.find((entry) => entry.kind === "lexical-dictionary")!.path;
      for (const route of built.descriptor.lexicalRouting as string[][]) route.splice(0);
      for (const bucket of buckets) (built.descriptor.lexicalRouting[bucket] as string[]).push(dictionaryPath);
      memory.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(built.descriptor));
      resealPointer(memory, built.descriptor);
      const opened = await store(memory.storage).openCurrent();
      await expect(opened!.validateClosure()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    }
  });

  it("typed-rejects an adjacent postings block whose content is shifted across a continuous ref boundary", async () => {
    const memory = memoryStorage();
    const id = "gen-shifted-postings-content";
    const built = multiBlockLexicalFixture(id, 1);
    const healthy = await store(memory.storage).stageAndPromote({
      ...built,
      writerToken: `writer-shifted-postings-${"f".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 1,
    });
    await expect(healthy.validateClosure()).resolves.toBeUndefined();

    const descriptor = built.descriptor;
    const postingRefs = descriptor.objects.filter(
      (reference): reference is GenerationObjectReference & { kind: "lexical-postings" } => reference.kind === "lexical-postings",
    );
    const sourceRef = postingRefs[1]!;
    const targetRef = postingRefs[2]!;
    expect([sourceRef.recordStart, sourceRef.recordCount, targetRef.recordStart, targetRef.recordCount]).toEqual([1, 2, 3, 2]);
    const source = decodeLexicalPostingsBlock(memory.binary.get(generationPath(id, sourceRef.path))!);
    const shift = targetRef.recordStart - sourceRef.recordStart;
    const shifted = encodeLexicalPostingsBlock({
      postingOrdinal: 2,
      chunkStart: targetRef.recordStart,
      chunks: source.chunks.map((chunkRecord) => ({ ...chunkRecord })),
      occurrences: source.occurrences.map((occurrence) => ({ ...occurrence, chunkOrdinal: occurrence.chunkOrdinal + shift })),
      termCatalog: [...source.termCatalog],
    });
    memory.binary.set(generationPath(id, targetRef.path), shifted);
    Object.assign(targetRef, { byteLength: shifted.byteLength, checksum: blockObjectChecksum(shifted) });
    memory.text.set(generationPath(id, "descriptor.json"), encodeGenerationDescriptor(descriptor));
    resealPointer(memory, descriptor);

    vi.mocked(memory.storage.readBinary!).mockClear();
    const opened = await store(memory.storage).openCurrent();
    expect(memory.storage.readBinary).not.toHaveBeenCalled();
    await expect(opened!.validateClosure()).rejects.toMatchObject({
      code: "corrupt-or-unreadable",
      cause: expect.objectContaining({ message: expect.stringMatching(/postings chunk identity/i) }),
    });
  });

  it("accepts misaligned evidence/postings/dictionary boundaries with linear exact reads", async () => {
    const memory = memoryStorage();
    const built = multiBlockLexicalFixture("gen-misaligned-lexical", 1);
    const opened = await store(memory.storage).stageAndPromote({
      ...built,
      writerToken: `writer-misaligned-lexical-${"f".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 1,
    });
    const before = opened.diagnostics.objectReads;
    await opened.validateClosure();
    expect(opened.diagnostics.objectReads - before).toBe(24); // 3V + 9E + 8P + 2M + 2D.
    expect(opened.diagnostics.maxLiveBlocks).toBe(2);
  });

  it("reads each lexical object only once per ordered pass and keeps at most two blocks live", async () => {
    const { memory } = await openedLexical("gen-linear-reads", ["哈哈 alpha", "beta beta"]);
    const opened = await store(memory.storage).openCurrent();
    const before = opened!.diagnostics.objectReads;
    await opened!.validateClosure();
    expect(opened!.diagnostics.objectReads - before).toBe(8); // V + 3E + 2P + M + D for one ref of each kind.
    expect(opened!.diagnostics.maxLiveBlocks).toBe(2);
  });

  it("accepts duplicate Han text and a tokenless metadata-only BM25 generation", async () => {
    const duplicate = await openedLexical("gen-duplicate-han", ["哈哈"]);
    await expect(duplicate.opened.validateClosure()).resolves.toBeUndefined();

    const memory = memoryStorage();
    const built = lexicalFixture("gen-tokenless", 1, ["---"]);
    const descriptor: GenerationDescriptor = {
      ...built.descriptor,
      corpusStats: { ...built.descriptor.corpusStats, totalLexicalTokenCount: 0, avgdl: 0, totalLexicalTokenCountWithHanSingles: 0, avgdlWithHanSingles: 0 },
      lexicalRouting: Array.from({ length: 256 }, () => []),
      objects: built.descriptor.objects.filter((reference) => reference.kind !== "lexical-postings" && reference.kind !== "lexical-dictionary"),
    };
    const objects = built.objects.filter((object) => object.path !== "objects/postings.bin" && object.path !== "objects/dictionary.bin");
    const opened = await store(memory.storage).stageAndPromote({ descriptor, objects, writerToken: `writer-tokenless-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1 });
    await expect(opened.validateClosure()).resolves.toBeUndefined();
  });
});

describe("canonical generation mean", () => {
  it("uses canonical ref/row float64 accumulation order and survives exact JSON roundtrip under cancellation", () => {
    const blocks = [
      encodeVectorBlock({ rowStart: 0, dimension: 1, paperOrdinals: new Uint32Array([0, 0]), vectors: new Float32Array([1e20, 1]) }),
      encodeVectorBlock({ rowStart: 2, dimension: 1, paperOrdinals: new Uint32Array([1, 1]), vectors: new Float32Array([-1e20, 3]) }),
    ].map((bytes) => {
      const decoded = decodeVectorBlock(bytes);
      return { dimension: decoded.dimension, vectors: decoded.vectors };
    });
    const mean = computeCanonicalVectorMean(blocks, 1);
    expect(mean).toEqual([0.75]);
    expect(JSON.parse(JSON.stringify(mean))).toEqual(mean);
    expect(computeCanonicalVectorMean([...blocks].reverse(), 1)).toEqual([0.25]);
  });
});

describe("host-authorized generation maintenance", () => {
  it("keeps local admission closed until an opened reader is explicitly closed", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const opened = await publish(index, "gen-maintenance-live", 1);

    const pending = index.beginHostAuthorizedMaintenance();
    let settled = false;
    void pending.then(() => { settled = true; });
    await Promise.resolve();
    expect(settled).toBe(false);
    await expect(index.openCurrent()).rejects.toMatchObject({ code: "concurrent" });

    await opened.close();
    const session = await pending;
    await expect(opened.close()).resolves.toBeUndefined();
    await expect(opened.validateClosure()).rejects.toMatchObject({ code: "invalid" });
    session.release();
    await expect(index.openCurrent()).resolves.toMatchObject({ descriptor: { generationId: "gen-maintenance-live" } });
  });

  it("repairs a committed promotion claim and removes only unreferenced claimless generations", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const first = await publish(index, "gen-maintenance-first", 1);
    await first.close();
    const second = await publish(index, "gen-maintenance-second", 2, {
      generationId: "gen-maintenance-first", sourceRevision: 1,
    });
    await second.close();
    const third = await publish(index, "gen-maintenance-third", 3, {
      generationId: "gen-maintenance-second", sourceRevision: 2,
    });
    await third.close();

    memory.text.set("legacy/papers/keep.json", "authoritative legacy data");
    memory.dirs.add("legacy/papers");
    const orphan = "gen-maintenance-claimed-orphan";
    memory.dirs.add(generationPath(orphan, "").replace(/\/$/, ""));
    memory.text.set(generationPath(orphan, ".staging-claim.json"), "foreign claim");

    const session = await index.beginHostAuthorizedMaintenance();
    const report = await session.run();
    session.release();

    expect(report.removedGenerationIds).toContain("gen-maintenance-first");
    expect(report.removedGenerationIds).not.toContain("gen-maintenance-second");
    expect(report.removedGenerationIds).not.toContain("gen-maintenance-third");
    expect(report.removedGenerationIds).not.toContain(orphan);
    await expect(memory.storage.exists(generationPath(orphan, "").replace(/\/$/, ""))).resolves.toBe(true);
    expect(memory.text.has("legacy/papers/keep.json")).toBe(true);
    expect(report.promotionClaim).toBe("absent");
  });

  it("repairs a promotion claim only after proving that its candidate is committed", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const first = await publish(index, "gen-claim-repair-first", 1);
    await first.close();
    memory.setRemoveHook(async (path) => {
      if (path === PROMOTION_CLAIM) throw new Error("promotion claim cleanup lost");
    });
    const second = await publish(index, "gen-claim-repair-second", 2, {
      generationId: "gen-claim-repair-first", sourceRevision: 1,
    });
    await second.close();
    memory.setRemoveHook();
    expect(memory.text.has(PROMOTION_CLAIM)).toBe(true);

    const session = await index.beginHostAuthorizedMaintenance();
    const report = await session.run();
    session.release();
    expect(report.promotionClaim).toBe("repaired-committed");
    expect(memory.text.has(PROMOTION_CLAIM)).toBe(false);
  });

  it("retains an unproven promotion claim and performs no age-based orphan cleanup", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const first = await publish(index, "gen-unproven-claim-first", 1);
    await first.close();
    const firstPointer = memory.text.get(CURRENT)!;
    memory.setRemoveHook(async (path) => {
      if (path === PROMOTION_CLAIM) throw new Error("promotion claim cleanup lost");
    });
    const second = await publish(index, "gen-unproven-claim-second", 2, {
      generationId: "gen-unproven-claim-first", sourceRevision: 1,
    });
    await second.close();
    memory.setRemoveHook();
    memory.text.set(CURRENT, firstPointer);

    const session = await index.beginHostAuthorizedMaintenance();
    const report = await session.run();
    session.release();

    expect(report.promotionClaim).toBe("retained-unproven");
    expect(report.removedGenerationIds).toEqual([]);
    expect(memory.text.has(PROMOTION_CLAIM)).toBe(true);
    await expect(memory.storage.exists(generationPath("gen-unproven-claim-second", "").replace(/\/$/, "")))
      .resolves.toBe(true);
  });

  it("fails closed instead of collecting recovery candidates when cutover lost both pointers", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const opened = await publish(index, "gen-missing-pointer-maintenance", 1);
    await opened.close();
    await memory.storage.remove(CURRENT);
    await memory.storage.remove(BACKUP);

    const session = await index.beginHostAuthorizedMaintenance();
    await expect(session.run()).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    session.release();
    await expect(memory.storage.exists(generationPath("gen-missing-pointer-maintenance", "").replace(/\/$/, "")))
      .resolves.toBe(true);
  });

  it("waits for a pinned old-generation reader before collecting that generation", async () => {
    const memory = memoryStorage();
    const index = store(memory.storage);
    const oldReader = await publish(index, "gen-active-old-reader", 1);
    const current = await publish(index, "gen-active-new-reader", 2, {
      generationId: "gen-active-old-reader", sourceRevision: 1,
    });
    await current.close();
    const newest = await publish(index, "gen-active-newest-reader", 3, {
      generationId: "gen-active-new-reader", sourceRevision: 2,
    });
    await newest.close();

    const pending = index.beginHostAuthorizedMaintenance();
    let settled = false;
    void pending.then(() => { settled = true; });
    await Promise.resolve();
    expect(settled).toBe(false);
    expect(memory.text.has(generationPath("gen-active-old-reader", "descriptor.json"))).toBe(true);
    await oldReader.close();
    const session = await pending;
    const report = await session.run();
    session.release();
    expect(report.removedGenerationIds).toContain("gen-active-old-reader");
  });
});

function resealPointer(memory: ReturnType<typeof memoryStorage>, descriptor: GenerationDescriptor) {
  const current = decodeCurrentGenerationPointer(memory.text.get(CURRENT)!);
  memory.text.set(CURRENT, encodeCurrentGenerationPointer({ ...current, descriptorChecksum: blockObjectChecksum(new TextEncoder().encode(encodeGenerationDescriptor(descriptor))), checksum: `sha256:${"0".repeat(64)}` }));
}
