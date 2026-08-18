import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { createEvidenceChunkId, type EvidenceChunk } from "../src/library/fulltext/evidence-chunk";
import { searchKnowledgeBaseBm25, tokenizeUnicode, tokenizeUnicodeWithHanSingles } from "../src/library/fulltext/bm25-retrieval";
import { searchGenerationBm25 } from "../src/library/fulltext/generation-bm25-index";
import { fusePaperRankingsRrf } from "../src/library/fulltext/hybrid-retrieval";
import { evaluateRetrieval, type RetrievalJudgment } from "../src/library/fulltext/retrieval-evaluation";
import { FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION, type FullTextPaperDocument } from "../src/library/fulltext/knowledge-base";
import {
  GENERATION_DESCRIPTOR_FORMAT_VERSION,
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
  blockObjectChecksum,
  decodeEvidenceBlock,
  decodeLexicalDictionaryBlock,
  decodeLexicalPostingsBlock,
  decodePaperMetadataBlock,
  decodeVectorBlock,
  encodeEvidenceBlock,
  encodeLexicalDictionaryBlock,
  encodeLexicalPostingsBlock,
  encodePaperMetadataBlock,
  encodeVectorBlock,
  lexicalTermBucket,
  type GenerationDescriptor,
  type GenerationObjectReference,
  type LexicalNamespace,
  type LexicalOccurrence,
} from "../src/library/fulltext/generation-index-format";
import { FullTextGenerationIndexStore, type GenerationObjectWrite, type OpenedFullTextGeneration } from "../src/library/fulltext/generation-index-store";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { searchGenerationDense } from "../src/library/fulltext/retrieval";

function chunk(index: number, text: string): EvidenceChunk {
  const identity = { text, headings: ["Section"], locator: { pageStart: index + 1 }, derivation: { parser: { id: "fixture", version: "1" }, chunkerVersion: 2, embeddingInputVersion: 1 } };
  return { id: createEvidenceChunkId(identity), index, page: index + 1, ...identity };
}

function compareUtf8(left: string, right: string): number {
  const a = new TextEncoder().encode(left); const b = new TextEncoder().encode(right);
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) if (a[index] !== b[index]) return a[index]! - b[index]!;
  return a.length - b.length;
}
const NS: Record<LexicalNamespace, number> = { alias: 0, base: 1, expanded: 2 };
const compact = (text: string) => text.normalize("NFKC").toLocaleLowerCase("und").replace(/[^\p{L}\p{N}]+/gu, "");

function generationFixture(papers: readonly FullTextPaperDocument[], postingSize = 2, denseBlockSize = Number.POSITIVE_INFINITY): { opened: OpenedFullTextGeneration; reads: string[]; descriptor: GenerationDescriptor; objects: GenerationObjectWrite[] } {
  const rows = papers.flatMap((paper, paperOrdinal) => paper.chunks.map((entry) => ({ paper, paperOrdinal, chunk: entry })));
  const writes = new Map<string, Uint8Array>(); const refs: GenerationObjectReference[] = []; const reads: string[] = [];
  const add = (kind: GenerationObjectReference["kind"], path: string, bytes: Uint8Array, start: number, count: number) => {
    writes.set(path, bytes); refs.push({ kind, path, byteLength: bytes.byteLength, recordStart: start, recordCount: count, checksum: blockObjectChecksum(bytes) });
  };
  const dimension = papers[0]?.dimension ?? 1;
  const vectorValues = rows.flatMap((row) => Array.from(row.paper.vectors.subarray(row.chunk.index * dimension, (row.chunk.index + 1) * dimension)));
  const actualDenseBlockSize = Number.isFinite(denseBlockSize) ? denseBlockSize : rows.length;
  for (let rowStart = 0, block = 0; rowStart < rows.length; rowStart += actualDenseBlockSize, block += 1) {
    const blockRows = rows.slice(rowStart, rowStart + actualDenseBlockSize); const suffix = String(block).padStart(3, "0");
    add("vector", `objects/vector-${suffix}.bin`, encodeVectorBlock({ rowStart, dimension, paperOrdinals: new Uint32Array(blockRows.map((row) => row.paperOrdinal)), vectors: new Float32Array(vectorValues.slice(rowStart * dimension, (rowStart + blockRows.length) * dimension)) }), rowStart, blockRows.length);
  }
  for (let rowStart = 0, block = 0; rowStart < rows.length; rowStart += actualDenseBlockSize, block += 1) {
    const blockRows = rows.slice(rowStart, rowStart + actualDenseBlockSize); const suffix = String(block).padStart(3, "0");
    add("evidence", `objects/evidence-${suffix}.bin`, encodeEvidenceBlock({ rowStart, records: blockRows.map((row, offset) => ({ paperIndex: row.paperOrdinal, paperKey: row.paper.paperKey, vectorRow: rowStart + offset, chunk: row.chunk as EvidenceChunk })) }), rowStart, blockRows.length);
  }
  add("paper-metadata", "objects/metadata.bin", encodePaperMetadataBlock({ paperStart: 0, records: papers.map((paper, paperOrdinal) => ({ paperOrdinal, paperKey: paper.paperKey, chunkStart: rows.findIndex((row) => row.paperOrdinal === paperOrdinal), chunkCount: paper.chunks.length, title: paper.title })) }), 0, papers.length);
  const postingBlocks: ReturnType<typeof decodeLexicalPostingsBlock>[] = [];
  for (let start = 0, postingOrdinal = 0; start < rows.length; start += postingSize, postingOrdinal += 1) {
    const blockRows = rows.slice(start, start + postingSize); const occurrences: LexicalOccurrence[] = [];
    const chunks = blockRows.map((row, offset) => {
      const chunkOrdinal = start + offset; const base = tokenizeUnicode(row.chunk.text); const expanded = tokenizeUnicodeWithHanSingles(row.chunk.text);
      for (const [namespace, tokens] of [["base", base], ["expanded", expanded]] as const) {
        const frequencies = new Map<string, number>(); for (const term of tokens) frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
        for (const [term, tf] of frequencies) occurrences.push({ chunkOrdinal, namespace, term, tf });
      }
      const compactText = compact(row.chunk.text); const chars = Array.from(compactText); const grams = new Set<string>();
      for (const size of [1, 2, 3]) for (let offset = 0; offset + size <= chars.length; offset += 1) grams.add(chars.slice(offset, offset + size).join(""));
      for (const term of grams) occurrences.push({ chunkOrdinal, namespace: "alias", term, tf: 1 });
      return { paperOrdinal: row.paperOrdinal, chunkIndex: row.chunk.index, baseLength: base.length, expandedLength: expanded.length, compactText };
    });
    occurrences.sort((a, b) => a.chunkOrdinal - b.chunkOrdinal || NS[a.namespace] - NS[b.namespace] || compareUtf8(a.term, b.term));
    const termCatalog = occurrences.map((_, index) => index).sort((a, b) => NS[occurrences[a]!.namespace] - NS[occurrences[b]!.namespace] || compareUtf8(occurrences[a]!.term, occurrences[b]!.term) || occurrences[a]!.chunkOrdinal - occurrences[b]!.chunkOrdinal);
    const bytes = encodeLexicalPostingsBlock({ postingOrdinal, chunkStart: start, chunks, occurrences, termCatalog });
    add("lexical-postings", `objects/postings-${postingOrdinal}.bin`, bytes, start, blockRows.length); postingBlocks.push(decodeLexicalPostingsBlock(bytes));
  }
  const entries = postingBlocks.flatMap((block) => {
    const aggregate = new Map<string, { postingOrdinal: number; namespace: LexicalNamespace; term: string; chunkDf: number; totalTf: number }>();
    for (const occurrence of block.occurrences) { const key = `${occurrence.namespace}\0${occurrence.term}`; const old = aggregate.get(key); aggregate.set(key, old ? { ...old, chunkDf: old.chunkDf + 1, totalTf: old.totalTf + occurrence.tf } : { postingOrdinal: block.postingOrdinal, namespace: occurrence.namespace, term: occurrence.term, chunkDf: 1, totalTf: occurrence.tf }); }
    return [...aggregate.values()].sort((a, b) => NS[a.namespace] - NS[b.namespace] || compareUtf8(a.term, b.term));
  });
  const queryCatalog = entries.map((_, index) => index).sort((a, b) => lexicalTermBucket(entries[a]!.namespace, entries[a]!.term) - lexicalTermBucket(entries[b]!.namespace, entries[b]!.term) || NS[entries[a]!.namespace] - NS[entries[b]!.namespace] || compareUtf8(entries[a]!.term, entries[b]!.term) || entries[a]!.postingOrdinal - entries[b]!.postingOrdinal);
  const buckets = new Set(entries.map((entry) => lexicalTermBucket(entry.namespace, entry.term))); const mask = new Uint8Array(32); for (const bucket of buckets) mask[bucket >>> 3]! |= 1 << (bucket & 7);
  const dictionaryPath = "objects/dictionary.bin"; const dictionary = encodeLexicalDictionaryBlock({ dictionaryOrdinal: 0, postingStart: 0, postingCount: postingBlocks.length, entries, queryCatalog, bucketMask: Array.from(mask, (byte) => byte.toString(16).padStart(2, "0")).join("") });
  add("lexical-dictionary", dictionaryPath, dictionary, 0, postingBlocks.length);
  const routing = Array.from({ length: 256 }, () => [] as string[]); for (const bucket of buckets) routing[bucket] = [dictionaryPath];
  const total = rows.reduce((sum, row) => sum + tokenizeUnicode(row.chunk.text).length, 0); const expandedTotal = rows.reduce((sum, row) => sum + tokenizeUnicodeWithHanSingles(row.chunk.text).length, 0);
  const sums = new Float64Array(dimension); for (let offset = 0; offset < vectorValues.length; offset += dimension) for (let column = 0; column < dimension; column += 1) sums[column]! += vectorValues[offset + column]!;
  const descriptor: GenerationDescriptor = { formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION, schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION, generationId: "bm25-fixture", sourceRevision: 1, scopeFingerprint: `sha256:${"a".repeat(64)}`, identificationFingerprint: `sha256:${"b".repeat(64)}`, modelId: "fixture", dimension, corpusMean: Array.from(sums, (sum) => sum / rows.length), corpusStats: { indexedPaperCount: papers.length, chunkCount: rows.length, totalLexicalTokenCount: total, avgdl: total / rows.length, totalLexicalTokenCountWithHanSingles: expandedTotal, avgdlWithHanSingles: expandedTotal / rows.length }, lexicalCapability: "bm25-v1", lexicalRouting: routing, indexDerivation: { builderVersion: 1, denseCenteringVersion: 1, tokenizerVersion: 1, postingsVersion: 1 }, objects: refs };
  const read = async (reference: GenerationObjectReference) => { reads.push(reference.path); const bytes = writes.get(reference.path)!; const block = reference.kind === "vector" ? decodeVectorBlock(bytes) : reference.kind === "evidence" ? decodeEvidenceBlock(bytes) : reference.kind === "paper-metadata" ? decodePaperMetadataBlock(bytes) : reference.kind === "lexical-dictionary" ? decodeLexicalDictionaryBlock(bytes) : decodeLexicalPostingsBlock(bytes); return { reference, block }; };
  return {
    opened: { descriptor, diagnostics: { maxObjectBytes: 0, objectReads: 0, maxLiveBlocks: 0 }, readObject: read, readPaperMetadata: read, readLexicalDictionary: read, readLexicalPostings: read, iterateVectorBlocks: async function* () { for (const reference of refs.filter((entry) => entry.kind === "vector")) yield await read(reference) as any; } } as unknown as OpenedFullTextGeneration,
    reads,
    descriptor,
    objects: refs.map((reference) => ({ path: reference.path, bytes: writes.get(reference.path)! })),
  };
}

function paper(paperKey: string, title: string, texts: readonly string[]): FullTextPaperDocument {
  return { schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION, paperKey, title, modelId: "fixture", dimension: 1, textHash: `sha256:${"c".repeat(64)}`, filePaths: [`${paperKey}.pdf`], observationFingerprints: [`sha256:${"d".repeat(64)}`], chunks: texts.map((text, index) => chunk(index, text)), vectors: new Float32Array(texts.length), updatedAt: "2026-08-17T00:00:00.000Z" };
}

function memoryStorage() {
  const text = new Map<string, string>(); const binary = new Map<string, Uint8Array>(); const dirs = new Set<string>();
  const normalize = (path: string) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
  const storage: StorageAdapter = {
    normalizePath: normalize,
    readText: vi.fn(async (path) => { const value = text.get(path); if (value === undefined) throw new Error(`missing ${path}`); return value; }),
    writeText: vi.fn(async (path, value) => { text.set(path, value); }),
    writeTextAtomic: vi.fn(async (path, value) => { text.set(path, value); }),
    createTextExclusive: vi.fn(async (path, value) => { if (text.has(path) || binary.has(path) || dirs.has(path)) return false; text.set(path, value); return true; }),
    exists: vi.fn(async (path) => text.has(path) || binary.has(path) || dirs.has(path)),
    mkdir: vi.fn(async (path) => { dirs.add(path); }),
    remove: vi.fn(async (path) => { const prefix = `${path}/`; for (const key of [...text.keys()]) if (key === path || key.startsWith(prefix)) text.delete(key); for (const key of [...binary.keys()]) if (key === path || key.startsWith(prefix)) binary.delete(key); }),
    rename: vi.fn(async () => undefined),
    list: vi.fn(async (dir) => {
      const prefix = `${dir}/`; const entries = new Map<string, "file" | "folder">();
      for (const path of [...text.keys(), ...binary.keys(), ...dirs]) {
        if (!path.startsWith(prefix)) continue;
        const suffix = path.slice(prefix.length); if (!suffix) continue;
        const child = suffix.split("/")[0]!; const childPath = `${dir}/${child}`;
        entries.set(childPath, suffix.includes("/") || dirs.has(childPath) ? "folder" : "file");
      }
      return [...entries].map(([path, type]) => ({ path, type }));
    }),
    writeBinary: vi.fn(async (path, value) => { binary.set(path, new Uint8Array(value).slice()); }),
    readBinary: vi.fn(async (path) => { const value = binary.get(path); if (!value) throw new Error(`missing ${path}`); return value.slice().buffer; }),
  };
  return { storage, binary };
}

function stats() {
  return { dictionaryReads: 0, postingsReads: 0, metadataReads: 0, evidenceReads: 0, peakChunkAccumulators: 0, peakCandidates: 0, peakHits: 0, totalRetainedHits: 0, maxLiveBlocks: 0 };
}

describe("generation BM25 reader", () => {
  it("matches the P3 oracle exactly across posting boundaries and title-only papers", async () => {
    const papers = [paper("paper:a", "Alpha methods", ["alpha noise", "late alpha alpha"]), paper("paper:b", "Exact title only", ["unrelated"]), paper("paper:c", "Other", ["alpha"] )];
    const fixture = generationFixture(papers, 1); const titles = new Map(papers.map((entry) => [entry.paperKey, entry.title!]));
    for (const queryText of ["alpha", "Exact title only", "alpha alpha", "ＡＬＰＨＡ"]) {
      const actual = await searchGenerationBm25({ generation: fixture.opened, queryText, limit: 10, maxHitsPerPaper: 3 });
      const expected = searchKnowledgeBaseBm25({ papers, titles, queryText, limit: 10, maxHitsPerPaper: 3 });
      expect(actual).toEqual(expected);
      actual.forEach((match, paperIndex) => { expect(Object.is(match.score, expected[paperIndex]!.score)).toBe(true); match.hits.forEach((hit, hitIndex) => expect(Object.is(hit.score, expected[paperIndex]!.hits[hitIndex]!.score)).toBe(true)); });
    }
  });

  it("preserves mixed Han, repeated Han, compact alias, term order, and deterministic ties", async () => {
    const papers = [
      paper("paper:a", "Prefix title extended", ["中文证据 哈哈哈 pan starrs"]),
      paper("paper:m", "Alias", ["nothing"]),
      paper("paper:z", "证 中文", ["证据 中文检索 哈哈 Pan-STARRS"]),
    ];
    const fixture = generationFixture(papers, 1); const titles = new Map(papers.map((entry) => [entry.paperKey, entry.title!]));
    for (const queryText of ["证 中文", "哈哈", "哈哈 哈哈", "panstarrs", "Prefix title", "中文 证", "证 中文"]) {
      const actual = await searchGenerationBm25({ generation: fixture.opened, queryText, limit: 3, maxHitsPerPaper: 2, k1: 1.7, b: 0.4 });
      const expected = searchKnowledgeBaseBm25({ papers, titles, queryText, limit: 3, maxHitsPerPaper: 2, k1: 1.7, b: 0.4 });
      expect(actual).toEqual(expected);
      actual.forEach((match, index) => expect(Object.is(match.score, expected[index]!.score)).toBe(true));
    }
  });

  it("routes unknown terms without postings reads, enforces caps, and reports real bounded stats", async () => {
    const papers = Array.from({ length: 120 }, (_, index) => paper(`paper:${String(index).padStart(3, "0")}`, `Title ${index}`, [index % 2 === 0 ? "hot alpha" : "cold beta"]));
    const fixture = generationFixture(papers, 3);
    const occupied = new Set(fixture.opened.descriptor.lexicalRouting.flatMap((paths, bucket) => paths.length > 0 ? [bucket] : []));
    let unknown = "unknown"; while (occupied.has(lexicalTermBucket("base", unknown)) || occupied.has(lexicalTermBucket("alias", Array.from(unknown).sort().slice(0, 3).join("")))) unknown += "x";
    const emptyStats = { dictionaryReads: 0, postingsReads: 0, metadataReads: 0, evidenceReads: 0, peakChunkAccumulators: 0, peakCandidates: 0, peakHits: 0, totalRetainedHits: 0, maxLiveBlocks: 0 };
    expect(await searchGenerationBm25({ generation: fixture.opened, queryText: unknown, stats: emptyStats })).toEqual([]);
    expect(emptyStats.dictionaryReads).toBe(0); expect(emptyStats.postingsReads).toBe(0); expect(emptyStats.metadataReads).toBe(1);
    const alphaBucket = lexicalTermBucket("base", "alpha"); let collision = "collision0";
    for (let index = 1; lexicalTermBucket("base", collision) !== alphaBucket; index += 1) collision = `collision${index}`;
    const collisionStats = { ...emptyStats };
    expect(await searchGenerationBm25({ generation: fixture.opened, queryText: `${collision}${"!".repeat(81)}`, stats: collisionStats })).toEqual([]);
    expect(collisionStats.dictionaryReads).toBe(1); expect(collisionStats.postingsReads).toBe(0);
    expect(await searchGenerationBm25({ generation: fixture.opened, queryText: "alphabeta" })).toEqual([]);
    const stats = { ...emptyStats };
    const matches = await searchGenerationBm25({ generation: fixture.opened, queryText: "alpha", limit: 7, maxHitsPerPaper: 2, stats });
    expect(matches).toHaveLength(7); expect(stats.peakCandidates).toBeLessThanOrEqual(7); expect(stats.totalRetainedHits).toBeLessThanOrEqual(14); expect(stats.maxLiveBlocks).toBeLessThanOrEqual(2); expect(stats.peakChunkAccumulators).toBeLessThanOrEqual(3);
    await expect(searchGenerationBm25({ generation: fixture.opened, queryText: "x", limit: 1001 })).rejects.toBeInstanceOf(TypeError);
    await expect(searchGenerationBm25({ generation: fixture.opened, queryText: "x", maxHitsPerPaper: 101 })).rejects.toBeInstanceOf(TypeError);
    await expect(searchGenerationBm25({ generation: fixture.opened, queryText: "x".repeat(65_537) })).rejects.toBeInstanceOf(TypeError);
    await expect(searchGenerationBm25({ generation: fixture.opened, queryText: Array.from({ length: 65 }, (_, index) => `term${index}`).join(" ") })).rejects.toBeInstanceOf(TypeError);
  });

  it("tracks retained top hits plus a full current paper even when that paper is ultimately evicted", async () => {
    const fixture = generationFixture([
      paper("paper:a", "A", ["alpha alpha alpha", "alpha alpha alpha"]),
      paper("paper:b", "B", ["alpha alpha", "alpha alpha"]),
      paper("paper:c", "C", ["alpha", "alpha"]),
    ], 1);
    const observed = stats();
    const matches = await searchGenerationBm25({ generation: fixture.opened, queryText: "alpha", limit: 2, maxHitsPerPaper: 2, stats: observed });
    expect(matches.map((entry) => entry.paperKey)).toEqual(["paper:a", "paper:b"]);
    expect(observed.totalRetainedHits).toBe(4);
    expect(observed.peakHits).toBe(6);
    expect(observed.peakHits).toBeLessThanOrEqual(2 * 2 + 2);
  });

  it("keeps fixed BM25 and production RRF rankings and metrics equal to P3", async () => {
    const axis = (index: number) => { const value = new Float32Array(6); value[index] = 1; return value; };
    const corpus = [
      ["attention", "Graph Attention Networks", "masked self-attention over graph neighborhoods", 0],
      ["chinese", "中文科研文献检索", "中文检索与证据定位方法", 2],
      ["hard-negative", "Contamination Keywords", "robust estimation under contamination mentioned only as background", 0],
      ["panstarrs", "The Pan-STARRS1 Surveys", "Pan-STARRS photometric survey data products", 1],
      ["robust", "Robust Estimation", "bounded influence estimation under adversarial contamination", 5],
      ["semantic", "Invariant Representation Alignment", "latent representation alignment across domains", 3],
      ["sky", "Deep Sky Calibration", "wide field photometric calibration for deep galaxy surveys", 4],
      ["survey-negative", "Survey Instrument Status", "survey calibration hardware status report", 1],
    ].map(([key, title, text, vector]) => ({ ...paper(String(key), String(title), [String(text)]), dimension: 6, vectors: axis(Number(vector)) }));
    const queries = [
      ["exact", "exact-title", "Graph Attention Networks", axis(1)],
      ["alias", "compact-alias", "panstarrs", axis(0)],
      ["cjk", "cjk-keyword", "中文检索", axis(1)],
      ["semantic", "semantic-rewrite", "meaning-preserving domain features", axis(3)],
      ["long", "title-abstract", "Deep Sky Calibration", axis(4)],
      ["hard", "hard-negative", "reliable statistics with outliers", axis(5)],
    ] as const;
    const judgments: RetrievalJudgment[] = [
      { queryId: "exact", category: "exact-title", grades: { attention: 3 } }, { queryId: "alias", category: "compact-alias", grades: { panstarrs: 3 } },
      { queryId: "cjk", category: "cjk-keyword", grades: { chinese: 3 } }, { queryId: "semantic", category: "semantic-rewrite", grades: { semantic: 3 } },
      { queryId: "long", category: "title-abstract", grades: { sky: 3 } }, { queryId: "hard", category: "hard-negative", grades: { robust: 3, "hard-negative": 0 } },
    ];
    const fixture = generationFixture(corpus); const titles = new Map(corpus.map((entry) => [entry.paperKey, entry.title!]));
    const rankings: Record<string, Record<string, string[]>> = { dense: {}, bm25: {}, hybrid: {} };
    for (const [id, , queryText, queryVector] of queries) {
      const dense = await searchGenerationDense({ generation: fixture.opened, queryVector, centerCorpus: false, limit: corpus.length });
      const bm25 = await searchGenerationBm25({ generation: fixture.opened, queryText, limit: corpus.length });
      const legacy = searchKnowledgeBaseBm25({ papers: corpus, titles, queryText, limit: corpus.length });
      expect(bm25).toEqual(legacy);
      const hybrid = fusePaperRankingsRrf({ rankings: [dense, bm25], candidateLimit: corpus.length, limit: corpus.length });
      const legacyHybrid = fusePaperRankingsRrf({ rankings: [dense, legacy], candidateLimit: corpus.length, limit: corpus.length });
      expect(hybrid).toEqual(legacyHybrid);
      rankings.dense[id] = dense.map((entry) => entry.paperKey); rankings.bm25[id] = bm25.map((entry) => entry.paperKey); rankings.hybrid[id] = hybrid.map((entry) => entry.paperKey);
    }
    const report = evaluateRetrieval({ judgments, rankings, k: 5 });
    expect(report.modes.bm25.overall).toEqual({ recall: 2 / 3, mrr: 2 / 3, ndcg: 2 / 3 });
    expect(report.modes.hybrid.overall).toEqual({ recall: 1, mrr: 1, ndcg: 1 });
  });

  it("runs stageAndPromote, lexical closure, openCurrent, and BM25 through the real store handle", async () => {
    const memory = memoryStorage();
    const built = generationFixture([paper("paper:a", "Alpha", ["alpha"]), paper("paper:b", "Title only", ["beta"])], 1, 1);
    built.descriptor.generationId = "bm25-store-roundtrip";
    const store = new FullTextGenerationIndexStore(memory.storage, DEFAULT_SETTINGS.output, built.descriptor.scopeFingerprint, built.descriptor.identificationFingerprint);
    const staged = await store.stageAndPromote({ descriptor: built.descriptor, objects: built.objects, writerToken: `writer-roundtrip-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1 });
    await expect(staged.validateClosure()).resolves.toBeUndefined();
    const opened = await store.openCurrent(); expect(opened).not.toBeNull();
    await expect(searchGenerationBm25({ generation: opened!, queryText: "alpha" })).resolves.toMatchObject([{ paperKey: "paper:a" }]);
    const titleOnly = await searchGenerationBm25({ generation: opened!, queryText: "Title only" });
    expect(titleOnly[0]).toMatchObject({ paperKey: "paper:b", hits: [{ chunkIndex: 0, score: 0, text: "beta" }] });
  });

  it("ignores post-promotion corruption in unselected evidence and typed-rejects selected corruption", async () => {
    const memory = memoryStorage();
    const built = generationFixture([paper("paper:a", "Healthy", ["alpha"]), paper("paper:b", "Broken", ["beta"])], 1, 1);
    built.descriptor.generationId = "bm25-corruption";
    const store = new FullTextGenerationIndexStore(memory.storage, DEFAULT_SETTINGS.output, built.descriptor.scopeFingerprint, built.descriptor.identificationFingerprint);
    await store.stageAndPromote({ descriptor: built.descriptor, objects: built.objects, writerToken: `writer-corrupt-${"f".repeat(32)}`, expectedCurrent: null, sourceCurrentRevision: () => 1 });
    const opened = await store.openCurrent(); expect(opened).not.toBeNull();
    const evidenceRefs = opened!.descriptor.objects.filter((reference) => reference.kind === "evidence"); expect(evidenceRefs).toHaveLength(2);
    const badPath = `${store.paths.generationsDirectory}/${opened!.descriptor.generationId}/${evidenceRefs[1]!.path}`;
    const damaged = memory.binary.get(badPath)!.slice(); damaged[damaged.length - 1]! ^= 1; memory.binary.set(badPath, damaged);
    const healthyStats = stats();
    await expect(searchGenerationBm25({ generation: opened!, queryText: "alpha", stats: healthyStats })).resolves.toMatchObject([{ paperKey: "paper:a" }]);
    expect(healthyStats.evidenceReads).toBe(1);
    await expect(searchGenerationBm25({ generation: opened!, queryText: "beta" })).rejects.toMatchObject({ name: "FullTextGenerationIndexStoreError", code: "corrupt-or-unreadable" });
  });

  it("late-materializes each selected evidence ref once and never reads unselected evidence", async () => {
    const fixture = generationFixture([paper("paper:a", "Alpha", ["alpha", "alpha alpha"]), paper("paper:b", "Other", ["beta"])]);
    const original = fixture.opened.readObject.bind(fixture.opened); let evidenceReads = 0;
    const opened = { ...fixture.opened, readObject: async (reference: GenerationObjectReference) => { if (reference.kind === "evidence") evidenceReads += 1; return original(reference); } } as OpenedFullTextGeneration;
    const stats = { dictionaryReads: 0, postingsReads: 0, metadataReads: 0, evidenceReads: 0, peakChunkAccumulators: 0, peakCandidates: 0, peakHits: 0, totalRetainedHits: 0, maxLiveBlocks: 0 };
    await searchGenerationBm25({ generation: opened, queryText: "alpha", limit: 1, maxHitsPerPaper: 2, stats });
    expect(evidenceReads).toBe(1); expect(stats.evidenceReads).toBe(1);
    const corruptSelected = { ...fixture.opened, readObject: async (reference: GenerationObjectReference) => { if (reference.kind === "evidence") throw new Error("selected corrupt"); return original(reference); } } as OpenedFullTextGeneration;
    await expect(searchGenerationBm25({ generation: corruptSelected, queryText: "alpha", limit: 1 })).rejects.toThrow("selected corrupt");
    const noEvidence = { ...fixture.opened, readObject: async (reference: GenerationObjectReference) => { if (reference.kind === "evidence") throw new Error("must not read unselected evidence"); return original(reference); } } as OpenedFullTextGeneration;
    await expect(searchGenerationBm25({ generation: noEvidence, queryText: "unknown-no-match" })).resolves.toEqual([]);
  });

  it("rejects v2/dense-only capability", async () => {
    const fixture = generationFixture([paper("paper:a", "Alpha", ["alpha"])]);
    const denseOnly = { ...fixture.opened, descriptor: { ...fixture.opened.descriptor, schemaVersion: 2, lexicalCapability: "none" } } as OpenedFullTextGeneration;
    await expect(searchGenerationBm25({ generation: denseOnly, queryText: "alpha" })).rejects.toMatchObject({ name: "FullTextGenerationIndexStoreError", code: "capability-unsupported" });
  });

  it("counts one completed read before abort-after-I/O-return at every reader await", async () => {
    // This seam aborts after the mocked I/O promise returns. It deliberately does not claim pending I/O is interruptible.
    const fixture = generationFixture([paper("paper:a", "Alpha", ["alpha"])]);
    const cases = [
      ["readLexicalDictionary", "dictionaryReads"],
      ["readLexicalPostings", "postingsReads"],
      ["readPaperMetadata", "metadataReads"],
      ["readObject", "evidenceReads"],
    ] as const;
    for (const [method, counter] of cases) {
      const controller = new AbortController(); const original = fixture.opened[method].bind(fixture.opened) as (...args: any[]) => Promise<any>;
      const opened = { ...fixture.opened, [method]: async (...args: any[]) => { const value = await original(...args); controller.abort(); return value; } } as OpenedFullTextGeneration;
      const observed = stats();
      await expect(searchGenerationBm25({ generation: opened, queryText: "alpha", signal: controller.signal, stats: observed })).rejects.toMatchObject({ name: "AbortError" });
      expect(observed[counter]).toBe(1);
    }
  });
});
