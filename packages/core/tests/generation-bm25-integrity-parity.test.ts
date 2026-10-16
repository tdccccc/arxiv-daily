import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  FullTextGenerationIndexStore,
  createEvidenceChunkId,
  synchronizeFullTextGenerationIndex,
  type FullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBaseStore,
  type FullTextPaperDocument,
} from "../src/index";
import { searchKnowledgeBaseBm25 } from "../src/library/fulltext/bm25-retrieval";
import { searchGenerationBm25 } from "../src/library/fulltext/generation-bm25-index";
import { decodePaperMetadataBlock, encodePaperMetadataBlock, lexicalTermBucket } from "../src/library/fulltext/generation-index-format";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

// Pins that the optimized hashing / bucket-derivation path returns exactly the
// same ranked results as the independent in-memory BM25, and that every layer
// of integrity checking on the real read path still rejects modified files.

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const DERIVATION = { parser: { id: "fixture-parser", version: "1" }, chunkerVersion: 2, embeddingInputVersion: 1 } as const;

const PAPERS: ReadonlyArray<{ key: string; title: string; chunks: readonly string[] }> = [
  { key: "arxiv:2501.00001", title: "Weak lensing of galaxy clusters", chunks: ["weak lensing mass of galaxy clusters from shear catalogs", "photometric redshift calibration for the survey", "dark energy constraints from cluster counts"] },
  { key: "arxiv:2501.00002", title: "Protein folding kinetics", chunks: ["protein folding kinetics in lipid membranes", "molecular dynamics of membrane proteins"] },
  { key: "arxiv:2501.00003", title: "Supernova photometry", chunks: ["supernova light curve photometry and redshift", "type Ia supernova distance ladder calibration", "galaxy host mass step"] },
  { key: "arxiv:2501.00004", title: "星系团引力透镜", chunks: ["星系团 引力透镜 质量 重建", "lensing mass reconstruction pipeline"] },
  { key: "arxiv:2501.00005", title: "Cosmic voids", chunks: ["cosmic voids in galaxy surveys and the dark energy equation of state", "void lensing profiles"] },
  { key: "arxiv:2501.00006", title: "Radio dipole", chunks: ["cosmic radio dipole from source counts", "kinematic dipole and the SKA", "survey systematics"] },
];
const GOLDEN: ReadonlyArray<ReadonlyArray<{ key: string; score: number; hits: ReadonlyArray<readonly [number, number]> }>> = [[{"key":"arxiv:2501.00001","score":5.651114269159564,"hits":[[0,5.651114269159564],[2,2.3341960831776998]]},{"key":"arxiv:2501.00003","score":3.4816834143229287,"hits":[[2,3.4816834143229287]]},{"key":"arxiv:2501.00004","score":3.4816834143229287,"hits":[[1,3.4816834143229287]]},{"key":"arxiv:2501.00005","score":1.8938453728299525,"hits":[[1,1.8938453728299525],[0,1.0574142103478033]]}],[{"key":"arxiv:2501.00001","score":3.6609524508601843,"hits":[[1,3.6609524508601843]]},{"key":"arxiv:2501.00003","score":3.6609524508601843,"hits":[[0,3.6609524508601843],[1,3.6609524508601843]]}],[{"key":"arxiv:2501.00001","score":7.825624759467977,"hits":[[2,7.825624759467977],[1,1.8304762254300921],[0,1.239960141801708]]},{"key":"arxiv:2501.00005","score":2.5830275198544594,"hits":[[0,2.5830275198544594],[1,1.8938453728299525]]},{"key":"arxiv:2501.00006","score":2.536013100328201,"hits":[[2,2.536013100328201],[0,1.8304762254300921]]},{"key":"arxiv:2501.00004","score":1.7408417071614644,"hits":[[1,1.7408417071614644]]}],[{"key":"arxiv:2501.00002","score":2.5086791790962986,"hits":[[1,2.5086791790962986],[0,2.3341960831776998]]}],[{"key":"arxiv:2501.00006","score":10.159820842645676,"hits":[[0,10.159820842645676],[1,1.9673058436100828]]},{"key":"arxiv:2501.00001","score":1.8304762254300921,"hits":[[2,1.8304762254300921]]},{"key":"arxiv:2501.00005","score":1.2915137599272297,"hits":[[0,1.2915137599272297]]}],[{"key":"arxiv:2501.00004","score":4.364811750913213,"hits":[[0,4.364811750913213]]}],[]];
const QUERIES = [
  "weak lensing galaxy cluster mass",
  "supernova redshift calibration",
  "dark energy survey cluster counts lensing",
  "protein membrane",
  "cosmic dipole radio source counts",
  "星系团",
  "completely unrelated zebra",
];

function document(paper: (typeof PAPERS)[number], ordinal: number): FullTextPaperDocument {
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey: paper.key,
    modelId: "model-a",
    dimension: 2,
    textHash: `sha256:${"c".repeat(64)}`,
    title: paper.title,
    filePaths: [`${ordinal}.pdf`],
    observationFingerprints: [`sha256:${"d".repeat(64)}`],
    derivation: DERIVATION,
    chunks: paper.chunks.map((text, index) => {
      const identity = { text, headings: ["Body"], locator: { pageStart: index + 1 }, derivation: DERIVATION };
      return { id: createEvidenceChunkId(identity), index, page: index + 1, ...identity };
    }),
    vectors: new Float32Array(paper.chunks.flatMap(() => [1, 0])),
    updatedAt: "2026-10-15T00:00:00.000Z",
  };
}

function memoryStorage() {
  const text = new Map<string, string>();
  const binary = new Map<string, Uint8Array>();
  const dirs = new Set<string>();
  const normalizePath = (path: string) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
  const storage: StorageAdapter = {
    normalizePath,
    readText: async (path) => { const value = text.get(normalizePath(path)); if (value === undefined) throw new Error(`missing ${path}`); return value; },
    writeText: async (path, value) => { text.set(normalizePath(path), value); },
    writeTextAtomic: async (path, value) => { text.set(normalizePath(path), value); },
    createTextExclusive: async (path, value) => {
      const normalized = normalizePath(path);
      if (text.has(normalized) || binary.has(normalized) || dirs.has(normalized)) return false;
      text.set(normalized, value);
      return true;
    },
    exists: async (path) => { const n = normalizePath(path); return text.has(n) || binary.has(n) || dirs.has(n); },
    mkdir: async (path) => { dirs.add(normalizePath(path)); },
    remove: async (path) => {
      const normalized = normalizePath(path); const prefix = `${normalized}/`;
      for (const key of [...text.keys()]) if (key === normalized || key.startsWith(prefix)) text.delete(key);
      for (const key of [...binary.keys()]) if (key === normalized || key.startsWith(prefix)) binary.delete(key);
      for (const key of [...dirs]) if (key === normalized || key.startsWith(prefix)) dirs.delete(key);
    },
    rename: async (from, to) => { const value = text.get(normalizePath(from)); if (value === undefined) throw new Error(`missing ${from}`); text.delete(normalizePath(from)); text.set(normalizePath(to), value); },
    writeBinary: async (path, value) => { binary.set(normalizePath(path), new Uint8Array(value).slice()); },
    readBinary: async (path) => { const value = binary.get(normalizePath(path)); if (!value) throw new Error(`missing ${path}`); return value.slice().buffer; },
    list: async (directory) => {
      const prefix = `${normalizePath(directory)}/`; const entries = new Map<string, "file" | "folder">();
      for (const path of [...text.keys(), ...binary.keys(), ...dirs]) {
        if (!path.startsWith(prefix)) continue;
        const suffix = path.slice(prefix.length); if (!suffix) continue;
        const [head, ...rest] = suffix.split("/");
        entries.set(`${prefix}${head}`, rest.length > 0 || dirs.has(`${prefix}${head}`) ? "folder" : "file");
      }
      return [...entries].map(([path, type]) => ({ path, type }));
    },
  };
  return { storage, binary };
}

async function buildIndex() {
  const documents = PAPERS.map(document);
  const byKey = new Map(documents.map((entry) => [entry.paperKey, entry]));
  const manifest: FullTextKnowledgeBaseManifest = {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION, revision: 3, scopeFingerprint: SCOPE, identificationFingerprint: IDENTIFICATION,
    modelId: "model-a", dimension: 2, updatedAt: "2026-10-15T00:00:00.000Z",
    papers: Object.fromEntries(documents.map((entry) => [entry.paperKey, {
      paperKey: entry.paperKey, status: "ready" as const, modelId: entry.modelId, dimension: entry.dimension, textHash: entry.textHash, title: entry.title,
      filePaths: [...entry.filePaths], observationFingerprints: [...entry.observationFingerprints], derivation: entry.derivation, chunkCount: entry.chunks.length, updatedAt: entry.updatedAt,
    }])),
  };
  const sourceStore: FullTextKnowledgeBaseStore = {
    paths: { directory: "legacy", manifest: { directory: "legacy", documentPath: "legacy/manifest.json", backupPath: "legacy/manifest.json.backup" }, papersDirectory: "legacy/papers" },
    loadManifest: vi.fn(async () => manifest),
    loadPaper: vi.fn(async (paperKey: string) => byKey.get(paperKey) ?? null),
    replaceManifest: vi.fn(async (next) => next), savePaper: vi.fn(async () => {}), removePaper: vi.fn(async () => {}), removeAll: vi.fn(async () => {}),
  };
  const memory = memoryStorage();
  const generationStore = new FullTextGenerationIndexStore(memory.storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION);
  await synchronizeFullTextGenerationIndex({
    sourceStore, generationStore, storage: memory.storage, output: DEFAULT_SETTINGS.output,
    scopeFingerprint: SCOPE, identificationFingerprint: IDENTIFICATION, writerToken: `writer-integrity-${"f".repeat(40)}`.slice(0, 64),
  });
  return { documents, generationStore, binary: memory.binary };
}

async function search(generationStore: FullTextGenerationIndexStore, queryText: string) {
  const generation = await generationStore.openCurrent();
  try { return await searchGenerationBm25({ generation: generation!, queryText, limit: 10, maxHitsPerPaper: 3 }); }
  finally { await generation!.close(); }
}

const summarize = (matches: Awaited<ReturnType<typeof search>>) =>
  matches.map((match) => ({ key: match.paperKey, score: match.score, hits: match.hits.map((hit) => [hit.chunkIndex, hit.score]) }));

function objectPath(binary: Map<string, Uint8Array>, marker: string): string {
  const path = [...binary.keys()].find((key) => key.includes(marker));
  if (!path) throw new Error(`no object path containing ${marker}: ${[...binary.keys()].join(", ")}`);
  return path;
}

describe("optimized hashing leaves BM25 results unchanged", () => {
  it("returns the same ranked papers, chunk hits and scores as the independent in-memory BM25", async () => {
    const { documents, generationStore } = await buildIndex();
    for (const queryText of QUERIES) {
      const expected = searchKnowledgeBaseBm25({ papers: documents, queryText, limit: 10, maxHitsPerPaper: 3 });
      const actual = await search(generationStore, queryText);
      expect(actual.map((match) => match.paperKey), queryText).toEqual(expected.map((match) => match.paperKey));
      for (const [position, match] of actual.entries()) {
        expect(match.score).toBeCloseTo(expected[position]!.score, 10);
        expect(match.hits.map((hit) => hit.chunkIndex)).toEqual(expected[position]!.hits.map((hit) => hit.chunkIndex));
      }
    }
  });

  it("is repeatable across searches and fresh opens (no state leaks between queries)", async () => {
    const { generationStore } = await buildIndex();
    const first = QUERIES.map(async (query) => summarize(await search(generationStore, query)));
    const firstRun = await Promise.all(first);
    const secondRun: unknown[] = [];
    for (const query of QUERIES) secondRun.push(summarize(await search(generationStore, query)));
    expect(secondRun).toEqual(firstRun);
    // Golden ranking (keys, chunk hits, BM25 scores) captured from the pre-optimization implementation.
    expect(firstRun.map((matches) => matches.map((match) => match.key))).toEqual(GOLDEN.map((matches) => matches.map((match) => match.key)));
    for (const [queryIndex, matches] of GOLDEN.entries()) {
      for (const [position, expected] of matches.entries()) {
        const actual = firstRun[queryIndex]![position]!;
        expect(actual.score).toBeCloseTo(expected.score, 12);
        expect(actual.hits.map((hit) => hit[0])).toEqual(expected.hits.map((hit) => hit[0]));
        actual.hits.forEach((hit, hitIndex) => expect(hit[1] as number).toBeCloseTo(expected.hits[hitIndex]![1]!, 12));
      }
    }
  });

  it("derives the same term buckets as a direct SHA-256, cold or cached", () => {
    const direct = (namespace: string, term: string) =>
      Number.parseInt(createHash("sha256").update(Buffer.concat([Buffer.from(namespace), Buffer.from([0]), Buffer.from(term)])).digest("hex").slice(0, 2), 16);
    const terms = ["galaxy", "cluster", "星系团", "a", "lensing", "ab", "x".repeat(200), "naïve", "\u{1F680}"];
    for (let pass = 0; pass < 2; pass += 1) {
      for (const namespace of ["base", "expanded", "alias"] as const) {
        for (const term of terms) expect(lexicalTermBucket(namespace, term), `${namespace}/${term}/${pass}`).toBe(direct(namespace, term));
      }
    }
    expect(() => lexicalTermBucket("bogus" as never, "galaxy")).toThrow();
    expect(() => lexicalTermBucket("base", "galaxy")).not.toThrow();
  });
});

describe("a modified index file is still detected on the search path", () => {
  const query = "weak lensing galaxy cluster mass";
  const kinds = ["metadata", "postings", "dictionary", "evidence"] as const;

  for (const kind of kinds) {
    it(`rejects a one-byte change in a ${kind} object`, async () => {
      const { generationStore, binary } = await buildIndex();
      expect((await search(generationStore, query)).length).toBeGreaterThan(0);
      const path = objectPath(binary, kind);
      const bytes = binary.get(path)!.slice();
      bytes[Math.floor(bytes.length / 2)]! ^= 0x01;
      binary.set(path, bytes);
      await expect(search(generationStore, query)).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    });
  }

  it("detects a change made after a successful search on the SAME opened generation (no verified-once cache)", async () => {
    const { generationStore, binary } = await buildIndex();
    const generation = (await generationStore.openCurrent())!;
    const ok = await searchGenerationBm25({ generation, queryText: query });
    expect(ok.length).toBeGreaterThan(0);
    const path = objectPath(binary, "postings");
    const bytes = binary.get(path)!.slice();
    bytes[bytes.length - 1]! ^= 0xff;
    binary.set(path, bytes);
    await expect(searchGenerationBm25({ generation, queryText: query })).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    await generation.close();
  });

  it("rejects a truncated object and an object with trailing bytes", async () => {
    for (const mutate of [(b: Uint8Array) => b.slice(0, b.length - 1), (b: Uint8Array) => { const out = new Uint8Array(b.length + 1); out.set(b); return out; }]) {
      const { generationStore, binary } = await buildIndex();
      const path = objectPath(binary, "metadata");
      binary.set(path, mutate(binary.get(path)!));
      await expect(search(generationStore, query)).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
    }
  });

  it("still verifies each block's embedded checksum independently of the outer reference checksum", () => {
    const bytes = encodePaperMetadataBlock({ paperStart: 0, records: [{ paperOrdinal: 0, paperKey: "paper:a", chunkStart: 0, chunkCount: 1, title: "A" }] });
    expect(() => decodePaperMetadataBlock(bytes)).not.toThrow();
    const tampered = bytes.slice();
    tampered[tampered.length - 1] = (tampered[tampered.length - 1] ?? 0) ^ 1;
    expect(() => decodePaperMetadataBlock(tampered)).toThrow();
    const checksumFlipped = bytes.slice();
    checksumFlipped[20]! ^= 1;
    expect(() => decodePaperMetadataBlock(checksumFlipped)).toThrow(/checksum/);
  });
});
