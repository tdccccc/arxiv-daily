import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  FullTextGenerationIndexStore,
  createEvidenceChunkId,
  searchFullTextKnowledgeBase,
  synchronizeFullTextGenerationIndex,
  type EmbeddingModel,
  type FullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBaseStore,
  type FullTextPaperDocument,
} from "../src/index";
import { tokenizeUnicode } from "../src/library/fulltext/bm25-retrieval";
import { MAX_BM25_QUERY_TERMS, limitBm25QueryTerms } from "../src/library/fulltext/generation-bm25-index";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

// Free-text searches (Dashboard search box, palette command, CLI
// `library search`, workbench search) can carry a pasted abstract. The real
// schema-v4 BM25 generation index rejects queries with more than 64 unique
// terms, so core search must cap the lexical leg itself: a fake search port
// cannot catch that.

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const DERIVATION = { parser: { id: "fixture-parser", version: "1" }, chunkerVersion: 2, embeddingInputVersion: 1 } as const;

const ALPHA_KEY = "arxiv:2501.00001";
const BETA_KEY = "arxiv:2501.00002";
const GAMMA_KEY = "arxiv:2501.00003";

function uniqueWords(prefix: string, count: number): string[] {
  return Array.from({ length: count }, (_, index) => `${prefix}${String.fromCharCode(97 + (index % 26))}${Math.floor(index / 26)}`);
}

// 120 distinct filler words: far beyond the index's per-query term limit.
const ABSTRACT_WORDS = uniqueWords("term", 120);
const QUERY_TITLE = "Galaxy cluster lensing survey";
const QUERY_ABSTRACT = `${ABSTRACT_WORDS.join(" ")} lensing.`;

function document(paperKey: string, text: string, vector: readonly number[]): FullTextPaperDocument {
  const identity = { text, headings: ["Body"], locator: { pageStart: 1 }, derivation: DERIVATION };
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey,
    modelId: "model-a",
    dimension: 2,
    textHash: `sha256:${"c".repeat(64)}`,
    title: `Title ${paperKey}`,
    filePaths: [`${paperKey.replaceAll(":", "-")}.pdf`],
    observationFingerprints: [`sha256:${"d".repeat(64)}`],
    derivation: DERIVATION,
    chunks: [{ id: createEvidenceChunkId(identity), index: 0, page: 1, ...identity }],
    vectors: new Float32Array(vector),
    updatedAt: "2026-10-15T00:00:00.000Z",
  };
}

function memoryStorage(): StorageAdapter {
  const text = new Map<string, string>();
  const binary = new Map<string, Uint8Array>();
  const dirs = new Set<string>();
  const normalizePath = (path: string) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
  return {
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
      const value = text.get(normalizePath(from));
      if (value === undefined) throw new Error(`missing ${from}`);
      text.delete(normalizePath(from));
      text.set(normalizePath(to), value);
    },
    writeBinary: async (path, value) => { binary.set(normalizePath(path), new Uint8Array(value).slice()); },
    readBinary: async (path) => {
      const value = binary.get(normalizePath(path));
      if (!value) throw new Error(`missing ${path}`);
      return value.slice().buffer;
    },
    list: async (directory) => {
      const prefix = `${normalizePath(directory)}/`;
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
}

async function buildRealIndex() {
  const documents = [
    // Alpha shares the title topic and a slice of the abstract's words.
    document(ALPHA_KEY, `galaxy cluster lensing survey ${ABSTRACT_WORDS.slice(0, 10).join(" ")}`, [1, 0]),
    document(BETA_KEY, "protein folding kinetics in membranes", [0, 1]),
    document(GAMMA_KEY, "supernova light curve photometry", [0, 1]),
  ];
  const byKey = new Map(documents.map((entry) => [entry.paperKey, entry]));
  const manifest: FullTextKnowledgeBaseManifest = {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    revision: 3,
    scopeFingerprint: SCOPE,
    identificationFingerprint: IDENTIFICATION,
    modelId: "model-a",
    dimension: 2,
    updatedAt: "2026-10-15T00:00:00.000Z",
    papers: Object.fromEntries(documents.map((entry) => [entry.paperKey, {
      paperKey: entry.paperKey,
      status: "ready" as const,
      modelId: entry.modelId,
      dimension: entry.dimension,
      textHash: entry.textHash,
      title: entry.title,
      filePaths: [...entry.filePaths],
      observationFingerprints: [...entry.observationFingerprints],
      derivation: entry.derivation,
      chunkCount: entry.chunks.length,
      updatedAt: entry.updatedAt,
    }])),
  };
  const store: FullTextKnowledgeBaseStore = {
    paths: {
      directory: "legacy",
      manifest: { directory: "legacy", documentPath: "legacy/manifest.json", backupPath: "legacy/manifest.json.backup" },
      papersDirectory: "legacy/papers",
    },
    loadManifest: vi.fn(async () => manifest),
    loadPaper: vi.fn(async (paperKey: string) => byKey.get(paperKey) ?? null),
    replaceManifest: vi.fn(async (next) => next),
    savePaper: vi.fn(async () => {}),
    removePaper: vi.fn(async () => {}),
    removeAll: vi.fn(async () => {}),
  };
  const storage = memoryStorage();
  const generationStore = new FullTextGenerationIndexStore(storage, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION);
  await synchronizeFullTextGenerationIndex({
    sourceStore: store,
    generationStore,
    storage,
    output: DEFAULT_SETTINGS.output,
    scopeFingerprint: SCOPE,
    identificationFingerprint: IDENTIFICATION,
    writerToken: `writer-real-index-${"f".repeat(40)}`.slice(0, 64),
  });
  const embedding: EmbeddingModel = {
    modelId: "model-a",
    dimension: 2,
    prefixPolicy: "none",
    embed: vi.fn(async (texts) => texts.map(() => new Float32Array([1, 0]))),
  };
  return { store, generationStore, embedding, manifest };
}

const SEARCH_BASE = async () => {
  const index = await buildRealIndex();
  return {
    ...index,
    search: (options: { queryText: string; lexicalQueryText?: string; mode: "lexical" | "hybrid" | "dense" }) => searchFullTextKnowledgeBase({
      store: index.store,
      generationStore: index.generationStore,
      sourceManifest: index.manifest,
      embedding: index.embedding,
      limit: 5,
      maxHitsPerPaper: 1,
      ...options,
    }),
  };
};

const LONG_QUERY = `${QUERY_TITLE} ${QUERY_ABSTRACT}`;

describe("searchFullTextKnowledgeBase lexical query cap over a real BM25 generation index", () => {
  it("accepts a 120-term free-text query in lexical mode", async () => {
    const { search } = await SEARCH_BASE();
    const found = await search({ queryText: LONG_QUERY, mode: "lexical" });
    expect(found[0]?.paperKey).toBe(ALPHA_KEY);
  });

  it("accepts a 120-term free-text query in hybrid mode and still embeds the whole text", async () => {
    const { search, embedding } = await SEARCH_BASE();
    const found = await search({ queryText: LONG_QUERY, mode: "hybrid" });
    expect(found[0]?.paperKey).toBe(ALPHA_KEY);
    const embedded = vi.mocked(embedding.embed).mock.calls[0]![0][0]!;
    expect(embedded).toContain(ABSTRACT_WORDS.at(-1)!);
  });

  it("caps an explicit lexicalQueryText that is over the limit too", async () => {
    const { search } = await SEARCH_BASE();
    const found = await search({ queryText: "galaxy", lexicalQueryText: LONG_QUERY, mode: "lexical" });
    expect(found[0]?.paperKey).toBe(ALPHA_KEY);
  });

  it("returns the same ranking for a query within the limit as before", async () => {
    const { search } = await SEARCH_BASE();
    const short = "galaxy cluster lensing survey";
    const direct = await search({ queryText: short, mode: "lexical" });
    const viaCap = await search({ queryText: short, lexicalQueryText: limitBm25QueryTerms([short]), mode: "lexical" });
    expect(direct.map((match) => match.paperKey)).toEqual(viaCap.map((match) => match.paperKey));
    expect(direct[0]?.paperKey).toBe(ALPHA_KEY);
    expect(direct.map((match) => match.score)).toEqual(viaCap.map((match) => match.score));
  });
});

describe("limitBm25QueryTerms", () => {
  const count = (text: string) => new Set(tokenizeUnicode(text)).size;

  it("leaves text within the limit unchanged", () => {
    expect(limitBm25QueryTerms(["Galaxy survey", "We map galaxies. Galaxy!"])).toBe("Galaxy survey\nWe map galaxies. Galaxy!");
  });

  it("keeps the first parts' terms in order and counts unique normalized terms exactly", () => {
    const title = "Deep NIR Imaging";
    const abstract = `${uniqueWords("w", 100).join(" ")} DEEP imaging`;
    const bounded = limitBm25QueryTerms([title, abstract]);
    expect(count(bounded)).toBe(MAX_BM25_QUERY_TERMS);
    expect(tokenizeUnicode(bounded).slice(0, 4)).toEqual(["deep", "nir", "imaging", "wa0"]);
  });

  it("honors the limit for Han bigrams and full-width forms", () => {
    const han = "星系团引力透镜巡天观测".repeat(2) + Array.from({ length: 80 }, (_, i) => `Ｔｅｒｍ${i}`).join(" ");
    const bounded = limitBm25QueryTerms([han]);
    expect(count(bounded)).toBe(MAX_BM25_QUERY_TERMS);
    expect(tokenizeUnicode(bounded)).toEqual([...new Set(tokenizeUnicode(han))].slice(0, MAX_BM25_QUERY_TERMS));
  });

  it("can be cut to a smaller limit", () => {
    expect(count(limitBm25QueryTerms(["one two three four five"], 3))).toBe(3);
  });
});
