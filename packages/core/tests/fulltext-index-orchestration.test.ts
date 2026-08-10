import { describe, expect, it } from "vitest";
import {
  indexPersonalLibraryFullText,
  searchFullTextKnowledgeBase,
} from "../src/library/fulltext/index-orchestration";
import { chunkFullText } from "../src/library/fulltext/chunking";
import type {
  FullTextKnowledgeBaseManifest,
  FullTextKnowledgeBaseStore,
  FullTextPaperDocument,
} from "../src/library/fulltext/knowledge-base";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  decodeFullTextKnowledgeBaseManifest,
  decodeFullTextPaperDocument,
  serializeFullTextPaperDocument,
} from "../src/library/fulltext/knowledge-base";
import type { EmbeddingModel, PdfTextExtractor } from "../src/library/fulltext/ports";
import type { ScopedLibrarySource } from "../src/library/scoped-library-source";
import type { PersonalLibraryCatalog } from "../src/library/personal-library-catalog";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const NOW = "2026-08-05T00:00:00.000Z";

function fingerprint(seed: string): string {
  return `sha256:${seed.padEnd(64, "0").slice(0, 64)}`;
}

/** In-memory store fake with the same CAS semantics the real store enforces. */
class MemoryStore implements FullTextKnowledgeBaseStore {
  paths = { directory: "kb", manifest: { directory: "kb", documentPath: "kb/manifest.json", backupPath: "kb/manifest.json.backup" }, papersDirectory: "kb/papers" };
  manifest: FullTextKnowledgeBaseManifest;
  private readonly papers = new Map<string, FullTextPaperDocument>();
  removed = new Set<string>();

  constructor(modelId = "fake-e5-q8", dimension = 4) {
    this.manifest = {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      revision: 0,
      scopeFingerprint: SCOPE,
      identificationFingerprint: IDENTIFICATION,
      modelId,
      dimension,
      updatedAt: NOW,
      papers: {},
    };
  }

  async loadManifest(): Promise<FullTextKnowledgeBaseManifest> {
    // Round-trip through the strict decoder, mirroring the real store.
    const decoded = decodeFullTextKnowledgeBaseManifest(JSON.parse(JSON.stringify(this.manifest)));
    if (!decoded) throw new Error("manifest failed strict decode");
    return decoded;
  }

  async replaceManifest(
    next: FullTextKnowledgeBaseManifest,
    expectedRevision: number,
  ): Promise<FullTextKnowledgeBaseManifest> {
    if (expectedRevision !== this.manifest.revision) {
      throw new Error(`stale revision: expected ${expectedRevision}, current ${this.manifest.revision}`);
    }
    if (Object.keys(this.manifest.papers).length > 0 && this.manifest.modelId !== next.modelId) {
      throw new Error("model switch requires rebuilding");
    }
    const decoded = decodeFullTextKnowledgeBaseManifest(JSON.parse(JSON.stringify(next)));
    if (!decoded) throw new Error("manifest failed strict decode");
    decoded.revision = this.manifest.revision + 1;
    this.manifest = decoded;
    return this.loadManifest();
  }

  async loadPaper(paperKey: string): Promise<FullTextPaperDocument | null> {
    const document = this.papers.get(paperKey);
    if (!document) return null;
    // Mirror the real store's per-paper serialization/decode round-trip.
    const decoded = decodeFullTextPaperDocument(JSON.parse(serializeFullTextPaperDocument(document)));
    if (!decoded) throw new Error("paper document failed strict decode");
    return decoded;
  }

  async savePaper(document: FullTextPaperDocument): Promise<void> {
    const decoded = decodeFullTextPaperDocument(JSON.parse(serializeFullTextPaperDocument(document)));
    if (!decoded) throw new Error("paper document failed strict decode");
    this.papers.set(document.paperKey, decoded);
  }

  async removePaper(paperKey: string): Promise<void> {
    this.papers.delete(paperKey);
    this.removed.add(paperKey);
  }

  async removeAll(): Promise<void> {
    this.papers.clear();
    this.manifest.revision = 0;
  }
}

class FakeSource implements ScopedLibrarySource {
  async inventory(): Promise<never> {
    throw new Error("not used in this test");
  }

  async readBinary(path: string): Promise<ArrayBuffer> {
    return new TextEncoder().encode(path).buffer as ArrayBuffer;
  }
}

class FakeExtractor implements PdfTextExtractor {
  calls = 0;
  failFor = new Set<string>();

  constructor(private readonly texts: Record<string, string[]>) {}

  async extractPdfText(bytes: Uint8Array): Promise<{ pages: readonly string[] }> {
    this.calls += 1;
    const path = new TextDecoder().decode(bytes);
    if (this.failFor.has(path)) throw new Error("extraction boom");
    return { pages: this.texts[path] ?? ["fallback page with enough text content"] };
  }
}

class FakeEmbedding implements EmbeddingModel {
  readonly modelId = "fake-e5-q8";
  readonly dimension = 4;
  readonly prefixPolicy = "e5" as const;
  calls = 0;

  constructor(private readonly vectors: Readonly<Record<string, Float32Array>> = {}) {}

  async embed(texts: readonly string[]): Promise<readonly Float32Array[]> {
    this.calls += 1;
    return texts.map((text) => this.vectors[text] ?? new Float32Array(this.dimension));
  }
}

function makeCatalog(
  papers: Array<{ paperKey: string; filePaths: string[]; fingerprint: string }>,
): PersonalLibraryCatalog {
  const files: PersonalLibraryCatalog["files"] = Object.create(null);
  const paperRecords: PersonalLibraryCatalog["papers"] = Object.create(null);
  for (const paper of papers) {
    for (const path of paper.filePaths) {
      files[path] = {
        path,
        status: "ready",
        observationFingerprint: paper.fingerprint,
        paperKey: paper.paperKey,
        arxivId: paper.paperKey.slice("arxiv:".length),
        updatedAt: NOW,
      };
    }
    paperRecords[paper.paperKey] = {
      paperKey: paper.paperKey,
      source: "arxiv",
      externalId: paper.paperKey.slice("arxiv:".length),
      title: `Title of ${paper.paperKey}`,
      authors: ["A. Author"],
      abstract: "Abstract text.",
      published: "2026-01-01T00:00:00.000Z",
      updated: "2026-01-01T00:00:00.000Z",
      primaryCategory: "cs.LG",
      categories: ["cs.LG"],
      evidenceDepth: "metadata-and-abstract",
      filePaths: [...paper.filePaths],
    };
  }
  return {
    schemaVersion: 1,
    revision: 1,
    scopeFingerprint: SCOPE,
    identificationFingerprint: IDENTIFICATION,
    updatedAt: NOW,
    lastScan: null,
    files,
    papers: paperRecords,
  };
}

const LONG_ALPHA = Array.from({ length: 600 }, () => "alpha").join(" ");
const LONG_BETA = Array.from({ length: 600 }, () => "beta").join(" ");

describe("full-text indexing orchestration", () => {
  it("indexes every paper on the first run and persists per-paper documents", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] });
    const embedding = new FakeEmbedding();

    const summary = await indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      extractor,
      embedding,
      store,
      now: () => new Date(NOW),
    });

    expect(summary.indexed).toBe(1);
    expect(summary.reused).toBe(0);
    expect(summary.failed).toBe(0);
    expect(summary.pruned).toBe(0);
    expect(summary.manifestRevision).toBe(1);
    const document = await store.loadPaper("arxiv:2403.19236");
    expect(document).not.toBeNull();
    expect(document!.chunks.length).toBeGreaterThan(0);
    expect(document!.vectors.length).toBe(document!.chunks.length * 4);
    expect(document!.textHash).toMatch(/^sha256:[a-f0-9]{64}$/);
    const manifest = await store.loadManifest();
    expect(manifest.papers["arxiv:2403.19236"]!.status).toBe("ready");
    expect(manifest.papers["arxiv:2403.19236"]!.chunkCount).toBe(document!.chunks.length);
  });

  it("reuses unchanged papers without re-extracting", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] });
    const embedding = new FakeEmbedding();

    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });
    expect(extractor.calls).toBe(1);
    const summary = await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });

    expect(summary.reused).toBe(1);
    expect(summary.indexed).toBe(0);
    expect(extractor.calls).toBe(1); // no re-extraction
    expect((await store.loadManifest()).revision).toBe(2);
  });

  it("re-indexes only the paper whose observation fingerprint changed", async () => {
    const unchanged = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] });
    const embedding = new FakeEmbedding();
    await indexPersonalLibraryFullText({ catalog: unchanged, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });

    const changed = makeCatalog([
      {
        paperKey: "arxiv:2403.19236",
        filePaths: ["lib/a.pdf"],
        fingerprint: fingerprint("f2"),
      },
      {
        paperKey: "arxiv:2309.11425",
        filePaths: ["lib/b.pdf"],
        fingerprint: fingerprint("f3"),
      },
    ]);
    const changedExtractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA], "lib/b.pdf": [LONG_BETA] });
    const summary = await indexPersonalLibraryFullText({ catalog: changed, source: new FakeSource(), extractor: changedExtractor, embedding, store, now: () => new Date(NOW) });

    expect(summary.indexed).toBe(2);
    expect(summary.reused).toBe(0);
    const manifest = await store.loadManifest();
    expect(manifest.papers["arxiv:2309.11425"]!.status).toBe("ready");
  });

  it("records a failed paper without failing the run", async () => {
    const catalog = makeCatalog([
      {
        paperKey: "arxiv:2403.19236",
        filePaths: ["lib/good.pdf"],
        fingerprint: fingerprint("f1"),
      },
      {
        paperKey: "arxiv:2309.11425",
        filePaths: ["lib/bad.pdf"],
        fingerprint: fingerprint("f2"),
      },
    ]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/good.pdf": [LONG_ALPHA], "lib/bad.pdf": [LONG_BETA] });
    extractor.failFor.add("lib/bad.pdf");
    const embedding = new FakeEmbedding();

    const summary = await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });

    expect(summary.indexed).toBe(1);
    expect(summary.failed).toBe(1);
    const manifest = await store.loadManifest();
    expect(manifest.papers["arxiv:2309.11425"]!.status).toBe("failed");
    expect(manifest.papers["arxiv:2309.11425"]!.error).toContain("extraction boom");
    expect(manifest.papers["arxiv:2403.19236"]!.status).toBe("ready");
    expect(await store.loadPaper("arxiv:2309.11425")).toBeNull();
  });

  it("retries a failed paper on the next run", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] });
    const embedding = new FakeEmbedding();
    extractor.failFor.add("lib/a.pdf");
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });
    expect((await store.loadManifest()).papers["arxiv:2403.19236"]!.status).toBe("failed");

    extractor.failFor.clear();
    const summary = await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });
    expect(summary.indexed).toBe(1);
    expect((await store.loadManifest()).papers["arxiv:2403.19236"]!.status).toBe("ready");
  });

  it("prunes papers that left the catalog and their documents", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] });
    const embedding = new FakeEmbedding();
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });

    const shrunk = makeCatalog([]);
    const summary = await indexPersonalLibraryFullText({ catalog: shrunk, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });

    expect(summary.pruned).toBe(1);
    expect(store.removed.has("arxiv:2403.19236")).toBe(true);
    expect(await store.loadPaper("arxiv:2403.19236")).toBeNull();
    expect((await store.loadManifest()).papers).toEqual({});
  });

  it("rejects a model switch over a populated knowledge base", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] });
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding: new FakeEmbedding(), store, now: () => new Date(NOW) });

    const otherModel = new FakeEmbedding();
    (otherModel as unknown as { modelId: string }).modelId = "other-model";
    await expect(indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      extractor,
      embedding: otherModel,
      store,
      now: () => new Date(NOW),
    })).rejects.toThrow(/rebuild/);
  });

  it("rejects a store bound to different fingerprints", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    store.manifest.scopeFingerprint = `sha256:${"c".repeat(64)}`;
    await expect(indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      extractor: new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] }),
      embedding: new FakeEmbedding(),
      store,
      now: () => new Date(NOW),
    })).rejects.toThrow(/do not match/);
  });

  it("searches with explainable hit evidence", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA, LONG_BETA] });
    // Build the embedding map from the real chunk texts so the fixture is
    // robust to chunk-boundary decisions: the alpha page becomes one chunk
    // (page 1), the beta page another (page 2).
    const chunks = chunkFullText([LONG_ALPHA, LONG_BETA]);
    const alphaChunk = chunks.find((chunk) => chunk.text.includes("alpha"))!;
    const betaChunk = chunks.find((chunk) => chunk.text.includes("beta"))!;
    const alpha = new Float32Array([1, 0, 0, 0]);
    const beta = new Float32Array([0, 1, 0, 0]);
    const embedding = new FakeEmbedding({
      [`query: ${alphaChunk.text}`]: alpha,
      [`passage: ${alphaChunk.text}`]: alpha,
      [`passage: ${betaChunk.text}`]: beta,
    });
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });

    const matches = await searchFullTextKnowledgeBase({
      store,
      embedding,
      queryText: alphaChunk.text,
    });

    expect(matches.length).toBe(1);
    expect(matches[0]!.paperKey).toBe("arxiv:2403.19236");
    expect(matches[0]!.hits[0]!.text).toContain("alpha");
    expect(matches[0]!.hits[0]!.page).toBe(1);
    expect(matches[0]!.score).toBeCloseTo(1, 5);
  });

  it("searches an empty knowledge base to an empty result", async () => {
    const store = new MemoryStore();
    const matches = await searchFullTextKnowledgeBase({
      store,
      embedding: new FakeEmbedding(),
      queryText: "anything",
    });
    expect(matches).toEqual([]);
    expect((new FakeEmbedding()).calls).toBe(0);
  });
});
