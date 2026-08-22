import { describe, expect, it } from "vitest";
import {
  indexPersonalLibraryFullText,
  searchFullTextKnowledgeBase,
  TITLE_EXTRACTION_VERSION,
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
import type { DocumentParser, DocumentParserSelector } from "../src/documents/parsed-document";
import type { EmbeddingModel, PdfTextExtractor } from "../src/library/fulltext/ports";
import { FullTextKnowledgeBaseStoreError } from "../src/library/fulltext/knowledge-base-store";
import type { ScopedLibrarySource } from "../src/library/scoped-library-source";
import type { PersonalLibraryCatalog } from "../src/library/personal-library-catalog";
import { sha256Hex } from "../src/utils/digest";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const NOW = "2026-08-05T00:00:00.000Z";

function fingerprint(seed: string): string {
  return `sha256:${seed.padEnd(64, "0").slice(0, 64)}`;
}

function fallbackPaperKey(content: string): string {
  return `file:sha256:${sha256Hex(content)}`;
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
  constructor(private readonly contents: Readonly<Record<string, string>> = {}) {}

  async inventory(): Promise<never> {
    throw new Error("not used in this test");
  }

  async readBinary(path: string): Promise<ArrayBuffer> {
    return new TextEncoder().encode(this.contents[path] ?? path).buffer as ArrayBuffer;
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
  unresolved: Array<{ path: string; fingerprint: string }> = [],
  titleOverrides: Record<string, string> = {},
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
      title: titleOverrides[paper.paperKey] ?? `Title of ${paper.paperKey}`,
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
  for (const file of unresolved) {
    files[file.path] = {
      path: file.path,
      status: "unresolved",
      observationFingerprint: file.fingerprint,
      updatedAt: NOW,
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

  it("indexes ParsedDocument input with parser derivation and structured headings", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const parser: DocumentParser = {
      capabilities: ["page-text", "document-structure"],
      provenance: { id: "fixture-structured", version: "2" },
      async parse() {
        return {
          mediaType: "application/pdf",
          blocks: [
            { kind: "heading", text: "Methods", headingLevel: 1, locator: { page: 2, block: 0 } },
            { kind: "paragraph", text: "Structured method evidence with enough text.", locator: { page: 2, block: 1 } },
          ],
        };
      },
    };
    const embedding = new FakeEmbedding();

    await indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      parser,
      embedding,
      store,
      now: () => new Date(NOW),
    });

    const document = await store.loadPaper("arxiv:2403.19236");
    expect(document?.derivation?.parser).toEqual(parser.provenance);
    expect(document?.chunks[0]?.headings).toEqual(["Methods"]);
    expect(document?.chunks[0]?.locator).toEqual({ pageStart: 2, pageEnd: 2, blockStart: 1, blockEnd: 1 });
  });

  it("persists the actual parser selected for each indexed document", async () => {
    const catalog = makeCatalog([
      { paperKey: "arxiv:2403.19236", filePaths: ["lib/sidecar.pdf"], fingerprint: fingerprint("f1") },
      { paperKey: "arxiv:2501.00001", filePaths: ["lib/fallback.pdf"], fingerprint: fingerprint("f2") },
    ]);
    const sidecar: DocumentParser = {
      capabilities: ["page-text", "document-structure"],
      provenance: { id: "docling", version: "2.0" },
      async parse() { throw new Error("selector owns sidecar parsing"); },
    };
    const fallback: DocumentParser = {
      capabilities: ["page-text"],
      provenance: { id: "pdfjs", version: "4.10" },
      async parse() { throw new Error("selector owns fallback parsing"); },
    };
    const selector: DocumentParserSelector = {
      preferredParser: sidecar,
      async parse(bytes) {
        const path = new TextDecoder().decode(bytes);
        if (path === "lib/sidecar.pdf") {
          return {
            parser: sidecar,
            document: {
              mediaType: "application/pdf",
              blocks: [
                { kind: "heading", text: "Methods", headingLevel: 1, locator: { page: 2, block: 0 } },
                { kind: "paragraph", text: LONG_ALPHA, locator: { page: 2, block: 1 } },
              ],
            },
          };
        }
        return {
          parser: fallback,
          document: {
            mediaType: "application/pdf",
            blocks: [{ kind: "page", text: LONG_BETA, locator: { page: 1, block: 0 } }],
          },
        };
      },
    };
    const store = new MemoryStore();

    await indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      parserSelector: selector,
      embedding: new FakeEmbedding(),
      store,
      now: () => new Date(NOW),
    });

    const sidecarDocument = await store.loadPaper("arxiv:2403.19236");
    const fallbackDocument = await store.loadPaper("arxiv:2501.00001");
    expect(sidecarDocument?.derivation?.parser).toEqual(sidecar.provenance);
    expect(sidecarDocument?.chunks[0]?.headings).toEqual(["Methods"]);
    expect(fallbackDocument?.derivation?.parser).toEqual(fallback.provenance);
    expect(fallbackDocument?.chunks[0]?.headings).toEqual([]);
  });

  it("re-indexes unchanged v2 content when parser derivation changes", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const makeParser = (version: string, calls: { value: number }): DocumentParser => ({
      capabilities: ["page-text"],
      provenance: { id: "fixture-parser", version },
      async parse() {
        calls.value += 1;
        return { mediaType: "application/pdf", blocks: [{ kind: "page", text: LONG_ALPHA, locator: { page: 1, block: 0 } }] };
      },
    });
    const firstCalls = { value: 0 };
    const embedding = new FakeEmbedding();
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), parser: makeParser("1", firstCalls), embedding, store, now: () => new Date(NOW) });
    const secondCalls = { value: 0 };
    const embeddingCallsBefore = embedding.calls;
    const summary = await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), parser: makeParser("2", secondCalls), embedding, store, now: () => new Date(NOW) });
    expect(summary.indexed).toBe(1);
    expect(summary.reused).toBe(0);
    expect(secondCalls.value).toBe(1);
    expect(embedding.calls).toBe(embeddingCallsBefore + 1);
    expect((await store.loadPaper("arxiv:2403.19236"))?.derivation?.parser.version).toBe("2");
  });

  it("reuses an unchanged promoted v1 paper without parsing or embedding", async () => {
    const catalog = makeCatalog([{
      paperKey: "arxiv:2403.19236",
      filePaths: ["lib/a.pdf"],
      fingerprint: fingerprint("f1"),
    }]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] });
    const embedding = new FakeEmbedding();
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });
    const stored = await store.loadPaper("arxiv:2403.19236");
    expect(stored).not.toBeNull();
    stored!.derivation = undefined;
    await store.savePaper(stored!);
    const record = store.manifest.papers["arxiv:2403.19236"]!;
    record.derivation = undefined;
    const parser: DocumentParser = {
      capabilities: ["page-text"],
      provenance: { id: "new-parser", version: "9" },
      async parse() { throw new Error("must not parse legacy reuse"); },
    };
    const callsBefore = embedding.calls;
    const summary = await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), parser, embedding, store, now: () => new Date(NOW) });
    expect(summary.reused).toBe(1);
    expect(embedding.calls).toBe(callsBefore);
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

  it("aborts without deleting or downgrading a ready paper when a normal save rejects a future schema", async () => {
    const paperKey = "arxiv:2403.19236";
    const initial = makeCatalog([{ paperKey, filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") }]);
    const store = new MemoryStore();
    const embedding = new FakeEmbedding();
    await indexPersonalLibraryFullText({
      catalog: initial, source: new FakeSource(), extractor: new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] }),
      embedding, store, now: () => new Date(NOW),
    });
    const readyBefore = structuredClone(store.manifest.papers[paperKey]!);
    const originalSave = store.savePaper.bind(store);
    store.savePaper = async (document) => {
      if (document.paperKey === paperKey) {
        throw new FullTextKnowledgeBaseStoreError("future paper schema", "incompatible");
      }
      return originalSave(document);
    };

    await expect(indexPersonalLibraryFullText({
      catalog: makeCatalog([{ paperKey, filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f2") }]),
      source: new FakeSource(), extractor: new FakeExtractor({ "lib/a.pdf": [LONG_BETA] }),
      embedding, store, now: () => new Date(NOW),
    })).rejects.toMatchObject({ code: "incompatible" });

    expect(store.removed.has(paperKey)).toBe(false);
    expect(store.manifest.papers[paperKey]).toEqual(readyBefore);
    expect(store.manifest.papers[paperKey]?.status).toBe("ready");
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
    expect(matches[0]!.rankingScore).toBeCloseTo(2 / 61, 12);
  });

  it("fuses lexical title matches into ranking without extra embedding calls", async () => {
    const catalog = makeCatalog([
      { paperKey: "arxiv:2403.19236", filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") },
      { paperKey: "arxiv:2501.00001", filePaths: ["lib/b.pdf"], fingerprint: fingerprint("f2") },
    ]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA], "lib/b.pdf": [LONG_BETA] });
    // The alpha chunk scores 0.9 against the query; the second paper's title
    // matches the query exactly ("Title of arxiv:2501.00001"), so the lexical
    // fusion must outrank the chunk evidence.
    const alphaChunk = chunkFullText([LONG_ALPHA]).find((chunk) => chunk.text.includes("alpha"))!;
    const betaChunk = chunkFullText([LONG_BETA]).find((chunk) => chunk.text.includes("beta"))!;
    const queryVec = new Float32Array([1, 0, 0, 0]);
    const nearQuery = new Float32Array([0.9, Math.sqrt(1 - 0.9 * 0.9), 0, 0]);
    const unrelated = new Float32Array([0, 1, 0, 0]);
    const embedding = new FakeEmbedding({
      [`query: ${alphaChunk.text}`]: queryVec,
      [`passage: ${alphaChunk.text}`]: nearQuery,
      [`passage: ${betaChunk.text}`]: unrelated,
    });
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });

    const callsBeforeSearch = embedding.calls;
    const matches = await searchFullTextKnowledgeBase({
      store,
      embedding,
      queryText: "Title of arxiv:2501.00001",
      titles: new Map([
        ["arxiv:2403.19236", "Title of arxiv:2403.19236"],
        ["arxiv:2501.00001", "Title of arxiv:2501.00001"],
      ]),
    });

    expect(matches.length).toBe(2);
    expect(matches[0]!.paperKey).toBe("arxiv:2501.00001");
    expect(matches[0]!.score).toBeCloseTo(1, 5);
    expect(matches[0]!.rankingScore).toBeCloseTo(1 / 62 + 1 / 61, 12);
    expect(matches[1]!.paperKey).toBe("arxiv:2403.19236");
    // Lexical fusion adds no extra embedding calls: exactly one for the query.
    expect(embedding.calls).toBe(callsBeforeSearch + 1);
  });

  it("rejects a search before embedding when the knowledge base was built with a different model", async () => {
    const catalog = makeCatalog([
      { paperKey: "arxiv:2403.19236", filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") },
    ]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] });
    const embedding = new FakeEmbedding();
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });

    // Same dimension, different modelId: cross-model cosine is meaningless and
    // the query embedding must not be spent before the guard rejects.
    let embedCalls = 0;
    const otherModel: EmbeddingModel = {
      modelId: "other-model",
      dimension: 4,
      prefixPolicy: "e5",
      async embed(texts: readonly string[]) {
        embedCalls += texts.length;
        return texts.map(() => new Float32Array(4));
      },
    };

    await expect(
      searchFullTextKnowledgeBase({ store, embedding: otherModel, queryText: "anything" }),
    ).rejects.toThrow(/built with model fake-e5-q8/);
    expect(embedCalls).toBe(0);
  });

  it("indexes unresolved files with content-addressed keys and extracted titles", async () => {
    const catalog = makeCatalog(
      [{ paperKey: "arxiv:2403.19236", filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") }],
      [{ path: "lib/local.pdf", fingerprint: fingerprint("f2") }],
    );
    const store = new MemoryStore();
    const extractor = new FakeExtractor({
      "lib/a.pdf": [LONG_ALPHA],
      "lib/local.pdf": ["Attention Local Paper\nAbstract body of the local paper."],
    });
    const embedding = new FakeEmbedding();
    const summary = await indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      extractor,
      embedding,
      store,
      now: () => new Date(NOW),
    });

    expect(summary.indexed).toBe(2);
    const fallbackKey = fallbackPaperKey("lib/local.pdf");
    const manifest = await store.loadManifest();
    expect(manifest.papers[fallbackKey]?.status).toBe("ready");
    expect(manifest.papers[fallbackKey]?.title).toBe("Attention Local Paper");
    expect(manifest.papers[fallbackKey]?.filePaths).toEqual(["lib/local.pdf"]);
    const document = await store.loadPaper(fallbackKey);
    expect(document?.title).toBe("Attention Local Paper");
    expect(document?.chunks.length).toBeGreaterThan(0);
    // arXiv papers keep their catalog title and no extracted title.
    expect(manifest.papers["arxiv:2403.19236"]?.title).toBeUndefined();
  });

  it("reuses unchanged unresolved files and prunes removed ones", async () => {
    const makeCatalogWithLocal = (present: boolean) => makeCatalog(
      [{ paperKey: "arxiv:2403.19236", filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") }],
      present ? [{ path: "lib/local.pdf", fingerprint: fingerprint("f2") }] : [],
    );
    const run = async (catalog: PersonalLibraryCatalog, store: MemoryStore) => {
      const extractor = new FakeExtractor({
        "lib/a.pdf": [LONG_ALPHA],
        "lib/local.pdf": ["Attention Local Paper\nAbstract body."],
      });
      return indexPersonalLibraryFullText({
        catalog,
        source: new FakeSource(),
        extractor,
        embedding: new FakeEmbedding(),
        store,
        now: () => new Date(NOW),
      });
    };
    const store = new MemoryStore();
    await run(makeCatalogWithLocal(true), store);
    // Unchanged file reuses; nothing new indexed.
    const second = await run(makeCatalogWithLocal(true), store);
    expect(second.indexed).toBe(0);
    expect(second.reused).toBe(2);
    // Removed file prunes its fallback document.
    const third = await run(makeCatalogWithLocal(false), store);
    expect(third.pruned).toBe(1);
    const manifest = await store.loadManifest();
    expect(manifest.papers[fallbackPaperKey("lib/local.pdf")]).toBeUndefined();
  });

  it("migrates legacy observation-key fallback documents without re-embedding", async () => {
    const path = "lib/legacy.pdf";
    const pdfBytes = "legacy-pdf-bytes";
    const observation = fingerprint("c3");
    const legacyKey = `file:${observation}`;
    const contentKey = fallbackPaperKey(pdfBytes);
    const source = new FakeSource({ [path]: pdfBytes });
    const store = new MemoryStore();
    const legacyDocument: FullTextPaperDocument = {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      paperKey: legacyKey,
      modelId: "fake-e5-q8",
      dimension: 4,
      textHash: fingerprint("d4"),
      title: "Legacy Local Paper",
      titleVersion: TITLE_EXTRACTION_VERSION,
      filePaths: [path],
      observationFingerprints: [observation],
      chunks: [{ index: 0, page: 1, text: "legacy chunk" }],
      vectors: new Float32Array([1, 2, 3, 4]),
      updatedAt: NOW,
    };
    await store.savePaper(legacyDocument);
    await store.replaceManifest({
      ...(await store.loadManifest()),
      papers: { [legacyKey]: {
        paperKey: legacyKey,
        status: "ready",
        modelId: legacyDocument.modelId,
        dimension: legacyDocument.dimension,
        textHash: legacyDocument.textHash,
        title: legacyDocument.title,
        titleVersion: legacyDocument.titleVersion,
        filePaths: legacyDocument.filePaths,
        observationFingerprints: legacyDocument.observationFingerprints,
        chunkCount: legacyDocument.chunks.length,
        updatedAt: NOW,
      } },
    }, 0);
    const extractor = new FakeExtractor({ [pdfBytes]: ["should not extract"] });
    const embedding = new FakeEmbedding();

    const summary = await indexPersonalLibraryFullText({
      catalog: makeCatalog([], [{ path, fingerprint: observation }]),
      source,
      extractor,
      embedding,
      store,
      now: () => new Date(NOW),
    });

    expect(summary).toMatchObject({ indexed: 0, reused: 1, failed: 0, pruned: 0 });
    expect(extractor.calls).toBe(0);
    expect(embedding.calls).toBe(0);
    const manifest = await store.loadManifest();
    expect(manifest.papers[legacyKey]).toBeUndefined();
    expect(manifest.papers[contentKey]).toMatchObject({
      status: "ready",
      contentHash: `sha256:${sha256Hex(pdfBytes)}`,
      filePaths: [path],
      observationFingerprints: [observation],
    });
    expect(Array.from((await store.loadPaper(contentKey))!.vectors)).toEqual([1, 2, 3, 4]);
  });

  it("aborts without deleting or downgrading a fallback when rebind save rejects a future schema", async () => {
    const oldPath = "lib/original.pdf";
    const newPath = "lib/renamed.pdf";
    const bytes = "same-future-protected-pdf";
    const paperKey = fallbackPaperKey(bytes);
    const source = new FakeSource({ [oldPath]: bytes, [newPath]: bytes });
    const store = new MemoryStore();
    const embedding = new FakeEmbedding();
    await indexPersonalLibraryFullText({
      catalog: makeCatalog([], [{ path: oldPath, fingerprint: fingerprint("a1") }]), source,
      extractor: new FakeExtractor({ [bytes]: [LONG_ALPHA] }), embedding, store, now: () => new Date(NOW),
    });
    const readyBefore = structuredClone(store.manifest.papers[paperKey]!);
    store.savePaper = async () => {
      throw new FullTextKnowledgeBaseStoreError("future fallback schema", "incompatible");
    };

    await expect(indexPersonalLibraryFullText({
      catalog: makeCatalog([], [{ path: newPath, fingerprint: fingerprint("a2") }]), source,
      extractor: new FakeExtractor({ [bytes]: [LONG_ALPHA] }), embedding, store, now: () => new Date(NOW),
    })).rejects.toMatchObject({ code: "incompatible" });

    expect(store.removed.has(paperKey)).toBe(false);
    expect(store.manifest.papers[paperKey]).toEqual(readyBefore);
    expect(store.manifest.papers[paperKey]?.status).toBe("ready");
  });

  it("does not content-reuse a v2 fallback when parser derivation changes", async () => {
    const path = "lib/local.pdf";
    const pdfBytes = "stable-fallback-bytes";
    const source = new FakeSource({ [path]: pdfBytes });
    const store = new MemoryStore();
    const calls = { first: 0, second: 0 };
    const parser = (version: string, key: keyof typeof calls): DocumentParser => ({
      capabilities: ["page-text"],
      provenance: { id: "fallback-parser", version },
      async parse() {
        calls[key] += 1;
        return { mediaType: "application/pdf", blocks: [{ kind: "page", text: LONG_ALPHA, locator: { page: 1, block: 0 } }] };
      },
    });
    const embedding = new FakeEmbedding();
    await indexPersonalLibraryFullText({
      catalog: makeCatalog([], [{ path, fingerprint: fingerprint("a1") }]),
      source,
      parser: parser("1", "first"),
      embedding,
      store,
      now: () => new Date(NOW),
    });
    const before = embedding.calls;
    const summary = await indexPersonalLibraryFullText({
      catalog: makeCatalog([], [{ path, fingerprint: fingerprint("a2") }]),
      source,
      parser: parser("2", "second"),
      embedding,
      store,
      now: () => new Date(NOW),
    });
    expect(summary).toMatchObject({ indexed: 1, reused: 0 });
    expect(calls.second).toBe(1);
    expect(embedding.calls).toBe(before + 1);
  });

  it("does not reuse a v2 migration source when parser derivation changes", async () => {
    const path = "lib/legacy.pdf";
    const bytes = "migration-pdf-bytes";
    const observation = fingerprint("c1");
    const legacyKey = `file:${observation}`;
    const source = new FakeSource({ [path]: bytes });
    const store = new MemoryStore();
    const legacyDocument: FullTextPaperDocument = {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      paperKey: legacyKey,
      modelId: "fake-e5-q8",
      dimension: 4,
      textHash: fingerprint("c2"),
      titleVersion: TITLE_EXTRACTION_VERSION,
      filePaths: [path],
      observationFingerprints: [observation],
      derivation: { parser: { id: "old-parser", version: "1" }, chunkerVersion: 2, embeddingInputVersion: 1 },
      chunks: [{ index: 0, page: 1, text: "legacy chunk" }],
      vectors: new Float32Array([1, 2, 3, 4]),
      updatedAt: NOW,
    };
    await store.savePaper(legacyDocument);
    await store.replaceManifest({
      ...(await store.loadManifest()),
      papers: { [legacyKey]: {
        paperKey: legacyKey, status: "ready", modelId: legacyDocument.modelId,
        dimension: legacyDocument.dimension, textHash: legacyDocument.textHash,
        filePaths: legacyDocument.filePaths, observationFingerprints: legacyDocument.observationFingerprints,
        derivation: legacyDocument.derivation, chunkCount: 1, titleVersion: TITLE_EXTRACTION_VERSION, updatedAt: NOW,
      } },
    }, 0);
    let parseCalls = 0;
    const parser: DocumentParser = {
      capabilities: ["page-text"],
      provenance: { id: "new-parser", version: "2" },
      async parse() {
        parseCalls += 1;
        return { mediaType: "application/pdf", blocks: [{ kind: "page", text: LONG_BETA, locator: { page: 1, block: 0 } }] };
      },
    };
    const embedding = new FakeEmbedding();
    const summary = await indexPersonalLibraryFullText({
      catalog: makeCatalog([], [{ path, fingerprint: observation }]), source, parser, embedding, store,
      now: () => new Date(NOW),
    });
    expect(summary).toMatchObject({ indexed: 1, reused: 0 });
    expect(parseCalls).toBe(1);
    expect(embedding.calls).toBe(1);
  });

  it("keeps a fallback paper key and vectors when the PDF is renamed", async () => {
    const oldPath = "lib/old-name.pdf";
    const newPath = "lib/renamed.pdf";
    const pdfBytes = "same-pdf-bytes";
    const source = new FakeSource({ [oldPath]: pdfBytes, [newPath]: pdfBytes });
    const store = new MemoryStore();
    const firstExtractor = new FakeExtractor({
      [pdfBytes]: ["Stable Local Paper Title\nAbstract body."],
    });
    const firstEmbedding = new FakeEmbedding();
    const first = await indexPersonalLibraryFullText({
      catalog: makeCatalog([], [{ path: oldPath, fingerprint: fingerprint("a1") }]),
      source,
      extractor: firstExtractor,
      embedding: firstEmbedding,
      store,
      now: () => new Date(NOW),
    });
    const paperKey = fallbackPaperKey(pdfBytes);
    expect(first.indexed).toBe(1);
    expect((await store.loadManifest()).papers[paperKey]?.filePaths).toEqual([oldPath]);

    const secondExtractor = new FakeExtractor({
      [pdfBytes]: ["Stable Local Paper Title\nAbstract body."],
    });
    const secondEmbedding = new FakeEmbedding();
    const second = await indexPersonalLibraryFullText({
      catalog: makeCatalog([], [{ path: newPath, fingerprint: fingerprint("b2") }]),
      source,
      extractor: secondExtractor,
      embedding: secondEmbedding,
      store,
      now: () => new Date(NOW),
    });

    expect(second.indexed).toBe(0);
    expect(second.reused).toBe(1);
    expect(secondExtractor.calls).toBe(0);
    expect(secondEmbedding.calls).toBe(0);
    expect((await store.loadManifest()).papers[paperKey]?.filePaths).toEqual([newPath]);
    expect((await store.loadPaper(paperKey))?.filePaths).toEqual([newPath]);
  });

  it("fails only the affected fallback paper when its stored document is corrupt", async () => {
    const oldPath = "lib/old-name.pdf";
    const newPath = "lib/renamed.pdf";
    const pdfBytes = "same-pdf-bytes";
    const source = new FakeSource({ [oldPath]: pdfBytes, [newPath]: pdfBytes });
    const store = new MemoryStore();
    const first = await indexPersonalLibraryFullText({
      catalog: makeCatalog([], [{ path: oldPath, fingerprint: fingerprint("a1") }]),
      source,
      extractor: new FakeExtractor({ [pdfBytes]: ["Stable Local Paper Title\nAbstract body."] }),
      embedding: new FakeEmbedding(),
      store,
      now: () => new Date(NOW),
    });
    const paperKey = fallbackPaperKey(pdfBytes);
    expect(first.indexed).toBe(1);

    // The stored document for the fallback paper is now corrupt.
    const originalLoadPaper = store.loadPaper.bind(store);
    store.loadPaper = async (key: string) => {
      if (key === paperKey) throw new Error("corrupt-or-unreadable");
      return originalLoadPaper(key);
    };

    const second = await indexPersonalLibraryFullText({
      catalog: makeCatalog(
        [{ paperKey: "arxiv:2403.19236", filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") }],
        [{ path: newPath, fingerprint: fingerprint("b2") }],
      ),
      source,
      extractor: new FakeExtractor({ "lib/a.pdf": [LONG_ALPHA] }),
      embedding: new FakeEmbedding(),
      store,
      now: () => new Date(NOW),
    });

    // The corrupt fallback paper fails alone; the arXiv paper still indexes.
    expect(second.failed).toBe(1);
    expect(second.indexed).toBe(1);
    const manifest = await store.loadManifest();
    expect(manifest.papers[paperKey]?.status).toBe("failed");
    expect(manifest.papers["arxiv:2403.19236"]?.status).toBe("ready");
    // The overwritten ready document leaves no orphan behind.
    expect(store.removed.has(paperKey)).toBe(true);
  });

  it("skips a corrupt ready document during search instead of failing the query", async () => {
    const catalog = makeCatalog([
      { paperKey: "arxiv:2403.19236", filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") },
    ], [
      { path: "lib/local.pdf", fingerprint: fingerprint("f2") },
    ]);
    const store = new MemoryStore();
    const extractor = new FakeExtractor({
      "lib/a.pdf": [LONG_ALPHA],
      "lib/local.pdf": ["Local Paper Title\nAbstract body of the local paper."],
    });
    const embedding = new FakeEmbedding();
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });
    const paperKey = fallbackPaperKey("lib/local.pdf");

    const originalLoadPaper = store.loadPaper.bind(store);
    store.loadPaper = async (key: string) => {
      if (key === paperKey) throw new Error("corrupt-or-unreadable");
      return originalLoadPaper(key);
    };

    const matches = await searchFullTextKnowledgeBase({
      store,
      embedding,
      queryText: "alpha",
    });
    expect(matches.length).toBe(1);
    expect(matches[0]!.paperKey).toBe("arxiv:2403.19236");
  });

  it("ranks a literal token hit above misleading chunk similarity", async () => {
    const catalog = makeCatalog(
      [
        { paperKey: "arxiv:2403.19236", filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") },
        { paperKey: "arxiv:2501.00001", filePaths: ["lib/b.pdf"], fingerprint: fingerprint("f2") },
        { paperKey: "arxiv:2601.00001", filePaths: ["lib/c.pdf"], fingerprint: fingerprint("f3") },
        { paperKey: "arxiv:2601.00002", filePaths: ["lib/d.pdf"], fingerprint: fingerprint("f4") },
      ],
      [],
      { "arxiv:2403.19236": "The Pan-STARRS Survey" },
    );
    const store = new MemoryStore();
    const alphaPage = "Pan-STARRS survey. Pan-STARRS data. Pan-STARRS imaging.";
    const alphaChunk = chunkFullText([alphaPage]).find((chunk) => chunk.text.includes("Pan-STARRS"))!;
    const betaChunk = chunkFullText([LONG_BETA]).find((chunk) => chunk.text.includes("beta"))!;
    const extractor = new FakeExtractor({
      "lib/a.pdf": [alphaPage],
      "lib/b.pdf": [LONG_BETA],
      "lib/c.pdf": [LONG_ALPHA],
      "lib/d.pdf": [LONG_ALPHA],
    });
    // Only the alpha paper literally contains the query token; the beta chunk
    // scores 0.9 against the query, so the lexical hit must win. Four papers
    // keep the token's document frequency under the common-word cutoff.
    const queryVec = new Float32Array([1, 0, 0, 0]);
    const nearQuery = new Float32Array([0.9, Math.sqrt(1 - 0.9 * 0.9), 0, 0]);
    const unrelated = new Float32Array([0, 1, 0, 0]);
    const embedding = new FakeEmbedding({
      [`query: panstarrs`]: queryVec,
      [`passage: ${alphaChunk.text}`]: unrelated,
      [`passage: ${betaChunk.text}`]: nearQuery,
    });
    await indexPersonalLibraryFullText({ catalog, source: new FakeSource(), extractor, embedding, store, now: () => new Date(NOW) });

    const matches = await searchFullTextKnowledgeBase({
      store,
      embedding,
      queryText: "panstarrs",
      titles: new Map([
        ["arxiv:2403.19236", "The Pan-STARRS Survey"],
        ["arxiv:2501.00001", "Title of arxiv:2501.00001"],
        ["arxiv:2601.00001", "Title of arxiv:2601.00001"],
        ["arxiv:2601.00002", "Title of arxiv:2601.00002"],
      ]),
    });

    expect(matches[0]!.paperKey).toBe("arxiv:2403.19236");
    expect(matches[0]!.scoreKind).toBe("cosine");
    expect(matches[0]!.rankingScore).toBeCloseTo(1 / 62 + 1 / 61, 12);
    expect(matches[1]!.paperKey).toBe("arxiv:2501.00001");
  });

  it("ignores body-token fusion for long title+abstract queries so vector ranking wins", async () => {
    const catalog = makeCatalog(
      [
        { paperKey: "arxiv:2101.00001", filePaths: ["lib/theme.pdf"], fingerprint: fingerprint("a1") },
        { paperKey: "arxiv:2101.00002", filePaths: ["lib/token.pdf"], fingerprint: fingerprint("a2") },
        { paperKey: "arxiv:2101.00003", filePaths: ["lib/c.pdf"], fingerprint: fingerprint("a3") },
        { paperKey: "arxiv:2101.00004", filePaths: ["lib/d.pdf"], fingerprint: fingerprint("a4") },
      ],
      [],
    );
    const store = new MemoryStore();
    const themePage = "Theme-aligned methods and results for deep surveys of galaxies.";
    const tokenPage = "This paper mentions galaxies once while discussing unrelated instrumentation.";
    const themeChunk = chunkFullText([themePage])[0]!;
    const tokenChunk = chunkFullText([tokenPage])[0]!;
    const fillerChunk = chunkFullText([LONG_ALPHA])[0]!;
    const extractor = new FakeExtractor({
      "lib/theme.pdf": [themePage],
      "lib/token.pdf": [tokenPage],
      "lib/c.pdf": [LONG_ALPHA],
      "lib/d.pdf": [LONG_ALPHA],
    });
    const longQuery = [
      "Deep galaxy surveys with wide-field imaging",
      "",
      "We present a study of galaxies, surveys, imaging, photometry, and redshift",
      "measurements across a wide field. Methods include calibration, catalogs,",
      "and multi-band photometry of galaxies in deep fields.",
    ].join("\n");
    const queryVec = new Float32Array([1, 0, 0, 0]);
    const themeVec = new Float32Array([0.95, Math.sqrt(1 - 0.95 * 0.95), 0, 0]);
    const weakVec = new Float32Array([0.2, Math.sqrt(1 - 0.2 * 0.2), 0, 0]);
    const embedding = new FakeEmbedding({
      [`query: ${longQuery}`]: queryVec,
      [`passage: ${themeChunk.text}`]: themeVec,
      [`passage: ${tokenChunk.text}`]: weakVec,
      [`passage: ${fillerChunk.text}`]: weakVec,
    });
    await indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      extractor,
      embedding,
      store,
      now: () => new Date(NOW),
    });

    const matches = await searchFullTextKnowledgeBase({
      store,
      embedding,
      queryText: longQuery,
      titles: new Map([
        ["arxiv:2101.00001", "Theme paper"],
        ["arxiv:2101.00002", "Token paper"],
        ["arxiv:2101.00003", "Filler C"],
        ["arxiv:2101.00004", "Filler D"],
      ]),
    });

    expect(matches[0]!.paperKey).toBe("arxiv:2101.00001");
    expect(matches[0]!.score).toBeGreaterThan(matches.find((match) => match.paperKey === "arxiv:2101.00002")!.score);
  });

  it("scores titles against the first paragraph of a title+abstract query", async () => {
    const catalog = makeCatalog(
      [
        { paperKey: "arxiv:2102.00001", filePaths: ["lib/target.pdf"], fingerprint: fingerprint("b1") },
        { paperKey: "arxiv:2102.00002", filePaths: ["lib/other.pdf"], fingerprint: fingerprint("b2") },
        { paperKey: "arxiv:2102.00003", filePaths: ["lib/c.pdf"], fingerprint: fingerprint("b3") },
        { paperKey: "arxiv:2102.00004", filePaths: ["lib/d.pdf"], fingerprint: fingerprint("b4") },
      ],
      [],
    );
    const store = new MemoryStore();
    const extractor = new FakeExtractor({
      "lib/target.pdf": [LONG_ALPHA],
      "lib/other.pdf": [LONG_BETA],
      "lib/c.pdf": [LONG_ALPHA],
      "lib/d.pdf": [LONG_ALPHA],
    });
    const queryText = "The Pan-STARRS1 Surveys\n\nWe describe the surveys, data products, and calibration.";
    const queryVec = new Float32Array([0, 1, 0, 0]);
    const weakVec = new Float32Array([1, 0, 0, 0]);
    const alphaChunk = chunkFullText([LONG_ALPHA])[0]!;
    const betaChunk = chunkFullText([LONG_BETA])[0]!;
    const embedding = new FakeEmbedding({
      [`query: ${queryText}`]: queryVec,
      [`passage: ${alphaChunk.text}`]: weakVec,
      [`passage: ${betaChunk.text}`]: weakVec,
    });
    await indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      extractor,
      embedding,
      store,
      now: () => new Date(NOW),
    });

    const matches = await searchFullTextKnowledgeBase({
      store,
      embedding,
      queryText,
      titles: new Map([
        ["arxiv:2102.00001", "The Pan-STARRS1 Surveys"],
        ["arxiv:2102.00002", "Unrelated Instrumentation Paper"],
        ["arxiv:2102.00003", "Filler C"],
        ["arxiv:2102.00004", "Filler D"],
      ]),
    });

    expect(matches[0]!.paperKey).toBe("arxiv:2102.00001");
    expect(matches[0]!.rankingScore).toBeCloseTo(1 / 61, 12);
  });

  it("refreshes fallback titles when the extraction rules advanced", async () => {
    const catalog = makeCatalog(
      [],
      [{ path: "lib/local.pdf", fingerprint: fingerprint("f2") }],
    );
    const store = new MemoryStore();
    const run = async (firstPage: string) => indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      extractor: new FakeExtractor({ "lib/local.pdf": [`${firstPage}\nAbstract body.`] }),
      embedding: new FakeEmbedding(),
      store,
      now: () => new Date(NOW),
    });
    const fallbackKey = fallbackPaperKey("lib/local.pdf");

    await run("Old Title Line");
    expect((await store.loadManifest()).papers[fallbackKey]?.title).toBe("Old Title Line");

    // Simulate a knowledge base indexed before the version field existed.
    const manifest = await store.loadManifest();
    const record = { ...manifest.papers[fallbackKey]! };
    delete record.titleVersion;
    await store.replaceManifest(
      { ...manifest, papers: { ...manifest.papers, [fallbackKey]: record } },
      manifest.revision,
    );

    // The same file re-runs; the title refresh re-reads the first page.
    const second = await run("New Title Line");
    expect(second.titlesRefreshed).toBe(1);
    expect(second.reused).toBe(1);
    const refreshed = await store.loadPaper(fallbackKey);
    expect(refreshed?.title).toBe("New Title Line");
    expect(refreshed?.derivation).toBeDefined();
    expect((await store.loadManifest()).papers[fallbackKey]?.derivation).toEqual(refreshed?.derivation);
    expect(refreshed?.chunks.length).toBeGreaterThan(0);

    // A further run is a plain reuse (no re-read, no refresh).
    const third = await run("New Title Line");
    expect(third.titlesRefreshed).toBe(0);
    expect(third.reused).toBe(1);
  });

  it("defaults to hybrid and supports dense/lexical diagnostic modes", async () => {
    const catalog = makeCatalog([
      { paperKey: "arxiv:2403.19236", filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") },
      { paperKey: "arxiv:2501.00001", filePaths: ["lib/b.pdf"], fingerprint: fingerprint("f2") },
    ]);
    const store = new MemoryStore();
    const exactPage = "rarelexeme rarelexeme rarelexeme";
    const semanticPage = "semantic neighboring concept";
    const exactChunk = chunkFullText([exactPage])[0]!;
    const semanticChunk = chunkFullText([semanticPage])[0]!;
    const query = new Float32Array([1, 0, 0, 0]);
    const embedding = new FakeEmbedding({
      "query: rarelexeme": query,
      [`passage: ${exactChunk.text}`]: new Float32Array([0, 1, 0, 0]),
      [`passage: ${semanticChunk.text}`]: query,
    });
    await indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      extractor: new FakeExtractor({ "lib/a.pdf": [exactPage], "lib/b.pdf": [semanticPage] }),
      embedding,
      store,
      now: () => new Date(NOW),
    });

    const dense = await searchFullTextKnowledgeBase({ store, embedding, queryText: "rarelexeme", mode: "dense" });
    const callsBeforeLexical = embedding.calls;
    const lexical = await searchFullTextKnowledgeBase({ store, embedding, queryText: "rarelexeme", mode: "lexical" });
    expect(embedding.calls).toBe(callsBeforeLexical);
    const hybrid = await searchFullTextKnowledgeBase({ store, embedding, queryText: "rarelexeme" });
    expect(dense[0]!.paperKey).toBe("arxiv:2501.00001");
    expect(lexical[0]!.paperKey).toBe("arxiv:2403.19236");
    expect(hybrid[0]!.paperKey).toBe("arxiv:2403.19236");
  });

  it("uses explicit lexicalQueryText while dense embeds the complete title and abstract", async () => {
    const catalog = makeCatalog([
      { paperKey: "arxiv:2403.19236", filePaths: ["lib/a.pdf"], fingerprint: fingerprint("f1") },
      { paperKey: "arxiv:2501.00001", filePaths: ["lib/b.pdf"], fingerprint: fingerprint("f2") },
    ]);
    const store = new MemoryStore();
    const lexicalPage = "laterterm laterterm evidence";
    const semanticPage = `semantic target ${"context ".repeat(30)}`;
    const lexicalChunk = chunkFullText([lexicalPage])[0]!;
    const semanticChunk = chunkFullText([semanticPage])[0]!;
    const queryText = `Unrelated title\n\n${"background ".repeat(30)}laterterm`;
    const queryVector = new Float32Array([1, 0, 0, 0]);
    const embedding = new FakeEmbedding({
      [`query: ${queryText}`]: queryVector,
      [`passage: ${lexicalChunk.text}`]: new Float32Array([0, 1, 0, 0]),
      [`passage: ${semanticChunk.text}`]: queryVector,
    });
    await indexPersonalLibraryFullText({
      catalog,
      source: new FakeSource(),
      extractor: new FakeExtractor({ "lib/a.pdf": [lexicalPage], "lib/b.pdf": [semanticPage] }),
      embedding,
      store,
      now: () => new Date(NOW),
    });

    const ordinary = await searchFullTextKnowledgeBase({ store, embedding, queryText });
    const findSimilar = await searchFullTextKnowledgeBase({
      store,
      embedding,
      queryText,
      lexicalQueryText: "Unrelated title",
    });
    expect(ordinary[0]!.paperKey).toBe("arxiv:2403.19236");
    expect(findSimilar[0]!.paperKey).toBe("arxiv:2501.00001");
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
