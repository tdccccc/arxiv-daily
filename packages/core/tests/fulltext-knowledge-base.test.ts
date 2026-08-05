import { describe, expect, it } from "vitest";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  createFullTextKnowledgeBasePaperPath,
  decodeFullTextKnowledgeBaseManifest,
  decodeFullTextPaperDocument,
  deriveFullTextKnowledgeBasePaths,
  serializeFullTextPaperDocument,
  type FullTextKnowledgeBaseManifest,
  type FullTextPaperDocument,
} from "../src/library/fulltext/knowledge-base";
import { E5_PASSAGE_PREFIX, E5_QUERY_PREFIX, applyEmbeddingPrefix } from "../src/library/fulltext/ports";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const SCOPE_FINGERPRINT = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION_FINGERPRINT = `sha256:${"b".repeat(64)}`;
const TEXT_HASH = `sha256:${"c".repeat(64)}`;

const normalizePath = (path: string) => path;

function makeDocument(): FullTextPaperDocument {
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey: "arxiv:2403.19236",
    modelId: "multilingual-e5-small-q8",
    dimension: 4,
    textHash: TEXT_HASH,
    filePaths: ["library/paper.pdf"],
    observationFingerprints: [SCOPE_FINGERPRINT],
    chunks: [
      { index: 0, page: 1, text: "First chunk" },
      { index: 1, page: 1, text: "Second chunk" },
    ],
    vectors: new Float32Array([0.1, -0.2, 0.3, 0.4, 1.0, 2.0, -3.0, 4.5]),
    updatedAt: "2026-08-05T00:00:00.000Z",
  };
}

describe("embedding prefix policy", () => {
  it("applies the e5 query prefix to queries", () => {
    expect(E5_QUERY_PREFIX).toBe("query: ");
    expect(applyEmbeddingPrefix("query", "graph neural networks"))
      .toBe("query: graph neural networks");
  });

  it("applies the e5 passage prefix to passages", () => {
    expect(E5_PASSAGE_PREFIX).toBe("passage: ");
    expect(applyEmbeddingPrefix("passage", "We propose a new architecture"))
      .toBe("passage: We propose a new architecture");
  });
});

describe("knowledge base path derivation", () => {
  it("shards by scope and identification fingerprint hex", () => {
    const paths = deriveFullTextKnowledgeBasePaths(
      { normalizePath },
      DEFAULT_SETTINGS.output,
      SCOPE_FINGERPRINT,
      IDENTIFICATION_FINGERPRINT,
    );
    expect(paths.directory)
      .toBe(`arxiv-daily/.index/personal-library-knowledge-base/${"a".repeat(64)}/${"b".repeat(64)}`);
    expect(paths.manifest.documentPath)
      .toBe(`arxiv-daily/.index/personal-library-knowledge-base/${"a".repeat(64)}/${"b".repeat(64)}/manifest.json`);
    expect(paths.manifest.backupPath)
      .toBe(`arxiv-daily/.index/personal-library-knowledge-base/${"a".repeat(64)}/${"b".repeat(64)}/manifest.json.backup`);
    expect(paths.papersDirectory)
      .toBe(`arxiv-daily/.index/personal-library-knowledge-base/${"a".repeat(64)}/${"b".repeat(64)}/papers`);
  });

  it("rejects fingerprints that are not SHA-256", () => {
    expect(() => deriveFullTextKnowledgeBasePaths(
      { normalizePath },
      DEFAULT_SETTINGS.output,
      "sha256:not-hex",
      IDENTIFICATION_FINGERPRINT,
    )).toThrow(/must be a SHA-256 fingerprint/);
  });

  it("names per-paper files by sha256 of the paper key", () => {
    const paths = deriveFullTextKnowledgeBasePaths(
      { normalizePath },
      DEFAULT_SETTINGS.output,
      SCOPE_FINGERPRINT,
      IDENTIFICATION_FINGERPRINT,
    );
    const path = createFullTextKnowledgeBasePaperPath({ normalizePath }, paths, "arxiv:2403.19236");
    expect(path).toMatch(/^arxiv-daily\/\.index\/personal-library-knowledge-base\/[a-f0-9]{64}\/[a-f0-9]{64}\/papers\/[a-f0-9]{64}\.json$/);
  });
});

describe("full-text paper document serialization", () => {
  it("round-trips chunks and float vectors through base64", () => {
    const document = makeDocument();
    const decoded = decodeFullTextPaperDocument(JSON.parse(serializeFullTextPaperDocument(document)));
    expect(decoded).not.toBeNull();
    expect(decoded!.paperKey).toBe(document.paperKey);
    expect(decoded!.chunks).toEqual(document.chunks);
    expect(Array.from(decoded!.vectors)).toEqual(Array.from(document.vectors));
    expect(decoded!.vectors).toBeInstanceOf(Float32Array);
  });

  it("rejects a vector payload with the wrong length", () => {
    const value = JSON.parse(serializeFullTextPaperDocument(makeDocument()));
    value.chunks.push({ index: 2, page: 2, text: "Third chunk" });
    expect(decodeFullTextPaperDocument(value)).toBeNull();
  });

  it("rejects non-sequential chunk indices", () => {
    const value = JSON.parse(serializeFullTextPaperDocument(makeDocument()));
    value.chunks[1]!.index = 2;
    expect(decodeFullTextPaperDocument(value)).toBeNull();
  });

  it("rejects corrupt base64 vectors", () => {
    const value = JSON.parse(serializeFullTextPaperDocument(makeDocument()));
    value.vectors.data = "!!!not-base64!!!";
    expect(decodeFullTextPaperDocument(value)).toBeNull();
  });

  it("rejects an unknown schema version", () => {
    const value = JSON.parse(serializeFullTextPaperDocument(makeDocument()));
    value.schemaVersion = 2;
    expect(decodeFullTextPaperDocument(value)).toBeNull();
  });
});

describe("knowledge base manifest decoder", () => {
  function makeManifest(): FullTextKnowledgeBaseManifest {
    return {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      revision: 3,
      scopeFingerprint: SCOPE_FINGERPRINT,
      identificationFingerprint: IDENTIFICATION_FINGERPRINT,
      modelId: "multilingual-e5-small-q8",
      dimension: 384,
      updatedAt: "2026-08-05T00:00:00.000Z",
      papers: {
        "arxiv:2403.19236": {
          paperKey: "arxiv:2403.19236",
          status: "ready",
          modelId: "multilingual-e5-small-q8",
          dimension: 384,
          textHash: TEXT_HASH,
          filePaths: ["library/paper.pdf"],
          observationFingerprints: [SCOPE_FINGERPRINT],
          chunkCount: 21,
          updatedAt: "2026-08-05T00:00:00.000Z",
        },
        "arxiv:2309.11425": {
          paperKey: "arxiv:2309.11425",
          status: "failed",
          modelId: "multilingual-e5-small-q8",
          dimension: 384,
          filePaths: ["library/other.pdf"],
          observationFingerprints: [IDENTIFICATION_FINGERPRINT],
          chunkCount: 0,
          error: "extraction failed",
          updatedAt: "2026-08-05T00:00:00.000Z",
        },
      },
    };
  }

  it("accepts a valid manifest with ready and failed records", () => {
    const manifest = makeManifest();
    const decoded = decodeFullTextKnowledgeBaseManifest(JSON.parse(JSON.stringify(manifest)));
    expect(decoded).not.toBeNull();
    expect(decoded!.revision).toBe(3);
    expect(decoded!.papers["arxiv:2403.19236"]!.status).toBe("ready");
    expect(decoded!.papers["arxiv:2309.11425"]!.status).toBe("failed");
    expect(decoded!.papers["arxiv:2309.11425"]!.error).toBe("extraction failed");
  });

  it("rejects a record key that does not match its paperKey field", () => {
    const manifest = makeManifest();
    manifest.papers["arxiv:2403.19236"]!.paperKey = "arxiv:9999.99999";
    expect(decodeFullTextKnowledgeBaseManifest(JSON.parse(JSON.stringify(manifest)))).toBeNull();
  });

  it("rejects a failed record that carries a textHash", () => {
    const manifest = makeManifest();
    manifest.papers["arxiv:2309.11425"]!.textHash = TEXT_HASH;
    expect(decodeFullTextKnowledgeBaseManifest(JSON.parse(JSON.stringify(manifest)))).toBeNull();
  });

  it("rejects a ready record without a textHash", () => {
    const manifest = makeManifest();
    delete manifest.papers["arxiv:2403.19236"]!.textHash;
    expect(decodeFullTextKnowledgeBaseManifest(JSON.parse(JSON.stringify(manifest)))).toBeNull();
  });

  it("rejects mismatched observation fingerprint count", () => {
    const manifest = makeManifest();
    manifest.papers["arxiv:2403.19236"]!.observationFingerprints = [];
    expect(decodeFullTextKnowledgeBaseManifest(JSON.parse(JSON.stringify(manifest)))).toBeNull();
  });

  it("rejects a non-integer revision", () => {
    const manifest = makeManifest();
    manifest.revision = 1.5;
    expect(decodeFullTextKnowledgeBaseManifest(JSON.parse(JSON.stringify(manifest)))).toBeNull();
  });
});
