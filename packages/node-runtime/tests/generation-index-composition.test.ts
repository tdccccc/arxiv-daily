import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  DEFAULT_SETTINGS,
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  FullTextGenerationIndexStore,
  FullTextKnowledgeBaseFileStore,
  createEvidenceChunkId,
  searchFullTextKnowledgeBase,
  synchronizeFullTextGenerationIndex,
  type EmbeddingModel,
  type FullTextPaperDocument,
  type FullTextPaperKnowledgeRecord,
} from "@arxiv-daily/core";
import { NodeStorageAdapter } from "../src/index";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const TEXT_HASH = `sha256:${"c".repeat(64)}`;
const OBSERVATION = `sha256:${"d".repeat(64)}`;
const NOW = "2026-08-18T00:00:00.000Z";
const DERIVATION = {
  parser: { id: "node-composition-parser", version: "1" },
  chunkerVersion: 2,
  embeddingInputVersion: 1,
} as const;
const tempDirs: string[] = [];

afterEach(async () => {
  vi.restoreAllMocks();
  while (tempDirs.length > 0) {
    const directory = tempDirs.pop();
    if (directory) await rm(directory, { recursive: true, force: true });
  }
});

async function makeTempDir(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "arxiv-daily-generation-"));
  tempDirs.push(directory);
  return directory;
}

function paper(
  paperKey: string,
  text: string,
  vector: readonly [number, number],
): FullTextPaperDocument {
  const identity = {
    text,
    headings: ["Methods"],
    locator: { pageStart: 2 },
    derivation: DERIVATION,
  };
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey,
    modelId: "node-composition-model",
    dimension: 2,
    textHash: TEXT_HASH,
    title: `Title ${paperKey}`,
    filePaths: [`library/${paperKey.replaceAll(":", "-")}.pdf`],
    observationFingerprints: [OBSERVATION],
    derivation: DERIVATION,
    chunks: [{
      id: createEvidenceChunkId(identity),
      index: 0,
      page: 2,
      ...identity,
    }],
    vectors: new Float32Array(vector),
    updatedAt: NOW,
  };
}

function readyRecord(document: FullTextPaperDocument): FullTextPaperKnowledgeRecord {
  return {
    paperKey: document.paperKey,
    status: "ready",
    modelId: document.modelId,
    dimension: document.dimension,
    textHash: document.textHash,
    title: document.title,
    filePaths: [...document.filePaths],
    observationFingerprints: [...document.observationFingerprints],
    derivation: document.derivation,
    chunkCount: document.chunks.length,
    updatedAt: document.updatedAt,
  };
}

function embedding(queryVector: readonly [number, number]): EmbeddingModel {
  return {
    modelId: "node-composition-model",
    dimension: 2,
    prefixPolicy: "none",
    embed: vi.fn(async (texts) => texts.map(() => new Float32Array(queryVector))),
  };
}

describe("Node full-text generation composition", () => {
  it.skipIf(process.platform !== "linux")(
    "persists a committed legacy library through spool, promotion, and generation search",
    async () => {
      const root = await makeTempDir();
      const storage = new NodeStorageAdapter(root);
      expect(storage.createTextExclusive).toBeTypeOf("function");

      const source = new FullTextKnowledgeBaseFileStore(
        storage,
        DEFAULT_SETTINGS.output,
        SCOPE,
        IDENTIFICATION,
        { now: () => new Date(NOW) },
      );
      const documents = [
        paper("paper:alpha", "alpha telescope survey", [1, 0]),
        paper("paper:beta", "unrelated chemistry", [0, 1]),
      ];
      for (const document of documents) await source.savePaper(document);
      const initial = await source.loadManifest();
      const committed = await source.replaceManifest({
        ...initial,
        modelId: "node-composition-model",
        dimension: 2,
        papers: Object.fromEntries(
          documents.map((document) => [document.paperKey, readyRecord(document)]),
        ),
      }, initial.revision);
      expect(committed).toMatchObject({ revision: 1, modelId: "node-composition-model" });

      const generations = new FullTextGenerationIndexStore(
        storage,
        DEFAULT_SETTINGS.output,
        SCOPE,
        IDENTIFICATION,
      );
      const writeBinary = vi.spyOn(storage, "writeBinary");
      const sourcePaperLoads = vi.spyOn(source, "loadPaper");
      const synchronized = await synchronizeFullTextGenerationIndex({
        sourceStore: source,
        generationStore: generations,
        storage,
        output: DEFAULT_SETTINGS.output,
        scopeFingerprint: SCOPE,
        identificationFingerprint: IDENTIFICATION,
        writerToken: `node-composition-${"e".repeat(48)}`,
      });

      expect(synchronized).toMatchObject({
        kind: "rebuilt",
        sourceRevision: committed.revision,
        indexedPaperCount: 2,
        chunkCount: 2,
      });
      expect(sourcePaperLoads).toHaveBeenCalledTimes(2);
      const binaryPaths = writeBinary.mock.calls.map(([path]) => path);
      const spoolObjectPath = binaryPaths.find((path) => path.includes("/spool/"));
      expect(spoolObjectPath).toContain("/objects/");
      expect(binaryPaths.some((path) => path.includes(
        `/generations/${synchronized.generationId}/objects/`,
      ))).toBe(true);
      const spoolAttemptDirectory = spoolObjectPath?.split("/objects/")[0];
      expect(spoolAttemptDirectory).toBeTruthy();
      await expect(storage.exists(spoolAttemptDirectory!)).resolves.toBe(false);
      await expect(storage.exists(generations.paths.currentPath)).resolves.toBe(true);

      const reopenedStorage = new NodeStorageAdapter(root);
      const reopenedSource = new FullTextKnowledgeBaseFileStore(
        reopenedStorage,
        DEFAULT_SETTINGS.output,
        SCOPE,
        IDENTIFICATION,
      );
      const reopenedGenerations = new FullTextGenerationIndexStore(
        reopenedStorage,
        DEFAULT_SETTINGS.output,
        SCOPE,
        IDENTIFICATION,
      );
      await expect(reopenedGenerations.openCurrent()).resolves.toMatchObject({
        descriptor: {
          generationId: synchronized.generationId,
          sourceRevision: committed.revision,
        },
      });

      const legacyPaperLoads = vi.spyOn(reopenedSource, "loadPaper");
      const model = embedding([1, 0]);
      for (const mode of ["dense", "lexical", "hybrid"] as const) {
        const matches = await searchFullTextKnowledgeBase({
          store: reopenedSource,
          generationStore: reopenedGenerations,
          embedding: model,
          queryText: "alpha telescope",
          mode,
          limit: 2,
        });
        expect(matches[0]).toMatchObject({ paperKey: "paper:alpha" });
      }
      expect(legacyPaperLoads).not.toHaveBeenCalled();
      expect(model.embed).toHaveBeenCalledTimes(2);
    },
  );
});
