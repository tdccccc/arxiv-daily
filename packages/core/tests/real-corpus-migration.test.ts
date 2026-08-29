/**
 * Opt-in check that a real legacy knowledge base migrates to a generation index
 * and answers identically through both retrieval paths.
 *
 * Fixtures cannot stand in for this: the routing capacity defect that blocked
 * every real library was invisible to three-word chunks, and only appeared once
 * a corpus carried real vocabulary density.
 *
 *   REAL_KB_DIR=<.../personal-library-knowledge-base/<scope>/<ident>> \
 *     npm test -- tests/real-corpus-migration.test.ts
 *
 * The source directory is copied into a temporary root and never written to.
 */
import { cp, mkdir, rm } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_SETTINGS,
  FullTextGenerationIndexStore,
  FullTextKnowledgeBaseFileStore,
  MAX_GENERATION_OBJECTS,
  searchFullTextKnowledgeBase,
  synchronizeFullTextGenerationIndex,
  type EmbeddingModel,
} from "../src/index";
import { NodeStorageAdapter } from "../../node-runtime/src/index";

const SOURCE_KB = process.env.REAL_KB_DIR;
const EMBED_URL = process.env.REAL_EMBED_URL ?? "http://127.0.0.1:11434/v1/embeddings";
const EMBED_MODEL = process.env.REAL_EMBED_MODEL ?? "nomic-embed-text";

function embedding(modelId: string, dimension: number): EmbeddingModel {
  return {
    modelId,
    dimension,
    prefixPolicy: "none",
    async embed(texts) {
      const response = await fetch(EMBED_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: EMBED_MODEL, input: [...texts] }),
      });
      const json = (await response.json()) as { data: { embedding: number[] }[] };
      return json.data.map((row) => Float32Array.from(row.embedding));
    },
  };
}

describe.skipIf(!SOURCE_KB)("a real legacy knowledge base migrates to a generation index", () => {
  it("rebuilds from the committed source and ranks identically to the legacy path", async () => {
    const manifest = JSON.parse(readFileSync(join(SOURCE_KB as string, "manifest.json"), "utf8")) as {
      schemaVersion: number; revision: number; scopeFingerprint: string;
      identificationFingerprint: string; modelId: string; dimension: number;
      papers: Record<string, unknown>;
    };
    // eslint-disable-next-line no-console
    console.log(`source: schema v${manifest.schemaVersion} rev${manifest.revision} ${manifest.modelId} papers=${Object.keys(manifest.papers).length}`);

    // A fixed root, cleared up front: a timeout kills the test before its
    // cleanup runs, and a fresh temp directory each time would leak gigabytes.
    const root = join(tmpdir(), "real-corpus-migration");
    await rm(root, { recursive: true, force: true });
    await mkdir(root, { recursive: true });
    try {
      const storage = new NodeStorageAdapter(root);
      const target = join(
        root, "arxiv-daily/.index/personal-library-knowledge-base",
        manifest.scopeFingerprint.replace("sha256:", ""),
        manifest.identificationFingerprint.replace("sha256:", ""),
      );
      await cp(SOURCE_KB as string, target, { recursive: true });

      const args = [storage, DEFAULT_SETTINGS.output, manifest.scopeFingerprint, manifest.identificationFingerprint] as const;
      const source = new FullTextKnowledgeBaseFileStore(...args);
      const generations = new FullTextGenerationIndexStore(...args);

      const startedAt = Date.now();
      const synchronized = await synchronizeFullTextGenerationIndex({
        sourceStore: source, generationStore: generations, storage,
        output: DEFAULT_SETTINGS.output,
        scopeFingerprint: manifest.scopeFingerprint,
        identificationFingerprint: manifest.identificationFingerprint,
        writerToken: `real-migration-${"f".repeat(48)}`,
      });
      const opened = await generations.openCurrent();
      const routeRefs = (opened?.descriptor.lexicalRouting ?? []).reduce((total, route) => total + route.length, 0);
      // eslint-disable-next-line no-console
      console.log(
        `migrated ${synchronized.kind}: papers=${synchronized.indexedPaperCount} chunks=${synchronized.chunkCount} ` +
        `objects=${opened?.descriptor.objects.length}/${MAX_GENERATION_OBJECTS} routeRefs=${routeRefs} ` +
        `in ${((Date.now() - startedAt) / 1000).toFixed(0)}s`,
      );

      expect(synchronized.kind).toBe("rebuilt");
      expect(synchronized.chunkCount).toBeGreaterThan(0);
      // The defect this guards: routing outgrew the object budget long before
      // the objects themselves did.
      expect(routeRefs).toBeGreaterThan(opened!.descriptor.objects.length);

      const model = embedding(manifest.modelId, manifest.dimension);
      for (const queryText of [
        "galaxy morphology classification with deep learning",
        "stellar population synthesis and star formation history",
      ]) {
        for (const mode of ["dense", "lexical", "hybrid"] as const) {
          const viaGeneration = await searchFullTextKnowledgeBase({
            store: source, generationStore: generations, embedding: model, queryText, mode, limit: 5,
          });
          const viaLegacy = await searchFullTextKnowledgeBase({
            store: source, embedding: model, queryText, mode, limit: 5,
          });
          // eslint-disable-next-line no-console
          console.log(`[${mode}] ${queryText.slice(0, 42)}… generation=${viaGeneration.length} legacy=${viaLegacy.length}`);
          expect(viaGeneration.length).toBeGreaterThan(0);
          expect(viaGeneration.map((match) => match.paperKey)).toEqual(viaLegacy.map((match) => match.paperKey));
          for (const match of viaGeneration) expect(match.hits.length).toBeGreaterThan(0);
        }
      }
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }, 10_800_000);
});
