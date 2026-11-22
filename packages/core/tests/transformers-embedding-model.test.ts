import { describe, expect, it, vi } from "vitest";
import {
  createTransformersEmbeddingModelFromLoader,
  type TransformersFeatureExtractor,
} from "../src/library/fulltext/transformers-embedding-model";

describe("shared local embedding factory", () => {
  it("uses a host loader and preserves caller-applied prefixes while cancelling between batches", async () => {
    const controller = new AbortController();
    const extractor = vi.fn<TransformersFeatureExtractor>(async input => ({
      dims: [typeof input === "string" ? 1 : input.length, 384],
      data: new Float32Array(8 * 384),
    }));
    const loadPipeline = vi.fn(async () => extractor);
    const yieldBetweenBatches = vi.fn(async () => { controller.abort("stop after batch"); });
    const model = createTransformersEmbeddingModelFromLoader({ loadPipeline, yieldBetweenBatches });
    expect(loadPipeline).not.toHaveBeenCalled();
    const texts = Array.from({ length: 9 }, (_, i) => `passage: document ${i}`);
    await expect(model.embed(texts, { signal: controller.signal }))
      .rejects.toMatchObject({ name: "AbortError", message: "stop after batch" });
    expect(loadPipeline).toHaveBeenCalledOnce();
    expect(extractor.mock.calls.map(call => call[0])).toEqual(["dimension probe", texts.slice(0, 8)]);
    expect(yieldBetweenBatches).toHaveBeenCalledOnce();
  });

  it("retries a failed pipeline load on the next embed instead of caching the rejection", async () => {
    const extractor = vi.fn<TransformersFeatureExtractor>(async input => ({
      dims: [typeof input === "string" ? 1 : input.length, 384],
      data: new Float32Array(384),
    }));
    const loadPipeline = vi.fn()
      .mockRejectedValueOnce(new Error("transient download failure"))
      .mockResolvedValueOnce(extractor);
    const model = createTransformersEmbeddingModelFromLoader({ loadPipeline });

    await expect(model.embed(["query: first"]))
      .rejects.toThrow("transient download failure");
    await expect(model.embed(["query: second"])).resolves.toHaveLength(1);

    expect(loadPipeline).toHaveBeenCalledTimes(2);
  });

  it("disposes the loaded pipeline and reloads on the next embed", async () => {
    const dispose = vi.fn(async () => undefined);
    const extractor = vi.fn<TransformersFeatureExtractor>(async input => ({
      dims: [typeof input === "string" ? 1 : input.length, 384],
      data: new Float32Array(384),
    }));
    extractor.dispose = dispose;
    const loadPipeline = vi.fn(async () => extractor);
    const model = createTransformersEmbeddingModelFromLoader({ loadPipeline });

    await model.embed(["query: first"]);
    await model.dispose?.();
    expect(dispose).toHaveBeenCalledOnce();
    await model.embed(["query: second"]);
    expect(loadPipeline).toHaveBeenCalledTimes(2);
  });

  it("rejects incompatible batch output even after a successful dimension probe", async () => {
    const extractor = vi.fn<TransformersFeatureExtractor>()
      .mockResolvedValueOnce({ dims: [1, 384], data: new Float32Array(384) })
      .mockResolvedValueOnce({ dims: [1, 12], data: new Float32Array(12) });
    const model = createTransformersEmbeddingModelFromLoader({ loadPipeline: async () => extractor });
    await expect(model.embed(["query: calibration"]))
      .rejects.toThrow("Embedding model output dimension mismatch");
  });
});
