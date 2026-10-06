import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTransformersEmbeddingModel } from "../src/hosts/obsidian/embedding-model";

const runtime = vi.hoisted(() => ({
  pipeline: vi.fn(),
  env: { remoteHost: "", cacheDir: "" },
}));

vi.mock("@huggingface/transformers", () => runtime);

beforeEach(() => {
  runtime.pipeline.mockReset();
  runtime.env.remoteHost = "";
  runtime.env.cacheDir = "";
});

describe("local embedding runtime contract", () => {
  it("loads lazily once, probes 384 dimensions, batches eight texts, and copies reused tensors", async () => {
    const buffer = new Float32Array(8 * 384);
    let sequence = 0;
    const extractor = vi.fn(async (input: string | readonly string[], _options: { pooling: "mean"; normalize: true }) => {
      buffer.fill(++sequence);
      return { dims: [typeof input === "string" ? 1 : input.length, 384], data: buffer };
    });
    runtime.pipeline.mockResolvedValue(extractor);
    const model = createTransformersEmbeddingModel({
      huggingfaceMirror: "https://mirror.example",
      cacheDir: "/isolated-model-cache",
    });
    expect(model).toMatchObject({ modelId: "multilingual-e5-small-q8", dimension: 384, prefixPolicy: "e5" });
    expect(await model.embed([])).toEqual([]);
    expect(runtime.pipeline).not.toHaveBeenCalled();
    const texts = Array.from({ length: 9 }, (_, index) => `passage: text ${index}`);
    const vectors = await model.embed(texts);
    expect(runtime.pipeline).toHaveBeenCalledOnce();
    expect(runtime.pipeline).toHaveBeenCalledWith("feature-extraction", "Xenova/multilingual-e5-small", { dtype: "q8", device: "cpu" });
    expect(runtime.env).toEqual({ remoteHost: "https://mirror.example/", cacheDir: "/isolated-model-cache" });
    expect(extractor.mock.calls.map(call => call[0])).toEqual(["dimension probe", texts.slice(0, 8), texts.slice(8)]);
    for (const call of extractor.mock.calls) {
      expect(call[1]).toEqual({ pooling: "mean", normalize: true });
    }
    expect(vectors).toHaveLength(9);
    expect(vectors.every(vector => vector instanceof Float32Array && vector.length === 384)).toBe(true);
    expect(vectors[0]![0]).toBe(2);
    expect(vectors[8]![0]).toBe(3);
    await model.embed(["query: calibration"]);
    expect(runtime.pipeline).toHaveBeenCalledOnce();
    expect(vectors[0]![0]).toBe(2);
    expect(vectors[8]![0]).toBe(3);
  });

  it("rejects a mismatched model dimension before embedding user texts", async () => {
    const extractor = vi.fn(async () => ({ dims: [1, 12], data: new Float32Array(12) }));
    runtime.pipeline.mockResolvedValue(extractor);
    await expect(createTransformersEmbeddingModel().embed(["query: test"]))
      .rejects.toThrow("expected [*, 384]");
    expect(extractor).toHaveBeenCalledOnce();
    expect(extractor).toHaveBeenCalledWith("dimension probe", { pooling: "mean", normalize: true });
  });

  it("cancels a waiting caller while allowing a later call to reuse the completed model load", async () => {
    let complete!: (extractor: (input: string | readonly string[]) => Promise<unknown>) => void;
    runtime.pipeline.mockReturnValue(new Promise(resolve => { complete = resolve; }));
    const model = createTransformersEmbeddingModel();
    const controller = new AbortController();
    const pending = model.embed(["query: cancelled"], { signal: controller.signal });
    const rejected = expect(pending).rejects.toMatchObject({ name: "AbortError", message: "cancel load wait" });
    await vi.waitFor(() => expect(runtime.pipeline).toHaveBeenCalledOnce());
    controller.abort("cancel load wait");
    await rejected;
    complete(async input => ({ dims: [typeof input === "string" ? 1 : input.length, 384], data: new Float32Array(384) }));
    expect(await model.embed(["query: resumed"])).toHaveLength(1);
    expect(runtime.pipeline).toHaveBeenCalledOnce();
  });

  it("does not load the model for a pre-cancelled request", async () => {
    const controller = new AbortController();
    controller.abort("not now");
    await expect(createTransformersEmbeddingModel().embed(["query: test"], { signal: controller.signal }))
      .rejects.toMatchObject({ name: "AbortError", message: "not now" });
    expect(runtime.pipeline).not.toHaveBeenCalled();
  });
});
