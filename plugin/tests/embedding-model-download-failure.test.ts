import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * A network failure while downloading the model (config/tokenizer/ONNX
 * weights, on the first `pipeline()` call) must not surface as whatever raw
 * fetch/DNS error transformers.js happened to throw — it should read as a
 * network problem downloading the embedding model, and be recognizable by
 * `isEmbeddingModelDownloadNetworkError` for the completion notice that
 * reports it (see plugin/src/library/index-completion.ts).
 */
const pipelineMock = vi.fn();
vi.mock("@huggingface/transformers", () => ({
  env: {},
  pipeline: (...args: unknown[]) => pipelineMock(...args),
}));

const {
  createTransformersEmbeddingModel,
  isEmbeddingModelDownloadNetworkError,
} = await import("../src/hosts/obsidian/embedding-model");

afterEach(() => {
  pipelineMock.mockReset();
});

describe("embedding model download failure classification", () => {
  it("reports a browser-style fetch failure as a clear, actionable message", async () => {
    pipelineMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    const model = createTransformersEmbeddingModel();
    await expect(model.embed(["hello"])).rejects.toThrow(
      /Couldn't download the embedding model \(network problem\)/,
    );
  });

  it("reports a Node-style fetch failure as the same clear message", async () => {
    pipelineMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    const model = createTransformersEmbeddingModel();
    await expect(model.embed(["hello"])).rejects.toThrow(
      /Couldn't download the embedding model \(network problem\)/,
    );
  });

  it("looks one level into `cause` for a wrapped DNS failure", async () => {
    const dnsFailure = new Error("getaddrinfo ENOTFOUND huggingface.co");
    pipelineMock.mockRejectedValueOnce(new TypeError("fetch failed", { cause: dnsFailure }));
    const model = createTransformersEmbeddingModel();
    await expect(model.embed(["hello"])).rejects.toThrow(
      /Couldn't download the embedding model \(network problem\)/,
    );
  });

  it("points a reader at the developer console for the underlying error", async () => {
    pipelineMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    const model = createTransformersEmbeddingModel();
    await expect(model.embed(["hello"])).rejects.toThrow(/developer console/);
  });

  it("is recognized by isEmbeddingModelDownloadNetworkError once stringified to a message", async () => {
    pipelineMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    const model = createTransformersEmbeddingModel();
    const failure = await model.embed(["hello"]).catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(Error);
    expect(isEmbeddingModelDownloadNetworkError((failure as Error).message)).toBe(true);
  });

  it("leaves an unrelated load failure alone, unclassified as a network problem", async () => {
    pipelineMock.mockRejectedValueOnce(new Error("dtype q8 is not supported for this model"));
    const model = createTransformersEmbeddingModel();
    const failure = await model.embed(["hello"]).catch((error: unknown) => error);
    expect((failure as Error).message).toBe("dtype q8 is not supported for this model");
    expect(isEmbeddingModelDownloadNetworkError((failure as Error).message)).toBe(false);
  });

  it("treats undefined as not a network-download failure", () => {
    expect(isEmbeddingModelDownloadNetworkError(undefined)).toBe(false);
  });
});
