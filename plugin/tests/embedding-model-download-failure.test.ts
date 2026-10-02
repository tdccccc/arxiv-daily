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

function featureExtractor() {
  return vi.fn(async (texts: string | string[]) => ({
    dims: [typeof texts === "string" ? 1 : texts.length, 384],
    data: new Float32Array((typeof texts === "string" ? 1 : texts.length) * 384),
  }));
}

describe("local model preparation progress", () => {
  it("reports loading and file progress, then ready before embedding starts", async () => {
    const onProgress = vi.fn();
    const extractor = featureExtractor();
    pipelineMock.mockImplementationOnce(async (_task, _repo, options) => {
      // Transformers emits 'download' even when reading browser cache, so the
      // callback cannot truthfully distinguish this from a network transfer.
      options.progress_callback?.({ status: "download", name: "model", file: "weights.onnx" });
      options.progress_callback?.({ status: "progress", name: "model", file: "weights.onnx", progress: 25, loaded: 25, total: 100 });
      return extractor;
    });
    const model = createTransformersEmbeddingModel({ onProgress });
    expect(onProgress).not.toHaveBeenCalled();

    await model.embed(["hello"]);

    expect(onProgress.mock.calls[0]?.[0]).toMatchObject({ phase: "loading" });
    expect(onProgress.mock.calls.some(([event]) => event.phase === "loading" && event.progress === 25)).toBe(true);
    expect(onProgress.mock.calls.at(-1)?.[0]).toMatchObject({ phase: "ready" });
    expect(onProgress.mock.calls.some(([event]) => event.phase === "downloading")).toBe(false);
    expect(onProgress.mock.calls.some(([event]) => /first use.*130 MB/i.test(event.message))).toBe(true);
    expect(onProgress.mock.invocationCallOrder.at(-1)).toBeLessThan(extractor.mock.invocationCallOrder[1]!);

    onProgress.mockClear();
    await model.embed(["second call"]);
    expect(pipelineMock).toHaveBeenCalledTimes(1);
    expect(onProgress.mock.calls.map(([event]) => event.phase)).toEqual(["ready"]);
  });

  it.each(["factory", "embed"] as const)("stops notices after %s cancellation while loading continues", async (signalOwner) => {
    let finishLoad!: () => void;
    let report!: (event: Record<string, unknown>) => void;
    const extractor = featureExtractor();
    pipelineMock.mockImplementationOnce((_task, _repo, options) => {
      report = options.progress_callback;
      return new Promise((resolve) => { finishLoad = () => resolve(extractor); });
    });
    const controller = new AbortController();
    const onProgress = vi.fn();
    const model = createTransformersEmbeddingModel({
      signal: signalOwner === "factory" ? controller.signal : undefined,
      onProgress,
    });
    const pending = model.embed(["hello"], signalOwner === "embed" ? { signal: controller.signal } : undefined);
    const rejection = expect(pending).rejects.toMatchObject({ name: "AbortError" });
    await vi.waitFor(() => expect(pipelineMock).toHaveBeenCalledTimes(1));
    expect(onProgress).toHaveBeenCalledWith(expect.objectContaining({ phase: "loading" }));
    expect(report).toBeTypeOf("function");
    controller.abort("cancelled by user");
    await rejection;
    const callsAtCancellation = onProgress.mock.calls.length;

    report?.({ status: "progress", name: "model", file: "weights.onnx", progress: 75, loaded: 75, total: 100 });
    finishLoad();
    await vi.waitFor(() => expect(extractor).toHaveBeenCalledTimes(1));

    expect(onProgress.mock.calls.length).toBe(callsAtCancellation);
    expect(onProgress.mock.calls.some(([event]) => event.phase === "ready")).toBe(false);
    if (signalOwner === "embed") {
      onProgress.mockClear();
      await model.embed(["retry"]);
      expect(pipelineMock).toHaveBeenCalledTimes(1);
      expect(onProgress.mock.calls.map(([event]) => event.phase)).toEqual(["ready"]);
    }
  });

  it("does not announce ready after model loading fails", async () => {
    const onProgress = vi.fn();
    pipelineMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    await expect(createTransformersEmbeddingModel({ onProgress }).embed(["hello"])).rejects.toThrow(/download/);
    expect(onProgress).toHaveBeenCalledWith(expect.objectContaining({ phase: "loading" }));
    expect(onProgress.mock.calls.some(([event]) => event.phase === "ready")).toBe(false);
  });
});
