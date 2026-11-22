import type { EmbeddingModel, EmbeddingOptions } from "./ports";

export const LOCAL_EMBEDDING_MODEL_ID = "multilingual-e5-small-q8";
export const LOCAL_EMBEDDING_MODEL_REPO = "Xenova/multilingual-e5-small";
export const LOCAL_EMBEDDING_DIMENSION = 384;
const EMBED_BATCH_SIZE = 8;

/** The small tensor surface shared by the CPU and WASM feature extractors. */
export interface TransformersEmbeddingTensor {
  readonly dims: readonly number[];
  readonly data: ArrayLike<unknown>;
}

export interface TransformersFeatureExtractor {
  (input: string | string[], options: { pooling: "mean"; normalize: true }): Promise<TransformersEmbeddingTensor>;
  /** Optional engine hook to release the loaded model (transformers.js pipelines have one). */
  dispose?(): Promise<void>;
}

export interface ModelPreparationProgress {
  phase: "loading" | "downloading" | "ready";
  message: string;
  progress?: number;
}

export interface TransformersEmbeddingLoaderOptions {
  /** Preparation updates only for callers still waiting for model readiness. */
  onProgress?: (progress: ModelPreparationProgress) => void;
  /** Host loads the q8 feature-extraction pipeline, including engine and model assets. */
  loadPipeline(report: (progress: ModelPreparationProgress) => void): Promise<TransformersFeatureExtractor>;
  /** Default caller cancellation; the underlying model load remains reusable. */
  signal?: AbortSignal;
  /** Host scheduling hook, e.g. yielding renderer events between inference batches. */
  yieldBetweenBatches?(): Promise<void>;
}

/**
 * Shared inference contract with an injected engine loader. The host retains
 * all runtime selection, downloads and cache configuration. Core supplies the
 * same model identity, dimension probe, ordered batches and owned output rows.
 */
export function createTransformersEmbeddingModelFromLoader(
  options: TransformersEmbeddingLoaderOptions,
): EmbeddingModel {
  let pipelinePromise: Promise<TransformersFeatureExtractor> | null = null;

  let ready = false;
  /** Embeds that already hold the current extractor; dispose() waits for them. */
  let activeEmbeds = 0;
  const idleWaiters = new Set<() => void>();
  const listeners = new Set<(progress: ModelPreparationProgress) => void>();

  function startLoad(): Promise<TransformersFeatureExtractor> {
    const loading = (async () => {
      const extractor = await options.loadPipeline(event => {
        for (const report of listeners) report(event);
      });
      const probe = await extractor("dimension probe", { pooling: "mean", normalize: true });
      const dims = probe.dims;
      if (dims.length !== 2 || dims[1] !== LOCAL_EMBEDDING_DIMENSION) {
        throw new Error(
          `Embedding model ${LOCAL_EMBEDDING_MODEL_ID} produced dimension ` +
            `[${dims.join(", ")}], expected [*, ${LOCAL_EMBEDDING_DIMENSION}]. ` +
            "The knowledge base embedding model and the host model must agree; " +
            "refusing to serve inconsistent vectors.",
        );
      }
      return extractor;
    })();
    // Published before readiness is set: a dispose() that lands while the
    // pipeline is still loading clears the current promise first. Only the
    // load that is still current may report the model ready; a stale load's
    // extractor is left for dispose() to release.
    const tracked = loading.then((extractor) => {
      if (pipelinePromise === tracked) ready = true;
      return extractor;
    });
    pipelinePromise = tracked;
    return tracked;
  }

  async function waitForIdleEmbeds(): Promise<void> {
    if (activeEmbeds === 0) return;
    await new Promise<void>((resolve) => idleWaiters.add(resolve));
  }

  return {
    modelId: LOCAL_EMBEDDING_MODEL_ID,
    dimension: LOCAL_EMBEDDING_DIMENSION,
    prefixPolicy: "e5",
    async embed(texts: readonly string[], embedOptions?: EmbeddingOptions): Promise<readonly Float32Array[]> {
      const signal = embedOptions?.signal ?? options.signal;
      if (signal?.aborted) throw abortError(signal);
      if (texts.length === 0) return [];

      const report = (event: ModelPreparationProgress) => {
        if (!signal?.aborted && !options.signal?.aborted) options.onProgress?.(event);
      };
      listeners.add(report);
      let extractor: TransformersFeatureExtractor;
      try {
        if (!ready) report({
          phase: "loading",
          message: "Loading local model files; first use may download about 130 MB.",
        });
        const loading = pipelinePromise ?? startLoad();
        // A failed load is not a permanent property of this model: clear the
        // cached promise so a later embed retries (a transient network failure
        // must not poison every future search). Only the load's own rejection
        // clears it — a caller abort leaves the load running for reuse.
        loading.catch(() => {
          if (pipelinePromise === loading) pipelinePromise = null;
        });
        extractor = await raceWithAbort(loading, signal);
        // dispose() may have released the model while this call was waiting.
        // Using the destroyed extractor would read a freed session, and the
        // next embed would skip the "loading" phase for a model it no longer
        // has. The stale load's extractor is dispose()'s to release.
        if (pipelinePromise !== loading) throw releasedError();
        report({ phase: "ready", message: "Local model ready; extracting and embedding titles and abstracts." });
      } finally {
        listeners.delete(report);
      }
      activeEmbeds += 1;
      try {
        const vectors: Float32Array[] = [];
        for (let start = 0; start < texts.length; start += EMBED_BATCH_SIZE) {
          if (signal?.aborted) throw abortError(signal);
          const output = await raceWithAbort(
            extractor(texts.slice(start, start + EMBED_BATCH_SIZE), { pooling: "mean", normalize: true }),
            signal,
          );
          vectors.push(...tensorToVectors(output, LOCAL_EMBEDDING_DIMENSION));
          await options.yieldBetweenBatches?.();
        }
        return vectors;
      } finally {
        activeEmbeds -= 1;
        if (activeEmbeds === 0) {
          for (const resolve of idleWaiters) resolve();
          idleWaiters.clear();
        }
      }
    },
    /** Release the loaded engine so a host that stops using it can free it. */
    async dispose(): Promise<void> {
      const loading = pipelinePromise;
      pipelinePromise = null;
      ready = false;
      // An embed that already holds the extractor is allowed to finish; only
      // then is the engine released under it.
      await waitForIdleEmbeds();
      if (!loading) return;
      try {
        const extractor = await loading;
        await extractor.dispose?.();
      } catch {
        // A load that never completed holds nothing to release.
      }
    },
  };
}

/** Copy rows: an inference engine may reuse the tensor's underlying storage. */
function tensorToVectors(tensor: TransformersEmbeddingTensor, expectedDimension: number): Float32Array[] {
  const dims = tensor.dims;
  if (dims.length !== 2 || dims[1] !== expectedDimension) {
    throw new Error(
      `Embedding model output dimension mismatch: expected [*, ${expectedDimension}], ` +
        `got [${dims.join(", ")}]. The knowledge base must be rebuilt with the ` +
        "model whose vectors are stored.",
    );
  }
  const rows = dims[0];
  if (rows === undefined || !Number.isInteger(rows)) {
    throw new Error("Embedding model returned an invalid batch output");
  }
  const vectors = new Array<Float32Array>(rows);
  for (let i = 0; i < rows; i++) {
    const offset = i * expectedDimension;
    const row = new Float32Array(expectedDimension);
    for (let j = 0; j < expectedDimension; j++) row[j] = tensor.data[offset + j] as number;
    vectors[i] = row;
  }
  return vectors;
}

/** Cancellation affects the caller; an in-flight engine load continues. */
function raceWithAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal));
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      value => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      error => {
        signal.removeEventListener("abort", onAbort);
        reject(error instanceof Error ? error : new Error(String(error)));
      },
    );
  });
}

function abortError(signal: AbortSignal): Error {
  const reason = signal.reason as unknown;
  const error = new Error(typeof reason === "string" && reason ? reason : "cancelled by user");
  error.name = "AbortError";
  return error;
}

/** The model was released while this caller was waiting for its pipeline. */
function releasedError(): Error {
  const error = new Error(
    "The local embedding model was released while this request was loading; retry the request.",
  );
  error.name = "EmbeddingModelReleasedError";
  return error;
}
