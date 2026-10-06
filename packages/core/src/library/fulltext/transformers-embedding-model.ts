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
  const listeners = new Set<(progress: ModelPreparationProgress) => void>();
  async function load(): Promise<TransformersFeatureExtractor> {
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
    ready = true;
    return extractor;
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
        const loading = pipelinePromise ?? (pipelinePromise = load());
        extractor = await raceWithAbort(loading, signal);
        report({ phase: "ready", message: "Local model ready; extracting and embedding titles and abstracts." });
      } finally {
        listeners.delete(report);
      }
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
