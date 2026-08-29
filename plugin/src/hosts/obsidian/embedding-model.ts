/**
 * Obsidian host implementation of core's `EmbeddingModel` port, backed by
 * transformers.js (`feature-extraction` with Xenova/multilingual-e5-small,
 * q8-quantized, CPU).
 *
 * Host boundaries: core owns the embedding policy (query/passage prefixes via
 * `applyEmbeddingPrefix`) and the vector contract; this file supplies the
 * local inference engine. Model download, quantization loading, and ONNX
 * Runtime inference all happen here, never in core, and model files never
 * enter the vault: transformers.js caches them in the browser Cache API
 * (Obsidian renderer) or a file-system cache dir (Node hosts).
 *
 * Loading strategy: everything is lazy. The transformers.js module is only
 * imported on the first `embed()` call (dynamic import; esbuild bundles it
 * into main.js at build time), the model is downloaded on first use (HF Hub
 * by default, mirror-configurable via `env.remoteHost`), and the ONNX
 * session is created once and reused. The transformers.js `env` singleton is
 * configured inside this factory, before the first pipeline load, and only
 * when an option requires it.
 *
 * Device selection: transformers.js v4 accepts `device: "cpu"` only on real
 * Node (onnxruntime-node); in web/Electron-renderer environments the wasm
 * backend is the CPU execution provider ("wasm" device). The same probe
 * transformers.js uses internally (`process.release.name === "node"`) decides
 * which side we are on. One wrinkle: Obsidian's renderer runs with Node.js
 * integration, so `process.release.name` is "node" there even though the
 * runtime is a Chromium renderer — that would push transformers.js into its
 * Node branch, which the plugin's web bundle does not contain (the
 * onnxruntime-node module is tree-shaken out), crashing with
 * `Cannot read properties of undefined (reading 'create')` at module init.
 * Electron renderers expose `process.versions.electron`; when present, the
 * factory aligns the probe to "electron" first (see
 * `alignElectronReleaseProbe`), keeping transformers.js and `isNodeRuntime()`
 * on the same wasm/browser branch.
 *
 * Cancellation: transformers.js has no AbortSignal plumbing, so embeds are
 * batched (bounded memory, abort granularity) with the caller's signal
 * checked between batches, and every await is raced against the signal,
 * converting aborts into `AbortError`-named errors that core's
 * `isCancellationError` recognizes (same convention as pdf-text-extractor).
 * An aborted load keeps running in the background (the download/cache
 * completes and the next embed reuses it); only the caller sees the abort.
 */

import type {
  EmbeddingModel,
  EmbeddingOptions,
} from "@arxiv-daily/core";
import type { FeatureExtractionPipeline } from "@huggingface/transformers";

/** Stable model identifier shared with core's knowledge-base manifest. */
export const EMBEDDING_MODEL_ID = "multilingual-e5-small-q8";
/**
 * Transformers.js model repository. Xenova's ONNX export ships
 * `onnx/model_quantized.onnx` (q8), which is what `dtype: "q8"` resolves to.
 */
const TRANSFORMERS_MODEL_REPO = "Xenova/multilingual-e5-small";
/** Embedding width of multilingual-e5-small; asserted against the loaded model. */
const EXPECTED_DIMENSION = 384;
/** Texts per inference call: bounds q8-session memory and gives abort granularity. */
const EMBED_BATCH_SIZE = 8;

export interface TransformersEmbeddingModelOptions {
  /**
   * Hugging Face mirror base URL, e.g. `"https://hf-mirror.com"`. Applied to
   * transformers.js `env.remoteHost` before the first model load; the default
   * path template `{model}/resolve/{revision}/` is kept. Only model files go
   * through this host — the ONNX Runtime wasm binaries are fetched from
   * transformers.js' own CDN default (`env.backends.onnx.wasm.wasmPaths`) and
   * cached via the Cache API.
   */
  huggingfaceMirror?: string;
  /**
   * Abort signal for the lazy model load, and the default signal for
   * `embed()` calls that do not pass their own.
   */
  signal?: AbortSignal;
  /**
   * File-system cache directory override (transformers.js `env.cacheDir`).
   * Only meaningful on Node/CLI hosts; the Obsidian renderer caches via the
   * Cache API by default and must not set this.
   */
  cacheDir?: string;
}

/** The module namespace of `@huggingface/transformers` (types only). */
type TransformersModule = typeof import("@huggingface/transformers");
/** Environment object type (the `env` export's type is not re-exported at root). */
type TransformersEnv = TransformersModule["env"];
/** Output tensor of the feature-extraction pipeline. */
type PipelineOutputTensor = Awaited<ReturnType<FeatureExtractionPipeline["_call"]>>;

/**
 * Create the local embedding model. The factory is cheap: all heavy work is
 * deferred to the first `embed()` call (module import, model download,
 * session creation, dimension probe).
 */
export function createTransformersEmbeddingModel(
  options?: TransformersEmbeddingModelOptions,
): EmbeddingModel {
  alignElectronReleaseProbe();
  const loader = new LazyModelLoader(options);
  return {
    modelId: EMBEDDING_MODEL_ID,
    dimension: EXPECTED_DIMENSION,
    prefixPolicy: "e5",
    async embed(
      texts: readonly string[],
      embedOptions?: EmbeddingOptions,
    ): Promise<readonly Float32Array[]> {
      const signal = embedOptions?.signal ?? options?.signal;
      if (signal?.aborted) throw abortError(signal);
      if (texts.length === 0) return [];

      const extractor = await loader.ensure(signal);
      const vectors: Float32Array[] = [];
      for (let start = 0; start < texts.length; start += EMBED_BATCH_SIZE) {
        if (signal?.aborted) throw abortError(signal);
        const batch = texts.slice(start, start + EMBED_BATCH_SIZE);
        const output = await raceWithAbort(
          extractor(batch, { pooling: "mean", normalize: true }),
          signal,
        );
        vectors.push(...tensorToVectors(output, EXPECTED_DIMENSION));
        // Yield between inference batches so a long index run does not
        // freeze the host UI (the Obsidian renderer processes queued events
        // between batches); harmless on Node hosts.
        await yieldToEventLoop();
      }
      return vectors;
    },
  };
}

/**
 * One-time, memoized load of the transformers.js module and the ONNX
 * feature-extraction session, with dimension assertion.
 */
class LazyModelLoader {
  private modulePromise: Promise<TransformersModule> | null = null;
  private pipelinePromise: Promise<FeatureExtractionPipeline> | null = null;

  constructor(private readonly options?: TransformersEmbeddingModelOptions) {}

  /**
   * Resolve the pipeline, racing the (possibly in-flight) load against the
   * signal. The underlying load is never cancelled — aborting only rejects
   * this caller — so a later embed reuses the completed load.
   */
  ensure(signal?: AbortSignal): Promise<FeatureExtractionPipeline> {
    const loading = this.pipelinePromise ?? this.startLoad();
    return signal ? raceWithAbort(loading, signal) : loading;
  }

  private startLoad(): Promise<FeatureExtractionPipeline> {
    this.pipelinePromise = this.load();
    return this.pipelinePromise;
  }

  private async load(): Promise<FeatureExtractionPipeline> {
    const transformers = await this.loadModule();
    configureTransformersEnv(transformers.env, this.options);

    const device = isNodeRuntime() ? "cpu" : "wasm";
    const extractor = await transformers.pipeline("feature-extraction", TRANSFORMERS_MODEL_REPO, {
      dtype: "q8",
      device,
    });

    // Probe the loaded model and assert the documented dimension so the port
    // contract (`dimension === 384`) is verified against the real model
    // output once, up front, with a descriptive error on mismatch.
    const probe = await extractor("dimension probe", { pooling: "mean", normalize: true });
    const dims = probe.dims;
    if (dims.length !== 2 || dims[1] !== EXPECTED_DIMENSION) {
      throw new Error(
        `Embedding model ${EMBEDDING_MODEL_ID} produced dimension ` +
          `[${dims.join(", ")}], expected [*, ${EXPECTED_DIMENSION}]. ` +
          "The knowledge base embedding model and the host model must agree; " +
          "refusing to serve inconsistent vectors.",
      );
    }
    return extractor;
  }

  private loadModule(): Promise<TransformersModule> {
    if (!this.modulePromise) {
      // Lazy, deferred import: nothing from transformers.js runs at plugin
      // startup. esbuild rewrites this to a bundled require at build time.
      this.modulePromise = import("@huggingface/transformers");
    }
    return this.modulePromise;
  }
}

/**
 * Apply factory options to the transformers.js `env` singleton. Must run
 * before the first pipeline load; `env` is global, so this is done once per
 * process in practice (documented: configuring after a model is already
 * loaded has no effect on the loaded session).
 */
function configureTransformersEnv(
  env: TransformersEnv,
  options: TransformersEmbeddingModelOptions | undefined,
): void {
  if (!options) return;
  const mirror = options.huggingfaceMirror;
  if (mirror !== undefined && mirror !== "") {
    env.remoteHost = mirror.endsWith("/") ? mirror : `${mirror}/`;
    // env.remotePathTemplate keeps the default "{model}/resolve/{revision}/".
  }
  if (options.cacheDir !== undefined) {
    env.cacheDir = options.cacheDir;
  }
}

/** Read-only facts about the transformers.js `env` singleton. */
export interface TransformersEnvFacts {
  remoteHost?: string;
  wasmPaths?: string;
  cacheDir?: string;
}

/**
 * Inspect the transformers.js `env` singleton for diagnostics (the
 * `diagnose-fulltext-runtime` command). Safe to call anytime: it only reads
 * the env after a lazy module import; a failed model load still leaves the
 * configured values readable. The ONNX wasm paths default is populated by
 * the backend on initialization, so read them after a load attempt.
 * Returns null when the module itself cannot be imported.
 */
export async function inspectTransformersEnv(): Promise<TransformersEnvFacts | null> {
  try {
    const transformers = await import("@huggingface/transformers");
    const env = transformers.env as unknown as {
      remoteHost?: string;
      cacheDir?: string;
      backends?: { onnx?: { wasm?: { wasmPaths?: string | Record<string, string> } } };
    };
    const wasmPaths = env.backends?.onnx?.wasm?.wasmPaths;
    return {
      remoteHost: env.remoteHost,
      wasmPaths:
        typeof wasmPaths === "string" ? wasmPaths : wasmPaths ? JSON.stringify(wasmPaths) : undefined,
      cacheDir: env.cacheDir,
    };
  } catch {
    return null;
  }
}

/**
 * Same runtime probe transformers.js uses internally: a Node.js process has
 * `process.release.name === "node"`; the Obsidian renderer exposes `process`
 * too (Node.js integration), but with `process.versions.electron` present —
 * see `alignElectronReleaseProbe`, which makes the release name "electron"
 * so this resolves to the web/wasm path.
 */
function isNodeRuntime(): boolean {
  return (
    typeof process !== "undefined"
    && typeof process.release === "object"
    && process.release !== null
    && process.release.name === "node"
  );
}

/**
 * Electron renderers run with Node.js integration, so `process.release.name`
 * is "node" there even though the runtime is a Chromium renderer. Without
 * alignment, transformers.js v4 selects its Node branch (onnxruntime-node),
 * which the plugin's web bundle does not contain (tree-shaken), and module
 * init crashes with `Cannot read properties of undefined (reading 'create')`
 * (`ONNX.InferenceSession` is undefined). Electron renderers expose
 * `process.versions.electron`; when present, report the release name as
 * "electron" — semantically accurate — so transformers.js and
 * `isNodeRuntime()` both take the wasm/browser branch. Idempotent; must run
 * before the transformers.js module initializes (the factory calls it before
 * the lazy import). Real Node processes have no `versions.electron` and are
 * left untouched.
 */
export function alignElectronReleaseProbe(): void {
  if (typeof process === "undefined") return;
  const versions = (process as { versions?: { electron?: unknown } }).versions;
  if (versions?.electron === undefined) return;
  const release = process.release;
  if (release?.name === "electron") return;
  try {
    Object.defineProperty(process, "release", {
      value: { ...release, name: "electron" },
      configurable: true,
    });
  } catch {
    // Non-configurable release (unexpected in Electron renderers): leave the
    // probe as-is; the wasm branch cannot be forced through this path.
  }
}

/**
 * Human-readable runtime probe for the `diagnose-fulltext-runtime` report:
 * which branch transformers.js will take (`process.release.name`) and whether
 * the Electron marker is present.
 */
export function describeRuntimeProbe(): string {
  if (typeof process === "undefined") return "no process (plain browser)";
  const versions = (process as { versions?: { electron?: unknown } }).versions;
  const electron = versions?.electron as string | undefined;
  const isolated = typeof window !== "undefined"
    && typeof (window as { crossOriginIsolated?: unknown }).crossOriginIsolated === "boolean"
    ? String((window as { crossOriginIsolated?: unknown }).crossOriginIsolated)
    : "n/a";
  return electron !== undefined
    ? `process.release.name=${process.release?.name ?? "?"} (electron ${electron}, crossOriginIsolated=${isolated})`
    : `process.release.name=${process.release?.name ?? "?"} (node)`;
}

/** Let queued host events run (timers, clicks, renders) before continuing. */
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    const schedule = typeof window !== "undefined"
      ? window.setTimeout.bind(window)
      : setTimeout;
    if (typeof schedule === "function") schedule(resolve, 0);
    else resolve();
  });
}

/**
 * Copy the pipeline output rows into fresh Float32Array instances. The copy
 * is required: transformers.js may reuse the underlying tensor buffer across
 * calls, so callers must not retain views into it.
 */function tensorToVectors(
  tensor: PipelineOutputTensor,
  expectedDimension: number,
): Float32Array[] {
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
  // transformers.js types the tensor data loosely; treat it as a flat numeric
  // sequence (Float32Array on onnxruntime-node, number[] elsewhere) and copy
  // rows into fresh Float32Array instances. The copy is required: the runtime
  // may reuse the underlying buffer across calls.
  const data = tensor.data as unknown as ArrayLike<number>;
  const vectors = new Array<Float32Array>(rows);
  for (let i = 0; i < rows; i++) {
    const offset = i * expectedDimension;
    const row = new Float32Array(expectedDimension);
    for (let j = 0; j < expectedDimension; j++) {
      row[j] = data[offset + j] as number;
    }
    vectors[i] = row;
  }
  return vectors;
}

/** Race a promise against the abort signal (mirrors pdf-text-extractor). */
function raceWithAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal));
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        reject(error instanceof Error ? error : new Error(String(error)));
      },
    );
  });
}

/**
 * Mirror core's cancellation convention (`isCancellationError` matches on
 * `name === "AbortError"`): an Error named "AbortError" carrying the signal
 * reason as message.
 */
function abortError(signal: AbortSignal): Error {
  const reason = (signal as { reason?: unknown }).reason;
  const message =
    typeof reason === "string" && reason ? reason : "cancelled by user";
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}
