/**
 * Remote (OpenAI-compatible) implementation of the `EmbeddingModel` port
 * (ADR 0008): POSTs text batches to `{baseUrl}/embeddings` and turns the
 * returned vectors into Float32Array rows. No local model, no wasm — the
 * speed path for users who opt into full-text chunks leaving the machine.
 *
 * The endpoint contract is the de-facto OpenAI embeddings shape:
 * `POST {baseUrl}/embeddings` with `{ model, input: string[] }` returning
 * `{ data: [{ index, embedding: number[] }] }`. The `dimension` option is
 * asserted against every response so a misconfigured model cannot silently
 * poison the knowledge base.
 */

import type { HttpClient } from "../../core/adapters";
import { redactText } from "../../utils/redaction";
import type { EmbeddingModel, EmbeddingOptions } from "./ports";

export interface RemoteEmbeddingModelOptions {
  /** OpenAI-compatible embeddings endpoint base URL (e.g. `https://api.openai.com/v1`). */
  baseUrl: string;
  apiKey: string;
  /** Model name sent in the request body, e.g. `text-embedding-3-small`. */
  model: string;
  /** Expected vector width; asserted against every response. */
  dimension: number;
  http: HttpClient;
  /** Texts per request: bounds response size and timeout exposure. */
  batchSize?: number;
  /** Per-request timeout. */
  timeoutMs?: number;
}

/**
 * Stable knowledge-base model identity for a remote embedding model: the
 * same model + dimension from any compatible endpoint produces
 * interchangeable vectors, so the identity deliberately excludes the
 * endpoint URL.
 */
export function remoteEmbeddingModelId(model: string, dimension: number): string {
  return `remote:${model}:${dimension}`;
}

export function createRemoteEmbeddingModel(
  options: RemoteEmbeddingModelOptions,
): EmbeddingModel {
  const baseUrl = trimTrailingSlashes(options.baseUrl);
  const apiKey = options.apiKey;
  const model = options.model;
  const dimension = options.dimension;
  const batchSize = options.batchSize ?? 64;
  const timeoutMs = options.timeoutMs ?? 60_000;
  if (!baseUrl || !apiKey || !model) {
    throw new Error("Remote embedding requires an endpoint base URL, API key, and model");
  }
  if (!Number.isInteger(dimension) || dimension <= 0) {
    throw new TypeError(
      `Remote embedding dimension must be a positive integer, got ${JSON.stringify(dimension)}`,
    );
  }

  async function requestBatch(
    batch: readonly string[],
    signal?: AbortSignal,
  ): Promise<Float32Array[]> {
    const response = await options.http.request({
      url: `${baseUrl}/embeddings`,
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model, input: batch }),
      timeoutMs,
      signal,
    });
    if (response.status < 200 || response.status >= 300) {
      throw statusError(response.status, response.bodyText ?? "", apiKey);
    }
    const data = parseData(response.bodyText ?? "", apiKey);
    if (!Array.isArray(data) || data.length !== batch.length) {
      throw new Error("Remote embedding response is missing a complete `data` array");
    }
    const byIndex = new Map<number, { embedding?: unknown }>();
    for (const item of data) {
      const index = typeof item?.index === "number" ? item.index : byIndex.size;
      byIndex.set(index, item);
    }
    return batch.map((_, index) => {
      const item = byIndex.get(index);
      if (!item || !Array.isArray(item.embedding)) {
        throw new Error(`Remote embedding response is missing item ${index}`);
      }
      if (item.embedding.length !== dimension) {
        throw new Error(
          `Remote embedding model ${model} produced dimension ${item.embedding.length}, ` +
            `expected ${dimension}; the knowledge base dimension and the remote model must agree`,
        );
      }
      return Float32Array.from(item.embedding as number[]);
    });
  }

  return {
    modelId: remoteEmbeddingModelId(model, dimension),
    dimension,
    // OpenAI-compatible models embed plain text; the e5 query/passage
    // prefixes would add asymmetric noise.
    prefixPolicy: "none",
    async embed(
      texts: readonly string[],
      embedOptions?: EmbeddingOptions,
    ): Promise<readonly Float32Array[]> {
      const signal = embedOptions?.signal;
      const vectors: Float32Array[] = [];
      for (let start = 0; start < texts.length; start += batchSize) {
        if (signal?.aborted) throw abortError(signal);
        const batch = texts.slice(start, start + batchSize);
        vectors.push(...await requestBatch(batch, signal));
      }
      return vectors;
    },
  };
}

function parseData(bodyText: string, apiKey: string): unknown {
  try {
    return (JSON.parse(bodyText) as { data?: unknown }).data;
  } catch {
    throw new Error(
      `Remote embedding returned invalid JSON: ${redactText(bodyText.slice(0, 200), { secrets: [apiKey] })}`,
    );
  }
}

function statusError(status: number, bodyText: string, apiKey: string): Error {
  let message = bodyText.trim();
  try {
    const parsed = JSON.parse(bodyText) as { error?: { message?: string } };
    message = parsed.error?.message ?? message;
  } catch {
    // Use raw response text.
  }
  const error = new Error(
    redactText(message || `Remote embedding request failed with HTTP ${status}`, { secrets: [apiKey] }),
  ) as Error & { status?: number };
  error.status = status;
  return error;
}

function abortError(signal: AbortSignal): Error {
  const reason = (signal as { reason?: unknown }).reason;
  const message = typeof reason === "string" && reason ? reason : "cancelled by user";
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}

function trimTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) end -= 1;
  return value.slice(0, end);
}
