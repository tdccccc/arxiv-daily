import { describe, expect, it, vi } from "vitest";
import {
  createRemoteEmbeddingModel,
  remoteEmbeddingModelId,
  type RemoteEmbeddingModelOptions,
} from "../src/library/fulltext/remote-embedding-model";
import type { HttpClient, HttpRequest, HttpResponse } from "../src/core/adapters";

const BASE_URL = "https://embeddings.example.test/v1";
const API_KEY = "sk-test-secret";
const MODEL = "text-embedding-3-small";
const DIMENSION = 8;

function vector(length = DIMENSION, seed = 0): number[] {
  // Integer values: exact in Float32Array comparisons.
  return Array.from({ length }, (_, i) => seed + i);
}

function makeHttp(
  respond: (request: HttpRequest) => HttpResponse | Promise<HttpResponse>,
): HttpClient {
  return { request: vi.fn(async (request: HttpRequest) => respond(request)) } as unknown as HttpClient;
}

function modelOptions(http: HttpClient, overrides: Partial<RemoteEmbeddingModelOptions> = {}): RemoteEmbeddingModelOptions {
  return {
    baseUrl: BASE_URL,
    apiKey: API_KEY,
    model: MODEL,
    dimension: DIMENSION,
    http,
    batchSize: 2,
    ...overrides,
  };
}

function embeddingResponse(batch: string[], seed: number): HttpResponse {
  return {
    status: 200,
    headers: {},
    bodyText: JSON.stringify({
      data: batch.map((_, i) => ({
        index: i,
        embedding: vector(DIMENSION, seed + i),
      })),
    }),
  };
}

describe("createRemoteEmbeddingModel", () => {
  it("posts batches to {baseUrl}/embeddings and returns vectors in input order", async () => {
    const requests: HttpRequest[] = [];
    const http = makeHttp(async (request) => {
      requests.push(request);
      const body = JSON.parse(String(request.body)) as { input: string[] };
      return embeddingResponse(body.input, (requests.length - 1) * 10);
    });
    const model = createRemoteEmbeddingModel(modelOptions(http));

    const vectors = await model.embed(["a", "b", "c", "d", "e"]);

    expect(model.modelId).toBe(remoteEmbeddingModelId(MODEL, DIMENSION));
    expect(model.dimension).toBe(DIMENSION);
    expect(model.prefixPolicy).toBe("none");
    expect(vectors.map((v) => Array.from(v))).toEqual([
      vector(DIMENSION, 0),
      vector(DIMENSION, 1),
      vector(DIMENSION, 10),
      vector(DIMENSION, 11),
      vector(DIMENSION, 20),
    ]);
    expect(requests).toHaveLength(3); // batchSize 2 -> 3 batches
    expect(requests[0]!.url).toBe(`${BASE_URL}/embeddings`);
    expect(requests[0]!.headers?.["Authorization"]).toBe(`Bearer ${API_KEY}`);
    expect(JSON.parse(String(requests[0]!.body))).toEqual({ model: MODEL, input: ["a", "b"] });
  });

  it("orders vectors by the response index, not array position", async () => {
    const http = makeHttp(() => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({
        data: [
          { index: 1, embedding: vector(DIMENSION, 1) },
          { index: 0, embedding: vector(DIMENSION, 0) },
        ],
      }),
    }));
    const model = createRemoteEmbeddingModel(modelOptions(http));

    const vectors = await model.embed(["a", "b"]);

    expect(Array.from(vectors[0]!)).toEqual(vector(DIMENSION, 0));
    expect(Array.from(vectors[1]!)).toEqual(vector(DIMENSION, 1));
  });

  it("rejects dimension mismatches so the knowledge base cannot be poisoned", async () => {
    const http = makeHttp(() => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ data: [{ index: 0, embedding: vector(4) }] }),
    }));
    const model = createRemoteEmbeddingModel(modelOptions(http));

    await expect(model.embed(["a"])).rejects.toThrow("produced dimension 4, expected 8");
  });

  it("surfaces HTTP errors with the API key redacted", async () => {
    const http = makeHttp(() => ({
      status: 401,
      headers: {},
      bodyText: JSON.stringify({ error: { message: `invalid key ${API_KEY}` } }),
    }));
    const model = createRemoteEmbeddingModel(modelOptions(http));

    await expect(model.embed(["a"])).rejects.toThrow("invalid key [REDACTED]");
    await expect(model.embed(["a"])).rejects.not.toThrow(API_KEY);
  });

  it("rejects invalid JSON bodies", async () => {
    const http = makeHttp(() => ({ status: 200, headers: {}, bodyText: "not json" }));
    const model = createRemoteEmbeddingModel(modelOptions(http));

    await expect(model.embed(["a"])).rejects.toThrow("invalid JSON");
  });

  it("rejects incomplete data arrays", async () => {
    const http = makeHttp(() => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ data: [{ index: 0, embedding: vector() }] }),
    }));
    const model = createRemoteEmbeddingModel(modelOptions(http));

    await expect(model.embed(["a", "b"])).rejects.toThrow("missing a complete `data` array");
  });

  it("propagates abort signals between batches", async () => {
    const controller = new AbortController();
    const http = makeHttp(async (request) => {
      controller.abort();
      const body = JSON.parse(String(request.body)) as { input: string[] };
      return embeddingResponse(body.input, 0);
    });
    const model = createRemoteEmbeddingModel(modelOptions(http, { batchSize: 1 }));

    await expect(model.embed(["a", "b"], { signal: controller.signal }))
      .rejects.toMatchObject({ name: "AbortError" });
  });

  it("validates configuration eagerly", () => {
    const http = makeHttp(() => embeddingResponse(["a"], 0));
    expect(() => createRemoteEmbeddingModel(modelOptions(http, { baseUrl: "" })))
      .toThrow("requires an endpoint base URL");
    expect(() => createRemoteEmbeddingModel(modelOptions(http, { apiKey: "" })))
      .toThrow("requires an endpoint base URL, API key, and model");
    expect(() => createRemoteEmbeddingModel(modelOptions(http, { dimension: -1 })))
      .toThrow("dimension must be a positive integer");
  });
});

describe("remoteEmbeddingModelId", () => {
  it("combines model and dimension, excluding the endpoint", () => {
    expect(remoteEmbeddingModelId("text-embedding-3-small", 1536))
      .toBe("remote:text-embedding-3-small:1536");
  });
});
