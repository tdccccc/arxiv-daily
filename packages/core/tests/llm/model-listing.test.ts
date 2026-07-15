import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import {
  buildModelUrlCandidates,
  LLM_STREAM_IDLE_TIMEOUT_MS,
  LlmClient,
  StreamIdleTimeoutError,
  collectStreamWithIdleTimeout,
  normalizeOpenAiBaseUrl,
} from "../../src/llm/client";
import { isCancellationError } from "../../src/services/cancellation";

describe("Model listing", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("tries /models directly when the base URL already includes /v1", () => {
    const candidates = buildModelUrlCandidates("https://api.deepseek.com/v1");

    expect(candidates[0]).toBe("https://api.deepseek.com/v1/models");
    expect(candidates).not.toContain("https://api.deepseek.com/v1/v1/models");
  });

  it("keeps fallback candidates unique and in priority order", () => {
    const candidates = buildModelUrlCandidates("https://llm.example.com");

    expect(candidates).toEqual([
      "https://llm.example.com/v1/models",
      "https://llm.example.com/models",
    ]);
    expect(new Set(candidates).size).toBe(candidates.length);
  });

  it("normalizes root OpenAI-compatible gateways to /v1 for chat calls", () => {
    expect(normalizeOpenAiBaseUrl("http://59.64.32.247:5001")).toBe(
      "http://59.64.32.247:5001/v1",
    );
    expect(normalizeOpenAiBaseUrl("http://59.64.32.247:5001/")).toBe(
      "http://59.64.32.247:5001/v1",
    );
  });

  it("does not duplicate an existing /v1 suffix for chat calls", () => {
    expect(normalizeOpenAiBaseUrl("http://59.64.32.247:5001/v1")).toBe(
      "http://59.64.32.247:5001/v1",
    );
  });

  it("posts chat calls to the normalized OpenAI-compatible chat URL", async () => {
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ choices: [{ message: { content: "ok" } }] }),
    }));
    const client = new LlmClient(
      {
        apiKey: "sk-test",
        provider: "custom",
        baseUrl: "http://59.64.32.247:5001",
        model: "gpt-5.5",
        thinkingMode: false,
        reasoningEffort: "high",
      },
      { warn: vi.fn() } as any,
      { request },
    );

    await expect(client.testConnection()).resolves.toEqual({ success: true });

    expect(request).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://59.64.32.247:5001/v1/chat/completions",
        method: "POST",
      }),
    );
  });

  it("does not retry permanent non-429 4xx chat errors", async () => {
    const request = vi.fn(async () => ({
      status: 401,
      headers: {},
      bodyText: JSON.stringify({ error: { message: "Unauthorized" } }),
    }));
    const client = new LlmClient(
      {
        apiKey: "sk-test",
        provider: "custom",
        baseUrl: "https://llm.example.com/v1",
        model: "gpt-test",
        thinkingMode: false,
        reasoningEffort: "high",
      },
      { warn: vi.fn() } as any,
      { request },
    );

    await expect(client.call([{ role: "user", content: "hello" }]))
      .rejects.toThrow("Unauthorized");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("includes the last endpoint error when model listing fails", async () => {
    const request = vi.fn(async () => ({
      status: 401,
      headers: {},
      bodyText: JSON.stringify({ error: { message: "bad api key" } }),
    }));
    const client = new LlmClient(
      {
        apiKey: "sk-test",
        provider: "custom",
        baseUrl: "https://llm.example.com/v1",
        model: "gpt-test",
        thinkingMode: false,
        reasoningEffort: "high",
      },
      { warn: vi.fn() } as any,
      { request },
    );

    await expect(client.fetchModels()).rejects.toThrow(
      "Failed to fetch models from any endpoint: bad api key",
    );
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("collects OpenAI-compatible SSE chat completion chunks", async () => {
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText:
        `data: ${JSON.stringify({ choices: [{ delta: { content: "hel" } }] })}\n\n` +
        `data: ${JSON.stringify({ choices: [{ delta: { content: "lo" } }] })}\n\n` +
        "data: [DONE]\n\n",
    }));
    const client = new LlmClient(
      {
        apiKey: "sk-test",
        provider: "custom",
        baseUrl: "https://llm.example.com/v1",
        model: "gpt-test",
        thinkingMode: false,
        reasoningEffort: "high",
      },
      { warn: vi.fn() } as any,
      { request },
    );

    await expect(client.call([{ role: "user", content: "hello" }]))
      .resolves.toBe("hello");
  });

  it("rejects stream idle timeouts with a retryable timeout error", async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    async function* neverYields() {
      await new Promise(() => undefined);
    }

    const read = collectStreamWithIdleTimeout(
      neverYields(),
      controller,
      LLM_STREAM_IDLE_TIMEOUT_MS,
    );
    const caught = read.catch((error) => error);
    await vi.advanceTimersByTimeAsync(LLM_STREAM_IDLE_TIMEOUT_MS);

    const error = await caught;
    expect(error).toBeInstanceOf(StreamIdleTimeoutError);
    expect(isCancellationError(error)).toBe(false);
    expect(controller.signal.aborted).toBe(true);
  });
});
