import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import OpenAI from "openai";
import {
  buildModelUrlCandidates,
  LLM_STREAM_IDLE_TIMEOUT_MS,
  LlmClient,
  collectStreamWithIdleTimeout,
  normalizeOpenAiBaseUrl,
} from "../../src/llm/client";

vi.mock("openai", () => {
  const OpenAIMock = vi.fn().mockImplementation(() => ({
    chat: { completions: { create: vi.fn() } },
  }));
  return { default: OpenAIMock };
});

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

  it("constructs the OpenAI client with the normalized chat base URL", () => {
    new LlmClient(
      {
        apiKey: "sk-test",
        provider: "custom",
        baseUrl: "http://59.64.32.247:5001",
        model: "gpt-5.5",
        thinkingMode: false,
        reasoningEffort: "high",
      },
      { warn: vi.fn() } as any,
    );

    expect(OpenAI).toHaveBeenCalledWith(
      expect.objectContaining({
        baseURL: "http://59.64.32.247:5001/v1",
      }),
    );
  });

  it("does not retry permanent non-429 4xx chat errors", async () => {
    vi.useFakeTimers();
    const create = vi.fn().mockRejectedValue(
      Object.assign(new Error("Unauthorized"), { status: 401 }),
    );
    vi.mocked(OpenAI).mockImplementationOnce(
      () => ({ chat: { completions: { create } } }) as any,
    );
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
    );

    const call = client.call([{ role: "user", content: "hello" }]);
    const assertion = expect(call).rejects.toThrow("Unauthorized");
    await vi.runAllTimersAsync();

    await assertion;
    expect(create).toHaveBeenCalledTimes(1);
  });

  it("aborts a stream when no chunk arrives before the idle timeout", async () => {
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
    const assertion = expect(read).rejects.toThrow("LLM stream idle timeout");
    await vi.advanceTimersByTimeAsync(LLM_STREAM_IDLE_TIMEOUT_MS);

    await assertion;
    expect(controller.signal.aborted).toBe(true);
  });
});
