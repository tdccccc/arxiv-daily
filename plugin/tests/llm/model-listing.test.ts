import { describe, it, expect, vi } from "vitest";
import OpenAI from "openai";
import {
  buildModelUrlCandidates,
  LlmClient,
  normalizeOpenAiBaseUrl,
} from "../../src/llm/client";

vi.mock("openai", () => {
  const OpenAIMock = vi.fn().mockImplementation(() => ({
    chat: { completions: { create: vi.fn() } },
  }));
  return { default: OpenAIMock };
});

describe("Model listing", () => {
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
});
