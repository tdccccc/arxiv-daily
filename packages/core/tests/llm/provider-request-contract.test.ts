import { describe, expect, it, vi } from "vitest";
import { LlmClient } from "../../src/llm/client";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import { Logger } from "../../src/services/logger";
import type { HttpRequest } from "../../src/core/adapters";
import type { LlmSettings } from "../../src/settings/types";
import { buildCheckpointGenerationIdentity } from "../../src/services/daily-summary-checkpoint-store";

const answer = {
  status: 200, headers: {},
  bodyText: 'data: {"choices":[{"delta":{"content":"2"}}]}\n\ndata: [DONE]\n\n',
};
const prompt = [{ role: "user" as const, content: "1+1?" }];

function fixture(provider: string, thinkingMode = true, overrides: Partial<LlmSettings> = {}) {
  const request = vi.fn(async (_req: HttpRequest) => answer);
  const client = new LlmClient({
    ...DEFAULT_SETTINGS.llm, provider, thinkingMode,
    baseUrl: "https://provider.example/v1", model: "test-model", apiKey: "test-key",
    reasoningEffort: "high",
    ...overrides,
  }, new Logger("error"), { request });
  return { client, request, body: () => JSON.parse(String(request.mock.calls[0]![0].body)) };
}

describe("provider request contract", () => {
  it("uses native Anthropic adaptive thinking and decodes only answer text with usage", async () => {
    const { client, request, body } = fixture("anthropic", true, {
      baseUrl: "https://api.anthropic.com/v1", model: "claude-opus-4-7",
    });
    request.mockResolvedValueOnce({ ...answer, bodyText: [
      { type: "message_start", message: { usage: { input_tokens: 5, cache_read_input_tokens: 3, output_tokens: 1 } } },
      { type: "content_block_delta", delta: { type: "thinking_delta", thinking: "private reasoning" } },
      { type: "content_block_delta", delta: { type: "text_delta", text: "2" } },
      { type: "message_delta", usage: { output_tokens: 2 } },
      { type: "message_stop" },
    ].map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join("") });
    const metrics = vi.fn();
    expect(await client.call([{ role: "system", content: "Be concise." }, ...prompt], {
      maxCompletionTokens: 4096, onMetrics: metrics,
    })).toBe("2");
    expect(request.mock.calls[0]![0]).toMatchObject({
      url: "https://api.anthropic.com/v1/messages",
      headers: { "x-api-key": "test-key", "anthropic-version": "2023-06-01" },
    });
    expect(body()).toMatchObject({
      system: "Be concise.", messages: prompt, max_tokens: 4096,
      thinking: { type: "adaptive" }, output_config: { effort: "high" },
    });
    expect(body()).not.toHaveProperty("stream_options");
    expect(body().thinking).not.toHaveProperty("budget_tokens");
    expect(metrics).toHaveBeenCalledWith(expect.objectContaining({ inputTokens: 8, outputTokens: 2 }));
  });

  it("uses a model-aware checkpoint identity and the native Anthropic endpoint", () => {
    const settings = { ...DEFAULT_SETTINGS.llm, provider: "anthropic", baseUrl: "https://api.anthropic.com/v1", model: "claude-opus-4-7", thinkingMode: true, reasoningEffort: "high" };
    const native = buildCheckpointGenerationIdentity(settings, 0);
    expect(native.mode).toEqual({ kind: "anthropic-adaptive", reasoningEffort: "high" });
    expect(native.endpointDigest).not.toBe(buildCheckpointGenerationIdentity({ ...settings, provider: "custom" }, 0).endpointDigest);
  });

  it("rejects native Anthropic error events without exposing the API key", async () => {
    const { client, request } = fixture("anthropic", true, { baseUrl: "https://api.anthropic.com/v1", model: "claude-opus-4-7" });
    request.mockResolvedValueOnce({ ...answer, bodyText: 'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error","message":"bad test-key"}}\n\n' });
    await expect(client.call(prompt)).rejects.toThrow("[REDACTED]");
    expect(request).toHaveBeenCalledOnce();
  });

  it("enforces the output limit on native Anthropic text deltas", async () => {
    const { client, request } = fixture("anthropic", true, { baseUrl: "https://api.anthropic.com/v1", model: "claude-opus-4-7" });
    request.mockResolvedValueOnce({ ...answer, bodyText: 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"long answer"}}\n\n' });
    await expect(client.call(prompt, { maxOutputCodeUnits: 3 })).rejects.toThrow("output exceeded");
    expect(request).toHaveBeenCalledOnce();
  });

  it.each(["openai", "custom"])("sends only standard reasoning parameters to %s", async (provider) => {
    const { client, body } = fixture(provider);
    expect(await client.call(prompt)).toBe("2");
    expect(body()).toMatchObject({ reasoning_effort: "high" });
    expect(body()).not.toHaveProperty("thinking");
    expect(body()).not.toHaveProperty("extra_body");
    expect(body()).not.toHaveProperty("temperature");
  });

  it("sends DeepSeek thinking and effort at the top level", async () => {
    const { client, body } = fixture("deepseek");
    await client.call(prompt);
    expect(body()).toMatchObject({ thinking: { type: "enabled" }, reasoning_effort: "high" });
    expect(body()).not.toHaveProperty("extra_body");
  });

  it("sends GLM thinking without unsupported effort fields", async () => {
    const { client, body } = fixture("zhipu");
    await client.call(prompt);
    expect(body()).toMatchObject({ thinking: { type: "enabled" } });
    expect(body()).not.toHaveProperty("reasoning_effort");
    expect(body()).not.toHaveProperty("extra_body");
  });

  it.each(["deepseek", "zhipu", "anthropic"])("explicitly disables %s thinking when off", async (provider) => {
    const { client, body } = fixture(provider, false);
    await client.call(prompt, { temperature: 0.2 });
    expect(body()).toMatchObject({ thinking: { type: "disabled" }, temperature: 0.2 });
    expect(body()).not.toHaveProperty("reasoning_effort");
    expect(body()).not.toHaveProperty("extra_body");
  });

  it("keeps the ordinary custom-provider temperature contract", async () => {
    const { client, body } = fixture("custom", false);
    await client.call(prompt, { temperature: 0.2, maxCompletionTokens: 128 });
    expect(body()).toMatchObject({ temperature: 0.2, max_tokens: 128 });
    expect(body()).not.toHaveProperty("thinking");
    expect(body()).not.toHaveProperty("reasoning_effort");
  });

  it("uses the OpenAI completion-token field and retains it in stream fallback", async () => {
    const { client, request } = fixture("openai");
    request.mockResolvedValueOnce({ status: 400, headers: {}, bodyText: '{"error":{"message":"stream_options unsupported"}}' });
    await client.call(prompt, { maxCompletionTokens: 4096 });
    expect(request).toHaveBeenCalledTimes(2);
    for (const [req] of request.mock.calls) {
      const body = JSON.parse(String(req.body));
      expect(body).toMatchObject({ max_completion_tokens: 4096, reasoning_effort: "high" });
      expect(body).not.toHaveProperty("max_tokens");
      expect(body).not.toHaveProperty("extra_body");
    }
  });

  it("keeps the Anthropic thinking budget below an explicit completion cap", async () => {
    const { client, body } = fixture("anthropic");
    await client.call(prompt, { maxCompletionTokens: 4096 });
    expect(body().thinking.type).toBe("enabled");
    expect(body().thinking.budget_tokens).toBeGreaterThanOrEqual(1024);
    expect(body().thinking.budget_tokens).toBeLessThan(4096);
    expect(body().max_tokens).toBe(4096);
    expect(body()).not.toHaveProperty("extra_body");
    expect(body()).not.toHaveProperty("reasoning_effort");
  });

  it("supplies a valid Anthropic completion cap when the caller leaves it unset", async () => {
    const { client, body } = fixture("anthropic");
    await client.call(prompt);
    expect(body().thinking.budget_tokens).toBe(16384);
    expect(body().max_tokens).toBeGreaterThan(16384);
  });

  it("rejects an impossible Anthropic thinking budget before making a request", async () => {
    const { client, request } = fixture("anthropic");
    await expect(client.call(prompt, { maxCompletionTokens: 1024 })).rejects.toThrow("thinking");
    expect(request).not.toHaveBeenCalled();
  });
});
