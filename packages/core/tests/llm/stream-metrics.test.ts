import { afterEach, describe, expect, it, vi } from "vitest";
import {
  LlmClient,
  LlmTransientExhaustedError,
  isUnsupportedStreamOptionsError,
} from "../../src/llm/client";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type { HttpClient, HttpRequest, HttpResponse } from "../../src/core/adapters";

function response(status: number, bodyText: string): HttpResponse {
  return { status, bodyText, headers: {} };
}

function sse(events: unknown[]): string {
  return events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") + "data: [DONE]\n\n";
}

function client(request: (req: HttpRequest) => Promise<HttpResponse>) {
  return new LlmClient(
    { ...DEFAULT_SETTINGS.llm, apiKey: "sk-full-secret", model: "test" },
    new Logger("error"),
    { request } satisfies HttpClient,
  );
}

describe("LLM stream metrics", () => {
  afterEach(() => vi.useRealTimers());

  it("requests streamed usage and reports provider aliases", async () => {
    const request = vi.fn(async (req: HttpRequest) => response(200, sse([
      { choices: [{ delta: { content: "hello" } }] },
      { choices: [], usage: { input_tokens: 8, output_tokens: 2, total_tokens: 10 } },
    ])));
    const metrics = vi.fn();

    await expect(client(request).call([{ role: "user", content: "x" }], { onMetrics: metrics }))
      .resolves.toBe("hello");
    expect(JSON.parse(String(request.mock.calls[0]?.[0].body))).toMatchObject({
      stream: true,
      stream_options: { include_usage: true },
    });
    expect(metrics).toHaveBeenCalledWith(expect.objectContaining({
      logicalCalls: 1, attempts: 1, usageComplete: true,
      inputTokens: 8, outputTokens: 2, totalTokens: 10,
    }));
  });

  it("falls back once only for explicit unsupported stream_options 400/422", async () => {
    const request = vi.fn()
      .mockResolvedValueOnce(response(400, JSON.stringify({ error: { message: "stream_options is unsupported" } })))
      .mockResolvedValueOnce(response(200, sse([{ choices: [{ delta: { content: "ok" } }] }])));
    const metrics = vi.fn();

    await expect(client(request).call([{ role: "user", content: "x" }], { onMetrics: metrics }))
      .resolves.toBe("ok");
    expect(request).toHaveBeenCalledTimes(2);
    expect(JSON.parse(String(request.mock.calls[1]?.[0].body))).not.toHaveProperty("stream_options");
    expect(metrics).toHaveBeenCalledWith(expect.objectContaining({
      logicalCalls: 1, attempts: 2, usageComplete: false,
    }));
  });

  it("types exhausted transient retries at the LLM client boundary", async () => {
    vi.useFakeTimers();
    const request = vi.fn(async () => response(503, "provider unavailable"));
    const metrics = vi.fn();

    const call = client(request)
      .call([{ role: "user", content: "x" }], { onMetrics: metrics })
      .catch((error) => error);
    await vi.runAllTimersAsync();

    const error = await call;
    expect(error).toBeInstanceOf(LlmTransientExhaustedError);
    expect(error).toMatchObject({ message: "provider unavailable" });
    expect(request).toHaveBeenCalledTimes(3);
    expect(metrics).toHaveBeenCalledWith(
      expect.objectContaining({ logicalCalls: 1, attempts: 3 }),
    );
  });

  it("does not treat generic client errors as unsupported options and sanitizes provider errors", async () => {
    expect(isUnsupportedStreamOptionsError(Object.assign(new Error("bad request"), { status: 400 }))).toBe(false);
    const request = vi.fn(async () => response(401, JSON.stringify({
      error: { message: "invalid sk-full-secret Authorization: Bearer sk-full-secret" },
    })));
    await expect(client(request).call([{ role: "user", content: "x" }]))
      .rejects.not.toThrow(/sk-full-secret/);
    expect(request).toHaveBeenCalledTimes(1);
  });
});
