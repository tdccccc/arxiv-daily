import { afterEach, describe, expect, it, vi } from "vitest";
import {
  LLM_MAX_COMPLETION_TOKENS,
  LLM_MAX_OUTPUT_CODE_UNITS,
  LlmClient,
  LlmOutputLimitExceededError,
  LlmOutputTruncatedError,
  LlmTransientExhaustedError,
  isLlmOutputTruncatedError,
  isPermanentLlmError,
  isUnsupportedStreamOptionsError,
} from "../../src/llm/client";
import {
  RunCancelledError,
  isCancellationError,
} from "../../src/services/cancellation";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import {
  HttpTransportError,
  type HttpClient,
  type HttpRequest,
  type HttpResponse,
} from "../../src/core/adapters";

function response(status: number, bodyText: string): HttpResponse {
  return { status, bodyText, headers: {} };
}

function sse(events: unknown[]): string {
  return events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") + "data: [DONE]\n\n";
}

function client(
  request: (req: HttpRequest) => Promise<HttpResponse>,
  logger = new Logger("error"),
) {
  return new LlmClient(
    { ...DEFAULT_SETTINGS.llm, apiKey: "sk-full-secret", model: "test" },
    logger,
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

  it("sends validated completion bounds and retains them in stream-options fallback", async () => {
    const request = vi.fn()
      .mockResolvedValueOnce(response(400, JSON.stringify({ error: { message: "stream_options is unsupported" } })))
      .mockResolvedValueOnce(response(200, sse([{ choices: [{ delta: { content: "ok" } }] }])));
    await expect(client(request).call([{ role: "user", content: "x" }], {
      maxCompletionTokens: 4096,
      maxOutputCodeUnits: 64_000,
    })).resolves.toBe("ok");
    expect(request).toHaveBeenCalledTimes(2);
    expect(JSON.parse(String(request.mock.calls[0]?.[0].body))).toMatchObject({ max_tokens: 4096 });
    expect(JSON.parse(String(request.mock.calls[1]?.[0].body))).toMatchObject({ max_tokens: 4096 });
    for (const value of [0, -1, 1.5, Number.MAX_SAFE_INTEGER, LLM_MAX_OUTPUT_CODE_UNITS + 1]) {
      await expect(client(vi.fn()).call([{ role: "user", content: "x" }], {
        maxOutputCodeUnits: value,
      })).rejects.toBeInstanceOf(TypeError);
    }
    await expect(client(vi.fn()).call([{ role: "user", content: "x" }], {
      maxCompletionTokens: LLM_MAX_COMPLETION_TOKENS + 1,
    })).rejects.toBeInstanceOf(TypeError);
  });

  it("rejects streamed output at the accumulation limit without transport retry or content logging", async () => {
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const request = vi.fn(async () => response(200, sse([
      { choices: [{ delta: { content: "1234" } }] },
      { choices: [{ delta: { content: "SECRET-OVERFLOW" } }] },
      { choices: [{ delta: { content: "never-accumulated" } }] },
    ])));
    const error = await client(request, logger).call([{ role: "user", content: "x" }], {
      maxOutputCodeUnits: 5,
    }).catch((caught) => caught);
    expect(error).toBeInstanceOf(LlmOutputLimitExceededError);
    expect(error).toMatchObject({ code: "ARXIV_LLM_OUTPUT_LIMIT_EXCEEDED" });
    expect(error.message).not.toContain("SECRET-OVERFLOW");
    expect(request).toHaveBeenCalledTimes(1);
    expect(warn).not.toHaveBeenCalled();
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

  it("does not log or schedule a retry when a request is cancelled", async () => {
    const controller = new AbortController();
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const request = vi.fn(async () => {
      controller.abort("cancelled by user");
      throw new RunCancelledError("cancelled by user");
    });

    const error = await client(request, logger)
      .call([{ role: "user", content: "x" }], { signal: controller.signal })
      .catch((caught) => caught);

    expect(isCancellationError(error)).toBe(true);
    expect(error).toBeInstanceOf(RunCancelledError);
    expect(request).toHaveBeenCalledTimes(1);
    expect(warn).not.toHaveBeenCalled();
  });

  it("does not retry when the stream-options fallback is cancelled", async () => {
    const controller = new AbortController();
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const request = vi.fn()
      .mockResolvedValueOnce(response(400, JSON.stringify({
        error: { message: "stream_options is unsupported" },
      })))
      .mockImplementationOnce(async () => {
        controller.abort("cancelled by user");
        throw new RunCancelledError("cancelled by user");
      });

    const error = await client(request, logger)
      .call([{ role: "user", content: "x" }], { signal: controller.signal })
      .catch((caught) => caught);

    expect(isCancellationError(error)).toBe(true);
    expect(error).toBeInstanceOf(RunCancelledError);
    expect(request).toHaveBeenCalledTimes(2);
    expect(warn).not.toHaveBeenCalled();
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

  it("does not re-send a request whose transport failure may still be running", async () => {
    const request = vi.fn(async (): Promise<HttpResponse> => {
      throw new HttpTransportError("timeout", "HTTP timeout after 300000ms", { retryableAttempt: false });
    });

    const error = await client(request).call([{ role: "user", content: "x" }]).catch((e) => e);

    expect(error).toBeInstanceOf(LlmTransientExhaustedError);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("retries a transport failure the host marked as safe to repeat", async () => {
    vi.useFakeTimers();
    const request = vi.fn()
      .mockRejectedValueOnce(new HttpTransportError("network", "reset", { retryableAttempt: true }))
      .mockResolvedValueOnce(response(200, sse([{ choices: [{ delta: { content: "ok" } }] }])));

    const call = client(request).call([{ role: "user", content: "x" }]);
    await vi.runAllTimersAsync();

    await expect(call).resolves.toBe("ok");
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("gives the request the per-attempt signal that follows the caller's abort", async () => {
    const controller = new AbortController();
    let seen: AbortSignal | undefined;
    const request = vi.fn(async (req: HttpRequest): Promise<HttpResponse> => {
      seen = req.signal;
      controller.abort("cancelled by user");
      expect(req.signal?.aborted).toBe(true);
      throw new RunCancelledError("cancelled by user");
    });

    const error = await client(request)
      .call([{ role: "user", content: "x" }], { signal: controller.signal })
      .catch((e) => e);

    expect(isCancellationError(error)).toBe(true);
    expect(seen).toBeDefined();
    expect(seen).not.toBe(controller.signal);
  });

  it.each(["length", "max_tokens"])("throws a typed error instead of returning text cut off by finish_reason %s", async (reason) => {
    const request = vi.fn(async () => response(200, sse([
      { choices: [{ delta: { content: "# partial note" } }] },
      { choices: [{ delta: {}, finish_reason: reason }] },
    ])));
    const metrics = vi.fn();

    const error = await client(request)
      .call([{ role: "user", content: "x" }], { onMetrics: metrics })
      .catch((e) => e);

    expect(error).toBeInstanceOf(LlmOutputTruncatedError);
    expect(isLlmOutputTruncatedError(error)).toBe(true);
    expect(request).toHaveBeenCalledTimes(1);
    expect(metrics).toHaveBeenCalledWith(expect.objectContaining({ logicalCalls: 1, attempts: 1 }));
  });

  it("accepts a normal stop and an Anthropic end_turn", async () => {
    const stop = vi.fn(async () => response(200, sse([
      { choices: [{ delta: { content: "done" }, finish_reason: "stop" }] },
    ])));
    await expect(client(stop).call([{ role: "user", content: "x" }])).resolves.toBe("done");

    const anthropic = new LlmClient(
      { ...DEFAULT_SETTINGS.llm, provider: "anthropic", baseUrl: "https://api.anthropic.com/v1", apiKey: "sk-full-secret", model: "claude-sonnet-4-6", thinkingMode: false },
      new Logger("error"),
      { request: vi.fn(async () => response(200, [
        { type: "content_block_delta", delta: { type: "text_delta", text: "done" } },
        { type: "message_delta", delta: { stop_reason: "end_turn" } },
        { type: "message_stop" },
      ].map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""))) },
    );
    await expect(anthropic.call([{ role: "user", content: "x" }])).resolves.toBe("done");
  });

  it("types an Anthropic max_tokens stop_reason as truncation", async () => {
    const anthropic = new LlmClient(
      { ...DEFAULT_SETTINGS.llm, provider: "anthropic", baseUrl: "https://api.anthropic.com/v1", apiKey: "sk-full-secret", model: "claude-sonnet-4-6", thinkingMode: false },
      new Logger("error"),
      { request: vi.fn(async () => response(200, [
        { type: "content_block_delta", delta: { type: "text_delta", text: "# partial" } },
        { type: "message_delta", delta: { stop_reason: "max_tokens" } },
        { type: "message_stop" },
      ].map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""))) },
    );
    await expect(anthropic.call([{ role: "user", content: "x" }])).rejects.toBeInstanceOf(LlmOutputTruncatedError);
  });

  it.each([408, 409, 425, 429])("treats HTTP %i as transient and retries it", async (status) => {
    vi.useFakeTimers();
    const request = vi.fn(async () => response(status, "try again"));

    const call = client(request).call([{ role: "user", content: "x" }]).catch((e) => e);
    await vi.runAllTimersAsync();

    expect(await call).toBeInstanceOf(LlmTransientExhaustedError);
    expect(request).toHaveBeenCalledTimes(3);
    expect(isPermanentLlmError(Object.assign(new Error("x"), { status }))).toBe(false);
  });

  it.each([400, 401, 403, 404, 413, 422])("keeps HTTP %i permanent", (status) => {
    expect(isPermanentLlmError(Object.assign(new Error("x"), { status }))).toBe(true);
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
