import { afterEach, describe, expect, it, vi } from "vitest";
import {
  LLM_STREAM_IDLE_TIMEOUT_MS,
  LlmClient,
  LlmOutputLimitExceededError,
  LlmTransientExhaustedError,
  StreamIdleTimeoutError,
  isPermanentLlmError,
} from "../../src/llm/client";
import { isCancellationError } from "../../src/services/cancellation";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type {
  HttpClient,
  HttpRequest,
  HttpResponse,
  HttpStreamResponse,
} from "../../src/core/adapters";

const messages = [{ role: "user" as const, content: "x" }];

function event(content: string): string {
  return `data: ${JSON.stringify({ choices: [{ delta: { content } }] })}\n\n`;
}

/** A hand-driven network body: the test decides when chunks arrive or the stream ends. */
function liveBody() {
  const queue: Array<IteratorResult<string>> = [];
  let waiter: ((result: IteratorResult<string>) => void) | undefined;
  const deliver = (result: IteratorResult<string>) => {
    if (waiter) {
      const resolve = waiter;
      waiter = undefined;
      resolve(result);
    } else {
      queue.push(result);
    }
  };
  const returned = vi.fn();
  const iterator: AsyncIterator<string> = {
    next: () => {
      const ready = queue.shift();
      if (ready) return Promise.resolve(ready);
      return new Promise((resolve) => { waiter = resolve; });
    },
    return: async () => {
      returned();
      waiter?.({ done: true, value: undefined });
      waiter = undefined;
      return { done: true, value: undefined };
    },
  };
  return {
    body: { [Symbol.asyncIterator]: () => iterator } as AsyncIterable<string>,
    push: (text: string) => deliver({ done: false, value: text }),
    end: () => deliver({ done: true, value: undefined }),
    returned,
  };
}

function streamingClient(
  requestStream: (req: HttpRequest) => Promise<HttpStreamResponse>,
  settings: Partial<typeof DEFAULT_SETTINGS.llm> = {},
) {
  const request = vi.fn(async (): Promise<HttpResponse> => {
    throw new Error("buffered request() must not be used when requestStream exists");
  });
  const http: HttpClient = { request, requestStream };
  return {
    request,
    llm: new LlmClient(
      { ...DEFAULT_SETTINGS.llm, apiKey: "sk-full-secret", model: "test", ...settings },
      new Logger("error"),
      http,
    ),
  };
}

describe("LLM real streaming", () => {
  afterEach(() => vi.useRealTimers());

  it("parses events that arrive in arbitrary network chunks", async () => {
    const live = liveBody();
    const requestStream = vi.fn(async () => ({ status: 200, headers: {}, body: live.body }));
    const { llm } = streamingClient(requestStream);
    const wire = event("hel") + event("lo") + "data: [DONE]\n\n";
    const crlf = wire.replace(/\n/g, "\r\n");
    const result = llm.call(messages);
    // Split mid-JSON and between the two characters of a CRLF separator.
    for (const part of [crlf.slice(0, 12), crlf.slice(12, 40), crlf.slice(40, 41), crlf.slice(41)]) {
      live.push(part);
      await Promise.resolve();
    }
    live.end();

    await expect(result).resolves.toBe("hello");
    expect(requestStream).toHaveBeenCalledTimes(1);
  });

  it("aborts a stalled stream after the idle gap, not after the total timeout", async () => {
    vi.useFakeTimers();
    const live = liveBody();
    let signal: AbortSignal | undefined;
    const requestStream = vi.fn(async (req: HttpRequest) => {
      signal = req.signal;
      return { status: 200, headers: {}, body: live.body };
    });
    const { llm, request } = streamingClient(requestStream);

    const outcome = llm.call(messages).then(() => undefined, (error: unknown) => error);
    await vi.advanceTimersByTimeAsync(0);
    live.push(event("partial"));
    await vi.advanceTimersByTimeAsync(LLM_STREAM_IDLE_TIMEOUT_MS - 1);
    expect(signal?.aborted).toBe(false);
    expect(live.returned).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);
    const error = await outcome;
    expect(error).toBeInstanceOf(LlmTransientExhaustedError);
    expect((error as LlmTransientExhaustedError).cause.name).toBe("StreamIdleTimeoutError");
    expect(signal?.aborted).toBe(true);
    expect(signal?.reason).toBeInstanceOf(StreamIdleTimeoutError);
    expect(live.returned).toHaveBeenCalledTimes(1);
    // A silent stream may still be generating at the provider: no blind resend.
    expect(requestStream).toHaveBeenCalledTimes(1);
    expect(request).not.toHaveBeenCalled();
  });

  it("treats keep-alive comments as activity", async () => {
    vi.useFakeTimers();
    const live = liveBody();
    const { llm } = streamingClient(async () => ({ status: 200, headers: {}, body: live.body }));

    const result = llm.call(messages);
    await vi.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 2; i += 1) {
      await vi.advanceTimersByTimeAsync(LLM_STREAM_IDLE_TIMEOUT_MS - 1_000);
      live.push(": keep-alive\n\n");
    }
    live.push(event("done") + "data: [DONE]\n\n");
    live.end();

    await expect(result).resolves.toBe("done");
  });

  it("ends a trickling stream at the total attempt deadline without resending", async () => {
    vi.useFakeTimers();
    const live = liveBody();
    const requestStream = vi.fn(async () => ({ status: 200, headers: {}, body: live.body }));
    const { llm } = streamingClient(requestStream);

    const outcome = llm.call(messages).then(() => undefined, (error: unknown) => error);
    await vi.advanceTimersByTimeAsync(0);
    for (let elapsed = 0; elapsed < 300_000; elapsed += 60_000) {
      live.push(": keep-alive\n\n");
      await vi.advanceTimersByTimeAsync(60_000);
    }

    const error = await outcome;
    expect(error).toBeInstanceOf(LlmTransientExhaustedError);
    expect((error as LlmTransientExhaustedError).cause).toMatchObject({
      name: "HttpTransportError", kind: "timeout",
    });
    expect(requestStream).toHaveBeenCalledTimes(1);
    expect(live.returned).toHaveBeenCalledTimes(1);
  });

  it("aborts as soon as the output cap is exceeded and cancels the reader", async () => {
    const live = liveBody();
    let signal: AbortSignal | undefined;
    const requestStream = vi.fn(async (req: HttpRequest) => {
      signal = req.signal;
      return { status: 200, headers: {}, body: live.body };
    });
    const { llm } = streamingClient(requestStream);

    const result = llm.call(messages, { maxOutputCodeUnits: 10 });
    const outcome = result.then(() => undefined, (error: unknown) => error);
    live.push(event("12345"));
    live.push(event("678901"));
    // The stream is still open (no end, no [DONE]): the cap alone stops it.

    expect(await outcome).toBeInstanceOf(LlmOutputLimitExceededError);
    expect(signal?.aborted).toBe(true);
    expect(signal?.reason).toBeInstanceOf(LlmOutputLimitExceededError);
    expect(live.returned).toHaveBeenCalledTimes(1);
    expect(requestStream).toHaveBeenCalledTimes(1);
  });

  it("cancels the reader and reports cancellation when the caller aborts mid-stream", async () => {
    const live = liveBody();
    const controller = new AbortController();
    const { llm } = streamingClient(async () => ({ status: 200, headers: {}, body: live.body }));

    const outcome = llm.call(messages, { signal: controller.signal })
      .then(() => undefined, (error: unknown) => error);
    live.push(event("a"));
    await Promise.resolve();
    controller.abort("stop now");

    expect(isCancellationError(await outcome)).toBe(true);
    expect(live.returned).toHaveBeenCalledTimes(1);
  });

  it("turns a non-2xx streamed response into the same permanent error as before", async () => {
    const live = liveBody();
    const requestStream = vi.fn(async () => ({ status: 401, headers: {}, body: live.body }));
    const { llm } = streamingClient(requestStream);
    live.push('{"error":{"message":"bad key sk-full-secret"}}');
    live.end();

    const error = await llm.call(messages).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(Error);
    expect(isPermanentLlmError(error)).toBe(true);
    expect((error as Error).message).toContain("bad key");
    expect((error as Error).message).not.toContain("sk-full-secret");
    expect(requestStream).toHaveBeenCalledTimes(1);
  });

  it("retries a 5xx streamed response", async () => {
    vi.useFakeTimers();
    const bodies = [liveBody(), liveBody()];
    bodies[0].push("upstream down");
    bodies[0].end();
    bodies[1].push(event("ok") + "data: [DONE]\n\n");
    bodies[1].end();
    const requestStream = vi.fn(async () => ({
      status: requestStream.mock.calls.length === 1 ? 503 : 200,
      headers: {},
      body: bodies[requestStream.mock.calls.length - 1].body,
    }));
    const { llm } = streamingClient(requestStream);

    const result = llm.call(messages);
    await vi.advanceTimersByTimeAsync(60_000);

    await expect(result).resolves.toBe("ok");
    expect(requestStream).toHaveBeenCalledTimes(2);
  });

  it("falls back without stream_options after an unsupported-option 400", async () => {
    const first = liveBody();
    first.push('{"error":{"message":"Unsupported parameter: stream_options"}}');
    first.end();
    const second = liveBody();
    second.push(event("fine") + "data: [DONE]\n\n");
    second.end();
    const sent: Array<Record<string, unknown>> = [];
    const requestStream = vi.fn(async (req: HttpRequest) => {
      sent.push(JSON.parse(String(req.body)) as Record<string, unknown>);
      return sent.length === 1
        ? { status: 400, headers: {}, body: first.body }
        : { status: 200, headers: {}, body: second.body };
    });
    const { llm } = streamingClient(requestStream);

    await expect(llm.call(messages)).resolves.toBe("fine");
    expect(sent[0]).toHaveProperty("stream_options");
    expect(sent[1]).not.toHaveProperty("stream_options");
    expect(first.returned).toHaveBeenCalledTimes(1);
  });

  it("streams native Anthropic events and stops at message_stop", async () => {
    const live = liveBody();
    const { llm } = streamingClient(
      async () => ({ status: 200, headers: {}, body: live.body }),
      { provider: "anthropic", baseUrl: "https://api.anthropic.com" },
    );
    const sse = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`;
    live.push(sse({ type: "message_start", message: { usage: { input_tokens: 3 } } }));
    live.push(sse({ type: "content_block_delta", delta: { type: "text_delta", text: "hi" } }));
    live.push(sse({ type: "message_stop" }));

    await expect(llm.call(messages)).resolves.toBe("hi");
    // message_stop ends parsing; the rest of the body is not drained.
    expect(live.returned).toHaveBeenCalled();
  });

  it("still works on hosts that can only buffer the body", async () => {
    const request = vi.fn(async (): Promise<HttpResponse> => ({
      status: 200,
      headers: {},
      bodyText: event("buf") + event("fered") + "data: [DONE]\n\n",
    }));
    const llm = new LlmClient(
      { ...DEFAULT_SETTINGS.llm, apiKey: "sk-full-secret", model: "test" },
      new Logger("error"),
      { request },
    );

    await expect(llm.call(messages)).resolves.toBe("buffered");
    expect(request).toHaveBeenCalledTimes(1);
  });
});
