import { afterEach, describe, expect, it, vi } from "vitest";
import {
  LlmClient,
  LlmContentBlockedError,
  LlmOutputTruncatedError,
  LlmTransientExhaustedError,
  isPermanentLlmError,
} from "../../src/llm/client";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type {
  HttpClient,
  HttpRequest,
  HttpResponse,
  HttpStreamResponse,
} from "../../src/core/adapters";

const messages = [{ role: "user" as const, content: "x" }];
const PARTIAL = "# Title\n\n## 研究问题\n部分内容，被截断于此";

const data = (event: unknown) => `data: ${JSON.stringify(event)}\n\n`;
const text = (content: string, finish?: string) =>
  data({ choices: [{ delta: { content }, ...(finish ? { finish_reason: finish } : {}) }] });
const DONE = "data: [DONE]\n\n";

function ok(bodyText: string): HttpResponse {
  return { status: 200, headers: {}, bodyText };
}

function clientFor(bodies: string[], logger = new Logger("error")) {
  const request = vi.fn(async (_req: HttpRequest): Promise<HttpResponse> => {
    const body = bodies[Math.min(request.mock.calls.length - 1, bodies.length - 1)]!;
    return ok(body);
  });
  const llm = new LlmClient(
    { ...DEFAULT_SETTINGS.llm, apiKey: "sk-full-secret", model: "test" },
    logger,
    { request } satisfies HttpClient,
  );
  return { llm, request };
}

function anthropicFor(body: string) {
  const request = vi.fn(async (): Promise<HttpResponse> => ok(body));
  const llm = new LlmClient(
    {
      ...DEFAULT_SETTINGS.llm, provider: "anthropic", baseUrl: "https://api.anthropic.com/v1",
      apiKey: "sk-full-secret", model: "claude-sonnet-4-6", thinkingMode: false,
    },
    new Logger("error"),
    { request },
  );
  return { llm, request };
}

/** Runs `call()` to its outcome, letting the retry back-off elapse on fake timers. */
async function outcome(run: Promise<string>): Promise<unknown> {
  const settled = run.then((value) => ({ value }), (error: unknown) => ({ error }));
  await vi.runAllTimersAsync();
  const result = await settled;
  if ("error" in result) return result.error;
  throw new Error(`call() returned ${JSON.stringify(result.value)} instead of failing`);
}

describe("an unfinished reply is never returned as an answer", () => {
  afterEach(() => vi.useRealTimers());

  it.each(["content_filter", "sensitive", "refusal"])(
    "types a reply the provider stopped with finish_reason %s and does not resend it",
    async (reason) => {
      vi.useFakeTimers();
      const { llm, request } = clientFor([text(PARTIAL) + data({ choices: [{ delta: {}, finish_reason: reason }] }) + DONE]);

      const error = await outcome(llm.call(messages));

      expect(error).toBeInstanceOf(LlmContentBlockedError);
      expect((error as Error).message).toContain(reason);
      expect((error as Error).message).not.toContain("部分内容");
      expect(isPermanentLlmError(error)).toBe(true);
      expect(request).toHaveBeenCalledTimes(1);
    },
  );

  it.each(["error", "network_error", "insufficient_system_resource", "abort", "timeout"])(
    "treats finish_reason %s as an interrupted stream and re-sends once",
    async (reason) => {
      vi.useFakeTimers();
      const { llm, request } = clientFor([text(PARTIAL) + data({ choices: [{ delta: {}, finish_reason: reason }] }) + DONE]);

      const error = await outcome(llm.call(messages));

      expect(error).toBeInstanceOf(LlmTransientExhaustedError);
      expect((error as Error).message).toContain(reason);
      expect(isPermanentLlmError(error)).toBe(false);
      expect(request).toHaveBeenCalledTimes(2);
    },
  );

  it("recovers when the re-sent request finishes normally", async () => {
    vi.useFakeTimers();
    const { llm, request } = clientFor([
      text(PARTIAL),
      text("complete", "stop") + DONE,
    ]);

    const run = llm.call(messages);
    await vi.runAllTimersAsync();

    await expect(run).resolves.toBe("complete");
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("rejects a stream that just ends, with no finish reason and no [DONE]", async () => {
    vi.useFakeTimers();
    const { llm, request } = clientFor([text("part one, ") + text("part two")]);

    const error = await outcome(llm.call(messages));

    expect(error).toBeInstanceOf(LlmTransientExhaustedError);
    expect((error as Error).message).toMatch(/before the provider reported completion/);
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("rejects an empty 200 body instead of returning an empty answer", async () => {
    vi.useFakeTimers();
    const { llm } = clientFor([""]);

    expect(await outcome(llm.call(messages))).toBeInstanceOf(LlmTransientExhaustedError);
  });

  it("rejects an in-stream error event that is followed by [DONE]", async () => {
    vi.useFakeTimers();
    const { llm, request } = clientFor([
      text(PARTIAL) + data({ error: { message: "upstream provider disconnected", code: "server_error" } }) + DONE,
    ]);

    const error = await outcome(llm.call(messages));

    expect(error).toBeInstanceOf(LlmTransientExhaustedError);
    expect((error as Error).message).toBe("upstream provider disconnected");
    // A server-side failure is worth the usual attempts.
    expect(request).toHaveBeenCalledTimes(3);
  });

  it("does not repeat an in-stream error that the provider says is the request's fault", async () => {
    vi.useFakeTimers();
    const { llm, request } = clientFor([
      data({ error: { message: "invalid api key sk-full-secret", code: 401 } }) + DONE,
    ]);

    const error = await outcome(llm.call(messages));

    expect(isPermanentLlmError(error)).toBe(true);
    expect((error as { status?: number }).status).toBe(401);
    expect((error as Error).message).not.toContain("sk-full-secret");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("types an in-stream content-safety error like an HTTP refusal", async () => {
    vi.useFakeTimers();
    const { llm, request } = clientFor([
      text(PARTIAL) + data({ error: { message: "Content Exists Risk", code: "invalid_request_error" } }) + DONE,
    ]);

    const error = await outcome(llm.call(messages));

    expect(error).toBeInstanceOf(LlmContentBlockedError);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("accepts an error object that is null on a normal chunk", async () => {
    const { llm } = clientFor([data({ choices: [{ delta: { content: "fine" } }], error: null }) + text("", "stop") + DONE]);

    await expect(llm.call(messages)).resolves.toBe("fine");
  });

  it("keeps text cut off at the token limit a truncation, not an interruption", async () => {
    const { llm, request } = clientFor([text(PARTIAL) + data({ choices: [{ delta: {}, finish_reason: "length" }] }) + DONE]);

    await expect(llm.call(messages)).rejects.toBeInstanceOf(LlmOutputTruncatedError);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("rejects a native Anthropic stream that ends before message_stop", async () => {
    vi.useFakeTimers();
    const { llm, request } = anthropicFor([
      data({ type: "message_start", message: { usage: { input_tokens: 3 } } }),
      data({ type: "content_block_delta", delta: { type: "text_delta", text: PARTIAL } }),
    ].join(""));

    const error = await outcome(llm.call(messages));

    expect(error).toBeInstanceOf(LlmTransientExhaustedError);
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("types an Anthropic refusal stop_reason as a content block", async () => {
    const { llm, request } = anthropicFor([
      data({ type: "content_block_delta", delta: { type: "text_delta", text: PARTIAL } }),
      data({ type: "message_delta", delta: { stop_reason: "refusal" } }),
      data({ type: "message_stop" }),
    ].join(""));

    await expect(llm.call(messages)).rejects.toBeInstanceOf(LlmContentBlockedError);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("still fails an Anthropic overloaded error event as a transient failure", async () => {
    vi.useFakeTimers();
    const { llm, request } = anthropicFor([
      data({ type: "content_block_delta", delta: { type: "text_delta", text: PARTIAL } }),
      data({ type: "error", error: { type: "overloaded_error", message: "Overloaded" } }),
    ].join(""));

    const error = await outcome(llm.call(messages));

    expect(error).toBeInstanceOf(LlmTransientExhaustedError);
    expect((error as Error).message).toBe("Overloaded");
    expect(request).toHaveBeenCalledTimes(3);
  });
});

describe("a server that ignores stream: true", () => {
  afterEach(() => vi.useRealTimers());

  const completion = (content: string | null, finish = "stop") => JSON.stringify({
    id: "chatcmpl-1",
    object: "chat.completion",
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: finish }],
    usage: { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15 },
  });

  it("uses a complete plain JSON reply, with its usage", async () => {
    const { llm } = clientFor([completion("# Title\n\nwhole note")]);
    const onMetrics = vi.fn();

    await expect(llm.call(messages, { onMetrics })).resolves.toBe("# Title\n\nwhole note");
    expect(onMetrics).toHaveBeenCalledWith(expect.objectContaining({
      usageComplete: true, inputTokens: 11, outputTokens: 4, totalTokens: 15,
    }));
  });

  it("uses a pretty-printed plain JSON reply", async () => {
    const { llm } = clientFor([JSON.stringify(JSON.parse(completion("ok")), null, 2)]);

    await expect(llm.call(messages)).resolves.toBe("ok");
  });

  it("still applies the finish reason to a plain JSON reply", async () => {
    await expect(clientFor([completion("cut", "length")]).llm.call(messages))
      .rejects.toBeInstanceOf(LlmOutputTruncatedError);
    await expect(clientFor([completion(null, "content_filter")]).llm.call(messages))
      .rejects.toBeInstanceOf(LlmContentBlockedError);
  });

  it("reports a plain JSON error body with the provider's message", async () => {
    vi.useFakeTimers();
    const { llm } = clientFor([JSON.stringify({ error: { message: "gateway is over capacity" } })]);

    const error = await outcome(llm.call(messages));

    expect((error as Error).message).toBe("gateway is over capacity");
  });

  it("explains a reply that is neither an event stream nor a chat completion", async () => {
    vi.useFakeTimers();
    const { llm, request } = clientFor(["<html><body>Please sign in to continue</body></html>"]);

    const error = await outcome(llm.call(messages));

    expect(error).toBeInstanceOf(LlmTransientExhaustedError);
    expect((error as Error).message).toMatch(/not an event stream/);
    expect((error as Error).message).toContain("Please sign in");
    expect(request).toHaveBeenCalledTimes(2);
  });
});

describe("streams that finish normally are unaffected", () => {
  it.each([
    ["stop", text("answer", "stop") + DONE],
    ["stop without [DONE]", text("answer", "stop")],
    ["[DONE] without a finish reason", text("answer") + DONE],
    ["a separate finish chunk and a usage chunk", text("answer") +
      data({ choices: [{ delta: {}, finish_reason: "stop" }] }) +
      data({ choices: [], usage: { prompt_tokens: 1, completion_tokens: 1 } }) + DONE],
    ["keep-alive comments before [DONE]", ": keep-alive\n\n" + text("answer") + ": keep-alive\n\n" + text("", "stop") + ": keep-alive\n\n" + DONE],
    ["CRLF separators", (text("answer", "stop") + DONE).replace(/\n/g, "\r\n")],
    ["[DONE] as the last line, unterminated", text("answer") + "data: [DONE]"],
    ["an upper-case STOP", text("answer", "STOP") + DONE],
    ["end_turn relayed by a gateway", text("answer", "end_turn") + DONE],
    ["tool_calls", text("answer", "tool_calls") + DONE],
  ])("returns the text for %s", async (_name, body) => {
    const { llm, request } = clientFor([body]);

    await expect(llm.call(messages)).resolves.toBe("answer");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("accepts an unrecognised finish reason only when the stream ends with [DONE], and says so", async () => {
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const { llm } = clientFor([text("answer", "all_done") + DONE], logger);

    await expect(llm.call(messages)).resolves.toBe("answer");
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("all_done"));
  });

  it("rejects an unrecognised finish reason when nothing else says the stream completed", async () => {
    vi.useFakeTimers();
    try {
      const { llm } = clientFor([text("answer", "all_done")]);

      expect(await outcome(llm.call(messages))).toBeInstanceOf(LlmTransientExhaustedError);
    } finally {
      vi.useRealTimers();
    }
  });

  it("accepts an Anthropic end_turn and stop_sequence", async () => {
    for (const stopReason of ["end_turn", "stop_sequence"]) {
      const { llm } = anthropicFor([
        data({ type: "content_block_delta", delta: { type: "text_delta", text: "answer" } }),
        data({ type: "message_delta", delta: { stop_reason: stopReason } }),
        data({ type: "message_stop" }),
      ].join(""));

      await expect(llm.call(messages)).resolves.toBe("answer");
    }
  });
});

describe("the same rules over a host that streams the body", () => {
  afterEach(() => vi.useRealTimers());

  function streamingClient(wire: string[], settings: Partial<typeof DEFAULT_SETTINGS.llm> = {}) {
    const requestStream = vi.fn(async (_req: HttpRequest): Promise<HttpStreamResponse> => ({
      status: 200,
      headers: {},
      body: (async function* () { for (const part of wire) yield part; })(),
    }));
    const llm = new LlmClient(
      { ...DEFAULT_SETTINGS.llm, apiKey: "sk-full-secret", model: "test", ...settings },
      new Logger("error"),
      { request: vi.fn(), requestStream } as HttpClient,
    );
    return { llm, requestStream };
  }

  it("returns a reply that finished", async () => {
    const wire = text("hel") + text("lo") + data({ choices: [{ delta: {}, finish_reason: "stop" }] }) + DONE;
    const { llm } = streamingClient([wire.slice(0, 20), wire.slice(20)]);

    await expect(llm.call(messages)).resolves.toBe("hello");
  });

  it("rejects a body that closes cleanly in the middle of a reply", async () => {
    vi.useFakeTimers();
    const { llm, requestStream } = streamingClient([text("hel"), text("lo")]);

    expect(await outcome(llm.call(messages))).toBeInstanceOf(LlmTransientExhaustedError);
    expect(requestStream).toHaveBeenCalledTimes(2);
  });

  it("reads a plain JSON reply delivered in several network chunks", async () => {
    const body = JSON.stringify({ choices: [{ message: { content: "chunked" }, finish_reason: "stop" }] });
    const { llm } = streamingClient([body.slice(0, 10), body.slice(10, 30), body.slice(30)]);

    await expect(llm.call(messages)).resolves.toBe("chunked");
  });

  it("rejects a native Anthropic body that stops before message_stop", async () => {
    vi.useFakeTimers();
    const { llm } = streamingClient(
      [data({ type: "content_block_delta", delta: { type: "text_delta", text: "hi" } })],
      { provider: "anthropic", baseUrl: "https://api.anthropic.com" },
    );

    expect(await outcome(llm.call(messages))).toBeInstanceOf(LlmTransientExhaustedError);
  });
});
