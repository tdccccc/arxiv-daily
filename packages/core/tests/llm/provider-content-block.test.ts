import { describe, expect, it, vi } from "vitest";
import {
  LlmClient,
  LlmContentBlockedError,
  isLlmContentBlockedError,
  isPermanentLlmError,
} from "../../src/llm/client";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type { HttpClient, HttpRequest, HttpResponse } from "../../src/core/adapters";

const messages = [{ role: "user" as const, content: "x" }];

function llmWith(request: (req: HttpRequest) => Promise<HttpResponse>) {
  return new LlmClient(
    { ...DEFAULT_SETTINGS.llm, apiKey: "sk-full-secret", model: "test" },
    new Logger("error"),
    { request } satisfies HttpClient,
  );
}

function rejectedWith(status: number, body: unknown) {
  const request = vi.fn(async (): Promise<HttpResponse> => ({
    status,
    bodyText: typeof body === "string" ? body : JSON.stringify(body),
    headers: {},
  }));
  return { request, call: () => llmWith(request).call(messages).catch((error: unknown) => error) };
}

// Bodies as the providers document or return them for input their content-safety filter refuses.
const REFUSALS: Array<[string, unknown, string]> = [
  ["DeepSeek", {
    error: { message: "Content Exists Risk", type: "invalid_request_error", param: null, code: "invalid_request_error" },
  }, "Content Exists Risk"],
  ["Zhipu GLM 1301", {
    error: { code: "1301", message: "系统检测到输入或生成内容可能包含不安全或敏感内容，请您避免输入易产生敏感内容的提示语，感谢您的配合。" },
  }, "不安全或敏感内容"],
  ["Azure OpenAI", {
    error: {
      message: "The response was filtered due to the prompt triggering Azure OpenAI's content management policy.",
      type: null, param: "prompt", code: "content_filter", status: 400,
    },
  }, "content management policy"],
  ["Moonshot", {
    error: { message: "The request was rejected because it was considered high risk", type: "content_filter" },
  }, "considered high risk"],
  ["Alibaba DashScope", {
    error: { code: "data_inspection_failed", message: "Input data may contain inappropriate content.", type: "data_inspection_failed" },
  }, "inappropriate content"],
  ["OpenAI usage policy", {
    error: {
      message: "Invalid prompt: your prompt was flagged as potentially violating our usage policy.",
      type: "invalid_request_error", param: null, code: "invalid_prompt",
    },
  }, "flagged as potentially violating"],
];

describe("provider content-safety refusal", () => {
  it.each(REFUSALS)("types an HTTP 400 from %s and does not repeat the request", async (_name, body, text) => {
    const { request, call } = rejectedWith(400, body);

    const error = await call();

    expect(error).toBeInstanceOf(LlmContentBlockedError);
    expect(isLlmContentBlockedError(error)).toBe(true);
    expect(error).toMatchObject({ code: "ARXIV_LLM_CONTENT_BLOCKED", status: 400 });
    expect((error as Error).message).toContain(text);
    // The same input is refused again, so the client must not retry it.
    expect(request).toHaveBeenCalledTimes(1);
    // Callers that predate the type still see a permanent failure.
    expect(isPermanentLlmError(error)).toBe(true);
  });

  it("keeps the API key out of the typed error", async () => {
    const { call } = rejectedWith(400, {
      error: { message: "Content Exists Risk for key sk-full-secret", code: "content_filter" },
    });

    const error = await call();

    expect(error).toBeInstanceOf(LlmContentBlockedError);
    expect((error as Error).message).not.toContain("sk-full-secret");
  });

  it("reports a refused connection test as a failure with the provider's text", async () => {
    const { request } = rejectedWith(400, REFUSALS[0]![1]);

    await expect(llmWith(request).testConnection()).resolves.toEqual({
      success: false,
      error: "Content Exists Risk",
    });
  });

  // A request the provider cannot understand must stay a visible, permanent failure:
  // treating it as a content refusal would turn a bug into an all-fallback report.
  it.each([
    ["unsupported parameter", {
      error: { message: "Unsupported parameter: 'max_tokens' is not supported with this model.", type: "invalid_request_error", param: "max_tokens", code: "unsupported_parameter" },
    }],
    ["empty content block", {
      error: { message: "messages: text content blocks must be non-empty", type: "invalid_request_error" },
    }],
    ["context length", {
      error: { message: "This model's maximum context length is 128000 tokens.", type: "invalid_request_error", code: "context_length_exceeded" },
    }],
    ["wrong content type", {
      error: { message: "Invalid content type for message 0", type: "invalid_request_error", code: "invalid_value" },
    }],
    ["invalid_prompt without a policy reason", {
      error: { message: "Invalid prompt: the prompt is empty.", type: "invalid_request_error", code: "invalid_prompt" },
    }],
    ["plain text body", "Bad Request"],
  ])("does not mistake a 400 for %s as a content refusal", async (_name, body) => {
    const { request, call } = rejectedWith(400, body);

    const error = await call();

    expect(error).toBeInstanceOf(Error);
    expect(isLlmContentBlockedError(error)).toBe(false);
    expect(isPermanentLlmError(error)).toBe(true);
    expect(request).toHaveBeenCalledTimes(1);
  });

  // Account-wide failures say nothing about one input; they keep failing the whole run.
  it.each([
    [401, { error: { message: "Incorrect API key provided", type: "invalid_request_error", code: "invalid_api_key" } }],
    [402, { error: { message: "Insufficient Balance" } }],
    [403, { error: { message: "blocked by content filter policy", code: "content_filter" } }],
    [404, { error: { message: "The model `nope` does not exist", code: "model_not_found" } }],
  ])("leaves HTTP %i an ordinary permanent error", async (status, body) => {
    const { request, call } = rejectedWith(status, body);

    const error = await call();

    expect(isLlmContentBlockedError(error)).toBe(false);
    expect(isPermanentLlmError(error)).toBe(true);
    expect((error as { status?: number }).status).toBe(status);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("recognizes the typed error across a realm or copy boundary and nothing else", () => {
    const copy = Object.assign(new Error("Content Exists Risk"), {
      name: "LlmContentBlockedError",
      code: "ARXIV_LLM_CONTENT_BLOCKED",
    });
    expect(isLlmContentBlockedError(copy)).toBe(true);
    expect(isLlmContentBlockedError(new Error("Content Exists Risk"))).toBe(false);
    expect(isLlmContentBlockedError(Object.assign(new Error("x"), { name: "LlmContentBlockedError" }))).toBe(false);
    expect(isLlmContentBlockedError("Content Exists Risk")).toBe(false);
    expect(isPermanentLlmError(new LlmContentBlockedError("refused"))).toBe(true);
  });
});
