import { describe, expect, it, vi } from "vitest";
import {
  LlmClient,
  isPermanentLlmError,
  rejectedOptionalParameter,
} from "../../src/llm/client";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type { HttpRequest, HttpResponse } from "../../src/core/adapters";
import type { LlmSettings } from "../../src/settings/types";

const messages = [{ role: "user" as const, content: "x" }];

function response(status: number, bodyText: string): HttpResponse {
  return { status, bodyText, headers: {} };
}

function openAiError(message: string): string {
  return JSON.stringify({ error: { message, type: "invalid_request_error" } });
}

const ok = `data: ${JSON.stringify({ choices: [{ delta: { content: "fine" } }] })}\n\ndata: [DONE]\n\n`;
const okJson = JSON.stringify({ choices: [{ message: { content: "Hi" } }] });

/**
 * An endpoint that knows only `known` optional fields and answers any other one
 * the way OpenAI does. Records every body it was sent.
 */
function strictEndpoint(rejectMessage: (field: string) => string, knows: string[] = []) {
  const sent: Array<Record<string, unknown>> = [];
  const optional = ["thinking", "output_config", "reasoning_effort", "temperature"];
  const request = vi.fn(async (req: HttpRequest) => {
    const body = JSON.parse(String(req.body)) as Record<string, unknown>;
    sent.push(body);
    const unknown = optional.find((field) => field in body && !knows.includes(field));
    if (unknown) return response(400, openAiError(rejectMessage(unknown)));
    return response(200, body.stream ? ok : okJson);
  });
  return { sent, request };
}

function clientFor(
  request: (req: HttpRequest) => Promise<HttpResponse>,
  settings: Partial<LlmSettings> = {},
  logger = new Logger("error"),
) {
  return new LlmClient(
    { ...DEFAULT_SETTINGS.llm, apiKey: "sk-full-secret", model: "gpt-4.1", ...settings },
    logger,
    { request },
  );
}

describe("optional request fields a provider rejects", () => {
  it("drops thinking when the DeepSeek preset points at an OpenAI-style endpoint", async () => {
    // The reported case: provider left on DeepSeek (thinking on), base URL elsewhere.
    const { sent, request } = strictEndpoint(
      (field) => `Unknown parameter: '${field}'.`,
      ["reasoning_effort"],
    );
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const llm = clientFor(request, { provider: "deepseek", thinkingMode: true }, logger);

    await expect(llm.call(messages)).resolves.toBe("fine");
    expect(sent[0]).toHaveProperty("thinking");
    expect(sent[1]).not.toHaveProperty("thinking");
    expect(sent[1]).toHaveProperty("reasoning_effort", DEFAULT_SETTINGS.llm.reasoningEffort);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('"thinking"'));

    // The next call of the same run does not pay for the rejection again.
    await expect(llm.call(messages)).resolves.toBe("fine");
    expect(sent).toHaveLength(3);
    expect(sent[2]).not.toHaveProperty("thinking");
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("drops several rejected fields one after another", async () => {
    const { sent, request } = strictEndpoint((field) => `Unsupported parameter: '${field}' is not supported with this model.`);
    const llm = clientFor(request, { provider: "deepseek", thinkingMode: true });

    await expect(llm.call(messages)).resolves.toBe("fine");
    expect(sent).toHaveLength(3);
    expect(sent[2]).not.toHaveProperty("thinking");
    expect(sent[2]).not.toHaveProperty("reasoning_effort");
  });

  it("drops a temperature the model does not accept", async () => {
    const { sent, request } = strictEndpoint(
      (field) => `Unsupported value: '${field}' does not support 0.2 with this model. Only the default (1) value is supported.`,
      ["thinking", "reasoning_effort"],
    );
    const llm = clientFor(request, { provider: "openai", thinkingMode: false });

    await expect(llm.call(messages)).resolves.toBe("fine");
    expect(sent[0]).toHaveProperty("temperature");
    expect(sent[1]).not.toHaveProperty("temperature");
  });

  it("lets the connection test pass where a real call would", async () => {
    const { sent, request } = strictEndpoint(
      (field) => `Unknown parameter: '${field}'.`,
      ["reasoning_effort"],
    );
    const llm = clientFor(request, { provider: "deepseek", thinkingMode: true });

    await expect(llm.testConnection()).resolves.toEqual({ success: true });
    expect(sent[1]).not.toHaveProperty("thinking");
  });

  it("keeps any other 400 permanent and sends it once", async () => {
    const request = vi.fn(async () => response(400, openAiError("Invalid model: gpt-nope")));
    const error = await clientFor(request, { provider: "deepseek" }).call(messages).catch((e: unknown) => e);

    expect(isPermanentLlmError(error)).toBe(true);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("drops fields an Anthropic proxy rejects as OpenAI-compatible parameters", async () => {
    const { sent, request } = strictEndpoint((field) => `Unknown parameter: '${field}'.`);
    const llm = clientFor(request, {
      provider: "anthropic",
      baseUrl: "https://proxy.example.com/v1",
      model: "claude-opus-4-7",
      thinkingMode: true,
    });

    await expect(llm.call(messages)).resolves.toBe("fine");
    expect(sent[0]).toHaveProperty("reasoning_effort");
    expect(sent[0]).not.toHaveProperty("thinking");
    expect(sent[0]).not.toHaveProperty("output_config");
    expect(sent[1]).not.toHaveProperty("reasoning_effort");
  });

  it("never drops thinking from native Anthropic requests", async () => {
    const request = vi.fn(async () => response(400, JSON.stringify({
      type: "error",
      error: { type: "invalid_request_error", message: "thinking.type.enabled is not supported for this model." },
    })));
    const llm = clientFor(request, {
      provider: "anthropic",
      baseUrl: "https://api.anthropic.com/v1",
      model: "claude-sonnet-4-5",
      thinkingMode: true,
    });

    const error = await llm.call(messages).catch((e: unknown) => e);
    expect(isPermanentLlmError(error)).toBe(true);
    expect(request).toHaveBeenCalledTimes(1);
  });

  describe("rejectedOptionalParameter", () => {
    const body = { thinking: { type: "enabled" }, reasoning_effort: "high", temperature: 0.2 };
    const status400 = (message: string) => Object.assign(new Error(message), { status: 400 });
    const deepseek = { provider: "deepseek", baseUrl: "https://api.deepseek.com/v1" };

    it.each([
      ["Unknown parameter: 'thinking'.", "thinking"],
      ["Unrecognized request argument supplied: reasoning_effort", "reasoning_effort"],
      ["Unsupported parameter: 'reasoning_effort' is not supported with this model.", "reasoning_effort"],
      ["Unsupported value: 'temperature' does not support 0.2 with this model.", "temperature"],
      ["[{'type': 'extra_forbidden', 'loc': ('body', 'thinking'), 'msg': 'Extra inputs are not permitted'}]", "thinking"],
    ])("reads %j as rejecting %s", (message, field) => {
      expect(rejectedOptionalParameter(status400(message), body, deepseek)).toBe(field);
    });

    it("ignores errors that do not name a field as rejected", () => {
      for (const message of [
        "Invalid model: gpt-nope",
        "invalid_request_error: max_tokens is too large",
        "Your thinking budget must be at least 1024 tokens",
      ]) {
        expect(rejectedOptionalParameter(status400(message), body, deepseek)).toBeUndefined();
      }
    });

    it("only names fields the request carried, and only on 400 or 422", () => {
      const err = status400("Unknown parameter: 'thinking'.");
      expect(rejectedOptionalParameter(err, { temperature: 0.2 }, deepseek)).toBeUndefined();
      expect(rejectedOptionalParameter(
        Object.assign(new Error("Unknown parameter: 'thinking'."), { status: 401 }), body, deepseek,
      )).toBeUndefined();
      expect(rejectedOptionalParameter(
        Object.assign(new Error("Unknown parameter: 'thinking'."), { status: 422 }), body, deepseek,
      )).toBe("thinking");
    });
  });
});
