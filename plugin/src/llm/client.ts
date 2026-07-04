import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import type { LlmSettings } from "../settings/types";
import type { HttpClient } from "../core/adapters";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";

const LLM_TIMEOUT_MS = 300_000; // 5 minutes
export const LLM_STREAM_IDLE_TIMEOUT_MS = 120_000;
export const LLM_TEMPERATURE = 0.1;

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CallOptions {
  /** Overrides default temperature. Ignored when thinkingMode = true. */
  temperature?: number;
  signal?: AbortSignal;
}

interface LlmStatusError extends Error {
  status?: number;
}

const KNOWN_MODEL_BASE_SUFFIXES = [
  "/api/claudecode",
  "/api/anthropic",
  "/apps/anthropic",
  "/api/coding",
  "/claudecode",
  "/anthropic",
  "/step_plan",
  "/coding",
  "/claude",
];

export function buildModelUrlCandidates(baseUrl: string): string[] {
  const normalized = baseUrl.replace(/\/+$/, "");
  const candidates: string[] = [];
  const addCandidate = (url: string): void => {
    if (!candidates.includes(url)) candidates.push(url);
  };

  if (normalized.endsWith("/v1")) {
    addCandidate(`${normalized}/models`);
  } else {
    addCandidate(`${normalized}/v1/models`);
  }

  for (const suffix of KNOWN_MODEL_BASE_SUFFIXES) {
    if (normalized.endsWith(suffix)) {
      const stripped = normalized.slice(0, -suffix.length);
      addCandidate(`${stripped}/v1/models`);
      break;
    }
  }

  addCandidate(`${normalized}/models`);
  return candidates;
}

export function normalizeOpenAiBaseUrl(baseUrl: string): string {
  const normalized = baseUrl.trim().replace(/\/+$/, "");
  if (!normalized) return normalized;
  if (normalized.endsWith("/v1")) return normalized;

  try {
    const parsed = new URL(normalized);
    const path = parsed.pathname.replace(/\/+$/, "");
    if (!path || path === "/") {
      return `${normalized}/v1`;
    }
  } catch {
    return normalized;
  }

  return normalized;
}

export class LlmClient {
  constructor(
    private settings: LlmSettings,
    private logger: Logger,
    private http?: HttpClient,
  ) {}

  async testConnection(): Promise<{ success: boolean; error?: string }> {
    try {
      await this.postChatJson({
        model: this.settings.model,
        messages: [{ role: "user", content: "Hello" }],
        max_tokens: 5,
      });
      return { success: true };
    } catch (e) {
      return { success: false, error: (e as Error).message };
    }
  }

  async fetchModels(): Promise<string[]> {
    const baseUrl = this.settings.baseUrl.replace(/\/+$/, "");
    const apiKey = this.settings.apiKey;

    if (!baseUrl || !apiKey) {
      throw new Error("Please fill in API Base URL and API Key first");
    }

    const candidates = buildModelUrlCandidates(baseUrl);

    for (const url of candidates) {
      try {
        let data: unknown;
        if (this.http) {
          const res = await this.http.request({
            url,
            method: "GET",
            headers: {
              "Authorization": `Bearer ${apiKey}`,
              "Content-Type": "application/json",
            },
          });
          if (res.status >= 200 && res.status < 300) {
            data = JSON.parse(res.bodyText);
            return this.parseModelList(data);
          }
        } else {
          const response = await fetch(url, {
            method: "GET",
            headers: {
              "Authorization": `Bearer ${apiKey}`,
              "Content-Type": "application/json",
            },
          });
          if (response.ok) {
            data = await response.json();
            return this.parseModelList(data);
          }
        }
      } catch {
        // Try next candidate
        continue;
      }
    }

    throw new Error("Failed to fetch models from any endpoint");
  }

  private parseModelList(data: unknown): string[] {
    if (
      data &&
      typeof data === "object" &&
      "data" in data &&
      Array.isArray((data as { data: unknown }).data)
    ) {
      return (data as { data: Array<{ id?: string }> }).data
        .map((model) => model.id)
        .filter((id): id is string => Boolean(id));
    }
    if (Array.isArray(data)) {
      return data
        .map((model: { id?: string; name?: string }) => model.id || model.name)
        .filter((id): id is string => Boolean(id));
    }
    throw new Error("Invalid model list format");
  }

  async call(messages: ChatMessage[], opts: CallOptions = {}): Promise<string> {
    return retry(
      async () => {
        throwIfCancelled(opts.signal);
        const params: Record<string, unknown> = {
          model: this.settings.model,
          messages,
          stream: true,
        };
        if (this.settings.thinkingMode) {
          if (this.settings.provider === "anthropic") {
            // Anthropic extended thinking via OpenAI-compat proxy
            const budgets: Record<string, number> = {
              low: 2048,
              medium: 8192,
              high: 16384,
            };
            (params as any).extra_body = {
              thinking: {
                type: "enabled",
                budget_tokens: budgets[this.settings.reasoningEffort] ?? 8192,
              },
            };
          } else {
            params.reasoning_effort = this.settings.reasoningEffort;
            (params as any).extra_body = { thinking: { type: "enabled" } };
          }
        } else {
          params.temperature = opts.temperature ?? LLM_TEMPERATURE;
        }
        const abort = createAttemptAbortController(opts.signal);
        try {
          const content = await this.postChatStream(
            params,
            abort.controller,
            opts.signal,
          );
          throwIfCancelled(opts.signal);
          return content;
        } finally {
          abort.cleanup();
        }
      },
      {
        maxAttempts: 3,
        baseDelayMs: 5000,
        signal: opts.signal,
        shouldRetry: (err) =>
          !isCancellationError(err) && !isPermanentLlmError(err),
        onRetry: (err, attempt, wait) =>
          this.logger.warn(
            `LLM retry #${attempt} after ${wait}ms: ${(err as Error).message}`,
          ),
      },
    );
  }

  private async postChatJson(
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<unknown> {
    const res = await this.requestChat(body, false, signal);
    return JSON.parse(res);
  }

  private async postChatStream(
    body: Record<string, unknown>,
    controller: AbortController,
    signal?: AbortSignal,
  ): Promise<string> {
    const requestBody = { ...body, stream: true };
    if (this.http) {
      const raw = await this.requestChat(requestBody, true, signal);
      return collectStreamWithIdleTimeout(
        parseSseText(raw),
        controller,
        LLM_STREAM_IDLE_TIMEOUT_MS,
        signal,
      );
    }

    const response = await fetch(this.chatCompletionsUrl(), {
      method: "POST",
      headers: this.chatHeaders(),
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw createStatusError(response.status, await response.text());
    }
    if (!response.body) {
      return collectStreamWithIdleTimeout(
        parseSseText(await response.text()),
        controller,
        LLM_STREAM_IDLE_TIMEOUT_MS,
        signal,
      );
    }
    return collectStreamWithIdleTimeout(
      parseSseReadableStream(response.body),
      controller,
      LLM_STREAM_IDLE_TIMEOUT_MS,
      signal,
    );
  }

  private async requestChat(
    body: Record<string, unknown>,
    stream: boolean,
    signal?: AbortSignal,
  ): Promise<string> {
    const requestBody = { ...body, stream };
    if (this.http) {
      const res = await this.http.request({
        url: this.chatCompletionsUrl(),
        method: "POST",
        headers: this.chatHeaders(),
        body: JSON.stringify(requestBody),
        timeoutMs: LLM_TIMEOUT_MS,
        signal,
      });
      if (res.status < 200 || res.status >= 300) {
        throw createStatusError(res.status, res.bodyText);
      }
      return res.bodyText;
    }

    const response = await fetch(this.chatCompletionsUrl(), {
      method: "POST",
      headers: this.chatHeaders(),
      body: JSON.stringify(requestBody),
      signal,
    });
    const text = await response.text();
    if (!response.ok) throw createStatusError(response.status, text);
    return text;
  }

  private chatCompletionsUrl(): string {
    return `${normalizeOpenAiBaseUrl(this.settings.baseUrl)}/chat/completions`;
  }

  private chatHeaders(): Record<string, string> {
    return {
      "Authorization": `Bearer ${this.settings.apiKey}`,
      "Content-Type": "application/json",
    };
  }
}

export function isPermanentLlmError(err: unknown): boolean {
  const status = (err as LlmStatusError | undefined)?.status;
  return (
    typeof status === "number" &&
    status >= 400 &&
    status < 500 &&
    status !== 429
  );
}

function createStatusError(status: number, bodyText: string): LlmStatusError {
  let message = bodyText.trim();
  try {
    const parsed = JSON.parse(bodyText) as { error?: { message?: string } };
    message = parsed.error?.message ?? message;
  } catch {
    // Use raw response text.
  }
  const error = new Error(message || `LLM request failed with HTTP ${status}`) as LlmStatusError;
  error.status = status;
  return error;
}

async function* parseSseText(raw: string): AsyncIterable<unknown> {
  for (const event of raw.split(/\r?\n\r?\n/)) {
    const parsed = parseSseEvent(event);
    if (parsed.done) return;
    if (parsed.value !== undefined) yield parsed.value;
  }
}

async function* parseSseReadableStream(
  stream: ReadableStream<Uint8Array>,
): AsyncIterable<unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() ?? "";
      for (const event of events) {
        const parsed = parseSseEvent(event);
        if (parsed.done) return;
        if (parsed.value !== undefined) yield parsed.value;
      }
    }
    buffer += decoder.decode();
    const parsed = parseSseEvent(buffer);
    if (!parsed.done && parsed.value !== undefined) yield parsed.value;
  } finally {
    reader.releaseLock();
  }
}

function parseSseEvent(event: string): { done: boolean; value?: unknown } {
  const data = event
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trim())
    .join("\n");
  if (!data) return { done: false };
  if (data === "[DONE]") return { done: true };
  return { done: false, value: JSON.parse(data) };
}

export async function collectStreamWithIdleTimeout(
  stream: AsyncIterable<unknown>,
  controller: AbortController,
  idleTimeoutMs = LLM_STREAM_IDLE_TIMEOUT_MS,
  signal?: AbortSignal,
): Promise<string> {
  const chunks: string[] = [];
  const iterator = stream[Symbol.asyncIterator]();
  while (true) {
    throwIfCancelled(signal);
    const next = await nextStreamChunk(iterator, controller, idleTimeoutMs, signal);
    if (next.done) break;
    throwIfCancelled(signal);
    const delta = streamDeltaContent(next.value);
    if (delta) chunks.push(delta);
  }
  throwIfCancelled(signal);
  return chunks.join("");
}

function nextStreamChunk(
  iterator: AsyncIterator<unknown>,
  controller: AbortController,
  idleTimeoutMs: number,
  signal?: AbortSignal,
): Promise<IteratorResult<unknown>> {
  throwIfCancelled(signal);
  return new Promise((resolve, reject) => {
    let settled = false;
    const timeout = setTimeout(() => {
      finish();
      controller.abort("LLM stream idle timeout");
      reject(new Error("LLM stream idle timeout"));
    }, idleTimeoutMs);
    const onAbort = () => {
      finish();
      try {
        throwIfCancelled(signal);
      } catch (e) {
        reject(e);
      }
    };
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) {
      onAbort();
      return;
    }
    iterator.next().then(
      (value) => {
        finish();
        resolve(value);
      },
      (error) => {
        finish();
        reject(error);
      },
    );
  });
}

function streamDeltaContent(chunk: unknown): string | undefined {
  const choices = (chunk as { choices?: Array<{ delta?: { content?: unknown } }> })
    .choices;
  const content = choices?.[0]?.delta?.content;
  return typeof content === "string" ? content : undefined;
}

function createAttemptAbortController(signal?: AbortSignal): {
  controller: AbortController;
  cleanup: () => void;
} {
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal?.reason);
  signal?.addEventListener("abort", onAbort, { once: true });
  if (signal?.aborted) onAbort();
  return {
    controller,
    cleanup: () => signal?.removeEventListener("abort", onAbort),
  };
}
