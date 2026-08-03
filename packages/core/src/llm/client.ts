import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import type { LlmSettings } from "../settings/types";
import type { HttpClient } from "../core/adapters";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import { redactError, redactText } from "../utils/redaction";
import {
  parseTokenUsage,
  usageIsComplete,
  type LlmCallMetrics,
  type MetricsObserver,
  type TokenUsage,
} from "../metrics/generation";

const LLM_TIMEOUT_MS = 300_000; // 5 minutes
export const LLM_STREAM_IDLE_TIMEOUT_MS = 120_000;
export const LLM_TEMPERATURE = 0.1;
export const LLM_MAX_OUTPUT_CODE_UNITS = 10_000_000;
export const LLM_MAX_COMPLETION_TOKENS = 1_000_000;
export const LLM_OUTPUT_LIMIT_EXCEEDED_ERROR_CODE = "ARXIV_LLM_OUTPUT_LIMIT_EXCEEDED" as const;

export class LlmOutputLimitExceededError extends Error {
  readonly code = LLM_OUTPUT_LIMIT_EXCEEDED_ERROR_CODE;

  constructor() {
    super("LLM output exceeded configured limit");
    this.name = "LlmOutputLimitExceededError";
  }
}

export class StreamIdleTimeoutError extends Error {
  constructor(message = "LLM stream idle timeout") {
    super(message);
    this.name = "StreamIdleTimeoutError";
  }
}

export const LLM_TRANSIENT_EXHAUSTED_ERROR_CODE =
  "ARXIV_LLM_TRANSIENT_EXHAUSTED" as const;

/** A logical LLM call exhausted the client's internal transient retries. */
export class LlmTransientExhaustedError extends Error {
  readonly code = LLM_TRANSIENT_EXHAUSTED_ERROR_CODE;
  readonly cause: Error;

  constructor(cause: Error) {
    super(cause.message);
    this.name = "LlmTransientExhaustedError";
    this.cause = cause;
  }
}

export function isLlmTransientExhaustedError(
  error: unknown,
): error is LlmTransientExhaustedError {
  if (error instanceof LlmTransientExhaustedError) return true;
  if (!isErrorLike(error) ||
      error.name !== "LlmTransientExhaustedError" ||
      error.code !== LLM_TRANSIENT_EXHAUSTED_ERROR_CODE) return false;
  return isErrorLike(error.cause);
}

function isErrorLike(value: unknown): value is Error & Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.name === "string" &&
    typeof candidate.message === "string" &&
    typeof candidate.stack === "string";
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CallOptions {
  /** Overrides default temperature. Ignored when thinkingMode = true. */
  temperature?: number;
  signal?: AbortSignal;
  onMetrics?: MetricsObserver;
  /** Reject accumulated response content above this UTF-16 code-unit count. */
  maxOutputCodeUnits?: number;
  /** Provider-compatible completion-token request bound (`max_tokens`). */
  maxCompletionTokens?: number;
}

interface StreamResult {
  content: string;
  usage?: TokenUsage;
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

/** Exact effective URL identity used by chat requests. */
export function buildChatCompletionsUrl(baseUrl: string): string {
  return `${normalizeOpenAiBaseUrl(baseUrl)}/chat/completions`;
}

export class LlmClient {
  constructor(
    private settings: LlmSettings,
    private logger: Logger,
    private http: HttpClient,
  ) {
    this.logger.setSensitiveValues?.([settings.apiKey]);
  }

  async testConnection(): Promise<{ success: boolean; error?: string }> {
    try {
      await this.postChatJson({
        model: this.settings.model,
        messages: [{ role: "user", content: "Hello" }],
        max_tokens: 5,
      });
      return { success: true };
    } catch (e) {
      return { success: false, error: this.safeError(e).message };
    }
  }

  async fetchModels(): Promise<string[]> {
    const baseUrl = this.settings.baseUrl.replace(/\/+$/, "");
    const apiKey = this.settings.apiKey;

    if (!baseUrl || !apiKey) {
      throw new Error("Please fill in API Base URL and API Key first");
    }

    const candidates = buildModelUrlCandidates(baseUrl);
    let lastError: unknown;

    for (const url of candidates) {
      try {
        const res = await this.http.request({
          url,
          method: "GET",
          headers: {
            "Authorization": `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
        });
        if (res.status >= 200 && res.status < 300) {
          return this.parseModelList(JSON.parse(res.bodyText));
        }
        throw createStatusError(res.status, res.bodyText, [apiKey]);
      } catch (e) {
        lastError = e;
        // Try next candidate
        continue;
      }
    }

    const safeLastError = this.safeError(lastError);
    const suffix = safeLastError.message ? `: ${safeLastError.message}` : "";
    throw new Error(`Failed to fetch models from any endpoint${suffix}`);
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
    const limits = validateCallLimits(opts);
    const started = Date.now();
    let attempts = 0;
    let finalUsage: TokenUsage | undefined;
    let hadRetriedGenerationFailure = false;
    try {
      try {
        return await retry(
          async () => {
            attempts += 1;
            throwIfCancelled(opts.signal);
            const params: Record<string, unknown> = {
              model: this.settings.model,
              messages,
              stream: true,
              stream_options: { include_usage: true },
            };
            if (limits.maxCompletionTokens !== undefined) {
              params.max_tokens = limits.maxCompletionTokens;
            }
            if (this.settings.thinkingMode) {
              if (this.settings.provider === "anthropic") {
                const budgets: Record<string, number> = { low: 2048, medium: 8192, high: 16384 };
                params.extra_body = { thinking: { type: "enabled", budget_tokens: budgets[this.settings.reasoningEffort] ?? 8192 } };
              } else {
                params.reasoning_effort = this.settings.reasoningEffort;
                params.extra_body = { thinking: { type: "enabled" } };
              }
            } else {
              params.temperature = opts.temperature ?? LLM_TEMPERATURE;
            }
            const abort = createAttemptAbortController(opts.signal);
            try {
              const result = await this.postChatStream(
                params, abort.controller, opts.signal, limits.maxOutputCodeUnits,
              );
              finalUsage = result.usage;
              throwIfCancelled(opts.signal);
              return result.content;
            } catch (error) {
              if (isCancellationError(error) || error instanceof LlmOutputLimitExceededError) throw error;
              if (!isUnsupportedStreamOptionsError(error)) throw this.safeError(error);
              attempts += 1;
              const fallback = { ...params };
              delete fallback.stream_options;
              try {
                const result = await this.postChatStream(
                  fallback, abort.controller, opts.signal, limits.maxOutputCodeUnits,
                );
                finalUsage = result.usage;
                return result.content;
              } catch (fallbackError) {
                if (isCancellationError(fallbackError)
                  || fallbackError instanceof LlmOutputLimitExceededError) throw fallbackError;
                throw this.safeError(fallbackError);
              }
            } finally {
              abort.cleanup();
            }
          },
          {
            maxAttempts: 3,
            baseDelayMs: 5000,
            signal: opts.signal,
            shouldRetry: (err) => !isCancellationError(err)
              && !(err instanceof LlmOutputLimitExceededError)
              && !isPermanentLlmError(err),
            onRetry: (err, attempt, wait) => {
              hadRetriedGenerationFailure = true;
              this.logger.warn(
                `LLM retry #${attempt} after ${wait}ms: ${this.safeError(err).message}`,
              );
            },
          },
        );
      } catch (error) {
        if (isCancellationError(error)
          || error instanceof LlmOutputLimitExceededError
          || isPermanentLlmError(error)) throw error;
        throw new LlmTransientExhaustedError(this.safeError(error));
      }
    } finally {
      const metrics: LlmCallMetrics = {
        logicalCalls: 1,
        attempts,
        elapsedMs: Date.now() - started,
        usageComplete: usageIsComplete(finalUsage) && !hadRetriedGenerationFailure,
        ...finalUsage,
      };
      opts.onMetrics?.(metrics);
    }
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
    maxOutputCodeUnits?: number,
  ): Promise<StreamResult> {
    const raw = await this.requestChat({ ...body, stream: true }, true, signal);
    return collectStreamResultWithIdleTimeout(
      parseSseText(raw),
      controller,
      LLM_STREAM_IDLE_TIMEOUT_MS,
      signal,
      maxOutputCodeUnits,
    );
  }

  private async requestChat(
    body: Record<string, unknown>,
    stream: boolean,
    signal?: AbortSignal,
  ): Promise<string> {
    const requestBody = { ...body, stream };
    const res = await this.http.request({
      url: buildChatCompletionsUrl(this.settings.baseUrl),
      method: "POST",
      headers: this.chatHeaders(),
      body: JSON.stringify(requestBody),
      timeoutMs: LLM_TIMEOUT_MS,
      signal,
    });
    if (res.status < 200 || res.status >= 300) {
      throw createStatusError(res.status, res.bodyText, [this.settings.apiKey]);
    }
    return res.bodyText;
  }

  private chatHeaders(): Record<string, string> {
    return {
      "Authorization": `Bearer ${this.settings.apiKey}`,
      "Content-Type": "application/json",
    };
  }

  private safeError(error: unknown): Error {
    return redactError(error, { secrets: [this.settings.apiKey] });
  }
}

export function isUnsupportedStreamOptionsError(err: unknown): boolean {
  const status = (err as LlmStatusError | undefined)?.status;
  if (status !== 400 && status !== 422) return false;
  const message = err instanceof Error ? err.message : String(err);
  return /stream[_ -]?options|include[_ -]?usage/i.test(message) &&
    /unsupported|unknown|unrecognized|not (?:allowed|supported)|extra fields?|invalid/i.test(message);
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

function createStatusError(
  status: number,
  bodyText: string,
  secrets: readonly string[] = [],
): LlmStatusError {
  let message = bodyText.trim();
  try {
    const parsed = JSON.parse(bodyText) as { error?: { message?: string } };
    message = parsed.error?.message ?? message;
  } catch {
    // Use raw response text.
  }
  const error = new Error(
    redactText(message || `LLM request failed with HTTP ${status}`, { secrets }),
  ) as LlmStatusError;
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
  maxOutputCodeUnits?: number,
): Promise<string> {
  return (await collectStreamResultWithIdleTimeout(
    stream, controller, idleTimeoutMs, signal, maxOutputCodeUnits,
  )).content;
}

export async function collectStreamResultWithIdleTimeout(
  stream: AsyncIterable<unknown>,
  controller: AbortController,
  idleTimeoutMs = LLM_STREAM_IDLE_TIMEOUT_MS,
  signal?: AbortSignal,
  maxOutputCodeUnits?: number,
): Promise<StreamResult> {
  const outputLimit = validatePositiveBoundedInteger(
    "maxOutputCodeUnits", maxOutputCodeUnits, LLM_MAX_OUTPUT_CODE_UNITS,
  );
  const chunks: string[] = [];
  let outputCodeUnits = 0;
  let usage: TokenUsage | undefined;
  const iterator = stream[Symbol.asyncIterator]();
  while (true) {
    throwIfCancelled(signal);
    const next = await nextStreamChunk(iterator, controller, idleTimeoutMs, signal);
    if (next.done) break;
    throwIfCancelled(signal);
    const delta = streamDeltaContent(next.value);
    if (delta) {
      outputCodeUnits += delta.length;
      if (outputLimit !== undefined && outputCodeUnits > outputLimit) {
        controller.abort(new LlmOutputLimitExceededError());
        throw new LlmOutputLimitExceededError();
      }
      chunks.push(delta);
    }
    usage = parseTokenUsage(next.value) ?? usage;
  }
  throwIfCancelled(signal);
  return { content: chunks.join(""), usage };
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
      const error = new StreamIdleTimeoutError();
      finish();
      reject(error);
      controller.abort(error);
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

function validateCallLimits(opts: CallOptions): {
  maxOutputCodeUnits?: number;
  maxCompletionTokens?: number;
} {
  return {
    maxOutputCodeUnits: validatePositiveBoundedInteger(
      "maxOutputCodeUnits", opts.maxOutputCodeUnits, LLM_MAX_OUTPUT_CODE_UNITS,
    ),
    maxCompletionTokens: validatePositiveBoundedInteger(
      "maxCompletionTokens", opts.maxCompletionTokens, LLM_MAX_COMPLETION_TOKENS,
    ),
  };
}

function validatePositiveBoundedInteger(
  name: string,
  value: number | undefined,
  maximum: number,
): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || value <= 0 || value > maximum) {
    throw new TypeError(`${name} must be a positive safe integer no greater than ${maximum}`);
  }
  return value;
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
