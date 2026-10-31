import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import type { LlmSettings } from "../settings/types";
import {
  HttpTransportError,
  isHttpTransportError,
  type HttpClient,
  type HttpRequest,
} from "../core/adapters";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import { redactError, redactText } from "../utils/redaction";
import { setTimer, clearTimer } from "../utils/timers";
import {
  parseTokenUsage,
  usageIsComplete,
  type LlmCallMetrics,
  type MetricsObserver,
  type TokenUsage,
} from "../metrics/generation";

/**
 * Total wall-clock bound for one request attempt, streaming or not. A stream
 * that keeps trickling bytes (so never goes idle) still ends here.
 */
const LLM_TIMEOUT_MS = 300_000; // 5 minutes
/**
 * Longest silence allowed on the wire while a response streams. With a host
 * that implements `HttpClient.requestStream` (the Node host) this acts on
 * network bytes as they arrive, so a stalled provider is cut off after this
 * gap instead of after {@link LLM_TIMEOUT_MS}. Hosts that can only buffer the
 * whole body (Obsidian's `requestUrl`) have no such protection: the response
 * is already complete when parsing starts, and only the total bound applies.
 */
export const LLM_STREAM_IDLE_TIMEOUT_MS = 120_000;
/** Largest single SSE event kept pending while waiting for its terminator. */
const SSE_MAX_PENDING_EVENT_CHARS = 16_000_000;
/** Largest non-2xx response body read to build an error message. */
const ERROR_BODY_MAX_CHARS = 65_536;
export const LLM_TEMPERATURE = 0.1;
export const LLM_MAX_OUTPUT_CODE_UNITS = 10_000_000;
export const LLM_MAX_COMPLETION_TOKENS = 1_000_000;
/** Invalidate generated checkpoints when the on-wire generation contract changes. */
export const LLM_REQUEST_CONTRACT_VERSION = 2 as const;
export const LLM_OUTPUT_LIMIT_EXCEEDED_ERROR_CODE = "ARXIV_LLM_OUTPUT_LIMIT_EXCEEDED" as const;

export class LlmOutputLimitExceededError extends Error {
  readonly code = LLM_OUTPUT_LIMIT_EXCEEDED_ERROR_CODE;

  constructor() {
    super("LLM output exceeded configured limit");
    this.name = "LlmOutputLimitExceededError";
  }
}

export const LLM_OUTPUT_TRUNCATED_ERROR_CODE = "ARXIV_LLM_OUTPUT_TRUNCATED" as const;

/**
 * The provider stopped because it hit its completion-token limit
 * (`finish_reason: length`, Anthropic `stop_reason: max_tokens`). The text it
 * returned is cut off, so it must never be treated as a finished answer.
 */
export class LlmOutputTruncatedError extends Error {
  readonly code = LLM_OUTPUT_TRUNCATED_ERROR_CODE;

  constructor() {
    super("LLM output was truncated at the completion-token limit");
    this.name = "LlmOutputTruncatedError";
  }
}

export function isLlmOutputTruncatedError(error: unknown): error is LlmOutputTruncatedError {
  if (error instanceof LlmOutputTruncatedError) return true;
  return isErrorLike(error) &&
    error.name === "LlmOutputTruncatedError" &&
    error.code === LLM_OUTPUT_TRUNCATED_ERROR_CODE;
}

export const LLM_CONTENT_BLOCKED_ERROR_CODE = "ARXIV_LLM_CONTENT_BLOCKED" as const;

/**
 * The provider's content-safety filter refused this input (an HTTP 400 whose
 * error code or text says so). The same input is refused every time, so it is
 * never retried; unlike an auth or quota failure it says nothing about the
 * other papers of the day, which lets callers set this one input aside.
 */
export class LlmContentBlockedError extends Error {
  readonly code = LLM_CONTENT_BLOCKED_ERROR_CODE;
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "LlmContentBlockedError";
    if (status !== undefined) this.status = status;
  }
}

export function isLlmContentBlockedError(error: unknown): error is LlmContentBlockedError {
  if (error instanceof LlmContentBlockedError) return true;
  return isErrorLike(error) &&
    error.name === "LlmContentBlockedError" &&
    error.code === LLM_CONTENT_BLOCKED_ERROR_CODE;
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
  /** Provider-compatible completion-token request bound. */
  maxCompletionTokens?: number;
}

interface StreamResult {
  content: string;
  usage?: TokenUsage;
  /** True when the provider reported that it stopped at the token limit. */
  truncated?: boolean;
  /** The last `finish_reason` the provider reported, as sent. */
  finishReason?: string;
  /** True when the provider sent its explicit end of stream (`data: [DONE]`, Anthropic `message_stop`). */
  ended?: boolean;
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

function stripTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) end--;
  return value.slice(0, end);
}

export function buildModelUrlCandidates(baseUrl: string): string[] {
  const normalized = stripTrailingSlashes(baseUrl);
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
  const normalized = stripTrailingSlashes(baseUrl.trim());
  if (!normalized) return normalized;
  if (normalized.endsWith("/v1")) return normalized;

  try {
    const parsed = new URL(normalized);
    const path = stripTrailingSlashes(parsed.pathname);
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

function usesNativeAnthropic(settings: Pick<LlmSettings, "provider" | "baseUrl">): boolean {
  if (settings.provider !== "anthropic") return false;
  try { return new URL(settings.baseUrl).hostname === "api.anthropic.com"; }
  catch { return false; }
}

export function buildLlmRequestUrl(settings: Pick<LlmSettings, "provider" | "baseUrl">): string {
  if (!usesNativeAnthropic(settings)) return buildChatCompletionsUrl(settings.baseUrl);
  const url = new URL(normalizeOpenAiBaseUrl(settings.baseUrl));
  url.pathname = `${stripTrailingSlashes(url.pathname)}/messages`;
  return url.toString();
}

export function usesAdaptiveAnthropicThinking(model: string): boolean {
  const match = /^claude-(?:opus|sonnet|haiku)-(\d+)(?:[.-](\d+))?/.exec(model);
  return Boolean(match && (Number(match[1]) > 4 || (Number(match[1]) === 4 && Number(match[2]) >= 6)));
}

export function anthropicAdaptiveEffort(effort: string): string {
  return ["low", "medium", "high"].includes(effort) ? effort : "high";
}

export class LlmClient {
  constructor(
    private settings: LlmSettings,
    private logger: Logger,
    private http: HttpClient,
  ) {
    this.logger.addSensitiveValues?.([settings.apiKey]);
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
    const baseUrl = stripTrailingSlashes(this.settings.baseUrl);
    const apiKey = this.settings.apiKey;

    if (!baseUrl || !apiKey) {
      throw new Error("Please fill in API Base URL and API Key first");
    }

    const candidates = buildModelUrlCandidates(baseUrl);
    let lastError: unknown;

    for (const url of candidates) {
      try {
        // The same credentials header as chat: Anthropic's own endpoint takes
        // x-api-key plus anthropic-version, not a Bearer token.
        const res = await this.http.request({
          url,
          method: "GET",
          headers: this.chatHeaders(),
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
      Array.isArray(data.data)
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
    const generationParams = generationParameters(this.settings, opts, limits.maxCompletionTokens);
    const started = Date.now();
    let attempts = 0;
    let finalUsage: TokenUsage | undefined;
    let hadRetriedGenerationFailure = false;
    // One blind re-send is allowed after an ambiguous mid-request network
    // failure (e.g. ECONNRESET): bounded duplicate-billing risk for a flaky link.
    let ambiguousResends = 0;
    const shouldResendAfterTransportFailure = (err: unknown): boolean => {
      // A stream that went silent mid-generation was (partly) billed and may
      // still be running at the provider: same reasoning as a request timeout.
      if (err instanceof StreamIdleTimeoutError
        || (err as { name?: unknown } | null)?.name === "StreamIdleTimeoutError") return false;
      if (!isHttpTransportError(err)) return true;
      // Provably never sent (connect/TLS phase): always safe to repeat.
      if (err.retryableAttempt) return true;
      // A timeout means the request may still be generating at the provider.
      if (err.kind === "timeout") return false;
      if (ambiguousResends >= 1) return false;
      ambiguousResends += 1;
      return true;
    };
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
              ...generationParams,
            };
            const abort = createAttemptAbortController(opts.signal);
            try {
              const result = await this.postChatStream(
                params, abort.controller, opts.signal, limits.maxOutputCodeUnits,
              );
              finalUsage = result.usage;
              throwIfCancelled(opts.signal);
              this.assertCompleted(result);
              return result.content;
            } catch (error) {
              if (isCancellationError(error) || isDeterministicOutputFailure(error)) throw error;
              if (!isUnsupportedStreamOptionsError(error)) throw this.safeError(error);
              attempts += 1;
              const fallback = { ...params };
              delete fallback.stream_options;
              try {
                const result = await this.postChatStream(
                  fallback, abort.controller, opts.signal, limits.maxOutputCodeUnits,
                );
                finalUsage = result.usage;
                this.assertCompleted(result);
                return result.content;
              } catch (fallbackError) {
                if (isCancellationError(fallbackError)
                  || isDeterministicOutputFailure(fallbackError)) throw fallbackError;
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
              && !isDeterministicOutputFailure(err)
              && !isPermanentLlmError(err)
              && shouldResendAfterTransportFailure(err),
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
          || isDeterministicOutputFailure(error)
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

  /**
   * Sends a `stream: true` request. When the host can stream, the body is parsed
   * as SSE while it arrives: the idle timeout runs between network chunks and
   * the output cap aborts the request the moment it is exceeded. Otherwise the
   * host buffers the whole body and it is parsed afterwards. `controller`
   * merges the caller's signal with this attempt's own abort, and its signal is
   * what the request is given.
   */
  private async postChatStream(
    body: Record<string, unknown>,
    controller: AbortController,
    signal?: AbortSignal,
    maxOutputCodeUnits?: number,
  ): Promise<StreamResult> {
    const native = usesNativeAnthropic(this.settings);
    if (!this.http.requestStream) {
      const raw = await this.requestChat({ ...body, stream: true }, true, controller.signal);
      const events = parseSseStream(singleChunk(raw));
      return collectStreamResultWithIdleTimeout(
        native ? parseAnthropicEvents(events) : events,
        controller,
        LLM_STREAM_IDLE_TIMEOUT_MS,
        signal,
        maxOutputCodeUnits,
      );
    }

    const request = this.chatRequest({ ...body, stream: true }, controller.signal);
    // Wall-clock bound for the attempt; its reason is surfaced as the error.
    const deadline = setTimer(() => {
      controller.abort(new HttpTransportError(
        "timeout",
        `HTTP timeout after ${LLM_TIMEOUT_MS}ms: ${request.url}`,
        { retryableAttempt: false },
      ));
    }, LLM_TIMEOUT_MS);
    let bodyIterator: AsyncIterator<string> | undefined;
    try {
      const res = await this.http.requestStream(request);
      bodyIterator = res.body[Symbol.asyncIterator]();
      if (res.status < 200 || res.status >= 300) {
        const errorText = await readTextWithIdleTimeout(
          bodyIterator, controller, LLM_STREAM_IDLE_TIMEOUT_MS, signal, ERROR_BODY_MAX_CHARS,
        );
        throw createStatusError(res.status, errorText, [this.settings.apiKey]);
      }
      const watched = idleWatchedChunks(
        bodyIterator, controller, LLM_STREAM_IDLE_TIMEOUT_MS, signal,
      );
      const events = parseSseStream(watched);
      // Idle is enforced per network chunk above, so a keep-alive comment
      // keeps the stream alive even though it yields no event.
      return await collectParsedStream(
        native ? parseAnthropicEvents(events) : events,
        controller,
        null,
        signal,
        maxOutputCodeUnits,
      );
    } catch (error) {
      // The attempt was stopped by us (total deadline); the host may report that
      // as a plain cancellation, so restore the real reason.
      const reason: unknown = controller.signal.reason;
      if (!signal?.aborted && controller.signal.aborted && reason instanceof Error) throw reason;
      throw error;
    } finally {
      clearTimer(deadline);
      // Stop reading and close the connection (a no-op once the body ended).
      await bodyIterator?.return?.().catch(() => undefined);
    }
  }

  private chatRequest(body: Record<string, unknown>, signal?: AbortSignal): HttpRequest {
    let requestBody: Record<string, unknown> = { ...body };
    if (usesNativeAnthropic(this.settings)) {
      const { messages, stream_options: _streamOptions, ...parameters } = requestBody;
      const chatMessages = messages as ChatMessage[];
      const system = chatMessages.filter((message) => message.role === "system")
        .map((message) => message.content).join("\n");
      requestBody = {
        ...parameters,
        messages: chatMessages.filter((message) => message.role !== "system"),
        ...(system ? { system } : {}),
      };
    }
    return {
      url: buildLlmRequestUrl(this.settings),
      method: "POST",
      headers: this.chatHeaders(),
      body: JSON.stringify(requestBody),
      timeoutMs: LLM_TIMEOUT_MS,
      signal,
    };
  }

  private async requestChat(
    body: Record<string, unknown>,
    stream: boolean,
    signal?: AbortSignal,
  ): Promise<string> {
    const res = await this.http.request(this.chatRequest({ ...body, stream }, signal));
    if (res.status < 200 || res.status >= 300) {
      throw createStatusError(res.status, res.bodyText, [this.settings.apiKey]);
    }
    return res.bodyText;
  }

  private chatHeaders(): Record<string, string> {
    if (usesNativeAnthropic(this.settings)) {
      return {
        "x-api-key": this.settings.apiKey,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
      };
    }
    return {
      "Authorization": `Bearer ${this.settings.apiKey}`,
      "Content-Type": "application/json",
    };
  }

  private safeError(error: unknown): Error {
    const safe = redactError(error, { secrets: [this.settings.apiKey] });
    if (isHttpTransportError(error)) {
      // Keep the structural transport markers the retry policy keys off.
      Object.assign(safe, { kind: error.kind, retryableAttempt: error.retryableAttempt });
    }
    return safe;
  }

  /**
   * A reply is an answer only when the provider says it finished. Text from a
   * stream that was cut off, filtered or ended without a completion signal is
   * a fragment, and callers persist what `call()` returns.
   */
  private assertCompleted(result: StreamResult): void {
    const reason = result.finishReason;
    if (result.truncated) throw new LlmOutputTruncatedError();
    if (reason !== undefined) {
      switch (classifyFinishReason(reason)) {
        case "complete":
          return;
        case "truncated":
          throw new LlmOutputTruncatedError();
        case "blocked":
          throw new LlmContentBlockedError(
            `LLM reply was stopped by the provider's content filter (finish_reason: ${reason})`,
          );
        case "interrupted":
          throw streamInterrupted(`the provider stopped the reply (finish_reason: ${reason})`);
        case "unknown":
          // Providers invent finish reasons; an explicit end of stream still
          // says the reply is whole, so only that case is let through.
          if (!result.ended) {
            throw streamInterrupted(`the provider ended the reply with finish_reason: ${reason}`);
          }
          this.logger.warn(
            `LLM finish_reason "${reason}" is not recognised; accepting the reply because the stream ended normally`,
          );
          return;
      }
    }
    if (!result.ended) {
      throw streamInterrupted("the stream ended before the provider reported completion");
    }
  }
}

/**
 * The reply stopped without being finished. Same standing as a connection reset
 * mid-reply: the client re-sends it once, never more (the first try was billed).
 */
function streamInterrupted(detail: string): HttpTransportError {
  return new HttpTransportError("network", `LLM reply was cut off: ${detail}`, {
    retryableAttempt: false,
  });
}

type FinishReasonKind = "complete" | "truncated" | "blocked" | "interrupted" | "unknown";

const COMPLETE_FINISH_REASONS: ReadonlySet<string> = new Set([
  "stop", "end_turn", "stop_sequence", "eos", "eos_token", "end", "normal", "complete", "completed",
  "finished", "done", "tool_calls", "tool_call", "tool_use", "function_call",
]);
const TRUNCATED_FINISH_REASONS: ReadonlySet<string> = new Set([
  "length", "max_tokens", "model_length", "model_context_window_exceeded",
]);
/** The provider's safety layer ended the reply (Zhipu `sensitive`, Azure `content_filter`, Anthropic `refusal`, ...). */
const BLOCKED_FINISH_REASONS: ReadonlySet<string> = new Set([
  "content_filter", "content_filtered", "sensitive", "refusal", "safety", "blocklist",
  "prohibited_content", "spii", "recitation", "guardrail_intervened",
]);
/** The provider gave up on the reply (DeepSeek `insufficient_system_resource`, Zhipu `network_error`, gateway `error`, ...). */
const INTERRUPTED_FINISH_REASONS: ReadonlySet<string> = new Set([
  "error", "network_error", "insufficient_system_resource", "abort", "aborted", "cancelled",
  "canceled", "timeout", "interrupted", "incomplete", "failed", "failure", "unavailable", "overloaded",
]);

function classifyFinishReason(reason: string): FinishReasonKind {
  const value = reason.trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (COMPLETE_FINISH_REASONS.has(value)) return "complete";
  if (TRUNCATED_FINISH_REASONS.has(value)) return "truncated";
  if (BLOCKED_FINISH_REASONS.has(value)) return "blocked";
  if (INTERRUPTED_FINISH_REASONS.has(value)) return "interrupted";
  return "unknown";
}

/** Failures a repeat of the same request would reproduce; never retried or re-typed. */
function isDeterministicOutputFailure(error: unknown): boolean {
  return error instanceof LlmOutputLimitExceededError
    || isLlmOutputTruncatedError(error)
    || isLlmContentBlockedError(error);
}

/**
 * Default completion budget when the caller sets none. Anthropic requires an
 * explicit `max_tokens` and counts thinking tokens against it, so 4096 cut
 * long answers off. Without thinking, 8192 is the largest value every Claude
 * model accepts; with thinking (only newer models) 32k is within each model's
 * output cap. Other providers keep their own default.
 */
const ANTHROPIC_DEFAULT_MAX_TOKENS = 8_192;
const ANTHROPIC_THINKING_MAX_TOKENS = 32_000;

function generationParameters(
  settings: LlmSettings,
  options: CallOptions,
  maxCompletionTokens?: number,
): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  if (maxCompletionTokens !== undefined) {
    params[settings.provider === "openai" ? "max_completion_tokens" : "max_tokens"] = maxCompletionTokens;
  } else if (settings.provider === "anthropic") {
    params.max_tokens = settings.thinkingMode
      ? ANTHROPIC_THINKING_MAX_TOKENS
      : ANTHROPIC_DEFAULT_MAX_TOKENS;
  }
  const hasThinkingToggle = ["anthropic", "deepseek", "zhipu"].includes(settings.provider);
  if (!settings.thinkingMode) {
    params.temperature = options.temperature ?? LLM_TEMPERATURE;
    if (hasThinkingToggle) params.thinking = { type: "disabled" };
    return params;
  }
  if (settings.provider === "anthropic") {
    if (usesAdaptiveAnthropicThinking(settings.model)) {
      params.thinking = { type: "adaptive" };
      params.output_config = { effort: anthropicAdaptiveEffort(settings.reasoningEffort) };
      return params;
    }
    const budgets: Record<string, number> = { low: 2048, medium: 8192, high: 16384 };
    const requested = budgets[settings.reasoningEffort] ?? 8192;
    const cap = maxCompletionTokens
      ?? Math.min(requested + 16_000, ANTHROPIC_THINKING_MAX_TOKENS);
    if (cap <= 1024) {
      throw new TypeError("Anthropic thinking requires maxCompletionTokens greater than 1024");
    }
    params.max_tokens = cap;
    params.thinking = { type: "enabled", budget_tokens: Math.min(requested, cap - 1) };
  } else {
    if (hasThinkingToggle) params.thinking = { type: "enabled" };
    if (settings.provider !== "zhipu") params.reasoning_effort = settings.reasoningEffort;
  }
  return params;
}

export function isUnsupportedStreamOptionsError(err: unknown): boolean {
  const status = (err as LlmStatusError | undefined)?.status;
  if (status !== 400 && status !== 422) return false;
  const message = err instanceof Error ? err.message : String(err);
  return /stream[_ -]?options|include[_ -]?usage/i.test(message) &&
    /unsupported|unknown|unrecognized|not (?:allowed|supported)|extra fields?|invalid/i.test(message);
}

/** Client-error statuses a later attempt can legitimately succeed on. */
const TRANSIENT_CLIENT_STATUSES: ReadonlySet<number> = new Set([408, 409, 425, 429]);

export function isPermanentLlmError(err: unknown): boolean {
  if (isLlmContentBlockedError(err)) return true;
  const status = (err as LlmStatusError | undefined)?.status;
  return (
    typeof status === "number" &&
    status >= 400 &&
    status < 500 &&
    !TRANSIENT_CLIENT_STATUSES.has(status)
  );
}

interface ProviderErrorFields {
  message?: string;
  code?: string;
  type?: string;
}

/** Error codes (and types) providers document for input their content-safety filter refused. */
const CONTENT_BLOCK_CODES: ReadonlySet<string> = new Set([
  "content_filter", // Azure OpenAI, Moonshot
  "content_policy_violation",
  "data_inspection_failed", // Alibaba DashScope
  "1301", // Zhipu GLM: input or output may contain unsafe or sensitive content
]);
/** Provider texts that name a content-safety refusal without a dedicated error code. */
const CONTENT_BLOCK_MESSAGES: readonly RegExp[] = [
  /content exists risk/i, // DeepSeek
  /不安全或敏感内容/, // Zhipu GLM 1301
];

/** `code`, `type` and `message` of a provider error body, nested under `error` or at the top level. */
function providerErrorFields(body: unknown): ProviderErrorFields {
  if (typeof body !== "object" || body === null) return {};
  const nested = (body as { error?: unknown }).error;
  const source = typeof nested === "object" && nested !== null ? nested : body;
  const { message, code, type } = source as Record<string, unknown>;
  return {
    message: typeof message === "string" ? message : undefined,
    code: typeof code === "string" || typeof code === "number" ? String(code) : undefined,
    type: typeof type === "string" ? type : undefined,
  };
}

/**
 * True only when the provider's own error code or text says its content-safety
 * filter refused the input. Deliberately narrow: any other client error is a
 * request problem the caller has to see, not an input to quietly set aside.
 */
function isContentBlockedBody(fields: ProviderErrorFields, message: string): boolean {
  const code = fields.code?.toLowerCase();
  const type = fields.type?.toLowerCase();
  if ((code && CONTENT_BLOCK_CODES.has(code)) || (type && CONTENT_BLOCK_CODES.has(type))) return true;
  // OpenAI flags a prompt with the generic `invalid_prompt` code; only its policy wording makes it a refusal.
  if (code === "invalid_prompt" && /usage polic|safety/i.test(message)) return true;
  return CONTENT_BLOCK_MESSAGES.some((pattern) => pattern.test(message));
}

function createStatusError(
  status: number,
  bodyText: string,
  secrets: readonly string[] = [],
): LlmStatusError {
  let message = bodyText.trim();
  let fields: ProviderErrorFields = {};
  try {
    const parsed = JSON.parse(bodyText) as { error?: { message?: string } };
    message = parsed.error?.message ?? message;
    fields = providerErrorFields(parsed);
  } catch {
    // Use raw response text.
  }
  const text = redactText(message || `LLM request failed with HTTP ${status}`, { secrets });
  // Refusals arrive as HTTP 400; a 401/403/404 names the account or the model, never one input.
  if (status === 400 && isContentBlockedBody(fields, message)) {
    return new LlmContentBlockedError(text, status);
  }
  const error = new Error(text) as LlmStatusError;
  error.status = status;
  return error;
}

/**
 * The failure behind an `error` event a provider or gateway sent inside a 200
 * response stream, where there is no HTTP status to go by. A content-safety
 * block is typed like an HTTP refusal; a numeric code in the HTTP range is the
 * provider's own verdict; anything else is a server-side failure worth a retry.
 */
function createStreamEventError(error: unknown): Error {
  const fields = providerErrorFields({ error: typeof error === "string" ? { message: error } : error });
  const message = fields.message?.trim() || "the provider reported an error in the response stream";
  const text = redactText(message);
  if (isContentBlockedBody(fields, message)) return new LlmContentBlockedError(text);
  const code = Number(fields.code);
  const status = Number.isInteger(code) && code >= 400 && code < 600
    ? code
    : fields.type === "invalid_request_error" ? 400 : 500;
  return Object.assign(new Error(text), { status });
}

/** The `error` of a stream chunk, when the chunk reports a failure instead of content. */
function streamErrorEvent(chunk: unknown): unknown {
  if (typeof chunk !== "object" || chunk === null) return undefined;
  const error = (chunk as { error?: unknown }).error;
  if (typeof error === "string") return error.trim() ? error : undefined;
  return typeof error === "object" && error !== null && Object.keys(error).length > 0
    ? error
    : undefined;
}

/** In-band marker the parsers emit when the provider itself says the reply is over. */
const STREAM_END = Symbol("llm-stream-end");

/** Anthropic `stop_reason` in the `finish_reason` vocabulary the collector reads. */
function anthropicFinishReason(stopReason: string): string {
  if (stopReason === "max_tokens" || stopReason === "model_context_window_exceeded") return "length";
  if (stopReason === "refusal") return "content_filter";
  if (stopReason === "end_turn" || stopReason === "stop_sequence") return "stop";
  return stopReason;
}

async function* parseAnthropicEvents(events: AsyncIterable<unknown>): AsyncIterable<unknown> {
  let usage: Record<string, number> = {};
  for await (const value of events) {
    if (value === STREAM_END) {
      yield STREAM_END;
      return;
    }
    const event = value as {
      type?: string;
      message?: { usage?: Record<string, number> };
      usage?: Record<string, number>;
      delta?: { type?: string; text?: string; stop_reason?: string };
      content_block?: { type?: string; text?: string };
      error?: { type?: string; message?: string };
    };
    if (event.type === "error") throw createStreamEventError(event.error);
    if (event.type === "content_block_start" && event.content_block?.type === "text") {
      yield { choices: [{ delta: { content: event.content_block.text ?? "" } }] };
    }
    if (event.type === "content_block_delta" && event.delta?.type === "text_delta") {
      yield { choices: [{ delta: { content: event.delta.text ?? "" } }] };
    }
    const nextUsage = event.type === "message_start" ? event.message?.usage : event.usage;
    if (nextUsage) {
      usage = { ...usage, ...nextUsage };
      yield { usage: {
        ...usage,
        input_tokens: (usage.input_tokens ?? 0) + (usage.cache_read_input_tokens ?? 0) + (usage.cache_creation_input_tokens ?? 0),
      } };
    }
    const stopReason = event.delta?.stop_reason;
    if (event.type === "message_delta" && typeof stopReason === "string") {
      yield { choices: [{ delta: {}, finish_reason: anthropicFinishReason(stopReason) }] };
    }
    if (event.type === "message_stop") {
      yield STREAM_END;
      return;
    }
  }
}

async function* singleChunk(text: string): AsyncIterable<string> {
  yield text;
}

/** How an event stream opens: an SSE field (`data:`, `event:`, `id:`, `retry:`) or a comment line. */
const SSE_OPENING = /^(?:(?:data|event|id|retry)[ \t]*:|:)/;

/**
 * Incrementally parses an SSE byte stream into JSON event payloads. `data:
 * [DONE]` ends it with {@link STREAM_END}; a body that is not an event stream
 * at all is read as the plain reply of a server that ignored `stream: true`.
 */
async function* parseSseStream(chunks: AsyncIterable<string>): AsyncIterable<unknown> {
  let buffer = "";
  // Kept only until the body shows itself to be an event stream (it opens with
  // a field or a comment); a plain JSON or HTML reply never does.
  let plain: string | undefined = "";
  let sawEvent = false;
  for await (const chunk of chunks) {
    buffer += chunk;
    if (plain !== undefined) {
      plain += chunk;
      if (SSE_OPENING.test(plain.trimStart())) plain = undefined;
      else if (plain.length > SSE_MAX_PENDING_EVENT_CHARS) {
        throw new Error("LLM stream event exceeded the size limit");
      }
    }
    const separator = /\r?\n\r?\n/g;
    let start = 0;
    for (let match = separator.exec(buffer); match; match = separator.exec(buffer)) {
      const parsed = parseSseEvent(buffer.slice(start, match.index));
      start = match.index + match[0].length;
      if (parsed.done) {
        yield STREAM_END;
        return;
      }
      if (parsed.value !== undefined) {
        sawEvent = true;
        yield parsed.value;
      }
    }
    buffer = buffer.slice(start);
    if (buffer.length > SSE_MAX_PENDING_EVENT_CHARS) {
      throw new Error("LLM stream event exceeded the size limit");
    }
  }
  const parsed = parseSseEvent(buffer);
  if (parsed.done) {
    yield STREAM_END;
  } else if (parsed.value !== undefined) {
    yield parsed.value;
  } else if (!sawEvent && plain?.trim()) {
    yield* plainBodyEvents(plain.trim());
  }
}

/**
 * Events equivalent to a non-streamed reply, for a server that ignored
 * `stream: true` and answered with one chat-completion JSON object (or an
 * error object). Anything else is reported rather than read as an empty reply.
 */
function* plainBodyEvents(body: string): Iterable<unknown> {
  let value: unknown;
  try {
    value = JSON.parse(body);
  } catch {
    throw notAnEventStream(body);
  }
  const reply = (typeof value === "object" && value !== null ? value : {}) as {
    error?: unknown;
    choices?: unknown;
    usage?: unknown;
  };
  if (streamErrorEvent(reply) !== undefined) {
    yield reply;
    return;
  }
  const choice = (Array.isArray(reply.choices) ? reply.choices[0] : undefined) as
    | { message?: { content?: unknown }; finish_reason?: unknown }
    | undefined;
  if (typeof choice !== "object" || choice === null) throw notAnEventStream(body);
  const content = choice.message?.content;
  yield {
    choices: [{
      delta: typeof content === "string" ? { content } : {},
      finish_reason: choice.finish_reason,
    }],
    usage: reply.usage,
  };
  yield STREAM_END;
}

function notAnEventStream(body: string): HttpTransportError {
  const snippet = redactText(body.replace(/\s+/g, " ").trim().slice(0, 200));
  return new HttpTransportError(
    "network",
    `LLM server answered with something that is not an event stream or a chat completion: ${snippet}`,
    { retryableAttempt: false },
  );
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

export function collectStreamResultWithIdleTimeout(
  stream: AsyncIterable<unknown>,
  controller: AbortController,
  idleTimeoutMs = LLM_STREAM_IDLE_TIMEOUT_MS,
  signal?: AbortSignal,
  maxOutputCodeUnits?: number,
): Promise<StreamResult> {
  return collectParsedStream(stream, controller, idleTimeoutMs, signal, maxOutputCodeUnits);
}

/** `idleTimeoutMs = null` leaves idleness to the caller (see {@link idleWatchedChunks}). */
async function collectParsedStream(
  stream: AsyncIterable<unknown>,
  controller: AbortController,
  idleTimeoutMs: number | null,
  signal?: AbortSignal,
  maxOutputCodeUnits?: number,
): Promise<StreamResult> {
  const outputLimit = validatePositiveBoundedInteger(
    "maxOutputCodeUnits", maxOutputCodeUnits, LLM_MAX_OUTPUT_CODE_UNITS,
  );
  const chunks: string[] = [];
  let outputCodeUnits = 0;
  let usage: TokenUsage | undefined;
  let truncated = false;
  let finishReason: string | undefined;
  let ended = false;
  const iterator = stream[Symbol.asyncIterator]();
  while (true) {
    throwIfCancelled(signal);
    const next = await nextStreamChunk(iterator, controller, idleTimeoutMs, signal);
    if (next.done) break;
    throwIfCancelled(signal);
    if (next.value === STREAM_END) {
      ended = true;
      continue;
    }
    const failure = streamErrorEvent(next.value);
    if (failure !== undefined) throw createStreamEventError(failure);
    const delta = streamDeltaContent(next.value);
    if (delta) {
      outputCodeUnits += delta.length;
      if (outputLimit !== undefined && outputCodeUnits > outputLimit) {
        controller.abort(new LlmOutputLimitExceededError());
        throw new LlmOutputLimitExceededError();
      }
      chunks.push(delta);
    }
    const reason = streamFinishReason(next.value);
    if (reason) finishReason = reason;
    if (reason === "length" || reason === "max_tokens") truncated = true;
    usage = parseTokenUsage(next.value) ?? usage;
  }
  throwIfCancelled(signal);
  return {
    content: chunks.join(""),
    usage,
    ...(truncated ? { truncated } : {}),
    ...(finishReason ? { finishReason } : {}),
    ...(ended ? { ended } : {}),
  };
}

/**
 * Wraps a raw body iterator so each wait for the next network chunk is bounded
 * by the idle timeout. Closing the wrapper closes the source immediately.
 */
function idleWatchedChunks(
  source: AsyncIterator<string>,
  controller: AbortController,
  idleTimeoutMs: number,
  signal?: AbortSignal,
): AsyncIterable<string> {
  const iterator: AsyncIterator<string> = {
    next: () => nextStreamChunk(source, controller, idleTimeoutMs, signal),
    return: async () => {
      await source.return?.();
      return { done: true, value: undefined };
    },
  };
  return { [Symbol.asyncIterator]: () => iterator };
}

/** Reads at most `maxChars` of a text body, each chunk bounded by the idle timeout. */
async function readTextWithIdleTimeout(
  source: AsyncIterator<string>,
  controller: AbortController,
  idleTimeoutMs: number,
  signal: AbortSignal | undefined,
  maxChars: number,
): Promise<string> {
  let text = "";
  while (text.length < maxChars) {
    const next = await nextStreamChunk(source, controller, idleTimeoutMs, signal);
    if (next.done) break;
    text += next.value;
  }
  return text.slice(0, maxChars);
}

function nextStreamChunk<T>(
  iterator: AsyncIterator<T>,
  controller: AbortController,
  idleTimeoutMs: number | null,
  signal?: AbortSignal,
): Promise<IteratorResult<T>> {
  throwIfCancelled(signal);
  return new Promise((resolve, reject) => {
    let settled = false;
    const timeout = idleTimeoutMs === null ? undefined : setTimer(() => {
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
        reject(e instanceof Error ? e : new Error(String(e), { cause: e }));
      }
    };
    // The attempt may be stopped from outside this read (total deadline). A host
    // is expected to fail its pending read then, but do not depend on it.
    const onAttemptAbort = () => {
      if (signal?.aborted) {
        onAbort();
        return;
      }
      finish();
      const reason: unknown = controller.signal.reason;
      reject(reason instanceof Error ? reason : new Error(String(reason), { cause: reason }));
    };
    const finish = () => {
      if (settled) return;
      settled = true;
      if (timeout) clearTimer(timeout);
      signal?.removeEventListener("abort", onAbort);
      controller.signal.removeEventListener("abort", onAttemptAbort);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    controller.signal.addEventListener("abort", onAttemptAbort, { once: true });
    if (signal?.aborted) {
      onAbort();
      return;
    }
    if (controller.signal.aborted) {
      onAttemptAbort();
      return;
    }
    iterator.next().then(
      (value) => {
        finish();
        resolve(value);
      },
      (error: unknown) => {
        finish();
        reject(error instanceof Error ? error : new Error(String(error), { cause: error }));
      },
    );
  });
}

function streamFinishReason(chunk: unknown): string | undefined {
  const choices = (chunk as { choices?: Array<{ finish_reason?: unknown }> }).choices;
  const reason = choices?.[0]?.finish_reason;
  return typeof reason === "string" ? reason : undefined;
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
