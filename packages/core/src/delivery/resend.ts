import type { HttpClient } from "../core/adapters";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { ResendEmailPayload } from "./types";

export const RESEND_API_URL = "https://api.resend.com/emails";

export interface SendViaResendOptions {
  http: HttpClient;
  apiKey: string;
  payload: ResendEmailPayload;
  /** Total attempts including the first try. Default 3. */
  maxAttempts?: number;
  /** Base backoff between retries in ms. Default 400. */
  baseDelayMs?: number;
  signal?: AbortSignal;
  /** Injectable sleep for tests. */
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

export interface SendViaResendResult {
  providerMessageId?: string;
  attempts: number;
  status: number;
}

export class ResendSendError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly permanent = false,
    readonly attempts = 1,
  ) {
    super(message);
    this.name = "ResendSendError";
  }
}

export async function sendViaResend(
  opts: SendViaResendOptions,
): Promise<SendViaResendResult> {
  const maxAttempts = Math.max(1, opts.maxAttempts ?? 3);
  const baseDelayMs = Math.max(0, opts.baseDelayMs ?? 400);
  const sleepFn = opts.sleep ?? defaultSleep;
  const apiKey = opts.apiKey.trim();
  if (!apiKey) {
    throw new ResendSendError("Resend API key is empty", undefined, true, 0);
  }

  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    throwIfCancelled(opts.signal);
    try {
      const response = await opts.http.request({
        url: RESEND_API_URL,
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          from: formatFrom(opts.payload),
          to: [opts.payload.to],
          subject: opts.payload.subject,
          html: opts.payload.html,
          text: opts.payload.text,
        }),
        responseType: "text",
        signal: opts.signal,
      });

      if (response.status >= 200 && response.status < 300) {
        return {
          providerMessageId: parseProviderMessageId(response.bodyText),
          attempts: attempt,
          status: response.status,
        };
      }

      const permanent = response.status === 401 || response.status === 403;
      const retryable =
        !permanent &&
        (response.status === 429 || response.status >= 500 || response.status === 408);
      const message =
        `Resend HTTP ${response.status}` +
        (response.bodyText ? `: ${truncate(response.bodyText, 200)}` : "");

      if (!retryable || attempt >= maxAttempts) {
        throw new ResendSendError(message, response.status, permanent, attempt);
      }
      lastError = new ResendSendError(message, response.status, permanent, attempt);
      await sleepFn(baseDelayMs * attempt, opts.signal);
      continue;
    } catch (error) {
      if (isCancellationError(error)) throw error;
      if (error instanceof ResendSendError) {
        if (error.permanent || attempt >= maxAttempts) {
          throw new ResendSendError(error.message, error.status, error.permanent, attempt);
        }
        lastError = error;
        await sleepFn(baseDelayMs * attempt, opts.signal);
        continue;
      }
      // Network / transport errors — retry.
      lastError = error;
      if (attempt >= maxAttempts) {
        throw new ResendSendError(
          error instanceof Error ? error.message : String(error),
          undefined,
          false,
          attempt,
        );
      }
      await sleepFn(baseDelayMs * attempt, opts.signal);
    }
  }

  throw lastError instanceof ResendSendError
    ? lastError
    : new ResendSendError(
        lastError instanceof Error ? lastError.message : String(lastError),
        undefined,
        false,
        maxAttempts,
      );
}

function formatFrom(payload: ResendEmailPayload): string {
  // payload.from already includes optional display name when prepared by deliver-email.
  return payload.from;
}

export function formatResendFrom(fromEmail: string, fromName?: string): string {
  const email = fromEmail.trim();
  const name = fromName?.trim();
  if (!name) return email;
  // Avoid breaking header if name contains quotes.
  const safeName = name.replace(/"/g, "");
  return `${safeName} <${email}>`;
}

function parseProviderMessageId(bodyText: string): string | undefined {
  if (!bodyText.trim()) return undefined;
  try {
    const parsed = JSON.parse(bodyText) as { id?: unknown };
    return typeof parsed.id === "string" && parsed.id ? parsed.id : undefined;
  } catch {
    return undefined;
  }
}

function truncate(value: string, max: number): string {
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length <= max ? compact : `${compact.slice(0, max)}…`;
}

function defaultSleep(ms: number, signal?: AbortSignal): Promise<void> {
  throwIfCancelled(signal);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timeout);
      cleanup();
      try {
        throwIfCancelled(signal);
      } catch (e) {
        reject(e);
      }
    };
    function cleanup() {
      signal?.removeEventListener("abort", onAbort);
    }
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
  });
}
