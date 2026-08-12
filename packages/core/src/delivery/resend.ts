import { isHttpTransportError, type HttpClient } from "../core/adapters";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";
import type { ResendEmailPayload } from "./types";

export const RESEND_API_URL = "https://api.resend.com/emails";

export interface SendViaResendOptions {
  http: HttpClient;
  apiKey: string;
  payload: ResendEmailPayload;
  /** Stable logical provider key, reused by every internal attempt. */
  idempotencyKey: string;
  /** Total attempts including the first try. Default 3. */
  maxAttempts?: number;
  /** Base backoff between retries in ms. Default 400. */
  baseDelayMs?: number;
  signal?: AbortSignal;
  /** Injectable sleep for tests. */
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  /** Completes the durable local attempt marker before each physical request. */
  beforeProviderAttempt?: () => Promise<void>;
  /** Called synchronously immediately before HttpClient.request is invoked. */
  onProviderInvocation?: () => void;
}

export interface SendViaResendResult {
  attempts: number;
  status: number;
}

export class ResendSendError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly permanent = false,
    readonly attempts = 1,
    readonly ambiguous = false,
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
  const idempotencyKey = opts.idempotencyKey.trim();
  if (!idempotencyKey || idempotencyKey.length > 128) {
    throw new ResendSendError(
      "Resend Idempotency-Key must be 1-128 characters",
      undefined,
      true,
      0,
    );
  }

  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    throwIfCancelled(opts.signal);
    await opts.beforeProviderAttempt?.();
    try {
      opts.onProviderInvocation?.();
      const response = await opts.http.request({
        url: RESEND_API_URL,
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          "Idempotency-Key": idempotencyKey,
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
        if (!hasProviderAcceptance(response.bodyText)) {
          throw new ResendSendError(
            "Resend success response did not contain an acceptance marker",
            response.status,
            false,
            attempt,
            true,
          );
        }
        return {
          attempts: attempt,
          status: response.status,
        };
      }

      const ambiguous =
        response.status === 408 ||
        response.status === 409 ||
        response.status >= 500;
      const definitiveRejection =
        response.status === 400 ||
        response.status === 401 ||
        response.status === 403 ||
        response.status === 404 ||
        response.status === 422 ||
        response.status === 429;
      const error = new ResendSendError(
        `Resend HTTP ${response.status}`,
        response.status,
        definitiveRejection,
        attempt,
        ambiguous || !definitiveRejection,
      );

      // Stable provider idempotency permits bounded retries for 408/409/5xx.
      // The local claim remains blocking throughout, even after retry exhaustion.
      if (!ambiguous || attempt >= maxAttempts) throw error;
      lastError = error;
      await sleepFn(baseDelayMs * attempt, opts.signal);
      continue;
    } catch (error) {
      if (isCancellationError(error)) throw error;
      if (error instanceof ResendSendError) {
        if (error.permanent || attempt >= maxAttempts) {
          throw new ResendSendError(
            error.message,
            error.status,
            error.permanent,
            attempt,
            error.ambiguous,
          );
        }
        lastError = error;
        await sleepFn(baseDelayMs * attempt, opts.signal);
        continue;
      }
      const message = isHttpTransportError(error)
        ? `Resend ${error.kind} transport failure`
        : "Resend transport outcome is unknown";
      const canRetryPhysicalAttempt =
        isHttpTransportError(error) && error.retryableAttempt;
      if (!canRetryPhysicalAttempt || attempt >= maxAttempts) {
        throw new ResendSendError(
          message,
          undefined,
          false,
          attempt,
          true,
        );
      }
      lastError = error;
      await sleepFn(baseDelayMs * attempt, opts.signal);
    }
  }

  throw lastError instanceof ResendSendError
    ? lastError
    : new ResendSendError(
        "Resend transport outcome is unknown",
        undefined,
        false,
        maxAttempts,
        true,
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

function hasProviderAcceptance(bodyText: string): boolean {
  if (!bodyText.trim()) return false;
  try {
    const parsed = JSON.parse(bodyText) as { id?: unknown };
    return typeof parsed.id === "string" && parsed.id.length > 0;
  } catch {
    return false;
  }
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
