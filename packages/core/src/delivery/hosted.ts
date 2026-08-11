import type { HttpClient } from "../core/adapters";
import {
  OFFICIAL_DELIVERY_AVAILABLE,
  type DailyDigest,
  type ResendEmailPayload,
} from "./types";

/**
 * Default project relay base URL.
 * Must match the Worker custom domain (currently mail.arxiv-daily.top).
 * Override via settings.email.hostedBaseUrl if needed.
 */
export const DEFAULT_HOSTED_DELIVERY_BASE_URL = "https://mail.arxiv-daily.top";

export class HostedDeliveryError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly cause?: unknown,
    readonly ambiguous?: boolean,
  ) {
    super(message);
    this.name = "HostedDeliveryError";
  }
}

export interface HostedDeliverRequest {
  to: string;
  digest: DailyDigest;
  /** Idempotency hint for the server (date|to). */
  idempotencyKey: string;
  subject: string;
  html: string;
  text: string;
}

export interface HostedDeliverResult {
  attempts: number;
}

/**
 * POST digest to the project relay. Offline until OFFICIAL_DELIVERY_AVAILABLE.
 * Server holds the send API key; plugin only sends auth token + payload.
 */
export function resolveHostedBaseUrl(baseUrl?: string): string {
  return (baseUrl?.trim() || DEFAULT_HOSTED_DELIVERY_BASE_URL).replace(
    /\/$/,
    "",
  );
}

/** Request a magic-link verification email for Official delivery (Beta). */
export async function startHostedEmailVerification(opts: {
  http: HttpClient;
  baseUrl?: string;
  email: string;
  signal?: AbortSignal;
}): Promise<void> {
  if (!OFFICIAL_DELIVERY_AVAILABLE) {
    throw new HostedDeliveryError(
      "Official delivery (Beta) client support is disabled in this build.",
    );
  }
  const email = opts.email.trim();
  if (!email) {
    throw new HostedDeliveryError("Enter your email address first.");
  }
  const base = resolveHostedBaseUrl(opts.baseUrl);
  const url = `${base}/v1/verify/start`;
  let res;
  try {
    res = await opts.http.request({
      url,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
      signal: opts.signal,
    });
  } catch {
    throw new HostedDeliveryError(
      "Cannot reach Official delivery. Check your network and try again later.",
    );
  }
  if (res.status < 200 || res.status >= 300) {
    throw new HostedDeliveryError(
      `Could not send verification email (HTTP ${res.status}).`,
      res.status,
    );
  }
}

export async function sendViaHosted(opts: {
  http: HttpClient;
  baseUrl?: string;
  token: string;
  request: HostedDeliverRequest;
  signal?: AbortSignal;
  beforeProviderAttempt?: () => Promise<void>;
  onProviderInvocation?: () => void;
}): Promise<HostedDeliverResult> {
  if (!OFFICIAL_DELIVERY_AVAILABLE) {
    throw new HostedDeliveryError(
      "Official delivery (Beta) is not online yet. Use Send yourself with your Resend API key, or wait for the hosted service.",
    );
  }

  const base = resolveHostedBaseUrl(opts.baseUrl);
  // Strip all whitespace — paste from HTML <pre> often includes newlines.
  const token = opts.token.replace(/\s+/g, "").trim();
  if (!token) {
    throw new HostedDeliveryError(
      "Verification code is missing. Send a verification email and paste the code from the page.",
    );
  }

  let res;
  try {
    await opts.beforeProviderAttempt?.();
    opts.onProviderInvocation?.();
    res = await opts.http.request({
      url: `${base}/v1/deliver`,
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        "Idempotency-Key": opts.request.idempotencyKey,
      },
      body: JSON.stringify({
        to: opts.request.to,
        date: opts.request.digest.date,
        subject: opts.request.subject,
        html: opts.request.html,
        text: opts.request.text,
        digest: opts.request.digest,
      }),
      signal: opts.signal,
    });
  } catch {
    throw new HostedDeliveryError(
      "Cannot reach Official delivery.",
      undefined,
      undefined,
      true,
    );
  }

  if (res.status < 200 || res.status >= 300) {
    const ambiguous =
      res.status === 408 || res.status === 409 || res.status >= 500;
    let hint = "";
    if (res.status === 401) {
      hint =
        " Verification code is invalid or expired. Send a new verification email and use the latest code.";
    } else if (res.status === 403) {
      hint = " The requested recipient does not match the verified inbox.";
    } else if (res.status === 429) {
      hint = " Daily send limit reached for this verified inbox. Try again tomorrow.";
    } else if (res.status === 422) {
      hint = " The relay definitively rejected this delivery before acceptance.";
    }
    throw new HostedDeliveryError(
      `Official delivery failed (HTTP ${res.status}).${hint}`,
      res.status,
      undefined,
      ambiguous,
    );
  }

  try {
    const parsed = JSON.parse(res.bodyText) as unknown;
    if (
      !parsed ||
      typeof parsed !== "object" ||
      Array.isArray(parsed) ||
      Object.keys(parsed as Record<string, unknown>).length !== 1 ||
      (parsed as { ok?: unknown }).ok !== true
    ) {
      throw new Error("invalid safe success contract");
    }
  } catch {
    throw new HostedDeliveryError(
      "Official delivery returned an invalid success response.",
      res.status,
      undefined,
      true,
    );
  }

  return { attempts: 1 };
}

/** Map rendered mail into the shape the future relay expects (docs/contract). */
export function hostedPayloadFromRendered(
  to: string,
  digest: DailyDigest,
  mail: Pick<ResendEmailPayload, "subject" | "html" | "text">,
  idempotencyKey: string,
): HostedDeliverRequest {
  return {
    to,
    digest,
    idempotencyKey,
    subject: mail.subject,
    html: mail.html,
    text: mail.text,
  };
}
