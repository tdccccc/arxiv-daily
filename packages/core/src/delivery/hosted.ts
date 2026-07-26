import type { HttpClient } from "../core/adapters";
import {
  OFFICIAL_DELIVERY_AVAILABLE,
  type DailyDigest,
  type ResendEmailPayload,
} from "./types";

/** Default project relay base URL (Cloudflare Worker custom domain). */
export const DEFAULT_HOSTED_DELIVERY_BASE_URL =
  "https://email.arxiv-daily.top";

export class HostedDeliveryError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly cause?: unknown,
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
  providerMessageId?: string;
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
    throw new HostedDeliveryError("email is required");
  }
  const base = resolveHostedBaseUrl(opts.baseUrl);
  const res = await opts.http.request({
    url: `${base}/v1/verify/start`,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
    signal: opts.signal,
  });
  if (res.status < 200 || res.status >= 300) {
    throw new HostedDeliveryError(
      `Verify start HTTP ${res.status}: ${res.bodyText.slice(0, 300)}`,
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
}): Promise<HostedDeliverResult> {
  if (!OFFICIAL_DELIVERY_AVAILABLE) {
    throw new HostedDeliveryError(
      "Official delivery (Beta) is not online yet. Use Send yourself with your Resend API key, or wait for the hosted service.",
    );
  }

  const base = resolveHostedBaseUrl(opts.baseUrl);
  const token = opts.token.trim();
  if (!token) {
    throw new HostedDeliveryError("hosted delivery token is missing");
  }

  const res = await opts.http.request({
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

  if (res.status < 200 || res.status >= 300) {
    throw new HostedDeliveryError(
      `Hosted delivery HTTP ${res.status}: ${res.bodyText.slice(0, 300)}`,
      res.status,
    );
  }

  let providerMessageId: string | undefined;
  try {
    const parsed = JSON.parse(res.bodyText) as { id?: unknown };
    if (typeof parsed.id === "string" && parsed.id) {
      providerMessageId = parsed.id;
    }
  } catch {
    // optional body
  }

  return { providerMessageId, attempts: 1 };
}

/** Map rendered mail into the shape the future relay expects (docs/contract). */
export function hostedPayloadFromRendered(
  to: string,
  digest: DailyDigest,
  mail: Pick<ResendEmailPayload, "subject" | "html" | "text">,
): HostedDeliverRequest {
  return {
    to,
    digest,
    idempotencyKey: `${digest.date}|${to.trim().toLowerCase()}`,
    subject: mail.subject,
    html: mail.html,
    text: mail.text,
  };
}
