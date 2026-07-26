import type { HttpClient } from "../core/adapters";
import {
  OFFICIAL_DELIVERY_AVAILABLE,
  type DailyDigest,
  type ResendEmailPayload,
} from "./types";

/** Default project relay base URL when 官方代发 (Beta) goes live. Override via settings later. */
export const DEFAULT_HOSTED_DELIVERY_BASE_URL =
  "https://email.arxiv-daily.example";

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
export async function sendViaHosted(opts: {
  http: HttpClient;
  baseUrl?: string;
  token: string;
  request: HostedDeliverRequest;
  signal?: AbortSignal;
}): Promise<HostedDeliverResult> {
  if (!OFFICIAL_DELIVERY_AVAILABLE) {
    throw new HostedDeliveryError(
      "官方代发 (Beta) is not online yet. Use 自己发送 with your Resend API key, or wait for the hosted service.",
    );
  }

  const base = (opts.baseUrl ?? DEFAULT_HOSTED_DELIVERY_BASE_URL).replace(
    /\/$/,
    "",
  );
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
