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

// Legacy relay provider IDs are untrusted transition data. This conservative
// cap matches the existing 128-character delivery/provider key contracts.
const LEGACY_PROVIDER_ID_MAX_LENGTH = 128;
// Accepted bodies are tiny even when every ID character is a JSON escape. Bound
// lexical work before decoding member names and values.
const HOSTED_SUCCESS_BODY_MAX_LENGTH = 4096;

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

  if (!isHostedSuccessBody(res.bodyText)) {
    throw new HostedDeliveryError(
      "Official delivery returned an invalid success response.",
      res.status,
      undefined,
      true,
    );
  }

  return { attempts: 1 };
}

function isHostedSuccessBody(bodyText: string): boolean {
  if (
    bodyText.length === 0 ||
    bodyText.length > HOSTED_SUCCESS_BODY_MAX_LENGTH ||
    hasDuplicateTopLevelJsonMember(bodyText)
  ) {
    return false;
  }

  try {
    return isHostedSuccessResponse(JSON.parse(bodyText) as unknown);
  } catch {
    return false;
  }
}

function isHostedSuccessResponse(value: unknown): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;

  const response = value as Record<string, unknown>;
  const keys = Object.keys(response);
  if (response.ok !== true) return false;
  if (keys.length === 1) return keys[0] === "ok";

  if (
    typeof response.id !== "string" ||
    !response.id.trim() ||
    response.id.length > LEGACY_PROVIDER_ID_MAX_LENGTH
  ) {
    return false;
  }

  // The validated provider ID is transition evidence only. Return a boolean so
  // it cannot reach delivery results, persisted state, or logs.
  if (keys.length === 2) {
    return keys.includes("ok") && keys.includes("id");
  }
  return (
    keys.length === 3 &&
    keys.includes("ok") &&
    keys.includes("id") &&
    keys.includes("deduped") &&
    response.deduped === true
  );
}

/**
 * Reject duplicate members before JSON.parse can collapse them. This scanner is
 * bounded by HOSTED_SUCCESS_BODY_MAX_LENGTH and decodes quoted member names with
 * JSON.parse, so escaped names compare by their actual JSON string value.
 */
function hasDuplicateTopLevelJsonMember(bodyText: string): boolean {
  let index = skipJsonWhitespace(bodyText, 0);
  if (bodyText[index] !== "{") return false;
  index = skipJsonWhitespace(bodyText, index + 1);

  const names = new Set<string>();
  while (bodyText[index] !== "}") {
    if (bodyText[index] !== '"') return false;
    const keyEnd = jsonStringEnd(bodyText, index);
    if (keyEnd < 0) return false;

    let name: string;
    try {
      name = JSON.parse(bodyText.slice(index, keyEnd)) as string;
    } catch {
      return false;
    }
    if (names.has(name)) return true;
    names.add(name);

    index = skipJsonWhitespace(bodyText, keyEnd);
    if (bodyText[index] !== ":") return false;
    index = jsonTopLevelValueEnd(bodyText, index + 1);
    if (index < 0) return false;
    index = skipJsonWhitespace(bodyText, index);
    if (bodyText[index] === ",") {
      index = skipJsonWhitespace(bodyText, index + 1);
      continue;
    }
    if (bodyText[index] !== "}") return false;
  }
  return false;
}

function jsonTopLevelValueEnd(bodyText: string, start: number): number {
  let index = skipJsonWhitespace(bodyText, start);
  let objectDepth = 0;
  let arrayDepth = 0;
  while (index < bodyText.length) {
    const char = bodyText[index];
    if (char === '"') {
      index = jsonStringEnd(bodyText, index);
      if (index < 0) return -1;
      continue;
    }
    if (char === "{") objectDepth += 1;
    else if (char === "}") {
      if (objectDepth === 0 && arrayDepth === 0) return index;
      objectDepth -= 1;
      if (objectDepth < 0) return -1;
    } else if (char === "[") arrayDepth += 1;
    else if (char === "]") {
      arrayDepth -= 1;
      if (arrayDepth < 0) return -1;
    } else if (char === "," && objectDepth === 0 && arrayDepth === 0) {
      return index;
    }
    index += 1;
  }
  return index;
}

function jsonStringEnd(bodyText: string, start: number): number {
  let escaped = false;
  for (let index = start + 1; index < bodyText.length; index += 1) {
    const char = bodyText[index];
    if (escaped) {
      escaped = false;
    } else if (char === "\\") {
      escaped = true;
    } else if (char === '"') {
      return index + 1;
    }
  }
  return -1;
}

function skipJsonWhitespace(bodyText: string, start: number): number {
  let index = start;
  while (
    bodyText[index] === " " ||
    bodyText[index] === "\n" ||
    bodyText[index] === "\r" ||
    bodyText[index] === "\t"
  ) {
    index += 1;
  }
  return index;
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
