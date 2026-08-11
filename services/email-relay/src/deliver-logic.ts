import { normalizeEmail, sha256Hex } from "./crypto";
import type { AuthenticatedDevice } from "./kv";

export type DeliverBody = {
  to?: string;
  date?: string;
  subject?: string;
  html?: string;
  text?: string;
};

export type DeliveryKeyKind = "auto" | "test";

export interface ValidatedDeliverRequest {
  to: string;
  date: string;
  subject: string;
  html: string;
  text: string;
  logicalKeyHash: string;
  fingerprint: string;
  keyKind: DeliveryKeyKind;
  providerKey: string;
}

export type DeliverValidationResult =
  | { ok: true; value: ValidatedDeliverRequest }
  | { ok: false; status: number; error: string };

const AUTO_KEY = /^arxiv-daily:auto:[0-9a-f]{64}$/;
const TEST_KEY = /^arxiv-daily:test:[0-9a-f]{32}$/;
const MAX_SUBJECT_LENGTH = 500;
const MAX_BODY_LENGTH = 2_000_000;

export async function validateDeliverRequest(input: {
  device: AuthenticatedDevice;
  idempotencyHeader: string | null;
  body: DeliverBody;
}): Promise<DeliverValidationResult> {
  const to = typeof input.body.to === "string"
    ? normalizeEmail(input.body.to)
    : "";
  if (!to || to !== input.device.email) {
    return {
      ok: false,
      status: 403,
      error: "to must match the verified email bound to this token",
    };
  }

  const date = typeof input.body.date === "string" ? input.body.date.trim() : "";
  const subject =
    typeof input.body.subject === "string" ? input.body.subject.trim() : "";
  const html = typeof input.body.html === "string" ? input.body.html : "";
  const text = typeof input.body.text === "string" ? input.body.text : "";
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return { ok: false, status: 400, error: "date must be YYYY-MM-DD" };
  }
  if (!subject || (!html && !text)) {
    return { ok: false, status: 400, error: "subject and html or text required" };
  }
  if (
    subject.length > MAX_SUBJECT_LENGTH ||
    html.length > MAX_BODY_LENGTH ||
    text.length > MAX_BODY_LENGTH
  ) {
    return { ok: false, status: 413, error: "delivery payload is too large" };
  }

  const logicalKey = input.idempotencyHeader?.trim() ?? "";
  const keyKind: DeliveryKeyKind | undefined = AUTO_KEY.test(logicalKey)
    ? "auto"
    : TEST_KEY.test(logicalKey)
      ? "test"
      : undefined;
  if (!keyKind || logicalKey.length > 128) {
    return {
      ok: false,
      status: 400,
      error: "Idempotency-Key must be a supported bounded auto or test key",
    };
  }

  // Automatic delivery identity is authoritative server state. The client key is
  // only a kind marker; allowing its hash into the ledger/provider identity would
  // let one logical digest select multiple provider keys.
  const logicalKeyHash = await sha256Hex(JSON.stringify(
    keyKind === "auto"
      ? ["delivery-v2", "auto", input.device.identity, input.device.recipientIdentity, date]
      : ["delivery-v2", "test", input.device.identity, input.device.recipientIdentity, logicalKey],
  ));
  const fingerprint = await sha256Hex(JSON.stringify([
    input.device.identity,
    input.device.recipientIdentity,
    date,
    subject,
    html,
    text,
    keyKind,
  ]));
  const providerKey = [
    "arxiv-daily:relay:v2",
    keyKind,
    logicalKeyHash,
  ].join(":");
  if (providerKey.length > 128) {
    return { ok: false, status: 500, error: "provider key construction failed" };
  }

  return {
    ok: true,
    value: {
      to,
      date,
      subject,
      html,
      text,
      logicalKeyHash,
      fingerprint,
      keyKind,
      providerKey,
    },
  };
}
