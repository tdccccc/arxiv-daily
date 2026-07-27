/**
 * Pure-ish deliver steps shared by the HTTP handler and Durable Object.
 * Callers must serialize concurrent calls for the same idempotency key
 * (Durable Object) — KV alone cannot CAS.
 */
import { normalizeEmail, randomToken } from "./crypto";
import {
  clearIdempotent,
  completeIdempotent,
  dailyQuotaLimit,
  getDevice,
  getQuotaCount,
  incrementQuota,
  reserveIdempotent,
  type Env,
} from "./kv";
import { sendResendEmail } from "./resend";

export type DeliverBody = {
  to?: string;
  date?: string;
  subject?: string;
  html?: string;
  text?: string;
};

export type DeliverOutcome =
  | { status: number; body: Record<string, unknown> };

export async function runDeliver(opts: {
  env: Env;
  authorizationHeader: string | null;
  idempotencyHeader: string | null;
  body: DeliverBody;
}): Promise<DeliverOutcome> {
  const { env } = opts;
  const auth = opts.authorizationHeader ?? "";
  const m = /^Bearer\s+(.+)$/i.exec(auth);
  const deviceToken = m?.[1]?.trim() ?? "";
  if (!deviceToken) {
    return { status: 401, body: { error: "missing bearer token" } };
  }

  const device = await getDevice(env, deviceToken);
  if (!device) {
    return { status: 401, body: { error: "invalid or revoked token" } };
  }

  const to =
    typeof opts.body.to === "string" ? normalizeEmail(opts.body.to) : "";
  if (!to || to !== device.email) {
    return {
      status: 403,
      body: { error: "to must match the verified email bound to this token" },
    };
  }

  const subject =
    typeof opts.body.subject === "string" ? opts.body.subject.trim() : "";
  const html = typeof opts.body.html === "string" ? opts.body.html : "";
  const text = typeof opts.body.text === "string" ? opts.body.text : "";
  if (!subject || (!html && !text)) {
    return { status: 400, body: { error: "subject and html or text required" } };
  }

  const idempotency =
    opts.idempotencyHeader?.trim() ||
    `${typeof opts.body.date === "string" ? opts.body.date : ""}|${to}`;

  const claim = randomToken(16);
  if (idempotency) {
    const reserved = await reserveIdempotent(env, idempotency, claim);
    if (reserved.status === "done") {
      return {
        status: 200,
        body: { ok: true, id: reserved.id, deduped: true },
      };
    }
    if (reserved.status === "pending_other") {
      return {
        status: 409,
        body: {
          error: "delivery already in progress for this idempotency key",
        },
      };
    }
  }

  const limit = dailyQuotaLimit(env);
  const used = await getQuotaCount(env, to);
  if (used >= limit) {
    if (idempotency) await clearIdempotent(env, idempotency);
    return {
      status: 429,
      body: {
        error: `daily quota exceeded (${limit} per UTC day)`,
        quota: used,
      },
    };
  }

  // Re-check claim immediately before Resend (narrow race if not using DO).
  if (idempotency) {
    const again = await reserveIdempotent(env, idempotency, claim);
    if (again.status === "done") {
      return {
        status: 200,
        body: { ok: true, id: again.id, deduped: true },
      };
    }
    if (again.status === "pending_other") {
      return {
        status: 409,
        body: {
          error: "delivery already in progress for this idempotency key",
        },
      };
    }
  }

  let sent: { id?: string };
  try {
    sent = await sendResendEmail(env, {
      to,
      subject,
      html: html || `<pre>${escapeHtml(text)}</pre>`,
      text: text || stripTags(html),
    });
  } catch (e) {
    if (idempotency) await clearIdempotent(env, idempotency);
    const message = e instanceof Error ? e.message : String(e);
    return { status: 502, body: { error: message } };
  }

  await incrementQuota(env, to);
  const messageId = sent.id?.trim() || `local:${claim}`;
  if (idempotency) {
    await completeIdempotent(env, idempotency, messageId);
  }

  return { status: 200, body: { ok: true, id: messageId } };
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function stripTags(html: string): string {
  return html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
}
