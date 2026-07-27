import { isPlausibleEmail, normalizeEmail, randomToken } from "./crypto";
import {
  checkAndIncrRateLimit,
  clearIdempotent,
  completeIdempotent,
  dailyQuotaLimit,
  getDevice,
  getQuotaCount,
  incrementQuota,
  putDevice,
  putPending,
  reserveIdempotent,
  takePending,
  type Env,
} from "./kv";
import { sendResendEmail } from "./resend";

export type { Env };

/** Verify-start limits (per rolling TTL window on the KV key). */
const VERIFY_EMAIL_LIMIT = 3;
const VERIFY_EMAIL_WINDOW_SEC = 3600;
const VERIFY_IP_LIMIT = 10;
const VERIFY_IP_WINDOW_SEC = 3600;

// CORS: plugin uses Obsidian requestUrl (no browser CORS). Keep minimal OPTIONS
// for local curl/debug only — do not reflect arbitrary origins for credentialed use.
const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers":
    "Content-Type, Authorization, Idempotency-Key",
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      return await handle(request, env);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      return json({ error: message }, 500);
    }
  },
};

async function handle(request: Request, env: Env): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";

  if (request.method === "GET" && (path === "/" || path === "/health")) {
    return json({
      ok: true,
      service: "arxiv-daily-email-relay",
      beta: true,
    });
  }

  if (request.method === "POST" && path === "/v1/verify/start") {
    return verifyStart(request, env);
  }
  if (request.method === "GET" && path === "/v1/verify") {
    return verifyComplete(url, env);
  }
  if (request.method === "POST" && path === "/v1/deliver") {
    return deliver(request, env);
  }

  return json({ error: "not found" }, 404);
}

async function verifyStart(request: Request, env: Env): Promise<Response> {
  assertSecrets(env);
  let body: { email?: string };
  try {
    body = (await request.json()) as { email?: string };
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }
  const email = typeof body.email === "string" ? body.email.trim() : "";
  if (!isPlausibleEmail(email)) {
    return json({ error: "invalid email" }, 400);
  }

  const emailNorm = normalizeEmail(email);
  const ip = clientIp(request);

  const emailRl = await checkAndIncrRateLimit(
    env,
    "verify-email",
    emailNorm,
    VERIFY_EMAIL_LIMIT,
    VERIFY_EMAIL_WINDOW_SEC,
  );
  if (!emailRl.ok) {
    // Generic message — do not confirm whether email exists / was sent.
    return json({ ok: true, message: "verification email sent" });
  }
  const ipRl = await checkAndIncrRateLimit(
    env,
    "verify-ip",
    ip,
    VERIFY_IP_LIMIT,
    VERIFY_IP_WINDOW_SEC,
  );
  if (!ipRl.ok) {
    return json({ ok: true, message: "verification email sent" });
  }

  const token = randomToken(24);
  await putPending(env, token, email, 3600);
  const link = `${env.PUBLIC_BASE_URL.replace(/\/$/, "")}/v1/verify?token=${token}`;

  try {
    await sendResendEmail(env, {
      to: emailNorm,
      subject: "Verify your arXiv Daily email (Beta)",
      text:
        `Confirm this address for Official delivery (Beta).\n\n` +
        `Open this link within 1 hour:\n${link}\n\n` +
        `If you did not request this, ignore this email.`,
      html:
        `<p>Confirm this address for <strong>arXiv Daily Official delivery (Beta)</strong>.</p>` +
        `<p><a href="${escapeHtml(link)}">Verify email</a></p>` +
        `<p>Or copy: <code>${escapeHtml(link)}</code></p>` +
        `<p>Link expires in 1 hour. If you did not request this, ignore this email.</p>`,
    });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return json({ error: message }, 500);
  }

  return json({ ok: true, message: "verification email sent" });
}

async function verifyComplete(url: URL, env: Env): Promise<Response> {
  assertSecrets(env);
  const token = url.searchParams.get("token")?.trim() ?? "";
  if (!token) {
    return htmlPage("Missing token", "<p>Invalid verification link.</p>", 400);
  }
  const pending = await takePending(env, token);
  if (!pending) {
    return htmlPage(
      "Link expired",
      "<p>This verification link is invalid or expired. Request a new one from the plugin.</p>",
      400,
    );
  }

  const deviceToken = randomToken(32);
  await putDevice(env, deviceToken, pending.email);

  return htmlPage(
    "Email verified",
    `<p>Verified <strong>${escapeHtml(pending.email)}</strong> for Official delivery (Beta).</p>` +
      `<p>Copy this token into Obsidian → arXiv Daily → Email → <em>Hosted token</em>:</p>` +
      `<pre style="white-space:pre-wrap;word-break:break-all;background:#f4f4f5;padding:12px;border-radius:8px">${escapeHtml(deviceToken)}</pre>` +
      `<p>Keep this token private. You can close this tab after pasting it into the plugin.</p>`,
    200,
  );
}

async function deliver(request: Request, env: Env): Promise<Response> {
  assertSecrets(env);
  const auth = request.headers.get("Authorization") ?? "";
  const m = /^Bearer\s+(.+)$/i.exec(auth);
  const deviceToken = m?.[1]?.trim() ?? "";
  if (!deviceToken) {
    return json({ error: "missing bearer token" }, 401);
  }

  const device = await getDevice(env, deviceToken);
  if (!device) {
    return json({ error: "invalid or revoked token" }, 401);
  }

  let body: {
    to?: string;
    date?: string;
    subject?: string;
    html?: string;
    text?: string;
  };
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  const to = typeof body.to === "string" ? normalizeEmail(body.to) : "";
  if (!to || to !== device.email) {
    return json(
      { error: "to must match the verified email bound to this token" },
      403,
    );
  }

  const subject = typeof body.subject === "string" ? body.subject.trim() : "";
  const html = typeof body.html === "string" ? body.html : "";
  const text = typeof body.text === "string" ? body.text : "";
  if (!subject || (!html && !text)) {
    return json({ error: "subject and html or text required" }, 400);
  }

  const idempotency =
    request.headers.get("Idempotency-Key")?.trim() ||
    `${typeof body.date === "string" ? body.date : ""}|${to}`;

  const claim = randomToken(16);
  if (idempotency) {
    const reserved = await reserveIdempotent(env, idempotency, claim);
    if (reserved.status === "done") {
      return json({ ok: true, id: reserved.id, deduped: true });
    }
    if (reserved.status === "pending_other") {
      return json(
        { error: "delivery already in progress for this idempotency key" },
        409,
      );
    }
  }

  const limit = dailyQuotaLimit(env);
  const used = await getQuotaCount(env, to);
  if (used >= limit) {
    if (idempotency) await clearIdempotent(env, idempotency);
    return json(
      { error: `daily quota exceeded (${limit} per UTC day)`, quota: used },
      429,
    );
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
    return json({ error: message }, 502);
  }

  await incrementQuota(env, to);
  // Always store a stable id so retries dedupe even if Resend omits id.
  const messageId = sent.id?.trim() || `local:${claim}`;
  if (idempotency) {
    await completeIdempotent(env, idempotency, messageId);
  }

  return json({ ok: true, id: messageId });
}

function clientIp(request: Request): string {
  return (
    request.headers.get("CF-Connecting-IP") ||
    request.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() ||
    "unknown"
  );
}

function assertSecrets(env: Env): void {
  if (!env.RESEND_API_KEY?.trim()) {
    throw new Error("RESEND_API_KEY secret is not configured");
  }
  if (!env.TOKEN_SECRET?.trim()) {
    throw new Error("TOKEN_SECRET secret is not configured");
  }
  if (!env.FROM_EMAIL?.trim()) {
    throw new Error("FROM_EMAIL is not configured");
  }
  if (!env.PUBLIC_BASE_URL?.trim()) {
    throw new Error("PUBLIC_BASE_URL is not configured");
  }
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...CORS_HEADERS,
    },
  });
}

function htmlPage(title: string, body: string, status = 200): Response {
  const doc =
    `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>` +
    `<meta name="viewport" content="width=device-width,initial-scale=1"/>` +
    `<title>${escapeHtml(title)} · arXiv Daily</title>` +
    `<style>body{font-family:system-ui,sans-serif;max-width:36rem;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#18181b}</style>` +
    `</head><body><h1>${escapeHtml(title)}</h1>${body}</body></html>`;
  return new Response(doc, {
    status,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      ...CORS_HEADERS,
    },
  });
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

// Re-export for tests that import handler pieces via default fetch
export { handle as handleRequest };
