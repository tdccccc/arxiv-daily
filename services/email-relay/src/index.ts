import { isPlausibleEmail, normalizeEmail, randomToken } from "./crypto";
import {
  dailyQuotaLimit,
  getDevice,
  getIdempotent,
  getQuotaCount,
  incrementQuota,
  putDevice,
  putIdempotent,
  putPending,
  takePending,
  type Env,
} from "./kv";
import { sendResendEmail } from "./resend";

export type { Env };

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
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

  const token = randomToken(24);
  await putPending(env, token, email, 3600);
  const link = `${env.PUBLIC_BASE_URL.replace(/\/$/, "")}/v1/verify?token=${token}`;

  await sendResendEmail(env, {
    to: normalizeEmail(email),
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

  if (idempotency) {
    const existing = await getIdempotent(env, idempotency);
    if (existing) {
      return json({ ok: true, id: existing, deduped: true });
    }
  }

  const limit = dailyQuotaLimit(env);
  const used = await getQuotaCount(env, to);
  if (used >= limit) {
    return json(
      { error: `daily quota exceeded (${limit} per UTC day)`, quota: used },
      429,
    );
  }

  const sent = await sendResendEmail(env, {
    to,
    subject,
    html: html || `<pre>${escapeHtml(text)}</pre>`,
    text: text || stripTags(html),
  });

  await incrementQuota(env, to);
  if (idempotency && sent.id) {
    await putIdempotent(env, idempotency, sent.id);
  }

  return json({ ok: true, id: sent.id ?? null });
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
