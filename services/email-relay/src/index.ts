import { isPlausibleEmail, normalizeEmail, randomToken, sha256Hex } from "./crypto";
import {
  checkAndIncrRateLimit,
  putDevice,
  putPending,
  takePending,
  type Env,
} from "./kv";
import { sendResendEmail } from "./resend";
import { runDeliver, type DeliverBody } from "./deliver-logic";

export type { Env };
export { DeliverGate } from "./deliver-gate";

/** Verify-start limits (per rolling TTL window on the KV key). */
const VERIFY_EMAIL_LIMIT = 3;
const VERIFY_EMAIL_WINDOW_SEC = 3600;
const VERIFY_IP_LIMIT = 10;
const VERIFY_IP_WINDOW_SEC = 3600;

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
    return deliverViaGate(request, env);
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

/**
 * Route deliver through a Durable Object keyed by Idempotency-Key (or token),
 * so concurrent sends for the same logical mail are single-threaded.
 */
async function deliverViaGate(request: Request, env: Env): Promise<Response> {
  assertSecrets(env);

  // Clone body for possible fallback; DO gets a new Request.
  const auth = request.headers.get("Authorization");
  const idemp = request.headers.get("Idempotency-Key");
  let bodyText: string;
  try {
    bodyText = await request.text();
  } catch {
    return json({ error: "invalid body" }, 400);
  }

  let parsed: DeliverBody = {};
  try {
    parsed = JSON.parse(bodyText) as DeliverBody;
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  const gate = env.DELIVER_GATE;
  if (gate) {
    const keyMaterial =
      idemp?.trim() ||
      `${typeof parsed.date === "string" ? parsed.date : ""}|${typeof parsed.to === "string" ? parsed.to : ""}|${auth ?? ""}`;
    const objectId = gate.idFromName(await sha256Hex(keyMaterial || "default"));
    const stub = gate.get(objectId);
    const doReq = new Request("https://deliver-gate/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(auth ? { Authorization: auth } : {}),
        ...(idemp ? { "Idempotency-Key": idemp } : {}),
      },
      body: bodyText,
    });
    const res = await stub.fetch(doReq);
    const text = await res.text();
    return new Response(text, {
      status: res.status,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        ...CORS_HEADERS,
      },
    });
  }

  // Fallback if DO binding missing (local misconfig): still run logic (weaker).
  const outcome = await runDeliver({
    env,
    authorizationHeader: auth,
    idempotencyHeader: idemp,
    body: parsed,
  });
  return json(outcome.body, outcome.status);
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
