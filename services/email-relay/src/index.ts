import { isPlausibleEmail, normalizeEmail, randomToken, sha256Hex } from "./crypto";
import {
  authenticateDevice,
  checkAndIncrRateLimit,
  putDevice,
  putPending,
  takePending,
  type Env,
} from "./kv";
import { sendResendEmail } from "./resend";
import type { DeliverBody } from "./deliver-logic";
import {
  fetchCutoverStatus,
  isCutoverOperationId,
  postCutoverAction,
  type CutoverAction,
} from "./cutover-control";

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
    } catch {
      return json({ error: "relay request failed" }, 500);
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
  if (
    (request.method === "GET" || request.method === "POST") &&
    path === "/internal/delivery-v2/cutover"
  ) {
    return cutoverControl(request, env);
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
  } catch {
    return json({ error: "verification delivery failed" }, 500);
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
    `<p>Your address is verified for Official delivery (Beta).</p>` +
      `<p>Copy this token into Obsidian → arXiv Daily → Email → <em>Hosted token</em>:</p>` +
      `<pre style="white-space:pre-wrap;word-break:break-all;background:#f4f4f5;padding:12px;border-radius:8px">${escapeHtml(deviceToken)}</pre>` +
      `<p>Keep this token private. You can close this tab after pasting it into the plugin.</p>`,
    200,
  );
}

/** Authenticate first, then route every delivery for one device to one DO. */
async function deliverViaGate(request: Request, env: Env): Promise<Response> {
  assertSecrets(env);
  const auth = request.headers.get("Authorization") ?? "";
  const match = /^Bearer\s+(.+)$/i.exec(auth);
  const rawToken = match?.[1]?.trim() ?? "";
  if (!rawToken) return json({ error: "missing bearer token" }, 401);

  const device = await authenticateDevice(env, rawToken);
  if (!device) return json({ error: "invalid or revoked token" }, 401);

  const idemp = request.headers.get("Idempotency-Key");
  let bodyText: string;
  try {
    bodyText = await request.text();
  } catch {
    return json({ error: "invalid body" }, 400);
  }
  try {
    JSON.parse(bodyText) as DeliverBody;
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  const gate = env.DELIVER_GATE;
  if (!gate) {
    return json(
      {
        error: "DELIVER_GATE Durable Object binding is not configured",
        ambiguous: false,
      },
      503,
    );
  }

  const objectId = gate.idFromName(`device-v2:${device.identity}`);
  const stub = gate.get(objectId);
  const doReq = new Request("https://deliver-gate/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Device-Identity": device.identity,
      "X-Device-Created-At": device.createdAt,
      "X-Device-Delivery-Generation": String(device.deliveryGeneration ?? ""),
      "X-Recipient-Identity": device.recipientIdentity,
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

async function cutoverControl(request: Request, env: Env): Promise<Response> {
  const configuredToken = env.DELIVERY_V2_CUTOVER_TOKEN?.trim() ?? "";
  const auth = /^Bearer\s+(.+)$/i.exec(
    request.headers.get("Authorization") ?? "",
  )?.[1]?.trim() ?? "";
  if (
    !configuredToken ||
    !auth ||
    await sha256Hex(auth) !== await sha256Hex(configuredToken)
  ) {
    return json({ error: "not found" }, 404);
  }
  if (!env.DELIVER_GATE || !env.TOKEN_SECRET?.trim()) {
    return json({ error: "cutover control is unavailable" }, 503);
  }

  try {
    const response = request.method === "GET"
      ? await fetchCutoverStatus(env)
      : await forwardCutoverAction(request, env);
    return relayJson(response);
  } catch {
    return json({ error: "cutover control is unavailable" }, 503);
  }
}

async function forwardCutoverAction(
  request: Request,
  env: Env,
): Promise<Response> {
  let body: Record<string, unknown>;
  try {
    body = await request.json() as Record<string, unknown>;
  } catch {
    return json({ error: "invalid cutover action" }, 400);
  }
  if (
    Object.keys(body).some(
      (key) => key !== "action" && key !== "operationId" && key !== "attestation",
    ) ||
    (body.action !== "inventory" &&
      body.action !== "provider-fence" &&
      body.action !== "observe" &&
      body.action !== "seal" &&
      body.action !== "repair") ||
    !isCutoverOperationId(body.operationId) ||
    (body.attestation !== undefined && typeof body.attestation !== "string")
  ) {
    return json({ error: "invalid cutover action" }, 400);
  }
  return postCutoverAction(
    env,
    body.action as CutoverAction,
    body.operationId,
    typeof body.attestation === "string" ? body.attestation : undefined,
  );
}

async function relayJson(response: Response): Promise<Response> {
  return new Response(await response.text(), {
    status: response.status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...CORS_HEADERS,
    },
  });
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
