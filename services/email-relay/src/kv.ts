import { normalizeEmail, sha256Hex, utcDateKey } from "./crypto";

export interface Env {
  STORE: KVNamespace;
  /** Durable Object namespace: serializes deliver per idempotency key (KV has no CAS). */
  DELIVER_GATE?: DurableObjectNamespace;
  RESEND_API_KEY: string;
  TOKEN_SECRET: string;
  PUBLIC_BASE_URL: string;
  FROM_EMAIL: string;
  FROM_NAME: string;
  DAILY_QUOTA: string;
}

export interface DeviceRecord {
  email: string;
  createdAt: string;
}

export interface PendingVerify {
  email: string;
  createdAt: string;
  expiresAt: string;
}

const PENDING_PREFIX = "pending:";
const DEVICE_PREFIX = "device:";
const QUOTA_PREFIX = "quota:";
const IDEMP_PREFIX = "idemp:";
const RL_PREFIX = "rl:";

export function pendingKey(tokenHash: string): string {
  return `${PENDING_PREFIX}${tokenHash}`;
}

export function deviceKey(tokenHash: string): string {
  return `${DEVICE_PREFIX}${tokenHash}`;
}

export function quotaKey(email: string, day: string = utcDateKey()): string {
  return `${QUOTA_PREFIX}${normalizeEmail(email)}:${day}`;
}

export function idempKey(key: string): string {
  return `${IDEMP_PREFIX}${key}`;
}

export function rateLimitKey(kind: string, id: string): string {
  return `${RL_PREFIX}${kind}:${id}`;
}

export async function hashDeviceToken(
  rawToken: string,
  secret: string,
): Promise<string> {
  return sha256Hex(`${secret}:device:${rawToken}`);
}

export async function hashPendingToken(
  rawToken: string,
  secret: string,
): Promise<string> {
  return sha256Hex(`${secret}:pending:${rawToken}`);
}

export async function putPending(
  env: Env,
  rawToken: string,
  email: string,
  ttlSeconds = 3600,
): Promise<void> {
  const now = new Date();
  const expires = new Date(now.getTime() + ttlSeconds * 1000);
  const value: PendingVerify = {
    email: normalizeEmail(email),
    createdAt: now.toISOString(),
    expiresAt: expires.toISOString(),
  };
  const hash = await hashPendingToken(rawToken, env.TOKEN_SECRET);
  await env.STORE.put(pendingKey(hash), JSON.stringify(value), {
    expirationTtl: ttlSeconds,
  });
}

export async function takePending(
  env: Env,
  rawToken: string,
): Promise<PendingVerify | null> {
  const hash = await hashPendingToken(rawToken, env.TOKEN_SECRET);
  const key = pendingKey(hash);
  const raw = await env.STORE.get(key);
  if (!raw) return null;
  await env.STORE.delete(key);
  try {
    return JSON.parse(raw) as PendingVerify;
  } catch {
    return null;
  }
}

export async function putDevice(
  env: Env,
  rawToken: string,
  email: string,
): Promise<void> {
  const hash = await hashDeviceToken(rawToken, env.TOKEN_SECRET);
  const value: DeviceRecord = {
    email: normalizeEmail(email),
    createdAt: new Date().toISOString(),
  };
  // ~1 year
  await env.STORE.put(deviceKey(hash), JSON.stringify(value), {
    expirationTtl: 60 * 60 * 24 * 365,
  });
}

export async function getDevice(
  env: Env,
  rawToken: string,
): Promise<DeviceRecord | null> {
  const hash = await hashDeviceToken(rawToken, env.TOKEN_SECRET);
  const raw = await env.STORE.get(deviceKey(hash));
  if (!raw) return null;
  try {
    return JSON.parse(raw) as DeviceRecord;
  } catch {
    return null;
  }
}

export async function getQuotaCount(env: Env, email: string): Promise<number> {
  const raw = await env.STORE.get(quotaKey(email));
  if (!raw) return 0;
  const n = Number(raw);
  return Number.isFinite(n) ? n : 0;
}

export async function incrementQuota(
  env: Env,
  email: string,
): Promise<number> {
  const key = quotaKey(email);
  const next = (await getQuotaCount(env, email)) + 1;
  await env.STORE.put(key, String(next), { expirationTtl: 60 * 60 * 48 });
  return next;
}

export type IdempValue =
  | { kind: "done"; id: string }
  | { kind: "pending"; claim: string };

export async function getIdempotentRaw(
  env: Env,
  key: string,
): Promise<string | null> {
  return env.STORE.get(idempKey(key));
}

export function parseIdemp(raw: string | null): IdempValue | null {
  if (!raw) return null;
  if (raw.startsWith("pending:")) {
    return { kind: "pending", claim: raw.slice("pending:".length) };
  }
  if (raw.startsWith("done:")) {
    return { kind: "done", id: raw.slice("done:".length) };
  }
  // Legacy plain message id
  return { kind: "done", id: raw };
}

/**
 * Reserve idempotency before calling Resend.
 * Returns existing done id, conflict if another pending, or claim string if reserved.
 */
export async function reserveIdempotent(
  env: Env,
  key: string,
  claim: string,
): Promise<
  | { status: "reserved"; claim: string }
  | { status: "done"; id: string }
  | { status: "pending_other" }
> {
  const existing = parseIdemp(await getIdempotentRaw(env, key));
  if (existing?.kind === "done") {
    return { status: "done", id: existing.id };
  }
  if (existing?.kind === "pending" && existing.claim !== claim) {
    return { status: "pending_other" };
  }
  // Best-effort reserve (KV has no CAS; claim reduces double-send window)
  await env.STORE.put(idempKey(key), `pending:${claim}`, {
    expirationTtl: 120,
  });
  const after = parseIdemp(await getIdempotentRaw(env, key));
  if (after?.kind === "done") {
    return { status: "done", id: after.id };
  }
  if (after?.kind === "pending" && after.claim !== claim) {
    return { status: "pending_other" };
  }
  return { status: "reserved", claim };
}

export async function completeIdempotent(
  env: Env,
  key: string,
  messageId: string,
): Promise<void> {
  await env.STORE.put(idempKey(key), `done:${messageId}`, {
    expirationTtl: 60 * 60 * 24 * 14,
  });
}

export async function clearIdempotent(
  env: Env,
  key: string,
): Promise<void> {
  await env.STORE.delete(idempKey(key));
}

/** Increment rate-limit counter; returns false if over limit. */
export async function checkAndIncrRateLimit(
  env: Env,
  kind: string,
  id: string,
  limit: number,
  windowSeconds: number,
): Promise<{ ok: true; count: number } | { ok: false; count: number }> {
  const key = rateLimitKey(kind, id);
  const raw = await env.STORE.get(key);
  const count = raw && Number.isFinite(Number(raw)) ? Number(raw) : 0;
  if (count >= limit) {
    return { ok: false, count };
  }
  const next = count + 1;
  await env.STORE.put(key, String(next), { expirationTtl: windowSeconds });
  return { ok: true, count: next };
}

export function dailyQuotaLimit(env: Env): number {
  const n = Number(env.DAILY_QUOTA || "5");
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 5;
}

// --- legacy aliases used by older tests (if any) ---
export async function getIdempotent(
  env: Env,
  key: string,
): Promise<string | null> {
  const v = parseIdemp(await getIdempotentRaw(env, key));
  return v?.kind === "done" ? v.id : null;
}

export async function putIdempotent(
  env: Env,
  key: string,
  messageId: string,
): Promise<void> {
  await completeIdempotent(env, key, messageId);
}
