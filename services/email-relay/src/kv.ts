import { normalizeEmail, sha256Hex, utcDateKey } from "./crypto";

export interface Env {
  STORE: KVNamespace;
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

export function pendingKey(token: string): string {
  return `${PENDING_PREFIX}${token}`;
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

export async function hashDeviceToken(
  rawToken: string,
  secret: string,
): Promise<string> {
  return sha256Hex(`${secret}:device:${rawToken}`);
}

export async function putPending(
  env: Env,
  token: string,
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
  await env.STORE.put(pendingKey(token), JSON.stringify(value), {
    expirationTtl: ttlSeconds,
  });
}

export async function takePending(
  env: Env,
  token: string,
): Promise<PendingVerify | null> {
  const key = pendingKey(token);
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
  // Expire shortly after UTC day end (~2 days safety)
  await env.STORE.put(key, String(next), { expirationTtl: 60 * 60 * 48 });
  return next;
}

export async function getIdempotent(
  env: Env,
  key: string,
): Promise<string | null> {
  return env.STORE.get(idempKey(key));
}

export async function putIdempotent(
  env: Env,
  key: string,
  messageId: string,
): Promise<void> {
  await env.STORE.put(idempKey(key), messageId, {
    expirationTtl: 60 * 60 * 24 * 14,
  });
}

export function dailyQuotaLimit(env: Env): number {
  const n = Number(env.DAILY_QUOTA || "2");
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 2;
}
