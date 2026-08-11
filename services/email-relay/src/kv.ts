import {
  hmacSha256Hex,
  isPlausibleEmail,
  normalizeEmail,
  sha256Hex,
} from "./crypto";

export interface Env {
  STORE: KVNamespace;
  /** Recipient-scoped automatic gates, device-scoped test gates, and cutover control. */
  DELIVER_GATE?: DurableObjectNamespace;
  /** Operator-only bearer secret for the single-version cutover control API. */
  DELIVERY_V2_CUTOVER_TOKEN?: string;
  RESEND_API_KEY: string;
  TOKEN_SECRET: string;
  /** Long-lived root for recipient routing and automatic delivery identity. */
  IDENTITY_SECRET?: string;
  PUBLIC_BASE_URL: string;
  FROM_EMAIL: string;
  FROM_NAME: string;
  DAILY_QUOTA: string;
  /** Non-secret immutable deployment/build identity supplied by Worker config. */
  BUILD_IDENTITY?: string;
}

export const DELIVERY_PROTOCOL_GENERATION = 2 as const;

export interface DeliveryBinding {
  protocolGeneration: typeof DELIVERY_PROTOCOL_GENERATION;
  buildIdentity: string;
  readyGeneration: number;
}

export interface DeviceRecord {
  email: string;
  createdAt: string;
  /** Present only on identities issued by the delivery-v2 Worker. */
  deliveryGeneration?: 2;
  protocolGeneration?: number;
  buildIdentity?: string;
  readyGeneration?: number;
}

export interface AuthenticatedDevice {
  /** Secret-scoped hash of the raw token. Safe for routing and internal binding. */
  identity: string;
  /** Normalized verified recipient; never used in DO names or provider keys. */
  email: string;
  recipientIdentity: string;
  createdAt: string;
  deliveryGeneration?: 2;
  protocolGeneration?: number;
  buildIdentity?: string;
  readyGeneration?: number;
}

export interface PendingVerify {
  email: string;
  createdAt: string;
  expiresAt: string;
}

const PENDING_PREFIX = "pending:";
const DEVICE_PREFIX = "device:";
const DEVICE_V2_PREFIX = "device-v2:";
const RL_PREFIX = "rl:";
export const DELIVERY_V3_CUTOVER_AUDIT_KEY = "cutover:delivery-v3-audit";
export const DELIVERY_V2_KV_VISIBILITY_MS = 60 * 1000;
export const DELIVERY_V2_LEGACY_PENDING_TTL_MS = 120 * 1000;
const IDEMP_PREFIX = "idemp:";
const HASHED_LEGACY_AUTO_KEY = /^arxiv-daily:auto:[0-9a-f]{64}$/;
const PLAIN_LEGACY_AUTO_KEY = /^(\d{4}-\d{2}-\d{2})\|([^|\s]+@[^|\s]+)$/;
const HASHED_TEST_KEY = /^arxiv-daily:test:[0-9a-f]{32}$/;
const PLAIN_TEST_KEY = /^test\|\d{4}-\d{2}-\d{2}\|[^|\s]+@[^|\s]+\|[^|]+$/;

export type LegacyDeliveryEvidence = "none" | "done" | "attempted";

export interface DeliveryV3CutoverAuditMarker {
  schemaVersion: 3;
  kind: "delivery-v2-cutover-audit";
  proofVersion: 1;
  providerFence: "old-resend-credential-revoked";
  providerFencedAt: string;
  inventoryStartedAt: string;
  inventoryCompletedAt: string;
  inventoryAutomaticKeyCount: number;
  postFenceScanStartedAt: string;
  postFenceScanCompletedAt: string;
  postFenceAutomaticKeyCount: number;
  followupScanStartedAt: string;
  followupScanCompletedAt: string;
  followupAutomaticKeyCount: number;
  legacyAutoEvidenceSnapshot: "exact-canonical-map";
  legacyAutoEvidence: Record<string, "done" | "attempted">;
  identitySecretFingerprint: string;
  buildIdentity: string;
  protocolGeneration: typeof DELIVERY_PROTOCOL_GENERATION;
  constructedAt: string;
  proof: string;
}

export function pendingKey(tokenHash: string): string {
  return `${PENDING_PREFIX}${tokenHash}`;
}

export function deviceKey(tokenHash: string): string {
  return `${DEVICE_PREFIX}${tokenHash}`;
}

export function deviceV2Key(tokenHash: string): string {
  return `${DEVICE_V2_PREFIX}${tokenHash}`;
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

export async function hashRecipientIdentity(
  normalizedEmail: string,
  identitySecret: string,
): Promise<string> {
  return hmacSha256Hex(
    identitySecret,
    `delivery-v2-recipient\u0000${normalizeEmail(normalizedEmail)}`,
  );
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

export async function peekPendingByIdentity(
  env: Env,
  pendingIdentity: string,
): Promise<PendingVerify | null> {
  if (!isHash(pendingIdentity)) return null;
  const raw = await env.STORE.get(pendingKey(pendingIdentity));
  if (!raw) return null;
  const parsed = parsePending(raw);
  if (!parsed || Date.parse(parsed.expiresAt) <= Date.now()) return null;
  return parsed;
}

export async function deletePendingByIdentity(
  env: Env,
  pendingIdentity: string,
): Promise<void> {
  if (!isHash(pendingIdentity)) {
    throw new Error("pending identity is invalid");
  }
  await env.STORE.delete(pendingKey(pendingIdentity));
}

export async function putDevice(
  env: Env,
  rawToken: string,
  email: string,
  binding: DeliveryBinding,
  now: Date = new Date(),
): Promise<void> {
  const hash = await hashDeviceToken(rawToken, env.TOKEN_SECRET);
  const value: DeviceRecord = {
    email: normalizeEmail(email),
    createdAt: now.toISOString(),
    deliveryGeneration: 2,
    ...binding,
  };
  // Approximately one year; expiry revokes inactive devices.
  await env.STORE.put(deviceV2Key(hash), JSON.stringify(value), {
    expirationTtl: 60 * 60 * 24 * 365,
  });
}

export async function authenticateDevice(
  env: Env,
  rawToken: string,
): Promise<AuthenticatedDevice | null> {
  const identitySecret = configuredIdentitySecret(env);
  if (!identitySecret) return null;
  const identity = await hashDeviceToken(rawToken, env.TOKEN_SECRET);
  const v2Raw = await env.STORE.get(deviceV2Key(identity));
  const legacyRaw = v2Raw === null
    ? await env.STORE.get(deviceKey(identity))
    : null;
  const device = v2Raw !== null
    ? parseDevice(v2Raw, true)
    : legacyRaw !== null ? parseDevice(legacyRaw, false) : null;
  if (!device) return null;
  return {
    identity,
    email: device.email,
    recipientIdentity: await hashRecipientIdentity(device.email, identitySecret),
    createdAt: device.createdAt,
    deliveryGeneration: device.deliveryGeneration,
    protocolGeneration: device.protocolGeneration,
    buildIdentity: device.buildIdentity,
    readyGeneration: device.readyGeneration,
  };
}

export async function getDevice(
  env: Env,
  rawToken: string,
): Promise<DeviceRecord | null> {
  const authenticated = await authenticateDevice(env, rawToken);
  return authenticated
    ? { email: authenticated.email, createdAt: "authenticated" }
    : null;
}

/**
 * Verification rate limits remain best-effort KV counters. They are not used
 * for delivery authorization, idempotency, or quota correctness.
 */
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
  if (count >= limit) return { ok: false, count };
  const next = count + 1;
  await env.STORE.put(key, String(next), { expirationTtl: windowSeconds });
  return { ok: true, count: next };
}

export function dailyQuotaLimit(env: Env): number {
  const n = Number(env.DAILY_QUOTA || "5");
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 5;
}

export async function writeDeliveryV3CutoverAuditMarker(
  env: Env,
  marker: DeliveryV3CutoverAuditMarker,
): Promise<string> {
  const serialized = JSON.stringify(marker);
  await env.STORE.put(DELIVERY_V3_CUTOVER_AUDIT_KEY, serialized);
  return sha256Hex(serialized);
}

/**
 * Scans the complete historical idempotency namespace. Every supported automatic
 * generation is projected onto one secret-scoped logical identity. Supported
 * test generations are ignored; any other idemp key makes the scan fail closed.
 */
export async function scanLegacyAutoDeliveryEvidence(
  env: Env,
  options: {
    collect?: boolean;
    onEvidence?: (
      identity: string,
      evidence: "done" | "attempted",
    ) => void | Promise<void>;
  } = {},
): Promise<Record<string, "done" | "attempted">> {
  const identitySecret = configuredIdentitySecret(env);
  if (!identitySecret) {
    throw new Error("legacy automatic delivery scan is unavailable");
  }
  const evidence: Record<string, "done" | "attempted"> = {};
  let cursor: string | undefined;
  do {
    const page = await env.STORE.list({
      prefix: IDEMP_PREFIX,
      ...(cursor ? { cursor } : {}),
    });
    if (!page || !Array.isArray(page.keys)) {
      throw new Error("legacy automatic delivery scan is invalid");
    }
    for (const entry of page.keys) {
      const key = entry?.name ?? "";
      if (!key.startsWith(IDEMP_PREFIX)) {
        throw new Error("legacy automatic delivery scan encountered an unsupported key");
      }
      const logicalKey = key.slice(IDEMP_PREFIX.length);
      if (HASHED_TEST_KEY.test(logicalKey) || PLAIN_TEST_KEY.test(logicalKey)) {
        continue;
      }

      let identity: string;
      if (HASHED_LEGACY_AUTO_KEY.test(logicalKey)) {
        identity = await hashLegacyAutomaticLogicalIdentity(
          identitySecret,
          logicalKey,
        );
      } else {
        const plain = PLAIN_LEGACY_AUTO_KEY.exec(logicalKey);
        if (!plain) {
          throw new Error(
            "legacy automatic delivery scan encountered an unsupported key",
          );
        }
        identity = await hashLegacyAutoDeliveryIdentity(
          identitySecret,
          plain[1]!,
          plain[2]!,
        );
      }
      const raw = await env.STORE.get(key);
      const observed = raw?.startsWith("pending:")
        ? "attempted"
        : raw ? "done" : "attempted";
      if (options.collect !== false) {
        evidence[identity] = evidence[identity] === "attempted" || observed === "attempted"
          ? "attempted"
          : "done";
      }
      await options.onEvidence?.(identity, observed);
    }
    if (page.list_complete) break;
    if (!page.cursor || page.cursor === cursor) {
      throw new Error("legacy automatic delivery scan did not advance");
    }
    cursor = page.cursor;
  } while (true);
  return evidence;
}

export async function hashLegacyAutoDeliveryIdentity(
  identitySecret: string,
  date: string,
  normalizedRecipient: string,
): Promise<string> {
  const logicalKey = `arxiv-daily:auto:${await sha256Hex(
    `${date}\u0000${normalizeEmail(normalizedRecipient)}`,
  )}`;
  return hashLegacyAutomaticLogicalIdentity(identitySecret, logicalKey);
}

async function hashLegacyAutomaticLogicalIdentity(
  identitySecret: string,
  logicalKey: string,
): Promise<string> {
  return hmacSha256Hex(
    identitySecret,
    `delivery-v2-legacy-auto-key\u0000${logicalKey}`,
  );
}

function validTimestamp(value: unknown): value is string {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}

function parseDevice(raw: string, expectV2: boolean): DeviceRecord | null {
  try {
    const value = JSON.parse(raw) as Partial<DeviceRecord>;
    const email = typeof value.email === "string" ? normalizeEmail(value.email) : "";
    if (
      !email ||
      !validTimestamp(value.createdAt) ||
      (expectV2 ? value.deliveryGeneration !== 2 : value.deliveryGeneration !== undefined) ||
      (value.protocolGeneration !== undefined &&
        (!Number.isSafeInteger(value.protocolGeneration) || value.protocolGeneration < 1)) ||
      (value.buildIdentity !== undefined &&
        !isBuildIdentity(value.buildIdentity)) ||
      (value.readyGeneration !== undefined &&
        (!Number.isSafeInteger(value.readyGeneration) || value.readyGeneration < 1))
    ) {
      return null;
    }
    return {
      email,
      createdAt: value.createdAt,
      ...(value.deliveryGeneration === 2 ? { deliveryGeneration: 2 as const } : {}),
      ...(value.protocolGeneration !== undefined
        ? { protocolGeneration: value.protocolGeneration }
        : {}),
      ...(value.buildIdentity !== undefined
        ? { buildIdentity: value.buildIdentity }
        : {}),
      ...(value.readyGeneration !== undefined
        ? { readyGeneration: value.readyGeneration }
        : {}),
    };
  } catch {
    return null;
  }
}

export function configuredBuildIdentity(env: Env): string | null {
  const value = env.BUILD_IDENTITY?.trim() ?? "";
  return /^email-relay-v2-[0-9a-f]{40}$/.test(value) ? value : null;
}

export function configuredIdentitySecret(env: Env): string | null {
  const value = env.IDENTITY_SECRET?.trim() ?? "";
  return value.length >= 16 && value.length <= 1024 ? value : null;
}

export function automaticRuntimeConfigured(env: Env): boolean {
  if (
    !env.DELIVER_GATE ||
    !env.STORE ||
    !configuredBuildIdentity(env) ||
    !env.TOKEN_SECRET?.trim() ||
    !configuredIdentitySecret(env) ||
    !env.RESEND_API_KEY?.trim() ||
    !isPlausibleEmail(env.FROM_EMAIL ?? "")
  ) return false;
  try {
    const publicUrl = new URL(env.PUBLIC_BASE_URL ?? "");
    return publicUrl.protocol === "https:" && Boolean(publicUrl.hostname);
  } catch {
    return false;
  }
}

function isBuildIdentity(value: unknown): value is string {
  return typeof value === "string" &&
    /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value);
}

function isHash(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function parsePending(raw: string): PendingVerify | null {
  try {
    const value = JSON.parse(raw) as Partial<PendingVerify>;
    const email = typeof value.email === "string" ? normalizeEmail(value.email) : "";
    if (
      !email ||
      typeof value.createdAt !== "string" ||
      typeof value.expiresAt !== "string" ||
      !Number.isFinite(Date.parse(value.createdAt)) ||
      !Number.isFinite(Date.parse(value.expiresAt))
    ) {
      return null;
    }
    return { email, createdAt: value.createdAt, expiresAt: value.expiresAt };
  } catch {
    return null;
  }
}
