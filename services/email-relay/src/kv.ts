import { normalizeEmail, sha256Hex } from "./crypto";

export interface Env {
  STORE: KVNamespace;
  /** One Durable Object per authenticated device plus one cutover-proof singleton. */
  DELIVER_GATE?: DurableObjectNamespace;
  /** Operator-only bearer secret for staging the single-version cutover proof. */
  DELIVERY_V2_CUTOVER_TOKEN?: string;
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
  /** Present only on identities issued by the delivery-v2 Worker. */
  deliveryGeneration?: 2;
}

export interface AuthenticatedDevice {
  /** Secret-scoped hash of the raw token. Safe for routing and internal binding. */
  identity: string;
  /** Normalized verified recipient; never used in DO names or provider keys. */
  email: string;
  recipientIdentity: string;
  createdAt: string;
  deliveryGeneration?: 2;
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
const DELIVERY_V2_CUTOVER_KEY = "cutover:delivery-v2";
export const DELIVERY_V2_KV_VISIBILITY_MS = 60 * 1000;
export const DELIVERY_V2_LEGACY_PENDING_TTL_MS = 120 * 1000;
const LEGACY_AUTO_IDEMP_PREFIX = "idemp:arxiv-daily:auto:";

export type LegacyDeliveryEvidence = "none" | "done" | "attempted";

export interface DeliveryV2CutoverMarker {
  schemaVersion: 2;
  oldWorkerWritesQuiesced: true;
  legacyAutoEvidenceSnapshot: "positive-evidence-only";
  preQuiesceScanStartedAt: string;
  preQuiesceScanCompletedAt: string;
  oldWorkerWritesQuiescedAt: string;
  postQuiesceScanStartedAt: string;
  postQuiesceScanCompletedAt: string;
  enabledAt: string;
  /** Secret-scoped hashes of legacy automatic date + recipient identities only. */
  legacyAutoEvidence: Record<string, "done" | "attempted">;
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
  secret: string,
): Promise<string> {
  return sha256Hex(`${secret}:recipient:${normalizeEmail(normalizedEmail)}`);
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
  const parsed = parsePending(raw);
  if (!parsed || Date.parse(parsed.expiresAt) <= Date.now()) return null;
  return parsed;
}

export async function putDevice(
  env: Env,
  rawToken: string,
  email: string,
  now: Date = new Date(),
): Promise<void> {
  const hash = await hashDeviceToken(rawToken, env.TOKEN_SECRET);
  const value: DeviceRecord = {
    email: normalizeEmail(email),
    createdAt: now.toISOString(),
    deliveryGeneration: 2,
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
    recipientIdentity: await hashRecipientIdentity(device.email, env.TOKEN_SECRET),
    createdAt: device.createdAt,
    deliveryGeneration: device.deliveryGeneration,
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

/**
 * Parses the operator's single-version cutover declaration and positive legacy
 * evidence. Empty scans never prove absence: the Durable Object adds its own
 * observations, time barrier, and v2 device-provenance boundary before auto is
 * enabled. Mixed-version operation and rollback are intentionally unsupported.
 */
export async function assertDeliveryV2CutoverReady(
  env: Env,
  now: Date = new Date(),
): Promise<DeliveryV2CutoverMarker> {
  const raw = await env.STORE.get(DELIVERY_V2_CUTOVER_KEY);
  if (!raw) throw new Error("delivery-v2 cutover marker is missing");
  return parseDeliveryV2CutoverMarker(raw, now);
}

export function parseDeliveryV2CutoverMarker(
  raw: string,
  now: Date = new Date(),
): DeliveryV2CutoverMarker {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("delivery-v2 cutover marker is invalid");
  }
  if (!parsed || typeof parsed !== "object") {
    throw new Error("delivery-v2 cutover marker is invalid");
  }
  const value = parsed as Partial<DeliveryV2CutoverMarker>;
  if (
    value.schemaVersion !== 2 ||
    value.oldWorkerWritesQuiesced !== true ||
    value.legacyAutoEvidenceSnapshot !== "positive-evidence-only" ||
    !validTimestamp(value.preQuiesceScanStartedAt) ||
    !validTimestamp(value.preQuiesceScanCompletedAt) ||
    !validTimestamp(value.oldWorkerWritesQuiescedAt) ||
    !validTimestamp(value.postQuiesceScanStartedAt) ||
    !validTimestamp(value.postQuiesceScanCompletedAt) ||
    !validTimestamp(value.enabledAt) ||
    !value.legacyAutoEvidence ||
    typeof value.legacyAutoEvidence !== "object" ||
    Array.isArray(value.legacyAutoEvidence)
  ) {
    throw new Error("delivery-v2 cutover marker is invalid");
  }

  const preStartedAt = Date.parse(value.preQuiesceScanStartedAt);
  const preCompletedAt = Date.parse(value.preQuiesceScanCompletedAt);
  const quiescedAt = Date.parse(value.oldWorkerWritesQuiescedAt);
  const postStartedAt = Date.parse(value.postQuiesceScanStartedAt);
  const postCompletedAt = Date.parse(value.postQuiesceScanCompletedAt);
  const enabledAt = Date.parse(value.enabledAt);
  if (
    preCompletedAt < preStartedAt ||
    preCompletedAt > quiescedAt ||
    postStartedAt < quiescedAt + DELIVERY_V2_KV_VISIBILITY_MS ||
    postCompletedAt < postStartedAt ||
    postCompletedAt >= preStartedAt + DELIVERY_V2_LEGACY_PENDING_TTL_MS ||
    enabledAt < postCompletedAt ||
    enabledAt < quiescedAt + DELIVERY_V2_LEGACY_PENDING_TTL_MS ||
    now.getTime() < enabledAt + DELIVERY_V2_KV_VISIBILITY_MS
  ) {
    throw new Error("delivery-v2 cutover timing proof is invalid");
  }

  const legacyAutoEvidence: Record<string, "done" | "attempted"> = {};
  for (const [identity, evidence] of Object.entries(value.legacyAutoEvidence)) {
    if (
      !/^[0-9a-f]{64}$/.test(identity) ||
      (evidence !== "done" && evidence !== "attempted")
    ) {
      throw new Error("delivery-v2 legacy automatic evidence snapshot is invalid");
    }
    legacyAutoEvidence[identity] = evidence;
  }
  return { ...value, legacyAutoEvidence } as DeliveryV2CutoverMarker;
}

export async function hashDeliveryV2CutoverProof(
  marker: DeliveryV2CutoverMarker,
): Promise<string> {
  return sha256Hex(JSON.stringify([
    marker.schemaVersion,
    marker.oldWorkerWritesQuiesced,
    marker.legacyAutoEvidenceSnapshot,
    marker.preQuiesceScanStartedAt,
    marker.preQuiesceScanCompletedAt,
    marker.oldWorkerWritesQuiescedAt,
    marker.postQuiesceScanStartedAt,
    marker.postQuiesceScanCompletedAt,
    marker.enabledAt,
    Object.entries(marker.legacyAutoEvidence).sort(([left], [right]) =>
      left.localeCompare(right)
    ),
  ]));
}

/** Reads only the staged automatic snapshot; no expiring legacy KV key is trusted. */
export async function scanLegacyAutoDeliveryEvidence(
  env: Env,
): Promise<Record<string, "done" | "attempted">> {
  const evidence: Record<string, "done" | "attempted"> = {};
  let cursor: string | undefined;
  do {
    const page = await env.STORE.list({
      prefix: LEGACY_AUTO_IDEMP_PREFIX,
      ...(cursor ? { cursor } : {}),
    });
    if (!page || !Array.isArray(page.keys)) {
      throw new Error("legacy automatic delivery scan is invalid");
    }
    for (const entry of page.keys) {
      const key = entry?.name ?? "";
      const logicalKey = key.startsWith("idemp:") ? key.slice("idemp:".length) : "";
      if (!/^arxiv-daily:auto:[0-9a-f]{64}$/.test(logicalKey)) {
        throw new Error("legacy automatic delivery key is invalid");
      }
      const identity = await sha256Hex(`${env.TOKEN_SECRET}:legacy-auto-key:${logicalKey}`);
      const raw = await env.STORE.get(key);
      const observed = raw?.startsWith("pending:") ? "attempted" : raw ? "done" : "attempted";
      evidence[identity] = evidence[identity] === "attempted" || observed === "attempted"
        ? "attempted"
        : "done";
    }
    if (page.list_complete) break;
    if (!page.cursor || page.cursor === cursor) {
      throw new Error("legacy automatic delivery scan did not advance");
    }
    cursor = page.cursor;
  } while (true);
  return evidence;
}

export function assertLegacyAutoSnapshotCovers(
  snapshot: Record<string, "done" | "attempted">,
  observed: Record<string, "done" | "attempted">,
): void {
  for (const [identity, status] of Object.entries(observed)) {
    const imported = snapshot[identity];
    if (!imported || (status === "attempted" && imported !== "attempted")) {
      throw new Error("legacy automatic evidence is missing from the snapshot");
    }
  }
}

export async function hashLegacyAutoDeliveryIdentity(
  secret: string,
  date: string,
  normalizedRecipient: string,
): Promise<string> {
  const legacyAutoKey = `arxiv-daily:auto:${await sha256Hex(
    `${date}\u0000${normalizeEmail(normalizedRecipient)}`,
  )}`;
  return sha256Hex(`${secret}:legacy-auto-key:${legacyAutoKey}`);
}

export async function readLegacyDeliveryEvidence(
  env: Env,
  cutover: DeliveryV2CutoverMarker,
  date: string,
  normalizedRecipient: string,
): Promise<LegacyDeliveryEvidence> {
  const identity = await hashLegacyAutoDeliveryIdentity(
    env.TOKEN_SECRET,
    date,
    normalizedRecipient,
  );
  return cutover.legacyAutoEvidence[identity] ?? "none";
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
      (expectV2 ? value.deliveryGeneration !== 2 : value.deliveryGeneration !== undefined)
    ) {
      return null;
    }
    return {
      email,
      createdAt: value.createdAt,
      ...(value.deliveryGeneration === 2 ? { deliveryGeneration: 2 as const } : {}),
    };
  } catch {
    return null;
  }
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
