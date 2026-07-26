import type { StorageAdapter } from "../core/adapters";
import { derivePaperInboxPaths } from "../services/paper-index";
import type { OutputSettings } from "../settings/types";
import {
  EMAIL_DELIVERY_CHANNEL,
  EMAIL_HOSTED_CHANNEL,
  type DeliveryRecord,
  type DeliveryStateFile,
  type EmailDeliveryChannel,
} from "./types";

export const DELIVERY_STATE_SCHEMA_VERSION = 1 as const;

const KNOWN_CHANNELS = new Set<string>([
  EMAIL_DELIVERY_CHANNEL,
  EMAIL_HOSTED_CHANNEL,
]);

export function deliveryRecordKey(
  date: string,
  recipient: string,
  channel: EmailDeliveryChannel = EMAIL_DELIVERY_CHANNEL,
): string {
  return `${date}|${recipient.trim().toLowerCase()}|${channel}`;
}

export function deliveryStatePath(
  output: OutputSettings,
  normalizePath: (path: string) => string = (p) => p,
): string {
  const { indexDir } = derivePaperInboxPaths(output, normalizePath);
  return normalizePath(`${indexDir}/delivery-state.json`);
}

export function emptyDeliveryState(now: Date = new Date()): DeliveryStateFile {
  return {
    schemaVersion: DELIVERY_STATE_SCHEMA_VERSION,
    updatedAt: now.toISOString(),
    records: {},
  };
}

/**
 * Auto-send gate: skip if this date+recipient already has any successful
 * delivery (cross-mode: self Resend vs hosted).
 */
export function shouldSendEmail(
  state: DeliveryStateFile,
  date: string,
  recipient: string,
  _channel?: EmailDeliveryChannel,
): boolean {
  const to = recipient.trim().toLowerCase();
  if (!date || !to) return true;
  for (const record of Object.values(state.records)) {
    if (
      record.date === date &&
      record.recipient.trim().toLowerCase() === to &&
      record.status === "delivered"
    ) {
      return false;
    }
  }
  return true;
}

export function markDelivered(
  state: DeliveryStateFile,
  input: {
    date: string;
    recipient: string;
    channel?: EmailDeliveryChannel;
    attempts: number;
    providerMessageId?: string;
    now?: Date;
  },
): DeliveryStateFile {
  const now = input.now ?? new Date();
  const channel = input.channel ?? EMAIL_DELIVERY_CHANNEL;
  const key = deliveryRecordKey(input.date, input.recipient, channel);
  const record: DeliveryRecord = {
    date: input.date,
    recipient: input.recipient.trim(),
    channel,
    status: "delivered",
    updatedAt: now.toISOString(),
    attempts: input.attempts,
    providerMessageId: input.providerMessageId,
  };
  return {
    schemaVersion: DELIVERY_STATE_SCHEMA_VERSION,
    updatedAt: now.toISOString(),
    records: { ...state.records, [key]: record },
  };
}

export function markFailed(
  state: DeliveryStateFile,
  input: {
    date: string;
    recipient: string;
    channel?: EmailDeliveryChannel;
    attempts: number;
    lastError?: string;
    now?: Date;
  },
): DeliveryStateFile {
  const now = input.now ?? new Date();
  const channel = input.channel ?? EMAIL_DELIVERY_CHANNEL;
  const key = deliveryRecordKey(input.date, input.recipient, channel);
  const record: DeliveryRecord = {
    date: input.date,
    recipient: input.recipient.trim(),
    channel,
    status: "failed",
    updatedAt: now.toISOString(),
    attempts: input.attempts,
    lastError: input.lastError,
  };
  return {
    schemaVersion: DELIVERY_STATE_SCHEMA_VERSION,
    updatedAt: now.toISOString(),
    records: { ...state.records, [key]: record },
  };
}

export async function loadDeliveryState(
  storage: StorageAdapter,
  output: OutputSettings,
): Promise<DeliveryStateFile> {
  const path = deliveryStatePath(output, (p) => storage.normalizePath(p));
  if (!(await storage.exists(path))) {
    return emptyDeliveryState();
  }
  try {
    const raw = await storage.readText(path);
    return normalizeDeliveryState(JSON.parse(raw));
  } catch {
    return emptyDeliveryState();
  }
}

export async function saveDeliveryState(
  storage: StorageAdapter,
  output: OutputSettings,
  state: DeliveryStateFile,
  now: Date = new Date(),
): Promise<void> {
  const path = deliveryStatePath(output, (p) => storage.normalizePath(p));
  const dir = parentDir(path);
  if (dir) await ensureDirDeep(storage, dir);
  const next: DeliveryStateFile = {
    ...state,
    schemaVersion: DELIVERY_STATE_SCHEMA_VERSION,
    updatedAt: now.toISOString(),
  };
  const content = `${JSON.stringify(next, null, 2)}\n`;
  await writeAtomic(storage, path, content);
}

export function normalizeDeliveryState(raw: unknown): DeliveryStateFile {
  if (!isRecord(raw)) return emptyDeliveryState();
  const records: Record<string, DeliveryRecord> = {};
  const rawRecords = isRecord(raw.records) ? raw.records : {};
  for (const [key, value] of Object.entries(rawRecords)) {
    const record = normalizeRecord(value);
    if (record) records[key] = record;
  }
  return {
    schemaVersion: DELIVERY_STATE_SCHEMA_VERSION,
    updatedAt:
      typeof raw.updatedAt === "string" && raw.updatedAt
        ? raw.updatedAt
        : new Date().toISOString(),
    records,
  };
}

function normalizeRecord(raw: unknown): DeliveryRecord | null {
  if (!isRecord(raw)) return null;
  if (typeof raw.date !== "string" || !raw.date) return null;
  if (typeof raw.recipient !== "string" || !raw.recipient.trim()) return null;
  if (typeof raw.channel !== "string" || !KNOWN_CHANNELS.has(raw.channel)) {
    return null;
  }
  if (raw.status !== "delivered" && raw.status !== "failed") return null;
  return {
    date: raw.date,
    recipient: raw.recipient.trim(),
    channel: raw.channel as EmailDeliveryChannel,
    status: raw.status,
    updatedAt:
      typeof raw.updatedAt === "string" && raw.updatedAt
        ? raw.updatedAt
        : new Date().toISOString(),
    attempts:
      typeof raw.attempts === "number" && Number.isFinite(raw.attempts)
        ? Math.max(0, Math.floor(raw.attempts))
        : 0,
    providerMessageId:
      typeof raw.providerMessageId === "string"
        ? raw.providerMessageId
        : undefined,
    lastError: typeof raw.lastError === "string" ? raw.lastError : undefined,
  };
}

async function writeAtomic(
  storage: StorageAdapter,
  path: string,
  content: string,
): Promise<void> {
  if (storage.writeTextAtomic) {
    await storage.writeTextAtomic(path, content);
    return;
  }
  const tmp = `${path}.tmp`;
  const bak = `${path}.bak`;
  await storage.writeText(tmp, content);
  if (!(await storage.exists(path))) {
    await storage.rename(tmp, path);
    return;
  }
  if (await storage.exists(bak)) {
    await storage.remove(bak);
  }
  await storage.rename(path, bak);
  try {
    await storage.rename(tmp, path);
    await storage.remove(bak);
  } catch (e) {
    if (await storage.exists(bak)) {
      await storage.rename(bak, path);
    }
    throw e;
  }
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  const normalized = storage.normalizePath(dir);
  if (!normalized || normalized === ".") return;
  const parts = normalized.split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) {
      await storage.mkdir(current);
    }
  }
}

function parentDir(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx <= 0 ? "" : path.slice(0, idx);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
