import type {
  StorageAdapter,
  StorageNamespaceGuard,
} from "../core/adapters";
import { derivePaperInboxPaths } from "../services/paper-index";
import type { OutputSettings } from "../settings/types";
import {
  EMAIL_DELIVERY_CHANNEL,
  EMAIL_HOSTED_CHANNEL,
  type DeliveryPhase,
  type DeliveryRecord,
  type DeliveryStateFile,
  type EmailDeliveryChannel,
  type EmailDeliveryReason,
} from "./types";

export const DELIVERY_STATE_SCHEMA_VERSION = 1 as const;
export const DELIVERY_CLAIM_RECOVERY_GRACE_MS = 5 * 60 * 1000;

const CLAIM_SCHEMA_VERSION = 1 as const;
const STATE_REBUILD_ATTEMPTS = 8;
const KNOWN_CHANNELS = new Set<string>([
  EMAIL_DELIVERY_CHANNEL,
  EMAIL_HOSTED_CHANNEL,
]);
const KNOWN_PHASES = new Set<string>(["claimed", "delivered", "ambiguous"]);
const KNOWN_RESULT_ERROR_CODES = new Set<DeliveryResultErrorCode>([
  "cancelled_before_provider_attempt",
  "provider_definitive_rejection",
  "provider_outcome_ambiguous",
]);

export type DeliveryStateReadResult =
  | { kind: "missing"; state: DeliveryStateFile }
  | { kind: "valid"; state: DeliveryStateFile }
  | { kind: "corrupt"; reason: "state_parse_failed" | "state_invalid" }
  | { kind: "unreadable"; reason: "state_read_failed" };

export type DeliveryResultErrorCode =
  | "cancelled_before_provider_attempt"
  | "provider_definitive_rejection"
  | "provider_outcome_ambiguous";

export interface DeliveryClaimHandle {
  claimPath: string;
  decisionPath: string;
  resultPath: string;
  owner: string;
  generation: number;
  date: string;
  /** Raw recipient is retained in memory only so v1 main state stays compatible. */
  recipient?: string;
  recipientIdentity: string;
  channel: EmailDeliveryChannel;
  namespaceGuard?: StorageNamespaceGuard;
}

export type DeliveryClaimResult =
  | ({ kind: "claimed" } & DeliveryClaimHandle)
  | { kind: "blocked"; reason: EmailDeliveryReason }
  | { kind: "failed"; reason: EmailDeliveryReason };

type ClaimDecisionPhase =
  | "provider_attempt_started"
  | "recovered_before_provider_attempt";
type ClaimResultPhase =
  | "delivered"
  | "ambiguous"
  | "definitive_rejection"
  | "cancelled_before_provider_attempt";

interface DeliveryClaimDocument {
  schemaVersion: typeof CLAIM_SCHEMA_VERSION;
  phase: "claimed";
  owner: string;
  generation: number;
  date: string;
  recipientIdentity: string;
  channel: EmailDeliveryChannel;
  createdAt: string;
}

interface DeliveryDecisionDocument {
  schemaVersion: typeof CLAIM_SCHEMA_VERSION;
  phase: ClaimDecisionPhase;
  owner: string;
  createdAt: string;
  previousOwner?: string;
}

interface DeliveryResultDocument {
  schemaVersion: typeof CLAIM_SCHEMA_VERSION;
  phase: ClaimResultPhase;
  owner: string;
  createdAt: string;
  attempts: number;
  errorCode?: DeliveryResultErrorCode;
}

interface ClaimGeneration {
  stem: string;
  handle: DeliveryClaimHandle;
  claim: DeliveryClaimDocument;
  decision?: DeliveryDecisionDocument;
  result?: DeliveryResultDocument;
}

export function deliveryRecordKey(
  date: string,
  recipient: string,
  channel: EmailDeliveryChannel = EMAIL_DELIVERY_CHANNEL,
): string {
  return `${date}|${normalizeRecipient(recipient)}|${channel}`;
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

/** All delivered-compatible v1 records, including claims, block automatic send. */
export function shouldSendEmail(
  state: DeliveryStateFile,
  date: string,
  recipient: string,
  _channel?: EmailDeliveryChannel,
): boolean {
  const to = normalizeRecipient(recipient);
  if (!date || !to) return true;
  for (const record of Object.values(state.records)) {
    if (
      record.date === date &&
      normalizeRecipient(record.recipient) === to &&
      record.status === "delivered"
    ) {
      return false;
    }
  }
  return true;
}

export function markClaimed(
  state: DeliveryStateFile,
  input: {
    date: string;
    recipient: string;
    channel?: EmailDeliveryChannel;
    now?: Date;
  },
): DeliveryStateFile {
  return putRecord(state, input, {
    status: "delivered",
    deliveryPhase: "claimed",
    attempts: 0,
  });
}

export function markDelivered(
  state: DeliveryStateFile,
  input: {
    date: string;
    recipient: string;
    channel?: EmailDeliveryChannel;
    attempts: number;
    now?: Date;
  },
): DeliveryStateFile {
  return putRecord(state, input, {
    status: "delivered",
    deliveryPhase: "delivered",
    attempts: input.attempts,
  });
}

export function markAmbiguous(
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
  return putRecord(state, input, {
    status: "delivered",
    deliveryPhase: "ambiguous",
    attempts: input.attempts,
    lastError: input.lastError,
  });
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
  return putRecord(state, input, {
    status: "failed",
    attempts: input.attempts,
    lastError: input.lastError,
  });
}

export async function readDeliveryState(
  storage: StorageAdapter,
  output: OutputSettings,
): Promise<DeliveryStateReadResult> {
  const path = deliveryStatePath(output, (p) => storage.normalizePath(p));
  let exists: boolean;
  try {
    await storage.recoverTextAtomic?.(path, 0o600);
    exists = await storage.exists(path);
  } catch {
    return { kind: "unreadable", reason: "state_read_failed" };
  }
  if (!exists) return { kind: "missing", state: emptyDeliveryState() };

  let raw: string;
  try {
    raw = await storage.readText(path);
  } catch {
    return { kind: "unreadable", reason: "state_read_failed" };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { kind: "corrupt", reason: "state_parse_failed" };
  }
  try {
    return { kind: "valid", state: normalizeDeliveryState(parsed) };
  } catch {
    return { kind: "corrupt", reason: "state_invalid" };
  }
}

export async function loadDeliveryState(
  storage: StorageAdapter,
  output: OutputSettings,
): Promise<DeliveryStateFile> {
  const read = await readDeliveryState(storage, output);
  if (read.kind === "missing" || read.kind === "valid") return read.state;
  throw new Error(`delivery state ${read.kind}: ${read.reason}`);
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
  await writeAtomic(storage, path, `${JSON.stringify(next, null, 2)}\n`);
}

/** Strict v1 parser: invalid documents or records are never silently dropped. */
export function normalizeDeliveryState(raw: unknown): DeliveryStateFile {
  if (!isRecord(raw)) throw new Error("delivery state must be an object");
  if (raw.schemaVersion !== DELIVERY_STATE_SCHEMA_VERSION) {
    throw new Error("unsupported delivery state schemaVersion");
  }
  if (typeof raw.updatedAt !== "string" || !raw.updatedAt) {
    throw new Error("delivery state updatedAt is invalid");
  }
  if (!isRecord(raw.records)) throw new Error("delivery state records are invalid");

  const records: Record<string, DeliveryRecord> = {};
  for (const [key, value] of Object.entries(raw.records)) {
    const record = normalizeRecord(value);
    if (!record) throw new Error(`delivery state record ${key} is invalid`);
    records[key] = record;
  }
  return {
    schemaVersion: DELIVERY_STATE_SCHEMA_VERSION,
    updatedAt: raw.updatedAt,
    records,
  };
}

/**
 * Claims are immutable generations. A same-generation decision file is the
 * exclusive race between starting the provider attempt and recovering a stale
 * pre-attempt owner. There is deliberately no global lock to orphan.
 */
export async function claimAutomaticDelivery(
  storage: StorageAdapter,
  output: OutputSettings,
  input: {
    date: string;
    recipient: string;
    channel: EmailDeliveryChannel;
    now?: Date;
    recoveryGraceMs?: number;
    owner?: string;
  },
): Promise<DeliveryClaimResult> {
  if (
    !storage.createTextExclusive ||
    !storage.guardClaimNamespace ||
    !storage.list
  ) {
    return {
      kind: "failed",
      reason: "delivery_storage_unsupported",
    };
  }

  const initialRead = await readDeliveryState(storage, output);
  if (initialRead.kind === "corrupt" || initialRead.kind === "unreadable") {
    return {
      kind: "failed",
      reason: "delivery_state_unavailable",
    };
  }

  const now = input.now ?? new Date();
  const graceMs = Math.max(
    0,
    input.recoveryGraceMs ?? DELIVERY_CLAIM_RECOVERY_GRACE_MS,
  );
  const owner = input.owner ?? newOwner();
  const dir = claimDirectory(storage, output);
  const recipientHash = await recipientIdentity(input.recipient);
  const stem = await logicalClaimStem(input.date, recipientHash);

  try {
    await ensureDirDeep(storage, dir);
    for (let pass = 0; pass < STATE_REBUILD_ATTEMPTS; pass += 1) {
      const generations = await readLogicalGenerations(storage, dir, stem);
      const latest = generations.at(-1);
      let nextGeneration = 0;

      if (!latest) {
        if (!shouldSendEmail(initialRead.state, input.date, input.recipient)) {
          return { kind: "blocked", reason: "already_delivered" };
        }
      } else {
        const status = classifyGeneration(latest);
        if (status === "blocking") {
          return { kind: "blocked", reason: blockingReason(latest) };
        }
        if (status === "waiting") {
          const ageMs = now.getTime() - Date.parse(latest.claim.createdAt);
          if (!Number.isFinite(ageMs) || ageMs < graceMs) {
            return {
              kind: "blocked",
              reason: "delivery_claim_active",
            };
          }
          const recovered = await createDecision(storage, latest.handle, {
            schemaVersion: CLAIM_SCHEMA_VERSION,
            phase: "recovered_before_provider_attempt",
            owner,
            previousOwner: latest.claim.owner,
            createdAt: now.toISOString(),
          });
          if (!recovered) continue;
        }
        nextGeneration = latest.claim.generation + 1;
      }

      const handle = claimHandle(storage, dir, stem, {
        generation: nextGeneration,
        owner,
        date: input.date,
        recipient: input.recipient,
        recipientIdentity: recipientHash,
        channel: input.channel,
      });
      const document: DeliveryClaimDocument = {
        schemaVersion: CLAIM_SCHEMA_VERSION,
        phase: "claimed",
        owner,
        generation: nextGeneration,
        date: input.date,
        recipientIdentity: recipientHash,
        channel: input.channel,
        createdAt: now.toISOString(),
      };
      const created = await storage.createTextExclusive(
        handle.claimPath,
        serializeDocument(document),
      );
      if (!created) continue;

      let namespaceGuard: StorageNamespaceGuard | undefined;
      try {
        namespaceGuard = await storage.guardClaimNamespace(handle.claimPath);
        namespaceGuard.assertCurrent();
        await assertCurrentClaim(storage, handle);
        await rebuildDeliveryStateFromClaims(
          storage,
          output,
          now,
          input.recipient,
          handle,
        );
      } catch {
        const recovered = await (async () => {
          try {
            namespaceGuard?.assertCurrent();
            await assertCurrentClaim(storage, handle);
            return await createDecision(storage, handle, {
              schemaVersion: CLAIM_SCHEMA_VERSION,
              phase: "recovered_before_provider_attempt",
              owner,
              previousOwner: owner,
              createdAt: now.toISOString(),
            });
          } catch {
            return false;
          }
        })();
        await namespaceGuard?.release().catch(() => undefined);
        return {
          kind: "failed",
          reason: recovered
            ? "delivery_state_update_failed"
            : "delivery_claim_storage_failed",
        };
      }
      return { kind: "claimed", ...handle, namespaceGuard };
    }
    return { kind: "failed", reason: "delivery_claim_contention" };
  } catch {
    return { kind: "failed", reason: "delivery_claim_storage_failed" };
  }
}

/** Persist the blocking attempt decision before invoking a provider. */
export async function markAutomaticDeliveryAttemptStarted(
  storage: StorageAdapter,
  output: OutputSettings,
  handle: DeliveryClaimHandle,
  now: Date = new Date(),
): Promise<void> {
  await assertCurrentClaim(storage, handle);
  const decision: DeliveryDecisionDocument = {
    schemaVersion: CLAIM_SCHEMA_VERSION,
    phase: "provider_attempt_started",
    owner: handle.owner,
    createdAt: now.toISOString(),
  };
  const created = await createDecision(storage, handle, decision);
  if (!created) {
    const existing = await readOptionalDecision(storage, handle.decisionPath);
    if (
      existing?.phase !== "provider_attempt_started" ||
      existing.owner !== handle.owner
    ) {
      throw new Error("delivery claim was recovered before provider attempt");
    }
  }
  await rebuildDeliveryStateFromClaims(
    storage,
    output,
    now,
    handle.recipient,
    handle,
  );
}

/**
 * A durable pre-attempt release wins only when no provider-attempt decision was
 * already recorded. It creates no retry window by deletion.
 */
export async function releaseAutomaticDeliveryBeforeAttempt(
  storage: StorageAdapter,
  output: OutputSettings,
  handle: DeliveryClaimHandle,
  _reason: string,
  now: Date = new Date(),
): Promise<void> {
  const existing = await readOptionalDecision(storage, handle.decisionPath);
  if (!existing) {
    const recovered = await createDecision(storage, handle, {
      schemaVersion: CLAIM_SCHEMA_VERSION,
      phase: "recovered_before_provider_attempt",
      owner: handle.owner,
      previousOwner: handle.owner,
      createdAt: now.toISOString(),
    });
    if (!recovered) {
      throw new Error("delivery claim release lost the provider decision race");
    }
  } else if (existing.owner !== handle.owner) {
    throw new Error("another owner controls this delivery claim");
  }
  // The owning caller may have persisted provider_attempt_started and then
  // observed cancellation before the synchronous invocation boundary. Its
  // not-attempted result is durable evidence that makes this generation
  // releasable; callers must never use this API after onProviderInvocation.

  await createResult(storage, handle, {
    schemaVersion: CLAIM_SCHEMA_VERSION,
    phase: "cancelled_before_provider_attempt",
    owner: handle.owner,
    createdAt: now.toISOString(),
    attempts: 0,
    errorCode: "cancelled_before_provider_attempt",
  });
  await rebuildDeliveryStateFromClaims(storage, output, now, handle.recipient);
}

export async function finalizeAutomaticDelivery(
  storage: StorageAdapter,
  output: OutputSettings,
  input: DeliveryClaimHandle & {
    outcome: "delivered" | "ambiguous" | "failed";
    attempts: number;
    errorCode?: DeliveryResultErrorCode;
    lastError?: string;
    now?: Date;
  },
): Promise<void> {
  const now = input.now ?? new Date();
  const phase: ClaimResultPhase =
    input.outcome === "failed" ? "definitive_rejection" : input.outcome;
  await createResult(storage, input, {
    schemaVersion: CLAIM_SCHEMA_VERSION,
    phase,
    owner: input.owner,
    createdAt: now.toISOString(),
    attempts: Math.max(0, Math.floor(input.attempts)),
    errorCode:
      input.errorCode ??
      (input.outcome === "failed"
        ? "provider_definitive_rejection"
        : input.outcome === "ambiguous"
          ? "provider_outcome_ambiguous"
          : undefined),
  });
  await rebuildDeliveryStateFromClaims(storage, output, now, input.recipient);
}

function putRecord(
  state: DeliveryStateFile,
  input: {
    date: string;
    recipient: string;
    channel?: EmailDeliveryChannel;
    now?: Date;
  },
  value: {
    status: DeliveryRecord["status"];
    deliveryPhase?: DeliveryPhase;
    attempts: number;
    lastError?: string;
  },
): DeliveryStateFile {
  const now = input.now ?? new Date();
  const channel = input.channel ?? EMAIL_DELIVERY_CHANNEL;
  const key = deliveryRecordKey(input.date, input.recipient, channel);
  const record: DeliveryRecord = {
    date: input.date,
    recipient: input.recipient.trim(),
    channel,
    status: value.status,
    deliveryPhase: value.deliveryPhase,
    updatedAt: now.toISOString(),
    attempts: Math.max(0, Math.floor(value.attempts)),
    lastError: value.lastError,
  };
  return {
    schemaVersion: DELIVERY_STATE_SCHEMA_VERSION,
    updatedAt: now.toISOString(),
    records: { ...state.records, [key]: record },
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
  if (typeof raw.updatedAt !== "string" || !raw.updatedAt) return null;
  if (
    typeof raw.attempts !== "number" ||
    !Number.isFinite(raw.attempts) ||
    raw.attempts < 0
  ) {
    return null;
  }
  if (
    raw.deliveryPhase !== undefined &&
    (typeof raw.deliveryPhase !== "string" || !KNOWN_PHASES.has(raw.deliveryPhase))
  ) {
    return null;
  }
  if (raw.status === "failed" && raw.deliveryPhase !== undefined) return null;
  // Legacy v1 files may contain a provider id; accept a string for compatibility
  // but intentionally discard it from normalized/public state.
  if (
    raw.providerMessageId !== undefined &&
    typeof raw.providerMessageId !== "string"
  ) {
    return null;
  }
  if (raw.lastError !== undefined && typeof raw.lastError !== "string") return null;
  return {
    date: raw.date,
    recipient: raw.recipient.trim(),
    channel: raw.channel as EmailDeliveryChannel,
    status: raw.status,
    deliveryPhase: raw.deliveryPhase as DeliveryPhase | undefined,
    updatedAt: raw.updatedAt,
    attempts: Math.floor(raw.attempts),
    lastError: typeof raw.lastError === "string" ? raw.lastError : undefined,
  };
}

async function rebuildDeliveryStateFromClaims(
  storage: StorageAdapter,
  output: OutputSettings,
  now: Date,
  recipientHint?: string,
  expectedClaim?: DeliveryClaimHandle,
): Promise<void> {
  if (!storage.list) throw new Error("storage cannot enumerate delivery claims");
  const statePath = deliveryStatePath(output, (p) => storage.normalizePath(p));
  const dir = claimDirectory(storage, output);

  for (let attempt = 0; attempt < STATE_REBUILD_ATTEMPTS; attempt += 1) {
    const read = await readDeliveryState(storage, output);
    if (read.kind === "corrupt" || read.kind === "unreadable") {
      throw new Error(`delivery state ${read.kind}: ${read.reason}`);
    }
    const latest = await readLatestGenerations(storage, dir);
    if (expectedClaim) assertClaimGenerationVisible(latest, expectedClaim);
    const next = await stateWithClaimRecords(
      read.state,
      latest,
      now,
      recipientHint,
    );
    const parent = parentDir(statePath);
    if (parent) await ensureDirDeep(storage, parent);
    await writeAtomic(storage, statePath, `${JSON.stringify(next, null, 2)}\n`);

    const afterRead = await readDeliveryState(storage, output);
    if (afterRead.kind !== "valid") continue;
    const afterLatest = await readLatestGenerations(storage, dir);
    if (expectedClaim) assertClaimGenerationVisible(afterLatest, expectedClaim);
    if (await claimRecordsMatch(afterRead.state, afterLatest)) return;
  }
  throw new Error("delivery state did not converge with durable claims");
}

async function stateWithClaimRecords(
  state: DeliveryStateFile,
  latest: ClaimGeneration[],
  now: Date,
  recipientHint?: string,
): Promise<DeliveryStateFile> {
  let next: DeliveryStateFile = {
    ...state,
    updatedAt: now.toISOString(),
    records: { ...state.records },
  };
  const recipients = await recipientsByIdentity(next, recipientHint);
  for (const generation of latest) {
    const recipient = recipients.get(generation.claim.recipientIdentity);
    // A concurrent host may have durably claimed another recipient before its
    // compatible main-state record is visible. Preserve existing state and retry.
    if (!recipient) continue;
    for (const [key, record] of Object.entries(next.records)) {
      if (
        record.date === generation.claim.date &&
        normalizeRecipient(record.recipient) === normalizeRecipient(recipient)
      ) {
        delete next.records[key];
      }
    }
    next = putClaimRecord(next, generation, recipient);
  }
  return next;
}

function putClaimRecord(
  state: DeliveryStateFile,
  generation: ClaimGeneration,
  recipient: string,
): DeliveryStateFile {
  const common = {
    date: generation.claim.date,
    recipient,
    channel: generation.claim.channel,
  };
  const result = generation.result;
  if (result?.phase === "delivered") {
    return markDelivered(state, {
      ...common,
      attempts: result.attempts,
      now: new Date(result.createdAt),
    });
  }
  if (result?.phase === "ambiguous") {
    return markAmbiguous(state, {
      ...common,
      attempts: result.attempts,
      lastError: result.errorCode ?? "provider_outcome_ambiguous",
      now: new Date(result.createdAt),
    });
  }
  if (result?.phase === "definitive_rejection") {
    return markAmbiguous(state, {
      ...common,
      attempts: result.attempts,
      lastError: result.errorCode ?? "provider_definitive_rejection",
      now: new Date(result.createdAt),
    });
  }
  if (
    result?.phase === "cancelled_before_provider_attempt" ||
    generation.decision?.phase === "recovered_before_provider_attempt"
  ) {
    return markFailed(state, {
      ...common,
      attempts: result?.attempts ?? 0,
      lastError: result?.errorCode ?? "cancelled_before_provider_attempt",
      now: new Date(result?.createdAt ?? generation.decision!.createdAt),
    });
  }
  if (generation.decision?.phase === "provider_attempt_started") {
    return markAmbiguous(state, {
      ...common,
      attempts: 0,
      lastError: "provider attempt started; outcome not yet recorded",
      now: new Date(generation.decision.createdAt),
    });
  }
  return markClaimed(state, {
    ...common,
    now: new Date(generation.claim.createdAt),
  });
}

async function claimRecordsMatch(
  state: DeliveryStateFile,
  latest: ClaimGeneration[],
): Promise<boolean> {
  const recipients = await recipientsByIdentity(state);
  for (const generation of latest) {
    const recipient = recipients.get(generation.claim.recipientIdentity);
    if (!recipient) return false;
    const matches = Object.values(state.records).filter(
      (record) =>
        record.date === generation.claim.date &&
        normalizeRecipient(record.recipient) === normalizeRecipient(recipient),
    );
    if (matches.length !== 1) return false;
    const expected = putClaimRecord(
      emptyDeliveryState(),
      generation,
      recipient,
    );
    const expectedRecord = Object.values(expected.records)[0];
    if (!expectedRecord || JSON.stringify(matches[0]) !== JSON.stringify(expectedRecord)) {
      return false;
    }
  }
  return true;
}

async function recipientsByIdentity(
  state: DeliveryStateFile,
  recipientHint?: string,
): Promise<Map<string, string>> {
  const recipients = new Map<string, string>();
  for (const record of Object.values(state.records)) {
    recipients.set(await recipientIdentity(record.recipient), record.recipient);
  }
  if (recipientHint?.trim()) {
    recipients.set(await recipientIdentity(recipientHint), recipientHint.trim());
  }
  return recipients;
}

async function assertCurrentClaim(
  storage: StorageAdapter,
  handle: DeliveryClaimHandle,
): Promise<void> {
  const claim = normalizeClaimDocument(
    await readJsonDocument(storage, handle.claimPath),
  );
  if (
    claim.owner !== handle.owner ||
    claim.generation !== handle.generation ||
    claim.date !== handle.date ||
    claim.recipientIdentity !== handle.recipientIdentity ||
    claim.channel !== handle.channel
  ) {
    throw new Error("delivery claim is not visible in the current storage world");
  }
}

function assertClaimGenerationVisible(
  latest: ClaimGeneration[],
  handle: DeliveryClaimHandle,
): void {
  const visible = latest.some(
    (generation) =>
      generation.claim.owner === handle.owner &&
      generation.claim.generation === handle.generation &&
      generation.claim.date === handle.date &&
      generation.claim.recipientIdentity === handle.recipientIdentity &&
      generation.claim.channel === handle.channel,
  );
  if (!visible) {
    throw new Error("delivery claim disappeared from the current storage world");
  }
}

async function readLatestGenerations(
  storage: StorageAdapter,
  dir: string,
): Promise<ClaimGeneration[]> {
  if (!storage.list) throw new Error("storage cannot enumerate delivery claims");
  const entries = await storage.list(dir);
  const latestByStem = new Map<string, { stem: string; generation: number }>();
  for (const entry of entries) {
    if (entry.type !== "file") continue;
    const name = baseName(entry.path);
    const match = /^([0-9a-f]{64})\.g(\d+)\.claim\.json$/.exec(name);
    if (!match?.[1] || !match[2]) continue;
    const generation = Number(match[2]);
    if (!Number.isSafeInteger(generation) || generation < 0) continue;
    const current = latestByStem.get(match[1]);
    if (!current || generation > current.generation) {
      latestByStem.set(match[1], { stem: match[1], generation });
    }
  }
  const latest: ClaimGeneration[] = [];
  for (const item of latestByStem.values()) {
    latest.push(await readGeneration(storage, dir, item.stem, item.generation));
  }
  return latest;
}

async function readLogicalGenerations(
  storage: StorageAdapter,
  dir: string,
  stem: string,
): Promise<ClaimGeneration[]> {
  if (!storage.list) throw new Error("storage cannot enumerate delivery claims");
  const entries = await storage.list(dir);
  const generations: number[] = [];
  const pattern = new RegExp(`^${stem}\\.g(\\d+)\\.claim\\.json$`);
  for (const entry of entries) {
    if (entry.type !== "file") continue;
    const match = pattern.exec(baseName(entry.path));
    if (!match?.[1]) continue;
    const generation = Number(match[1]);
    if (Number.isSafeInteger(generation) && generation >= 0) generations.push(generation);
  }
  generations.sort((a, b) => a - b);
  return Promise.all(
    generations.map((generation) => readGeneration(storage, dir, stem, generation)),
  );
}

async function readGeneration(
  storage: StorageAdapter,
  dir: string,
  stem: string,
  generation: number,
): Promise<ClaimGeneration> {
  const claimPath = storage.normalizePath(`${dir}/${stem}.g${generation}.claim.json`);
  const claim = normalizeClaimDocument(await readJsonDocument(storage, claimPath));
  if (claim.generation !== generation) {
    throw new Error(`delivery claim generation mismatch: ${claimPath}`);
  }
  const handle = claimHandle(storage, dir, stem, claim);
  return {
    stem,
    handle,
    claim,
    decision: await readOptionalDecision(storage, handle.decisionPath),
    result: await readOptionalResult(storage, handle.resultPath),
  };
}

function classifyGeneration(
  generation: ClaimGeneration,
): "released" | "blocking" | "waiting" {
  if (
    generation.result?.phase === "cancelled_before_provider_attempt" ||
    generation.decision?.phase === "recovered_before_provider_attempt"
  ) {
    return "released";
  }
  if (
    generation.result?.phase === "delivered" ||
    generation.result?.phase === "ambiguous" ||
    generation.result?.phase === "definitive_rejection" ||
    generation.decision?.phase === "provider_attempt_started"
  ) {
    return "blocking";
  }
  return "waiting";
}

function blockingReason(generation: ClaimGeneration): EmailDeliveryReason {
  if (generation.result?.phase === "delivered") return "already_delivered";
  if (generation.result?.phase === "ambiguous") {
    return "provider_outcome_ambiguous";
  }
  if (generation.result?.phase === "definitive_rejection") {
    return "provider_definitive_rejection";
  }
  return "provider_attempt_started";
}

function claimDirectory(storage: StorageAdapter, output: OutputSettings): string {
  const statePath = deliveryStatePath(output, (p) => storage.normalizePath(p));
  return storage.normalizePath(`${statePath}.claims`);
}

function claimHandle(
  storage: StorageAdapter,
  dir: string,
  stem: string,
  input: {
    generation: number;
    owner: string;
    date: string;
    recipient?: string;
    recipientIdentity: string;
    channel: EmailDeliveryChannel;
  },
): DeliveryClaimHandle {
  const prefix = storage.normalizePath(`${dir}/${stem}.g${input.generation}`);
  return {
    claimPath: `${prefix}.claim.json`,
    decisionPath: `${prefix}.decision.json`,
    resultPath: `${prefix}.result.json`,
    owner: input.owner,
    generation: input.generation,
    date: input.date,
    recipient: input.recipient?.trim(),
    recipientIdentity: input.recipientIdentity,
    channel: input.channel,
  };
}

async function createDecision(
  storage: StorageAdapter,
  handle: DeliveryClaimHandle,
  document: DeliveryDecisionDocument,
): Promise<boolean> {
  if (!storage.createTextExclusive) {
    throw new Error("storage does not support system-wide exclusive create");
  }
  return storage.createTextExclusive(
    handle.decisionPath,
    serializeDocument(document),
  );
}

async function createResult(
  storage: StorageAdapter,
  handle: DeliveryClaimHandle,
  document: DeliveryResultDocument,
): Promise<void> {
  if (!storage.createTextExclusive) {
    throw new Error("storage does not support system-wide exclusive create");
  }
  const created = await storage.createTextExclusive(
    handle.resultPath,
    serializeDocument(document),
  );
  if (created) return;
  const existing = await readOptionalResult(storage, handle.resultPath);
  if (!existing || JSON.stringify(existing) !== JSON.stringify(document)) {
    throw new Error("delivery claim result already exists with a different outcome");
  }
}

async function readOptionalDecision(
  storage: StorageAdapter,
  path: string,
): Promise<DeliveryDecisionDocument | undefined> {
  if (!(await storage.exists(path))) return undefined;
  return normalizeDecisionDocument(await readJsonDocument(storage, path));
}

async function readOptionalResult(
  storage: StorageAdapter,
  path: string,
): Promise<DeliveryResultDocument | undefined> {
  if (!(await storage.exists(path))) return undefined;
  return normalizeResultDocument(await readJsonDocument(storage, path));
}

async function readJsonDocument(
  storage: StorageAdapter,
  path: string,
): Promise<unknown> {
  const raw = await storage.readText(path);
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error(`invalid delivery claim document: ${baseName(path)}`);
  }
}

function normalizeClaimDocument(raw: unknown): DeliveryClaimDocument {
  if (!isRecord(raw) || raw.schemaVersion !== CLAIM_SCHEMA_VERSION) {
    throw new Error("delivery claim schema is invalid");
  }
  if (raw.phase !== "claimed" || typeof raw.owner !== "string" || !raw.owner) {
    throw new Error("delivery claim owner or phase is invalid");
  }
  if (!Number.isSafeInteger(raw.generation) || (raw.generation as number) < 0) {
    throw new Error("delivery claim generation is invalid");
  }
  if (typeof raw.date !== "string" || !raw.date) {
    throw new Error("delivery claim date is invalid");
  }
  if (
    typeof raw.recipientIdentity !== "string" ||
    !/^[0-9a-f]{64}$/.test(raw.recipientIdentity)
  ) {
    throw new Error("delivery claim recipient identity is invalid");
  }
  if (typeof raw.channel !== "string" || !KNOWN_CHANNELS.has(raw.channel)) {
    throw new Error("delivery claim channel is invalid");
  }
  assertTimestamp(raw.createdAt, "delivery claim createdAt");
  return {
    schemaVersion: CLAIM_SCHEMA_VERSION,
    phase: "claimed",
    owner: raw.owner,
    generation: raw.generation as number,
    date: raw.date,
    recipientIdentity: raw.recipientIdentity,
    channel: raw.channel as EmailDeliveryChannel,
    createdAt: raw.createdAt as string,
  };
}

function normalizeDecisionDocument(raw: unknown): DeliveryDecisionDocument {
  if (!isRecord(raw) || raw.schemaVersion !== CLAIM_SCHEMA_VERSION) {
    throw new Error("delivery decision schema is invalid");
  }
  if (
    raw.phase !== "provider_attempt_started" &&
    raw.phase !== "recovered_before_provider_attempt"
  ) {
    throw new Error("delivery decision phase is invalid");
  }
  if (typeof raw.owner !== "string" || !raw.owner) {
    throw new Error("delivery decision owner is invalid");
  }
  if (raw.previousOwner !== undefined && typeof raw.previousOwner !== "string") {
    throw new Error("delivery decision previousOwner is invalid");
  }
  assertTimestamp(raw.createdAt, "delivery decision createdAt");
  return {
    schemaVersion: CLAIM_SCHEMA_VERSION,
    phase: raw.phase,
    owner: raw.owner,
    createdAt: raw.createdAt as string,
    previousOwner:
      typeof raw.previousOwner === "string" ? raw.previousOwner : undefined,
  };
}

function normalizeResultDocument(raw: unknown): DeliveryResultDocument {
  if (!isRecord(raw) || raw.schemaVersion !== CLAIM_SCHEMA_VERSION) {
    throw new Error("delivery result schema is invalid");
  }
  if (
    raw.phase !== "delivered" &&
    raw.phase !== "ambiguous" &&
    raw.phase !== "definitive_rejection" &&
    raw.phase !== "cancelled_before_provider_attempt"
  ) {
    throw new Error("delivery result phase is invalid");
  }
  if (typeof raw.owner !== "string" || !raw.owner) {
    throw new Error("delivery result owner is invalid");
  }
  if (
    typeof raw.attempts !== "number" ||
    !Number.isFinite(raw.attempts) ||
    raw.attempts < 0
  ) {
    throw new Error("delivery result attempts is invalid");
  }
  if (raw.providerMessageId !== undefined) {
    throw new Error("delivery result contains unsafe providerMessageId");
  }
  if (
    raw.errorCode !== undefined &&
    (typeof raw.errorCode !== "string" ||
      !KNOWN_RESULT_ERROR_CODES.has(raw.errorCode as DeliveryResultErrorCode))
  ) {
    throw new Error("delivery result errorCode is invalid");
  }
  if (raw.lastError !== undefined) {
    throw new Error("delivery result contains unsafe lastError");
  }
  assertTimestamp(raw.createdAt, "delivery result createdAt");
  return {
    schemaVersion: CLAIM_SCHEMA_VERSION,
    phase: raw.phase,
    owner: raw.owner,
    createdAt: raw.createdAt as string,
    attempts: Math.floor(raw.attempts),
    errorCode: raw.errorCode as DeliveryResultErrorCode | undefined,
  };
}

function assertTimestamp(value: unknown, name: string): void {
  if (
    typeof value !== "string" ||
    !value ||
    !Number.isFinite(Date.parse(value))
  ) {
    throw new Error(`${name} is invalid`);
  }
}

async function logicalClaimStem(
  date: string,
  recipientHash: string,
): Promise<string> {
  return sha256Hex(`${date}\u0000${recipientHash}`);
}

async function recipientIdentity(recipient: string): Promise<string> {
  return sha256Hex(normalizeRecipient(recipient));
}

async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

async function writeAtomic(
  storage: StorageAdapter,
  path: string,
  content: string,
): Promise<void> {
  if (!storage.writeTextAtomic) {
    throw new Error("private atomic storage is unavailable on this host");
  }
  await storage.writeTextAtomic(path, content, 0o600);
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  const normalized = storage.normalizePath(dir);
  if (!normalized || normalized === ".") return;
  const parts = normalized.split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (await storage.exists(current)) continue;
    try {
      await storage.mkdir(current);
    } catch (error) {
      if (!(await storage.exists(current))) throw error;
    }
  }
}

function serializeDocument(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function parentDir(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx <= 0 ? "" : path.slice(0, idx);
}

function baseName(path: string): string {
  const normalized = path.replace(/\\/g, "/");
  return normalized.slice(normalized.lastIndexOf("/") + 1);
}

function normalizeRecipient(recipient: string): string {
  return recipient.trim().toLowerCase();
}

function newOwner(): string {
  return crypto.randomUUID();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
