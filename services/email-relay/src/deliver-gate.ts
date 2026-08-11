import { sha256Hex, utcDateKey } from "./crypto";
import {
  validateDeliverRequest,
  type DeliverBody,
  type DeliveryKeyKind,
  type ValidatedDeliverRequest,
} from "./deliver-logic";
import {
  assertDeliveryV2CutoverReady,
  assertLegacyAutoSnapshotCovers,
  dailyQuotaLimit,
  DELIVERY_V2_KV_VISIBILITY_MS,
  DELIVERY_V2_LEGACY_PENDING_TTL_MS,
  hashDeliveryV2CutoverProof,
  parseDeliveryV2CutoverMarker,
  readLegacyDeliveryEvidence,
  scanLegacyAutoDeliveryEvidence,
  type AuthenticatedDevice,
  type DeliveryV2CutoverMarker,
  type Env,
  type LegacyDeliveryEvidence,
} from "./kv";
import {
  ResendProviderError,
  ResendTransportError,
  sendResendEmail,
} from "./resend";

const LEDGER_INDEX_KEY = "ledger:index:v2";
const LEDGER_PREFIX = "ledger:v2:";
const CUTOVER_PROOF_OBJECT = "delivery-cutover:v2";
const CUTOVER_PROOF_KEY = "cutover-proof:v2";
const CUTOVER_STAGE_PATH = "/cutover/stage";
const CUTOVER_READ_PATH = "/cutover/read";
const AUTO_LEDGER_LIMIT = 5_000;
const TEST_LEDGER_LIMIT = 1_000;
const TEST_TERMINAL_RETENTION_MS = 30 * 24 * 60 * 60 * 1000;
const PRE_ATTEMPT_RECOVERY_GRACE_MS = 5 * 60 * 1000;
const CUTOVER_SERVER_BARRIER_MS =
  DELIVERY_V2_KV_VISIBILITY_MS + DELIVERY_V2_LEGACY_PENDING_TTL_MS;

interface LedgerIndexEntry {
  keyHash: string;
  keyKind: DeliveryKeyKind;
}

interface LedgerIndex {
  schemaVersion: 2;
  entries: LedgerIndexEntry[];
}

interface LedgerRecord {
  schemaVersion: 2;
  keyHash: string;
  keyKind: DeliveryKeyKind;
  deviceIdentity: string;
  recipientIdentity: string;
  fingerprint: string;
  status: "reserved" | "attempted" | "done" | "rejected";
  createdAt: string;
  updatedAt: string;
  pendingAt: string;
  attemptedAt?: string;
  doneAt?: string;
  rejectedAt?: string;
  quotaDay?: string;
  quotaReserved: boolean;
  response?: { ok: true };
  /** Relay-mapped status only; provider status and body are never retained. */
  rejectionStatus?: 422 | 429;
  /** Imported from quiesced v1 KV; no raw key, recipient, or value is retained. */
  legacyImported?: true;
}

interface LegacyImportRecord {
  schemaVersion: 1;
  kind: "legacy-import";
  legacyImported: true;
  status: "done" | "attempted";
  deviceIdentity: string;
  recipientIdentity: string;
  importedAt: string;
  quotaDay: string;
}

interface CutoverProofRecord {
  schemaVersion: 2;
  proofHash: string;
  firstObservedAt: string;
  visibilityObservedAt?: string;
  readyAt: string;
  stagedAt?: string;
  status: "observed" | "visibility-observed" | "ready";
  observedAutoEvidence: Record<string, "done" | "attempted">;
  marker: DeliveryV2CutoverMarker;
}

interface PreparedDelivery {
  kind: "send";
  ledgerKey: string;
  record: LedgerRecord;
}

interface ImmediateResponse {
  kind: "response";
  response: Response;
}

type PrepareResult = PreparedDelivery | ImmediateResponse;

export async function stageDeliveryV2Cutover(env: Env): Promise<void> {
  const namespace = env.DELIVER_GATE;
  if (!namespace) throw new Error("delivery cutover binding is missing");
  if (!env.TOKEN_SECRET?.trim()) {
    throw new Error("delivery cutover identity secret is missing");
  }
  const marker = await assertDeliveryV2CutoverReady(env);
  const observedAutoEvidence = await scanLegacyAutoDeliveryEvidence(env);
  const stub = namespace.get(namespace.idFromName(CUTOVER_PROOF_OBJECT));
  const response = await stub.fetch(new Request(`https://delivery-cutover${CUTOVER_STAGE_PATH}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ marker, observedAutoEvidence }),
  }));
  if (!response.ok) throw new Error("delivery cutover proof could not be staged");
}

async function readStagedDeliveryV2Cutover(
  env: Env,
): Promise<{ marker: DeliveryV2CutoverMarker; stagedAt: string }> {
  const namespace = env.DELIVER_GATE;
  if (!namespace) throw new Error("delivery cutover binding is missing");
  const stub = namespace.get(namespace.idFromName(CUTOVER_PROOF_OBJECT));
  const response = await stub.fetch(
    new Request(`https://delivery-cutover${CUTOVER_READ_PATH}`),
  );
  if (!response.ok) throw new Error("delivery cutover proof is unavailable");
  const body = await response.json() as { marker?: unknown; stagedAt?: unknown };
  if (!validTimestamp(body.stagedAt)) {
    throw new Error("delivery cutover activation time is unavailable");
  }
  return {
    marker: parseDeliveryV2CutoverMarker(JSON.stringify(body.marker)),
    stagedAt: body.stagedAt,
  };
}

/** One instance is routed per authenticated device, serializing quota + ledger. */
export class DeliverGate {
  constructor(
    private readonly state: DurableObjectState,
    private readonly env: Env,
    private readonly now: () => Date = () => new Date(),
  ) {}

  async fetch(request: Request): Promise<Response> {
    const path = new URL(request.url).pathname;
    if (request.method === "POST" && path === CUTOVER_STAGE_PATH) {
      return this.state.blockConcurrencyWhile(() => this.stageCutover(request));
    }
    if (request.method === "GET" && path === CUTOVER_READ_PATH) {
      return this.state.blockConcurrencyWhile(() => this.readCutover());
    }
    if (request.method !== "POST") {
      return Response.json({ error: "method not allowed" }, { status: 405 });
    }
    return this.state.blockConcurrencyWhile(() => this.handle(request));
  }

  async alarm(): Promise<void> {
    await this.state.blockConcurrencyWhile(async () => {
      await this.cleanupExpiredTests(new Date());
      await this.scheduleCleanupAlarm(new Date());
    });
  }

  private async stageCutover(request: Request): Promise<Response> {
    let body: { marker?: unknown; observedAutoEvidence?: unknown };
    try {
      body = await request.json() as {
        marker?: unknown;
        observedAutoEvidence?: unknown;
      };
    } catch {
      return Response.json({ error: "invalid cutover proof" }, { status: 400 });
    }
    let marker: DeliveryV2CutoverMarker;
    let observedAutoEvidence: Record<string, "done" | "attempted">;
    try {
      marker = parseDeliveryV2CutoverMarker(JSON.stringify(body.marker));
      observedAutoEvidence = normalizeLegacyAutoEvidence(
        body.observedAutoEvidence,
      );
      assertLegacyAutoSnapshotCovers(
        marker.legacyAutoEvidence,
        observedAutoEvidence,
      );
    } catch {
      return Response.json({ error: "invalid cutover proof" }, { status: 400 });
    }
    const proofHash = await hashDeliveryV2CutoverProof(marker);
    const now = this.now();
    const existingRaw = await this.state.storage.get<unknown>(CUTOVER_PROOF_KEY);
    if (existingRaw === undefined) {
      await this.state.storage.put(CUTOVER_PROOF_KEY, {
        schemaVersion: 2,
        proofHash,
        firstObservedAt: now.toISOString(),
        readyAt: new Date(now.getTime() + CUTOVER_SERVER_BARRIER_MS).toISOString(),
        status: "observed",
        observedAutoEvidence,
        marker,
      } satisfies CutoverProofRecord);
      return cutoverObservationRequired();
    }
    try {
      const existing = await normalizeCutoverProof(existingRaw, false);
      if (existing.proofHash !== proofHash) {
        return Response.json(
          { error: "cutover proof changed after first observation" },
          { status: 409 },
        );
      }
      if (existing.status === "ready") {
        return Response.json({ ok: true }, { status: 200 });
      }
      const mergedEvidence = mergeLegacyAutoEvidence(
        existing.observedAutoEvidence,
        observedAutoEvidence,
      );
      assertLegacyAutoSnapshotCovers(marker.legacyAutoEvidence, mergedEvidence);
      const firstObservedAt = Date.parse(existing.firstObservedAt);
      if (existing.status === "observed") {
        const visibilityAt = firstObservedAt + DELIVERY_V2_KV_VISIBILITY_MS;
        const pendingExpiryAt =
          firstObservedAt + DELIVERY_V2_LEGACY_PENDING_TTL_MS;
        if (now.getTime() < visibilityAt) return cutoverObservationRequired();
        if (now.getTime() >= pendingExpiryAt) {
          return Response.json(
            { error: "cutover visibility observation was missed" },
            { status: 503 },
          );
        }
        await this.state.storage.put(CUTOVER_PROOF_KEY, {
          ...existing,
          status: "visibility-observed",
          visibilityObservedAt: now.toISOString(),
          observedAutoEvidence: mergedEvidence,
        } satisfies CutoverProofRecord);
        return cutoverObservationRequired();
      }
      if (now.getTime() < Date.parse(existing.readyAt)) {
        await this.state.storage.put(CUTOVER_PROOF_KEY, {
          ...existing,
          observedAutoEvidence: mergedEvidence,
        } satisfies CutoverProofRecord);
        return cutoverObservationRequired();
      }
      await this.state.storage.put(CUTOVER_PROOF_KEY, {
        ...existing,
        status: "ready",
        stagedAt: now.toISOString(),
        observedAutoEvidence: mergedEvidence,
      } satisfies CutoverProofRecord);
      return Response.json({ ok: true }, { status: 201 });
    } catch {
      return Response.json({ error: "cutover proof is invalid" }, { status: 503 });
    }
  }

  private async readCutover(): Promise<Response> {
    const raw = await this.state.storage.get<unknown>(CUTOVER_PROOF_KEY);
    if (raw === undefined) {
      return Response.json({ error: "cutover proof is not staged" }, { status: 503 });
    }
    try {
      const record = await normalizeCutoverProof(raw);
      parseDeliveryV2CutoverMarker(JSON.stringify(record.marker));
      return Response.json(
        { marker: record.marker, stagedAt: record.stagedAt },
        { status: 200 },
      );
    } catch {
      return Response.json({ error: "cutover proof is invalid" }, { status: 503 });
    }
  }

  private async readLegacyImport(
    request: ValidatedDeliverRequest,
    device: AuthenticatedDevice,
  ): Promise<LegacyImportRecord | undefined> {
    const raw = await this.state.storage.get<unknown>(
      await legacyImportStorageKey(request.date, device.recipientIdentity),
    );
    return raw === undefined ? undefined : normalizeLegacyImport(raw, device);
  }

  private async handle(request: Request): Promise<Response> {
    const deviceIdentity = request.headers.get("X-Device-Identity") ?? "";
    const recipientIdentity = request.headers.get("X-Recipient-Identity") ?? "";
    const deviceCreatedAt = request.headers.get("X-Device-Created-At") ?? "";
    const deliveryGeneration = request.headers.get(
      "X-Device-Delivery-Generation",
    );
    if (
      !isHash(deviceIdentity) ||
      !isHash(recipientIdentity) ||
      !validTimestamp(deviceCreatedAt)
    ) {
      return Response.json(
        { error: "authenticated device binding is missing", ambiguous: false },
        { status: 503 },
      );
    }

    let body: DeliverBody;
    try {
      body = (await request.json()) as DeliverBody;
    } catch {
      return Response.json({ error: "invalid JSON body" }, { status: 400 });
    }
    const normalizedTo = typeof body.to === "string" ? body.to.trim().toLowerCase() : "";
    const computedRecipient = await sha256Hex(
      `${this.env.TOKEN_SECRET}:recipient:${normalizedTo}`,
    );
    if (!normalizedTo || computedRecipient !== recipientIdentity) {
      return Response.json(
        { error: "authenticated recipient binding does not match" },
        { status: 403 },
      );
    }

    const device: AuthenticatedDevice = {
      identity: deviceIdentity,
      recipientIdentity,
      email: normalizedTo,
      createdAt: deviceCreatedAt,
      ...(deliveryGeneration === "2" ? { deliveryGeneration: 2 as const } : {}),
    };
    const validated = await validateDeliverRequest({
      device,
      idempotencyHeader: request.headers.get("Idempotency-Key"),
      body,
    });
    if (!validated.ok) {
      return Response.json(
        { error: validated.error },
        { status: validated.status },
      );
    }

    let legacyEvidence: LegacyDeliveryEvidence = "none";
    if (validated.value.keyKind === "auto") {
      try {
        const imported = await this.readLegacyImport(validated.value, device);
        if (imported) return legacyImportResponse(imported.status).response;
        const cutover = await readStagedDeliveryV2Cutover(this.env);
        legacyEvidence = await readLegacyDeliveryEvidence(
          this.env,
          cutover.marker,
          validated.value.date,
          device.email,
        );
        // Absence from eventually-consistent KV is never proof of no v1 attempt.
        // Only device identities created after the durable v2 proof became ready
        // are outside the old Worker's reachable history.
        if (
          legacyEvidence === "none" &&
          (device.deliveryGeneration !== 2 ||
            Date.parse(device.createdAt) <= Date.parse(cutover.stagedAt))
        ) {
          throw new Error("pre-cutover legacy absence cannot be proven");
        }
      } catch {
        return Response.json(
          { error: "delivery cutover is not ready", ambiguous: false },
          { status: 503 },
        );
      }
    }

    const now = new Date();
    let prepared: PrepareResult;
    try {
      await this.cleanupExpiredTests(now);
      prepared = await this.prepare(validated.value, device, legacyEvidence, now);
    } catch {
      return Response.json(
        { error: "delivery ledger is unavailable", ambiguous: false },
        { status: 503 },
      );
    }
    if (prepared.kind === "response") return prepared.response;

    const attempted: LedgerRecord = {
      ...prepared.record,
      status: "attempted",
      attemptedAt: now.toISOString(),
      updatedAt: now.toISOString(),
    };
    try {
      await this.state.storage.put(prepared.ledgerKey, attempted);
    } catch {
      // No provider invocation occurred. The reserved record may be reclaimed
      // after the bounded grace period without consuming another quota slot.
      return Response.json(
        { error: "delivery attempt could not be started", ambiguous: false },
        { status: 503 },
      );
    }

    try {
      await sendResendEmail(this.env, {
        to: validated.value.to,
        subject: validated.value.subject,
        html: validated.value.html || `<pre>${escapeHtml(validated.value.text)}</pre>`,
        text: validated.value.text || stripTags(validated.value.html),
        idempotencyKey: validated.value.providerKey,
      });
    } catch (error) {
      if (error instanceof ResendProviderError && !error.ambiguous) {
        const rejectionStatus = providerRejectionStatus(error.status);
        try {
          await this.recordDefinitiveRejection(
            prepared.ledgerKey,
            attempted,
            rejectionStatus,
            new Date(),
          );
        } catch {
          return Response.json(
            {
              error: "provider rejected delivery but ledger update is uncertain",
              ambiguous: true,
            },
            { status: 502 },
          );
        }
        return definitiveRejectionResponse(rejectionStatus);
      }
      const reason =
        error instanceof ResendTransportError || error instanceof ResendProviderError
          ? "provider_outcome_ambiguous"
          : "provider_outcome_unknown";
      return Response.json(
        { error: reason, ambiguous: true },
        { status: 502 },
      );
    }

    const completedAt = new Date();
    const done: LedgerRecord = {
      ...attempted,
      status: "done",
      updatedAt: completedAt.toISOString(),
      doneAt: completedAt.toISOString(),
      response: { ok: true },
    };
    try {
      await this.state.storage.put(prepared.ledgerKey, done);
    } catch {
      // The attempted record and quota remain blocking. Returning success here
      // would make replay fidelity depend on a write that did not complete.
      return Response.json(
        { error: "provider accepted delivery but completion storage failed", ambiguous: true },
        { status: 502 },
      );
    }

    if (done.keyKind === "test") {
      await this.scheduleCleanupAlarm(completedAt).catch(() => undefined);
    }
    return Response.json(done.response, { status: 200 });
  }

  private async prepare(
    request: ValidatedDeliverRequest,
    device: AuthenticatedDevice,
    legacyEvidence: LegacyDeliveryEvidence,
    now: Date,
  ): Promise<PrepareResult> {
    const ledgerKey = `${LEDGER_PREFIX}${request.logicalKeyHash}`;
    const legacyImportKey = request.keyKind === "auto"
      ? await legacyImportStorageKey(request.date, device.recipientIdentity)
      : undefined;
    return this.state.storage.transaction(async (txn) => {
      if (legacyImportKey) {
        const existingImportRaw = await txn.get<unknown>(legacyImportKey);
        if (existingImportRaw !== undefined) {
          const imported = normalizeLegacyImport(existingImportRaw, device);
          return legacyImportResponse(imported.status);
        }
      }
      if (legacyImportKey && legacyEvidence !== "none") {
        const quotaDay = utcDateKey(now);
        const quotaKey = quotaStorageKey("auto", quotaDay);
        const used = normalizeQuota(await txn.get<unknown>(quotaKey));
        await txn.put(quotaKey, used + 1);
        await txn.put(legacyImportKey, {
          schemaVersion: 1,
          kind: "legacy-import",
          legacyImported: true,
          status: legacyEvidence,
          deviceIdentity: device.identity,
          recipientIdentity: device.recipientIdentity,
          importedAt: now.toISOString(),
          quotaDay,
        } satisfies LegacyImportRecord);
        return legacyImportResponse(legacyEvidence);
      }

      const index = await readIndex(txn);
      const existingRaw = await txn.get<unknown>(ledgerKey);
      const indexed = index.entries.some(
        (entry) => entry.keyHash === request.logicalKeyHash,
      );
      if ((existingRaw === undefined) !== !indexed) {
        throw new Error("delivery ledger index binding is inconsistent");
      }

      const existing = existingRaw === undefined
        ? undefined
        : normalizeLedgerRecord(existingRaw);
      if (existing) {
        if (
          existing.deviceIdentity !== device.identity ||
          existing.recipientIdentity !== device.recipientIdentity ||
          existing.fingerprint !== request.fingerprint ||
          existing.keyKind !== request.keyKind
        ) {
          return {
            kind: "response" as const,
            response: Response.json(
              { error: "idempotency key is already bound to another request" },
              { status: 409 },
            ),
          };
        }
        if (existing.status === "done") {
          if (!existing.response) throw new Error("done ledger response is missing");
          return {
            kind: "response" as const,
            response: Response.json(existing.response, { status: 200 }),
          };
        }
        if (existing.status === "rejected") {
          if (!existing.rejectionStatus) {
            throw new Error("rejected ledger response is missing");
          }
          return {
            kind: "response" as const,
            response: definitiveRejectionResponse(existing.rejectionStatus),
          };
        }
        if (existing.status === "attempted") {
          return {
            kind: "response" as const,
            response: Response.json(
              { error: "delivery outcome is pending", ambiguous: true },
              { status: 409 },
            ),
          };
        }
        if (existing.status === "reserved") {
          const age = now.getTime() - Date.parse(existing.pendingAt);
          if (!Number.isFinite(age) || age < PRE_ATTEMPT_RECOVERY_GRACE_MS) {
            return {
              kind: "response" as const,
              response: Response.json(
                { error: "delivery reservation is still active", ambiguous: false },
                { status: 409 },
              ),
            };
          }
          const recovered = {
            ...existing,
            updatedAt: now.toISOString(),
            pendingAt: now.toISOString(),
          };
          await txn.put(ledgerKey, recovered);
          return { kind: "send" as const, ledgerKey, record: recovered };
        }
        throw new Error("delivery ledger status is unsupported");
      } else {
        const used = index.entries.filter(
          (entry) => entry.keyKind === request.keyKind,
        ).length;
        const limit = request.keyKind === "auto"
          ? AUTO_LEDGER_LIMIT
          : TEST_LEDGER_LIMIT;
        if (used >= limit) {
          return {
            kind: "response" as const,
            response: Response.json(
              { error: "delivery ledger capacity reached" },
              { status: 507 },
            ),
          };
        }
      }

      const quotaDay = utcDateKey(now);
      const quotaKey = quotaStorageKey(request.keyKind, quotaDay);
      const used = normalizeQuota(await txn.get<unknown>(quotaKey));
      const limit = dailyQuotaLimit(this.env);
      if (used >= limit) {
        return {
          kind: "response" as const,
          response: Response.json(
            { error: `daily quota exceeded (${limit} per UTC day)`, quota: used },
            { status: 429 },
          ),
        };
      }

      if (!existing) {
        index.entries.push({
          keyHash: request.logicalKeyHash,
          keyKind: request.keyKind,
        });
        await txn.put(LEDGER_INDEX_KEY, index);
      }

      const createdAt = now.toISOString();
      const record: LedgerRecord = {
        schemaVersion: 2,
        keyHash: request.logicalKeyHash,
        keyKind: request.keyKind,
        deviceIdentity: device.identity,
        recipientIdentity: device.recipientIdentity,
        fingerprint: request.fingerprint,
        status: "reserved",
        createdAt,
        updatedAt: now.toISOString(),
        pendingAt: now.toISOString(),
        quotaDay,
        quotaReserved: true,
      };
      await txn.put(quotaKey, used + 1);
      await txn.put(ledgerKey, record);
      return { kind: "send" as const, ledgerKey, record };
    });
  }

  private async recordDefinitiveRejection(
    ledgerKey: string,
    attempted: LedgerRecord,
    rejectionStatus: 422 | 429,
    now: Date,
  ): Promise<void> {
    await this.state.storage.transaction(async (txn) => {
      const current = normalizeLedgerRecord(await txn.get<unknown>(ledgerKey));
      if (
        current.status !== "attempted" ||
        current.fingerprint !== attempted.fingerprint ||
        !current.quotaReserved ||
        !current.quotaDay
      ) {
        throw new Error("attempted ledger changed before rollback");
      }
      const quotaKey = quotaStorageKey(current.keyKind, current.quotaDay);
      const used = normalizeQuota(await txn.get<unknown>(quotaKey));
      if (used < 1) throw new Error("quota reservation is missing");
      await txn.put(quotaKey, used - 1);
      await txn.put(ledgerKey, {
        ...current,
        status: "rejected",
        quotaReserved: false,
        updatedAt: now.toISOString(),
        rejectedAt: now.toISOString(),
        rejectionStatus,
      } satisfies LedgerRecord);
    });
  }

  private async cleanupExpiredTests(now: Date): Promise<void> {
    await this.state.storage.transaction(async (txn) => {
      const index = await readIndex(txn);
      const retained: LedgerIndexEntry[] = [];
      for (const entry of index.entries) {
        if (entry.keyKind !== "test") {
          retained.push(entry);
          continue;
        }
        const key = `${LEDGER_PREFIX}${entry.keyHash}`;
        const record = normalizeLedgerRecord(await txn.get<unknown>(key));
        const terminalAt = record.doneAt ?? record.rejectedAt;
        const terminal = record.status === "done" || record.status === "rejected";
        if (
          terminal &&
          terminalAt &&
          now.getTime() - Date.parse(terminalAt) >= TEST_TERMINAL_RETENTION_MS
        ) {
          await txn.delete(key);
        } else {
          retained.push(entry);
        }
      }
      if (retained.length !== index.entries.length) {
        await txn.put(LEDGER_INDEX_KEY, {
          schemaVersion: 2,
          entries: retained,
        } satisfies LedgerIndex);
      }
    });
  }

  private async scheduleCleanupAlarm(now: Date): Promise<void> {
    await this.state.storage.setAlarm(
      now.getTime() + TEST_TERMINAL_RETENTION_MS,
    );
  }
}

async function readIndex(storage: DurableObjectTransaction): Promise<LedgerIndex> {
  const raw = await storage.get<unknown>(LEDGER_INDEX_KEY);
  if (raw === undefined) return { schemaVersion: 2, entries: [] };
  if (!raw || typeof raw !== "object") throw new Error("ledger index is invalid");
  const candidate = raw as Partial<LedgerIndex>;
  if (candidate.schemaVersion !== 2 || !Array.isArray(candidate.entries)) {
    throw new Error("ledger index is invalid");
  }
  const entries: LedgerIndexEntry[] = [];
  const seen = new Set<string>();
  for (const value of candidate.entries) {
    if (
      !value ||
      typeof value !== "object" ||
      !isHash((value as LedgerIndexEntry).keyHash) ||
      ((value as LedgerIndexEntry).keyKind !== "auto" &&
        (value as LedgerIndexEntry).keyKind !== "test") ||
      seen.has((value as LedgerIndexEntry).keyHash)
    ) {
      throw new Error("ledger index entry is invalid");
    }
    seen.add((value as LedgerIndexEntry).keyHash);
    entries.push({
      keyHash: (value as LedgerIndexEntry).keyHash,
      keyKind: (value as LedgerIndexEntry).keyKind,
    });
  }
  return { schemaVersion: 2, entries };
}

function normalizeLedgerRecord(raw: unknown): LedgerRecord {
  if (!raw || typeof raw !== "object") throw new Error("ledger record is missing");
  const value = raw as Partial<LedgerRecord>;
  if (
    value.schemaVersion !== 2 ||
    !isHash(value.keyHash ?? "") ||
    (value.keyKind !== "auto" && value.keyKind !== "test") ||
    !isHash(value.deviceIdentity ?? "") ||
    !isHash(value.recipientIdentity ?? "") ||
    !isHash(value.fingerprint ?? "") ||
    (value.status !== "reserved" &&
      value.status !== "attempted" &&
      value.status !== "done" &&
      value.status !== "rejected") ||
    !validTimestamp(value.createdAt) ||
    !validTimestamp(value.updatedAt) ||
    !validTimestamp(value.pendingAt) ||
    typeof value.quotaReserved !== "boolean"
  ) {
    throw new Error("ledger record is invalid");
  }
  if (value.attemptedAt !== undefined && !validTimestamp(value.attemptedAt)) {
    throw new Error("ledger attemptedAt is invalid");
  }
  if (value.doneAt !== undefined && !validTimestamp(value.doneAt)) {
    throw new Error("ledger doneAt is invalid");
  }
  if (value.rejectedAt !== undefined && !validTimestamp(value.rejectedAt)) {
    throw new Error("ledger rejectedAt is invalid");
  }
  if (value.quotaReserved && !/^\d{4}-\d{2}-\d{2}$/.test(value.quotaDay ?? "")) {
    throw new Error("ledger quota binding is invalid");
  }
  if (value.status === "done") {
    if (
      !value.response ||
      value.response.ok !== true ||
      Object.keys(value.response).some((key) => key !== "ok") ||
      value.rejectionStatus !== undefined
    ) {
      throw new Error("done ledger response is invalid");
    }
  } else if (value.response !== undefined) {
    throw new Error("non-done ledger response is invalid");
  }
  if (value.status === "rejected") {
    if (
      !value.rejectedAt ||
      value.quotaReserved ||
      (value.rejectionStatus !== 422 && value.rejectionStatus !== 429)
    ) {
      throw new Error("rejected ledger response is invalid");
    }
  } else if (value.rejectionStatus !== undefined) {
    throw new Error("non-rejected ledger response is invalid");
  }
  return value as LedgerRecord;
}

async function normalizeCutoverProof(
  raw: unknown,
  requireReady = true,
): Promise<CutoverProofRecord> {
  if (!raw || typeof raw !== "object") {
    throw new Error("cutover proof record is invalid");
  }
  const value = raw as Partial<CutoverProofRecord>;
  const statusValid = value.status === "observed" ||
    value.status === "visibility-observed" || value.status === "ready";
  if (
    value.schemaVersion !== 2 ||
    !isHash(value.proofHash ?? "") ||
    !validTimestamp(value.firstObservedAt) ||
    !validTimestamp(value.readyAt) ||
    !statusValid ||
    !value.marker ||
    Date.parse(value.readyAt) !==
      Date.parse(value.firstObservedAt) + CUTOVER_SERVER_BARRIER_MS ||
    (value.status === "observed" && value.visibilityObservedAt !== undefined) ||
    (value.status !== "observed" && !validTimestamp(value.visibilityObservedAt)) ||
    (value.status === "ready" && !validTimestamp(value.stagedAt)) ||
    (value.status !== "ready" && value.stagedAt !== undefined) ||
    (requireReady && value.status !== "ready")
  ) {
    throw new Error("cutover proof record is invalid");
  }
  const firstObservedAt = Date.parse(value.firstObservedAt);
  const readyAt = Date.parse(value.readyAt);
  if (value.stagedAt !== undefined && Date.parse(value.stagedAt) < readyAt) {
    throw new Error("cutover activation time is invalid");
  }
  if (value.visibilityObservedAt !== undefined) {
    const visibilityObservedAt = Date.parse(value.visibilityObservedAt);
    if (
      visibilityObservedAt < firstObservedAt + DELIVERY_V2_KV_VISIBILITY_MS ||
      visibilityObservedAt >= firstObservedAt + DELIVERY_V2_LEGACY_PENDING_TTL_MS
    ) {
      throw new Error("cutover visibility observation is invalid");
    }
  }
  const marker = parseDeliveryV2CutoverMarker(JSON.stringify(value.marker));
  const observedAutoEvidence = normalizeLegacyAutoEvidence(
    value.observedAutoEvidence,
  );
  assertLegacyAutoSnapshotCovers(
    marker.legacyAutoEvidence,
    observedAutoEvidence,
  );
  if (await hashDeliveryV2CutoverProof(marker) !== value.proofHash) {
    throw new Error("cutover proof hash does not match");
  }
  return { ...value, observedAutoEvidence, marker } as CutoverProofRecord;
}

function normalizeLegacyAutoEvidence(
  raw: unknown,
): Record<string, "done" | "attempted"> {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("legacy automatic evidence is invalid");
  }
  const evidence: Record<string, "done" | "attempted"> = {};
  for (const [identity, status] of Object.entries(raw)) {
    if (
      !isHash(identity) ||
      (status !== "done" && status !== "attempted")
    ) {
      throw new Error("legacy automatic evidence is invalid");
    }
    evidence[identity] = status;
  }
  return evidence;
}

function mergeLegacyAutoEvidence(
  left: Record<string, "done" | "attempted">,
  right: Record<string, "done" | "attempted">,
): Record<string, "done" | "attempted"> {
  const merged = { ...left };
  for (const [identity, status] of Object.entries(right)) {
    merged[identity] = merged[identity] === "attempted" || status === "attempted"
      ? "attempted"
      : "done";
  }
  return merged;
}

function cutoverObservationRequired(): Response {
  return Response.json(
    { error: "cutover proof requires a later observation" },
    { status: 503 },
  );
}

function legacyImportResponse(
  status: LegacyImportRecord["status"],
): ImmediateResponse {
  return {
    kind: "response",
    response: Response.json(
      status === "attempted"
        ? { error: "legacy_delivery_attempted", ambiguous: true }
        : { error: "legacy_delivery_done", ambiguous: false },
      { status: 409 },
    ),
  };
}

function normalizeLegacyImport(
  raw: unknown,
  device: AuthenticatedDevice,
): LegacyImportRecord {
  if (!raw || typeof raw !== "object") {
    throw new Error("legacy import record is invalid");
  }
  const value = raw as Partial<LegacyImportRecord>;
  if (
    value.schemaVersion !== 1 ||
    value.kind !== "legacy-import" ||
    value.legacyImported !== true ||
    (value.status !== "done" && value.status !== "attempted") ||
    value.deviceIdentity !== device.identity ||
    value.recipientIdentity !== device.recipientIdentity ||
    !validTimestamp(value.importedAt) ||
    !/^\d{4}-\d{2}-\d{2}$/.test(value.quotaDay ?? "")
  ) {
    throw new Error("legacy import record is invalid");
  }
  return value as LegacyImportRecord;
}

async function legacyImportStorageKey(
  date: string,
  recipientIdentity: string,
): Promise<string> {
  return `legacy-import:v2:${await sha256Hex(`${date}\u0000${recipientIdentity}`)}`;
}

function quotaStorageKey(kind: DeliveryKeyKind, day: string): string {
  return `quota:v2:${kind}:${day}`;
}

function normalizeQuota(raw: unknown): number {
  if (raw === undefined) return 0;
  if (typeof raw !== "number" || !Number.isSafeInteger(raw) || raw < 0) {
    throw new Error("quota record is invalid");
  }
  return raw;
}

function isHash(value: string): boolean {
  return /^[0-9a-f]{64}$/.test(value);
}

function validTimestamp(value: unknown): value is string {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}

function providerRejectionStatus(status: number): 422 | 429 {
  // Store only the relay contract classification. The provider status and body
  // stay outside the ledger, client response, and logs.
  return status === 429 ? 429 : 422;
}

function definitiveRejectionResponse(status: 422 | 429): Response {
  return Response.json(
    { error: "provider_definitive_rejection", ambiguous: false },
    { status },
  );
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
