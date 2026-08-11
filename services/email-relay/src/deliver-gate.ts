import { sha256Hex, utcDateKey } from "./crypto";
import {
  validateDeliverRequest,
  type DeliverBody,
  type DeliveryKeyKind,
  type ValidatedDeliverRequest,
} from "./deliver-logic";
import {
  dailyQuotaLimit,
  hashLegacyAutoDeliveryIdentity,
  type AuthenticatedDevice,
  type Env,
  type LegacyDeliveryEvidence,
} from "./kv";
import {
  authorizeAutomaticDelivery,
  handleCutoverControlFetch,
} from "./cutover-control";
import {
  ResendProviderError,
  ResendTransportError,
  sendResendEmail,
} from "./resend";

const LEDGER_INDEX_KEY = "ledger:index:v2";
const LEDGER_PREFIX = "ledger:v2:";
const AUTO_LEDGER_LIMIT = 5_000;
const TEST_LEDGER_LIMIT = 1_000;
const TEST_TERMINAL_RETENTION_MS = 30 * 24 * 60 * 60 * 1000;
const PRE_ATTEMPT_RECOVERY_GRACE_MS = 5 * 60 * 1000;

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
  status:
    | "reserved"
    | "attempted"
    | "done"
    | "rejected"
    | "legacy-done"
    | "legacy-attempted";
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
  /** Imported from provider-fenced legacy evidence; no raw key or recipient is retained. */
  legacyImported?: true;
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

/** One instance is routed per authenticated device, serializing quota + ledger. */
export class DeliverGate {
  constructor(
    private readonly state: DurableObjectState,
    private readonly env: Env,
    private readonly now: () => Date = () => new Date(),
  ) {}

  async fetch(request: Request): Promise<Response> {
    const control = await handleCutoverControlFetch(
      this.state,
      this.env,
      this.now,
      request,
    );
    if (control) return control;
    if (request.method !== "POST") {
      return Response.json({ error: "method not allowed" }, { status: 405 });
    }
    return this.state.blockConcurrencyWhile(() => this.handle(request));
  }

  async alarm(): Promise<void> {
    await this.state.blockConcurrencyWhile(async () => {
      const now = this.now();
      await this.cleanupExpiredTests(now);
      await this.scheduleCleanupAlarm(now);
    });
  }

  private async readImportedLegacyLedger(
    request: ValidatedDeliverRequest,
    device: AuthenticatedDevice,
  ): Promise<ImmediateResponse | undefined> {
    const raw = await this.state.storage.get<unknown>(
      `${LEDGER_PREFIX}${request.logicalKeyHash}`,
    );
    if (raw === undefined) return undefined;
    const record = normalizeLedgerRecord(raw);
    if (
      record.deviceIdentity !== device.identity ||
      record.recipientIdentity !== device.recipientIdentity ||
      record.fingerprint !== request.fingerprint ||
      record.keyKind !== "auto"
    ) {
      return undefined;
    }
    if (record.status === "legacy-done") return legacyImportResponse("done");
    if (record.status === "legacy-attempted") {
      return legacyImportResponse("attempted");
    }
    return undefined;
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
        const imported = await this.readImportedLegacyLedger(validated.value, device);
        if (imported) return imported.response;
        const evidenceIdentity = await hashLegacyAutoDeliveryIdentity(
          this.env.TOKEN_SECRET,
          validated.value.date,
          device.email,
        );
        const decision = await authorizeAutomaticDelivery(this.env, {
          evidenceIdentity,
          deviceCreatedAt: device.createdAt,
          deliveryGeneration: device.deliveryGeneration,
        });
        if (!decision.authorized) throw new Error("automatic delivery is locked");
        legacyEvidence = decision.legacyEvidence;
      } catch {
        return Response.json(
          { error: "delivery cutover is not ready", ambiguous: false },
          { status: 503 },
        );
      }
    }

    const now = this.now();
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
            this.now(),
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

    const completedAt = this.now();
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
    return this.state.storage.transaction(async (txn) => {
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
        if (existing.status === "legacy-done") {
          return legacyImportResponse("done");
        }
        if (existing.status === "legacy-attempted") {
          return legacyImportResponse("attempted");
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
          if (request.keyKind === "auto" && legacyEvidence !== "none") {
            const blocked: LedgerRecord = {
              ...existing,
              status: legacyEvidence === "attempted"
                ? "legacy-attempted"
                : "legacy-done",
              attemptedAt: now.toISOString(),
              updatedAt: now.toISOString(),
              legacyImported: true,
            };
            await txn.put(ledgerKey, blocked);
            return legacyImportResponse(legacyEvidence);
          }
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

      if (request.keyKind === "auto" && legacyEvidence !== "none") {
        const quotaDay = utcDateKey(now);
        const quotaKey = quotaStorageKey("auto", quotaDay);
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
        index.entries.push({
          keyHash: request.logicalKeyHash,
          keyKind: "auto",
        });
        const createdAt = now.toISOString();
        const record: LedgerRecord = {
          schemaVersion: 2,
          keyHash: request.logicalKeyHash,
          keyKind: "auto",
          deviceIdentity: device.identity,
          recipientIdentity: device.recipientIdentity,
          fingerprint: request.fingerprint,
          status: legacyEvidence === "attempted"
            ? "legacy-attempted"
            : "legacy-done",
          createdAt,
          updatedAt: createdAt,
          pendingAt: createdAt,
          attemptedAt: createdAt,
          quotaDay,
          quotaReserved: true,
          legacyImported: true,
        };
        await txn.put(LEDGER_INDEX_KEY, index);
        await txn.put(quotaKey, used + 1);
        await txn.put(ledgerKey, record);
        return legacyImportResponse(legacyEvidence);
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
      value.status !== "rejected" &&
      value.status !== "legacy-done" &&
      value.status !== "legacy-attempted") ||
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
  if (value.status === "legacy-done" || value.status === "legacy-attempted") {
    if (
      value.keyKind !== "auto" ||
      value.legacyImported !== true ||
      !value.attemptedAt ||
      !value.quotaReserved
    ) {
      throw new Error("legacy ledger record is invalid");
    }
  } else if (value.legacyImported !== undefined) {
    throw new Error("non-legacy ledger record is invalid");
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

function legacyImportResponse(
  status: "done" | "attempted",
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
