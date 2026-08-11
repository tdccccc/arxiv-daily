import { hmacSha256Hex, sha256Hex } from "./crypto";
import {
  DELIVERY_V2_KV_VISIBILITY_MS,
  DELIVERY_V2_LEGACY_PENDING_TTL_MS,
  DELIVERY_V3_CUTOVER_AUDIT_KEY,
  scanLegacyAutoDeliveryEvidence,
  writeDeliveryV3CutoverAuditMarker,
  type DeliveryV3CutoverAuditMarker,
  type Env,
  type LegacyDeliveryEvidence,
} from "./kv";

export const CUTOVER_CONTROL_OBJECT = "delivery-cutover:v3";
export const CUTOVER_STATUS_PATH = "/cutover/status";
export const CUTOVER_ACTION_PATH = "/cutover/action";
export const CUTOVER_AUTOMATIC_PATH = "/cutover/automatic";
export const PROVIDER_FENCE_ATTESTATION = "old-resend-credential-revoked";
export const MAX_LEGACY_AUTO_EVIDENCE = 512;
export const MAX_PREFLIGHT_SCAN_DURATION_MS = 30_000;
export const INVENTORY_FRESHNESS_MS = 5 * 60_000;

const CONTROL_KEY = "cutover-control:v3";
const OPERATION_PREFIX = "cutover-operation:v3:";
const MARKER_OBSERVATION_WAIT_MS = DELIVERY_V2_KV_VISIBILITY_MS;
const activeOperations = new WeakMap<DurableObjectState, Set<string>>();

type ExactEvidence = Record<string, "done" | "attempted">;

export type CutoverPhase =
  | "locked"
  | "inventoried"
  | "observing"
  | "sealed"
  | "ready"
  | "blocked";
export type CutoverAction =
  | "inventory"
  | "provider-fence"
  | "observe"
  | "seal"
  | "repair";

interface ScanRecord {
  startedAt: string;
  completedAt: string;
  automaticKeyCount: number;
  durationMs: number;
}

interface InventoryRecord extends ScanRecord {
  capacity: typeof MAX_LEGACY_AUTO_EVIDENCE;
  withinCapacity: boolean;
  durationBudgetMs: typeof MAX_PREFLIGHT_SCAN_DURATION_MS;
  credentialRevocationDeadline: string;
}

interface ProviderFenceRecord {
  attested: true;
  boundary: "old_resend_credential_revoked";
  attestedAt: string;
}

interface MarkerAuditRecord {
  markerHash: string;
  writeCompletedAt: string;
  observations: Array<{ observedAt: string }>;
}

interface SafeBlock {
  code:
    | "legacy_scan_unavailable"
    | "legacy_scan_unsupported_key"
    | "legacy_evidence_capacity_reached"
    | "legacy_pending_window_missed"
    | "marker_write_unavailable"
    | "marker_observation_unavailable"
    | "marker_observation_mismatch"
    | "control_state_invalid";
  at: string;
}

interface PendingOperation {
  operationId: string;
  action: CutoverAction;
  inputHash: string;
  baseRevision: number;
  basePhase: Exclude<CutoverPhase, "blocked">;
  startedAt: string;
  attestedAt?: string;
  recoverFromMarker?: true;
}

interface LastOperation {
  operationId: string;
  action: CutoverAction;
  status: number;
  completedAt: string;
}

interface CutoverControlRecord {
  schemaVersion: 3;
  phase: CutoverPhase;
  revision: number;
  updatedAt: string;
  legacyAutoEvidence: ExactEvidence;
  lastOperation?: LastOperation;
  preFenceInventory?: InventoryRecord;
  providerFence?: ProviderFenceRecord;
  postFenceScan?: ScanRecord;
  followupScan?: ScanRecord;
  markerAudit?: MarkerAuditRecord;
  readyAt?: string;
  blocked?: SafeBlock;
  recoverPhase?: Exclude<CutoverPhase, "blocked">;
  pendingOperation?: PendingOperation;
}

interface StoredOperation {
  schemaVersion: 1;
  operationId: string;
  action: CutoverAction;
  inputHash: string;
  status: number;
  body: string;
  completedAt: string;
}

interface ActionInput {
  action: CutoverAction;
  operationId: string;
  attestation?: string;
}

interface BeginExecute {
  kind: "execute";
  control: CutoverControlRecord;
  pending: PendingOperation;
}

interface BeginResponse {
  kind: "response";
  response: Response;
}

type BeginResult = BeginExecute | BeginResponse;

interface CompletedAction {
  control: CutoverControlRecord;
  status: number;
}

interface CompletedScan {
  record: ScanRecord;
  evidence: ExactEvidence;
}

interface ValidMarker {
  marker: DeliveryV3CutoverAuditMarker;
  hash: string;
}

export function isCutoverOperationId(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

export async function fetchCutoverStatus(env: Env): Promise<Response> {
  return controlStub(env).fetch(
    new Request(`https://cutover-control${CUTOVER_STATUS_PATH}`),
  );
}

export async function postCutoverAction(
  env: Env,
  action: CutoverAction,
  operationId: string,
  attestation?: string,
): Promise<Response> {
  return controlStub(env).fetch(
    new Request(`https://cutover-control${CUTOVER_ACTION_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        operationId,
        ...(attestation ? { attestation } : {}),
      }),
    }),
  );
}

export async function authorizeAutomaticDelivery(
  env: Env,
  input: {
    evidenceIdentity: string;
    deviceCreatedAt: string;
    deliveryGeneration?: 2;
  },
): Promise<
  | { authorized: true; legacyEvidence: LegacyDeliveryEvidence }
  | { authorized: false }
> {
  const response = await controlStub(env).fetch(
    new Request(`https://cutover-control${CUTOVER_AUTOMATIC_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
  if (!response.ok) return { authorized: false };
  const body = await response.json() as {
    authorized?: unknown;
    legacyEvidence?: unknown;
  };
  if (
    body.authorized !== true ||
    (body.legacyEvidence !== "none" &&
      body.legacyEvidence !== "done" &&
      body.legacyEvidence !== "attempted")
  ) {
    return { authorized: false };
  }
  return {
    authorized: true,
    legacyEvidence: body.legacyEvidence,
  };
}

export async function handleCutoverControlFetch(
  state: DurableObjectState,
  env: Env,
  now: () => Date,
  request: Request,
): Promise<Response | undefined> {
  const path = new URL(request.url).pathname;
  if (path === CUTOVER_STATUS_PATH) {
    if (request.method !== "GET") return methodNotAllowed();
    return state.blockConcurrencyWhile(() => readStatus(state, now()));
  }
  if (path === CUTOVER_ACTION_PATH) {
    if (request.method !== "POST") return methodNotAllowed();
    return runAction(state, env, now, request);
  }
  if (path === CUTOVER_AUTOMATIC_PATH) {
    if (request.method !== "POST") return methodNotAllowed();
    return state.blockConcurrencyWhile(() => authorizeAutomatic(state, request));
  }
  return undefined;
}

async function readStatus(state: DurableObjectState, now: Date): Promise<Response> {
  const raw = await state.storage.get<unknown>(CONTROL_KEY);
  if (raw === undefined) return statusResponse(lockedControl(new Date(0)), now);
  try {
    return statusResponse(normalizeControl(raw), now);
  } catch {
    return invalidControlResponse();
  }
}

async function authorizeAutomatic(
  state: DurableObjectState,
  request: Request,
): Promise<Response> {
  let body: {
    evidenceIdentity?: unknown;
    deviceCreatedAt?: unknown;
    deliveryGeneration?: unknown;
  };
  try {
    body = await request.json() as typeof body;
  } catch {
    return deniedAutomatic();
  }
  if (
    !isHash(body.evidenceIdentity) ||
    !validTimestamp(body.deviceCreatedAt) ||
    (body.deliveryGeneration !== undefined && body.deliveryGeneration !== 2)
  ) {
    return deniedAutomatic();
  }
  let control: CutoverControlRecord;
  try {
    control = normalizeControl(await state.storage.get<unknown>(CONTROL_KEY));
  } catch {
    return deniedAutomatic();
  }
  if (
    control.phase !== "ready" ||
    !control.readyAt ||
    control.pendingOperation?.action === "repair"
  ) {
    return deniedAutomatic();
  }
  const legacyEvidence: LegacyDeliveryEvidence =
    (control.legacyAutoEvidence[body.evidenceIdentity] as
      | "done"
      | "attempted"
      | undefined) ?? "none";
  if (
    legacyEvidence === "none" &&
    (body.deliveryGeneration !== 2 ||
      Date.parse(body.deviceCreatedAt) <= Date.parse(control.readyAt))
  ) {
    return deniedAutomatic();
  }
  return Response.json({ authorized: true, legacyEvidence });
}

async function runAction(
  state: DurableObjectState,
  env: Env,
  now: () => Date,
  request: Request,
): Promise<Response> {
  const input = await parseAction(request);
  if (input instanceof Response) return input;
  const inputHash = await sha256Hex(JSON.stringify([
    input.action,
    input.attestation ?? null,
  ]));
  const begun = await state.blockConcurrencyWhile(() =>
    beginAction(state, now(), input, inputHash)
  );
  if (begun.kind === "response") return begun.response;

  try {
    const completed = await executeAction(state, env, now, input, begun.control);
    return await state.blockConcurrencyWhile(() =>
      finalizeAction(state, now(), input, inputHash, begun.pending, completed)
    );
  } catch (error) {
    return await state.blockConcurrencyWhile(() =>
      finalizeFailure(state, now(), input, inputHash, begun.pending, error)
    );
  } finally {
    activeOperationSet(state).delete(input.operationId);
  }
}

async function parseAction(request: Request): Promise<ActionInput | Response> {
  let body: Record<string, unknown>;
  try {
    body = await request.json() as Record<string, unknown>;
  } catch {
    return safeError("invalid cutover action", 400);
  }
  if (
    Object.keys(body).some(
      (key) => key !== "action" && key !== "operationId" && key !== "attestation",
    ) ||
    !isAction(body.action) ||
    !isCutoverOperationId(body.operationId) ||
    (body.attestation !== undefined && typeof body.attestation !== "string")
  ) {
    return safeError("invalid cutover action", 400);
  }
  if (
    body.action === "provider-fence" &&
    body.attestation !== PROVIDER_FENCE_ATTESTATION
  ) {
    return safeError("provider fence attestation is required", 400);
  }
  if (body.action !== "provider-fence" && body.attestation !== undefined) {
    return safeError("invalid cutover action", 400);
  }
  return {
    action: body.action,
    operationId: body.operationId,
    ...(typeof body.attestation === "string"
      ? { attestation: body.attestation }
      : {}),
  };
}

async function beginAction(
  state: DurableObjectState,
  now: Date,
  input: ActionInput,
  inputHash: string,
): Promise<BeginResult> {
  const operationRaw = await state.storage.get<unknown>(
    `${OPERATION_PREFIX}${input.operationId}`,
  );
  if (operationRaw !== undefined) {
    try {
      const operation = normalizeOperation(operationRaw);
      if (operation.action !== input.action || operation.inputHash !== inputHash) {
        return { kind: "response", response: safeError("operation id is already bound", 409) };
      }
      return { kind: "response", response: storedOperationResponse(operation) };
    } catch {
      return { kind: "response", response: invalidControlResponse() };
    }
  }

  const raw = await state.storage.get<unknown>(CONTROL_KEY);
  let control: CutoverControlRecord;
  let recoverFromMarker = false;
  if (raw === undefined) {
    control = lockedControl(now);
    recoverFromMarker = input.action === "repair";
  } else {
    try {
      control = normalizeControl(raw);
    } catch {
      if (input.action !== "inventory" && input.action !== "repair") {
        return { kind: "response", response: invalidControlResponse() };
      }
      control = lockedControl(now);
      recoverFromMarker = input.action === "repair";
    }
  }

  if (control.pendingOperation) {
    if (
      control.pendingOperation.operationId !== input.operationId ||
      control.pendingOperation.action !== input.action ||
      control.pendingOperation.inputHash !== inputHash
    ) {
      return {
        kind: "response",
        response: Response.json(statusBody(control, now), { status: 409 }),
      };
    }
    if (activeOperationSet(state).has(input.operationId)) {
      return {
        kind: "response",
        response: Response.json(statusBody(control, now), { status: 409 }),
      };
    }
    activeOperationSet(state).add(input.operationId);
    return { kind: "execute", control, pending: control.pendingOperation };
  }

  const phase = effectivePhase(control);
  if (isMonotonicNoop(input.action, control, phase)) {
    return persistSuccessfulNoop(state, now, input, inputHash, control);
  }
  if (
    input.action === "provider-fence" &&
    !inventorySafeBeforeCredentialRevocation(control.preFenceInventory, now)
  ) {
    return {
      kind: "response",
      response: Response.json(statusBody(control, now), { status: 409 }),
    };
  }
  if (!actionAllowed(input.action, control, phase)) {
    return {
      kind: "response",
      response: Response.json(statusBody(control, now), { status: 409 }),
    };
  }
  if (input.action === "observe" && observationTiming(control, now) === "early") {
    return {
      kind: "response",
      response: Response.json(statusBody(control, now), { status: 409 }),
    };
  }

  const pending: PendingOperation = {
    operationId: input.operationId,
    action: input.action,
    inputHash,
    baseRevision: control.revision,
    basePhase: phase,
    startedAt: now.toISOString(),
    ...(input.action === "provider-fence"
      ? { attestedAt: now.toISOString() }
      : {}),
    ...(recoverFromMarker ? { recoverFromMarker: true as const } : {}),
  };
  const pendingControl: CutoverControlRecord = {
    ...control,
    updatedAt: now.toISOString(),
    pendingOperation: pending,
  };
  await state.storage.put(CONTROL_KEY, pendingControl);
  activeOperationSet(state).add(input.operationId);
  return { kind: "execute", control: pendingControl, pending };
}

async function persistSuccessfulNoop(
  state: DurableObjectState,
  completedAt: Date,
  input: ActionInput,
  inputHash: string,
  control: CutoverControlRecord,
): Promise<BeginResponse> {
  return state.storage.transaction(async (txn) => {
    const operationKey = `${OPERATION_PREFIX}${input.operationId}`;
    const existing = await txn.get<unknown>(operationKey);
    if (existing !== undefined) {
      const operation = normalizeOperation(existing);
      if (operation.action !== input.action || operation.inputHash !== inputHash) {
        return {
          kind: "response" as const,
          response: safeError("operation id is already bound", 409),
        };
      }
      return {
        kind: "response" as const,
        response: storedOperationResponse(operation),
      };
    }
    const current = normalizeControl(await txn.get<unknown>(CONTROL_KEY));
    if (
      current.revision !== control.revision ||
      current.phase !== control.phase ||
      current.pendingOperation !== undefined
    ) {
      return {
        kind: "response" as const,
        response: Response.json(statusBody(current, completedAt), { status: 409 }),
      };
    }
    const operation = buildStoredOperation(
      completedAt,
      input,
      inputHash,
      200,
      JSON.stringify(statusBody(current, completedAt, false)),
    );
    await txn.put(operationKey, operation);
    return {
      kind: "response" as const,
      response: storedOperationResponse(operation),
    };
  });
}

async function executeAction(
  state: DurableObjectState,
  env: Env,
  now: () => Date,
  input: ActionInput,
  control: CutoverControlRecord,
): Promise<CompletedAction> {
  if (input.action === "inventory") {
    const scan = await performLegacyScan(env, now, {});
    const inventory = inventoryRecord(scan.record, true);
    return {
      status: 200,
      control: {
        ...lockedControl(new Date(scan.record.completedAt)),
        phase: "inventoried",
        revision: control.revision + 1,
        legacyAutoEvidence: scan.evidence,
        preFenceInventory: inventory,
      },
    };
  }

  if (input.action === "provider-fence") {
    const attestedAt = control.pendingOperation?.attestedAt;
    if (!attestedAt) throw new CutoverFailure("control_state_invalid");
    try {
      const scan = await performLegacyScan(
        env,
        now,
        control.legacyAutoEvidence,
      );
      const fenced: CutoverControlRecord = {
        ...clearBlocked(control),
        phase: "observing",
        revision: control.revision + 1,
        updatedAt: scan.record.completedAt,
        legacyAutoEvidence: scan.evidence,
        providerFence: {
          attested: true,
          boundary: "old_resend_credential_revoked",
          attestedAt,
        },
        postFenceScan: scan.record,
        followupScan: undefined,
        markerAudit: undefined,
        readyAt: undefined,
      };
      if (
        Date.parse(scan.record.completedAt) >=
          Date.parse(attestedAt) + DELIVERY_V2_LEGACY_PENDING_TTL_MS
      ) {
        throw new CutoverFailure("legacy_pending_window_missed", {
          ...clearBlocked(control),
          phase: "inventoried",
          updatedAt: scan.record.completedAt,
          providerFence: fenced.providerFence,
        });
      }
      return { status: 200, control: fenced };
    } catch (error) {
      if (error instanceof CutoverFailure && !error.control) {
        throw new CutoverFailure(error.code, {
          ...clearBlocked(control),
          phase: "inventoried",
          providerFence: {
            attested: true,
            boundary: "old_resend_credential_revoked",
            attestedAt,
          },
        });
      }
      throw error;
    }
  }

  if (input.action === "observe" && effectivePhase(control) === "observing") {
    if (observationTiming(control, now()) === "legacy-window-missed") {
      throw new CutoverFailure("legacy_pending_window_missed", control);
    }
    const scan = await performLegacyScan(env, now, control.legacyAutoEvidence);
    const observedControl: CutoverControlRecord = {
      ...clearBlocked(control),
      legacyAutoEvidence: scan.evidence,
      followupScan: scan.record,
    };
    if (
      Date.parse(scan.record.completedAt) >=
        Date.parse(control.providerFence!.attestedAt) +
          DELIVERY_V2_LEGACY_PENDING_TTL_MS
    ) {
      throw new CutoverFailure("legacy_pending_window_missed", observedControl);
    }
    const marker = await buildAuditMarker(env, observedControl, now());
    let markerHash: string;
    try {
      markerHash = await writeDeliveryV3CutoverAuditMarker(env, marker);
    } catch {
      throw new CutoverFailure("marker_write_unavailable", observedControl);
    }
    const writeCompletedAt = now().toISOString();
    return {
      status: 200,
      control: {
        ...observedControl,
        phase: "sealed",
        revision: control.revision + 1,
        updatedAt: writeCompletedAt,
        markerAudit: { markerHash, writeCompletedAt, observations: [] },
      },
    };
  }

  if (input.action === "observe" && effectivePhase(control) === "sealed") {
    const marker = await readValidAuditMarker(env);
    const observedAt = now().toISOString();
    if (!marker || marker.hash !== control.markerAudit!.markerHash) {
      throw new CutoverFailure(
        marker ? "marker_observation_mismatch" : "marker_observation_unavailable",
        control,
      );
    }
    return {
      status: 200,
      control: {
        ...clearBlocked(control),
        phase: "sealed",
        revision: control.revision + 1,
        updatedAt: observedAt,
        markerAudit: {
          ...control.markerAudit!,
          observations: [
            ...control.markerAudit!.observations,
            { observedAt },
          ],
        },
      },
    };
  }

  if (input.action === "seal") {
    const marker = await readValidAuditMarker(env);
    if (!marker || marker.hash !== control.markerAudit!.markerHash) {
      throw new CutoverFailure(
        marker ? "marker_observation_mismatch" : "marker_observation_unavailable",
        control,
      );
    }
    const readyAt = now().toISOString();
    return {
      status: 200,
      control: {
        ...clearBlocked(control),
        phase: "ready",
        revision: control.revision + 1,
        updatedAt: readyAt,
        readyAt,
      },
    };
  }

  if (input.action === "repair") {
    const phase = effectivePhase(control);
    if (phase === "sealed" || phase === "ready") {
      const marker = await buildAuditMarker(env, control, now());
      let markerHash: string;
      try {
        markerHash = await writeDeliveryV3CutoverAuditMarker(env, marker);
      } catch {
        throw new CutoverFailure("marker_write_unavailable", control);
      }
      const writeCompletedAt = now().toISOString();
      return {
        status: 200,
        control: {
          ...clearBlocked(control),
          phase: "sealed",
          revision: control.revision + 1,
          updatedAt: writeCompletedAt,
          markerAudit: { markerHash, writeCompletedAt, observations: [] },
          readyAt: undefined,
        },
      };
    }
    const recovered = control.pendingOperation?.recoverFromMarker
      ? await readValidAuditMarker(env)
      : undefined;
    if (!recovered) {
      return {
        status: 503,
        control: {
          ...lockedControl(now()),
          revision: control.revision + 1,
        },
      };
    }
    return {
      status: 200,
      control: controlFromMarker(recovered, control.revision + 1, now()),
    };
  }

  throw new Error("unsupported cutover action");
}

async function performLegacyScan(
  env: Env,
  now: () => Date,
  baseEvidence: ExactEvidence,
): Promise<CompletedScan> {
  const started = now();
  const evidence = canonicalEvidence(baseEvidence);
  let automaticKeyCount = 0;
  let evidenceCapacityReached = false;
  try {
    await scanLegacyAutoDeliveryEvidence(env, {
      collect: false,
      onEvidence: (identity, status) => {
        automaticKeyCount += 1;
        if (automaticKeyCount > MAX_LEGACY_AUTO_EVIDENCE) {
          evidenceCapacityReached = true;
          return;
        }
        evidence[identity] = evidence[identity] === "attempted" || status === "attempted"
          ? "attempted"
          : "done";
        if (Object.keys(evidence).length > MAX_LEGACY_AUTO_EVIDENCE) {
          delete evidence[identity];
          evidenceCapacityReached = true;
        }
      },
    });
  } catch (error) {
    const completed = now();
    if (
      error instanceof Error &&
      error.message === "legacy automatic delivery scan encountered an unsupported key"
    ) {
      throw new CutoverFailure("legacy_scan_unsupported_key");
    }
    throw new CutoverFailure("legacy_scan_unavailable");
  }
  const completed = now();
  if (evidenceCapacityReached) {
    const record = scanRecord(started, completed, automaticKeyCount);
    throw new CutoverFailure(
      "legacy_evidence_capacity_reached",
      undefined,
      inventoryRecord(record, false),
    );
  }
  return {
    record: scanRecord(started, completed, automaticKeyCount),
    evidence: canonicalEvidence(evidence),
  };
}

async function finalizeAction(
  state: DurableObjectState,
  completedAt: Date,
  input: ActionInput,
  inputHash: string,
  pending: PendingOperation,
  completed: CompletedAction,
): Promise<Response> {
  const control = normalizeControl(withCompletedOperation(
    completed.control,
    input,
    completed.status,
    completedAt,
  ));
  const operation = buildStoredOperation(
    completedAt,
    input,
    inputHash,
    completed.status,
    JSON.stringify(statusBody(control, completedAt, false)),
  );
  return state.storage.transaction(async (txn) => {
    const existing = await txn.get<unknown>(`${OPERATION_PREFIX}${input.operationId}`);
    if (existing !== undefined) return storedOperationResponse(normalizeOperation(existing));
    const current = normalizeControl(await txn.get<unknown>(CONTROL_KEY));
    if (!samePending(current.pendingOperation, pending)) {
      return Response.json(statusBody(current, completedAt), { status: 409 });
    }
    await txn.put(CONTROL_KEY, control);
    await txn.put(`${OPERATION_PREFIX}${input.operationId}`, operation);
    return storedOperationResponse(operation);
  });
}

async function finalizeFailure(
  state: DurableObjectState,
  failedAt: Date,
  input: ActionInput,
  inputHash: string,
  pending: PendingOperation,
  error: unknown,
): Promise<Response> {
  return state.storage.transaction(async (txn) => {
    const existing = await txn.get<unknown>(`${OPERATION_PREFIX}${input.operationId}`);
    if (existing !== undefined) return storedOperationResponse(normalizeOperation(existing));
    const current = normalizeControl(await txn.get<unknown>(CONTROL_KEY));
    if (!samePending(current.pendingOperation, pending)) {
      return Response.json(statusBody(current, failedAt), { status: 409 });
    }
    const failure = error instanceof CutoverFailure
      ? error
      : new CutoverFailure("control_state_invalid");
    const source = failure.control ?? current;
    const recoverPhase = effectivePhase(source);
    let blocked: CutoverControlRecord = {
      ...source,
      phase: "blocked",
      revision: current.revision + 1,
      updatedAt: failedAt.toISOString(),
      blocked: { code: failure.code, at: failedAt.toISOString() },
      recoverPhase,
      pendingOperation: undefined,
    };
    if (failure.inventory && input.action === "inventory") {
      blocked = {
        ...blocked,
        legacyAutoEvidence: {},
        preFenceInventory: failure.inventory,
        recoverPhase: "locked",
      };
    }
    blocked = withCompletedOperation(blocked, input, 503, failedAt);
    const operation = buildStoredOperation(
      failedAt,
      input,
      inputHash,
      503,
      JSON.stringify(statusBody(blocked, failedAt, false)),
    );
    await txn.put(CONTROL_KEY, blocked);
    await txn.put(`${OPERATION_PREFIX}${input.operationId}`, operation);
    return storedOperationResponse(operation);
  });
}

async function buildAuditMarker(
  env: Env,
  control: CutoverControlRecord,
  constructedAt: Date,
): Promise<DeliveryV3CutoverAuditMarker> {
  if (
    !control.preFenceInventory ||
    !control.providerFence ||
    !control.postFenceScan ||
    !control.followupScan
  ) {
    throw new CutoverFailure("control_state_invalid");
  }
  const core: Omit<DeliveryV3CutoverAuditMarker, "proof"> = {
    schemaVersion: 3,
    kind: "delivery-v2-cutover-audit" as const,
    proofVersion: 1 as const,
    providerFence: PROVIDER_FENCE_ATTESTATION,
    providerFencedAt: control.providerFence.attestedAt,
    inventoryStartedAt: control.preFenceInventory.startedAt,
    inventoryCompletedAt: control.preFenceInventory.completedAt,
    inventoryAutomaticKeyCount: control.preFenceInventory.automaticKeyCount,
    postFenceScanStartedAt: control.postFenceScan.startedAt,
    postFenceScanCompletedAt: control.postFenceScan.completedAt,
    postFenceAutomaticKeyCount: control.postFenceScan.automaticKeyCount,
    followupScanStartedAt: control.followupScan.startedAt,
    followupScanCompletedAt: control.followupScan.completedAt,
    followupAutomaticKeyCount: control.followupScan.automaticKeyCount,
    legacyAutoEvidenceSnapshot: "exact-canonical-map" as const,
    legacyAutoEvidence: canonicalEvidence(control.legacyAutoEvidence),
    constructedAt: constructedAt.toISOString(),
  };
  return {
    ...core,
    proof: await markerProof(env, core),
  };
}

async function readValidAuditMarker(env: Env): Promise<ValidMarker | undefined> {
  let raw: string | null;
  try {
    raw = await env.STORE.get(DELIVERY_V3_CUTOVER_AUDIT_KEY);
  } catch {
    return undefined;
  }
  if (!raw) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return undefined;
  }
  const marker = await normalizeAuditMarker(env, parsed);
  if (!marker || JSON.stringify(marker) !== raw) return undefined;
  return { marker, hash: await sha256Hex(raw) };
}

async function normalizeAuditMarker(
  env: Env,
  raw: unknown,
): Promise<DeliveryV3CutoverAuditMarker | undefined> {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const value = raw as Partial<DeliveryV3CutoverAuditMarker>;
  if (
    Object.keys(value).sort().join("|") !== [
      "constructedAt",
      "followupAutomaticKeyCount",
      "followupScanCompletedAt",
      "followupScanStartedAt",
      "inventoryAutomaticKeyCount",
      "inventoryCompletedAt",
      "inventoryStartedAt",
      "kind",
      "legacyAutoEvidence",
      "legacyAutoEvidenceSnapshot",
      "postFenceAutomaticKeyCount",
      "postFenceScanCompletedAt",
      "postFenceScanStartedAt",
      "proof",
      "proofVersion",
      "providerFence",
      "providerFencedAt",
      "schemaVersion",
    ].sort().join("|") ||
    value.schemaVersion !== 3 ||
    value.kind !== "delivery-v2-cutover-audit" ||
    value.proofVersion !== 1 ||
    value.providerFence !== PROVIDER_FENCE_ATTESTATION ||
    value.legacyAutoEvidenceSnapshot !== "exact-canonical-map" ||
    !validTimestamp(value.providerFencedAt) ||
    !validTimestamp(value.inventoryStartedAt) ||
    !validTimestamp(value.inventoryCompletedAt) ||
    !Number.isSafeInteger(value.inventoryAutomaticKeyCount) ||
    value.inventoryAutomaticKeyCount! < 0 ||
    value.inventoryAutomaticKeyCount! > MAX_LEGACY_AUTO_EVIDENCE ||
    !validTimestamp(value.postFenceScanStartedAt) ||
    !validTimestamp(value.postFenceScanCompletedAt) ||
    !validEvidenceCount(value.postFenceAutomaticKeyCount) ||
    !validTimestamp(value.followupScanStartedAt) ||
    !validTimestamp(value.followupScanCompletedAt) ||
    !validEvidenceCount(value.followupAutomaticKeyCount) ||
    !validTimestamp(value.constructedAt) ||
    !isHash(value.proof)
  ) {
    return undefined;
  }
  let evidence: ExactEvidence;
  try {
    evidence = normalizeExactEvidence(value.legacyAutoEvidence);
  } catch {
    return undefined;
  }
  const core: Omit<DeliveryV3CutoverAuditMarker, "proof"> = {
    schemaVersion: 3,
    kind: "delivery-v2-cutover-audit" as const,
    proofVersion: 1 as const,
    providerFence: PROVIDER_FENCE_ATTESTATION,
    providerFencedAt: value.providerFencedAt,
    inventoryStartedAt: value.inventoryStartedAt,
    inventoryCompletedAt: value.inventoryCompletedAt,
    inventoryAutomaticKeyCount: value.inventoryAutomaticKeyCount!,
    postFenceScanStartedAt: value.postFenceScanStartedAt,
    postFenceScanCompletedAt: value.postFenceScanCompletedAt,
    postFenceAutomaticKeyCount: value.postFenceAutomaticKeyCount!,
    followupScanStartedAt: value.followupScanStartedAt,
    followupScanCompletedAt: value.followupScanCompletedAt,
    followupAutomaticKeyCount: value.followupAutomaticKeyCount!,
    legacyAutoEvidenceSnapshot: "exact-canonical-map" as const,
    legacyAutoEvidence: evidence,
    constructedAt: value.constructedAt,
  };
  if (await markerProof(env, core) !== value.proof) return undefined;
  return { ...core, proof: value.proof };
}

async function markerProof(
  env: Env,
  core: Omit<DeliveryV3CutoverAuditMarker, "proof">,
): Promise<string> {
  return hmacSha256Hex(
    env.TOKEN_SECRET,
    `delivery-v3-cutover-audit\u0000${JSON.stringify(core)}`,
  );
}

function controlFromMarker(
  valid: ValidMarker,
  revision: number,
  now: Date,
): CutoverControlRecord {
  const marker = valid.marker;
  const inventory: InventoryRecord = {
    startedAt: marker.inventoryStartedAt,
    completedAt: marker.inventoryCompletedAt,
    automaticKeyCount: marker.inventoryAutomaticKeyCount,
    durationMs: Date.parse(marker.inventoryCompletedAt) - Date.parse(marker.inventoryStartedAt),
    capacity: MAX_LEGACY_AUTO_EVIDENCE,
    withinCapacity: true,
    durationBudgetMs: MAX_PREFLIGHT_SCAN_DURATION_MS,
    credentialRevocationDeadline: new Date(
      Date.parse(marker.inventoryCompletedAt) + INVENTORY_FRESHNESS_MS,
    ).toISOString(),
  };
  return {
    schemaVersion: 3,
    phase: "sealed",
    revision,
    updatedAt: now.toISOString(),
    legacyAutoEvidence: canonicalEvidence(marker.legacyAutoEvidence),
    preFenceInventory: inventory,
    providerFence: {
      attested: true,
      boundary: "old_resend_credential_revoked",
      attestedAt: marker.providerFencedAt,
    },
    postFenceScan: {
      startedAt: marker.postFenceScanStartedAt,
      completedAt: marker.postFenceScanCompletedAt,
      automaticKeyCount: marker.postFenceAutomaticKeyCount,
      durationMs: Date.parse(marker.postFenceScanCompletedAt) -
        Date.parse(marker.postFenceScanStartedAt),
    },
    followupScan: {
      startedAt: marker.followupScanStartedAt,
      completedAt: marker.followupScanCompletedAt,
      automaticKeyCount: marker.followupAutomaticKeyCount,
      durationMs: Date.parse(marker.followupScanCompletedAt) -
        Date.parse(marker.followupScanStartedAt),
    },
    markerAudit: {
      markerHash: valid.hash,
      writeCompletedAt: now.toISOString(),
      observations: [],
    },
  };
}

function statusResponse(control: CutoverControlRecord, now: Date): Response {
  return Response.json(statusBody(control, now), {
    status: control.phase === "blocked" ? 503 : 200,
  });
}

function statusBody(
  control: CutoverControlRecord,
  now: Date,
  includeDynamicSafety = true,
): Record<string, unknown> {
  const automaticReady = control.phase === "ready" &&
    control.pendingOperation?.action !== "repair";
  return {
    schemaVersion: 3,
    phase: control.phase,
    revision: control.revision,
    automatic: automaticReady ? "ready" : "locked",
    updatedAt: control.updatedAt,
    ...(control.preFenceInventory
      ? {
        preFenceInventory: {
          ...control.preFenceInventory,
          ...(includeDynamicSafety
            ? {
              safeBeforeCredentialRevocation:
                inventorySafeBeforeCredentialRevocation(
                  control.preFenceInventory,
                  now,
                ),
            }
            : {}),
        },
      }
      : {}),
    ...(control.providerFence ? { providerFence: control.providerFence } : {}),
    ...(control.postFenceScan ? { postFenceScan: control.postFenceScan } : {}),
    ...(control.followupScan ? { followupScan: control.followupScan } : {}),
    ...(control.markerAudit
      ? {
        markerAudit: {
          observations: control.markerAudit.observations.length,
          writeCompletedAt: control.markerAudit.writeCompletedAt,
          globalVisibilityClaimed: false,
        },
      }
      : {}),
    ...(control.readyAt ? { readyAt: control.readyAt } : {}),
    ...(control.blocked ? { blocked: { code: control.blocked.code } } : {}),
    ...(control.lastOperation ? { lastOperation: control.lastOperation } : {}),
    ...(control.pendingOperation
      ? {
        pendingOperation: {
          operationId: control.pendingOperation.operationId,
          action: control.pendingOperation.action,
          startedAt: control.pendingOperation.startedAt,
          ...(control.pendingOperation.attestedAt
            ? { attestedAt: control.pendingOperation.attestedAt }
            : {}),
        },
      }
      : {}),
  };
}

function lockedControl(now: Date): CutoverControlRecord {
  return {
    schemaVersion: 3,
    phase: "locked",
    revision: 0,
    updatedAt: now.toISOString(),
    legacyAutoEvidence: {},
  };
}

function normalizeControl(raw: unknown): CutoverControlRecord {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("cutover control state is invalid");
  }
  const value = raw as Partial<CutoverControlRecord>;
  if (
    value.schemaVersion !== 3 ||
    !isPhase(value.phase) ||
    !Number.isSafeInteger(value.revision) ||
    value.revision! < 0 ||
    !validTimestamp(value.updatedAt)
  ) {
    throw new Error("cutover control state is invalid");
  }
  value.legacyAutoEvidence = normalizeExactEvidence(value.legacyAutoEvidence);
  if (value.phase === "blocked") {
    if (
      !value.blocked ||
      !isBlockCode(value.blocked.code) ||
      !validTimestamp(value.blocked.at) ||
      !isRecoverPhase(value.recoverPhase)
    ) {
      throw new Error("cutover blocked state is invalid");
    }
  } else if (value.blocked !== undefined || value.recoverPhase !== undefined) {
    throw new Error("cutover non-blocked state is invalid");
  }
  if (value.pendingOperation) normalizePending(value.pendingOperation);
  if (value.lastOperation) normalizeLastOperation(value.lastOperation);
  const phase = effectivePhase(value as CutoverControlRecord);
  if (
    value.pendingOperation &&
    (value.pendingOperation.baseRevision !== value.revision ||
      value.pendingOperation.basePhase !== phase)
  ) {
    throw new Error("cutover pending operation binding is invalid");
  }
  if (phase !== "locked" && !validInventory(value.preFenceInventory)) {
    throw new Error("cutover inventory is invalid");
  }
  if (
    (phase === "observing" || phase === "sealed" || phase === "ready") &&
    (!validProviderFence(value.providerFence) || !validScan(value.postFenceScan))
  ) {
    throw new Error("cutover provider fence is invalid");
  }
  if (
    (phase === "sealed" || phase === "ready") &&
    (!validScan(value.followupScan) || !validMarkerAudit(value.markerAudit))
  ) {
    throw new Error("cutover marker audit is invalid");
  }
  if (phase === "ready" && !validTimestamp(value.readyAt)) {
    throw new Error("cutover ready state is invalid");
  }
  validateControlTiming(value as CutoverControlRecord, phase);
  return value as CutoverControlRecord;
}

function normalizeExactEvidence(raw: unknown): ExactEvidence {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("cutover exact evidence is invalid");
  }
  const entries = Object.entries(raw);
  if (
    entries.length > MAX_LEGACY_AUTO_EVIDENCE ||
    entries.some(([identity, status]) =>
      !isHash(identity) || (status !== "done" && status !== "attempted")
    )
  ) {
    throw new Error("cutover exact evidence is invalid");
  }
  const result = canonicalEvidence(Object.fromEntries(entries) as ExactEvidence);
  if (JSON.stringify(result) !== JSON.stringify(raw)) {
    throw new Error("cutover exact evidence is not canonical");
  }
  return result;
}

function canonicalEvidence(evidence: ExactEvidence): ExactEvidence {
  return Object.fromEntries(
    Object.entries(evidence).sort(([left], [right]) => left.localeCompare(right)),
  ) as ExactEvidence;
}

function validateControlTiming(
  control: CutoverControlRecord,
  phase: Exclude<CutoverPhase, "blocked">,
): void {
  if (phase === "locked" || phase === "inventoried") return;
  const fence = control.providerFence!;
  if (
    Date.parse(fence.attestedAt) < Date.parse(control.preFenceInventory!.completedAt) ||
    Date.parse(control.postFenceScan!.startedAt) < Date.parse(fence.attestedAt) ||
    Date.parse(control.postFenceScan!.completedAt) >=
      Date.parse(fence.attestedAt) + DELIVERY_V2_LEGACY_PENDING_TTL_MS
  ) {
    throw new Error("cutover provider fence timing is invalid");
  }
  if (phase === "observing") return;
  const followup = control.followupScan!;
  if (
    Date.parse(followup.startedAt) <
      Date.parse(fence.attestedAt) + DELIVERY_V2_KV_VISIBILITY_MS ||
    Date.parse(followup.completedAt) >=
      Date.parse(fence.attestedAt) + DELIVERY_V2_LEGACY_PENDING_TTL_MS ||
    Date.parse(control.markerAudit!.writeCompletedAt) < Date.parse(followup.completedAt)
  ) {
    throw new Error("cutover followup timing is invalid");
  }
  let base = Date.parse(control.markerAudit!.writeCompletedAt);
  for (const observation of control.markerAudit!.observations) {
    const observedAt = Date.parse(observation.observedAt);
    if (observedAt < base + MARKER_OBSERVATION_WAIT_MS) {
      throw new Error("cutover marker observation timing is invalid");
    }
    base = observedAt;
  }
  if (
    phase === "ready" &&
    (control.markerAudit!.observations.length < 2 ||
      Date.parse(control.readyAt!) < base)
  ) {
    throw new Error("cutover ready timing is invalid");
  }
}

function actionAllowed(
  action: CutoverAction,
  control: CutoverControlRecord,
  phase: Exclude<CutoverPhase, "blocked">,
): boolean {
  if (action === "inventory") {
    return control.providerFence === undefined &&
      (phase === "locked" || phase === "inventoried");
  }
  if (action === "provider-fence") {
    return control.phase !== "blocked" && phase === "inventoried";
  }
  if (action === "observe") return phase === "observing" || phase === "sealed";
  if (action === "seal") {
    return phase === "sealed" &&
      (control.markerAudit?.observations.length ?? 0) >= 2;
  }
  return action === "repair" &&
    (phase === "locked" || phase === "sealed" || phase === "ready");
}

function isMonotonicNoop(
  action: CutoverAction,
  control: CutoverControlRecord,
  phase: Exclude<CutoverPhase, "blocked">,
): boolean {
  if (control.phase === "blocked") return false;
  if (action === "inventory") {
    return phase === "observing" || phase === "sealed" || phase === "ready";
  }
  if (action === "provider-fence") {
    return phase === "observing" || phase === "sealed" || phase === "ready";
  }
  if (action === "observe") return phase === "ready";
  return action === "seal" && phase === "ready";
}

function observationTiming(
  control: CutoverControlRecord,
  now: Date,
): "ok" | "early" | "legacy-window-missed" {
  const phase = effectivePhase(control);
  if (phase === "sealed") {
    const observations = control.markerAudit!.observations;
    const base = observations.length === 0
      ? control.markerAudit!.writeCompletedAt
      : observations[observations.length - 1]!.observedAt;
    return now.getTime() < Date.parse(base) + MARKER_OBSERVATION_WAIT_MS
      ? "early"
      : "ok";
  }
  const fencedAt = Date.parse(control.providerFence!.attestedAt);
  if (now.getTime() < fencedAt + DELIVERY_V2_KV_VISIBILITY_MS) return "early";
  if (now.getTime() >= fencedAt + DELIVERY_V2_LEGACY_PENDING_TTL_MS) {
    return "legacy-window-missed";
  }
  return "ok";
}

function effectivePhase(
  control: CutoverControlRecord,
): Exclude<CutoverPhase, "blocked"> {
  if (control.phase !== "blocked") return control.phase;
  if (!control.recoverPhase) throw new Error("cutover recovery phase is invalid");
  return control.recoverPhase;
}

function clearBlocked(control: CutoverControlRecord): CutoverControlRecord {
  if (control.phase !== "blocked") return control;
  const { blocked: _blocked, recoverPhase, ...rest } = control;
  if (!recoverPhase) throw new Error("cutover recovery phase is invalid");
  return { ...rest, phase: recoverPhase } as CutoverControlRecord;
}

function scanRecord(started: Date, completed: Date, count: number): ScanRecord {
  return {
    startedAt: started.toISOString(),
    completedAt: completed.toISOString(),
    automaticKeyCount: count,
    durationMs: Math.max(0, completed.getTime() - started.getTime()),
  };
}

function inventoryRecord(record: ScanRecord, withinCapacity: boolean): InventoryRecord {
  return {
    ...record,
    capacity: MAX_LEGACY_AUTO_EVIDENCE,
    withinCapacity,
    durationBudgetMs: MAX_PREFLIGHT_SCAN_DURATION_MS,
    credentialRevocationDeadline: new Date(
      Date.parse(record.completedAt) + INVENTORY_FRESHNESS_MS,
    ).toISOString(),
  };
}

function inventorySafeBeforeCredentialRevocation(
  inventory: InventoryRecord | undefined,
  now: Date,
): boolean {
  return Boolean(
    inventory?.withinCapacity &&
      inventory.durationMs <= inventory.durationBudgetMs &&
      now.getTime() <= Date.parse(inventory.credentialRevocationDeadline),
  );
}

function withCompletedOperation(
  control: CutoverControlRecord,
  input: ActionInput,
  status: number,
  completedAt: Date,
): CutoverControlRecord {
  return {
    ...control,
    pendingOperation: undefined,
    legacyAutoEvidence: canonicalEvidence(control.legacyAutoEvidence),
    lastOperation: {
      operationId: input.operationId,
      action: input.action,
      status,
      completedAt: completedAt.toISOString(),
    },
  };
}

function buildStoredOperation(
  completedAt: Date,
  input: ActionInput,
  inputHash: string,
  status: number,
  body: string,
): StoredOperation {
  return {
    schemaVersion: 1,
    operationId: input.operationId,
    action: input.action,
    inputHash,
    status,
    body,
    completedAt: completedAt.toISOString(),
  };
}

function normalizeOperation(raw: unknown): StoredOperation {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("cutover operation is invalid");
  }
  const value = raw as Partial<StoredOperation>;
  if (
    value.schemaVersion !== 1 ||
    !isCutoverOperationId(value.operationId) ||
    !isAction(value.action) ||
    !isHash(value.inputHash) ||
    !Number.isSafeInteger(value.status) ||
    value.status! < 100 ||
    value.status! > 599 ||
    typeof value.body !== "string" ||
    !validTimestamp(value.completedAt)
  ) {
    throw new Error("cutover operation is invalid");
  }
  JSON.parse(value.body);
  return value as StoredOperation;
}

function normalizePending(raw: PendingOperation): PendingOperation {
  if (
    !isCutoverOperationId(raw.operationId) ||
    !isAction(raw.action) ||
    !isHash(raw.inputHash) ||
    !Number.isSafeInteger(raw.baseRevision) ||
    raw.baseRevision < 0 ||
    !isRecoverPhase(raw.basePhase) ||
    !validTimestamp(raw.startedAt) ||
    (raw.action === "provider-fence"
      ? !validTimestamp(raw.attestedAt) || raw.attestedAt !== raw.startedAt
      : raw.attestedAt !== undefined) ||
    (raw.recoverFromMarker !== undefined &&
      (raw.action !== "repair" || raw.recoverFromMarker !== true))
  ) {
    throw new Error("cutover pending operation is invalid");
  }
  return raw;
}

function normalizeLastOperation(raw: LastOperation): LastOperation {
  if (
    !isCutoverOperationId(raw.operationId) ||
    !isAction(raw.action) ||
    !Number.isSafeInteger(raw.status) ||
    raw.status < 100 ||
    raw.status > 599 ||
    !validTimestamp(raw.completedAt)
  ) {
    throw new Error("cutover last operation is invalid");
  }
  return raw;
}

function samePending(
  left: PendingOperation | undefined,
  right: PendingOperation,
): boolean {
  return Boolean(
    left &&
      left.operationId === right.operationId &&
      left.action === right.action &&
      left.inputHash === right.inputHash &&
      left.baseRevision === right.baseRevision &&
      left.basePhase === right.basePhase &&
      left.startedAt === right.startedAt &&
      left.attestedAt === right.attestedAt &&
      left.recoverFromMarker === right.recoverFromMarker,
  );
}

function validEvidenceCount(value: unknown): value is number {
  return Number.isSafeInteger(value) &&
    (value as number) >= 0 &&
    (value as number) <= MAX_LEGACY_AUTO_EVIDENCE;
}

function validScan(value: unknown): value is ScanRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const scan = value as Partial<ScanRecord>;
  return validTimestamp(scan.startedAt) &&
    validTimestamp(scan.completedAt) &&
    Date.parse(scan.completedAt) >= Date.parse(scan.startedAt) &&
    validEvidenceCount(scan.automaticKeyCount) &&
    Number.isSafeInteger(scan.durationMs) &&
    scan.durationMs ===
      Date.parse(scan.completedAt) - Date.parse(scan.startedAt);
}

function validInventory(value: unknown): value is InventoryRecord {
  if (!validScan(value)) return false;
  const inventory = value as InventoryRecord;
  return inventory.capacity === MAX_LEGACY_AUTO_EVIDENCE &&
    inventory.withinCapacity === true &&
    inventory.durationBudgetMs === MAX_PREFLIGHT_SCAN_DURATION_MS &&
    validTimestamp(inventory.credentialRevocationDeadline) &&
    Date.parse(inventory.credentialRevocationDeadline) ===
      Date.parse(inventory.completedAt) + INVENTORY_FRESHNESS_MS;
}

function validProviderFence(value: unknown): value is ProviderFenceRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const fence = value as Partial<ProviderFenceRecord>;
  return fence.attested === true &&
    fence.boundary === "old_resend_credential_revoked" &&
    validTimestamp(fence.attestedAt);
}

function validMarkerAudit(value: unknown): value is MarkerAuditRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const marker = value as Partial<MarkerAuditRecord>;
  return isHash(marker.markerHash) &&
    validTimestamp(marker.writeCompletedAt) &&
    Array.isArray(marker.observations) &&
    marker.observations.length <= 2 &&
    marker.observations.every((entry) => validTimestamp(entry?.observedAt));
}

function isAction(value: unknown): value is CutoverAction {
  return value === "inventory" ||
    value === "provider-fence" ||
    value === "observe" ||
    value === "seal" ||
    value === "repair";
}

function isPhase(value: unknown): value is CutoverPhase {
  return value === "locked" ||
    value === "inventoried" ||
    value === "observing" ||
    value === "sealed" ||
    value === "ready" ||
    value === "blocked";
}

function isRecoverPhase(value: unknown): value is Exclude<CutoverPhase, "blocked"> {
  return value === "locked" ||
    value === "inventoried" ||
    value === "observing" ||
    value === "sealed" ||
    value === "ready";
}

function isBlockCode(value: unknown): value is SafeBlock["code"] {
  return value === "legacy_scan_unavailable" ||
    value === "legacy_scan_unsupported_key" ||
    value === "legacy_evidence_capacity_reached" ||
    value === "legacy_pending_window_missed" ||
    value === "marker_write_unavailable" ||
    value === "marker_observation_unavailable" ||
    value === "marker_observation_mismatch" ||
    value === "control_state_invalid";
}

function activeOperationSet(state: DurableObjectState): Set<string> {
  let operations = activeOperations.get(state);
  if (!operations) {
    operations = new Set<string>();
    activeOperations.set(state, operations);
  }
  return operations;
}

function storedOperationResponse(operation: StoredOperation): Response {
  return new Response(operation.body, {
    status: operation.status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}

function invalidControlResponse(): Response {
  return Response.json(
    {
      schemaVersion: 3,
      phase: "blocked",
      automatic: "locked",
      blocked: { code: "control_state_invalid" },
    },
    { status: 503 },
  );
}

function deniedAutomatic(): Response {
  return Response.json({ authorized: false }, { status: 503 });
}

function controlStub(env: Env): DurableObjectStub {
  const namespace = env.DELIVER_GATE;
  if (!namespace) throw new Error("delivery cutover binding is missing");
  return namespace.get(namespace.idFromName(CUTOVER_CONTROL_OBJECT));
}

function methodNotAllowed(): Response {
  return safeError("method not allowed", 405);
}

function safeError(error: string, status: number): Response {
  return Response.json({ error }, { status });
}

function isHash(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function validTimestamp(value: unknown): value is string {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}

class CutoverFailure extends Error {
  constructor(
    readonly code: SafeBlock["code"],
    readonly control?: CutoverControlRecord,
    readonly inventory?: InventoryRecord,
  ) {
    super(code);
  }
}
