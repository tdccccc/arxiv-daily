import { hmacSha256Hex, sha256Hex } from "./crypto";
import {
  authenticateDevice,
  automaticRuntimeConfigured,
  DELIVERY_V2_KV_VISIBILITY_MS,
  DELIVERY_V2_LEGACY_PENDING_TTL_MS,
  DELIVERY_V3_CUTOVER_AUDIT_KEY,
  configuredBuildIdentity,
  configuredIdentitySecret,
  deletePendingByIdentity,
  DELIVERY_PROTOCOL_GENERATION,
  hashRecipientIdentity,
  peekPendingByIdentity,
  putDevice,
  scanLegacyAutoDeliveryEvidence,
  writeDeliveryV3CutoverAuditMarker,
  type AuthenticatedDevice,
  type DeliveryV3CutoverAuditMarker,
  type Env,
  type LegacyDeliveryEvidence,
  type PendingVerify,
} from "./kv";

export const CUTOVER_CONTROL_OBJECT = "delivery-cutover:v3";
export const CUTOVER_STATUS_PATH = "/cutover/status";
export const CUTOVER_ACTION_PATH = "/cutover/action";
export const CUTOVER_AUTOMATIC_PATH = "/cutover/automatic";
export const CUTOVER_ISSUE_DEVICE_PATH = "/cutover/issue-device";
export const PROVIDER_FENCE_ATTESTATION = "old-resend-credential-revoked";
export const MAX_LEGACY_AUTO_EVIDENCE = 512;
export const MAX_PREFLIGHT_SCAN_DURATION_MS = 30_000;
export const INVENTORY_FRESHNESS_MS = 5 * 60_000;

const CONTROL_KEY = "cutover-control:v3";
const BINDING_KEY = "cutover-binding:v1";
const STATE_INDEX_KEY = "cutover-state-index:v1";
const OPERATION_PREFIX = "cutover-operation:v3:";
const ISSUANCE_CLAIM_PREFIX = "issuance-claim:v1:";
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

interface CutoverBindingRecord {
  schemaVersion: 1;
  identitySecretFingerprint: string;
  buildIdentity: string;
  protocolGeneration: typeof DELIVERY_PROTOCOL_GENERATION;
  boundAt: string;
}

interface CutoverStateIndex {
  schemaVersion: 1;
  stateCreatedAt: string;
}

interface CutoverControlRecord {
  schemaVersion: 3;
  phase: CutoverPhase;
  revision: number;
  updatedAt: string;
  legacyAutoEvidence: ExactEvidence;
  identitySecretFingerprint?: string;
  buildIdentity?: string;
  protocolGeneration?: typeof DELIVERY_PROTOCOL_GENERATION;
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

interface IssuanceClaim {
  schemaVersion: 1;
  status: "claimed" | "issued";
  protocolGeneration: typeof DELIVERY_PROTOCOL_GENERATION;
  buildIdentity: string;
  readyGeneration: number;
  createdAt: string;
  pendingExpiresAt: string;
  recipientIdentity: string;
  pendingProof: string;
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

export interface PublicReadiness {
  protocolGeneration: typeof DELIVERY_PROTOCOL_GENERATION;
  buildIdentity: string;
  phase: CutoverPhase;
  automatic: "ready" | "locked";
  readyGeneration: number | null;
}

async function currentIdentitySecretFingerprint(env: Env): Promise<string | null> {
  const identitySecret = configuredIdentitySecret(env);
  if (!identitySecret) return null;
  return hmacSha256Hex(
    identitySecret,
    "delivery-v3-identity-secret-fingerprint\u0000v1",
  );
}

async function currentBinding(
  env: Env,
  boundAt: Date,
): Promise<CutoverBindingRecord | null> {
  const identitySecretFingerprint = await currentIdentitySecretFingerprint(env);
  const buildIdentity = configuredBuildIdentity(env);
  if (!identitySecretFingerprint || !buildIdentity) return null;
  return {
    schemaVersion: 1,
    identitySecretFingerprint,
    buildIdentity,
    protocolGeneration: DELIVERY_PROTOCOL_GENERATION,
    boundAt: boundAt.toISOString(),
  };
}

function normalizeBinding(raw: unknown): CutoverBindingRecord {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("cutover binding is invalid");
  }
  const value = raw as Partial<CutoverBindingRecord>;
  if (
    Object.keys(value).sort().join("|") !== [
      "boundAt",
      "buildIdentity",
      "identitySecretFingerprint",
      "protocolGeneration",
      "schemaVersion",
    ].sort().join("|") ||
    value.schemaVersion !== 1 ||
    !isHash(value.identitySecretFingerprint) ||
    !isConfiguredBuildIdentity(value.buildIdentity) ||
    value.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION ||
    !validTimestamp(value.boundAt)
  ) {
    throw new Error("cutover binding is invalid");
  }
  return value as CutoverBindingRecord;
}

function bindingMatches(
  stored: CutoverBindingRecord,
  current: CutoverBindingRecord | null,
): boolean {
  return Boolean(
    current &&
      stored.identitySecretFingerprint === current.identitySecretFingerprint &&
      stored.buildIdentity === current.buildIdentity &&
      stored.protocolGeneration === current.protocolGeneration,
  );
}

function normalizeStateIndex(raw: unknown): CutoverStateIndex {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("cutover state index is invalid");
  }
  const value = raw as Partial<CutoverStateIndex>;
  if (
    Object.keys(value).sort().join("|") !== "schemaVersion|stateCreatedAt" ||
    value.schemaVersion !== 1 ||
    !validTimestamp(value.stateCreatedAt)
  ) {
    throw new Error("cutover state index is invalid");
  }
  return value as CutoverStateIndex;
}

function controlMatchesBinding(
  control: CutoverControlRecord,
  binding: CutoverBindingRecord,
): boolean {
  return control.identitySecretFingerprint === binding.identitySecretFingerprint &&
    control.buildIdentity === binding.buildIdentity &&
    control.protocolGeneration === binding.protocolGeneration;
}

async function readValidatedBinding(
  state: DurableObjectState,
  env: Env,
): Promise<CutoverBindingRecord | null> {
  try {
    const [bindingRaw, indexRaw, markerState] = await Promise.all([
      state.storage.get<unknown>(BINDING_KEY),
      state.storage.get<unknown>(STATE_INDEX_KEY),
      readAuditMarkerBinding(env),
    ]);
    const binding = normalizeBinding(bindingRaw);
    normalizeStateIndex(indexRaw);
    return bindingMatches(binding, await currentBinding(env, new Date(0))) &&
        (!markerState.present ||
          (markerState.binding && markerMatchesBinding(markerState.binding, binding)))
      ? binding
      : null;
  } catch {
    return null;
  }
}

export async function fetchPublicReadiness(
  env: Env,
): Promise<PublicReadiness | null> {
  if (!automaticRuntimeConfigured(env)) return null;
  const buildIdentity = configuredBuildIdentity(env)!;
  let response: Response;
  try {
    response = await fetchCutoverStatus(env);
  } catch {
    return null;
  }
  if (response.status !== 200) return null;
  let body: {
    schemaVersion?: unknown;
    phase?: unknown;
    revision?: unknown;
    automatic?: unknown;
  };
  try {
    body = await response.json() as typeof body;
  } catch {
    return null;
  }
  if (
    body.schemaVersion !== 3 ||
    !isPhase(body.phase) ||
    !Number.isSafeInteger(body.revision) ||
    (body.revision as number) < 0 ||
    (body.automatic !== "ready" && body.automatic !== "locked")
  ) {
    return null;
  }
  const readyGeneration = body.phase === "ready" && body.automatic === "ready" &&
      Number.isSafeInteger(body.revision) && (body.revision as number) > 0
    ? body.revision as number
    : null;
  if (body.automatic === "ready" && readyGeneration === null) return null;
  return {
    protocolGeneration: DELIVERY_PROTOCOL_GENERATION,
    buildIdentity,
    phase: body.phase,
    automatic: body.automatic,
    readyGeneration,
  };
}

export async function issueReadyBoundDevice(
  env: Env,
  pendingIdentity: string,
): Promise<
  | { status: "issued"; token: string }
  | { status: "invalid" | "locked" | "unavailable" }
> {
  let response: Response;
  try {
    response = await controlStub(env).fetch(
      new Request(`https://cutover-control${CUTOVER_ISSUE_DEVICE_PATH}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pendingIdentity }),
      }),
    );
  } catch {
    return { status: "unavailable" };
  }
  let body: { status?: unknown; token?: unknown };
  try {
    body = await response.json() as typeof body;
  } catch {
    return { status: "unavailable" };
  }
  if (
    response.status === 200 &&
    body.status === "issued" &&
    typeof body.token === "string" &&
    /^[0-9a-f]{64}$/.test(body.token)
  ) {
    return { status: "issued", token: body.token };
  }
  if (response.status === 400 && body.status === "invalid") {
    return { status: "invalid" };
  }
  if (response.status === 503 && body.status === "locked") {
    return { status: "locked" };
  }
  return { status: "unavailable" };
}

export async function authorizeAutomaticDelivery(
  env: Env,
  input: {
    evidenceIdentity: string;
    deliveryGeneration?: 2;
    protocolGeneration?: number;
    buildIdentity?: string;
    readyGeneration?: number;
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
    return state.blockConcurrencyWhile(() => readStatus(state, env, now()));
  }
  if (path === CUTOVER_ACTION_PATH) {
    if (request.method !== "POST") return methodNotAllowed();
    return runAction(state, env, now, request);
  }
  if (path === CUTOVER_AUTOMATIC_PATH) {
    if (request.method !== "POST") return methodNotAllowed();
    return state.blockConcurrencyWhile(() => authorizeAutomatic(state, env, request));
  }
  if (path === CUTOVER_ISSUE_DEVICE_PATH) {
    if (request.method !== "POST") return methodNotAllowed();
    const pendingIdentity = await parseIssuanceIdentity(request);
    if (!pendingIdentity) return invalidIssuance();
    return state.blockConcurrencyWhile(() =>
      issueDevice(state, env, now, pendingIdentity)
    );
  }
  return undefined;
}

async function readStatus(
  state: DurableObjectState,
  env: Env,
  now: Date,
): Promise<Response> {
  const [
    bindingRaw,
    indexRaw,
    controlRaw,
    markerState,
    operationArtifacts,
    issuanceArtifacts,
  ] = await Promise.all([
    state.storage.get<unknown>(BINDING_KEY),
    state.storage.get<unknown>(STATE_INDEX_KEY),
    state.storage.get<unknown>(CONTROL_KEY),
    readAuditMarkerBinding(env),
    state.storage.list({ prefix: OPERATION_PREFIX, limit: 1 }),
    state.storage.list({ prefix: ISSUANCE_CLAIM_PREFIX, limit: 1 }),
  ]);
  if (bindingRaw === undefined) {
    if (
      indexRaw !== undefined ||
      controlRaw !== undefined ||
      markerState.present ||
      operationArtifacts.size > 0 ||
      issuanceArtifacts.size > 0
    ) return identityLockedStatusResponse();
    return await currentBinding(env, new Date(0))
      ? statusResponse(lockedControl(new Date(0)), now)
      : identityLockedStatusResponse();
  }
  let binding: CutoverBindingRecord;
  try {
    binding = normalizeBinding(bindingRaw);
    normalizeStateIndex(indexRaw);
  } catch {
    return identityLockedStatusResponse();
  }
  if (
    !bindingMatches(binding, await currentBinding(env, new Date(0))) ||
    (markerState.present &&
      (!markerState.binding || !markerMatchesBinding(markerState.binding, binding)))
  ) {
    return identityLockedStatusResponse();
  }
  if (controlRaw === undefined) return identityLockedStatusResponse();
  try {
    const control = normalizeControl(controlRaw);
    const phase = effectivePhase(control);
    const markerConsistent = phase === "sealed" || phase === "ready"
      ? await authoritativeMarkerMatchesControl(env, control)
      : !markerState.present;
    return controlMatchesBinding(control, binding) && markerConsistent
      ? statusResponse(control, now)
      : identityLockedStatusResponse();
  } catch {
    return invalidControlResponse();
  }
}

async function parseIssuanceIdentity(request: Request): Promise<string | null> {
  let body: Record<string, unknown>;
  try {
    body = await request.json() as Record<string, unknown>;
  } catch {
    return null;
  }
  return Object.keys(body).length === 1 && isHash(body.pendingIdentity)
    ? body.pendingIdentity
    : null;
}

async function issueDevice(
  state: DurableObjectState,
  env: Env,
  now: () => Date,
  pendingIdentity: string,
): Promise<Response> {
  if (!automaticRuntimeConfigured(env)) return lockedIssuance();
  const binding = await readValidatedBinding(state, env);
  if (!binding) return lockedIssuance();
  let control: CutoverControlRecord;
  try {
    control = normalizeControl(await state.storage.get<unknown>(CONTROL_KEY));
  } catch {
    return lockedIssuance();
  }
  const buildIdentity = configuredBuildIdentity(env);
  if (
    !buildIdentity ||
    !controlMatchesBinding(control, binding) ||
    control.phase !== "ready" ||
    !control.readyAt ||
    control.pendingOperation?.action === "repair" ||
    !await authoritativeMarkerMatchesControl(env, control)
  ) {
    return lockedIssuance();
  }

  const claimKey = `${ISSUANCE_CLAIM_PREFIX}${pendingIdentity}`;
  let claim: IssuanceClaim;
  const claimRaw = await state.storage.get<unknown>(claimKey);
  if (claimRaw === undefined) {
    const pending = await peekPendingByIdentity(env, pendingIdentity);
    if (!pending) return invalidIssuance();
    const claimedAt = now();
    if (claimedAt.getTime() >= Date.parse(pending.expiresAt)) {
      return invalidIssuance();
    }
    claim = {
      schemaVersion: 1,
      status: "claimed",
      protocolGeneration: DELIVERY_PROTOCOL_GENERATION,
      buildIdentity,
      readyGeneration: control.revision,
      createdAt: claimedAt.toISOString(),
      pendingExpiresAt: pending.expiresAt,
      recipientIdentity: await hashRecipientIdentity(
        pending.email,
        configuredIdentitySecret(env)!,
      ),
      pendingProof: await issuancePendingProof(
        env,
        pendingIdentity,
        pending,
      ),
    };
    await state.storage.put(claimKey, claim);
  } else {
    try {
      claim = normalizeIssuanceClaim(claimRaw);
    } catch {
      return unavailableIssuance();
    }
  }

  if (
    claim.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION ||
    claim.buildIdentity !== buildIdentity ||
    claim.readyGeneration !== control.revision
  ) {
    return lockedIssuance();
  }
  if (now().getTime() >= Date.parse(claim.pendingExpiresAt)) {
    return invalidIssuance();
  }

  const deviceToken = await deterministicDeviceToken(env, pendingIdentity);
  const existingDevice = await authenticateDevice(env, deviceToken);
  if (claim.status === "issued") {
    return validClaimDevice(existingDevice, claim)
      ? Response.json({ status: "issued", token: deviceToken })
      : unavailableIssuance();
  }

  const pending = await peekPendingByIdentity(env, pendingIdentity);
  if (pending &&
    await issuancePendingProof(env, pendingIdentity, pending) !==
      claim.pendingProof) {
    return unavailableIssuance();
  }
  if (existingDevice) {
    if (!validClaimDevice(existingDevice, claim)) return unavailableIssuance();
  } else {
    if (!pending) return unavailableIssuance();
    await putDevice(env, deviceToken, pending.email, {
      protocolGeneration: claim.protocolGeneration,
      buildIdentity: claim.buildIdentity,
      readyGeneration: claim.readyGeneration,
    }, new Date(claim.createdAt));
  }

  await deletePendingByIdentity(env, pendingIdentity);
  const issued: IssuanceClaim = { ...claim, status: "issued" };
  await state.storage.put(claimKey, issued);
  return Response.json({ status: "issued", token: deviceToken });
}

async function deterministicDeviceToken(
  env: Env,
  pendingIdentity: string,
): Promise<string> {
  return hmacSha256Hex(
    env.TOKEN_SECRET,
    `delivery-v2-device-token\u0000${pendingIdentity}`,
  );
}

async function issuancePendingProof(
  env: Env,
  pendingIdentity: string,
  pending: PendingVerify,
): Promise<string> {
  return hmacSha256Hex(
    env.TOKEN_SECRET,
    `delivery-v2-pending-proof\u0000${JSON.stringify([
      pendingIdentity,
      pending.email,
      pending.createdAt,
      pending.expiresAt,
    ])}`,
  );
}

function normalizeIssuanceClaim(raw: unknown): IssuanceClaim {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error("issuance claim is invalid");
  }
  const claim = raw as Partial<IssuanceClaim>;
  const expectedKeys = [
    "buildIdentity",
    "createdAt",
    "pendingExpiresAt",
    "pendingProof",
    "protocolGeneration",
    "readyGeneration",
    "recipientIdentity",
    "schemaVersion",
    "status",
  ];
  if (
    Object.keys(claim).sort().join("|") !== expectedKeys.sort().join("|") ||
    claim.schemaVersion !== 1 ||
    (claim.status !== "claimed" && claim.status !== "issued") ||
    claim.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION ||
    !isBuildIdentity(claim.buildIdentity) ||
    !Number.isSafeInteger(claim.readyGeneration) ||
    claim.readyGeneration! < 1 ||
    !validTimestamp(claim.createdAt) ||
    !validTimestamp(claim.pendingExpiresAt) ||
    Date.parse(claim.pendingExpiresAt!) <= Date.parse(claim.createdAt!) ||
    !isHash(claim.recipientIdentity) ||
    !isHash(claim.pendingProof)
  ) {
    throw new Error("issuance claim is invalid");
  }
  return claim as IssuanceClaim;
}

function validClaimDevice(
  device: AuthenticatedDevice | null,
  claim: IssuanceClaim,
): boolean {
  return Boolean(
    device &&
      device.recipientIdentity === claim.recipientIdentity &&
      device.createdAt === claim.createdAt &&
      device.deliveryGeneration === 2 &&
      device.protocolGeneration === claim.protocolGeneration &&
      device.buildIdentity === claim.buildIdentity &&
      device.readyGeneration === claim.readyGeneration,
  );
}

function invalidIssuance(): Response {
  return Response.json({ status: "invalid" }, { status: 400 });
}

function lockedIssuance(): Response {
  return Response.json({ status: "locked" }, { status: 503 });
}

function unavailableIssuance(): Response {
  return Response.json({ status: "unavailable" }, { status: 503 });
}

async function authorizeAutomatic(
  state: DurableObjectState,
  env: Env,
  request: Request,
): Promise<Response> {
  if (!automaticRuntimeConfigured(env)) return deniedAutomatic();
  let body: {
    evidenceIdentity?: unknown;
    deliveryGeneration?: unknown;
    protocolGeneration?: unknown;
    buildIdentity?: unknown;
    readyGeneration?: unknown;
  };
  try {
    body = await request.json() as typeof body;
  } catch {
    return deniedAutomatic();
  }
  if (
    !isHash(body.evidenceIdentity) ||
    body.deliveryGeneration !== 2 ||
    body.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION ||
    !isBuildIdentity(body.buildIdentity) ||
    !Number.isSafeInteger(body.readyGeneration) ||
    (body.readyGeneration as number) < 1
  ) {
    return deniedAutomatic();
  }
  const binding = await readValidatedBinding(state, env);
  if (!binding) return deniedAutomatic();
  let control: CutoverControlRecord;
  try {
    control = normalizeControl(await state.storage.get<unknown>(CONTROL_KEY));
  } catch {
    return deniedAutomatic();
  }
  const buildIdentity = configuredBuildIdentity(env);
  if (
    !buildIdentity ||
    !controlMatchesBinding(control, binding) ||
    control.phase !== "ready" ||
    !control.readyAt ||
    control.pendingOperation?.action === "repair" ||
    body.readyGeneration !== control.revision ||
    body.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION ||
    body.buildIdentity !== buildIdentity ||
    !await authoritativeMarkerMatchesControl(env, control)
  ) {
    return deniedAutomatic();
  }
  const legacyEvidence: LegacyDeliveryEvidence =
    (control.legacyAutoEvidence[body.evidenceIdentity] as
      | "done"
      | "attempted"
      | undefined) ?? "none";
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
  const expectedBinding = await currentBinding(env, now());
  if (!expectedBinding) return identityLockedStatusResponse();
  const inputHash = await sha256Hex(JSON.stringify([
    input.action,
    input.attestation ?? null,
  ]));
  const begun = await state.blockConcurrencyWhile(() =>
    beginAction(state, env, now(), input, inputHash, expectedBinding)
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
  env: Env,
  now: Date,
  input: ActionInput,
  inputHash: string,
  expectedBinding: CutoverBindingRecord,
): Promise<BeginResult> {
  const operationKey = `${OPERATION_PREFIX}${input.operationId}`;
  const markerState = await readAuditMarkerBinding(env);
  const [bindingRaw, indexRaw, controlRaw, operationRaw] = await Promise.all([
    state.storage.get<unknown>(BINDING_KEY),
    state.storage.get<unknown>(STATE_INDEX_KEY),
    state.storage.get<unknown>(CONTROL_KEY),
    state.storage.get<unknown>(operationKey),
  ]);

  if (bindingRaw === undefined) {
    if (
      indexRaw !== undefined ||
      controlRaw !== undefined ||
      operationRaw !== undefined ||
      markerState.present ||
      input.action !== "inventory"
    ) {
      return { kind: "response", response: identityLockedStatusResponse() };
    }
    const pending: PendingOperation = {
      operationId: input.operationId,
      action: "inventory",
      inputHash,
      baseRevision: 0,
      basePhase: "locked",
      startedAt: now.toISOString(),
    };
    const pendingControl: CutoverControlRecord = {
      ...lockedControl(now),
      identitySecretFingerprint: expectedBinding.identitySecretFingerprint,
      buildIdentity: expectedBinding.buildIdentity,
      protocolGeneration: expectedBinding.protocolGeneration,
      pendingOperation: pending,
    };
    const stateIndex: CutoverStateIndex = {
      schemaVersion: 1,
      stateCreatedAt: now.toISOString(),
    };
    const created = await state.storage.transaction(async (txn) => {
      const [
        bindingCheck,
        indexCheck,
        controlCheck,
        operationArtifacts,
        issuanceArtifacts,
      ] = await Promise.all([
        txn.get<unknown>(BINDING_KEY),
        txn.get<unknown>(STATE_INDEX_KEY),
        txn.get<unknown>(CONTROL_KEY),
        txn.list({ prefix: OPERATION_PREFIX, limit: 1 }),
        txn.list({ prefix: ISSUANCE_CLAIM_PREFIX, limit: 1 }),
      ]);
      if (
        bindingCheck !== undefined ||
        indexCheck !== undefined ||
        controlCheck !== undefined ||
        operationArtifacts.size > 0 ||
        issuanceArtifacts.size > 0
      ) return false;
      await txn.put(BINDING_KEY, expectedBinding);
      await txn.put(STATE_INDEX_KEY, stateIndex);
      await txn.put(CONTROL_KEY, pendingControl);
      return true;
    });
    if (!created) return { kind: "response", response: identityLockedStatusResponse() };
    activeOperationSet(state).add(input.operationId);
    return { kind: "execute", control: pendingControl, pending };
  }

  let binding: CutoverBindingRecord;
  try {
    binding = normalizeBinding(bindingRaw);
    normalizeStateIndex(indexRaw);
  } catch {
    return { kind: "response", response: identityLockedStatusResponse() };
  }
  if (
    !bindingMatches(binding, expectedBinding) ||
    (markerState.present &&
      (!markerState.binding || !markerMatchesBinding(markerState.binding, binding)))
  ) {
    return { kind: "response", response: identityLockedStatusResponse() };
  }

  let control: CutoverControlRecord;
  let recoverFromMarker = false;
  if (controlRaw === undefined) {
    if (input.action !== "repair" || !markerState.binding) {
      return { kind: "response", response: identityLockedStatusResponse() };
    }
    control = {
      ...lockedControl(now),
      identitySecretFingerprint: binding.identitySecretFingerprint,
      buildIdentity: binding.buildIdentity,
      protocolGeneration: binding.protocolGeneration,
    };
    recoverFromMarker = true;
  } else {
    try {
      control = normalizeControl(controlRaw);
      if (!controlMatchesBinding(control, binding)) {
        return { kind: "response", response: identityLockedStatusResponse() };
      }
      recoverFromMarker = control.pendingOperation?.recoverFromMarker === true;
    } catch {
      if (input.action !== "repair" || !markerState.binding) {
        return { kind: "response", response: invalidControlResponse() };
      }
      control = {
        ...lockedControl(now),
        identitySecretFingerprint: binding.identitySecretFingerprint,
        buildIdentity: binding.buildIdentity,
        protocolGeneration: binding.protocolGeneration,
      };
      recoverFromMarker = true;
    }
  }

  const markerPhase = effectivePhase(control);
  if (
    ((markerPhase === "sealed" || markerPhase === "ready") &&
      (markerState.present
        ? !await authoritativeMarkerMatchesControl(env, control)
        : input.action !== "repair")) ||
    ((markerPhase === "locked" || markerPhase === "inventoried" ||
      markerPhase === "observing") && markerState.present &&
      !(recoverFromMarker && input.action === "repair"))
  ) {
    return { kind: "response", response: identityLockedStatusResponse() };
  }

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
        identitySecretFingerprint: control.identitySecretFingerprint,
        buildIdentity: control.buildIdentity,
        protocolGeneration: control.protocolGeneration,
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
          ...(control.identitySecretFingerprint && control.buildIdentity &&
              control.protocolGeneration === DELIVERY_PROTOCOL_GENERATION
            ? {
              identitySecretFingerprint: control.identitySecretFingerprint,
              buildIdentity: control.buildIdentity,
              protocolGeneration: control.protocolGeneration,
            }
            : {}),
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
    !control.followupScan ||
    !isHash(control.identitySecretFingerprint) ||
    !isConfiguredBuildIdentity(control.buildIdentity) ||
    control.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION ||
    await currentIdentitySecretFingerprint(env) !== control.identitySecretFingerprint ||
    configuredBuildIdentity(env) !== control.buildIdentity
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
    identitySecretFingerprint: control.identitySecretFingerprint!,
    buildIdentity: control.buildIdentity!,
    protocolGeneration: control.protocolGeneration!,
    constructedAt: constructedAt.toISOString(),
  };
  return {
    ...core,
    proof: await markerProof(env, core),
  };
}

interface AuditMarkerBindingState {
  present: boolean;
  binding?: Pick<
    CutoverBindingRecord,
    "identitySecretFingerprint" | "buildIdentity" | "protocolGeneration"
  >;
}

async function readAuditMarkerBinding(
  env: Env,
): Promise<AuditMarkerBindingState> {
  try {
    const raw = await env.STORE.get(DELIVERY_V3_CUTOVER_AUDIT_KEY);
    if (!raw) return { present: false };
    const parsed = JSON.parse(raw) as unknown;
    const marker = await normalizeAuditMarker(env, parsed);
    return marker && JSON.stringify(marker) === raw
      ? {
        present: true,
        binding: {
          identitySecretFingerprint: marker.identitySecretFingerprint,
          buildIdentity: marker.buildIdentity,
          protocolGeneration: marker.protocolGeneration,
        },
      }
      : { present: true };
  } catch {
    return { present: true };
  }
}

function markerMatchesBinding(
  marker: Pick<
    CutoverBindingRecord,
    "identitySecretFingerprint" | "buildIdentity" | "protocolGeneration"
  >,
  binding: CutoverBindingRecord,
): boolean {
  return marker.identitySecretFingerprint === binding.identitySecretFingerprint &&
    marker.buildIdentity === binding.buildIdentity &&
    marker.protocolGeneration === binding.protocolGeneration;
}

async function authoritativeMarkerMatchesControl(
  env: Env,
  control: CutoverControlRecord,
): Promise<boolean> {
  const marker = await readValidAuditMarker(env);
  return Boolean(
    marker &&
      control.markerAudit &&
      marker.hash === control.markerAudit.markerHash,
  );
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
      "buildIdentity",
      "constructedAt",
      "followupAutomaticKeyCount",
      "followupScanCompletedAt",
      "followupScanStartedAt",
      "inventoryAutomaticKeyCount",
      "inventoryCompletedAt",
      "inventoryStartedAt",
      "identitySecretFingerprint",
      "kind",
      "legacyAutoEvidence",
      "legacyAutoEvidenceSnapshot",
      "postFenceAutomaticKeyCount",
      "postFenceScanCompletedAt",
      "postFenceScanStartedAt",
      "proof",
      "proofVersion",
      "protocolGeneration",
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
    !isHash(value.identitySecretFingerprint) ||
    !isConfiguredBuildIdentity(value.buildIdentity) ||
    value.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION ||
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
    identitySecretFingerprint: value.identitySecretFingerprint,
    buildIdentity: value.buildIdentity,
    protocolGeneration: value.protocolGeneration,
    constructedAt: value.constructedAt,
  };
  if (
    await currentIdentitySecretFingerprint(env) !== core.identitySecretFingerprint ||
    configuredBuildIdentity(env) !== core.buildIdentity ||
    core.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION ||
    await markerProof(env, core) !== value.proof
  ) return undefined;
  return { ...core, proof: value.proof };
}

async function markerProof(
  env: Env,
  core: Omit<DeliveryV3CutoverAuditMarker, "proof">,
): Promise<string> {
  const identitySecret = configuredIdentitySecret(env);
  if (!identitySecret) throw new Error("identity secret is unavailable");
  return hmacSha256Hex(
    identitySecret,
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
    identitySecretFingerprint: marker.identitySecretFingerprint,
    buildIdentity: marker.buildIdentity,
    protocolGeneration: marker.protocolGeneration,
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
  const hasBoundState = value.phase !== "locked" ||
    value.preFenceInventory !== undefined ||
    value.pendingOperation?.action === "inventory";
  if (
    (value.identitySecretFingerprint !== undefined &&
      !isHash(value.identitySecretFingerprint)) ||
    (value.buildIdentity !== undefined &&
      !isConfiguredBuildIdentity(value.buildIdentity)) ||
    (value.protocolGeneration !== undefined &&
      value.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION) ||
    (hasBoundState &&
      (!isHash(value.identitySecretFingerprint) ||
        !isConfiguredBuildIdentity(value.buildIdentity) ||
        value.protocolGeneration !== DELIVERY_PROTOCOL_GENERATION)) ||
    (!hasBoundState &&
      ((value.identitySecretFingerprint === undefined) !==
        (value.buildIdentity === undefined) ||
        (value.identitySecretFingerprint === undefined) !==
          (value.protocolGeneration === undefined)))
  ) {
    throw new Error("cutover identity binding is invalid");
  }
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
      (phase === "inventoried" ||
        (phase === "locked" &&
          (control.phase === "blocked" ||
            control.identitySecretFingerprint === undefined)));
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

function identityLockedStatusResponse(): Response {
  return Response.json(
    { schemaVersion: 3, phase: "locked", automatic: "locked" },
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

function isConfiguredBuildIdentity(value: unknown): value is string {
  return typeof value === "string" &&
    /^email-relay-v2-[0-9a-f]{40}$/.test(value);
}

function isBuildIdentity(value: unknown): value is string {
  return typeof value === "string" &&
    /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value);
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
