import { afterEach, describe, expect, it, vi } from "vitest";
import worker from "../src/index";
import { DeliverGate } from "../src/deliver-gate";
import {
  fetchCutoverStatus,
  postCutoverAction,
  type CutoverAction,
} from "../src/cutover-control";
import { sha256Hex } from "../src/crypto";
import {
  hashDeviceToken,
  hashLegacyAutoDeliveryIdentity,
  putDevice,
  putPending,
  scanLegacyAutoDeliveryEvidence,
  type Env,
} from "../src/kv";

const AUTO_KEY_A = `arxiv-daily:auto:${"a".repeat(64)}`;
const AUTO_KEY_B = `arxiv-daily:auto:${"b".repeat(64)}`;
const TEST_KEY_A = `arxiv-daily:test:${"c".repeat(32)}`;
const TEST_KEY_B = `arxiv-daily:test:${"d".repeat(32)}`;
const DELIVERY_V2_CUTOVER_KEY = "cutover:delivery-v2";
const DELIVERY_V3_CUTOVER_AUDIT_KEY = "cutover:delivery-v3-audit";

type MemoryDurable = ReturnType<typeof durableState>;

function memoryKv() {
  const map = new Map<string, string>();
  const failGet = new Set<string>();
  const hideFromList = new Set<string>();
  let failList = false;
  let pageSize = Number.POSITIVE_INFINITY;
  let beforeList: (() => Promise<void>) | undefined;
  return {
    async get(key: string) {
      if (failGet.has(key)) throw new Error("injected KV read marker");
      return map.get(key) ?? null;
    },
    async put(key: string, value: string) {
      map.set(key, value);
    },
    async delete(key: string) {
      map.delete(key);
    },
    async list(options: { prefix?: string; cursor?: string } = {}) {
      if (failList) throw new Error("injected KV list failure");
      await beforeList?.();
      const prefix = options.prefix ?? "";
      const offset = options.cursor ? Number(options.cursor) : 0;
      const names = Array.from(map.keys())
        .filter((key) => key.startsWith(prefix) && !hideFromList.has(key))
        .sort();
      const page = names.slice(offset, offset + pageSize);
      const nextOffset = offset + page.length;
      const listComplete = nextOffset >= names.length;
      return {
        keys: page.map((name) => ({ name })),
        list_complete: listComplete,
        ...(listComplete ? {} : { cursor: String(nextOffset) }),
        cacheStatus: null,
      };
    },
    _map: map,
    _failGet: failGet,
    _hideFromList: hideFromList,
    set _failList(value: boolean) {
      failList = value;
    },
    set _pageSize(value: number) {
      pageSize = value;
    },
    set _beforeList(value: (() => Promise<void>) | undefined) {
      beforeList = value;
    },
  };
}

function durableState(options: { failDonePut?: boolean } = {}) {
  let tail: Promise<unknown> = Promise.resolve();
  const records = new Map<string, unknown>();
  let alarm: number | null = null;
  const storage = {
    async get<T>(key: string): Promise<T | undefined> {
      return records.get(key) as T | undefined;
    },
    async put(key: string, value: unknown): Promise<void> {
      if (
        options.failDonePut &&
        value &&
        typeof value === "object" &&
        (value as { status?: unknown }).status === "done"
      ) {
        throw new Error("completion storage unavailable");
      }
      records.set(key, structuredClone(value));
    },
    async delete(key: string): Promise<boolean> {
      return records.delete(key);
    },
    async transaction<T>(callback: (txn: unknown) => Promise<T>): Promise<T> {
      const snapshot = new Map(records);
      const txn = {
        get: storage.get,
        put: storage.put,
        delete: storage.delete,
      };
      try {
        return await callback(txn);
      } catch (error) {
        records.clear();
        for (const [key, value] of snapshot) records.set(key, value);
        throw error;
      }
    },
    async setAlarm(value: number): Promise<void> {
      alarm = value;
    },
    async getAlarm(): Promise<number | null> {
      return alarm;
    },
  };
  const state = {
    storage,
    blockConcurrencyWhile<T>(callback: () => Promise<T>): Promise<T> {
      const next = tail.then(callback, callback);
      tail = next.catch(() => undefined);
      return next;
    },
  } as unknown as DurableObjectState;
  return { state, records, storage };
}

class MemoryNamespace {
  readonly names: string[] = [];
  readonly durables = new Map<string, MemoryDurable>();
  readonly gates = new Map<string, DeliverGate>();
  now = new Date("2026-08-10T00:00:00.000Z");

  constructor(
    private readonly env: Env,
    private readonly stateOptions: { failDonePut?: boolean } = {},
  ) {}

  idFromName(name: string): DurableObjectId {
    this.names.push(name);
    return { toString: () => name } as DurableObjectId;
  }

  get(id: DurableObjectId): DurableObjectStub {
    const name = id.toString();
    let gate = this.gates.get(name);
    if (!gate) {
      const durable = durableState(this.stateOptions);
      this.durables.set(name, durable);
      gate = new DeliverGate(durable.state, this.env, () => this.now);
      this.gates.set(name, gate);
    }
    return { fetch: (request: Request) => gate!.fetch(request) } as DurableObjectStub;
  }
}

function envWith(
  kv: ReturnType<typeof memoryKv>,
  options: { quota?: number; failDonePut?: boolean; binding?: boolean } = {},
): Env & { _namespace?: MemoryNamespace } {
  kv._map.set(DELIVERY_V2_CUTOVER_KEY, JSON.stringify(cutoverMarker()));
  const env: Env & { _namespace?: MemoryNamespace } = {
    STORE: kv as unknown as KVNamespace,
    RESEND_API_KEY: "re_test",
    TOKEN_SECRET: "secret",
    PUBLIC_BASE_URL: "https://example.com",
    FROM_EMAIL: "daily@mail.arxiv-daily.top",
    FROM_NAME: "arXiv Daily",
    DAILY_QUOTA: String(options.quota ?? 5),
  };
  if (options.binding !== false) {
    const namespace = new MemoryNamespace(env, {
      failDonePut: options.failDonePut,
    });
    env.DELIVER_GATE = namespace as unknown as DurableObjectNamespace;
    env._namespace = namespace;
  }
  return env;
}

function cutoverMarker(
  legacyEvidence: Record<string, "done" | "attempted"> = {},
) {
  return {
    schemaVersion: 2,
    oldWorkerWritesQuiesced: true,
    legacyAutoEvidenceSnapshot: "positive-evidence-only",
    preQuiesceScanStartedAt: "2026-08-01T00:00:00.000Z",
    preQuiesceScanCompletedAt: "2026-08-01T00:00:01.000Z",
    oldWorkerWritesQuiescedAt: "2026-08-01T00:00:02.000Z",
    postQuiesceScanStartedAt: "2026-08-01T00:01:02.000Z",
    postQuiesceScanCompletedAt: "2026-08-01T00:01:03.000Z",
    enabledAt: "2026-08-01T00:02:02.000Z",
    legacyAutoEvidence: legacyEvidence,
  };
}

async function setLegacyEvidence(
  kv: ReturnType<typeof memoryKv>,
  legacyValue: string | undefined,
): Promise<void> {
  const date = "2026-08-10";
  const recipient = "recipient@example.com";
  const logicalKey = `arxiv-daily:auto:${await sha256Hex(`${date}\u0000${recipient}`)}`;
  if (legacyValue === undefined) {
    kv._map.delete(`idemp:${logicalKey}`);
  } else {
    kv._map.set(`idemp:${logicalKey}`, legacyValue);
  }
}

function deliveryRequest(input: {
  token?: string;
  key?: string;
  to?: string;
  date?: string;
  subject?: string;
  html?: string;
  text?: string;
} = {}): Request {
  return new Request("https://relay.test/v1/deliver", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${input.token ?? "device-token"}`,
      "Content-Type": "application/json",
      "Idempotency-Key": input.key ?? AUTO_KEY_A,
    },
    body: JSON.stringify({
      to: input.to ?? "recipient@example.com",
      date: input.date ?? "2026-08-10",
      subject: input.subject ?? "Daily",
      html: input.html ?? "<p>daily</p>",
      text: input.text ?? "daily",
    }),
  });
}

function deviceDurables(env: Env & { _namespace?: MemoryNamespace }): MemoryDurable[] {
  return Array.from(env._namespace!.durables.entries())
    .filter(([name]) => name.startsWith("device-v2:"))
    .map(([, durable]) => durable);
}

async function cutoverOperationId(
  action: CutoverAction,
  revision: number,
): Promise<string> {
  return sha256Hex(JSON.stringify(["test-cutover-v3", action, revision]));
}

async function applyCutoverAction(
  env: Env,
  action: CutoverAction,
  revision: number,
): Promise<Response> {
  return postCutoverAction(
    env,
    action,
    await cutoverOperationId(action, revision),
    action === "provider-fence" ? "old-resend-credential-revoked" : undefined,
  );
}

async function completeCutover(
  env: Env & { _namespace?: MemoryNamespace },
): Promise<void> {
  expect((await applyCutoverAction(env, "inventory", 0)).status).toBe(200);
  expect((await applyCutoverAction(env, "provider-fence", 1)).status).toBe(200);
  env._namespace!.now = new Date(env._namespace!.now.getTime() + 60 * 1000);
  expect((await applyCutoverAction(env, "observe", 2)).status).toBe(200);
  env._namespace!.now = new Date(env._namespace!.now.getTime() + 60 * 1000);
  expect((await applyCutoverAction(env, "observe", 3)).status).toBe(200);
  env._namespace!.now = new Date(env._namespace!.now.getTime() + 60 * 1000);
  expect((await applyCutoverAction(env, "observe", 4)).status).toBe(200);
  expect((await applyCutoverAction(env, "seal", 5)).status).toBe(200);
  const status = await fetchCutoverStatus(env);
  expect(status.status).toBe(200);
  expect(await status.json()).toMatchObject({ phase: "ready", automatic: "ready" });
}

async function authenticatedEnv(
  options: {
    quota?: number;
    failDonePut?: boolean;
    binding?: boolean;
    stageCutover?: boolean;
    legacyValue?: string;
  } = {},
) {
  const kv = memoryKv();
  const env = envWith(kv, options);
  if (options.legacyValue) {
    await setLegacyEvidence(kv, options.legacyValue);
  }
  if (options.binding !== false && options.stageCutover !== false) {
    await completeCutover(env);
    await putDevice(
      env,
      "device-token",
      "recipient@example.com",
      new Date(env._namespace!.now.getTime() + 1),
    );
  } else {
    await putDevice(env, "device-token", "recipient@example.com");
  }
  return { kv, env };
}

function providerSuccess(id = "msg_ok") {
  return vi.fn(async (_input: string, init?: RequestInit) => {
    const key = (init?.headers as Record<string, string>)["Idempotency-Key"];
    expect(key).toMatch(/^arxiv-daily:relay:v2:(auto|test):[0-9a-f]{64}$/);
    expect(key.length).toBeLessThanOrEqual(128);
    expect(key).not.toContain("recipient@example.com");
    expect(key).not.toContain("device-token");
    return new Response(JSON.stringify({ id }), { status: 200 });
  });
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("legacy automatic evidence scanner", () => {
  it("recognizes both production automatic key generations across pages and ignores supported tests", async () => {
    const kv = memoryKv();
    kv._pageSize = 2;
    const env = envWith(kv, { binding: false });
    const plainKey = "idemp:2026-08-10|recipient@example.com";
    const hashedKey = `idemp:arxiv-daily:auto:${"a".repeat(64)}`;
    kv._map.set(plainKey, "done:private-provider-id");
    kv._map.set(hashedKey, "pending:private-claim");
    kv._map.set(
      "idemp:test|2026-08-10|recipient@example.com|2026-08-10T00:00:00.000Z",
      "done:private-test-id",
    );
    kv._map.set(`idemp:arxiv-daily:test:${"b".repeat(32)}`, "done:private-test-id");

    const evidence = await scanLegacyAutoDeliveryEvidence(env);

    expect(Object.values(evidence).sort()).toEqual(["attempted", "done"]);
    expect(JSON.stringify(evidence)).not.toContain("recipient@example.com");
    expect(JSON.stringify(evidence)).not.toContain(plainKey);
    expect(JSON.stringify(evidence)).not.toContain(hashedKey);
  });

  it("fails closed on an unknown idemp key without exposing its raw key or recipient", async () => {
    const kv = memoryKv();
    const env = envWith(kv, { binding: false });
    const unknown = "idemp:private-recipient@example.com|unsupported";
    kv._map.set(unknown, "private-value");

    let message = "";
    try {
      await scanLegacyAutoDeliveryEvidence(env);
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }

    expect(message).toBe("legacy automatic delivery scan encountered an unsupported key");
    expect(message).not.toContain("private-recipient@example.com");
    expect(message).not.toContain(unknown);
  });
});

describe("cutover v3 safety contract", () => {
  const TOKEN = "cutover-v3-token";
  const ATTESTATION = "old-resend-credential-revoked";

  function operationId(n: number): string {
    return n.toString(16).padStart(64, "0");
  }

  function controlRequest(input: {
    method?: "GET" | "POST";
    token?: string;
    action?: string;
    operationId?: string;
    attestation?: string;
  } = {}): Request {
    const method = input.method ?? "POST";
    return new Request("https://relay.test/internal/delivery-v2/cutover", {
      method,
      headers: {
        Authorization: `Bearer ${input.token ?? TOKEN}`,
        ...(method === "POST" ? { "Content-Type": "application/json" } : {}),
      },
      ...(method === "POST"
        ? {
          body: JSON.stringify({
            action: input.action,
            operationId: input.operationId,
            ...(input.attestation ? { attestation: input.attestation } : {}),
          }),
        }
        : {}),
    });
  }

  async function apply(
    env: Env,
    action: string,
    id: number,
    attestation?: string,
  ): Promise<Response> {
    return worker.fetch(controlRequest({
      action,
      operationId: operationId(id),
      attestation,
    }), env);
  }

  async function readControl(env: Env): Promise<Record<string, unknown>> {
    const response = await worker.fetch(controlRequest({ method: "GET" }), env);
    expect([200, 503]).toContain(response.status);
    return await response.json() as Record<string, unknown>;
  }

  it("hides unauthenticated control requests and rejects client marker material", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;

    for (const request of [
      controlRequest({ method: "GET", token: "wrong" }),
      controlRequest({
        token: "wrong",
        action: "inventory",
        operationId: operationId(230),
      }),
    ]) {
      const response = await worker.fetch(request, env);
      expect(response.status).toBe(404);
      expect(await response.json()).toEqual({ error: "not found" });
    }

    const privateValue = "private-recipient@example.com";
    const invalidId = await worker.fetch(controlRequest({
      action: "inventory",
      operationId: privateValue,
    }), env);
    expect(invalidId.status).toBe(400);
    expect(await invalidId.text()).not.toContain(privateValue);

    const suppliedMarker = new Request(
      "https://relay.test/internal/delivery-v2/cutover",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          action: "inventory",
          operationId: operationId(231),
          marker: { recipient: privateValue },
        }),
      },
    );
    const rejected = await worker.fetch(suppliedMarker, env);
    expect(rejected.status).toBe(400);
    expect(await rejected.text()).not.toContain(privateValue);
  });

  it("defaults automatic to locked and ignores an old KV marker or old DO proof", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    await putDevice(env, "device-token", "recipient@example.com");
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    expect(await readControl(env)).toMatchObject({ phase: "locked", automatic: "locked" });
    expect((await worker.fetch(deliveryRequest(), env)).status).toBe(503);
    expect(provider).not.toHaveBeenCalled();

    const control = env._namespace!.get(
      env._namespace!.idFromName("delivery-cutover:v3"),
    );
    const durable = env._namespace!.durables.get("delivery-cutover:v3")!;
    durable.records.set("cutover-control:v3", {
      schemaVersion: 2,
      phase: "ready",
      readyAt: "2026-08-01T00:00:00.000Z",
    });
    expect((await control.fetch(new Request("https://cutover-control/cutover/status"))).status)
      .toBe(503);
    expect((await worker.fetch(deliveryRequest(), env)).status).toBe(503);
    expect(provider).not.toHaveBeenCalled();
  });

  it("requires inventory then explicit old-provider credential revocation attestation", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    await putDevice(env, "device-token", "recipient@example.com");
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const inventory = await apply(env, "inventory", 1);
    expect(inventory.status).toBe(200);
    expect(await inventory.json()).toMatchObject({ phase: "inventoried", automatic: "locked" });

    const missing = await apply(env, "provider-fence", 2);
    expect(missing.status).toBe(400);
    const wrong = await apply(env, "provider-fence", 3, "new-key-configured");
    expect(wrong.status).toBe(400);

    const fenced = await apply(env, "provider-fence", 4, ATTESTATION);
    expect(fenced.status).toBe(200);
    expect(await fenced.json()).toMatchObject({
      phase: "observing",
      automatic: "locked",
      providerFence: {
        attested: true,
        boundary: "old_resend_credential_revoked",
      },
    });
    expect((await worker.fetch(deliveryRequest(), env)).status).toBe(503);
    expect(provider).not.toHaveBeenCalled();
  });

  it("treats a missed legacy pending window as terminal and never re-attests the old fence", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    expect((await apply(env, "inventory", 10)).status).toBe(200);
    expect((await apply(env, "provider-fence", 11, ATTESTATION)).status).toBe(200);
    const fenced = await readControl(env) as {
      providerFence?: { attestedAt?: string };
    };
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 121_000);

    const missed = await apply(env, "observe", 12);
    expect(missed.status).toBe(503);
    expect(await missed.json()).toMatchObject({
      phase: "blocked",
      blocked: { code: "legacy_pending_window_missed" },
      automatic: "locked",
    });

    const repeated = await apply(env, "provider-fence", 13, ATTESTATION);
    expect(repeated.status).toBe(409);
    expect(await repeated.json()).toMatchObject({
      phase: "blocked",
      providerFence: { attestedAt: fenced.providerFence?.attestedAt },
    });
  });

  it("reuses the original fence timestamp when a pending attestation resumes", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    expect((await apply(env, "inventory", 16)).status).toBe(200);
    const durable = env._namespace!.durables.get("delivery-cutover:v3")!;
    const current = durable.records.get("cutover-control:v3") as Record<string, unknown>;
    const attestedAt = env._namespace!.now.toISOString();
    durable.records.set("cutover-control:v3", {
      ...current,
      updatedAt: attestedAt,
      pendingOperation: {
        operationId: operationId(17),
        action: "provider-fence",
        inputHash: await sha256Hex(JSON.stringify(["provider-fence", ATTESTATION])),
        baseRevision: current.revision,
        basePhase: "inventoried",
        startedAt: attestedAt,
        attestedAt,
      },
    });
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 5_000);

    const resumed = await apply(env, "provider-fence", 17, ATTESTATION);

    expect(resumed.status).toBe(200);
    expect(await resumed.json()).toMatchObject({
      phase: "observing",
      providerFence: { attestedAt },
      postFenceScan: { startedAt: env._namespace!.now.toISOString() },
    });
  });

  it("counts every automatic key and blocks before provider fencing when the exact cap is exceeded", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    for (let index = 0; index < 518; index += 1) {
      kv._map.set(
        `idemp:arxiv-daily:auto:${index.toString(16).padStart(64, "0")}`,
        "done:legacy",
      );
    }
    const date = "2026-08-10";
    const recipient = "duplicate@example.com";
    const duplicateLogical = `arxiv-daily:auto:${await sha256Hex(
      `${date}\u0000${recipient}`,
    )}`;
    kv._map.set(`idemp:${date}|${recipient}`, "done:legacy");
    kv._map.set(`idemp:${duplicateLogical}`, "pending:legacy");

    const inventory = await apply(env, "inventory", 14);

    expect(inventory.status).toBe(503);
    expect(await inventory.json()).toMatchObject({
      phase: "blocked",
      blocked: { code: "legacy_evidence_capacity_reached" },
      preFenceInventory: {
        automaticKeyCount: 520,
        capacity: 512,
        withinCapacity: false,
        durationBudgetMs: 30_000,
      },
    });
    expect(await readControl(env)).toMatchObject({
      preFenceInventory: { safeBeforeCredentialRevocation: false },
    });
    expect((await apply(env, "provider-fence", 15, ATTESTATION)).status).toBe(409);
  });

  it("fails closed when a post-fence scan itself exceeds the legacy pending window", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    expect((await apply(env, "inventory", 14)).status).toBe(200);
    let delayed = false;
    kv._beforeList = async () => {
      if (delayed) return;
      delayed = true;
      env._namespace!.now = new Date(env._namespace!.now.getTime() + 120_000);
    };

    const fenced = await apply(env, "provider-fence", 15, ATTESTATION);

    expect(fenced.status).toBe(503);
    expect(await fenced.json()).toMatchObject({
      phase: "blocked",
      automatic: "locked",
      blocked: { code: "legacy_pending_window_missed" },
    });
  });

  it("allows delayed marker observations without claiming global KV visibility", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    expect((await apply(env, "inventory", 20)).status).toBe(200);
    expect((await apply(env, "provider-fence", 21, ATTESTATION)).status).toBe(200);
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 60_000);
    expect((await apply(env, "observe", 22)).status).toBe(200);
    const marker = kv._map.get(DELIVERY_V3_CUTOVER_AUDIT_KEY);
    expect(marker).toBeDefined();
    expect(JSON.parse(marker!)).toMatchObject({
      schemaVersion: 3,
      kind: "delivery-v2-cutover-audit",
      providerFence: ATTESTATION,
      legacyAutoEvidenceSnapshot: "exact-canonical-map",
      legacyAutoEvidence: {},
      proof: expect.stringMatching(/^[0-9a-f]{64}$/),
    });
    expect(marker).not.toContain("recipient@example.com");
    expect(marker).not.toContain("idemp:");

    env._namespace!.now = new Date(env._namespace!.now.getTime() + 24 * 60 * 60 * 1000);
    const first = await apply(env, "observe", 23);
    expect(first.status).toBe(200);
    expect(await first.json()).toMatchObject({
      phase: "sealed",
      markerAudit: { observations: 1, globalVisibilityClaimed: false },
    });
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 60_000);
    const second = await apply(env, "observe", 24);
    expect(second.status).toBe(200);
    expect(await second.json()).toMatchObject({
      markerAudit: { observations: 2, globalVisibilityClaimed: false },
    });
    const ready = await apply(env, "seal", 25);
    expect(ready.status).toBe(200);
    expect(await ready.json()).toMatchObject({ phase: "ready", automatic: "ready" });
  });

  it("repairs a deleted marker from valid control and a corrupted control from a valid MACed marker", async () => {
    const first = envWith(memoryKv());
    first.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    await completeCutover(first);
    await putDevice(
      first,
      "device-token",
      "recipient@example.com",
      new Date(first._namespace!.now.getTime() + 1),
    );
    await first.STORE.delete(DELIVERY_V3_CUTOVER_AUDIT_KEY);

    const rewritten = await apply(first, "repair", 26);
    expect(rewritten.status).toBe(200);
    expect(await rewritten.json()).toMatchObject({
      phase: "sealed",
      automatic: "locked",
      markerAudit: { observations: 0, globalVisibilityClaimed: false },
    });
    expect((await worker.fetch(deliveryRequest(), first)).status).toBe(503);
    const mismatched = JSON.parse(
      (await first.STORE.get(DELIVERY_V3_CUTOVER_AUDIT_KEY))!,
    ) as Record<string, unknown>;
    await first.STORE.put(DELIVERY_V3_CUTOVER_AUDIT_KEY, JSON.stringify({
      ...mismatched,
      constructedAt: "2026-01-01T00:00:00.000Z",
    }));
    const mismatchRepair = await apply(first, "repair", 28);
    expect(mismatchRepair.status).toBe(200);
    expect(await mismatchRepair.json()).toMatchObject({
      phase: "sealed",
      automatic: "locked",
      markerAudit: { observations: 0 },
    });

    const secondKv = memoryKv();
    const second = envWith(secondKv);
    second.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    await completeCutover(second);
    const control = second._namespace!.durables.get("delivery-cutover:v3")!;
    const current = control.records.get("cutover-control:v3") as Record<string, unknown>;
    control.records.set("cutover-control:v3", {
      ...current,
      legacyAutoEvidence: { invalid: "done" },
    });

    const rebuilt = await apply(second, "repair", 27);
    expect(rebuilt.status).toBe(200);
    expect(await rebuilt.json()).toMatchObject({
      phase: "sealed",
      automatic: "locked",
      markerAudit: { observations: 0, globalVisibilityClaimed: false },
    });

    const thirdKv = memoryKv();
    const third = envWith(thirdKv);
    third.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    thirdKv._map.set(DELIVERY_V3_CUTOVER_AUDIT_KEY, "{invalid-marker");
    const thirdControl = third._namespace!.get(
      third._namespace!.idFromName("delivery-cutover:v3"),
    );
    const thirdDurable = third._namespace!.durables.get("delivery-cutover:v3")!;
    thirdDurable.records.set("cutover-control:v3", { schemaVersion: 2, phase: "ready" });
    const locked = await apply(third, "repair", 29);
    expect(locked.status).toBe(503);
    expect(await locked.json()).toMatchObject({ phase: "locked", automatic: "locked" });
    expect((await apply(third, "inventory", 30)).status).toBe(200);
    expect((await thirdControl.fetch(
      new Request("https://cutover-control/cutover/status"),
    )).status).toBe(200);
  });

  it("returns the exact stored operation result after response loss and later phase changes", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    const first = await apply(env, "inventory", 30);
    const firstText = await first.text();
    expect(JSON.parse(firstText)).toMatchObject({ phase: "inventoried", revision: 1 });
    expect(await readControl(env)).toMatchObject({
      lastOperation: {
        operationId: operationId(30),
        action: "inventory",
        status: 200,
      },
    });
    expect((await apply(env, "provider-fence", 31, ATTESTATION)).status).toBe(200);

    const replay = await apply(env, "inventory", 30);
    expect(replay.status).toBe(first.status);
    expect(await replay.text()).toBe(firstText);
  });

  it("keeps every state change and successful monotonic no-op permanently bound", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    const first = await apply(env, "inventory", 100);
    const firstText = await first.text();
    expect(first.status).toBe(200);
    expect((await apply(env, "provider-fence", 101, ATTESTATION)).status).toBe(200);
    let firstNoopText = "";
    for (let id = 102; id < 182; id += 1) {
      const response = await apply(env, "inventory", id);
      expect(response.status).toBe(200);
      const text = await response.text();
      if (id === 102) firstNoopText = text;
    }
    const controlRecords = env._namespace!.durables.get("delivery-cutover:v3")!.records;
    expect(Array.from(controlRecords.keys()).filter((key) =>
      key.startsWith("cutover-operation:v3:")
    )).toHaveLength(82);

    const stateChangeReplay = await apply(env, "inventory", 100);
    expect(stateChangeReplay.status).toBe(200);
    expect(await stateChangeReplay.text()).toBe(firstText);
    const noopReplay = await apply(env, "inventory", 102);
    expect(noopReplay.status).toBe(200);
    expect(await noopReplay.text()).toBe(firstNoopText);
    const rebound = await apply(env, "observe", 102);
    expect(rebound.status).toBe(409);
  });

  it("uses fresh completion timestamps after a slow scan", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    const startedAt = env._namespace!.now.toISOString();
    let delayed = false;
    kv._beforeList = async () => {
      if (delayed) return;
      delayed = true;
      env._namespace!.now = new Date(env._namespace!.now.getTime() + 5_000);
    };

    const response = await apply(env, "inventory", 200);
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      preFenceInventory: {
        startedAt,
        completedAt: env._namespace!.now.toISOString(),
        durationMs: 5_000,
        durationBudgetMs: 30_000,
      },
    });
    expect(await readControl(env)).toMatchObject({
      preFenceInventory: { safeBeforeCredentialRevocation: true },
    });
  });

  it("accepts the exact preflight duration budget boundary before fencing", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    let delayed = false;
    kv._beforeList = async () => {
      if (delayed) return;
      delayed = true;
      env._namespace!.now = new Date(env._namespace!.now.getTime() + 30_000);
    };

    const inventory = await apply(env, "inventory", 201);
    expect(inventory.status).toBe(200);
    const body = await inventory.json() as {
      preFenceInventory?: { completedAt?: string };
    };
    expect(body).toMatchObject({
      preFenceInventory: {
        durationMs: 30_000,
        durationBudgetMs: 30_000,
      },
    });
    expect(body.preFenceInventory?.completedAt).toBe(env._namespace!.now.toISOString());
    expect(await readControl(env)).toMatchObject({
      preFenceInventory: { safeBeforeCredentialRevocation: true },
    });
    kv._beforeList = undefined;
    expect((await apply(env, "provider-fence", 202, ATTESTATION)).status).toBe(200);
  });

  it("counts slow ignored test-key inventory time and rejects over-budget fencing", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    kv._map.set(
      "idemp:test|2026-08-10|ignored@example.com|fixture",
      "done:test",
    );
    kv._map.set(`idemp:arxiv-daily:test:${"e".repeat(32)}`, "done:test");
    let delayed = false;
    kv._beforeList = async () => {
      if (delayed) return;
      delayed = true;
      env._namespace!.now = new Date(env._namespace!.now.getTime() + 30_001);
    };

    const inventory = await apply(env, "inventory", 203);
    expect(inventory.status).toBe(200);
    expect(await inventory.json()).toMatchObject({
      preFenceInventory: {
        automaticKeyCount: 0,
        durationMs: 30_001,
        durationBudgetMs: 30_000,
      },
    });
    expect(await readControl(env)).toMatchObject({
      preFenceInventory: { safeBeforeCredentialRevocation: false },
    });
    kv._beforeList = undefined;
    const rejected = await apply(env, "provider-fence", 204, ATTESTATION);
    expect(rejected.status).toBe(409);
    expect(await rejected.json()).toMatchObject({
      phase: "inventoried",
      preFenceInventory: { safeBeforeCredentialRevocation: false },
    });
    const records = env._namespace!.durables.get("delivery-cutover:v3")!.records;
    expect(records.has(`cutover-operation:v3:${operationId(204)}`)).toBe(false);
    expect((await apply(env, "inventory", 205)).status).toBe(200);
    expect(await readControl(env)).toMatchObject({
      preFenceInventory: { safeBeforeCredentialRevocation: true },
    });
  });

  it("computes inventory freshness at response time and rejects stale fencing read-only", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    const inventory = await apply(env, "inventory", 206);
    const body = await inventory.json() as {
      preFenceInventory?: { credentialRevocationDeadline?: string };
    };
    expect(body.preFenceInventory?.credentialRevocationDeadline).toBe(
      new Date(env._namespace!.now.getTime() + 5 * 60_000).toISOString(),
    );
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 5 * 60_000 + 1);
    expect(await readControl(env)).toMatchObject({
      phase: "inventoried",
      preFenceInventory: { safeBeforeCredentialRevocation: false },
    });
    let scans = 0;
    kv._beforeList = async () => {
      scans += 1;
    };

    const rejected = await apply(env, "provider-fence", 207, ATTESTATION);
    expect(rejected.status).toBe(409);
    expect(scans).toBe(0);
    expect(await rejected.json()).toMatchObject({
      phase: "inventoried",
      preFenceInventory: { safeBeforeCredentialRevocation: false },
    });
    const records = env._namespace!.durables.get("delivery-cutover:v3")!.records;
    expect(records.has(`cutover-operation:v3:${operationId(207)}`)).toBe(false);
    kv._beforeList = undefined;
    expect((await apply(env, "inventory", 208)).status).toBe(200);
  });

  it("retries a failed stale inventory refresh before any provider attestation", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    expect((await apply(env, "inventory", 209)).status).toBe(200);
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 5 * 60_000 + 1);
    const unsupportedKey = "idemp:unsupported-refresh-fixture";
    kv._map.set(unsupportedKey, "private-fixture");

    const failed = await apply(env, "inventory", 210);
    const failedText = await failed.text();
    expect(failed.status).toBe(503);
    expect(JSON.parse(failedText)).toMatchObject({
      phase: "blocked",
      automatic: "locked",
      blocked: { code: "legacy_scan_unsupported_key" },
    });
    expect((await apply(env, "provider-fence", 211, ATTESTATION)).status).toBe(409);

    kv._map.delete(unsupportedKey);
    const refreshed = await apply(env, "inventory", 212);
    expect(refreshed.status).toBe(200);
    expect(await readControl(env)).toMatchObject({
      phase: "inventoried",
      automatic: "locked",
      preFenceInventory: { safeBeforeCredentialRevocation: true },
    });
    const failedReplay = await apply(env, "inventory", 210);
    expect(failedReplay.status).toBe(503);
    expect(await failedReplay.text()).toBe(failedText);
    const records = env._namespace!.durables.get("delivery-cutover:v3")!.records;
    expect(records.has(`cutover-operation:v3:${operationId(210)}`)).toBe(true);
    expect(records.has(`cutover-operation:v3:${operationId(212)}`)).toBe(true);
  });

  it("keeps GET status responsive while an inventory scan is outside the DO long critical section", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let scanStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      scanStarted = resolve;
    });
    kv._beforeList = async () => {
      scanStarted();
      await held;
    };

    const inventory = apply(env, "inventory", 210);
    await started;
    const status = await Promise.race([
      readControl(env),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("status blocked behind scan")), 100)
      ),
    ]);
    expect(status).toMatchObject({
      phase: "locked",
      pendingOperation: { action: "inventory" },
    });
    release();
    expect((await inventory).status).toBe(200);
  });

  it("blocks unknown idemp keys with safe status and no raw key or email", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    const privateKey = "idemp:private-recipient@example.com|unsupported";
    kv._map.set(privateKey, "private-value");

    const response = await apply(env, "inventory", 220);
    expect(response.status).toBe(503);
    const text = await response.text();
    expect(JSON.parse(text)).toMatchObject({
      phase: "blocked",
      blocked: { code: "legacy_scan_unsupported_key" },
    });
    expect(text).not.toContain(privateKey);
    expect(text).not.toContain("private-recipient@example.com");
  });
});

describe("ready-only automatic cutover", () => {
  it("keeps automatic locked until authoritative v3 state is ready", async () => {
    const { env } = await authenticatedEnv({ stageCutover: false });
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const locked = await worker.fetch(deliveryRequest(), env);

    expect(locked.status).toBe(503);
    expect(await locked.json()).toEqual({
      error: "delivery cutover is not ready",
      ambiguous: false,
    });
    expect(provider).not.toHaveBeenCalled();
  });

  it.each([
    ["done:msg_done", "legacy_delivery_done", false],
    ["pending:legacy-claim", "legacy_delivery_attempted", true],
  ] as const)(
    "imports scanned %s evidence into a permanent device ledger block",
    async (legacyValue, error, ambiguous) => {
      const { kv, env } = await authenticatedEnv({ quota: 5, legacyValue });
      kv._map.clear();
      await putDevice(
        env,
        "device-token",
        "recipient@example.com",
        new Date(env._namespace!.now.getTime() + 1),
      );
      const provider = providerSuccess("must_not_send");
      vi.stubGlobal("fetch", provider);

      const imported = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
      const control = env._namespace!.durables.get("delivery-cutover:v3")!;
      control.records.set("cutover-control:v3", { schemaVersion: 2, phase: "ready" });
      const replay = await worker.fetch(deliveryRequest({ key: AUTO_KEY_B }), env);

      expect(imported.status).toBe(409);
      expect(await imported.json()).toEqual({ error, ambiguous });
      expect(replay.status).toBe(409);
      expect(await replay.json()).toEqual({ error, ambiguous });
      expect(provider).not.toHaveBeenCalled();
      const durable = deviceDurables(env)[0]!;
      const index = durable.records.get("ledger:index:v2") as {
        entries?: Array<{ keyHash?: string; keyKind?: string }>;
      };
      expect(index.entries).toHaveLength(1);
      expect(index.entries?.[0]).toMatchObject({ keyKind: "auto" });
      expect(Array.from(durable.records.keys()).some((key) =>
        key.startsWith("legacy-import:")
      )).toBe(false);
      const stored = JSON.stringify(Array.from(durable.records));
      expect(stored).not.toContain("recipient@example.com");
      expect(stored).not.toContain(legacyValue);
    },
  );

  it("enforces normal automatic quota before importing positive legacy evidence", async () => {
    const { env } = await authenticatedEnv({ quota: 1, legacyValue: "done:legacy" });
    const identity = await hashDeviceToken("device-token", env.TOKEN_SECRET);
    env._namespace!.get(env._namespace!.idFromName(`device-v2:${identity}`));
    const durable = env._namespace!.durables.get(`device-v2:${identity}`)!;
    durable.records.set("quota:v2:auto:2026-08-10", 1);
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(429);
    expect(provider).not.toHaveBeenCalled();
    expect(Array.from(durable.records.keys()).some((key) =>
      key.startsWith("legacy-import:")
    )).toBe(false);
  });

  it("enforces normal automatic ledger capacity before importing legacy evidence", async () => {
    const { env } = await authenticatedEnv({ quota: 10, legacyValue: "done:legacy" });
    const identity = await hashDeviceToken("device-token", env.TOKEN_SECRET);
    env._namespace!.get(env._namespace!.idFromName(`device-v2:${identity}`));
    const durable = env._namespace!.durables.get(`device-v2:${identity}`)!;
    durable.records.set("ledger:index:v2", {
      schemaVersion: 2,
      entries: Array.from({ length: 5_000 }, (_, index) => ({
        keyHash: index.toString(16).padStart(64, "0"),
        keyKind: "auto",
      })),
    });
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(507);
    expect(provider).not.toHaveBeenCalled();
    expect(Array.from(durable.records.keys()).some((key) =>
      key.startsWith("legacy-import:")
    )).toBe(false);
  });

  it("keeps a pre-ready v2 identity without positive legacy evidence blocked", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await putDevice(
      env,
      "device-token",
      "recipient@example.com",
      new Date(env._namespace!.now.getTime() - 1),
    );
    await completeCutover(env);
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(provider).not.toHaveBeenCalled();
  });

  it("keeps a legacy identity without v2 provenance blocked after ready", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    const deviceHash = await hashDeviceToken("device-token", env.TOKEN_SECRET);
    kv._map.set(`device:${deviceHash}`, JSON.stringify({
      email: "recipient@example.com",
      createdAt: new Date(env._namespace!.now.getTime() + 1).toISOString(),
    }));
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(provider).not.toHaveBeenCalled();
  });
});

describe("authenticated device routing", () => {
  it("routes all keys for one device to one privacy-safe Durable Object", async () => {
    const { env } = await authenticatedEnv();
    const provider = providerSuccess();
    vi.stubGlobal("fetch", provider);

    await Promise.all([
      worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env),
      worker.fetch(deliveryRequest({ key: AUTO_KEY_B }), env),
    ]);

    const deviceNames = env._namespace!.names.filter((name) =>
      name.startsWith("device-v2:")
    );
    expect(new Set(deviceNames).size).toBe(1);
    const [name] = deviceNames;
    expect(name).toMatch(/^device-v2:[0-9a-f]{64}$/);
    expect(name).not.toContain("device-token");
    expect(name).not.toContain("recipient@example.com");
    expect(provider).toHaveBeenCalledTimes(1);
  });

  it("isolates different devices that use the same client key", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    const issuedAt = new Date(env._namespace!.now.getTime() + 1);
    await putDevice(env, "device-one", "one@example.com", issuedAt);
    await putDevice(env, "device-two", "two@example.com", issuedAt);
    const provider = vi.fn(async (_input: string, init?: RequestInit) =>
      new Response(
        JSON.stringify({ id: `msg_${provider.mock.calls.length}` }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", provider);

    const [first, second] = await Promise.all([
      worker.fetch(deliveryRequest({
        token: "device-one",
        to: "one@example.com",
        key: AUTO_KEY_A,
      }), env),
      worker.fetch(deliveryRequest({
        token: "device-two",
        to: "two@example.com",
        key: AUTO_KEY_A,
      }), env),
    ]);

    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(2);
    expect(new Set(env._namespace!.names.filter((name) =>
      name.startsWith("device-v2:")
    )).size).toBe(2);
    const keys = provider.mock.calls.map(
      ([, init]) => (init?.headers as Record<string, string>)["Idempotency-Key"],
    );
    expect(new Set(keys).size).toBe(2);
  });

  it("fails closed before provider when the DO binding is missing", async () => {
    const { env } = await authenticatedEnv({ binding: false });
    const provider = vi.fn();
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(provider).not.toHaveBeenCalled();
  });
});

describe("device-scoped ledger and quota", () => {
  it("deduplicates automatic delivery by server-derived date and recipient identity", async () => {
    const { env } = await authenticatedEnv({ quota: 5 });
    const provider = providerSuccess("msg_once");
    vi.stubGlobal("fetch", provider);

    const first = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
    const replay = await worker.fetch(deliveryRequest({ key: AUTO_KEY_B }), env);

    expect(first.status).toBe(200);
    expect(replay.status).toBe(200);
    expect(await replay.json()).toEqual({ ok: true });
    expect(provider).toHaveBeenCalledTimes(1);
  });

  it("uses the same automatic provider key for different valid client hints", async () => {
    const first = await authenticatedEnv();
    const second = await authenticatedEnv();
    const provider = providerSuccess("msg_ok");
    vi.stubGlobal("fetch", provider);

    expect((await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), first.env)).status)
      .toBe(200);
    expect((await worker.fetch(deliveryRequest({ key: AUTO_KEY_B }), second.env)).status)
      .toBe(200);

    const keys = provider.mock.calls.map(
      ([, init]) => (init?.headers as Record<string, string>)["Idempotency-Key"],
    );
    expect(keys).toHaveLength(2);
    expect(new Set(keys).size).toBe(1);
  });

  it("allows test delivery without consuming or depending on automatic cutover state", async () => {
    const { env } = await authenticatedEnv({ quota: 1, stageCutover: false });
    const provider = providerSuccess("msg_test");
    vi.stubGlobal("fetch", provider);

    const test = await worker.fetch(deliveryRequest({ key: TEST_KEY_A }), env);
    const automaticWithoutCutover = await worker.fetch(
      deliveryRequest({ key: AUTO_KEY_A }),
      env,
    );
    await completeCutover(env);
    const automatic = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);

    expect(test.status).toBe(200);
    expect(automaticWithoutCutover.status).toBe(503);
    expect(automatic.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(2);
  });

  it("keeps tests independent from legacy automatic evidence and automatic ledger state", async () => {
    const { kv, env } = await authenticatedEnv({
      quota: 5,
      legacyValue: "done:msg_done",
    });
    const provider = providerSuccess("msg_test");
    vi.stubGlobal("fetch", provider);

    const first = await worker.fetch(deliveryRequest({ key: TEST_KEY_A }), env);
    const retry = await worker.fetch(deliveryRequest({ key: TEST_KEY_A }), env);
    const automatic = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);

    expect(first.status).toBe(200);
    expect(retry.status).toBe(200);
    expect(automatic.status).toBe(409);
    expect(await automatic.json()).toEqual({
      error: "legacy_delivery_done",
      ambiguous: false,
    });
    expect(provider).toHaveBeenCalledTimes(1);
  });

  it("does not let an imported automatic block poison a later test key", async () => {
    const { env } = await authenticatedEnv({
      quota: 5,
      legacyValue: "pending:legacy-claim",
    });
    const provider = providerSuccess("msg_test_after_import");
    vi.stubGlobal("fetch", provider);

    const automatic = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
    const test = await worker.fetch(deliveryRequest({ key: TEST_KEY_A }), env);
    const retry = await worker.fetch(deliveryRequest({ key: TEST_KEY_A }), env);

    expect(automatic.status).toBe(409);
    expect(await automatic.json()).toEqual({
      error: "legacy_delivery_attempted",
      ambiguous: true,
    });
    expect(test.status).toBe(200);
    expect(retry.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(1);
    const providerKey = (provider.mock.calls[0]![1]?.headers as Record<string, string>)[
      "Idempotency-Key"
    ];
    expect(providerKey).toMatch(/^arxiv-daily:relay:v2:test:[0-9a-f]{64}$/);
  });

  it("keeps distinct test keys independent while retries reuse one provider key", async () => {
    const { env } = await authenticatedEnv({ quota: 5 });
    const provider = providerSuccess("msg_test");
    vi.stubGlobal("fetch", provider);

    const first = await worker.fetch(deliveryRequest({ key: TEST_KEY_A }), env);
    const retry = await worker.fetch(deliveryRequest({ key: TEST_KEY_A }), env);
    const secondTest = await worker.fetch(deliveryRequest({ key: TEST_KEY_B }), env);

    expect(first.status).toBe(200);
    expect(retry.status).toBe(200);
    expect(secondTest.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(2);
    const keys = provider.mock.calls.map(
      ([, init]) => (init?.headers as Record<string, string>)["Idempotency-Key"],
    );
    expect(new Set(keys).size).toBe(2);
  });

  it("stores no raw recipient, token, or payload in the DO ledger", async () => {
    const { env } = await authenticatedEnv({ quota: 5 });
    const provider = providerSuccess("msg_private");
    vi.stubGlobal("fetch", provider);

    await worker.fetch(deliveryRequest({
      key: AUTO_KEY_A,
      html: "<p>private-body-marker</p>",
      text: "private-text-marker",
    }), env);

    const stored = JSON.stringify(Array.from(deviceDurables(env)[0]!.records));
    expect(stored).not.toContain("recipient@example.com");
    expect(stored).not.toContain("device-token");
    expect(stored).not.toContain("private-body-marker");
    expect(stored).not.toContain("private-text-marker");
    expect(stored).not.toContain("msg_private");
  });

  it("serializes different automatic dates against one authoritative device quota", async () => {
    const { env } = await authenticatedEnv({ quota: 1 });
    const provider = providerSuccess("msg_only");
    vi.stubGlobal("fetch", provider);

    const responses = await Promise.all([
      worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env),
      worker.fetch(deliveryRequest({ key: AUTO_KEY_B, date: "2026-08-11" }), env),
    ]);

    expect(provider).toHaveBeenCalledTimes(1);
    expect(responses.map((response) => response.status).sort()).toEqual([200, 429]);
  });

  it("replays the stored success and rejects same key with another payload without leaking id", async () => {
    const { env } = await authenticatedEnv();
    const provider = providerSuccess("private_message_id");
    vi.stubGlobal("fetch", provider);

    const first = await worker.fetch(deliveryRequest(), env);
    const replay = await worker.fetch(deliveryRequest(), env);
    const conflict = await worker.fetch(
      deliveryRequest({ subject: "Different payload" }),
      env,
    );

    expect(await first.json()).toEqual({ ok: true });
    expect(await replay.json()).toEqual({ ok: true });
    expect(conflict.status).toBe(409);
    const conflictBody = await conflict.json() as Record<string, unknown>;
    expect(conflictBody).toEqual({
      error: "idempotency key is already bound to another request",
    });
    expect(JSON.stringify(conflictBody)).not.toContain("private_message_id");
    expect(provider).toHaveBeenCalledTimes(1);
  });

  it("consumes quota for an ambiguous physical attempt without exposing its error", async () => {
    const { env } = await authenticatedEnv({ quota: 1 });
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const provider = vi.fn(async () => {
      throw new Error("private low-level reset after request");
    });
    vi.stubGlobal("fetch", provider);

    const first = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
    const second = await worker.fetch(
      deliveryRequest({ key: AUTO_KEY_B, date: "2026-08-11" }),
      env,
    );
    const firstBody = await first.text();

    expect(first.status).toBe(502);
    expect(JSON.parse(firstBody)).toMatchObject({ ambiguous: true });
    expect(firstBody).not.toContain("private low-level reset");
    expect(second.status).toBe(429);
    expect(provider).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(Array.from(deviceDurables(env)[0]!.records))).not.toContain(
      "private low-level reset",
    );
    expect(log).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
  });

  it.each([418, 425] as const)(
    "keeps HTTP %s attempted, deduped, and quota-blocking",
    async (status) => {
      const { env } = await authenticatedEnv({ quota: 1 });
      const provider = vi.fn(async () =>
        new Response("body must not change classification", { status })
      );
      vi.stubGlobal("fetch", provider);

      const first = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
      const replay = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
      const another = await worker.fetch(
        deliveryRequest({ key: AUTO_KEY_B, date: "2026-08-11" }),
        env,
      );

      expect(first.status).toBe(502);
      expect(await first.json()).toMatchObject({ ambiguous: true });
      expect(replay.status).toBe(409);
      expect(await replay.json()).toMatchObject({ ambiguous: true });
      expect(another.status).toBe(429);
      expect(provider).toHaveBeenCalledTimes(1);
    },
  );

  it.each([
    [400, 422],
    [401, 422],
    [403, 422],
    [404, 422],
    [422, 422],
    [429, 429],
  ] as const)(
    "permanently replays definitive HTTP %s rejection without another automatic attempt",
    async (providerStatus, relayStatus) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date("2026-08-10T00:00:00.000Z"));
      const { env } = await authenticatedEnv({ quota: 1 });
      const provider = vi
        .fn()
        .mockResolvedValueOnce(new Response(
          "private provider body for recipient@example.com with private_provider_message_id",
          { status: providerStatus },
        ))
        .mockResolvedValueOnce(
          new Response(JSON.stringify({ id: "private_test_message_id" }), {
            status: 200,
          }),
        );
      vi.stubGlobal("fetch", provider);

      const rejected = await worker.fetch(
        deliveryRequest({ key: AUTO_KEY_A }),
        env,
      );
      const replay = await worker.fetch(
        deliveryRequest({ key: AUTO_KEY_B }),
        env,
      );

      expect(rejected.status).toBe(relayStatus);
      expect(replay.status).toBe(relayStatus);
      expect(await rejected.json()).toEqual({
        error: "provider_definitive_rejection",
        ambiguous: false,
      });
      expect(await replay.json()).toEqual({
        error: "provider_definitive_rejection",
        ambiguous: false,
      });
      expect(provider).toHaveBeenCalledTimes(1);

      const durable = deviceDurables(env)[0]!;
      expect(durable.records.get("quota:v2:auto:2026-08-10")).toBe(0);
      const stored = JSON.stringify(Array.from(durable.records.values()));
      expect(stored).not.toContain("private provider body");
      expect(stored).not.toContain("private_provider_message_id");
      expect(stored).not.toContain("recipient@example.com");

      const independentTest = await worker.fetch(
        deliveryRequest({ key: TEST_KEY_A }),
        env,
      );
      expect(independentTest.status).toBe(200);
      expect(await independentTest.json()).toEqual({ ok: true });
      expect(provider).toHaveBeenCalledTimes(2);
    },
  );

  it("keeps attempted delivery and quota blocking when completion storage fails", async () => {
    const { env } = await authenticatedEnv({ quota: 1, failDonePut: true });
    const provider = providerSuccess("msg_accepted");
    vi.stubGlobal("fetch", provider);

    const first = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
    const replay = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
    const another = await worker.fetch(
      deliveryRequest({ key: AUTO_KEY_B, date: "2026-08-11" }),
      env,
    );

    expect(first.status).toBe(502);
    expect(await first.json()).toMatchObject({ ambiguous: true });
    expect(replay.status).toBe(409);
    expect(await replay.json()).toMatchObject({ ambiguous: true });
    expect(another.status).toBe(429);
    expect(provider).toHaveBeenCalledTimes(1);
  });

  it("fails closed when ledger index binding is missing", async () => {
    const { env } = await authenticatedEnv();
    const provider = providerSuccess("msg_first");
    vi.stubGlobal("fetch", provider);
    await worker.fetch(deliveryRequest(), env);

    const durable = deviceDurables(env)[0]!;
    const ledgerKey = Array.from(durable.records.keys()).find((key) =>
      key.startsWith("ledger:v2:"),
    )!;
    durable.records.delete(ledgerKey);

    const replay = await worker.fetch(deliveryRequest(), env);

    expect(replay.status).toBe(503);
    expect(provider).toHaveBeenCalledTimes(1);
  });

  it("expires only terminal test records and retains automatic records", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-10T00:00:00.000Z"));
    const { env } = await authenticatedEnv({ quota: 10 });
    const provider = vi.fn(async () =>
      new Response(
        JSON.stringify({ id: `msg_${provider.mock.calls.length}` }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", provider);

    await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
    await worker.fetch(deliveryRequest({ key: TEST_KEY_A }), env);
    vi.setSystemTime(new Date("2026-09-11T00:00:00.000Z"));
    env._namespace!.now = new Date("2026-09-11T00:00:00.000Z");
    await worker.fetch(deliveryRequest({ key: TEST_KEY_B }), env);

    const durable = deviceDurables(env)[0]!;
    const ledgers = Array.from(durable.records.values()).filter(
      (value) => value && typeof value === "object" &&
        (value as { schemaVersion?: unknown }).schemaVersion === 2 &&
        "keyKind" in (value as object),
    ) as Array<{ keyKind: string }>;
    expect(ledgers.filter((record) => record.keyKind === "auto")).toHaveLength(1);
    expect(ledgers.filter((record) => record.keyKind === "test")).toHaveLength(1);
  });
});

describe("provider outcome classification", () => {
  it.each([
    [400, false, 422],
    [408, true, 502],
    [409, true, 502],
    [418, true, 502],
    [425, true, 502],
    [429, false, 429],
    [500, true, 502],
  ] as const)(
    "classifies provider HTTP %s with ambiguous=%s",
    async (status, ambiguous, expectedStatus) => {
      const { env } = await authenticatedEnv();
      vi.stubGlobal("fetch", vi.fn(async () =>
        new Response(
          status === 409
            ? JSON.stringify({ name: "concurrent_idempotent_requests" })
            : "provider error",
          { status },
        ),
      ));

      const response = await worker.fetch(deliveryRequest(), env);

      expect(response.status).toBe(expectedStatus);
      expect(await response.json()).toMatchObject({ ambiguous });
    },
  );

  it("does not echo provider error bodies that may contain the recipient", async () => {
    const { env } = await authenticatedEnv();
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(
        "invalid recipient recipient@example.com with private payload detail",
        { status: 400 },
      ),
    ));

    const response = await worker.fetch(deliveryRequest(), env);
    const body = await response.text();

    expect(response.status).toBe(422);
    expect(body).not.toContain("recipient@example.com");
    expect(body).not.toContain("private payload detail");
  });

  it("returns content-free success without retaining or logging provider data", async () => {
    const { env } = await authenticatedEnv();
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(JSON.stringify({
        id: "private_provider_message_id",
        recipient: "recipient@example.com",
        detail: "private provider body",
      }), { status: 200 }),
    ));

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ok: true });
    const stored = JSON.stringify(Array.from(deviceDurables(env)[0]!.records));
    expect(stored).not.toContain("private_provider_message_id");
    expect(stored).not.toContain("recipient@example.com");
    expect(stored).not.toContain("private provider body");
    expect(log).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
  });

  it("treats an invalid 2xx response as ambiguous without exposing the provider body", async () => {
    const { env } = await authenticatedEnv();
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response("private invalid success body for recipient@example.com", {
        status: 200,
      }),
    ));

    const response = await worker.fetch(deliveryRequest(), env);
    const body = await response.text();

    expect(response.status).toBe(502);
    expect(JSON.parse(body)).toMatchObject({ ambiguous: true });
    expect(body).not.toContain("recipient@example.com");
    expect(body).not.toContain("private invalid success body");
    const stored = JSON.stringify(Array.from(deviceDurables(env)[0]!.records));
    expect(stored).not.toContain("recipient@example.com");
    expect(stored).not.toContain("private invalid success body");
  });
});

describe("safe relay errors", () => {
  it("does not expose the verified raw recipient in the completion response", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await putPending(env, "verify-token", "private-recipient@example.com");

    const response = await worker.fetch(
      new Request("https://relay.test/v1/verify?token=verify-token"),
      env,
    );
    const body = await response.text();

    expect(response.status).toBe(200);
    expect(body).toContain("Email verified");
    expect(body).not.toContain("private-recipient@example.com");
  });

  it("does not expose an unexpected low-level KV error", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    kv._failGet.add("device-v2:" + await hashDeviceToken("device-token", env.TOKEN_SECRET));

    const response = await worker.fetch(deliveryRequest(), env);
    const body = await response.text();

    expect(response.status).toBe(500);
    expect(body).toContain("relay request failed");
    expect(body).not.toContain("injected KV read marker");
  });
});

describe("authentication", () => {
  it("issues v2 devices under a key prefix the legacy Worker cannot read", async () => {
    const kv = memoryKv();
    const env = envWith(kv, { binding: false });
    await putDevice(env, "new-v2-token", "recipient@example.com");
    const hash = await hashDeviceToken("new-v2-token", env.TOKEN_SECRET);

    expect(kv._map.has(`device-v2:${hash}`)).toBe(true);
    expect(kv._map.has(`device:${hash}`)).toBe(false);
  });

  it("derives a stable privacy-safe device identity from the token", async () => {
    const hash = await hashDeviceToken("device-token", "secret");
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    expect(hash).not.toContain("device-token");
  });
});
