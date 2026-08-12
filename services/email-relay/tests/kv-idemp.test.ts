import { afterEach, describe, expect, it, vi } from "vitest";
import worker from "../src/index";
import { DeliverGate } from "../src/deliver-gate";
import {
  CUTOVER_AUTOMATIC_PATH,
  CUTOVER_ISSUE_DEVICE_PATH,
  authorizeAutomaticDelivery,
  fetchCutoverStatus,
  issueReadyBoundDevice,
  postCutoverAction,
  type CutoverAction,
} from "../src/cutover-control";
import { hmacSha256Hex, sha256Hex } from "../src/crypto";
import {
  hashDeviceToken,
  hashLegacyAutoDeliveryIdentity,
  hashPendingToken,
  hashRecipientIdentity,
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
const CUTOVER_BINDING_KEY = "cutover-binding:v1";
const CUTOVER_STATE_INDEX_KEY = "cutover-state-index:v1";
const BUILD_IDENTITY = `email-relay-v2-${"a".repeat(40)}`;
const IDENTITY_SECRET = "identity-secret-stable-test-v1";
const PROTOCOL_GENERATION = 2;
const READY_GENERATION = 6;

type MemoryDurable = ReturnType<typeof durableState>;

function memoryKv() {
  const map = new Map<string, string>();
  const failGet = new Set<string>();
  const hideFromList = new Set<string>();
  let failList = false;
  let pageSize = Number.POSITIVE_INFINITY;
  let beforeList: (() => Promise<void>) | undefined;
  let beforePut: ((key: string) => Promise<void>) | undefined;
  let beforeDelete: ((key: string) => Promise<void>) | undefined;
  const failPutPrefixes = new Set<string>();
  const failPutAfterPrefixes = new Set<string>();
  const failDeletePrefixes = new Set<string>();
  const failDeleteAfterPrefixes = new Set<string>();
  const staleAfterDeletePrefixes = new Set<string>();
  const staleReads = new Map<string, string>();
  return {
    async get(key: string) {
      if (failGet.has(key)) throw new Error("injected KV read marker");
      return map.get(key) ?? staleReads.get(key) ?? null;
    },
    async put(key: string, value: string) {
      await beforePut?.(key);
      if (Array.from(failPutPrefixes).some((prefix) => key.startsWith(prefix))) {
        throw new Error("injected KV put failure");
      }
      map.set(key, value);
      if (
        Array.from(failPutAfterPrefixes).some((prefix) => key.startsWith(prefix))
      ) {
        throw new Error("injected KV put outcome unknown");
      }
    },
    async delete(key: string) {
      await beforeDelete?.(key);
      if (Array.from(failDeletePrefixes).some((prefix) => key.startsWith(prefix))) {
        throw new Error("injected KV delete failure");
      }
      const previous = map.get(key);
      if (
        previous !== undefined &&
        Array.from(staleAfterDeletePrefixes).some((prefix) => key.startsWith(prefix))
      ) {
        staleReads.set(key, previous);
      }
      map.delete(key);
      if (Array.from(failDeleteAfterPrefixes).some((prefix) => key.startsWith(prefix))) {
        throw new Error("injected KV delete outcome unknown");
      }
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
    _failPutPrefixes: failPutPrefixes,
    _failPutAfterPrefixes: failPutAfterPrefixes,
    _failDeletePrefixes: failDeletePrefixes,
    _failDeleteAfterPrefixes: failDeleteAfterPrefixes,
    _staleAfterDeletePrefixes: staleAfterDeletePrefixes,
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
    set _beforePut(value: ((key: string) => Promise<void>) | undefined) {
      beforePut = value;
    },
    set _beforeDelete(value: ((key: string) => Promise<void>) | undefined) {
      beforeDelete = value;
    },
  };
}

function durableState(
  options: { failDonePut?: boolean } = {},
  records: Map<string, unknown> = new Map(),
) {
  let tail: Promise<unknown> = Promise.resolve();
  let alarm: number | null = null;
  let failTransactionAfterCommit = false;
  const failPutPrefixes = new Set<string>();
  const failPutAfterPrefixes = new Set<string>();
  const failStatusPuts = new Set<string>();
  const failStatusPutsAfter = new Set<string>();
  const shouldFail = (prefixes: Set<string>, key: string) =>
    Array.from(prefixes).some((prefix) => key.startsWith(prefix));
  const putInto = async (
    target: Map<string, unknown>,
    key: string,
    value: unknown,
  ): Promise<void> => {
    if (
      options.failDonePut &&
      value &&
      typeof value === "object" &&
      (value as { status?: unknown }).status === "done"
    ) {
      throw new Error("completion storage unavailable");
    }
    if (shouldFail(failPutPrefixes, key)) {
      throw new Error("injected DO put failure");
    }
    if (
      value &&
      typeof value === "object" &&
      failStatusPuts.has(String((value as { status?: unknown }).status ?? ""))
    ) {
      throw new Error("injected DO status put failure");
    }
    target.set(key, structuredClone(value));
    if (
      value &&
      typeof value === "object" &&
      failStatusPutsAfter.has(String((value as { status?: unknown }).status ?? ""))
    ) {
      throw new Error("injected DO status put outcome unknown");
    }
    if (shouldFail(failPutAfterPrefixes, key)) {
      throw new Error("injected DO put outcome unknown");
    }
  };
  const listFrom = <T>(
    source: Map<string, unknown>,
    options: { prefix?: string; limit?: number } = {},
  ): Map<string, T> => {
    const prefix = options.prefix ?? "";
    const entries = Array.from(source.entries())
      .filter(([key]) => key.startsWith(prefix))
      .sort(([left], [right]) => left.localeCompare(right))
      .slice(0, options.limit);
    return new Map(
      entries.map(([key, value]) => [key, structuredClone(value) as T]),
    );
  };
  const storage = {
    async get<T>(key: string): Promise<T | undefined> {
      const value = records.get(key);
      return value === undefined ? undefined : structuredClone(value) as T;
    },
    async list<T>(
      options: { prefix?: string; limit?: number } = {},
    ): Promise<Map<string, T>> {
      return listFrom<T>(records, options);
    },
    async put(key: string, value: unknown): Promise<void> {
      await putInto(records, key, value);
    },
    async delete(key: string): Promise<boolean> {
      return records.delete(key);
    },
    async transaction<T>(callback: (txn: unknown) => Promise<T>): Promise<T> {
      const working = new Map(
        Array.from(records, ([key, value]) => [key, structuredClone(value)]),
      );
      const txn = {
        async get<T>(key: string): Promise<T | undefined> {
          const value = working.get(key);
          return value === undefined ? undefined : structuredClone(value) as T;
        },
        async list<T>(
          options: { prefix?: string; limit?: number } = {},
        ): Promise<Map<string, T>> {
          return listFrom<T>(working, options);
        },
        async put(key: string, value: unknown): Promise<void> {
          await putInto(working, key, value);
        },
        async delete(key: string): Promise<boolean> {
          return working.delete(key);
        },
      };
      const result = await callback(txn);
      records.clear();
      for (const [key, value] of working) {
        records.set(key, structuredClone(value));
      }
      if (failTransactionAfterCommit) {
        throw new Error("injected DO transaction outcome unknown");
      }
      return result;
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
  return {
    state,
    records,
    storage,
    failPutPrefixes,
    failPutAfterPrefixes,
    failStatusPuts,
    failStatusPutsAfter,
    get failTransactionAfterCommit() {
      return failTransactionAfterCommit;
    },
    set failTransactionAfterCommit(value: boolean) {
      failTransactionAfterCommit = value;
    },
  };
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

  rebuild(name: string): void {
    const current = this.durables.get(name);
    if (!current) throw new Error("durable object is missing");
    const durable = durableState(this.stateOptions, current.records);
    this.durables.set(name, durable);
    this.gates.set(
      name,
      new DeliverGate(durable.state, this.env, () => this.now),
    );
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
    IDENTITY_SECRET,
    PUBLIC_BASE_URL: "https://example.com",
    FROM_EMAIL: "daily@mail.arxiv-daily.top",
    FROM_NAME: "arXiv Daily",
    DAILY_QUOTA: String(options.quota ?? 5),
    BUILD_IDENTITY,
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

async function putBoundDeviceFixture(
  kv: ReturnType<typeof memoryKv>,
  env: Env & { _namespace?: MemoryNamespace },
  rawToken: string,
  email: string,
  overrides: Record<string, unknown> = {},
): Promise<void> {
  const identity = await hashDeviceToken(rawToken, env.TOKEN_SECRET);
  const now = env._namespace?.now ?? new Date("2026-08-10T00:00:00.000Z");
  kv._map.set(`device-v2:${identity}`, JSON.stringify({
    email,
    createdAt: new Date(now.getTime() + 1).toISOString(),
    deliveryGeneration: 2,
    protocolGeneration: PROTOCOL_GENERATION,
    buildIdentity: BUILD_IDENTITY,
    readyGeneration: READY_GENERATION,
    ...overrides,
  }));
}

function deviceDurables(env: Env & { _namespace?: MemoryNamespace }): MemoryDurable[] {
  return Array.from(env._namespace!.durables.entries())
    .filter(([name]) => name.startsWith("device-v2:"))
    .map(([, durable]) => durable);
}

function recipientDurables(env: Env & { _namespace?: MemoryNamespace }): MemoryDurable[] {
  return Array.from(env._namespace!.durables.entries())
    .filter(([name]) => name.startsWith("recipient-v2:"))
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
  }
  await putBoundDeviceFixture(
    kv,
    env,
    "device-token",
    "recipient@example.com",
  );
  return { kv, env };
}

function disclosedDeviceToken(body: string): string | undefined {
  return /<pre[^>]*>([0-9a-f]{64})<\/pre>/.exec(body)?.[1];
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

describe("public liveness and readiness contracts", () => {
  const LIVENESS = {
    ok: true,
    service: "arxiv-daily-email-relay",
    beta: true,
  };

  it("keeps root and health stable 200 without a DO binding or valid control", async () => {
    const withoutBinding = envWith(memoryKv(), { binding: false });
    for (const path of ["/", "/health"]) {
      const response = await worker.fetch(
        new Request(`https://relay.test${path}`),
        withoutBinding,
      );
      expect(response.status).toBe(200);
      expect(await response.json()).toEqual(LIVENESS);
    }

    const kv = memoryKv();
    const corrupted = envWith(kv);
    const control = corrupted._namespace!.get(
      corrupted._namespace!.idFromName("delivery-cutover:v3"),
    );
    corrupted._namespace!.durables.get("delivery-cutover:v3")!.records.set(
      "cutover-control:v3",
      { schemaVersion: 2, phase: "ready" },
    );
    expect((await control.fetch(
      new Request("https://cutover-control/cutover/status"),
    )).status).toBe(503);
    const health = await worker.fetch(
      new Request("https://relay.test/health"),
      corrupted,
    );
    expect(health.status).toBe(200);
    expect(await health.json()).toEqual(LIVENESS);
  });

  it("publishes only server-owned readiness identity and generation on ready", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const injectedBuild = "client-injected-build";

    const locked = await worker.fetch(new Request(
      `https://relay.test/ready?buildIdentity=${injectedBuild}&protocolGeneration=99`,
      { headers: { "X-Build-Identity": injectedBuild } },
    ), env);
    expect(locked.status).toBe(503);
    expect(await locked.json()).toEqual({
      protocolGeneration: PROTOCOL_GENERATION,
      buildIdentity: BUILD_IDENTITY,
      phase: "locked",
      automatic: "locked",
      readyGeneration: null,
    });

    await completeCutover(env);
    const ready = await worker.fetch(
      new Request("https://relay.test/ready", {
        headers: { "X-Build-Identity": injectedBuild },
      }),
      env,
    );
    const readyText = await ready.text();
    expect(ready.status).toBe(200);
    expect(JSON.parse(readyText)).toEqual({
      protocolGeneration: PROTOCOL_GENERATION,
      buildIdentity: BUILD_IDENTITY,
      phase: "ready",
      automatic: "ready",
      readyGeneration: READY_GENERATION,
    });
    expect(readyText).not.toContain(injectedBuild);
    expect(readyText).not.toContain(env.TOKEN_SECRET);
    expect(readyText).not.toContain(env.RESEND_API_KEY);
  });

  it("returns 503 for blocked control and while repair is pending", async () => {
    const blockedKv = memoryKv();
    const blocked = envWith(blockedKv);
    blockedKv._map.set("idemp:unsupported-ready-fixture", "private");
    expect((await applyCutoverAction(blocked, "inventory", 250)).status).toBe(503);
    const blockedReady = await worker.fetch(
      new Request("https://relay.test/ready"),
      blocked,
    );
    expect(blockedReady.status).toBe(503);
    expect(await blockedReady.json()).toEqual({ automatic: "locked" });

    const pendingKv = memoryKv();
    const pending = envWith(pendingKv);
    await completeCutover(pending);
    const durable = pending._namespace!.durables.get("delivery-cutover:v3")!;
    const control = durable.records.get("cutover-control:v3") as Record<string, unknown>;
    durable.records.set("cutover-control:v3", {
      ...control,
      pendingOperation: {
        operationId: "f".repeat(64),
        action: "repair",
        inputHash: await sha256Hex(JSON.stringify(["repair", null])),
        baseRevision: READY_GENERATION,
        basePhase: "ready",
        startedAt: pending._namespace!.now.toISOString(),
      },
    });
    const pendingReady = await worker.fetch(
      new Request("https://relay.test/ready"),
      pending,
    );
    expect(pendingReady.status).toBe(503);
    expect(await pendingReady.json()).toMatchObject({
      phase: "ready",
      automatic: "locked",
      readyGeneration: null,
    });
  });

  it("fails ready closed when an automatic runtime dependency is missing or invalid", async () => {
    const cases: Array<[string, (env: Env) => void]> = [
      ["durable binding", (env) => {
        env.DELIVER_GATE = undefined;
      }],
      ["token secret", (env) => {
        env.TOKEN_SECRET = " ";
      }],
      ["provider secret", (env) => {
        env.RESEND_API_KEY = " ";
      }],
      ["sender", (env) => {
        env.FROM_EMAIL = "not-an-email";
      }],
      ["public URL", (env) => {
        env.PUBLIC_BASE_URL = "not-a-url";
      }],
    ];

    for (const [label, invalidate] of cases) {
      const env = envWith(memoryKv());
      await completeCutover(env);
      invalidate(env);
      const response = await worker.fetch(
        new Request("https://relay.test/ready"),
        env,
      );
      expect(response.status, label).toBe(503);
      expect(await response.json(), label).toEqual({ automatic: "locked" });
    }
  });

  it("rejects non-200 or malformed internal cutover status as unready", async () => {
    const cases: Array<[string, number, Record<string, unknown>]> = [
      ["ready-shaped 503", 503, {
        schemaVersion: 3,
        phase: "ready",
        revision: READY_GENERATION,
        automatic: "ready",
      }],
      ["missing schema", 200, {
        phase: "ready",
        revision: READY_GENERATION,
        automatic: "ready",
      }],
      ["wrong schema", 200, {
        schemaVersion: 2,
        phase: "ready",
        revision: READY_GENERATION,
        automatic: "ready",
      }],
      ["invalid revision", 200, {
        schemaVersion: 3,
        phase: "ready",
        revision: Number.MAX_SAFE_INTEGER + 1,
        automatic: "ready",
      }],
    ];

    for (const [label, status, body] of cases) {
      const env = envWith(memoryKv(), { binding: false });
      env.DELIVER_GATE = {
        idFromName: () => ({ toString: () => "delivery-cutover:v3" }),
        get: () => ({
          fetch: async () => Response.json(body, { status }),
        }),
      } as unknown as DurableObjectNamespace;
      const response = await worker.fetch(
        new Request("https://relay.test/ready"),
        env,
      );
      expect(response.status, label).toBe(503);
      expect(await response.json(), label).toEqual({ automatic: "locked" });
    }
  });

  it("fails ready closed instead of accepting client identity when readiness is unverifiable", async () => {
    const injectedBuild = "client-injected-build";
    for (const env of [
      envWith(memoryKv(), { binding: false }),
      (() => {
        const configured = envWith(memoryKv());
        configured.BUILD_IDENTITY = "";
        return configured;
      })(),
      (() => {
        const malformed = envWith(memoryKv());
        malformed.BUILD_IDENTITY = "email-relay-v2-not-a-full-sha";
        return malformed;
      })(),
      (() => {
        const corrupted = envWith(memoryKv());
        corrupted._namespace!.get(
          corrupted._namespace!.idFromName("delivery-cutover:v3"),
        );
        corrupted._namespace!.durables.get("delivery-cutover:v3")!.records.set(
          "cutover-control:v3",
          { schemaVersion: 2, phase: "ready" },
        );
        return corrupted;
      })(),
    ]) {
      const response = await worker.fetch(new Request(
        `https://relay.test/ready?buildIdentity=${injectedBuild}&protocolGeneration=99`,
        { headers: { "X-Build-Identity": injectedBuild } },
      ), env);
      const text = await response.text();
      expect(response.status).toBe(503);
      expect(JSON.parse(text)).toEqual({ automatic: "locked" });
      expect(text).not.toContain(injectedBuild);
    }
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
    await putBoundDeviceFixture(kv, env, "device-token", "recipient@example.com");
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
    await putBoundDeviceFixture(kv, env, "device-token", "recipient@example.com");
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
    const firstKv = memoryKv();
    const first = envWith(firstKv);
    first.DELIVERY_V2_CUTOVER_TOKEN = TOKEN;
    await completeCutover(first);
    await putBoundDeviceFixture(
      firstKv,
      first,
      "device-token",
      "recipient@example.com",
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
    expect(mismatchRepair.status).toBe(503);
    expect(await mismatchRepair.json()).toMatchObject({
      phase: "locked",
      automatic: "locked",
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
    expect((await apply(third, "inventory", 30)).status).toBe(503);
    expect((await thirdControl.fetch(
      new Request("https://cutover-control/cutover/status"),
    )).status).toBe(503);
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

describe("ready-bound token issuance", () => {
  it("rejects verification before claim or KV mutation when automatic runtime config is invalid", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "invalid-runtime-pending-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    env.PUBLIC_BASE_URL = "http://relay.test";
    const control = env._namespace!.get(
      env._namespace!.idFromName("delivery-cutover:v3"),
    );
    const pendingIdentity = await hashPendingToken(rawPendingToken, env.TOKEN_SECRET);

    const readiness = await worker.fetch(
      new Request("https://relay.test/ready"),
      env,
    );
    const internal = await control.fetch(new Request(
      `https://cutover-control${CUTOVER_ISSUE_DEVICE_PATH}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pendingIdentity }),
      },
    ));
    const publicResponse = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );

    expect(readiness.status).toBe(503);
    expect(await readiness.json()).toEqual({ automatic: "locked" });
    expect(internal.status).toBe(503);
    expect(await internal.json()).toEqual({ status: "locked" });
    expect(publicResponse.status).toBe(503);
    expect(disclosedDeviceToken(await publicResponse.text())).toBeUndefined();
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    )).toHaveLength(1);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(0);
    expect(Array.from(
      env._namespace!.durables.get("delivery-cutover:v3")!.records.keys(),
    ).filter((key) => key.startsWith("issuance-claim:v1:"))).toHaveLength(0);
  });

  it("preserves pending verification before ready and binds the issued token after ready", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "private-pending-token";
    const privateRecipient = "private-recipient@example.com";
    await putPending(env, rawPendingToken, privateRecipient);
    const pendingKeys = () => Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    );

    const locked = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const lockedText = await locked.text();
    expect(locked.status).toBe(503);
    expect(pendingKeys()).toHaveLength(1);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(0);
    expect(lockedText).not.toContain(rawPendingToken);
    expect(lockedText).not.toContain(privateRecipient);

    await completeCutover(env);
    const issued = await worker.fetch(
      new Request(
        `https://relay.test/v1/verify?token=${rawPendingToken}` +
          "&buildIdentity=client-injected-build&protocolGeneration=99",
        { headers: { "X-Build-Identity": "client-injected-build" } },
      ),
      env,
    );
    const issuedText = await issued.text();
    expect(issued.status).toBe(200);
    expect(pendingKeys()).toHaveLength(0);
    expect(issuedText).not.toContain(rawPendingToken);
    expect(issuedText).not.toContain(privateRecipient);
    const storedDevices = Array.from(kv._map.entries()).filter(([key]) =>
      key.startsWith("device-v2:")
    );
    expect(storedDevices).toHaveLength(1);
    expect(JSON.parse(storedDevices[0]![1])).toEqual({
      email: privateRecipient,
      createdAt: expect.any(String),
      deliveryGeneration: 2,
      protocolGeneration: PROTOCOL_GENERATION,
      buildIdentity: BUILD_IDENTITY,
      readyGeneration: READY_GENERATION,
    });
  });

  it("preserves pending and discloses no token when device storage fails", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "device-put-failure-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    kv._failPutPrefixes.add("device-v2:");

    const response = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const body = await response.text();

    expect(response.status).toBe(503);
    expect(body).toContain("Verification unavailable");
    expect(body).not.toMatch(/<pre[^>]*>[0-9a-f]{64}<\/pre>/);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    )).toHaveLength(1);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(0);
  });

  it("discloses one deterministic token identity across concurrent clicks", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "concurrent-pending-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    let releaseDelete!: () => void;
    const heldDelete = new Promise<void>((resolve) => {
      releaseDelete = resolve;
    });
    let deleteStarted!: () => void;
    const startedDelete = new Promise<void>((resolve) => {
      deleteStarted = resolve;
    });
    kv._beforeDelete = async (key) => {
      if (!key.startsWith("pending:")) return;
      kv._beforeDelete = undefined;
      deleteStarted();
      await heldDelete;
    };
    const request = () => worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );

    const first = request();
    await startedDelete;
    const second = request();
    releaseDelete();
    const responses = await Promise.all([first, second]);
    const bodies = await Promise.all(responses.map((response) => response.text()));
    const disclosed = bodies.flatMap((body) =>
      Array.from(body.matchAll(/<pre[^>]*>([0-9a-f]{64})<\/pre>/g), (match) => match[1]!)
    );
    const devices = Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    );

    expect(responses.every((response) => response.status === 200)).toBe(true);
    expect(disclosed).toHaveLength(2);
    expect(new Set(disclosed).size).toBe(1);
    expect(devices).toHaveLength(1);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    )).toHaveLength(0);
    expect(Array.from(
      env._namespace!.durables.get("delivery-cutover:v3")!.records.keys(),
    ).filter((key) => key.startsWith("issuance-claim:v1:"))).toHaveLength(1);
  });

  it("replays one token across concurrent singleton issuance responses", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "singleton-concurrent-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    const pendingIdentity = await hashPendingToken(rawPendingToken, env.TOKEN_SECRET);
    let releaseDelete!: () => void;
    const heldDelete = new Promise<void>((resolve) => {
      releaseDelete = resolve;
    });
    let deleteStarted!: () => void;
    const startedDelete = new Promise<void>((resolve) => {
      deleteStarted = resolve;
    });
    kv._beforeDelete = async (key) => {
      if (!key.startsWith("pending:")) return;
      kv._beforeDelete = undefined;
      deleteStarted();
      await heldDelete;
    };

    const first = issueReadyBoundDevice(env, pendingIdentity);
    await startedDelete;
    const second = issueReadyBoundDevice(env, pendingIdentity);
    const secondOutcome = await Promise.race([
      second,
      new Promise<"queued">((resolve) => setTimeout(() => resolve("queued"), 20)),
    ]);
    releaseDelete();
    const outcomes = [await first, await second];

    expect(secondOutcome).toBe("queued");
    expect(outcomes.every((outcome) => outcome.status === "issued")).toBe(true);
    const tokens = outcomes.flatMap((outcome) =>
      outcome.status === "issued" ? [outcome.token] : []
    );
    expect(tokens).toHaveLength(2);
    expect(new Set(tokens).size).toBe(1);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(1);
    expect(Array.from(
      env._namespace!.durables.get("delivery-cutover:v3")!.records.keys(),
    ).filter((key) => key.startsWith("issuance-claim:v1:"))).toHaveLength(1);
  });

  it("holds repair behind issuance and binds the disclosed token to one ready generation", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "serialized-pending-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    let releaseDevicePut!: () => void;
    const heldDevicePut = new Promise<void>((resolve) => {
      releaseDevicePut = resolve;
    });
    let devicePutStarted!: () => void;
    const startedDevicePut = new Promise<void>((resolve) => {
      devicePutStarted = resolve;
    });
    kv._beforePut = async (key) => {
      if (!key.startsWith("device-v2:")) return;
      kv._beforePut = undefined;
      devicePutStarted();
      await heldDevicePut;
    };

    const verification = worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    await startedDevicePut;
    const repair = applyCutoverAction(env, "repair", 260);
    const repairTiming = await Promise.race([
      repair.then(() => "finished" as const),
      new Promise<"queued">((resolve) => setTimeout(() => resolve("queued"), 20)),
    ]);
    releaseDevicePut();
    const [issued, repaired] = await Promise.all([verification, repair]);

    expect(repairTiming).toBe("queued");
    expect(issued.status).toBe(200);
    expect(repaired.status).toBe(200);
    expect(await repaired.json()).toMatchObject({
      phase: "sealed",
      automatic: "locked",
    });
    const records = Array.from(kv._map.entries()).filter(([key]) =>
      key.startsWith("device-v2:")
    );
    expect(records).toHaveLength(1);
    expect(JSON.parse(records[0]![1])).toMatchObject({
      protocolGeneration: PROTOCOL_GENERATION,
      buildIdentity: BUILD_IDENTITY,
      readyGeneration: READY_GENERATION,
    });
  });

  it("replays one deterministic token despite stale pending reads and keeps a private permanent claim", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "stale-read-pending-token";
    const recipient = "stale-private@example.com";
    await putPending(env, rawPendingToken, recipient);
    await completeCutover(env);
    kv._staleAfterDeletePrefixes.add("pending:");

    const first = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const firstToken = disclosedDeviceToken(await first.text());
    const second = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const secondToken = disclosedDeviceToken(await second.text());

    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    expect(firstToken).toMatch(/^[0-9a-f]{64}$/);
    expect(secondToken).toBe(firstToken);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(1);
    const control = env._namespace!.durables.get("delivery-cutover:v3")!;
    const claims = Array.from(control.records.entries()).filter(([key]) =>
      key.startsWith("issuance-claim:v1:")
    );
    expect(claims).toHaveLength(1);
    expect(claims[0]![1]).toMatchObject({
      schemaVersion: 1,
      status: "issued",
      protocolGeneration: PROTOCOL_GENERATION,
      buildIdentity: BUILD_IDENTITY,
      readyGeneration: READY_GENERATION,
      createdAt: expect.any(String),
      pendingExpiresAt: expect.any(String),
      recipientIdentity: expect.stringMatching(/^[0-9a-f]{64}$/),
      pendingProof: expect.stringMatching(/^[0-9a-f]{64}$/),
    });
    const stored = JSON.stringify(Array.from(control.records));
    expect(stored).not.toContain(recipient);
    expect(stored).not.toContain(rawPendingToken);
    expect(stored).not.toContain(firstToken);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    )).toHaveLength(0);
  });

  it("replays the same token regardless of caller clock offset", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "clock-independent-replay-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    const first = await worker.fetch(
      new Request(
        `https://relay.test/v1/verify?token=${rawPendingToken}&requestStartedAt=999999999999999`,
        { headers: { "X-Request-Started-At": "999999999999999" } },
      ),
      env,
    );
    const firstToken = disclosedDeviceToken(await first.text());
    env._namespace!.rebuild("delivery-cutover:v3");
    const replay = await worker.fetch(
      new Request(
        `https://relay.test/v1/verify?token=${rawPendingToken}&requestStartedAt=-999999999999999`,
        { headers: { "X-Request-Started-At": "-999999999999999" } },
      ),
      env,
    );
    const replayToken = disclosedDeviceToken(await replay.text());

    expect(first.status).toBe(200);
    expect(replay.status).toBe(200);
    expect(replayToken).toBe(firstToken);
  });

  it("replays the same token after the first successful response is lost", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "lost-response-pending-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);

    const lost = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const lostToken = disclosedDeviceToken(await lost.text());
    env._namespace!.rebuild("delivery-cutover:v3");
    const replay = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const replayToken = disclosedDeviceToken(await replay.text());

    expect(lost.status).toBe(200);
    expect(replay.status).toBe(200);
    expect(replayToken).toBe(lostToken);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(1);
  });

  it.each([
    ["device put", "device-put"],
    ["unknown device put", "device-put-after"],
    ["pending delete", "pending-delete"],
    ["unknown pending delete", "pending-delete-after"],
    ["issued claim final write", "issued-write"],
  ] as const)("recovers after %s failure and DO rebuild", async (_label, failure) => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = `recover-${failure}-token`;
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    const control = env._namespace!.durables.get("delivery-cutover:v3")!;
    if (failure === "device-put") kv._failPutPrefixes.add("device-v2:");
    if (failure === "device-put-after") {
      kv._failPutAfterPrefixes.add("device-v2:");
    }
    if (failure === "pending-delete") kv._failDeletePrefixes.add("pending:");
    if (failure === "pending-delete-after") {
      kv._failDeleteAfterPrefixes.add("pending:");
    }
    if (failure === "issued-write") control.failStatusPuts.add("issued");

    const failed = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const failedBody = await failed.text();
    expect(failed.status).toBe(503);
    expect(disclosedDeviceToken(failedBody)).toBeUndefined();
    expect(Array.from(control.records.keys()).filter((key) =>
      key.startsWith("issuance-claim:v1:")
    )).toHaveLength(1);

    kv._failPutPrefixes.clear();
    kv._failPutAfterPrefixes.clear();
    kv._failDeletePrefixes.clear();
    kv._failDeleteAfterPrefixes.clear();
    control.failStatusPuts.clear();
    env._namespace!.rebuild("delivery-cutover:v3");
    const recovered = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const recoveredToken = disclosedDeviceToken(await recovered.text());

    expect(recovered.status).toBe(200);
    expect(recoveredToken).toMatch(/^[0-9a-f]{64}$/);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(1);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    )).toHaveLength(0);
  });

  it("recovers the same token when the issued claim put committed then threw", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "issued-write-unknown-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    const pendingIdentity = await hashPendingToken(rawPendingToken, env.TOKEN_SECRET);
    const expectedToken = await hmacSha256Hex(
      env.TOKEN_SECRET,
      `delivery-v2-device-token\u0000${pendingIdentity}`,
    );
    const control = env._namespace!.durables.get("delivery-cutover:v3")!;
    control.failStatusPutsAfter.add("issued");

    const failed = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const failedBody = await failed.text();
    expect(failed.status).toBe(503);
    expect(disclosedDeviceToken(failedBody)).toBeUndefined();
    expect(Array.from(control.records.values()).filter((value) =>
      (value as { status?: unknown }).status === "issued"
    )).toHaveLength(1);

    control.failStatusPutsAfter.clear();
    env._namespace!.rebuild("delivery-cutover:v3");
    const recovered = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );

    expect(recovered.status).toBe(200);
    expect(disclosedDeviceToken(await recovered.text())).toBe(expectedToken);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(1);
  });

  it("does not mutate KV when the initial durable claim write fails", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "claim-write-failure-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    const control = env._namespace!.durables.get("delivery-cutover:v3")!;
    control.failStatusPuts.add("claimed");

    const failed = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );

    expect(failed.status).toBe(503);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    )).toHaveLength(1);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(0);
    expect(Array.from(control.records.keys()).filter((key) =>
      key.startsWith("issuance-claim:v1:")
    )).toHaveLength(0);
  });

  it("recovers after the initial durable claim write outcome is unknown", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "claim-write-unknown-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    const control = env._namespace!.durables.get("delivery-cutover:v3")!;
    control.failPutAfterPrefixes.add("issuance-claim:v1:");

    const failed = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const failedBody = await failed.text();

    expect(failed.status).toBe(503);
    expect(disclosedDeviceToken(failedBody)).toBeUndefined();
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    )).toHaveLength(1);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(0);
    expect(Array.from(control.records.values()).filter((value) =>
      (value as { status?: unknown }).status === "claimed"
    )).toHaveLength(1);

    control.failPutAfterPrefixes.clear();
    env._namespace!.rebuild("delivery-cutover:v3");
    const recovered = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );

    expect(recovered.status).toBe(200);
    expect(disclosedDeviceToken(await recovered.text())).toMatch(/^[0-9a-f]{64}$/);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(1);
  });

  it("never trusts replacement pending content after a claim exists", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "replacement-pending-token";
    await putPending(env, rawPendingToken, "original-private@example.com");
    await completeCutover(env);
    kv._failPutPrefixes.add("device-v2:");
    expect((await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    )).status).toBe(503);
    kv._failPutPrefixes.clear();
    const pendingIdentity = await hashPendingToken(rawPendingToken, env.TOKEN_SECRET);
    const original = JSON.parse(kv._map.get(`pending:${pendingIdentity}`)!) as {
      createdAt: string;
      expiresAt: string;
    };
    kv._map.set(`pending:${pendingIdentity}`, JSON.stringify({
      email: "replacement-private@example.com",
      createdAt: original.createdAt,
      expiresAt: original.expiresAt,
    }));
    env._namespace!.rebuild("delivery-cutover:v3");

    const retry = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const retryText = await retry.text();

    expect(retry.status).toBe(503);
    expect(disclosedDeviceToken(retryText)).toBeUndefined();
    expect(retryText).not.toContain("original-private@example.com");
    expect(retryText).not.toContain("replacement-private@example.com");
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(0);
  });

  it("keeps an expired issued claim as a non-disclosing tombstone", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-10T00:00:00.000Z"));
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "expired-claim-token";
    await putPending(env, rawPendingToken, "recipient@example.com", 3600);
    await completeCutover(env);
    const issued = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    expect(issued.status).toBe(200);
    vi.setSystemTime(new Date("2026-08-10T01:00:01.000Z"));
    env._namespace!.now = new Date("2026-08-10T01:00:01.000Z");
    env._namespace!.rebuild("delivery-cutover:v3");

    const expired = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );
    const expiredText = await expired.text();

    expect(expired.status).toBe(400);
    expect(disclosedDeviceToken(expiredText)).toBeUndefined();
    expect(Array.from(
      env._namespace!.durables.get("delivery-cutover:v3")!.records.keys(),
    ).filter((key) => key.startsWith("issuance-claim:v1:"))).toHaveLength(1);
  });

  it("does not disclose an old claim after repair and re-ready changes generation", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPendingToken = "old-generation-claim-token";
    await putPending(env, rawPendingToken, "recipient@example.com");
    await completeCutover(env);
    expect((await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    )).status).toBe(200);
    expect((await applyCutoverAction(env, "repair", 270)).status).toBe(200);
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 60_000);
    expect((await applyCutoverAction(env, "observe", 271)).status).toBe(200);
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 60_000);
    expect((await applyCutoverAction(env, "observe", 272)).status).toBe(200);
    expect((await applyCutoverAction(env, "seal", 273)).status).toBe(200);

    const retry = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPendingToken}`),
      env,
    );

    expect(retry.status).toBe(503);
    expect(disclosedDeviceToken(await retry.text())).toBeUndefined();
  });

  it("sets no-store and no-referrer on every verification HTML response", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await putPending(env, "cache-pending-token", "recipient@example.com");
    const locked = await worker.fetch(
      new Request("https://relay.test/v1/verify?token=cache-pending-token"),
      env,
    );
    const missing = await worker.fetch(
      new Request("https://relay.test/v1/verify"),
      env,
    );
    await completeCutover(env);
    const success = await worker.fetch(
      new Request("https://relay.test/v1/verify?token=cache-pending-token"),
      env,
    );
    const replay = await worker.fetch(
      new Request("https://relay.test/v1/verify?token=unknown-token"),
      env,
    );
    for (const response of [locked, missing, success, replay]) {
      expect(response.headers.get("Cache-Control")).toBe("no-store");
      expect(response.headers.get("Referrer-Policy")).toBe("no-referrer");
    }
  });

  it("keeps test delivery valid for an authenticated build-mismatch token while automatic fails closed", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    await putBoundDeviceFixture(
      kv,
      env,
      "previous-build-token",
      "recipient@example.com",
      { buildIdentity: "previous-relay-build" },
    );
    const provider = providerSuccess("test_only");
    vi.stubGlobal("fetch", provider);

    const test = await worker.fetch(deliveryRequest({
      token: "previous-build-token",
      key: TEST_KEY_A,
    }), env);
    const automatic = await worker.fetch(deliveryRequest({
      token: "previous-build-token",
      key: AUTO_KEY_A,
    }), env);

    expect(test.status).toBe(200);
    expect(automatic.status).toBe(503);
    expect(provider).toHaveBeenCalledTimes(1);
  });

  it.each(["done", "legacy"] as const)(
    "authorizes a mismatch token before replaying an existing %s ledger",
    async (ledgerKind) => {
      const kv = memoryKv();
      const env = envWith(kv);
      if (ledgerKind === "legacy") {
        await setLegacyEvidence(kv, "done:legacy");
      }
      await completeCutover(env);
      await putBoundDeviceFixture(kv, env, "ready-token", "recipient@example.com");
      await putBoundDeviceFixture(
        kv,
        env,
        "mismatch-token",
        "recipient@example.com",
        { buildIdentity: "previous-relay-build" },
      );
      const provider = providerSuccess("done_once");
      vi.stubGlobal("fetch", provider);
      const established = await worker.fetch(
        deliveryRequest({ token: "ready-token" }),
        env,
      );
      expect(established.status).toBe(ledgerKind === "done" ? 200 : 409);

      const mismatch = await worker.fetch(
        deliveryRequest({ token: "mismatch-token" }),
        env,
      );

      expect(mismatch.status).toBe(503);
      expect(await mismatch.json()).toEqual({
        error: "delivery cutover is not ready",
        ambiguous: false,
      });
      expect(provider).toHaveBeenCalledTimes(ledgerKind === "done" ? 1 : 0);
    },
  );

  it.each([
    ["legacy", { deliveryGeneration: undefined }],
    ["pre-ready", { protocolGeneration: undefined, buildIdentity: undefined, readyGeneration: undefined }],
    ["generation mismatch", { readyGeneration: READY_GENERATION - 1 }],
    ["build mismatch", { buildIdentity: "previous-relay-build" }],
    ["protocol mismatch", { protocolGeneration: PROTOCOL_GENERATION - 1 }],
  ] as const)("fails automatic closed for a %s token", async (_label, overrides) => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    if (_label === "legacy") {
      const identity = await hashDeviceToken("mismatched-token", env.TOKEN_SECRET);
      kv._map.set(`device:${identity}`, JSON.stringify({
        email: "recipient@example.com",
        createdAt: new Date(env._namespace!.now.getTime() + 1).toISOString(),
      }));
    } else {
      await putBoundDeviceFixture(
        kv,
        env,
        "mismatched-token",
        "recipient@example.com",
        overrides,
      );
    }
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest({ token: "mismatched-token" }), env);

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "delivery cutover is not ready",
      ambiguous: false,
    });
    expect(provider).not.toHaveBeenCalled();
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
    "imports scanned %s evidence into a permanent recipient ledger block",
    async (legacyValue, error, ambiguous) => {
      const { kv, env } = await authenticatedEnv({ quota: 5, legacyValue });
      const auditMarker = kv._map.get(DELIVERY_V3_CUTOVER_AUDIT_KEY)!;
      kv._map.clear();
      kv._map.set(DELIVERY_V3_CUTOVER_AUDIT_KEY, auditMarker);
      await putBoundDeviceFixture(
        kv,
        env,
        "device-token",
        "recipient@example.com",
      );
      const provider = providerSuccess("must_not_send");
      vi.stubGlobal("fetch", provider);

      const imported = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
      const replay = await worker.fetch(deliveryRequest({ key: AUTO_KEY_B }), env);

      expect(imported.status).toBe(409);
      expect(await imported.json()).toEqual({ error, ambiguous });
      expect(replay.status).toBe(409);
      expect(await replay.json()).toEqual({ error, ambiguous });
      expect(provider).not.toHaveBeenCalled();
      const durable = recipientDurables(env)[0]!;
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
    const identity = await hashRecipientIdentity("recipient@example.com", IDENTITY_SECRET);
    env._namespace!.get(env._namespace!.idFromName(`recipient-v2:${identity}`));
    const durable = env._namespace!.durables.get(`recipient-v2:${identity}`)!;
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
    const identity = await hashRecipientIdentity("recipient@example.com", IDENTITY_SECRET);
    env._namespace!.get(env._namespace!.idFromName(`recipient-v2:${identity}`));
    const durable = env._namespace!.durables.get(`recipient-v2:${identity}`)!;
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
    await putBoundDeviceFixture(
      kv,
      env,
      "device-token",
      "recipient@example.com",
      {
        protocolGeneration: undefined,
        buildIdentity: undefined,
        readyGeneration: undefined,
      },
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

describe("automatic runtime gate", () => {
  it.each([
    ["non-HTTPS public URL", (env: Env) => {
      env.PUBLIC_BASE_URL = "http://relay.test";
    }],
    ["invalid sender", (env: Env) => {
      env.FROM_EMAIL = "not-an-email";
    }],
  ] as const)("blocks public automatic delivery for %s before recipient DO and provider", async (_label, invalidate) => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    await putBoundDeviceFixture(kv, env, "device-token", "recipient@example.com");
    invalidate(env);
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "automatic delivery is unavailable",
      ambiguous: false,
    });
    expect(provider).not.toHaveBeenCalled();
    expect(recipientDurables(env)).toHaveLength(0);
  });

  it("denies direct internal automatic authorization when provider config is missing", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    env.RESEND_API_KEY = " ";
    const control = env._namespace!.get(
      env._namespace!.idFromName("delivery-cutover:v3"),
    );

    const response = await control.fetch(new Request(
      `https://cutover-control${CUTOVER_AUTOMATIC_PATH}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          evidenceIdentity: "e".repeat(64),
          deliveryGeneration: 2,
          protocolGeneration: PROTOCOL_GENERATION,
          buildIdentity: BUILD_IDENTITY,
          readyGeneration: READY_GENERATION,
        }),
      },
    ));

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ authorized: false });
  });

  it("keeps test delivery device-scoped when automatic runtime config is invalid", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await putBoundDeviceFixture(kv, env, "device-token", "recipient@example.com");
    env.BUILD_IDENTITY = "invalid-build";
    env.PUBLIC_BASE_URL = "http://relay.test";
    const provider = providerSuccess("test_only");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest({ key: TEST_KEY_A }), env);

    expect(response.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(1);
    expect(deviceDurables(env)).toHaveLength(1);
    expect(recipientDurables(env)).toHaveLength(0);
  });
});

describe("stable automatic identity secret", () => {
  it("keeps one recipient ledger and provider key across TOKEN_SECRET rotation", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    await putPending(env, "first-rotation-pending", "recipient@example.com");
    const firstVerification = await worker.fetch(
      new Request("https://relay.test/v1/verify?token=first-rotation-pending"),
      env,
    );
    const firstToken = disclosedDeviceToken(await firstVerification.text())!;
    const provider = providerSuccess("one_physical_delivery");
    vi.stubGlobal("fetch", provider);
    const firstDelivery = await worker.fetch(deliveryRequest({ token: firstToken }), env);
    const firstProviderKey = (provider.mock.calls[0]![1]?.headers as Record<string, string>)[
      "Idempotency-Key"
    ];

    env.TOKEN_SECRET = "rotated-token-secret-v2";
    expect((await worker.fetch(deliveryRequest({ token: firstToken }), env)).status).toBe(401);
    await putPending(env, "second-rotation-pending", "recipient@example.com");
    const secondVerification = await worker.fetch(
      new Request("https://relay.test/v1/verify?token=second-rotation-pending"),
      env,
    );
    const secondToken = disclosedDeviceToken(await secondVerification.text())!;
    const secondDelivery = await worker.fetch(
      deliveryRequest({ token: secondToken, key: AUTO_KEY_B }),
      env,
    );

    expect(firstDelivery.status).toBe(200);
    expect(secondVerification.status).toBe(200);
    expect(secondDelivery.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(1);
    expect((provider.mock.calls[0]![1]?.headers as Record<string, string>)[
      "Idempotency-Key"
    ]).toBe(firstProviderKey);
    expect(new Set(env._namespace!.names.filter((name) =>
      name.startsWith("recipient-v2:")
    )).size).toBe(1);
  });

  it("locks every automatic gate on IDENTITY_SECRET rotation and recovers without rebinding", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    await putBoundDeviceFixture(kv, env, "device-token", "recipient@example.com");
    await putPending(env, "identity-rotation-pending", "recipient@example.com");
    const originalIdentitySecret = env.IDENTITY_SECRET!;
    env.IDENTITY_SECRET =
      "identity-secret-wrong-test-v2";
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);
    const control = env._namespace!.get(
      env._namespace!.idFromName("delivery-cutover:v3"),
    );
    const controlRecords = env._namespace!.durables.get("delivery-cutover:v3")!.records;
    const beforeAction = JSON.stringify(controlRecords.get("cutover-control:v3"));

    const status = await fetchCutoverStatus(env);
    const ready = await worker.fetch(new Request("https://relay.test/ready"), env);
    const verification = await worker.fetch(
      new Request("https://relay.test/v1/verify?token=identity-rotation-pending"),
      env,
    );
    const internal = await control.fetch(new Request(
      `https://cutover-control${CUTOVER_AUTOMATIC_PATH}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          evidenceIdentity: "e".repeat(64),
          deliveryGeneration: 2,
          protocolGeneration: PROTOCOL_GENERATION,
          buildIdentity: BUILD_IDENTITY,
          readyGeneration: READY_GENERATION,
        }),
      },
    ));
    const automatic = await worker.fetch(deliveryRequest(), env);
    const action = await applyCutoverAction(env, "inventory", 329);

    expect(status.status).toBe(503);
    expect(await status.json()).toEqual({
      schemaVersion: 3,
      phase: "locked",
      automatic: "locked",
    });
    expect(ready.status).toBe(503);
    expect(verification.status).toBe(503);
    expect(disclosedDeviceToken(await verification.text())).toBeUndefined();
    expect(internal.status).toBe(503);
    expect(await internal.json()).toEqual({ authorized: false });
    expect(automatic.status).toBe(503);
    expect(action.status).toBe(503);
    expect(JSON.stringify(controlRecords.get("cutover-control:v3"))).toBe(beforeAction);
    expect(recipientDurables(env)).toHaveLength(0);
    expect(provider).not.toHaveBeenCalled();
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    )).toHaveLength(1);
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("device-v2:")
    )).toHaveLength(1);

    env.IDENTITY_SECRET = originalIdentitySecret;
    const recoveredReady = await worker.fetch(new Request("https://relay.test/ready"), env);
    expect(recoveredReady.status).toBe(200);
  });

  it("repairs marker state across TOKEN_SECRET rotation but rejects another identity secret", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    const control = env._namespace!.durables.get("delivery-cutover:v3")!;
    const validControl = structuredClone(control.records.get("cutover-control:v3"));
    env.TOKEN_SECRET = "rotated-token-secret-for-marker";
    control.records.set("cutover-control:v3", { schemaVersion: 2, phase: "ready" });

    const repaired = await applyCutoverAction(env, "repair", 330);
    expect(repaired.status).toBe(200);
    expect(await repaired.json()).toMatchObject({ phase: "sealed", automatic: "locked" });

    control.records.set("cutover-control:v3", validControl);
    env.IDENTITY_SECRET = "wrong-identity-secret";
    control.records.set("cutover-control:v3", { schemaVersion: 2, phase: "ready" });
    const mismatchedStatus = await fetchCutoverStatus(env);
    const rejected = await applyCutoverAction(env, "repair", 331);
    const rebound = await applyCutoverAction(env, "inventory", 332);
    expect(mismatchedStatus.status).toBe(503);
    expect(await mismatchedStatus.json()).toEqual({
      schemaVersion: 3,
      phase: "locked",
      automatic: "locked",
    });
    expect(rejected.status).toBe(503);
    expect(rebound.status).toBe(503);

    env.IDENTITY_SECRET = IDENTITY_SECRET;
    const recovered = await applyCutoverAction(env, "repair", 333);
    expect(recovered.status).toBe(200);
  });

  it("locks actions without mutation when only another identity secret's marker remains", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    const durable = env._namespace!.durables.get("delivery-cutover:v3")!;
    durable.records.delete("cutover-control:v3");
    const markerBefore = kv._map.get(DELIVERY_V3_CUTOVER_AUDIT_KEY)!;
    const recordsBefore = JSON.stringify(Array.from(durable.records));
    const originalIdentitySecret = env.IDENTITY_SECRET!;
    const markerFingerprint = (JSON.parse(markerBefore) as {
      identitySecretFingerprint: string;
    }).identitySecretFingerprint;
    env.IDENTITY_SECRET = "identity-secret-missing-control-wrong-v2";

    const status = await fetchCutoverStatus(env);
    const inventory = await applyCutoverAction(env, "inventory", 0);
    const fence = await applyCutoverAction(env, "provider-fence", 1);
    const responseTexts = [
      await status.text(),
      await inventory.text(),
      await fence.text(),
    ];

    expect(status.status).toBe(503);
    expect(inventory.status).toBe(503);
    expect(fence.status).toBe(503);
    expect(kv._map.get(DELIVERY_V3_CUTOVER_AUDIT_KEY)).toBe(markerBefore);
    expect(JSON.stringify(Array.from(durable.records))).toBe(recordsBefore);
    for (const text of responseTexts) {
      expect(text).not.toContain(markerFingerprint);
      expect(text).not.toContain(originalIdentitySecret);
      expect(text).not.toContain(env.IDENTITY_SECRET);
    }

    env.IDENTITY_SECRET = originalIdentitySecret;
    const restoredInventory = await applyCutoverAction(env, "inventory", 0);
    const repaired = await applyCutoverAction(env, "repair", 334);
    expect(restoredInventory.status).toBe(503);
    expect(repaired.status).toBe(200);
    expect(await repaired.json()).toMatchObject({
      phase: "sealed",
      automatic: "locked",
    });
  });

  it("permits only marker repair when control is missing under the bound identity", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    const durable = env._namespace!.durables.get("delivery-cutover:v3")!;
    durable.records.delete("cutover-control:v3");
    const markerBefore = kv._map.get(DELIVERY_V3_CUTOVER_AUDIT_KEY)!;
    const recordsBefore = JSON.stringify(Array.from(durable.records));

    const inventory = await applyCutoverAction(env, "inventory", 336);
    expect(inventory.status).toBe(503);
    expect(kv._map.get(DELIVERY_V3_CUTOVER_AUDIT_KEY)).toBe(markerBefore);
    expect(JSON.stringify(Array.from(durable.records))).toBe(recordsBefore);

    const repaired = await applyCutoverAction(env, "repair", 337);
    expect(repaired.status).toBe(200);
    expect(await repaired.json()).toMatchObject({
      phase: "sealed",
      automatic: "locked",
    });
    expect(durable.records.get("cutover-control:v3")).toMatchObject({
      phase: "sealed",
      identitySecretFingerprint: expect.stringMatching(/^[0-9a-f]{64}$/),
    });
  });

  it("does not overwrite a bound marker with invalid proof when control is missing", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    const durable = env._namespace!.durables.get("delivery-cutover:v3")!;
    durable.records.delete("cutover-control:v3");
    const validMarker = JSON.parse(
      kv._map.get(DELIVERY_V3_CUTOVER_AUDIT_KEY)!,
    ) as Record<string, unknown>;
    const corruptedMarker = JSON.stringify({
      ...validMarker,
      proof: "f".repeat(64),
    });
    kv._map.set(DELIVERY_V3_CUTOVER_AUDIT_KEY, corruptedMarker);
    const recordsBefore = JSON.stringify(Array.from(durable.records));

    const inventory = await applyCutoverAction(env, "inventory", 338);
    const text = await inventory.text();

    expect(inventory.status).toBe(503);
    expect(kv._map.get(DELIVERY_V3_CUTOVER_AUDIT_KEY)).toBe(corruptedMarker);
    expect(JSON.stringify(Array.from(durable.records))).toBe(recordsBefore);
    expect(text).not.toContain(String(validMarker.identitySecretFingerprint));
    expect(text).not.toContain(env.IDENTITY_SECRET);
  });

  it("keeps permanent binding authoritative after control and marker loss", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const first = await applyCutoverAction(env, "inventory", 360);
    expect(first.status).toBe(200);
    const durable = env._namespace!.durables.get("delivery-cutover:v3")!;
    const bindingBefore = JSON.stringify(durable.records.get(CUTOVER_BINDING_KEY));
    const sentinelBefore = JSON.stringify(durable.records.get(CUTOVER_STATE_INDEX_KEY));
    expect(bindingBefore).not.toBeUndefined();
    expect(sentinelBefore).not.toBeUndefined();
    durable.records.delete("cutover-control:v3");
    kv._map.delete(DELIVERY_V3_CUTOVER_AUDIT_KEY);
    const recordsBefore = JSON.stringify(Array.from(durable.records));
    const originalIdentitySecret = env.IDENTITY_SECRET!;
    env.IDENTITY_SECRET = "identity-secret-no-control-or-marker-v2";

    const newInventory = await applyCutoverAction(env, "inventory", 361);
    const oldReplay = await applyCutoverAction(env, "inventory", 360);
    expect(newInventory.status).toBe(503);
    expect(oldReplay.status).toBe(503);
    expect(JSON.stringify(durable.records.get(CUTOVER_BINDING_KEY))).toBe(bindingBefore);
    expect(JSON.stringify(durable.records.get(CUTOVER_STATE_INDEX_KEY))).toBe(sentinelBefore);
    expect(JSON.stringify(Array.from(durable.records))).toBe(recordsBefore);

    env.IDENTITY_SECRET = originalIdentitySecret;
    expect((await applyCutoverAction(env, "inventory", 362)).status).toBe(503);
    expect((await applyCutoverAction(env, "repair", 363)).status).toBe(503);
    expect(JSON.stringify(durable.records.get(CUTOVER_BINDING_KEY))).toBe(bindingBefore);
  });

  it("locks build rotation at every gate and recovers on the bound build", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    await putBoundDeviceFixture(kv, env, "device-token", "recipient@example.com");
    await putPending(env, "build-rotation-pending", "recipient@example.com");
    const originalBuild = env.BUILD_IDENTITY!;
    env.BUILD_IDENTITY = `email-relay-v2-${"b".repeat(40)}`;
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);
    const control = env._namespace!.get(
      env._namespace!.idFromName("delivery-cutover:v3"),
    );

    const status = await fetchCutoverStatus(env);
    const ready = await worker.fetch(new Request("https://relay.test/ready"), env);
    const verification = await worker.fetch(
      new Request("https://relay.test/v1/verify?token=build-rotation-pending"),
      env,
    );
    const internal = await control.fetch(new Request(
      `https://cutover-control${CUTOVER_AUTOMATIC_PATH}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          evidenceIdentity: "e".repeat(64),
          deliveryGeneration: 2,
          protocolGeneration: PROTOCOL_GENERATION,
          buildIdentity: env.BUILD_IDENTITY,
          readyGeneration: READY_GENERATION,
        }),
      },
    ));
    const automatic = await worker.fetch(deliveryRequest(), env);
    const action = await applyCutoverAction(env, "inventory", 364);

    expect(status.status).toBe(503);
    expect(ready.status).toBe(503);
    expect(verification.status).toBe(503);
    expect(disclosedDeviceToken(await verification.text())).toBeUndefined();
    expect(internal.status).toBe(503);
    expect(automatic.status).toBe(503);
    expect(action.status).toBe(503);
    expect(provider).not.toHaveBeenCalled();
    expect(Array.from(kv._map.keys()).filter((key) =>
      key.startsWith("pending:")
    )).toHaveLength(1);

    env.BUILD_IDENTITY = originalBuild;
    expect((await worker.fetch(new Request("https://relay.test/ready"), env)).status).toBe(200);
  });

  it("recovers marker repair after its pending control write outcome is unknown", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    let durable = env._namespace!.durables.get("delivery-cutover:v3")!;
    durable.records.delete("cutover-control:v3");
    durable.failPutAfterPrefixes.add("cutover-control:v3");

    await expect(applyCutoverAction(env, "repair", 390)).rejects.toThrow(
      "injected DO put outcome unknown",
    );
    durable.failPutAfterPrefixes.clear();
    env._namespace!.rebuild("delivery-cutover:v3");
    durable = env._namespace!.durables.get("delivery-cutover:v3")!;

    const takeover = await applyCutoverAction(env, "repair", 391);
    const recovered = await applyCutoverAction(env, "repair", 390);

    expect(takeover.status).toBe(409);
    expect(await takeover.json()).toMatchObject({
      phase: "locked",
      automatic: "locked",
    });
    expect(recovered.status).toBe(200);
    expect(await recovered.json()).toMatchObject({
      phase: "sealed",
      automatic: "locked",
    });
    expect(durable.records.get("cutover-control:v3")).toMatchObject({
      phase: "sealed",
    });
  });

  it("recovers one binding and operation after binding write failures and unknown outcome", async () => {
    for (const outcome of ["before", "after"] as const) {
      const kv = memoryKv();
      const env = envWith(kv);
      env._namespace!.get(env._namespace!.idFromName("delivery-cutover:v3"));
      let durable = env._namespace!.durables.get("delivery-cutover:v3")!;
      if (outcome === "before") {
        durable.failPutPrefixes.add(CUTOVER_BINDING_KEY);
      } else {
        durable.failTransactionAfterCommit = true;
      }

      await expect(applyCutoverAction(env, "inventory", 365)).rejects.toThrow();
      if (outcome === "before") {
        durable.failPutPrefixes.clear();
      } else {
        durable.failTransactionAfterCommit = false;
      }
      env._namespace!.rebuild("delivery-cutover:v3");
      durable = env._namespace!.durables.get("delivery-cutover:v3")!;
      const recovered = await applyCutoverAction(env, "inventory", 365);

      expect(recovered.status, outcome).toBe(200);
      expect(await recovered.json(), outcome).toMatchObject({
        phase: "inventoried",
        automatic: "locked",
      });
      const binding = durable.records.get(CUTOVER_BINDING_KEY) as Record<string, unknown>;
      expect(binding, outcome).toMatchObject({
        schemaVersion: 1,
        buildIdentity: BUILD_IDENTITY,
        protocolGeneration: PROTOCOL_GENERATION,
        identitySecretFingerprint: expect.stringMatching(/^[0-9a-f]{64}$/),
        boundAt: expect.any(String),
      });
      expect(JSON.stringify(binding), outcome).not.toContain(IDENTITY_SECRET);
    }
  });

  it("never treats missing binding as fresh when any v3 state remains", async () => {
    for (const survivor of [
      "state-index",
      "control",
      "operation",
      "claim",
      "marker",
    ] as const) {
      const kv = memoryKv();
      const env = envWith(kv);
      await completeCutover(env);
      if (survivor === "claim") {
        await putPending(env, "binding-missing-claim", "recipient@example.com");
        expect((await worker.fetch(
          new Request("https://relay.test/v1/verify?token=binding-missing-claim"),
          env,
        )).status).toBe(200);
      }
      const durable = env._namespace!.durables.get("delivery-cutover:v3")!;
      durable.records.delete(CUTOVER_BINDING_KEY);
      const entries = Array.from(durable.records.entries());
      for (const [key] of entries) {
        const keep =
          (survivor === "state-index" && key === CUTOVER_STATE_INDEX_KEY) ||
          (survivor === "control" && key === "cutover-control:v3") ||
          (survivor === "operation" && key.startsWith("cutover-operation:v3:")) ||
          (survivor === "claim" && key.startsWith("issuance-claim:v1:"));
        if (!keep) durable.records.delete(key);
      }
      if (survivor !== "marker") kv._map.delete(DELIVERY_V3_CUTOVER_AUDIT_KEY);

      const status = await fetchCutoverStatus(env);
      const inventory = await applyCutoverAction(env, "inventory", 366);
      expect(status.status, survivor).toBe(503);
      expect(inventory.status, survivor).toBe(503);
      expect(durable.records.has(CUTOVER_BINDING_KEY), survivor).toBe(false);
    }
  });

  it("requires binding, build, protocol, and identity agreement for marker repair", async () => {
    for (const field of [
      "identitySecretFingerprint",
      "buildIdentity",
      "protocolGeneration",
    ] as const) {
      const kv = memoryKv();
      const env = envWith(kv);
      await completeCutover(env);
      const durable = env._namespace!.durables.get("delivery-cutover:v3")!;
      durable.records.delete("cutover-control:v3");
      const binding = structuredClone(
        durable.records.get(CUTOVER_BINDING_KEY),
      ) as Record<string, unknown>;
      durable.records.set(CUTOVER_BINDING_KEY, {
        ...binding,
        [field]: field === "identitySecretFingerprint"
          ? "f".repeat(64)
          : field === "buildIdentity"
          ? `email-relay-v2-${"c".repeat(40)}`
          : 3,
      });

      const repaired = await applyCutoverAction(env, "repair", 367);
      expect(repaired.status, field).toBe(503);
      expect(durable.records.has("cutover-control:v3"), field).toBe(false);
    }
  });

  it("stores only irreversible identity binding and no raw secret, email, or token", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const rawPending = "private-identity-binding-pending";
    const privateEmail = "private-identity-binding@example.com";
    await putPending(env, rawPending, privateEmail);
    await completeCutover(env);
    const verification = await worker.fetch(
      new Request(`https://relay.test/v1/verify?token=${rawPending}`),
      env,
    );
    const deviceToken = disclosedDeviceToken(await verification.text())!;
    const control = env._namespace!.durables.get("delivery-cutover:v3")!;
    const status = await fetchCutoverStatus(env);
    const durableText = JSON.stringify(Array.from(control.records));
    const markerText = kv._map.get(DELIVERY_V3_CUTOVER_AUDIT_KEY)!;
    const statusText = await status.text();
    const identitySecret = env.IDENTITY_SECRET!;

    expect(durableText).toContain("identitySecretFingerprint");
    expect(markerText).toContain("identitySecretFingerprint");
    for (const text of [durableText, markerText, statusText]) {
      expect(text).not.toContain(identitySecret);
      expect(text).not.toContain(privateEmail);
      expect(text).not.toContain(rawPending);
      expect(text).not.toContain(deviceToken);
    }
    expect(statusText).not.toContain("identitySecretFingerprint");
  });

  it("keeps legacy evidence and authorization identity stable across TOKEN_SECRET rotation", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await setLegacyEvidence(kv, "done:legacy-provider-id");
    const before = await scanLegacyAutoDeliveryEvidence(env);
    const beforeIdentity = Object.keys(before)[0]!;
    await completeCutover(env);
    const beforeDecision = await authorizeAutomaticDelivery(env, {
      evidenceIdentity: beforeIdentity,
      deliveryGeneration: 2,
      protocolGeneration: PROTOCOL_GENERATION,
      buildIdentity: BUILD_IDENTITY,
      readyGeneration: READY_GENERATION,
    });

    env.TOKEN_SECRET = "rotated-token-secret-for-legacy";
    const after = await scanLegacyAutoDeliveryEvidence(env);
    const afterIdentity = Object.keys(after)[0]!;
    const afterDecision = await authorizeAutomaticDelivery(env, {
      evidenceIdentity: afterIdentity,
      deliveryGeneration: 2,
      protocolGeneration: PROTOCOL_GENERATION,
      buildIdentity: BUILD_IDENTITY,
      readyGeneration: READY_GENERATION,
    });

    expect(afterIdentity).toBe(beforeIdentity);
    expect(beforeDecision).toEqual({ authorized: true, legacyEvidence: "done" });
    expect(afterDecision).toEqual(beforeDecision);
  });

  it("fails inventory closed without IDENTITY_SECRET before provider fencing", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    delete env.IDENTITY_SECRET;

    const inventory = await applyCutoverAction(env, "inventory", 340);
    const fence = await applyCutoverAction(env, "provider-fence", 341);

    expect(inventory.status).toBe(503);
    expect(fence.status).not.toBe(200);
    const records = Array.from(
      env._namespace!.durables.get("delivery-cutover:v3")!.records.values(),
    );
    expect(JSON.stringify(records)).not.toContain("old_resend_credential_revoked");
  });

  it("reuses the first inventory identity binding after its durable write outcome is unknown", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    env._namespace!.get(env._namespace!.idFromName("delivery-cutover:v3"));
    const durable = env._namespace!.durables.get("delivery-cutover:v3")!;
    durable.failTransactionAfterCommit = true;

    await expect(applyCutoverAction(env, "inventory", 350)).rejects.toThrow(
      "injected DO transaction outcome unknown",
    );
    const pending = JSON.stringify(durable.records.get("cutover-control:v3"));
    expect(pending).toContain("identitySecretFingerprint");
    expect(pending).not.toContain(IDENTITY_SECRET);

    durable.failTransactionAfterCommit = false;
    env._namespace!.rebuild("delivery-cutover:v3");
    const retried = await applyCutoverAction(env, "inventory", 350);

    expect(retried.status).toBe(200);
    expect(await retried.json()).toMatchObject({
      phase: "inventoried",
      revision: 1,
      automatic: "locked",
    });
  });
});

describe("delivery scope routing", () => {
  it("routes concurrent automatic requests from multiple tokens to one recipient ledger and provider key", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    await putBoundDeviceFixture(kv, env, "device-one", "recipient@example.com");
    await putBoundDeviceFixture(kv, env, "device-two", "recipient@example.com");
    const firstIdentity = await hashDeviceToken("device-one", env.TOKEN_SECRET);
    const secondIdentity = await hashDeviceToken("device-two", env.TOKEN_SECRET);
    const recipientIdentity = await hashRecipientIdentity(
      "recipient@example.com",
      IDENTITY_SECRET,
    );
    const provider = providerSuccess("msg_once");
    vi.stubGlobal("fetch", provider);

    const [first, second] = await Promise.all([
      worker.fetch(deliveryRequest({ token: "device-one", key: AUTO_KEY_A }), env),
      worker.fetch(deliveryRequest({ token: "device-two", key: AUTO_KEY_B }), env),
    ]);

    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(1);
    const recipientNames = env._namespace!.names.filter((name) =>
      name.startsWith("recipient-v2:")
    );
    expect(new Set(recipientNames)).toEqual(
      new Set([`recipient-v2:${recipientIdentity}`]),
    );
    const stored = JSON.stringify(Array.from(recipientDurables(env)[0]!.records));
    expect(stored).not.toContain(firstIdentity);
    expect(stored).not.toContain(secondIdentity);
    expect(stored).not.toContain("recipient@example.com");
  });

  it("shares one automatic quota across multiple tokens for the recipient", async () => {
    const kv = memoryKv();
    const env = envWith(kv, { quota: 1 });
    await completeCutover(env);
    await putBoundDeviceFixture(kv, env, "device-one", "recipient@example.com");
    await putBoundDeviceFixture(kv, env, "device-two", "recipient@example.com");
    const provider = providerSuccess("msg_only");
    vi.stubGlobal("fetch", provider);

    const responses = await Promise.all([
      worker.fetch(deliveryRequest({ token: "device-one", date: "2026-08-10" }), env),
      worker.fetch(deliveryRequest({ token: "device-two", date: "2026-08-11" }), env),
    ]);

    expect(responses.map((response) => response.status).sort()).toEqual([200, 429]);
    expect(provider).toHaveBeenCalledTimes(1);
  });

  it("keeps test requests from multiple tokens in device-scoped ledgers", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await putBoundDeviceFixture(kv, env, "device-one", "recipient@example.com");
    await putBoundDeviceFixture(kv, env, "device-two", "recipient@example.com");
    const provider = providerSuccess("msg_test");
    vi.stubGlobal("fetch", provider);

    const [first, second] = await Promise.all([
      worker.fetch(deliveryRequest({ token: "device-one", key: TEST_KEY_A }), env),
      worker.fetch(deliveryRequest({ token: "device-two", key: TEST_KEY_A }), env),
    ]);

    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(2);
    expect(new Set(env._namespace!.names.filter((name) =>
      name.startsWith("device-v2:")
    )).size).toBe(2);
    expect(env._namespace!.names.some((name) =>
      name.startsWith("recipient-v2:")
    )).toBe(false);
    const keys = provider.mock.calls.map(
      ([, init]) => (init?.headers as Record<string, string>)["Idempotency-Key"],
    );
    expect(new Set(keys).size).toBe(2);
  });

  it("isolates different recipients that use the same client key", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await completeCutover(env);
    await putBoundDeviceFixture(kv, env, "device-one", "one@example.com");
    await putBoundDeviceFixture(kv, env, "device-two", "two@example.com");
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
      name.startsWith("recipient-v2:")
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

describe("recipient-scoped automatic and device-scoped test ledgers", () => {
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

    const stored = JSON.stringify(Array.from(recipientDurables(env)[0]!.records));
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
    expect(JSON.stringify(Array.from(recipientDurables(env)[0]!.records))).not.toContain(
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

      const durable = recipientDurables(env)[0]!;
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

    const durable = recipientDurables(env)[0]!;
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

    const automaticLedgers = Array.from(recipientDurables(env)[0]!.records.values())
      .filter((value) => value && typeof value === "object" &&
        (value as { schemaVersion?: unknown }).schemaVersion === 2 &&
        "keyKind" in (value as object)) as Array<{ keyKind: string }>;
    const testLedgers = Array.from(deviceDurables(env)[0]!.records.values())
      .filter((value) => value && typeof value === "object" &&
        (value as { schemaVersion?: unknown }).schemaVersion === 2 &&
        "keyKind" in (value as object)) as Array<{ keyKind: string }>;
    expect(automaticLedgers.filter((record) => record.keyKind === "auto")).toHaveLength(1);
    expect(testLedgers.filter((record) => record.keyKind === "test")).toHaveLength(1);
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
    const stored = JSON.stringify(Array.from(recipientDurables(env)[0]!.records));
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
    const stored = JSON.stringify(Array.from(recipientDurables(env)[0]!.records));
    expect(stored).not.toContain("recipient@example.com");
    expect(stored).not.toContain("private invalid success body");
  });
});

describe("safe relay errors", () => {
  it("does not expose the verified raw recipient in the completion response", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    await putPending(env, "verify-token", "private-recipient@example.com");
    await completeCutover(env);

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
    await putDevice(env, "new-v2-token", "recipient@example.com", {
      protocolGeneration: PROTOCOL_GENERATION,
      buildIdentity: BUILD_IDENTITY,
      readyGeneration: READY_GENERATION,
    });
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
