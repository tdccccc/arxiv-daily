import { afterEach, describe, expect, it, vi } from "vitest";
import worker from "../src/index";
import { DeliverGate, stageDeliveryV2Cutover } from "../src/deliver-gate";
import { sha256Hex } from "../src/crypto";
import {
  hashDeviceToken,
  hashLegacyAutoDeliveryIdentity,
  putDevice,
  putPending,
  type Env,
} from "../src/kv";

const AUTO_KEY_A = `arxiv-daily:auto:${"a".repeat(64)}`;
const AUTO_KEY_B = `arxiv-daily:auto:${"b".repeat(64)}`;
const TEST_KEY_A = `arxiv-daily:test:${"c".repeat(32)}`;
const TEST_KEY_B = `arxiv-daily:test:${"d".repeat(32)}`;
const DELIVERY_V2_CUTOVER_KEY = "cutover:delivery-v2";

type MemoryDurable = ReturnType<typeof durableState>;

function memoryKv() {
  const map = new Map<string, string>();
  const failGet = new Set<string>();
  const hideFromList = new Set<string>();
  let failList = false;
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
    async list(options: { prefix?: string } = {}) {
      if (failList) throw new Error("injected KV list failure");
      const prefix = options.prefix ?? "";
      return {
        keys: Array.from(map.keys())
          .filter((key) => key.startsWith(prefix) && !hideFromList.has(key))
          .sort()
          .map((name) => ({ name })),
        list_complete: true,
        cacheStatus: null,
      };
    },
    _map: map,
    _failGet: failGet,
    _hideFromList: hideFromList,
    set _failList(value: boolean) {
      failList = value;
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
  const identity = await hashLegacyAutoDeliveryIdentity(
    "secret",
    date,
    recipient,
  );
  if (legacyValue === undefined) {
    kv._map.delete(`idemp:${logicalKey}`);
  } else {
    kv._map.set(`idemp:${logicalKey}`, legacyValue);
  }
  const evidence = legacyValue === undefined
    ? undefined
    : legacyValue.startsWith("pending:") ? "attempted" : "done";
  kv._map.set(
    DELIVERY_V2_CUTOVER_KEY,
    JSON.stringify(cutoverMarker(evidence ? { [identity]: evidence } : {})),
  );
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

async function completeCutover(
  env: Env & { _namespace?: MemoryNamespace },
): Promise<void> {
  await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
  env._namespace!.now = new Date(
    env._namespace!.now.getTime() + 60 * 1000,
  );
  await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
  env._namespace!.now = new Date(
    env._namespace!.now.getTime() + 120 * 1000,
  );
  await stageDeliveryV2Cutover(env);
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

describe("legacy KV delivery cutover", () => {
  it("does not enable automatic traffic after only the first server observation", async () => {
    const { env } = await authenticatedEnv({ stageCutover: false });
    await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "delivery cutover is not ready",
      ambiguous: false,
    });
    expect(provider).not.toHaveBeenCalled();
  });

  it("fails closed when the server-side legacy scan is unavailable", async () => {
    const { kv, env } = await authenticatedEnv({ stageCutover: false });
    kv._failList = true;
    await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(provider).not.toHaveBeenCalled();
  });

  it("fails closed when legacy automatic evidence appears after the first observation", async () => {
    const { kv, env } = await authenticatedEnv({ stageCutover: false });
    await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
    const date = "2026-08-10";
    const logicalKey = `arxiv-daily:auto:${await sha256Hex(
      `${date}\u0000recipient@example.com`,
    )}`;
    kv._map.set(`idemp:${logicalKey}`, "pending:late-v1-claim");
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 60 * 1000);
    await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(provider).not.toHaveBeenCalled();
  });

  it("blocks a pre-cutover identity when pending expires unseen during visibility wait", async () => {
    const { kv, env } = await authenticatedEnv({ stageCutover: false });
    await putDevice(
      env,
      "device-token",
      "recipient@example.com",
      new Date(env._namespace!.now.getTime() - 1),
    );
    const logicalKey = `arxiv-daily:auto:${await sha256Hex(
      "2026-08-10\u0000recipient@example.com",
    )}`;
    const legacyKey = `idemp:${logicalKey}`;
    kv._map.set(legacyKey, "pending:invisible-v1-claim");
    kv._hideFromList.add(legacyKey);

    await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 60 * 1000);
    kv._map.delete(legacyKey);
    await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
    env._namespace!.now = new Date(env._namespace!.now.getTime() + 120 * 1000);
    await stageDeliveryV2Cutover(env);
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "delivery cutover is not ready",
      ambiguous: false,
    });
    expect(provider).not.toHaveBeenCalled();
  });

  it("enforces the readiness propagation wait before accepting automatic traffic", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-10T00:00:30.000Z"));
    const { kv, env } = await authenticatedEnv({ stageCutover: false });
    kv._map.set(DELIVERY_V2_CUTOVER_KEY, JSON.stringify({
      ...cutoverMarker(),
      preQuiesceScanStartedAt: "2026-08-09T23:57:30.000Z",
      preQuiesceScanCompletedAt: "2026-08-09T23:57:31.000Z",
      oldWorkerWritesQuiescedAt: "2026-08-09T23:58:00.000Z",
      postQuiesceScanStartedAt: "2026-08-09T23:59:00.000Z",
      postQuiesceScanCompletedAt: "2026-08-09T23:59:01.000Z",
      enabledAt: "2026-08-10T00:00:00.000Z",
    }));
    await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "delivery cutover is not ready",
      ambiguous: false,
    });
    expect(provider).not.toHaveBeenCalled();
  });

  it("rejects an enabledAt declared before the 120-second quiesce interval elapsed", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-10T00:00:00.000Z"));
    const { kv, env } = await authenticatedEnv({ stageCutover: false });
    kv._map.set(DELIVERY_V2_CUTOVER_KEY, JSON.stringify({
      ...cutoverMarker(),
      enabledAt: "2026-08-01T00:01:03.000Z",
    }));
    await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "delivery cutover is not ready",
      ambiguous: false,
    });
    expect(provider).not.toHaveBeenCalled();
  });

  it("fails closed when the legacy scan cannot cover a 120-second pending lifetime", async () => {
    const { kv, env } = await authenticatedEnv({ stageCutover: false });
    kv._map.set(DELIVERY_V2_CUTOVER_KEY, JSON.stringify({
      ...cutoverMarker(),
      postQuiesceScanStartedAt: "2026-08-01T00:02:01.000Z",
      postQuiesceScanCompletedAt: "2026-08-01T00:02:02.000Z",
      enabledAt: "2026-08-01T00:02:02.000Z",
    }));
    await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "delivery cutover is not ready",
      ambiguous: false,
    });
    expect(provider).not.toHaveBeenCalled();
  });

  it("fails closed for a pre-cutover device when no legacy evidence is provable", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-31T00:00:00.000Z"));
    const { env } = await authenticatedEnv({ stageCutover: false });
    vi.setSystemTime(new Date("2026-08-10T00:00:00.000Z"));
    await completeCutover(env);
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "delivery cutover is not ready",
      ambiguous: false,
    });
    expect(provider).not.toHaveBeenCalled();
  });

  it("keeps a legacy device without v2 provenance blocked even with a future timestamp", async () => {
    const { kv, env } = await authenticatedEnv({ stageCutover: false });
    await completeCutover(env);
    const deviceHash = await hashDeviceToken("device-token", env.TOKEN_SECRET);
    kv._map.delete(`device-v2:${deviceHash}`);
    kv._map.set(`device:${deviceHash}`, JSON.stringify({
      email: "recipient@example.com",
      createdAt: new Date(env._namespace!.now.getTime() + 60 * 1000).toISOString(),
    }));
    const provider = providerSuccess("must_not_send");
    vi.stubGlobal("fetch", provider);

    const response = await worker.fetch(deliveryRequest(), env);

    expect(response.status).toBe(503);
    expect(provider).not.toHaveBeenCalled();
  });

  it.each([
    ["msg_plain", "done"],
    ["done:msg_done", "done"],
    ["pending:legacy-claim", "attempted"],
  ] as const)(
    "imports legacy %s evidence into a permanent v2 block",
    async (legacyValue, expectedStatus) => {
      const { env } = await authenticatedEnv({
        quota: 1,
        legacyValue,
      });
      const provider = providerSuccess("must_not_send");
      vi.stubGlobal("fetch", provider);

      const blocked = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
      const another = await worker.fetch(
        deliveryRequest({
          key: AUTO_KEY_B,
          date: "2026-08-11",
        }),
        env,
      );

      expect(blocked.status).toBe(409);
      expect(await blocked.json()).toMatchObject({
        ambiguous: expectedStatus === "attempted",
      });
      expect(another.status).toBe(429);
      expect(provider).not.toHaveBeenCalled();
      const durable = deviceDurables(env)[0]!;
      const imported = Array.from(durable.records.values()).find(
        (value) => value && typeof value === "object" &&
          (value as { legacyImported?: unknown }).legacyImported === true,
      ) as { status?: unknown } | undefined;
      expect(imported?.status).toBe(expectedStatus);
      expect(JSON.stringify(imported)).not.toContain("recipient@example.com");
      expect(JSON.stringify(imported)).not.toContain(legacyValue);
    },
  );

  it.each([
    ["done:msg_done", "legacy_delivery_done", false],
    ["pending:legacy-claim", "legacy_delivery_attempted", true],
  ] as const)(
    "imports durable %s proof after old KV disappears before first request",
    async (legacyValue, error, ambiguous) => {
      const { kv, env } = await authenticatedEnv({ quota: 5, legacyValue });
      kv._map.clear();
      await putDevice(
        env,
        "device-token",
        "recipient@example.com",
        new Date(env._namespace!.now.getTime() + 1),
      );
      kv._failGet.add(DELIVERY_V2_CUTOVER_KEY);
      const provider = providerSuccess("must_not_send");
      vi.stubGlobal("fetch", provider);

      const response = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);

      expect(response.status).toBe(409);
      expect(await response.json()).toEqual({ error, ambiguous });
      expect(provider).not.toHaveBeenCalled();
    },
  );

  it.each([
    ["done:msg_done", "legacy_delivery_done", false],
    ["pending:legacy-claim", "legacy_delivery_attempted", true],
  ] as const)(
    "keeps imported %s blocked after old KV and cutover marker disappear",
    async (legacyValue, error, ambiguous) => {
      const { kv, env } = await authenticatedEnv({ quota: 5, legacyValue });
      const provider = providerSuccess("must_not_send");
      vi.stubGlobal("fetch", provider);

      const imported = await worker.fetch(deliveryRequest({ key: AUTO_KEY_A }), env);
      kv._map.clear();
      const deviceHash = await hashDeviceToken("device-token", env.TOKEN_SECRET);
      await putDevice(
        env,
        "device-token",
        "recipient@example.com",
        new Date(env._namespace!.now.getTime() + 1),
      );
      expect(kv._map.has(`device-v2:${deviceHash}`)).toBe(true);
      kv._failGet.add(DELIVERY_V2_CUTOVER_KEY);
      const replay = await worker.fetch(deliveryRequest({ key: AUTO_KEY_B }), env);

      expect(imported.status).toBe(409);
      expect(replay.status).toBe(409);
      expect(await replay.json()).toEqual({ error, ambiguous });
      expect(provider).not.toHaveBeenCalled();
    },
  );

  it.each(["missing", "malformed", "read-failure"] as const)(
    "fails closed when readiness is %s",
    async (scenario) => {
      const { kv, env } = await authenticatedEnv({ stageCutover: false });
      if (scenario === "missing") kv._map.delete(DELIVERY_V2_CUTOVER_KEY);
      if (scenario === "malformed") {
        kv._map.set(DELIVERY_V2_CUTOVER_KEY, "{not-ready");
      }
      if (scenario === "read-failure") kv._failGet.add(DELIVERY_V2_CUTOVER_KEY);
      await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
      const provider = providerSuccess("must_not_send");
      vi.stubGlobal("fetch", provider);

      const response = await worker.fetch(deliveryRequest(), env);

      expect(response.status).toBe(503);
      expect(await response.json()).toEqual({
        error: "delivery cutover is not ready",
        ambiguous: false,
      });
      expect(provider).not.toHaveBeenCalled();
    },
  );

  it.each([
    { "not-a-hash": "done" },
    { ["a".repeat(64)]: "unknown" },
  ] as const)(
    "fails closed when the atomic legacy evidence snapshot is malformed",
    async (legacyEvidence) => {
      const { kv, env } = await authenticatedEnv({ stageCutover: false });
      kv._map.set(DELIVERY_V2_CUTOVER_KEY, JSON.stringify({
        ...cutoverMarker(),
        legacyAutoEvidence: legacyEvidence,
      }));
      await expect(stageDeliveryV2Cutover(env)).rejects.toThrow();
      const provider = providerSuccess("must_not_send");
      vi.stubGlobal("fetch", provider);

      const response = await worker.fetch(deliveryRequest(), env);

      expect(response.status).toBe(503);
      expect(await response.json()).toEqual({
        error: "delivery cutover is not ready",
        ambiguous: false,
      });
      expect(provider).not.toHaveBeenCalled();
    },
  );
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
