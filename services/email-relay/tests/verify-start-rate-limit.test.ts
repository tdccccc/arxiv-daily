import { afterEach, describe, expect, it, vi } from "vitest";
import worker from "../src/index";
import type { Env } from "../src/kv";
import { VerifyRateLimiter } from "../src/verify-rate-limiter";

const HOUR_MS = 3600 * 1000;
const START = Date.parse("2026-10-18T00:00:00.000Z");

interface Row { started_at: number; window_ms: number; count: number }

/** In-memory stand-in for a SQLite-backed DO's storage; only the statements the limiter issues. */
function fakeDurableState() {
  let row: Row | null = null;
  let alarm: number | null = null;
  let deleteAllCalls = 0;
  const sql = {
    exec(query: string, ...bindings: unknown[]) {
      const text = query.replace(/\s+/g, " ").trim();
      if (text.startsWith("CREATE TABLE IF NOT EXISTS verify_window")) {
        return { toArray: () => [] };
      }
      if (text.startsWith("SELECT started_at, window_ms, count FROM verify_window")) {
        return { toArray: () => (row ? [{ ...row }] : []) };
      }
      if (text.startsWith("INSERT OR REPLACE INTO verify_window")) {
        const [started_at, window_ms, count] = bindings as number[];
        row = { started_at, window_ms, count };
        return { toArray: () => [] };
      }
      throw new Error(`unexpected SQL: ${text}`);
    },
  };
  const state = {
    storage: {
      sql,
      transactionSync<T>(closure: () => T): T { return closure(); },
      async setAlarm(value: number) { alarm = value; },
      async deleteAll() { deleteAllCalls += 1; row = null; alarm = null; },
    },
  } as unknown as DurableObjectState;
  return {
    state,
    get row() { return row; },
    get alarm() { return alarm; },
    get deleteAllCalls() { return deleteAllCalls; },
  };
}

class FakeLimiterNamespace {
  now = START;
  readonly names: string[] = [];
  readonly objects = new Map<string, {
    durable: ReturnType<typeof fakeDurableState>;
    limiter: VerifyRateLimiter;
  }>();

  idFromName(name: string): DurableObjectId {
    this.names.push(name);
    return { toString: () => name } as DurableObjectId;
  }

  get(id: DurableObjectId): DurableObjectStub {
    const name = id.toString();
    let object = this.objects.get(name);
    if (!object) {
      const durable = fakeDurableState();
      object = {
        durable,
        limiter: new VerifyRateLimiter(durable.state, undefined, () => this.now),
      };
      this.objects.set(name, object);
    }
    const { limiter } = object;
    return {
      // Yield once like a network hop so concurrent callers really interleave.
      fetch: async (input: RequestInfo | URL, init?: RequestInit) => {
        await Promise.resolve();
        return limiter.fetch(new Request(input as string, init));
      },
    } as unknown as DurableObjectStub;
  }

  namesWith(prefix: string): string[] {
    return [...this.objects.keys()].filter((name) => name.startsWith(prefix));
  }
}

function env(options: { binding?: boolean } = {}) {
  const map = new Map<string, string>();
  const STORE = {
    async get(key: string) { return map.get(key) ?? null; },
    async put(key: string, value: string) { map.set(key, value); },
    async delete(key: string) { map.delete(key); },
  };
  const limiter = new FakeLimiterNamespace();
  return {
    map,
    limiter,
    env: {
      STORE,
      ...(options.binding === false
        ? {}
        : { VERIFY_RATE_LIMIT: limiter as unknown as DurableObjectNamespace }),
      RESEND_API_KEY: "re_test",
      TOKEN_SECRET: "token-secret-for-tests",
      FROM_EMAIL: "daily@mail.example.test",
      FROM_NAME: "arXiv Daily",
      PUBLIC_BASE_URL: "https://mail.example.test",
      DAILY_QUOTA: "5",
    } as unknown as Env,
  };
}

function start(testEnv: Env, email: string, ip: string) {
  return worker.fetch(new Request("https://mail.example.test/v1/verify/start", {
    method: "POST",
    headers: { "Content-Type": "application/json", "CF-Connecting-IP": ip },
    body: JSON.stringify({ email }),
  }), testEnv);
}

function stubProvider() {
  const provider = vi.fn(async () => new Response(JSON.stringify({ id: "msg_1" }), { status: 200 }));
  vi.stubGlobal("fetch", provider);
  return provider;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("verify/start rate limits", () => {
  it("sends at most 3 verification emails per address per window, answering identically", async () => {
    const { env: testEnv } = env();
    const provider = stubProvider();
    for (let index = 0; index < 5; index += 1) {
      const res = await start(testEnv, "Victim@Example.com", `198.51.100.${index}`);
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ok: true, message: "verification email sent" });
    }
    expect(provider).toHaveBeenCalledTimes(3);
  });

  it("stops after 10 sends per IP without spending the address allowance of rejected requests", async () => {
    const { env: testEnv, limiter } = env();
    const provider = stubProvider();
    for (let index = 0; index < 10; index += 1) {
      await start(testEnv, `person${index}@example.com`, "203.0.113.9");
    }
    expect(provider).toHaveBeenCalledTimes(10);
    const emailObjectsBefore = limiter.namesWith("verify-email:").length;

    // Over the IP limit: no mail, and the targeted address's counter must stay untouched.
    const blocked = await start(testEnv, "victim@example.com", "203.0.113.9");
    expect(blocked.status).toBe(200);
    expect(await blocked.json()).toEqual({ ok: true, message: "verification email sent" });
    expect(provider).toHaveBeenCalledTimes(10);
    expect(limiter.namesWith("verify-email:")).toHaveLength(emailObjectsBefore);

    // The address still has its full allowance from a fresh IP.
    for (let index = 0; index < 4; index += 1) {
      await start(testEnv, "victim@example.com", `198.51.100.${index}`);
    }
    expect(provider).toHaveBeenCalledTimes(13);
  });

  it("an attacker's IP-limited requests cannot lock the real owner out", async () => {
    const { env: testEnv } = env();
    const provider = stubProvider();
    for (let index = 0; index < 10; index += 1) {
      await start(testEnv, `filler${index}@example.com`, "203.0.113.9");
    }
    // The attacker keeps hammering the victim's address from the exhausted IP.
    for (let index = 0; index < 20; index += 1) {
      await start(testEnv, "victim@example.com", "203.0.113.9");
    }
    provider.mockClear();
    const owner = await start(testEnv, "victim@example.com", "198.51.100.77");
    expect(owner.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(1);
  });

  it("concurrent requests for one address send exactly 3 emails", async () => {
    const { env: testEnv } = env();
    const provider = stubProvider();
    const responses = await Promise.all(
      Array.from({ length: 25 }, (_, index) =>
        start(testEnv, "Victim@Example.com", `198.51.100.${index}`)),
    );
    for (const res of responses) {
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ok: true, message: "verification email sent" });
    }
    expect(provider).toHaveBeenCalledTimes(3);
  });

  it("concurrent requests from one IP send exactly 10 emails", async () => {
    const { env: testEnv } = env();
    const provider = stubProvider();
    await Promise.all(
      Array.from({ length: 30 }, (_, index) =>
        start(testEnv, `person${index}@example.com`, "203.0.113.9")),
    );
    expect(provider).toHaveBeenCalledTimes(10);
  });

  it("opens a fresh window one hour after the first hit, not before", async () => {
    const { env: testEnv, limiter } = env();
    const provider = stubProvider();
    for (let index = 0; index < 4; index += 1) {
      await start(testEnv, "victim@example.com", `198.51.100.${index}`);
    }
    expect(provider).toHaveBeenCalledTimes(3);

    limiter.now = START + HOUR_MS - 1;
    await start(testEnv, "victim@example.com", "198.51.100.50");
    expect(provider).toHaveBeenCalledTimes(3);

    limiter.now = START + HOUR_MS;
    for (let index = 0; index < 4; index += 1) {
      await start(testEnv, "victim@example.com", `198.51.100.${60 + index}`);
    }
    expect(provider).toHaveBeenCalledTimes(6);
  });

  it("keys counters by keyed hash and never stores the raw address or IP", async () => {
    const { env: testEnv, limiter, map } = env();
    stubProvider();
    await start(testEnv, "Victim@Example.com", "203.0.113.9");

    expect(limiter.names).toHaveLength(2);
    const [ipName, emailName] = limiter.names;
    expect(ipName).toMatch(/^verify-ip:[0-9a-f]{64}$/);
    expect(emailName).toMatch(/^verify-email:[0-9a-f]{64}$/);
    for (const name of limiter.names) {
      expect(name).not.toContain("203.0.113.9");
      expect(name.toLowerCase()).not.toContain("victim");
    }
    // The same address in another casing maps to the same counter.
    await start(testEnv, "  VICTIM@example.com ", "203.0.113.10");
    expect(limiter.namesWith("verify-email:")).toHaveLength(1);
    // The old KV counters are gone.
    expect([...map.keys()].some((key) => key.startsWith("rl:"))).toBe(false);
  });

  it("keeps a single row per counter and deletes it when the window alarm fires", async () => {
    const { env: testEnv, limiter } = env();
    stubProvider();
    for (let index = 0; index < 2; index += 1) {
      await start(testEnv, "victim@example.com", "203.0.113.9");
    }
    const emailName = limiter.namesWith("verify-email:")[0]!;
    const { durable, limiter: object } = limiter.objects.get(emailName)!;
    expect(durable.row).toMatchObject({ started_at: START, count: 2 });
    expect(durable.alarm).toBe(START + HOUR_MS);

    // An early alarm re-arms instead of deleting a live window.
    limiter.now = START + HOUR_MS - 1000;
    await object.alarm();
    expect(durable.deleteAllCalls).toBe(0);
    expect(durable.alarm).toBe(START + HOUR_MS);

    limiter.now = START + HOUR_MS;
    await object.alarm();
    expect(durable.deleteAllCalls).toBe(1);
    expect(durable.row).toBeNull();
  });

  it("fails closed without sending mail when the rate-limit binding is missing", async () => {
    const { env: testEnv } = env({ binding: false });
    const provider = stubProvider();
    const res = await start(testEnv, "victim@example.com", "203.0.113.9");
    expect(res.status).toBe(500);
    expect(provider).not.toHaveBeenCalled();
  });
});

describe("VerifyRateLimiter object", () => {
  it("rejects malformed hit requests and non-POST methods", async () => {
    const durable = fakeDurableState();
    const limiter = new VerifyRateLimiter(durable.state, undefined, () => START);
    const post = (body: unknown) =>
      limiter.fetch(new Request("https://verify-rate-limit/hit", {
        method: "POST",
        body: JSON.stringify(body),
      }));
    expect((await post({ limit: 0, windowSeconds: 3600 })).status).toBe(400);
    expect((await post({ limit: 3, windowSeconds: 0 })).status).toBe(400);
    expect((await post({ limit: "3", windowSeconds: 3600 })).status).toBe(400);
    expect((await limiter.fetch(new Request("https://verify-rate-limit/hit"))).status).toBe(405);
    expect(durable.row).toBeNull();
  });

  it("reports the remaining seconds of the window when rejecting", async () => {
    const durable = fakeDurableState();
    let now = START;
    const limiter = new VerifyRateLimiter(durable.state, undefined, () => now);
    const hit = async () =>
      (await limiter.fetch(new Request("https://verify-rate-limit/hit", {
        method: "POST",
        body: JSON.stringify({ limit: 1, windowSeconds: 3600 }),
      }))).json();
    expect(await hit()).toEqual({ ok: true, count: 1, retryAfterSeconds: 3600 });
    now = START + 600 * 1000;
    expect(await hit()).toEqual({ ok: false, count: 1, retryAfterSeconds: 3000 });
  });
});
