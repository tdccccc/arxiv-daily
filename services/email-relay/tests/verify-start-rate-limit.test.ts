import { afterEach, describe, expect, it, vi } from "vitest";
import worker from "../src/index";
import type { Env } from "../src/kv";

function env() {
  const map = new Map<string, string>();
  const STORE = {
    async get(key: string) { return map.get(key) ?? null; },
    async put(key: string, value: string) { map.set(key, value); },
    async delete(key: string) { map.delete(key); },
  };
  return {
    map,
    env: {
      STORE,
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
    const { env: testEnv, map } = env();
    const provider = stubProvider();
    for (let index = 0; index < 10; index += 1) {
      await start(testEnv, `person${index}@example.com`, "203.0.113.9");
    }
    expect(provider).toHaveBeenCalledTimes(10);

    // Over the IP limit: no mail, and the targeted address's counter must stay untouched.
    const blocked = await start(testEnv, "victim@example.com", "203.0.113.9");
    expect(blocked.status).toBe(200);
    expect(provider).toHaveBeenCalledTimes(10);
    expect(map.has("rl:verify-email:victim@example.com")).toBe(false);
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
});
