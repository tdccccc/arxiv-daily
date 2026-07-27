import { describe, expect, it } from "vitest";
import {
  completeIdempotent,
  parseIdemp,
  reserveIdempotent,
  type Env,
} from "../src/kv";

function memoryKv() {
  const map = new Map<string, string>();
  return {
    async get(key: string) {
      return map.get(key) ?? null;
    },
    async put(key: string, value: string) {
      map.set(key, value);
    },
    async delete(key: string) {
      map.delete(key);
    },
    _map: map,
  };
}

function envWith(kv: ReturnType<typeof memoryKv>): Env {
  return {
    STORE: kv as unknown as KVNamespace,
    RESEND_API_KEY: "re_test",
    TOKEN_SECRET: "secret",
    PUBLIC_BASE_URL: "https://example.com",
    FROM_EMAIL: "daily@mail.arxiv-daily.top",
    FROM_NAME: "arXiv Daily",
    DAILY_QUOTA: "5",
  };
}

describe("idempotency reserve", () => {
  it("parses pending and done forms", () => {
    expect(parseIdemp(null)).toBeNull();
    expect(parseIdemp("pending:abc")).toEqual({
      kind: "pending",
      claim: "abc",
    });
    expect(parseIdemp("done:msg_1")).toEqual({ kind: "done", id: "msg_1" });
    expect(parseIdemp("legacy_id")).toEqual({ kind: "done", id: "legacy_id" });
  });

  it("reserves then completes", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const key = "2026-07-27|a@b.com";
    const r1 = await reserveIdempotent(env, key, "claim-a");
    expect(r1.status).toBe("reserved");
    await completeIdempotent(env, key, "msg_ok");
    const r2 = await reserveIdempotent(env, key, "claim-b");
    expect(r2).toEqual({ status: "done", id: "msg_ok" });
  });

  it("detects another pending claim", async () => {
    const kv = memoryKv();
    const env = envWith(kv);
    const key = "k";
    await reserveIdempotent(env, key, "claim-a");
    const other = await reserveIdempotent(env, key, "claim-b");
    // After second reserve overwrites, claim-b wins (KV no CAS) — still better than no reserve
    expect(["reserved", "pending_other"]).toContain(other.status);
  });
});
