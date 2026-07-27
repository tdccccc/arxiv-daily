import { describe, expect, it, vi } from "vitest";
import {
  deliverDailyEmailIfEnabled,
  sampleDailyDigest,
} from "../../src/delivery/deliver-email";
import {
  loadDeliveryState,
  shouldSendEmail,
} from "../../src/delivery/delivery-state";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type { StorageAdapter } from "../../src/core/adapters";

function memoryStorage(): StorageAdapter {
  const files = new Map<string, string>();
  return {
    normalizePath: (p) => p.replace(/\\/g, "/"),
    async readText(path) {
      if (!files.has(path)) throw new Error(`missing ${path}`);
      return files.get(path)!;
    },
    async writeText(path, content) {
      files.set(path, content);
    },
    async exists(path) {
      return files.has(path);
    },
    async mkdir() {},
    async rename(from, to) {
      const v = files.get(from);
      if (v === undefined) throw new Error(`missing ${from}`);
      files.set(to, v);
      files.delete(from);
    },
    async remove(path) {
      files.delete(path);
    },
  };
}

describe("hosted deliverDailyEmailIfEnabled", () => {
  const output = DEFAULT_SETTINGS.output;
  const digest = sampleDailyDigest({ date: "2026-07-27", language: "zh" });

  it("force test-send uses test idempotency key and does not block auto-send", async () => {
    const storage = memoryStorage();
    const request = vi.fn(async (req: { url: string; headers?: Record<string, string> }) => {
      expect(req.url).toContain("/v1/deliver");
      const idemp = req.headers?.["Idempotency-Key"] ?? "";
      if (request.mock.calls.length === 1) {
        expect(idemp.startsWith("test|")).toBe(true);
      } else {
        expect(idemp).toBe("2026-07-27|you@example.com");
      }
      return {
        status: 200,
        headers: {},
        bodyText: JSON.stringify({ ok: true, id: `msg_${request.mock.calls.length}` }),
      };
    });

    const email = {
      enabled: true,
      mode: "hosted" as const,
      to: "you@example.com",
      fromEmail: "",
      hostedToken: "device_token_hex",
      apiKey: "",
    };

    const testSend = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      force: true,
    });
    expect(testSend.kind).toBe("delivered");
    expect(request).toHaveBeenCalledTimes(1);

    const state = await loadDeliveryState(storage, output);
    expect(shouldSendEmail(state, digest.date, email.to)).toBe(true);

    const auto = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });
    expect(auto.kind).toBe("delivered");
    expect(request).toHaveBeenCalledTimes(2);

    const state2 = await loadDeliveryState(storage, output);
    expect(shouldSendEmail(state2, digest.date, email.to)).toBe(false);
  });

  it("rejects hosted mode without token", async () => {
    const result = await deliverDailyEmailIfEnabled(digest, {
      storage: memoryStorage(),
      http: { request: vi.fn() },
      output,
      email: {
        enabled: true,
        mode: "hosted",
        to: "you@example.com",
        fromEmail: "",
        hostedToken: "",
      },
      force: true,
    });
    expect(result.kind).toBe("disabled");
    if (result.kind === "disabled") {
      expect(result.reason).toMatch(/verification code/i);
    }
  });
});
