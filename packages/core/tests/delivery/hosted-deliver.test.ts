import { describe, expect, it, vi } from "vitest";
import {
  deliverDailyEmailIfEnabled,
  sampleDailyDigest,
} from "../../src/delivery/deliver-email";
import {
  deliveryStatePath,
  loadDeliveryState,
  shouldSendEmail,
} from "../../src/delivery/delivery-state";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type { StorageAdapter } from "../../src/core/adapters";
import { startHostedEmailVerification } from "../../src/delivery/hosted";

async function persistedDeliveryText(
  storage: StorageAdapter,
): Promise<string> {
  const statePath = deliveryStatePath(DEFAULT_SETTINGS.output);
  const chunks: string[] = [];
  if (await storage.exists(statePath)) chunks.push(await storage.readText(statePath));
  for (const entry of (await storage.list?.(`${statePath}.claims`)) ?? []) {
    if (entry.type === "file") chunks.push(await storage.readText(entry.path));
  }
  return chunks.join("\n");
}

function memoryStorage(): StorageAdapter {
  const files = new Map<string, string>();
  return {
    normalizePath: (p) => p.replace(/\\/g, "/"),
    async createTextExclusive(path, content) {
      if (files.has(path)) return false;
      files.set(path, content);
      return true;
    },
    async guardClaimNamespace() {
      return { assertCurrent() {}, async release() {} };
    },
    async readText(path) {
      if (!files.has(path)) throw new Error(`missing ${path}`);
      return files.get(path)!;
    },
    async writeText(path, content) {
      files.set(path, content);
    },
    async writeTextAtomic(path, content, mode) {
      if (mode !== 0o600) throw new Error("private mode is required");
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
    async list(dir) {
      const prefix = dir ? `${dir}/` : "";
      return Array.from(files.keys())
        .filter((path) => path.startsWith(prefix))
        .map((path) => ({ path, type: "file" as const }));
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
      expect(idemp).toMatch(
        request.mock.calls.length === 1
          ? /^arxiv-daily:test:/
          : /^arxiv-daily:auto:/,
      );
      expect(idemp.length).toBeLessThanOrEqual(128);
      expect(idemp).not.toContain("you@example.com");
      return {
        status: 200,
        headers: {},
        bodyText: JSON.stringify({ ok: true }),
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

  it("keeps a hosted claim blocking when relay reports an ambiguous attempt", async () => {
    const storage = memoryStorage();
    const request = vi.fn(async () => ({
      status: 502,
      headers: {},
      bodyText: JSON.stringify({
        error: "Resend transport outcome is unknown",
        ambiguous: true,
      }),
    }));
    const email = {
      enabled: true,
      mode: "hosted" as const,
      to: "you@example.com",
      fromEmail: "",
      hostedToken: "device_token_hex",
      apiKey: "",
    };

    const first = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });
    const second = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });

    expect(first).toMatchObject({ kind: "ambiguous", attempts: 1 });
    expect(second.kind).toBe("skipped");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("does not expose or persist a sensitive hosted error body", async () => {
    const storage = memoryStorage();
    const sensitive = [
      "recipient@example.com",
      "hosted_token_super_secret",
      "private email body",
    ];
    const request = vi.fn(async () => ({
      status: 422,
      headers: {},
      bodyText: sensitive.join(" | "),
    }));
    const logger = {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    };

    const result = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email: {
        enabled: true,
        mode: "hosted",
        to: "you@example.com",
        hostedToken: "device_token_hex",
        fromEmail: "",
      },
      logger,
    });

    expect(result).toMatchObject({ kind: "failed", attempts: 1 });
    const exposed = [
      result.kind === "failed" ? result.reason : "",
      JSON.stringify(logger.warn.mock.calls),
      JSON.stringify(logger.error.mock.calls),
      await persistedDeliveryText(storage),
    ].join("\n");
    for (const value of sensitive) expect(exposed).not.toContain(value);
  });

  it.each([
    ["not-json", "invalid JSON"],
    [JSON.stringify({ id: "" }), "missing success marker"],
    [JSON.stringify({ ok: false }), "false success marker"],
    [JSON.stringify({ ok: true, id: "private-provider-id" }), "unsafe extra field"],
  ])("keeps an invalid hosted 2xx blocking (%s)", async (bodyText) => {
    const storage = memoryStorage();
    const request = vi.fn(async () => ({ status: 200, headers: {}, bodyText }));
    const email = {
      enabled: true,
      mode: "hosted" as const,
      to: "you@example.com",
      hostedToken: "device_token_hex",
      fromEmail: "",
    };

    const first = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });
    const second = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });

    expect(first).toMatchObject({ kind: "ambiguous", attempts: 1 });
    expect(second.kind).toBe("skipped");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("replays a relay definitive 422 decision without another provider call", async () => {
    const storage = memoryStorage();
    const request = vi
      .fn()
      .mockResolvedValueOnce({
        status: 422,
        headers: {},
        bodyText: JSON.stringify({ error: "provider rejected request", ambiguous: false }),
      })
      .mockResolvedValueOnce({
        status: 200,
        headers: {},
        bodyText: JSON.stringify({ ok: true }),
      });
    const email = {
      enabled: true,
      mode: "hosted" as const,
      to: "you@example.com",
      hostedToken: "device_token_hex",
      fromEmail: "",
    };

    const rejected = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });
    const retry = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });

    expect(rejected).toMatchObject({ kind: "failed", attempts: 1 });
    expect(retry).toMatchObject({
      kind: "skipped",
      reason: "provider_definitive_rejection",
    });
    expect(request).toHaveBeenCalledTimes(1);
  });

  it.each([408, 418, 500])(
    "does not trust ambiguous=false outside the explicit definitive contract on HTTP %s",
    async (status) => {
      const storage = memoryStorage();
      const request = vi.fn(async () => ({
        status,
        headers: {},
        bodyText: JSON.stringify({ error: "claimed definitive", ambiguous: false }),
      }));
      const email = {
        enabled: true,
        mode: "hosted" as const,
        to: "you@example.com",
        hostedToken: "device_token_hex",
        fromEmail: "",
      };

      const first = await deliverDailyEmailIfEnabled(digest, {
        storage,
        http: { request },
        output,
        email,
      });
      const second = await deliverDailyEmailIfEnabled(digest, {
        storage,
        http: { request },
        output,
        email,
      });

      expect(first).toMatchObject({ kind: "ambiguous", attempts: 1 });
      expect(second.kind).toBe("skipped");
      expect(request).toHaveBeenCalledTimes(1);
    },
  );

  it("does not expose a sensitive verification response body", async () => {
    const sensitiveBody =
      "recipient@example.com hosted_token_super_secret private email body";

    const error = await startHostedEmailVerification({
      http: {
        request: vi.fn(async () => ({
          status: 400,
          headers: {},
          bodyText: sensitiveBody,
        })),
      },
      baseUrl: "https://configurable-relay.example",
      email: "recipient@example.com",
    }).catch((value) => value);

    expect(error).toMatchObject({
      name: "HostedDeliveryError",
      status: 400,
    });
    expect(error.message).not.toContain("recipient@example.com");
    expect(error.message).not.toContain("hosted_token_super_secret");
    expect(error.message).not.toContain("private email body");
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
      expect(result.reason).toBe("verification_token_missing");
    }
  });
});
