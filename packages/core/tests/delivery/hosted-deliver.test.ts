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
import { HttpTransportError } from "../../src/core/adapters";

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
  // Matches the existing 128-character delivery/provider boundary while keeping
  // the transitional provider identifier strictly bounded.
  const legacyProviderIdMaxLength = 128;
  const escapedLegacyProviderId = 'private-"provider\\id,{braces}';

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
    ["canonical", JSON.stringify({ ok: true }), undefined],
    [
      "legacy provider ID",
      JSON.stringify({ id: "private-provider-id", ok: true }),
      "private-provider-id",
    ],
    [
      "legacy deduplicated provider ID at the bound",
      JSON.stringify({
        deduped: true,
        id: "d".repeat(legacyProviderIdMaxLength),
        ok: true,
      }),
      "d".repeat(legacyProviderIdMaxLength),
    ],
    [
      "legacy provider ID with escaped JSON characters",
      JSON.stringify({ id: escapedLegacyProviderId, ok: true }),
      escapedLegacyProviderId,
    ],
  ])(
    "accepts the exact %s success shape and discards its provider ID",
    async (_case, bodyText, providerId) => {
      const storage = memoryStorage();
      const request = vi.fn(async () => ({ status: 200, headers: {}, bodyText }));
      const logger = {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
      };
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
        logger,
      });
      const second = await deliverDailyEmailIfEnabled(digest, {
        storage,
        http: { request },
        output,
        email,
        logger,
      });

      expect(first).toEqual({ kind: "delivered", attempts: 1 });
      expect(second.kind).toBe("skipped");
      expect(request).toHaveBeenCalledTimes(1);
      if (providerId) {
        const exposed = [
          JSON.stringify(first),
          JSON.stringify(second),
          JSON.stringify([
            logger.info.mock.calls,
            logger.warn.mock.calls,
            logger.error.mock.calls,
            logger.debug.mock.calls,
          ]),
          await persistedDeliveryText(storage),
        ].join("\n");
        expect(exposed).not.toContain(providerId);
      }
    },
  );

  it.each([
    ["invalid JSON", "not-json"],
    ["oversized success body", `${" ".repeat(4096)}{"ok":true}`],
    ["null", "null"],
    ["array", JSON.stringify([{ ok: true }])],
    ["primitive", "true"],
    ["empty object", JSON.stringify({})],
    ["missing success marker", JSON.stringify({ id: "provider-id" })],
    ["false success marker", JSON.stringify({ ok: false })],
    ["string success marker", JSON.stringify({ ok: "true" })],
    ["duplicate success marker", '{"ok":false,"ok":true}'],
    [
      "duplicate provider ID",
      '{"ok":true,"id":"first-provider-id","id":"second-provider-id"}',
    ],
    [
      "escaped duplicate provider ID member",
      '{"ok":true,"\\u0069d":"first-provider-id","id":"second-provider-id"}',
    ],
    [
      "duplicate deduped marker",
      '{"ok":true,"id":"provider-id","deduped":false,"deduped":true}',
    ],
    ["empty provider ID", JSON.stringify({ ok: true, id: "" })],
    ["blank provider ID", JSON.stringify({ ok: true, id: "   " })],
    [
      "oversized provider ID",
      JSON.stringify({ ok: true, id: "p".repeat(legacyProviderIdMaxLength + 1) }),
    ],
    ["numeric provider ID", JSON.stringify({ ok: true, id: 1 })],
    ["null provider ID", JSON.stringify({ ok: true, id: null })],
    ["object provider ID", JSON.stringify({ ok: true, id: {} })],
    ["array provider ID", JSON.stringify({ ok: true, id: ["provider-id"] })],
    ["deduped without provider ID", JSON.stringify({ deduped: true, ok: true })],
    [
      "deduped false",
      JSON.stringify({ deduped: false, id: "provider-id", ok: true }),
    ],
    [
      "deduped string",
      JSON.stringify({ deduped: "true", id: "provider-id", ok: true }),
    ],
    ["extra canonical field", JSON.stringify({ extra: true, ok: true })],
    [
      "extra legacy field",
      JSON.stringify({ extra: true, id: "provider-id", ok: true }),
    ],
    [
      "extra deduplicated field",
      JSON.stringify({
        deduped: true,
        extra: true,
        id: "provider-id",
        ok: true,
      }),
    ],
  ])("keeps an invalid hosted 2xx blocking (%s)", async (_case, bodyText) => {
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

  describe("a failure that proves the relay was never asked to send", () => {
    const email = {
      enabled: true,
      mode: "hosted" as const,
      to: "you@example.com",
      hostedToken: "device_token_hex",
      fromEmail: "",
    };
    const ok = { status: 200, headers: {}, bodyText: JSON.stringify({ ok: true }) };
    const refusal = (error = "delivery cutover is not ready") =>
      ({ status: 503, headers: {}, bodyText: JSON.stringify({ error, ambiguous: false }) });
    const runner = (storage: StorageAdapter, request: ReturnType<typeof vi.fn>) => () =>
      deliverDailyEmailIfEnabled(digest, { storage, http: { request }, output, email });

    it("releases the claim after a connect-phase failure, then sends exactly once when the network is back", async () => {
      const storage = memoryStorage();
      const request = vi.fn()
        .mockRejectedValueOnce(new HttpTransportError("network", "connect ECONNREFUSED", { retryableAttempt: true, requestNotSent: true }))
        .mockResolvedValue(ok);
      const run = runner(storage, request);

      expect(await run()).toEqual({ kind: "failed", reason: "provider_not_invoked", attempts: 0 });
      expect(await run()).toEqual({ kind: "delivered", attempts: 1 });
      expect(await run()).toEqual({ kind: "skipped", reason: "already_delivered" });
      expect(request).toHaveBeenCalledTimes(2);
    });

    it.each([
      ["a timeout", new HttpTransportError("timeout", "no answer", { retryableAttempt: false })],
      ["a reset after sending", new HttpTransportError("network", "reset", { retryableAttempt: false })],
      ["a host that cannot tell", new HttpTransportError("network", "unknown", { retryableAttempt: true })],
      ["a non-transport error", new Error("something else")],
    ])("keeps %s ambiguous and blocking", async (_label, failure) => {
      const storage = memoryStorage();
      const request = vi.fn().mockRejectedValue(failure);
      const run = runner(storage, request);

      expect(await run()).toMatchObject({ kind: "ambiguous", attempts: 1 });
      expect(await run()).toMatchObject({ kind: "skipped", reason: "provider_outcome_ambiguous" });
      expect(request).toHaveBeenCalledTimes(1);
    });

    it.each([
      "delivery cutover is not ready",
      "delivery ledger is unavailable",
      "delivery attempt could not be started",
      "automatic delivery is unavailable",
    ])("treats the relay's 503 refusal \"%s\" as not sent and retries once it recovers", async (error) => {
      const storage = memoryStorage();
      const request = vi.fn().mockResolvedValueOnce(refusal(error)).mockResolvedValue(ok);
      const run = runner(storage, request);

      expect(await run()).toEqual({ kind: "failed", reason: "provider_not_invoked", attempts: 0 });
      expect(await run()).toEqual({ kind: "delivered", attempts: 1 });
      expect(request).toHaveBeenCalledTimes(2);
    });

    it.each([
      ["a 503 without a body", { status: 503, headers: {}, bodyText: "" }],
      ["a 503 error page", { status: 503, headers: {}, bodyText: "<html>Service Unavailable</html>" }],
      ["a 503 that says the outcome is ambiguous", { status: 503, headers: {}, bodyText: JSON.stringify({ error: "x", ambiguous: true }) }],
      ["a 503 without the ambiguous flag", { status: 503, headers: {}, bodyText: JSON.stringify({ error: "x" }) }],
      ["a 503 with extra members", { status: 503, headers: {}, bodyText: JSON.stringify({ error: "x", ambiguous: false, extra: 1 }) }],
      ["a 503 with a repeated member", { status: 503, headers: {}, bodyText: '{"error":"x","ambiguous":true,"ambiguous":false}' }],
      ["a 503 array", { status: 503, headers: {}, bodyText: JSON.stringify([{ error: "x", ambiguous: false }]) }],
      ["a 503 with a non-text error", { status: 503, headers: {}, bodyText: JSON.stringify({ error: 1, ambiguous: false }) }],
      ["a 500 with the refusal shape", { ...refusal(), status: 500 }],
      ["a 502 with the refusal shape", { ...refusal(), status: 502 }],
      ["a 409 with the refusal shape", { ...refusal(), status: 409 }],
    ])("does not trust %s", async (_label, response) => {
      const storage = memoryStorage();
      const request = vi.fn().mockResolvedValue(response);
      const run = runner(storage, request);

      expect(await run()).toMatchObject({ kind: "ambiguous", attempts: 1 });
      expect(await run()).toMatchObject({ kind: "skipped", reason: "provider_outcome_ambiguous" });
      expect(request).toHaveBeenCalledTimes(1);
    });
  });

  describe("a relay rejection says which kind it was", () => {
    const email = {
      enabled: true,
      mode: "hosted" as const,
      to: "you@example.com",
      hostedToken: "device_token_hex",
      fromEmail: "",
    };
    const reject = (status: number) => ({
      status,
      headers: {},
      bodyText: JSON.stringify({ error: "private relay detail you@example.com", ambiguous: false }),
    });
    const finerReasons = [
      "hosted_token_rejected",
      "hosted_recipient_mismatch",
      "provider_quota_exceeded",
      "provider_request_rejected",
    ];

    it.each([
      [401, "hosted_token_rejected"],
      [403, "hosted_recipient_mismatch"],
      [429, "provider_quota_exceeded"],
      [422, "provider_request_rejected"],
      [400, "provider_request_rejected"],
      [404, "provider_definitive_rejection"],
    ] as const)("reports HTTP %i as %s but keeps recording the coarse rejection", async (status, reason) => {
      const warn = vi.fn();
      const logger = { info: vi.fn(), warn, error: vi.fn(), debug: vi.fn() };
      const storage = memoryStorage();
      const request = vi.fn(async () => reject(status));

      const testSend = await deliverDailyEmailIfEnabled(digest, {
        storage, http: { request }, output, email, force: true, logger,
      });
      expect(testSend).toEqual({ kind: "failed", reason, attempts: 1 });

      const automatic = await deliverDailyEmailIfEnabled(digest, {
        storage, http: { request }, output, email, logger,
      });
      expect(automatic).toEqual({ kind: "failed", reason, attempts: 1 });
      expect(warn).toHaveBeenCalledWith(`email: ${reason} date=${digest.date} status=${status}`);

      // The durable record keeps the old code, so older clients read the same files.
      const persisted = await persistedDeliveryText(storage);
      expect(persisted).toContain("provider_definitive_rejection");
      for (const finer of finerReasons) expect(persisted).not.toContain(finer);
      // The next run is blocked with that recorded code, and the relay is not asked again.
      expect(await deliverDailyEmailIfEnabled(digest, { storage, http: { request }, output, email, logger }))
        .toEqual({ kind: "skipped", reason: "provider_definitive_rejection" });
      expect(request).toHaveBeenCalledTimes(2);
      // Neither the log nor the result carries the relay's response text.
      expect(JSON.stringify(warn.mock.calls)).not.toContain("you@example.com");
    });

    it("leaves uncertain outcomes ambiguous whatever the status", async () => {
      const storage = memoryStorage();
      const request = vi.fn(async () => reject(500));
      expect(await deliverDailyEmailIfEnabled(digest, { storage, http: { request }, output, email }))
        .toMatchObject({ kind: "ambiguous", reason: "provider_outcome_ambiguous" });
    });
  });

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
