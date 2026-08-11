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
import { RESEND_API_URL, sendViaResend } from "../../src/delivery/resend";
import {
  HttpTransportError,
  type HttpClient,
  type StorageAdapter,
} from "../../src/core/adapters";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import { RunCancelledError } from "../../src/services/cancellation";

async function persistedDeliveryText(
  storage: StorageAdapter,
  output = DEFAULT_SETTINGS.output,
): Promise<string> {
  const statePath = deliveryStatePath(output);
  const chunks: string[] = [];
  if (await storage.exists(statePath)) chunks.push(await storage.readText(statePath));
  const claimDir = `${statePath}.claims`;
  for (const entry of (await storage.list?.(claimDir)) ?? []) {
    if (entry.type === "file") chunks.push(await storage.readText(entry.path));
  }
  return chunks.join("\n");
}

function memoryStorage(
  initial: Record<string, string> = {},
  options: {
    unreadable?: boolean;
    failWrite?: (path: string) => boolean;
    failRemove?: (path: string) => boolean;
  } = {},
): StorageAdapter {
  const files = new Map(Object.entries(initial));
  const storage: StorageAdapter & {
    createTextExclusive(path: string, content: string): Promise<boolean>;
  } = {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/"),
    async readText(path) {
      if (options.unreadable) throw new Error(`unreadable ${path}`);
      const value = files.get(path);
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    },
    async writeText(path, content) {
      if (options.failWrite?.(path)) throw new Error(`cannot write ${path}`);
      files.set(path, content);
    },
    async writeTextAtomic(path, content) {
      if (options.failWrite?.(path)) throw new Error(`cannot write ${path}`);
      files.set(path, content);
    },
    async createTextExclusive(path, content) {
      if (files.has(path)) return false;
      files.set(path, content);
      return true;
    },
    async guardClaimNamespace() {
      return { assertCurrent() {}, async release() {} };
    },
    async exists(path) {
      return files.has(path);
    },
    async mkdir() {},
    async remove(path) {
      if (options.failRemove?.(path)) throw new Error(`cannot remove ${path}`);
      files.delete(path);
    },
    async rename(from, to) {
      const value = files.get(from);
      if (value === undefined) throw new Error(`missing ${from}`);
      files.set(to, value);
      files.delete(from);
    },
    async list(dir) {
      const prefix = dir ? `${dir}/` : "";
      return Array.from(files.keys())
        .filter((path) => path.startsWith(prefix))
        .map((path) => ({ path, type: "file" as const }));
    },
  };
  return storage;
}

describe("sendViaResend", () => {
  it("posts to Resend without exposing the provider response id", async () => {
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "email_123" }),
    }));
    const http: HttpClient = { request };
    const result = await sendViaResend({
      http,
      apiKey: "re_test",
      idempotencyKey: "arxiv-daily:test:direct",
      payload: {
        from: "Daily <from@example.com>",
        to: "you@example.com",
        subject: "test",
        html: "<p>hi</p>",
        text: "hi",
      },
      sleep: async () => {},
    });
    expect(result).toEqual({ attempts: 1, status: 200 });
    expect(JSON.stringify(result)).not.toContain("email_123");
    expect(request).toHaveBeenCalledWith(
      expect.objectContaining({
        url: RESEND_API_URL,
        method: "POST",
        headers: expect.objectContaining({
          Authorization: "Bearer re_test",
        }),
      }),
    );
  });

  it("retries 500 then fails after max attempts", async () => {
    const request = vi.fn(async () => ({
      status: 500,
      headers: {},
      bodyText: "nope",
    }));
    await expect(
      sendViaResend({
        http: { request },
        apiKey: "re_test",
        idempotencyKey: "arxiv-daily:test:retry",
        payload: {
          from: "from@example.com",
          to: "you@example.com",
          subject: "t",
          html: "h",
          text: "t",
        },
        maxAttempts: 3,
        baseDelayMs: 1,
        sleep: async () => {},
      }),
    ).rejects.toThrow(/Resend HTTP 500/);
    expect(request).toHaveBeenCalledTimes(3);
  });

  it.each([
    [400, false],
    [401, false],
    [403, false],
    [404, false],
    [422, false],
    [429, false],
    [408, true],
    [409, true],
    [500, true],
  ] as const)(
    "classifies HTTP %s with ambiguous=%s",
    async (status, ambiguous) => {
      const request = vi.fn(async () => ({
        status,
        headers: {},
        bodyText:
          status === 409
            ? JSON.stringify({ name: "concurrent_idempotent_requests", message: "in progress" })
            : "provider error",
      }));
      const error = await sendViaResend({
        http: { request },
        apiKey: "re_test",
        idempotencyKey: "arxiv-daily:auto:classification",
        payload: {
          from: "from@example.com",
          to: "you@example.com",
          subject: "t",
          html: "h",
          text: "t",
        },
        maxAttempts: 1,
        sleep: async () => {},
      }).catch((value) => value);

      expect(error).toMatchObject({
        name: "ResendSendError",
        status,
        ambiguous,
      });
      expect(request).toHaveBeenCalledTimes(1);
    },
  );

  it("treats an invalid successful response as ambiguous", async () => {
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: "not-json",
    }));

    await expect(sendViaResend({
      http: { request },
      apiKey: "re_test",
      idempotencyKey: "arxiv-daily:auto:invalid-success",
      payload: {
        from: "from@example.com",
        to: "you@example.com",
        subject: "t",
        html: "h",
        text: "t",
      },
      maxAttempts: 1,
    })).rejects.toMatchObject({
      name: "ResendSendError",
      ambiguous: true,
    });
  });

  it("fails fast on 401 without exhausting endless retries", async () => {
    const request = vi.fn(async () => ({
      status: 401,
      headers: {},
      bodyText: "unauthorized",
    }));
    await expect(
      sendViaResend({
        http: { request },
        apiKey: "bad",
        idempotencyKey: "arxiv-daily:test:bad-key",
        payload: {
          from: "from@example.com",
          to: "you@example.com",
          subject: "t",
          html: "h",
          text: "t",
        },
        maxAttempts: 3,
        sleep: async () => {},
      }),
    ).rejects.toThrow(/401/);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("reuses one bounded idempotency key across internal retries", async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce({ status: 500, headers: {}, bodyText: "retry" })
      .mockResolvedValueOnce({
        status: 200,
        headers: {},
        bodyText: JSON.stringify({ id: "email_ok" }),
      });
    const idempotencyKey = "arxiv-daily:auto:abcdef";

    await sendViaResend({
      http: { request },
      apiKey: "re_test",
      idempotencyKey,
      payload: {
        from: "from@example.com",
        to: "private-recipient@example.com",
        subject: "t",
        html: "h",
        text: "t",
      },
      maxAttempts: 2,
      sleep: async () => {},
    });

    const keys = request.mock.calls.map(
      ([req]) => req.headers?.["Idempotency-Key"],
    );
    expect(keys).toEqual([idempotencyKey, idempotencyKey]);
    expect(idempotencyKey.length).toBeLessThanOrEqual(128);
    expect(idempotencyKey).not.toContain("private-recipient@example.com");
  });
});

describe("deliverDailyEmailIfEnabled", () => {
  const output = DEFAULT_SETTINGS.output;
  const digest = sampleDailyDigest({ date: "2026-07-26", language: "zh" });

  it("delivers with empty From using quick sender, then skips when already delivered", async () => {
    const storage = memoryStorage();
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "msg_ok" }),
    }));
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "you@example.com",
      fromEmail: "",
      fromName: "",
      apiKey: "re_key",
    };

    const first = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      sleep: async () => {},
    });
    expect(first).toEqual({
      kind: "delivered",
      attempts: 1,
    });
    expect(JSON.stringify(first)).not.toContain("msg_ok");
    expect(await persistedDeliveryText(storage)).not.toContain("msg_ok");
    expect(request).toHaveBeenCalledTimes(1);
    const body = JSON.parse(String(request.mock.calls[0]![0].body));
    expect(body.from).toContain("onboarding@resend.dev");
    expect(body.from).toContain("arXiv Daily");

    const state = await loadDeliveryState(storage, output);
    expect(shouldSendEmail(state, digest.date, email.to)).toBe(false);

    const second = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      sleep: async () => {},
    });
    expect(second.kind).toBe("skipped");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("force test-send does not mark the day delivered for auto-send skip", async () => {
    const storage = memoryStorage();
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "msg_test" }),
    }));
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "you@example.com",
      fromEmail: "",
      apiKey: "re_key",
    };

    const testSend = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      force: true,
      sleep: async () => {},
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
      sleep: async () => {},
    });
    expect(auto.kind).toBe("delivered");
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("persists and replays a definitive provider rejection without retry", async () => {
    const storage = memoryStorage();
    const request = vi
      .fn()
      .mockResolvedValueOnce({ status: 400, headers: {}, bodyText: "invalid" })
      .mockResolvedValueOnce({
        status: 200,
        headers: {},
        bodyText: JSON.stringify({ id: "msg_retry" }),
      });
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "you@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
    };

    const failed = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      maxAttempts: 3,
      baseDelayMs: 1,
      sleep: async () => {},
    });
    expect(failed.kind).toBe("failed");
    expect(request).toHaveBeenCalledTimes(1);

    const state = await loadDeliveryState(storage, output);
    expect(shouldSendEmail(state, digest.date, email.to)).toBe(false);

    const retry = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      maxAttempts: 3,
      baseDelayMs: 1,
      sleep: async () => {},
    });
    expect(retry).toEqual({
      kind: "skipped",
      reason: "provider_definitive_rejection",
    });
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("does not expose or persist a sensitive Resend rejection body", async () => {
    const storage = memoryStorage();
    const sensitive = [
      "recipient@example.com",
      "tok_live_super_secret",
      "private email body",
    ];
    const request = vi.fn(async () => ({
      status: 400,
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
        mode: "self",
        to: "you@example.com",
        fromEmail: "from@example.com",
        apiKey: "re_key",
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

  it("allows only one provider request for concurrent automatic delivery", async () => {
    const storage = memoryStorage();
    let release!: () => void;
    const provider = new Promise<void>((resolve) => { release = resolve; });
    const request = vi.fn(async () => {
      await provider;
      return {
        status: 200,
        headers: {},
        bodyText: JSON.stringify({ id: "msg_once" }),
      };
    });
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "Concurrent@Example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
    };

    const first = deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      sleep: async () => {},
    });
    const second = deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email: { ...email, to: " concurrent@example.COM " },
      sleep: async () => {},
    });
    await vi.waitFor(() => expect(request).toHaveBeenCalled());
    release();

    const results = await Promise.all([first, second]);
    expect(request).toHaveBeenCalledTimes(1);
    expect(results.map((result) => result.kind).sort()).toEqual([
      "delivered",
      "skipped",
    ]);
  });

  it.each([
    ["corrupt", "{broken", {}],
    ["unreadable", "{}", { unreadable: true }],
  ] as const)("fails closed for %s state without a provider request", async (_kind, raw, options) => {
    const statePath = deliveryStatePath(output);
    const storage = memoryStorage({ [statePath]: raw }, options);
    const request = vi.fn();

    const result = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email: {
        enabled: true,
        mode: "self",
        to: "you@example.com",
        fromEmail: "from@example.com",
        apiKey: "re_key",
      },
    });

    expect(result).toMatchObject({ kind: "failed", attempts: 0 });
    expect(request).not.toHaveBeenCalled();
  });

  it("recovers a claim whose initial state snapshot failed before provider attempt", async () => {
    const statePath = deliveryStatePath(output);
    let fail = true;
    const storage = memoryStorage({}, {
      failWrite: (path) => path === statePath && fail,
    });
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "msg_after_state_recovery" }),
    }));
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "state-recovery@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
    };

    const failed = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });
    expect(failed).toMatchObject({ kind: "failed", attempts: 0 });
    expect(request).not.toHaveBeenCalled();

    fail = false;
    const retry = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });
    expect(retry.kind).toBe("delivered");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("returns delivered_unrecorded and keeps blocking after provider success if final save fails", async () => {
    const statePath = deliveryStatePath(output);
    let stateWrites = 0;
    const storage = memoryStorage({}, {
      // Claim snapshot + provider-attempt snapshot succeed; final delivered
      // snapshot fails after the immutable result marker has been created.
      failWrite: (path) => path === statePath && ++stateWrites >= 3,
    });
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "msg_physical_success" }),
    }));
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "you@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
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

    expect(first).toMatchObject({
      kind: "delivered_unrecorded",
      reason: "delivery_state_update_failed",
    });
    expect(JSON.stringify(first)).not.toContain("msg_physical_success");
    expect(second.kind).toBe("skipped");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("keeps a definitive rejection result blocking when the main-state rebuild fails", async () => {
    const statePath = deliveryStatePath(output);
    let stateWrites = 0;
    const storage = memoryStorage({}, {
      failWrite: (path) => path === statePath && ++stateWrites >= 3,
    });
    const request = vi.fn(async () => ({
      status: 400,
      headers: {},
      bodyText: "invalid",
    }));
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "rejection-rebuild@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
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

    expect(first).toEqual({
      kind: "ambiguous",
      reason: "delivery_state_update_failed",
      attempts: 1,
    });
    expect(second).toMatchObject({ kind: "skipped" });
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("releases a cancelled claim when cancellation wins before provider invocation", async () => {
    const storage = memoryStorage();
    const controller = new AbortController();
    controller.abort("cancel before send");
    const request = vi.fn();
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "cancel@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
    };

    const cancelled = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      signal: controller.signal,
    });
    expect(cancelled).toMatchObject({ kind: "failed", attempts: 0 });
    expect(request).not.toHaveBeenCalled();

    const retry = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: {
        request: vi.fn(async () => ({
          status: 200,
          headers: {},
          bodyText: JSON.stringify({ id: "msg_after_cancel" }),
        })),
      },
      output,
      email,
    });
    expect(retry.kind).toBe("delivered");
  });

  it("keeps cancellation during the provider invocation ambiguous and blocking", async () => {
    const storage = memoryStorage();
    const request = vi.fn(async () => {
      throw new RunCancelledError("cancelled during request");
    });
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "cancel@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
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

  it("does not rely on claim-file deletion to replay definitive rejection", async () => {
    const storage = memoryStorage({}, { failRemove: () => true });
    const request = vi
      .fn()
      .mockResolvedValueOnce({ status: 400, headers: {}, bodyText: "invalid" })
      .mockResolvedValueOnce({
        status: 200,
        headers: {},
        bodyText: JSON.stringify({ id: "msg_retry" }),
      });
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "remove@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
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

    expect(rejected.kind).toBe("failed");
    expect(retry).toMatchObject({
      kind: "skipped",
      reason: "provider_definitive_rejection",
    });
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("returns ambiguous and keeps the automatic claim on ambiguous transport", async () => {
    const storage = memoryStorage();
    const request = vi.fn(async () => {
      throw new HttpTransportError("timeout", "response was not observed", {
        retryableAttempt: false,
      });
    });
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "you@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
    };

    const first = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      maxAttempts: 3,
      sleep: async () => {},
    });
    const second = await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      maxAttempts: 3,
      sleep: async () => {},
    });

    expect(first).toMatchObject({ kind: "ambiguous", attempts: 1 });
    expect(second.kind).toBe("skipped");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("uses one stable PII-free formal key and fresh keys for explicit tests", async () => {
    const storage = memoryStorage();
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "msg_ok" }),
    }));
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "private-recipient@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_key",
    };

    await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      force: true,
    });
    await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
      force: true,
    });
    await deliverDailyEmailIfEnabled(digest, {
      storage,
      http: { request },
      output,
      email,
    });

    const keys = request.mock.calls.map(
      ([req]) => req.headers?.["Idempotency-Key"] ?? "",
    );
    expect(new Set(keys.slice(0, 2)).size).toBe(2);
    expect(keys[2]).toMatch(/^arxiv-daily:auto:/);
    for (const key of keys) {
      expect(key.length).toBeLessThanOrEqual(128);
      expect(key).not.toContain("private-recipient@example.com");
    }
  });

  it("preserves concurrent unrelated delivery-state mutations", async () => {
    const storage = memoryStorage();
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: `msg_${request.mock.calls.length}` }),
    }));
    const baseEmail = {
      enabled: true,
      mode: "self" as const,
      fromEmail: "from@example.com",
      apiKey: "re_key",
    };

    const results = await Promise.all([
      deliverDailyEmailIfEnabled(digest, {
        storage,
        http: { request },
        output,
        email: { ...baseEmail, to: "one@example.com" },
      }),
      deliverDailyEmailIfEnabled(digest, {
        storage,
        http: { request },
        output,
        email: { ...baseEmail, to: "two@example.com" },
      }),
    ]);

    expect(results.map((result) => result.kind)).toEqual([
      "delivered",
      "delivered",
    ]);
    const state = await loadDeliveryState(storage, output);
    expect(shouldSendEmail(state, digest.date, "one@example.com")).toBe(false);
    expect(shouldSendEmail(state, digest.date, "two@example.com")).toBe(false);
  });
});
