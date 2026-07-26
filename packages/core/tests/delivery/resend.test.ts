import { describe, expect, it, vi } from "vitest";
import {
  deliverDailyEmailIfEnabled,
  sampleDailyDigest,
} from "../../src/delivery/deliver-email";
import {
  loadDeliveryState,
  shouldSendEmail,
} from "../../src/delivery/delivery-state";
import { RESEND_API_URL, sendViaResend } from "../../src/delivery/resend";
import type { HttpClient, StorageAdapter } from "../../src/core/adapters";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";

function memoryStorage(initial: Record<string, string> = {}): StorageAdapter {
  const files = new Map(Object.entries(initial));
  return {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/"),
    async readText(path) {
      const value = files.get(path);
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    },
    async writeText(path, content) {
      files.set(path, content);
    },
    async exists(path) {
      return files.has(path);
    },
    async mkdir() {},
    async remove(path) {
      files.delete(path);
    },
    async rename(from, to) {
      const value = files.get(from);
      if (value === undefined) throw new Error(`missing ${from}`);
      files.set(to, value);
      files.delete(from);
    },
  };
}

describe("sendViaResend", () => {
  it("posts to Resend and returns provider id", async () => {
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "email_123" }),
    }));
    const http: HttpClient = { request };
    const result = await sendViaResend({
      http,
      apiKey: "re_test",
      payload: {
        from: "Daily <from@example.com>",
        to: "you@example.com",
        subject: "test",
        html: "<p>hi</p>",
        text: "hi",
      },
      sleep: async () => {},
    });
    expect(result.providerMessageId).toBe("email_123");
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
      providerMessageId: "msg_ok",
      attempts: 1,
    });
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

  it("marks failed after 500 retries and allows later retry", async () => {
    const storage = memoryStorage();
    const request = vi
      .fn()
      .mockResolvedValueOnce({ status: 500, headers: {}, bodyText: "err" })
      .mockResolvedValueOnce({ status: 500, headers: {}, bodyText: "err" })
      .mockResolvedValueOnce({ status: 500, headers: {}, bodyText: "err" })
      .mockResolvedValueOnce({
        status: 200,
        headers: {},
        bodyText: JSON.stringify({ id: "msg_retry" }),
      });
    const email = {
      enabled: true,
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
    expect(request).toHaveBeenCalledTimes(3);

    const state = await loadDeliveryState(storage, output);
    expect(shouldSendEmail(state, digest.date, email.to)).toBe(true);

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
      kind: "delivered",
      providerMessageId: "msg_retry",
      attempts: 1,
    });
  });
});
