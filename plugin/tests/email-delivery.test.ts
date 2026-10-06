import { describe, expect, it, vi } from "vitest";
import ArxivDailyPlugin from "../main.ts";
import { AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE, DEFAULT_SETTINGS, Logger, type HttpRequest, type StorageAdapter } from "@arxiv-daily/core";
import {
  createUnsupportedAutomaticEmailNotifier,
  testEmailResultMessage,
} from "../src/services/email-delivery";

describe("testEmailResultMessage", () => {
  it("confirms delivery when automatic email works here", () => {
    expect(testEmailResultMessage({ kind: "delivered", attempts: 1 }, true))
      .toBe("Test email delivered");
  });

  it("warns after a successful test that daily email will not be sent automatically", () => {
    expect(testEmailResultMessage({ kind: "delivered", attempts: 1 }, false))
      .toBe(`Test email delivered. ${AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE}`);
  });

  it("keeps the delivery-record note", () => {
    expect(testEmailResultMessage(
      { kind: "delivered_unrecorded", attempts: 1, reason: "delivery_state_write_failed" } as never,
      true,
    )).toBe("Test email delivered; delivery record unavailable: delivery_state_write_failed");
  });

  it("throws on failure", () => {
    expect(() => testEmailResultMessage(
      { kind: "failed", reason: "resend_http_error", attempts: 1 } as never,
      true,
    )).toThrow("failed: resend_http_error");
  });
});

describe("createUnsupportedAutomaticEmailNotifier", () => {
  it("tells the user once when an automatic send is refused on this system", () => {
    const notify = vi.fn();
    const onResult = createUnsupportedAutomaticEmailNotifier(notify);

    onResult({ kind: "delivered", attempts: 1 });
    onResult({ kind: "failed", reason: "resend_http_error", attempts: 1 } as never);
    expect(notify).not.toHaveBeenCalled();

    onResult({ kind: "failed", reason: "delivery_storage_unsupported", attempts: 0 });
    onResult({ kind: "failed", reason: "delivery_storage_unsupported", attempts: 0 });
    expect(notify).toHaveBeenCalledTimes(1);
    expect(notify).toHaveBeenCalledWith(AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE);
  });
});


describe("plugin settings email operations", () => {
  function fixture() {
    const files = new Map<string, string>();
    const storage: StorageAdapter = {
      normalizePath: path => path, readText: async path => { if (!files.has(path)) throw new Error("missing"); return files.get(path)!; },
      writeText: async (path,text) => { files.set(path,text); }, writeTextAtomic: async (path,text) => { files.set(path,text); },
      exists: async path => files.has(path), mkdir: async () => {}, remove: async () => {}, rename: async () => {}, list: async () => [],
    };
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.email = {...settings.email,enabled:false,mode:"self",to:"reader@example.test",apiKey:"private-test-mail"};
    settings.output.summaryLanguage = "en";
    const request = vi.fn(async (_request: HttpRequest) => ({status:200,headers:{},bodyText:JSON.stringify({id:"test-mail"})}));
    const plugin = Object.assign(Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin, {settings,host:{storage,http:{request}},logger:new Logger("error")});
    return {plugin,request,settings};
  }
  it("preserves the selected date, explicit test-send and host support message", async () => {
    const {plugin,request,settings} = fixture();
    expect(await plugin.sendTestEmail("2026-10-04")).toBe(`Test email delivered. ${AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE}`);
    expect(settings.email.enabled).toBe(false);
    expect(JSON.parse(request.mock.calls[0]![0].body!).subject).toContain("2026-10-04");
    expect(request.mock.calls[0]![0].headers?.Authorization).toBe("Bearer private-test-mail");
  });
  it("keeps the plugin verification endpoint and its empty-address message", async () => {
    const {plugin,request,settings} = fixture(); settings.email.hostedBaseUrl = "https://custom.example.test";
    expect(await plugin.sendHostedVerificationEmail()).toContain("Verification email sent");
    expect(request.mock.calls[0]![0].url).toBe("https://custom.example.test/v1/verify/start");
    settings.email.to = " ";
    await expect(plugin.sendHostedVerificationEmail()).rejects.toThrow("Enter your email before sending a verification message");
    expect(request).toHaveBeenCalledTimes(1);
  });
});
