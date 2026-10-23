import { describe, expect, it, vi } from "vitest";
import ArxivDailyPlugin from "../main.ts";
import { Notice } from "obsidian";
import { AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE, DEFAULT_SETTINGS, HttpTransportError, Logger, sampleDailyDigest, type HttpRequest, type StorageAdapter } from "@arxiv-daily/core";
import {
  createAutomaticEmailOutcomeNotifier,
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

  it.each([
    ["hosted_token_rejected", /verification code was rejected.*Send a new verification email/],
    ["hosted_recipient_mismatch", /not the address this verification code was issued for/],
    ["provider_quota_exceeded", /sending limit was reached/],
    ["provider_request_rejected", /rejected the request as invalid/],
    ["resend_api_key_rejected", /Resend rejected the API key/],
    ["resend_access_denied", /From domain is verified.*onboarding@resend\.dev/],
  ] as const)("tells the user what to do about %s and still names the code", (reason, advice) => {
    let message = "";
    try {
      testEmailResultMessage({ kind: "failed", reason, attempts: 1 }, true);
    } catch (error) {
      message = (error as Error).message;
    }
    expect(message).toMatch(advice);
    expect(message).toContain(`(failed: ${reason})`);
  });

  it("keeps the bare code for a rejection with no known fix", () => {
    expect(() => testEmailResultMessage(
      { kind: "failed", reason: "provider_definitive_rejection", attempts: 1 },
      true,
    )).toThrow(/^failed: provider_definitive_rejection$/);
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

describe("createAutomaticEmailOutcomeNotifier", () => {
  it("tells the user once per day when an email was not sent or may not have arrived", () => {
    const notify = vi.fn();
    const onResult = createAutomaticEmailOutcomeNotifier(notify);

    onResult("2026-10-20", { kind: "failed", reason: "provider_not_invoked", attempts: 0 });
    onResult("2026-10-20", { kind: "failed", reason: "provider_not_invoked", attempts: 0 });
    expect(notify).toHaveBeenCalledTimes(1);
    expect(notify.mock.calls[0]![0]).toContain("2026-10-20");
    expect(notify.mock.calls[0]![0]).toContain("could not be reached");

    onResult("2026-10-21", { kind: "ambiguous", reason: "provider_outcome_ambiguous", attempts: 1 });
    expect(notify).toHaveBeenCalledTimes(2);
    expect(notify.mock.calls[1]![0]).toContain("may or may not have been delivered");

    onResult("2026-10-22", { kind: "failed", reason: "delivery_claim_unreadable", attempts: 0 });
    expect(notify.mock.calls[2]![0]).toContain("unreadable");
    onResult("2026-10-23", { kind: "failed", reason: "provider_definitive_rejection", attempts: 1 });
    expect(notify.mock.calls[3]![0]).toContain("provider_definitive_rejection");
  });

  it("says what to do when the verification code was rejected", () => {
    const notify = vi.fn();
    const onResult = createAutomaticEmailOutcomeNotifier(notify);

    onResult("2026-10-24", { kind: "failed", reason: "hosted_token_rejected", attempts: 1 });
    expect(notify).toHaveBeenCalledTimes(1);
    expect(notify.mock.calls[0]![0]).toContain("The digest email for 2026-10-24 was not sent.");
    expect(notify.mock.calls[0]![0]).toContain("Send a new verification email");
    expect(notify.mock.calls[0]![0]).not.toContain("developer console");
  });

  it.each([
    ["hosted_recipient_mismatch", "not the address"],
    ["provider_quota_exceeded", "sending limit"],
    ["provider_request_rejected", "request as invalid"],
    ["resend_api_key_rejected", "Resend rejected the API key"],
    ["resend_access_denied", "From domain is verified"],
  ] as const)("explains automatic delivery failure %s", (reason, advice) => {
    const notify = vi.fn();
    const onResult = createAutomaticEmailOutcomeNotifier(notify);
    onResult("2026-10-24", { kind: "failed", reason, attempts: 1 });
    expect(notify).toHaveBeenCalledTimes(1);
    expect(notify.mock.calls[0]![0]).toContain(advice);
  });

  it("stays quiet for outcomes that need no action from the user", () => {
    const notify = vi.fn();
    const onResult = createAutomaticEmailOutcomeNotifier(notify);

    onResult("2026-10-20", { kind: "delivered", attempts: 1 });
    onResult("2026-10-20", { kind: "skipped", reason: "already_delivered" });
    onResult("2026-10-20", { kind: "disabled", reason: "email_delivery_disabled" });
    onResult("2026-10-20", { kind: "failed", reason: "delivery_storage_unsupported", attempts: 0 });
    onResult("2026-10-20", { kind: "failed", reason: "cancelled_before_provider_attempt", attempts: 0 });

    expect(notify).not.toHaveBeenCalled();
  });
});

describe("plugin automatic email after a completed run", () => {
  function claimStorage(): StorageAdapter {
    const files = new Map<string, string>();
    return {
      normalizePath: path => path,
      readText: async path => { if (!files.has(path)) throw new Error(`missing ${path}`); return files.get(path)!; },
      writeText: async (path, text) => { files.set(path, text); },
      writeTextAtomic: async (path, text) => { files.set(path, text); },
      createTextExclusive: async (path, text) => { if (files.has(path)) return false; files.set(path, text); return true; },
      guardClaimNamespace: async () => ({ assertCurrent() {}, async release() {} }),
      exists: async path => files.has(path), mkdir: async () => {}, remove: async path => { files.delete(path); },
      rename: async (from, to) => { files.set(to, files.get(from)!); files.delete(from); },
      list: async dir => [...files.keys()].filter(path => path.startsWith(`${dir}/`)).map(path => ({ path, type: "file" as const })),
    };
  }

  it("shows a notice when the mail service could not be reached, then stays quiet once it has been delivered", async () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.email = { ...settings.email, enabled: true, mode: "self", to: "reader@example.test", apiKey: "re_private" };
    const request = vi.fn()
      .mockRejectedValueOnce(new HttpTransportError("network", "getaddrinfo ENOTFOUND", { requestNotSent: true }))
      .mockResolvedValue({ status: 200, headers: {}, bodyText: JSON.stringify({ id: "sent" }) });
    const plugin = Object.assign(new ArxivDailyPlugin(), {
      settings, host: { storage: claimStorage(), http: { request } }, logger: new Logger("error"),
    });
    const result = (date: string) => ({ kind: "completed" as const, papersWritten: 1, digest: sampleDailyDigest({ date }) });
    Notice.calls = [];

    await plugin.deliverCompletedDigest("2026-10-20", result("2026-10-20"));
    expect(Notice.calls.map(call => call.message)).toEqual([
      expect.stringMatching(/^arXiv Daily: The digest email for 2026-10-20 was not sent because the mail service could not be reached/),
    ]);

    Notice.calls = [];
    await plugin.deliverCompletedDigest("2026-10-20", result("2026-10-20"));
    expect(Notice.calls).toEqual([]);
    expect(request).toHaveBeenCalledTimes(2);
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
