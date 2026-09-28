import { describe, expect, it, vi } from "vitest";
import { AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE } from "@arxiv-daily/core";
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
