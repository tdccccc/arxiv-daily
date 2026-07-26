import { describe, expect, it } from "vitest";
import {
  deliveryRecordKey,
  emptyDeliveryState,
  markDelivered,
  markFailed,
  shouldSendEmail,
} from "../../src/delivery/delivery-state";
import { EMAIL_DELIVERY_CHANNEL } from "../../src/delivery/types";

describe("delivery-state", () => {
  it("builds stable keys with lowercased recipient", () => {
    expect(deliveryRecordKey("2026-07-26", "You@Example.COM")).toBe(
      `2026-07-26|you@example.com|${EMAIL_DELIVERY_CHANNEL}`,
    );
  });

  it("skips when delivered and allows when missing or failed", () => {
    let state = emptyDeliveryState(new Date("2026-07-26T00:00:00.000Z"));
    expect(shouldSendEmail(state, "2026-07-26", "a@b.com")).toBe(true);

    state = markFailed(state, {
      date: "2026-07-26",
      recipient: "a@b.com",
      attempts: 3,
      lastError: "boom",
      now: new Date("2026-07-26T01:00:00.000Z"),
    });
    expect(shouldSendEmail(state, "2026-07-26", "a@b.com")).toBe(true);
    expect(
      state.records[deliveryRecordKey("2026-07-26", "a@b.com")]?.status,
    ).toBe("failed");

    state = markDelivered(state, {
      date: "2026-07-26",
      recipient: "a@b.com",
      attempts: 1,
      providerMessageId: "msg_1",
      now: new Date("2026-07-26T02:00:00.000Z"),
    });
    expect(shouldSendEmail(state, "2026-07-26", "a@b.com")).toBe(false);
    expect(
      state.records[deliveryRecordKey("2026-07-26", "a@b.com")],
    ).toMatchObject({
      status: "delivered",
      providerMessageId: "msg_1",
      attempts: 1,
    });
  });
});
