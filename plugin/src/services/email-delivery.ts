import {
  AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE,
  type DeliverEmailResult,
} from "@arxiv-daily/core";

/**
 * Notice text for a test send. A test send skips the automatic-delivery claim,
 * so on hosts that cannot claim it would succeed while every daily send is
 * refused; the success message says so instead of implying email is set up.
 */
export function testEmailResultMessage(
  result: DeliverEmailResult,
  automaticSupported: boolean,
): string {
  if (result.kind !== "delivered" && result.kind !== "delivered_unrecorded") {
    throw new Error(`${result.kind}: ${result.reason}`);
  }
  const message = "Test email delivered" +
    (result.kind === "delivered_unrecorded"
      ? `; delivery record unavailable: ${result.reason}`
      : "");
  return automaticSupported
    ? message
    : `${message}. ${AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE}`;
}

/** Tell the user once per session when an automatic send is refused on this host. */
export function createUnsupportedAutomaticEmailNotifier(
  notify: (message: string) => void,
): (result: DeliverEmailResult) => void {
  let notified = false;
  return (result) => {
    if (
      notified ||
      result.kind !== "failed" ||
      result.reason !== "delivery_storage_unsupported"
    ) return;
    notified = true;
    notify(AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE);
  };
}
