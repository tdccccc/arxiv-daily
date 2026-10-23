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

/**
 * Tell the user, once per day and outcome, when an automatic digest email did
 * not go out as expected. A host that cannot send automatically already has its
 * own one-time message, and a run the user cancelled needs no explanation.
 */
export function createAutomaticEmailOutcomeNotifier(
  notify: (message: string) => void,
): (date: string, result: DeliverEmailResult) => void {
  const told = new Set<string>();
  return (date, result) => {
    const message = describeAutomaticEmailOutcome(date, result);
    const key = `${date}|${result.kind}`;
    if (!message || told.has(key)) return;
    told.add(key);
    notify(message);
  };
}

function describeAutomaticEmailOutcome(
  date: string,
  result: DeliverEmailResult,
): string | undefined {
  if (result.kind === "ambiguous") {
    return `The digest email for ${date} may or may not have been delivered, and it will not be sent again automatically. Check your inbox.`;
  }
  if (result.kind !== "failed") return undefined;
  switch (result.reason) {
    case "delivery_storage_unsupported":
    case "cancelled_before_provider_attempt":
      return undefined;
    case "provider_not_invoked":
      return `The digest email for ${date} was not sent because the mail service could not be reached. Nothing was delivered.`;
    case "delivery_claim_unreadable":
      return `The digest email for ${date} was not sent because a delivery record file is unreadable. See the developer console for its name.`;
    default:
      return `The digest email for ${date} was not sent (${result.reason}). See the developer console for details.`;
  }
}
