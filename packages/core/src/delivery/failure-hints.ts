import type { EmailDeliveryReason } from "./types";

const HINTS: Partial<Record<EmailDeliveryReason, string>> = {
  hosted_token_rejected:
    "The verification code was rejected as invalid or expired. Send a new verification email and paste the new code into the Email settings.",
  hosted_recipient_mismatch:
    "The recipient is not the address this verification code was issued for. Verify the address you want to use, or change the recipient back.",
  provider_quota_exceeded:
    "The sending limit was reached (too many emails today, or too many requests). Try again later.",
  provider_request_rejected:
    "The mail service rejected the request as invalid. Check the recipient and sender addresses.",
  resend_api_key_rejected:
    "Resend rejected the API key. Check it, or create a new one, in the Email settings.",
  resend_access_denied:
    "Resend refused the send. Check that the API key is valid, that the From domain is verified, and that with the default onboarding@resend.dev sender the To address is your own Resend account email.",
};

/**
 * One actionable sentence for a delivery failure that has a known fix, or
 * undefined when the reason code is all there is to say. Fixed text only: never
 * a token, an address or a response body.
 */
export function emailFailureHint(reason: EmailDeliveryReason): string | undefined {
  return HINTS[reason];
}
