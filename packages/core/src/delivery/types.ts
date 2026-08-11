import type { SummaryLanguage } from "../settings/types";

/** Self-send via user Resend API key (自己发送). */
export const EMAIL_DELIVERY_CHANNEL = "email:resend" as const;
/** Hosted project relay (官方代发, Beta when shipped). */
export const EMAIL_HOSTED_CHANNEL = "email:hosted" as const;

export type EmailDeliveryChannel =
  | typeof EMAIL_DELIVERY_CHANNEL
  | typeof EMAIL_HOSTED_CHANNEL;

/** Settings: which exit path to use. Default self. */
export type EmailDeliveryMode = "self" | "hosted";

/**
 * Official delivery (Beta) client path is enabled in the plugin.
 * The Cloudflare Worker must be deployed for requests to succeed.
 * Keep UI labeled Beta until you are ready for wider users.
 */
export const OFFICIAL_DELIVERY_AVAILABLE = true;

export type DigestPaperKind = "structured" | "fallback";

export interface DigestStructuredFields {
  coreProblem: string;
  keyMethod: string;
  mainResult: string;
  whyRelevant: string;
  limitations: string;
}

export interface DigestPaper {
  id: string;
  title: string;
  authors: string;
  topicTag: string;
  sourceSections?: string;
  absUrl: string;
  pdfUrl: string;
  kind: DigestPaperKind;
  fields?: DigestStructuredFields;
  abstract?: string;
}

export interface DigestTopic {
  tag: string;
  name: string;
  papers: DigestPaper[];
}

export interface DailyDigest {
  date: string;
  summaryLanguage: SummaryLanguage;
  categories: string;
  dailyPath: string;
  paperCount: number;
  topics: DigestTopic[];
}

export type DeliveryStatus = "delivered" | "failed";
export type DeliveryPhase = "claimed" | "delivered" | "ambiguous";

export interface DeliveryRecord {
  date: string;
  recipient: string;
  channel: EmailDeliveryChannel;
  /**
   * Claims intentionally use v1's delivered status so older clients fail closed.
   * New clients use deliveryPhase to expose the more precise outcome.
   */
  status: DeliveryStatus;
  deliveryPhase?: DeliveryPhase;
  updatedAt: string;
  attempts: number;
  lastError?: string;
}

export interface DeliveryStateFile {
  schemaVersion: 1;
  updatedAt: string;
  records: Record<string, DeliveryRecord>;
}

export interface ResendEmailPayload {
  from: string;
  to: string;
  subject: string;
  html: string;
  text: string;
}

/** Stable, PII-free reason codes safe for state, logs, and caller output. */
export type EmailDeliveryReason =
  | "email_delivery_disabled"
  | "recipient_missing"
  | "official_delivery_unavailable"
  | "verification_token_missing"
  | "resend_api_key_missing"
  | "email_render_failed"
  | "already_delivered"
  | "delivery_claim_active"
  | "provider_attempt_started"
  | "provider_outcome_ambiguous"
  | "delivery_state_unavailable"
  | "delivery_storage_unsupported"
  | "delivery_claim_contention"
  | "delivery_claim_storage_failed"
  | "delivery_state_update_failed"
  | "cancelled_before_provider_attempt"
  | "provider_not_invoked"
  | "provider_definitive_rejection";

export type DeliverEmailResult =
  | { kind: "delivered"; attempts: number }
  | {
      kind: "delivered_unrecorded";
      reason: EmailDeliveryReason;
      attempts: number;
    }
  | { kind: "ambiguous"; reason: EmailDeliveryReason; attempts: number }
  | { kind: "skipped"; reason: EmailDeliveryReason }
  | { kind: "disabled"; reason: EmailDeliveryReason }
  | { kind: "failed"; reason: EmailDeliveryReason; attempts: number };
