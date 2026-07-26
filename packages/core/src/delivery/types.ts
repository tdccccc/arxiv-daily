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

export interface DeliveryRecord {
  date: string;
  recipient: string;
  channel: EmailDeliveryChannel;
  status: DeliveryStatus;
  updatedAt: string;
  attempts: number;
  providerMessageId?: string;
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

export type DeliverEmailResult =
  | { kind: "delivered"; providerMessageId?: string; attempts: number }
  | { kind: "skipped"; reason: string }
  | { kind: "disabled"; reason: string }
  | { kind: "failed"; reason: string; attempts: number };
