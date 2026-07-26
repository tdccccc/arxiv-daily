import type { SummaryLanguage } from "../settings/types";

export const EMAIL_DELIVERY_CHANNEL = "email:resend" as const;
export type EmailDeliveryChannel = typeof EMAIL_DELIVERY_CHANNEL;

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
