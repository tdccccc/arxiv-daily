import type { DetailSelectionSettings } from "./detail-selection";

export interface LlmSettings {
  apiKey: string;
  provider: string;
  baseUrl: string;
  model: string;
  thinkingMode: boolean;
  reasoningEffort: string;
}

export interface Topic {
  id: string;
  name: string;
  tag: string;
  description: string;
  detail: boolean;
}

export interface ArxivSettings {
  category: string;
  categories: string[];
  topics: Topic[];
  timezone: string;
}

export interface OutputSettings {
  dailyDir: string;
  papersDir: string;
  linkStyle?: LinkStyle;
  summaryLanguage?: SummaryLanguage;
}

export type LinkStyle = "wikilink" | "relative";
export type SummaryLanguage = "zh" | "en";

export interface ScheduleSettings {
  enabled: boolean;
  runAtLocal: string;
  runUntilLocal: string;
  tickIntervalMin: number;
}

export interface AdvancedSettings {
  requestDelayMs: number;
  cacheExpiryDays: number;
  sectionCharLimit: number;
  paperCharLimit: number;
  /** Retained for settings compatibility; sequential daily summarization does not batch by this limit. */
  dailyCharLimit: number;
  logLevel: "debug" | "info" | "warn" | "error";
}

export type EmailDeliveryModeSetting = "self" | "hosted";

export interface EmailSettings {
  /**
   * When true, auto-send after pipeline `completed`.
   * Quick setup: leave false until test-send succeeds.
   */
  enabled: boolean;
  /**
   * 自己发送 (`self`, default) vs 官方代发 (`hosted`, Beta — not online until
   * OFFICIAL_DELIVERY_AVAILABLE).
   */
  mode: EmailDeliveryModeSetting;
  /** Personal recipient (required). */
  to: string;
  /**
   * Optional custom From. Empty uses Resend quick sender
   * (`onboarding@resend.dev`) — fine for personal inbox testing.
   */
  fromEmail: string;
  /** Optional display name; defaults to "arXiv Daily" when empty. */
  fromName?: string;
  /** Plugin-local secret storage; CLI prefers ARXIV_DAILY_RESEND_API_KEY. */
  apiKey?: string;
  /**
   * Device token after magic-link verification (Official delivery Beta).
   */
  hostedToken?: string;
  /**
   * Optional override for the project relay base URL.
   * Default: https://email.arxiv-daily.top
   */
  hostedBaseUrl?: string;
}

export interface PluginSettings {
  llm: LlmSettings;
  arxiv: ArxivSettings;
  detailSelection: DetailSelectionSettings;
  output: OutputSettings;
  schedule: ScheduleSettings;
  advanced: AdvancedSettings;
  email: EmailSettings;
}

export type RunStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed_transient"
  | "failed_permanent"
  | "skipped";

export interface RunStateEntry {
  status: RunStatus;
  lastAttempt: number;
  attempts: number;
  error?: string;
  papersWritten?: number;
}

export type RunState = Record<string, RunStateEntry>;
