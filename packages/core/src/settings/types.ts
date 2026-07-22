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

export interface PluginSettings {
  llm: LlmSettings;
  arxiv: ArxivSettings;
  detailSelection: DetailSelectionSettings;
  output: OutputSettings;
  schedule: ScheduleSettings;
  advanced: AdvancedSettings;
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
