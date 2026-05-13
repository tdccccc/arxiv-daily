export interface LlmSettings {
  apiKey: string;
  provider: string;
  baseUrl: string;
  model: string;
  temperature: number;
  timeoutMs: number;
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
  topics: Topic[];
  timezone: string;
}

export interface OutputSettings {
  dailyDir: string;
  papersDir: string;
}

export interface ScheduleSettings {
  enabled: boolean;
  runAtLocal: string;
  tickIntervalMin: number;
  lookbackDays: number;
}

export interface AdvancedSettings {
  requestDelayMs: number;
  cacheExpiryDays: number;
  sectionCharLimit: number;
  paperCharLimit: number;
  dailyCharLimit: number;
  skipSections: string[];
  prioritySections: string[];
  logLevel: "debug" | "info" | "warn" | "error";
}

export interface PluginSettings {
  llm: LlmSettings;
  arxiv: ArxivSettings;
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
