import type { PluginSettings, RunState, RunStateEntry } from "../settings/types";
import { formatArxivCategories } from "../settings/categories";
import { validateFilterConfig, validateLlmConfig } from "../settings/validation";
import { daysBefore, formatDate, isWeekendDate, todayInTz } from "../utils/time";

export interface DiagnosticsInput {
  settings: PluginSettings;
  runState: RunState;
  version?: string;
  now?: Date;
  recentLimit?: number;
  paperIndex?: PaperIndexDiagnostics;
}

export interface PaperIndexDiagnostics {
  path: string;
  exists: boolean;
  schemaVersion?: number;
  total?: number;
  statusCounts?: Record<string, number>;
  unsupportedSchemaVersion?: string;
  invalidStatuses?: string[];
  invalidPriorities?: string[];
  invalidSeenDates?: string[];
  missingPaperPaths?: string[];
  noteArxivIdMismatches?: string[];
  error?: string;
}

export function buildDiagnosticsReport(input: DiagnosticsInput): string {
  const settings = input.settings;
  const runState = input.runState;
  const now = input.now ?? new Date();
  const version = input.version?.trim() || "unknown";
  const recentLimit = input.recentLimit ?? 10;
  const llmValidation = validateLlmConfig(settings);
  const filterValidation = validateFilterConfig(settings);
  const dateContext = getDateContext(now, settings, runState);
  const recentEntries = Object.entries(runState)
    .sort((a, b) => (a[0] < b[0] ? 1 : -1))
    .slice(0, recentLimit);
  const failedDates = Object.entries(runState)
    .filter(([, entry]) =>
      entry.status === "failed_transient" || entry.status === "failed_permanent",
    )
    .map(([date]) => date)
    .sort((a, b) => (a < b ? 1 : -1));

  const lines: string[] = [
    "arXiv Daily Diagnostics",
    `generatedAt: ${now.toISOString()}`,
    `pluginVersion: ${version}`,
    "",
    "validation:",
    `  llm: ${llmValidation.ok ? "ok" : "invalid"}`,
    ...formatReasons("  llmReasons", llmValidation.reasons),
    `  filter: ${filterValidation.ok ? "ok" : "invalid"}`,
    ...formatReasons("  filterReasons", filterValidation.reasons),
    "",
    "llm:",
    `  provider: ${settings.llm.provider}`,
    `  baseUrl: ${settings.llm.baseUrl}`,
    `  model: ${settings.llm.model}`,
    `  apiKeySet: ${settings.llm.apiKey.trim() ? "yes" : "no"}`,
    `  temperature: ${settings.llm.temperature}`,
    `  timeoutMs: ${settings.llm.timeoutMs}`,
    `  thinkingMode: ${settings.llm.thinkingMode}`,
    `  reasoningEffort: ${settings.llm.reasoningEffort}`,
    "",
    "arxiv:",
    `  category: ${settings.arxiv.category}`,
    `  categories: ${formatArxivCategories(settings.arxiv)}`,
    `  timezone: ${settings.arxiv.timezone}`,
    `  localDate: ${dateContext.localDate}`,
    `  localWeekday: ${dateContext.localWeekday}`,
    ...(dateContext.error ? [`  dateError: ${dateContext.error}`] : []),
    "",
    "topics:",
    ...formatTopics(settings),
    "",
    "output:",
    `  dailyDir: ${settings.output.dailyDir}`,
    `  papersDir: ${settings.output.papersDir}`,
    `  linkStyle: ${settings.output.linkStyle ?? "wikilink"}`,
    "",
    "schedule:",
    `  enabled: ${settings.schedule.enabled}`,
    `  runAtLocal: ${settings.schedule.runAtLocal}`,
    `  tickIntervalMin: ${settings.schedule.tickIntervalMin}`,
    `  lookbackDays: ${settings.schedule.lookbackDays}`,
    "",
    "advanced:",
    `  requestDelayMs: ${settings.advanced.requestDelayMs}`,
    `  cacheExpiryDays: ${settings.advanced.cacheExpiryDays}`,
    `  sectionCharLimit: ${settings.advanced.sectionCharLimit}`,
    `  paperCharLimit: ${settings.advanced.paperCharLimit}`,
    `  dailyCharLimit: ${settings.advanced.dailyCharLimit}`,
    `  logLevel: ${settings.advanced.logLevel}`,
    "",
    "lookbackDates:",
    ...dateContext.lookbackLines,
    "",
    "failedDates:",
    ...(failedDates.length ? failedDates.map((d) => `  - ${d}`) : ["  - none"]),
    "",
    `recentRunState(limit=${recentLimit}):`,
    ...formatRunState(recentEntries),
    "",
    "paperIndex:",
    ...formatPaperIndex(input.paperIndex),
  ];

  return `${lines.join("\n")}\n`;
}

function formatReasons(label: string, reasons: string[]): string[] {
  if (reasons.length === 0) return [`${label}: none`];
  return [`${label}:`, ...reasons.map((reason) => `    - ${reason}`)];
}

function formatTopics(settings: PluginSettings): string[] {
  if (settings.arxiv.topics.length === 0) return ["  - none"];
  return settings.arxiv.topics.map((topic, index) => {
    const name = topic.name.trim() || "(empty)";
    const tag = topic.tag.trim() || "(empty)";
    return (
      `  - ${index + 1}. name="${name}", tag="${tag}", ` +
      `detail=${topic.detail}, hasDescription=${topic.description.trim() ? "yes" : "no"}`
    );
  });
}

function formatRunState(entries: Array<[string, RunStateEntry]>): string[] {
  if (entries.length === 0) return ["  - none"];
  return entries.map(([date, entry]) => {
    const parts = [
      `${date}: ${entry.status}`,
      `attempts=${entry.attempts}`,
      `lastAttempt=${formatTimestamp(entry.lastAttempt)}`,
    ];
    if (entry.papersWritten != null) parts.push(`papers=${entry.papersWritten}`);
    if (entry.error) parts.push(`error=${entry.error.slice(0, 160)}`);
    return `  - ${parts.join(", ")}`;
  });
}

function getDateContext(now: Date, settings: PluginSettings, runState: RunState): {
  localDate: string;
  localWeekday: string;
  lookbackLines: string[];
  error?: string;
} {
  try {
    const today = todayInTz(now, settings.arxiv.timezone);
    const localDate = formatDate(today);
    const localWeekday = new Intl.DateTimeFormat("en-US", {
      timeZone: settings.arxiv.timezone,
      weekday: "long",
    }).format(now);
    const lookbackLines = Array.from(
      { length: settings.schedule.lookbackDays },
      (_, i) => {
        const dateObj = daysBefore(today, i);
        const date = formatDate(dateObj);
        const entry = runState[date];
        const parts = [
          `${date}: state=${entry?.status ?? "pending"}`,
          `weekend=${isWeekendDate(dateObj) ? "yes" : "no"}`,
        ];
        if (entry) parts.push(`attempts=${entry.attempts}`);
        return `  - ${parts.join(", ")}`;
      },
    );
    return { localDate, localWeekday, lookbackLines };
  } catch (e) {
    return {
      localDate: "unavailable",
      localWeekday: "unavailable",
      lookbackLines: ["  - unavailable"],
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

function formatTimestamp(timestamp: number): string {
  if (!timestamp) return "never";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return String(timestamp);
  return date.toISOString();
}

function formatList(items: string[]): string {
  return items.length ? items.join(", ") : "none";
}

function formatPaperIndex(diag: PaperIndexDiagnostics | undefined): string[] {
  if (!diag) return ["  unavailable"];
  const lines = [
    `  path: ${diag.path}`,
    `  exists: ${diag.exists ? "yes" : "no"}`,
  ];
  if (diag.error) {
    lines.push(`  error: ${diag.error}`);
    return lines;
  }
  if (diag.schemaVersion != null) lines.push(`  schemaVersion: ${diag.schemaVersion}`);
  if (diag.unsupportedSchemaVersion) {
    lines.push(`  unsupportedSchemaVersion: ${diag.unsupportedSchemaVersion}`);
  }
  if (diag.total != null) lines.push(`  total: ${diag.total}`);
  if (diag.statusCounts) {
    const counts = Object.entries(diag.statusCounts)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([status, count]) => `${status}=${count}`)
      .join(", ");
    lines.push(`  statusCounts: ${counts || "none"}`);
  }
  if (diag.invalidStatuses?.length) {
    lines.push("  invalidStatuses:", ...diag.invalidStatuses.map((s) => `    - ${s}`));
  } else {
    lines.push("  invalidStatuses: none");
  }
  if (diag.invalidPriorities?.length) {
    lines.push("  invalidPriorities:", ...diag.invalidPriorities.map((s) => `    - ${s}`));
  } else {
    lines.push("  invalidPriorities: none");
  }
  if (diag.invalidSeenDates?.length) {
    lines.push("  invalidSeenDates:", ...diag.invalidSeenDates.map((s) => `    - ${s}`));
  } else {
    lines.push("  invalidSeenDates: none");
  }
  if (diag.missingPaperPaths?.length) {
    lines.push("  missingPaperPaths:", ...diag.missingPaperPaths.map((p) => `    - ${p}`));
  } else {
    lines.push("  missingPaperPaths: none");
  }
  if (diag.noteArxivIdMismatches?.length) {
    lines.push(
      "  noteArxivIdMismatches:",
      ...diag.noteArxivIdMismatches.map((p) => `    - ${p}`),
    );
  } else {
    lines.push("  noteArxivIdMismatches: none");
  }
  return lines;
}
