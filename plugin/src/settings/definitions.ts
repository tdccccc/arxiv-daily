import type { PluginSettings } from "@arxiv-daily/core";

/**
 * Flat declarative-setting keys for the Obsidian 1.13+ settings API.
 * Keys are dotted paths into the nested PluginSettings object; the
 * framework calls getControlValue(key) / setControlValue(key, value) on
 * the setting tab, which resolve through readSettingValue /
 * writeSettingValue below.
 */
export const SETTING_KEYS = {
  llm: {
    apiKey: "llm.apiKey",
    baseUrl: "llm.baseUrl",
    model: "llm.model",
    thinkingMode: "llm.thinkingMode",
    reasoningEffort: "llm.reasoningEffort",
  },
  arxiv: {
    categories: "arxiv.categories",
    topics: "arxiv.topics",
    timezone: "arxiv.timezone",
  },
  output: {
    dailyDir: "output.dailyDir",
    papersDir: "output.papersDir",
    linkStyle: "output.linkStyle",
    summaryLanguage: "output.summaryLanguage",
  },
  schedule: {
    enabled: "schedule.enabled",
    runAtLocal: "schedule.runAtLocal",
    runUntilLocal: "schedule.runUntilLocal",
    tickIntervalMin: "schedule.tickIntervalMin",
  },
  advanced: {
    requestDelayMs: "advanced.requestDelayMs",
    cacheExpiryDays: "advanced.cacheExpiryDays",
    sectionCharLimit: "advanced.sectionCharLimit",
    paperCharLimit: "advanced.paperCharLimit",
    logLevel: "advanced.logLevel",
  },
  email: {
    enabled: "email.enabled",
    mode: "email.mode",
    to: "email.to",
    fromEmail: "email.fromEmail",
    fromName: "email.fromName",
    apiKey: "email.apiKey",
    hostedToken: "email.hostedToken",
    hostedBaseUrl: "email.hostedBaseUrl",
  },
} as const;

/** All flat keys registered above, for structural tests. */
export function allSettingKeys(): string[] {
  const out: string[] = [];
  for (const section of Object.values(SETTING_KEYS)) {
    for (const key of Object.values(section)) out.push(key);
  }
  return out;
}

/** Resolve a dotted key against the nested settings object. */
export function readSettingValue(
  settings: PluginSettings,
  key: string,
): unknown {
  const parts = key.split(".");
  let value: unknown = settings;
  for (const part of parts) {
    if (value == null || typeof value !== "object") return undefined;
    value = (value as Record<string, unknown>)[part];
  }
  return value;
}

/** Write a dotted key into the nested settings object (in place). */
export function writeSettingValue(
  settings: PluginSettings,
  key: string,
  value: unknown,
): void {
  const parts = key.split(".");
  if (parts.length === 0) return;
  let target: Record<string, unknown> = settings as unknown as Record<string, unknown>;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const next = target[parts[i]];
    if (next == null || typeof next !== "object") return;
    target = next as Record<string, unknown>;
  }
  target[parts[parts.length - 1]] = value;
}
