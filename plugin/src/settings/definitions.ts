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
    const part = parts[i];
    if (part === undefined) return;
    const next = target[part];
    if (next == null || typeof next !== "object") return;
    target = next as Record<string, unknown>;
  }
  const last = parts[parts.length - 1];
  if (last === undefined) return;
  target[last] = value;
}

import type { SettingDefinitionItem } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { validateOutputDirectoryDraft } from "./tab";
import {
  ARXIV_DAILY_DOCS_URL,
  ARXIV_DAILY_REPO_URL,
  buildBugReportUrl,
  buildFeatureRequestUrl,
} from "../feedback";
import { ObsidianResourceOpener } from "../hosts/obsidian/resource-opener";

/** Minimal host surface buildSettingDefinitions needs (the setting tab fits). */
export interface SettingDefinitionsHost {
  plugin: ArxivDailyPlugin;
}

/**
 * Declarative settings for Obsidian 1.13+. Complex rows (API key sentinel,
 * model picker, onboarding guide, topics, email verify, run window) are
 * added in later tasks of P2b; this block covers the control-expressible
 * settings. `display()` remains the <1.13 fallback.
 */
export function buildSettingDefinitions(
  host: SettingDefinitionsHost,
): SettingDefinitionItem[] {
  const { plugin } = host;
  return [
    {
      name: "Enable",
      desc: "When on, daily reports run automatically on weekdays (weekends are skipped).",
      control: {
        type: "toggle",
        key: SETTING_KEYS.schedule.enabled,
      },
    },
    {
      type: "group",
      heading: "AI model",
      items: [
        {
          name: "API base URL",
          desc: "Where chat requests are sent. Default is DeepSeek; change only if you use another provider.",
          control: {
            type: "text",
            key: SETTING_KEYS.llm.baseUrl,
            placeholder: "https://api.deepseek.com/v1",
          },
        },
        {
          name: "Thinking mode",
          desc: "Let the model spend extra effort on harder questions when the provider supports it.",
          control: {
            type: "toggle",
            key: SETTING_KEYS.llm.thinkingMode,
          },
        },
        {
          name: "Reasoning effort",
          desc: "How hard the model tries when thinking mode is on. Higher may be slower and cost more.",
          control: {
            type: "dropdown",
            key: SETTING_KEYS.llm.reasoningEffort,
            defaultValue: "medium",
            options: {
              low: "low",
              medium: "medium",
              high: "high",
            },
          },
        },
      ],
    },
    {
      type: "group",
      heading: "Output & schedule",
      items: [
        {
          name: "Daily reports folder",
          desc: "Folder in this vault for daily report notes (relative path).",
          control: {
            type: "text",
            key: SETTING_KEYS.output.dailyDir,
            validate: (value) => {
              const validation = validateOutputDirectoryDraft(value);
              return validation.ok ? undefined : (validation.reason ?? "Invalid path.");
            },
          },
        },
        {
          name: "Paper notes folder",
          desc: "Folder in this vault for per-paper notes (relative path).",
          control: {
            type: "text",
            key: SETTING_KEYS.output.papersDir,
            validate: (value) => {
              const validation = validateOutputDirectoryDraft(
                value,
                plugin.settings.output.dailyDir,
              );
              return validation.ok ? undefined : (validation.reason ?? "Invalid path.");
            },
          },
        },
        {
          name: "Link style",
          desc: "How links between notes are written in daily reports.",
          control: {
            type: "dropdown",
            key: SETTING_KEYS.output.linkStyle,
            defaultValue: "wikilink",
            options: {
              wikilink: "Obsidian wikilink",
              relative: "Standard relative link",
            },
          },
        },
        {
          name: "Summary language",
          desc: "Language for daily reports and paper notes.",
          control: {
            type: "dropdown",
            key: SETTING_KEYS.output.summaryLanguage,
            defaultValue: "zh",
            options: {
              zh: "Chinese",
              en: "English",
            },
          },
        },
        {
          name: "Check every (minutes)",
          desc: "How often the scheduler checks for new papers while Obsidian is open.",
          control: {
            type: "text",
            key: SETTING_KEYS.schedule.tickIntervalMin,
            validate: (value) => {
              const parsed = Number(value);
              return Number.isFinite(parsed) && parsed >= 1
                ? undefined
                : "Enter a positive number of minutes.";
            },
          },
        },
      ],
    },
    {
      type: "group",
      heading: "Advanced",
      items: [
        {
          name: "Log level",
          desc: "How much detail appears in the developer console. Use debug only when troubleshooting; info is the default.",
          control: {
            type: "dropdown",
            key: SETTING_KEYS.advanced.logLevel,
            options: {
              debug: "Debug",
              info: "Info",
              warn: "Warn",
              error: "Error",
            },
          },
        },
      ],
    },
    {
      type: "group",
      heading: "Help & feedback",
      desc: "Documentation and GitHub issues. A short note is enough; do not paste API keys.",
      items: [
        {
          name: "Report a bug",
          desc: "Opens a blank GitHub issue with the plugin version. A short description is enough.",
          action: () => {
            void new ObsidianResourceOpener(plugin.app).openUrl(
              buildBugReportUrl(plugin.manifest.version),
            );
          },
        },
        {
          name: "Request a feature",
          desc: "Opens a blank GitHub issue. Write freely.",
          action: () => {
            void new ObsidianResourceOpener(plugin.app).openUrl(
              buildFeatureRequestUrl(),
            );
          },
        },
        {
          name: "Documentation",
          desc: "Getting started guide on GitHub.",
          action: () => {
            void new ObsidianResourceOpener(plugin.app).openUrl(
              ARXIV_DAILY_DOCS_URL,
            );
          },
        },
        {
          name: "Repository",
          desc: ARXIV_DAILY_REPO_URL,
          action: () => {
            void new ObsidianResourceOpener(plugin.app).openUrl(
              ARXIV_DAILY_REPO_URL,
            );
          },
        },
      ],
    },
  ];
}
