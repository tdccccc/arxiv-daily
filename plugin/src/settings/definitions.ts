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
  detailSelection: {
    profile: "detailSelection.profile",
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

import type { Setting, SettingDefinitionItem } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { arxivCategories } from "@arxiv-daily/core";
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
  renderApiKeyRow?: (setting: Setting) => void;
  renderModelRow?: (setting: Setting) => void;
  renderSetupGuideRow?: (setting: Setting) => void;
  renderCategoryRow?: (setting: Setting, index: number) => void;
  renderQuickStartRow?: (setting: Setting) => void;
  renderTopicRow?: (setting: Setting, index: number) => void;
  renderTimezoneRow?: (setting: Setting) => void;
  addCategory?: () => void;
  deleteCategory?: (index: number) => void;
  reorderCategories?: (oldIndex: number, newIndex: number) => void;
  addTopic?: () => void;
  reorderTopics?: (oldIndex: number, newIndex: number) => void;
  renderScheduleEnabledRow?: (setting: Setting) => void;
  renderRunWindowRow?: (setting: Setting) => void;
  renderTickIntervalRow?: (setting: Setting) => void;
  renderEmailGuideRow?: (setting: Setting) => void;
  renderEmailModeRow?: (setting: Setting) => void;
  renderEmailApiKeyRow?: (setting: Setting) => void;
  renderHostedTokenRow?: (setting: Setting) => void;
  sendVerificationEmail?: () => void;
  sendTestEmail?: () => void;
}

/** Detail-notes profile options; mirrors display()'s conditional "custom" row. */
function detailNotesOptions(settings: PluginSettings): Record<string, string> {
  const options: Record<string, string> = {
    conservative: "Fewer",
    balanced: "Recommended",
    broad: "More",
  };
  if (settings.detailSelection.profile === "custom") {
    options.custom = "Custom (current values)";
  }
  return options;
}

/**
 * Declarative settings for Obsidian 1.13+. Complex rows (API-key sentinel,
 * model picker, onboarding guide, topic cards, email verify, run window)
 * use `action`/`render` callbacks; the rest are plain controls and lists.
 * `display()` remains the <1.13 fallback.
 */
export function buildSettingDefinitions(
  host: SettingDefinitionsHost,
): SettingDefinitionItem[] {
  const { plugin } = host;
  const categories = arxivCategories(plugin.settings.arxiv);
  const topics = plugin.settings.arxiv.topics;
  const hostedMode = plugin.settings.email.mode === "hosted";
  return [
    ...(host.renderSetupGuideRow
      ? [{
          name: "Getting started",
          render: (setting: Setting) => host.renderSetupGuideRow?.(setting),
        } satisfies SettingDefinitionItem]
      : []),
    ...(host.renderScheduleEnabledRow
      ? [{
          name: plugin.settings.schedule.enabled
            ? "Enable · Running"
            : "Enable · Paused",
          desc: "When on, daily reports run automatically on weekdays (weekends are skipped).",
          render: (setting: Setting) =>
            host.renderScheduleEnabledRow?.(setting),
        } satisfies SettingDefinitionItem]
      : []),
    {
      type: "group",
      heading: "AI model",
      items: [
        ...(host.renderApiKeyRow
          ? [{
              name: "API key",
              desc: "Saved only on this device. After saving, the key is hidden.",
              render: (setting: Setting) => host.renderApiKeyRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
        ...(host.renderModelRow
          ? [{
              name: "Model",
              desc: "Choose a model, or click Get models to load the list from your provider.",
              render: (setting: Setting) => host.renderModelRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
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
      type: "list",
      heading: "arXiv categories",
      emptyState: "No categories yet — use Add category to add one.",
      items: [
        {
          name: "",
          desc: "Which arXiv subject areas to watch. You can add several; the same paper is only kept once.",
        } satisfies SettingDefinitionItem,
        ...categories.map((category, index) => ({
          name: `Category ${index + 1}`,
          render: (setting: Setting) => host.renderCategoryRow?.(setting, index),
        })),
      ],
      addItem: {
        name: "Add category",
        action: () => void host.addCategory?.(),
      },
      onDelete: (index) => void host.deleteCategory?.(index),
      onReorder: (oldIndex, newIndex) =>
        void host.reorderCategories?.(oldIndex, newIndex),
    },
    ...(host.renderQuickStartRow
      ? [{
          name: "Quick start",
          desc: "Load a preset bundle of topics or add one manually.",
          render: (setting: Setting) => host.renderQuickStartRow?.(setting),
        } satisfies SettingDefinitionItem]
      : []),
    {
      type: "list",
      heading: "Research topics",
      emptyState:
        "No topics yet. Pick a template above or click Add topic to define what to track. Daily reports need at least one topic before AI runs.",
      items: [
        {
          name: "",
          desc: "Each topic becomes one section in the daily report.",
        } satisfies SettingDefinitionItem,
        ...topics.map((topic, index) => ({
          name: topic.name.trim() || "(unnamed)",
          render: (setting: Setting) => host.renderTopicRow?.(setting, index),
        })),
      ],
      addItem: {
        name: "Add topic",
        action: () => void host.addTopic?.(),
      },
      onReorder: (oldIndex, newIndex) =>
        void host.reorderTopics?.(oldIndex, newIndex),
    },
    {
      name: "Automatic detail notes",
      desc: "How often the plugin writes a longer note for a paper. Only topics with Detail report turned on are considered. Manual “summarize paper” is unchanged.",
      control: {
        type: "dropdown",
        key: SETTING_KEYS.detailSelection.profile,
        defaultValue: "balanced",
        options: detailNotesOptions(plugin.settings),
      },
    },
    ...(host.renderTimezoneRow
      ? [{
          name: "Timezone",
          desc: "Which timezone defines the current day for reports and schedules.",
          render: (setting: Setting) => host.renderTimezoneRow?.(setting),
        } satisfies SettingDefinitionItem]
      : []),
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
        ...(host.renderRunWindowRow
          ? [{
              name: "Run window",
              desc: "Local times when automatic runs may start (24-hour clock).",
              render: (setting: Setting) =>
                host.renderRunWindowRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
        ...(host.renderTickIntervalRow
          ? [{
              name: "Check every (minutes)",
              desc: "How often the plugin looks for a day that still needs a report. Default is 20 minutes.",
              render: (setting: Setting) =>
                host.renderTickIntervalRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
      ],
    },
    {
      type: "group",
      heading: "Email delivery",
      items: [
        ...(host.renderEmailGuideRow
          ? [{
              name: "",
              render: (setting: Setting) =>
                host.renderEmailGuideRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
        ...(host.renderEmailModeRow
          ? [{
              name: "How to send",
              desc: hostedMode
                ? "Official delivery (Beta) is a shared free service with a small daily limit. Prefer Send yourself if you need many messages or reliable high volume."
                : "Send yourself uses your own Resend account (no project quota). Official delivery (Beta) is a limited free option for light personal use.",
              render: (setting: Setting) =>
                host.renderEmailModeRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
        {
          name: "Your email",
          desc: hostedMode
            ? "Where verification and daily digests are sent."
            : "Where digests are delivered. With From empty, use the email on your Resend account.",
          control: {
            type: "text",
            key: SETTING_KEYS.email.to,
            placeholder: "you@example.com",
          },
        },
        ...(hostedMode
          ? [
              {
                name: "Send verification email",
                desc: "Sends a one-time link to confirm this address is yours.",
                action: () => void host.sendVerificationEmail?.(),
              } satisfies SettingDefinitionItem,
              ...(host.renderHostedTokenRow
                ? [{
                    name: "Verification code",
                    desc: "After you open the verification link, copy the long code shown on the web page (not the short code in the email link). Use the same email address as above.",
                    render: (setting: Setting) =>
                      host.renderHostedTokenRow?.(setting),
                  } satisfies SettingDefinitionItem]
                : []),
            ]
          : [
              ...(host.renderEmailApiKeyRow
                ? [{
                    name: "Resend API key",
                    desc: "From your Resend account. Saved only on this device; not shown again after you save.",
                    render: (setting: Setting) =>
                      host.renderEmailApiKeyRow?.(setting),
                  } satisfies SettingDefinitionItem]
                : []),
              {
                name: "From email",
                desc: "Optional. Leave blank for the simplest setup (mail may only go to your Resend account email). Use an address on a domain you verified in Resend to send more freely.",
                control: {
                  type: "text",
                  key: SETTING_KEYS.email.fromEmail,
                  placeholder: "Leave blank for simplest setup",
                },
              } satisfies SettingDefinitionItem,
              {
                name: "From name",
                desc: "Optional name shown as the sender. Default is \"arXiv Daily\".",
                control: {
                  type: "text",
                  key: SETTING_KEYS.email.fromName,
                  placeholder: "arXiv Daily",
                },
              } satisfies SettingDefinitionItem,
            ]),
        {
          name: "Send test email",
          desc: hostedMode
            ? "Sends a sample digest now. Needs your email and verification code. Tests count toward the daily limit."
            : "Sends a sample digest now. Needs your email and Resend API key.",
          action: () => void host.sendTestEmail?.(),
        },
        {
          name: "Daily auto-send",
          desc: hostedMode
            ? "When on, a digest is emailed after each successful daily report. Official delivery may stop for the day if the shared limit is reached; report generation still continues."
            : "When on, a digest is emailed after each successful daily report. Email problems do not stop report generation.",
          control: {
            type: "toggle",
            key: SETTING_KEYS.email.enabled,
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
