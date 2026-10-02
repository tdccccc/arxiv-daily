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
    maxDailyPapers: "output.maxDailyPapers",
    linkStyle: "output.linkStyle",
    summaryLanguage: "output.summaryLanguage",
  },
  schedule: {
    enabled: "schedule.enabled",
    runAtLocal: "schedule.runAtLocal",
    runUntilLocal: "schedule.runUntilLocal",
    tickIntervalMin: "schedule.tickIntervalMin",
  },
  embedding: {
    mode: "embedding.mode",
    baseUrl: "embedding.baseUrl",
    apiKey: "embedding.apiKey",
    model: "embedding.model",
    dimension: "embedding.dimension",
    initialChoiceDone: "embedding.initialChoiceDone",
  },
  pdfParserSidecar: {
    enabled: "pdfParserSidecar.enabled",
    capabilitiesUrl: "pdfParserSidecar.capabilitiesUrl",
    parseUrl: "pdfParserSidecar.parseUrl",
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
  // Reject the whole path before resolving any property, including accessors.
  const last = parts.pop();
  if (last === undefined || last === "__proto__" || last === "constructor" || last === "prototype") return;
  if (parts.some((part) => part === "__proto__" || part === "constructor" || part === "prototype")) return;
  let target: Record<string, unknown> = settings as unknown as Record<string, unknown>;
  for (const part of parts) {
    if (!Object.hasOwn(target, part)) return;
    const next = target[part];
    if (next == null || typeof next !== "object") return;
    target = next as Record<string, unknown>;
  }
  // Ordinary new leaf keys remain supported; inherited values/setters do not.
  if (!Object.hasOwn(target, last) && last in target) return;
  target[last] = value;
}

import type { Setting, SettingDefinitionItem } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE, arxivCategories } from "@arxiv-daily/core";
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
  renderLlmBaseUrlRow?: (setting: Setting) => void;
  renderApiKeyRow?: (setting: Setting) => void;
  renderModelRow?: (setting: Setting) => void;
  renderReasoningEffortRow?: (setting: Setting) => void;
  renderLibraryConnectionRow?: (setting: Setting) => void;
  renderLibraryDirectionsRow?: (setting: Setting) => void;
  renderLibraryGuideRow?: (setting: Setting) => void;
  renderSetupGuideRow?: (setting: Setting) => void;
  showSetupGuide?: boolean;
  /** False when this host refuses every automatic email send. */
  automaticEmailSupported?: boolean;
  renderCategoryRow?: (setting: Setting, index: number) => void;
  renderTopicRow?: (setting: Setting, index: number) => void;
  renderTimezoneRow?: (setting: Setting) => void;
  renderOutputDirectoryRow?: (setting: Setting, key: "dailyDir" | "papersDir") => void;
  renderEmailSenderRow?: (setting: Setting, key: "fromEmail" | "fromName") => void;
  addCategory?: () => void;
  deleteCategory?: (index: number) => void;
  addTopic?: () => void;
  renderScheduleEnabledRow?: (setting: Setting) => void;
  renderRunWindowRow?: (setting: Setting) => void;
  renderTickIntervalRow?: (setting: Setting) => void;
  renderDailyPaperLimitRow?: (setting: Setting) => void;
  renderEmailGuideRow?: (setting: Setting) => void;
  renderEmailModeRow?: (setting: Setting) => void;
  renderEmailToRow?: (setting: Setting) => void;
  renderEmailApiKeyRow?: (setting: Setting) => void;
  renderHostedTokenRow?: (setting: Setting) => void;
  renderEmbeddingModeRow?: (setting: Setting) => void;
  renderEmbeddingBaseUrlRow?: (setting: Setting) => void;
  renderEmbeddingApiKeyRow?: (setting: Setting) => void;
  renderEmbeddingModelRow?: (setting: Setting) => void;
  renderEmbeddingDimensionRow?: (setting: Setting) => void;
  renderPdfParserSidecarEnabledRow?: (setting: Setting) => void;
  renderPdfParserSidecarCapabilitiesUrlRow?: (setting: Setting) => void;
  renderPdfParserSidecarParseUrlRow?: (setting: Setting) => void;
}

export type SettingsSection =
  | "llm"
  | "library"
  | "arxiv"
  | "topics"
  | "schedule"
  | "email"
  | "advanced";

/**
 * Class marking a settings section on both render paths: the declarative
 * group/list element on 1.13+, the section heading row in display(). The
 * setup guide scrolls to it.
 */
export function settingsSectionClass(section: SettingsSection): string {
  return `arxiv-daily-settings__section--${section}`;
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

/** Daily auto-send description, shared by both render paths. */
export function dailyAutoSendDesc(hostedMode: boolean, automaticSupported: boolean): string {
  const desc = hostedMode
    ? "When on, a digest is emailed after each successful daily report. Official delivery may stop for the day if the shared limit is reached; report generation still continues."
    : "When on, a digest is emailed after each successful daily report. Email problems do not stop report generation.";
  return automaticSupported ? desc : `${AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE} ${desc}`;
}

/**
 * Research directions row description, shared by both render paths. The
 * saved count comes from the topics already loaded in settings.
 */
export function libraryDirectionsRowDesc(plugin: ArxivDailyPlugin): string {
  const base = "Review topic suggestions from indexed paper titles and abstracts. Only directions added to Research topics steer daily reports.";
  const index = plugin.libraryIndexStatus.snapshot();
  if (index.activity) return `${base} Preparation is running; review will be available when it finishes.`;
  if (index.preparationError) return `${base} ${index.preparationError} Use Retry preparation above before reviewing.`;
  if (!index.lastRun?.papers) return `${base} Finish preparing the library first. Use Retry preparation above if needed.`;
  const count = plugin.settings.arxiv.topics.reduce((total, topic) => total + topic.directions.length, 0);
  return count > 0 ? `${base} ${count} saved direction${count === 1 ? "" : "s"}.` : base;
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
    ...(host.renderSetupGuideRow && host.showSetupGuide
      ? [{
          // The card carries its own "Getting started" title; a row name
          // would repeat it and squeeze the card into the control column.
          name: "",
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
      heading: "LLM",
      cls: settingsSectionClass("llm"),
      items: [
        ...(host.renderLlmBaseUrlRow
          ? [{
              name: "API base URL",
              desc: "Where chat requests are sent.",
              render: (setting: Setting) => host.renderLlmBaseUrlRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
        ...(host.renderApiKeyRow
          ? [{
              name: "API key",
              desc: "Saved only on this device.",
              render: (setting: Setting) => host.renderApiKeyRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
        ...(host.renderModelRow
          ? [{
              name: "Model",
              desc: "Choose a model or load the list from your provider.",
              render: (setting: Setting) => host.renderModelRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
        ...(host.renderReasoningEffortRow
          ? [{
              name: "Reasoning effort",
              desc: "Higher levels may be slower and cost more.",
              render: (setting: Setting) =>
                host.renderReasoningEffortRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
      ],
    },
    {
      type: "list",
      heading: "arXiv categories",
      cls: settingsSectionClass("arxiv"),
      emptyState: "No categories yet — use Add category to add one.",
      items: categories.map((_category, index) => ({
        name: String(index + 1),
        render: (setting: Setting) => host.renderCategoryRow?.(setting, index),
      })),
      addItem: {
        name: "Add category",
        action: () => void host.addCategory?.(),
      },
      // The last category cannot be removed, so it gets no delete button.
      ...(categories.length > 1
        ? { onDelete: (index: number) => void host.deleteCategory?.(index) }
        : {}),
    },
    {
      type: "list",
      heading: "Research topics",
      cls: settingsSectionClass("topics"),
      emptyState:
        "No topics yet. Generate from your library or add a topic.",
      items: topics.map((topic, index) => ({
        name: topic.name.trim() || "(unnamed)",
        render: (setting: Setting) => host.renderTopicRow?.(setting, index),
      })),
      addItem: {
        name: "Add topic",
        action: () => void host.addTopic?.(),
      },
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
      cls: "arxiv-daily-settings__section-schedule",
      items: [
        {
          name: "Daily paper limit",
          desc: "Maximum papers across all topics in each daily report. Default is 20.",
          render: (setting: Setting) => host.renderDailyPaperLimitRow?.(setting),
        },
        ...(host.renderOutputDirectoryRow
          ? [{
              name: "Daily reports folder",
              desc: "Folder in this vault for daily report notes (relative path).",
              render: (setting: Setting) =>
                host.renderOutputDirectoryRow?.(setting, "dailyDir"),
            } satisfies SettingDefinitionItem, {
              name: "Paper notes folder",
              desc: "Folder in this vault for per-paper notes (relative path).",
              render: (setting: Setting) =>
                host.renderOutputDirectoryRow?.(setting, "papersDir"),
            } satisfies SettingDefinitionItem]
          : []),
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
    ...(host.renderLibraryConnectionRow
      ? [{
          type: "group",
          heading: "Personal library",
          items: [
            ...(host.renderLibraryGuideRow
              ? [{
                  // The box carries its own title; a row name would repeat it
                  // and squeeze the box into the control column.
                  name: "",
                  render: (setting: Setting) => host.renderLibraryGuideRow?.(setting),
                } satisfies SettingDefinitionItem]
              : []),
            {
              name: "Library",
              desc: "Choose a folder of PDFs to prepare its search index automatically. Only suggestions you accept change daily reports.",
              render: (setting: Setting) =>
                host.renderLibraryConnectionRow?.(setting),
            },
            ...(host.renderLibraryDirectionsRow
              && plugin.getLibraryConnectionStatus().kind !== "disconnected"
              ? [{
                  name: "Topics from library",
                  desc: libraryDirectionsRowDesc(plugin),
                  render: (setting: Setting) =>
                    host.renderLibraryDirectionsRow?.(setting),
                } satisfies SettingDefinitionItem]
              : []),
            ...(host.renderEmbeddingModeRow
              ? [{
                  name: "Embedding",
                  desc: plugin.settings.embedding.mode === "remote"
                    ? "Remote sends titles and abstracts to an embeddings API. Switching modes rebuilds the index."
                    : "Local downloads its model once (about 130 MB) on the first index build, then embeds on this device. Switch to remote only if you have an embeddings API.",
                  render: (setting: Setting) => host.renderEmbeddingModeRow?.(setting),
                } satisfies SettingDefinitionItem]
              : []),
            ...(plugin.settings.embedding.mode === "remote" && host.renderEmbeddingBaseUrlRow
              ? [{
                  name: "Embedding API base URL",
                  desc: "OpenAI-compatible embeddings endpoint.",
                  render: (setting: Setting) => host.renderEmbeddingBaseUrlRow?.(setting),
                } satisfies SettingDefinitionItem]
              : []),
            ...(plugin.settings.embedding.mode === "remote" && host.renderEmbeddingApiKeyRow
              ? [{
                  name: "Embedding API key",
                  desc: "Saved only on this device.",
                  render: (setting: Setting) => host.renderEmbeddingApiKeyRow?.(setting),
                } satisfies SettingDefinitionItem]
              : []),
            ...(plugin.settings.embedding.mode === "remote" && host.renderEmbeddingModelRow
              ? [{
                  name: "Embedding model",
                  desc: "Model name sent to the endpoint.",
                  render: (setting: Setting) => host.renderEmbeddingModelRow?.(setting),
                } satisfies SettingDefinitionItem]
              : []),
            ...(plugin.settings.embedding.mode === "remote" && host.renderEmbeddingDimensionRow
              ? [{
                  name: "Embedding dimension",
                  desc: "Vector width of the remote model. Must match the model.",
                  render: (setting: Setting) => host.renderEmbeddingDimensionRow?.(setting),
                } satisfies SettingDefinitionItem]
              : []),
            // The three PDF parser sidecar rows ("Better PDF parser" and its two
            // loopback URLs) are deliberately not listed here.
            //
            // Indexing covers each paper's title and abstract (ADR 0013), which
            // needs plain text from the leading pages, so the sidecar's
            // structured output has no consumer and switching it on would change
            // nothing a user could observe. A toggle promising a better parser
            // that silently does nothing is worse than no toggle.
            //
            // The renderers, the host wiring, the sidecar client and the stored
            // `pdfParserSidecar` settings all remain, so restoring the rows is a
            // matter of re-adding these entries. Whether to retire the sidecar
            // outright is an open question tied to the conclusion-section
            // increment in ADR 0013 §2, which would want section structure back.
          ],
        } satisfies SettingDefinitionItem]
      : []),
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
        ...(host.renderEmailToRow
          ? [{
              name: "Your email",
              desc: hostedMode
                ? "Where verification and daily digests are sent."
                : "Where digests are delivered. With From empty, use the email on your Resend account.",
              render: (setting: Setting) => host.renderEmailToRow?.(setting),
            } satisfies SettingDefinitionItem]
          : []),
        ...(hostedMode
          ? [
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
              ...(host.renderEmailSenderRow
                ? [{
                    name: "From email",
                    desc: "Optional. Leave blank for the simplest setup (mail may only go to your Resend account email). Use an address on a domain you verified in Resend to send more freely.",
                    render: (setting: Setting) =>
                      host.renderEmailSenderRow?.(setting, "fromEmail"),
                  } satisfies SettingDefinitionItem, {
                    name: "From name",
                    desc: "Optional name shown as the sender. Default is \"arXiv Daily\".",
                    render: (setting: Setting) =>
                      host.renderEmailSenderRow?.(setting, "fromName"),
                  } satisfies SettingDefinitionItem]
                : []),
            ]),
        {
          name: "Daily auto-send",
          desc: dailyAutoSendDesc(hostedMode, host.automaticEmailSupported ?? true),
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
      cls: "arxiv-daily-settings__section-advanced",
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
