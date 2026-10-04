import type { PluginSettings } from "@arxiv-daily/core";

import {
  SETTINGS_KEYS as SETTING_KEYS, getBusinessSettingsSections, businessSettingsContext,
  dailyAutoSendDescription, type BusinessSetting, type BusinessSettingId,
} from "@arxiv-daily/core";
export { SETTINGS_KEYS as SETTING_KEYS } from "@arxiv-daily/core";

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

import type { Setting, SettingDefinition, SettingDefinitionItem } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { arxivCategories } from "@arxiv-daily/core";
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

/** Compatibility export used by the legacy host renderer. */
export const dailyAutoSendDesc = dailyAutoSendDescription;

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
  const context = businessSettingsContext(plugin.settings, host.automaticEmailSupported ?? true);
  const renderers: Partial<Record<BusinessSettingId, (setting: Setting) => void>> = {
    apiBaseUrl: host.renderLlmBaseUrlRow?.bind(host),
    apiKey: host.renderApiKeyRow?.bind(host),
    model: host.renderModelRow?.bind(host),
    reasoningEffort: host.renderReasoningEffortRow?.bind(host),
    timezone: host.renderTimezoneRow?.bind(host),
    scheduleEnabled: host.renderScheduleEnabledRow?.bind(host),
    runWindow: host.renderRunWindowRow?.bind(host),
    tickInterval: host.renderTickIntervalRow?.bind(host),
    maxDailyPapers: host.renderDailyPaperLimitRow?.bind(host),
    library: host.renderLibraryConnectionRow?.bind(host),
    embeddingMode: host.renderEmbeddingModeRow?.bind(host),
    embeddingBaseUrl: host.renderEmbeddingBaseUrlRow?.bind(host),
    embeddingApiKey: host.renderEmbeddingApiKeyRow?.bind(host),
    embeddingModel: host.renderEmbeddingModelRow?.bind(host),
    embeddingDimension: host.renderEmbeddingDimensionRow?.bind(host),
    sidecarEnabled: host.renderPdfParserSidecarEnabledRow?.bind(host),
    sidecarCapabilitiesUrl: host.renderPdfParserSidecarCapabilitiesUrlRow?.bind(host),
    sidecarParseUrl: host.renderPdfParserSidecarParseUrlRow?.bind(host),
    emailMode: host.renderEmailModeRow?.bind(host),
    emailTo: host.renderEmailToRow?.bind(host),
    hostedToken: host.renderHostedTokenRow?.bind(host),
    emailApiKey: host.renderEmailApiKeyRow?.bind(host),
    dailyDir: host.renderOutputDirectoryRow ? setting => host.renderOutputDirectoryRow!(setting, "dailyDir") : undefined,
    papersDir: host.renderOutputDirectoryRow ? setting => host.renderOutputDirectoryRow!(setting, "papersDir") : undefined,
    fromEmail: host.renderEmailSenderRow ? setting => host.renderEmailSenderRow!(setting, "fromEmail") : undefined,
    fromName: host.renderEmailSenderRow ? setting => host.renderEmailSenderRow!(setting, "fromName") : undefined,
  };
  const declarative = new Set<BusinessSettingId>(['detailProfile','linkStyle','summaryLanguage','emailEnabled','logLevel']);
  function definition(field: BusinessSetting): SettingDefinition | undefined {
    // Title/abstract indexing has no structured-PDF consumer (ADR 0013).
    if (field.id.startsWith('sidecar')) return;
    const base = { name: field.name, desc: field.description };
    const render = renderers[field.id];
    if (render) return { ...base, render };
    if (!declarative.has(field.id)) return;
    if (field.control === 'toggle') return { ...base, control: { type: 'toggle', key: field.key! } };
    return { ...base, control: { type: 'dropdown', key: field.key!, options: field.options!,
      ...(typeof field.defaultValue === 'string' ? { defaultValue: field.defaultValue } : {}) } };
  }
  const business: SettingDefinitionItem[] = [];
  for (const section of getBusinessSettingsSections(context)) {
    if (section.type === 'field') {
      const item = definition(section.field); if (item) business.push(item);
    } else if (section.type === 'list') {
      if (section.id === 'categories') business.push({ type: 'list', heading: section.heading, cls: settingsSectionClass('arxiv'),
        emptyState: section.emptyState, items: categories.map((_category,index)=>({name:String(index+1),render:(setting:Setting)=>host.renderCategoryRow?.(setting,index)})),
        addItem:{name:section.addItemName,action:()=>void host.addCategory?.()},
        ...(categories.length>1 ? {onDelete:(index:number)=>void host.deleteCategory?.(index)} : {}) });
      else business.push({ type: 'list', heading: section.heading, cls: settingsSectionClass('topics'),
        emptyState: section.emptyState, items: topics.map((topic,index)=>({name:topic.name.trim()||'(unnamed)',render:(setting:Setting)=>host.renderTopicRow?.(setting,index)})),
        addItem:{name:section.addItemName,action:()=>void host.addTopic?.()} });
    } else {
      if (section.id === 'library' && !host.renderLibraryConnectionRow) continue;
      const items = section.items.map(definition).filter((item):item is SettingDefinition=>item!==undefined);
      if (section.id === 'library') {
        if (host.renderLibraryDirectionsRow && plugin.getLibraryConnectionStatus().kind !== 'disconnected') {
          const connectionIndex = section.items.findIndex(field => field.id === 'library');
          items.splice(connectionIndex + 1, 0, { name: 'Topics from library', desc: libraryDirectionsRowDesc(plugin), render: setting => host.renderLibraryDirectionsRow?.(setting) });
        }
        if (host.renderLibraryGuideRow) items.unshift({ name: '', render: setting => host.renderLibraryGuideRow?.(setting) });
      }
      if (section.id === 'email' && host.renderEmailGuideRow) items.unshift({name:'',render:setting=>host.renderEmailGuideRow?.(setting)});
      business.push({type:'group',heading:section.heading,...(section.id==='llm'?{cls:settingsSectionClass('llm')}:section.id==='output'?{cls:'arxiv-daily-settings__section-schedule'}:{}),items});
    }
  }
  return [
    ...(host.renderSetupGuideRow && host.showSetupGuide ? [{name:'',render:(setting:Setting)=>host.renderSetupGuideRow?.(setting)}] : []),
    ...business,
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
