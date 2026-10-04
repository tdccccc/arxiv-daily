import type { PluginSettings } from './types';
import { DEFAULT_SETTINGS } from './defaults';
import { AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE } from '../delivery/delivery-state';

/** Stable business metadata. Host-specific storage, controls and side effects stay in adapters. */
export const SETTINGS_KEYS = {
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


export const SETTING_KEYS = SETTINGS_KEYS;
export const TIMEZONE_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "Asia/Shanghai", label: "Shanghai (UTC+8)" },
  { value: "Asia/Tokyo", label: "Tokyo (UTC+9)" },
  { value: "US/Eastern", label: "US East (UTC-5)" },
  { value: "US/Pacific", label: "US West (UTC-8)" },
  { value: "Europe/London", label: "London (UTC+0)" },
  { value: "Europe/Berlin", label: "Berlin (UTC+1)" },
  { value: "Europe/Moscow", label: "Moscow (UTC+3)" },
  { value: "Australia/Sydney", label: "Sydney (UTC+10)" },
  { value: "UTC", label: "UTC" },
];

export type BusinessSettingId = "scheduleEnabled" | "apiBaseUrl" | "apiKey" | "model" | "reasoningEffort" | "categories" | "topics" | "detailProfile" | "timezone" | "maxDailyPapers" | "dailyDir" | "papersDir" | "linkStyle" | "summaryLanguage" | "runWindow" | "tickInterval" | "library" | "embeddingMode" | "embeddingBaseUrl" | "embeddingApiKey" | "embeddingModel" | "embeddingDimension" | "sidecarEnabled" | "sidecarCapabilitiesUrl" | "sidecarParseUrl" | "emailMode" | "emailTo" | "hostedToken" | "emailApiKey" | "fromEmail" | "fromName" | "emailEnabled" | "logLevel";
export interface BusinessSettingsContext {
  emailMode: PluginSettings['email']['mode'];
  embeddingMode: PluginSettings['embedding']['mode'];
  sidecarEnabled: boolean;
  detailProfile: PluginSettings['detailSelection']['profile'];
  scheduleEnabled: boolean;
  reasoningEffort?: string;
  automaticEmailSupported: boolean;
}
export type BusinessSettingControl = 'text' | 'secret' | 'model' | 'dropdown' | 'toggle' | 'list' | 'timezone' | 'time-window' | 'library' | 'number';
interface SettingMetadata {
  key?: string;
  name: string;
  description: string;
  control: BusinessSettingControl;
  options?: Record<string,string>;
  defaultValue?: string | number | boolean;
}
export interface BusinessSetting extends SettingMetadata { id: BusinessSettingId; visible: boolean }
export type BusinessSettingsSection =
  | { type: 'field'; field: BusinessSetting }
  | { type: 'list'; id: 'categories' | 'topics'; heading: string; field: BusinessSetting; emptyState: string; addItemName: string }
  | { type: 'group'; id: 'llm' | 'output' | 'library' | 'email' | 'advanced'; heading: string; items: BusinessSetting[] };
const fields: Record<BusinessSettingId, SettingMetadata> = {
  "scheduleEnabled": {
    "key": "schedule.enabled",
    "name": "Enable · Paused",
    "description": "When on, daily reports run automatically on weekdays (weekends are skipped).",
    "control": "toggle"
  },
  "apiBaseUrl": {
    "key": "llm.baseUrl",
    "name": "API base URL",
    "description": "Where chat requests are sent.",
    "control": "text"
  },
  "apiKey": {
    "key": "llm.apiKey",
    "name": "API key",
    "description": "Saved only on this device.",
    "control": "secret"
  },
  "model": {
    "key": "llm.model",
    "name": "Model",
    "description": "Choose a model or load the list from your provider.",
    "control": "model"
  },
  "reasoningEffort": {
    "key": "llm.reasoningEffort",
    "name": "Reasoning effort",
    "description": "Higher levels may be slower and cost more.",
    "control": "dropdown",
    "options": {
      "none": "None",
      "low": "Low",
      "medium": "Medium",
      "high": "High"
    }
  },
  "categories": {
    "key": "arxiv.categories",
    "name": "arXiv categories",
    "description": "",
    "control": "list"
  },
  "topics": {
    "key": "arxiv.topics",
    "name": "Research topics",
    "description": "",
    "control": "list"
  },
  "detailProfile": {
    "key": "detailSelection.profile",
    "name": "Automatic detail notes",
    "description": "How often the plugin writes a longer note for a paper. Only topics with Detail report turned on are considered. Manual “summarize paper” is unchanged.",
    "control": "dropdown",
    "options": {
      "conservative": "Fewer",
      "balanced": "Recommended",
      "broad": "More"
    },
    "defaultValue": "balanced"
  },
  "timezone": {
    "key": "arxiv.timezone",
    "name": "Timezone",
    "description": "Which timezone defines the current day for reports and schedules.",
    "control": "timezone"
  },
  "maxDailyPapers": {
    "key": "output.maxDailyPapers",
    "name": "Daily paper limit",
    "description": "Maximum papers across all topics in each daily report. Default is 20.",
    "control": "number",
    "defaultValue": 20
  },
  "dailyDir": {
    "key": "output.dailyDir",
    "name": "Daily reports folder",
    "description": "Folder in this vault for daily report notes (relative path).",
    "control": "text"
  },
  "papersDir": {
    "key": "output.papersDir",
    "name": "Paper notes folder",
    "description": "Folder in this vault for per-paper notes (relative path).",
    "control": "text"
  },
  "linkStyle": {
    "key": "output.linkStyle",
    "name": "Link style",
    "description": "How links between notes are written in daily reports.",
    "control": "dropdown",
    "options": {
      "wikilink": "Obsidian wikilink",
      "relative": "Standard relative link"
    },
    "defaultValue": "wikilink"
  },
  "summaryLanguage": {
    "key": "output.summaryLanguage",
    "name": "Summary language",
    "description": "Language for daily reports and paper notes.",
    "control": "dropdown",
    "options": {
      "zh": "Chinese",
      "en": "English"
    },
    "defaultValue": "zh"
  },
  "runWindow": {
    "name": "Run window",
    "description": "Local times when automatic runs may start (24-hour clock).",
    "control": "time-window"
  },
  "tickInterval": {
    "key": "schedule.tickIntervalMin",
    "name": "Check every (minutes)",
    "description": "How often the plugin looks for a day that still needs a report. Default is 20 minutes.",
    "control": "text"
  },
  "library": {
    "name": "Library",
    "description": "Choose a folder of PDFs to prepare its search index automatically. Only suggestions you accept change daily reports.",
    "control": "library"
  },
  "embeddingMode": {
    "key": "embedding.mode",
    "name": "Embedding",
    "description": "",
    "control": "dropdown",
    "options": {
      "local": "Local (default, one-time model download)",
      "remote": "Remote (titles and abstracts leave this device)"
    }
  },
  "embeddingBaseUrl": {
    "key": "embedding.baseUrl",
    "name": "Embedding API base URL",
    "description": "OpenAI-compatible embeddings endpoint.",
    "control": "text"
  },
  "embeddingApiKey": {
    "key": "embedding.apiKey",
    "name": "Embedding API key",
    "description": "Saved only on this device.",
    "control": "secret"
  },
  "embeddingModel": {
    "key": "embedding.model",
    "name": "Embedding model",
    "description": "Model name sent to the endpoint.",
    "control": "text"
  },
  "embeddingDimension": {
    "key": "embedding.dimension",
    "name": "Embedding dimension",
    "description": "Vector width of the remote model. Must match the model.",
    "control": "number"
  },
  "sidecarEnabled": {
    "key": "pdfParserSidecar.enabled",
    "name": "Better PDF parser",
    "description": "Optional local sidecar. Off by default; PDFs stay on this device either way.",
    "control": "toggle"
  },
  "sidecarCapabilitiesUrl": {
    "key": "pdfParserSidecar.capabilitiesUrl",
    "name": "Sidecar capability URL",
    "description": "Local loopback endpoint that reports parser capabilities.",
    "control": "text"
  },
  "sidecarParseUrl": {
    "key": "pdfParserSidecar.parseUrl",
    "name": "Sidecar parse URL",
    "description": "Same-origin local loopback endpoint that accepts one PDF byte buffer.",
    "control": "text"
  },
  "emailMode": {
    "key": "email.mode",
    "name": "How to send",
    "description": "",
    "control": "dropdown",
    "options": {
      "self": "Send yourself",
      "hosted": "Official delivery (beta)"
    }
  },
  "emailTo": {
    "key": "email.to",
    "name": "Your email",
    "description": "",
    "control": "text"
  },
  "hostedToken": {
    "key": "email.hostedToken",
    "name": "Verification code",
    "description": "After you open the verification link, copy the long code shown on the web page (not the short code in the email link). Use the same email address as above.",
    "control": "secret"
  },
  "emailApiKey": {
    "key": "email.apiKey",
    "name": "Resend API key",
    "description": "From your Resend account. Saved only on this device; not shown again after you save.",
    "control": "secret"
  },
  "fromEmail": {
    "key": "email.fromEmail",
    "name": "From email",
    "description": "Optional. Leave blank for the simplest setup (mail may only go to your Resend account email). Use an address on a domain you verified in Resend to send more freely.",
    "control": "text"
  },
  "fromName": {
    "key": "email.fromName",
    "name": "From name",
    "description": "Optional name shown as the sender. Default is \"arXiv Daily\".",
    "control": "text"
  },
  "emailEnabled": {
    "key": "email.enabled",
    "name": "Daily auto-send",
    "description": "",
    "control": "toggle"
  },
  "logLevel": {
    "key": "advanced.logLevel",
    "name": "Log level",
    "description": "How much detail appears in the developer console. Use debug only when troubleshooting; info is the default.",
    "control": "dropdown",
    "options": {
      "debug": "Debug",
      "info": "Info",
      "warn": "Warn",
      "error": "Error"
    }
  }
};

export const REASONING_EFFORT_OPTIONS: Readonly<Record<string,string>> = Object.freeze({ ...fields.reasoningEffort.options });
export function businessSettingsContext(settings: PluginSettings, automaticEmailSupported = true): BusinessSettingsContext {
  return { emailMode: settings.email.mode, embeddingMode: settings.embedding.mode, sidecarEnabled: settings.pdfParserSidecar.enabled,
    detailProfile: settings.detailSelection.profile, scheduleEnabled: settings.schedule.enabled, reasoningEffort: settings.llm.reasoningEffort, automaticEmailSupported };
}
function contextWithDefaults(context: Partial<BusinessSettingsContext>): BusinessSettingsContext {
  return { ...businessSettingsContext(DEFAULT_SETTINGS), ...context };
}
export function isBusinessSettingVisible(id: BusinessSettingId, context: Partial<BusinessSettingsContext> = {}): boolean {
  const current = contextWithDefaults(context);
  if (['embeddingBaseUrl','embeddingApiKey','embeddingModel','embeddingDimension'].includes(id)) return current.embeddingMode === 'remote';
  if (['sidecarCapabilitiesUrl','sidecarParseUrl'].includes(id)) return current.sidecarEnabled;
  if (id === 'hostedToken') return current.emailMode === 'hosted';
  if (['emailApiKey','fromEmail','fromName'].includes(id)) return current.emailMode !== 'hosted';
  return true;
}
export function dailyAutoSendDescription(hostedMode: boolean, automaticSupported: boolean): string {
  const description = hostedMode
    ? 'When on, a digest is emailed after each successful daily report. Official delivery may stop for the day if the shared limit is reached; report generation still continues.'
    : 'When on, a digest is emailed after each successful daily report. Email problems do not stop report generation.';
  return automaticSupported ? description : `${AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE} ${description}`;
}
/** Each result is a fresh projection so adapters cannot mutate the shared option catalog. */
export function getBusinessSetting(id: BusinessSettingId, context: Partial<BusinessSettingsContext> = {}): BusinessSetting {
  const current = contextWithDefaults(context);
  const metadata = fields[id];
  const field: BusinessSetting = { ...metadata, id, visible: isBusinessSettingVisible(id, current), ...(metadata.options ? { options: { ...metadata.options } } : {}) };
  if (metadata.defaultValue !== undefined && metadata.key) { const [group,key] = metadata.key.split('.'); field.defaultValue = (DEFAULT_SETTINGS[group as keyof PluginSettings] as unknown as Record<string,string|number|boolean>)[key!]!; }
  if (id === 'scheduleEnabled') field.name = current.scheduleEnabled ? 'Enable · Running' : 'Enable · Paused';
  if (id === 'reasoningEffort' && typeof current.reasoningEffort === 'string' && !Object.hasOwn(field.options!, current.reasoningEffort)) Object.defineProperty(field.options!, current.reasoningEffort, { value: 'Custom (current values)', enumerable: true, configurable: true, writable: true });
  if (id === 'detailProfile' && current.detailProfile === 'custom') field.options!.custom = 'Custom (current values)';
  if (id === 'embeddingMode') field.description = current.embeddingMode === 'remote'
    ? 'Remote sends titles and abstracts to an embeddings API. Switching modes rebuilds the index.'
    : 'Local downloads its model once (about 130 MB) on the first index build, then embeds on this device. Switch to remote only if you have an embeddings API.';
  if (id === 'emailMode') field.description = current.emailMode === 'hosted'
    ? 'Official delivery (Beta) is a shared free service with a small daily limit. Prefer Send yourself if you need many messages or reliable high volume.'
    : 'Send yourself uses your own Resend account (no project quota). Official delivery (Beta) is a limited free option for light personal use.';
  if (id === 'emailTo') field.description = current.emailMode === 'hosted'
    ? 'Where verification and daily digests are sent.'
    : 'Where digests are delivered. With From empty, use the email on your Resend account.';
  if (id === 'emailEnabled') field.description = dailyAutoSendDescription(current.emailMode === 'hosted', current.automaticEmailSupported);
  return field;
}
export function getBusinessSettingsSections(context: Partial<BusinessSettingsContext> = {}, options: { includeHidden?: boolean } = {}): BusinessSettingsSection[] {
  const field = (id: BusinessSettingId): BusinessSettingsSection => ({ type: 'field', field: getBusinessSetting(id, context) });
  const group = (id: Extract<BusinessSettingsSection,{type:'group'}>['id'], heading: string, ids: BusinessSettingId[]): BusinessSettingsSection => ({type:'group',id,heading,items:ids.map(key=>getBusinessSetting(key,context)).filter(item=>options.includeHidden || item.visible)});
  return [
    field('scheduleEnabled'),
    group('llm','LLM',['apiBaseUrl','apiKey','model','reasoningEffort']),
    {type:'list',id:'categories',heading:'arXiv categories',field:getBusinessSetting('categories',context),emptyState:'No categories yet — use Add category to add one.',addItemName:'Add category'},
    {type:'list',id:'topics',heading:'Research topics',field:getBusinessSetting('topics',context),emptyState:'No topics yet. Generate from your library or add a topic.',addItemName:'Add topic'},
    field('detailProfile'), field('timezone'),
    group('output','Output & schedule',['maxDailyPapers','dailyDir','papersDir','linkStyle','summaryLanguage','runWindow','tickInterval']),
    group('library','Personal library',['library','embeddingMode','embeddingBaseUrl','embeddingApiKey','embeddingModel','embeddingDimension','sidecarEnabled','sidecarCapabilitiesUrl','sidecarParseUrl']),
    group('email','Email delivery',['emailMode','emailTo','hostedToken','emailApiKey','fromEmail','fromName','emailEnabled']),
    group('advanced','Advanced',['logLevel']),
  ];
}

/** Topic subfields share labels and editing controls while hosts retain their editor widgets. */
const topicFields = {
 name: { name: "Name", control: "text", description: "Heading text used as the section title in the daily report.", placeholder: "Topic name" },
 tag: { name: "Tag", control: "text", description: "Kebab-case ASCII slug. Written into each paper's YAML frontmatter as an Obsidian #tag.", placeholder: "Topic tag" },
 description: { name: "Description", control: "textarea", description: "Plain-language description of what belongs here. The AI uses this to decide which papers go into this topic.", placeholder: "What papers belong in this topic?" },
 detail: { name: "Detail report", control: "checkbox", description: "Generate a detailed paper note for this topic.", placeholder: "" },
} as const;
export type TopicSettingKey = keyof typeof topicFields;
export function getTopicSettingField(key: TopicSettingKey) { return { key, ...topicFields[key] }; }
export function getTopicSettingFields() { return (Object.keys(topicFields) as TopicSettingKey[]).map(getTopicSettingField); }

export function isValidLocalTime(value: string): boolean {
  return /^(?:[01]\d|2[0-3]):[0-5]\d$/.test(value);
}

export interface RunWindowTimeOption {
  value: string;
  label: string;
  valid: boolean;
}

export function runWindowTimeOptions(current: string): RunWindowTimeOption[] {
  const values: RunWindowTimeOption[] = [];
  for (let hour = 0; hour < 24; hour += 1) {
    for (let minute = 0; minute < 60; minute += 15) {
      const value = `${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}`;
      values.push({ value, label: value, valid: true });
    }
  }

  if (!values.some((option) => option.value === current)) {
    const valid = isValidLocalTime(current);
    values.push({
      value: current,
      label: valid ? current : `${current || "(empty)"} — invalid`,
      valid,
    });
  }
  return values.sort((a, b) => a.value.localeCompare(b.value));
}
