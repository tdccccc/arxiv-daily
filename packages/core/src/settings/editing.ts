import { deriveTopicDescription, normalizeTopic } from "./topics";
import { isValidMaxDailyPapers } from "./daily-paper-limit";
import { getBusinessSetting } from "./schema";
import type { PluginSettings } from "./types";
import { isDetailSelectionProfile, sanitizeDetailSelection } from "./detail-selection";
import { validateLocalPdfParserSidecarConfig, validateScheduleConfig, validateSchedulerConfig, validateVaultRelativeDirectory, vaultRelativeDirectoriesCollide } from "./validation";

/** Host-neutral editing rules. Incomplete drafts are valid; execution checks readiness separately. */
export function normalizeSettingsEdits(candidate: PluginSettings, changedKeys: readonly string[]): PluginSettings {
  const result = structuredClone(candidate);
  const touches = (key: string) => changedKeys.some(changed => changed === key || changed.startsWith(`${key}.`) || key.startsWith(`${changed}.`));
  if (touches("llm")) {
    const llm = result.llm;
    for (const field of ["apiKey", "provider", "baseUrl", "model", "reasoningEffort"] as const) llm[field] = text(llm[field], `llm.${field}`);
    llm.baseUrl = endpoint(llm.baseUrl, "llm.baseUrl");
    bool(llm.thinkingMode, "llm.thinkingMode");
  }
  if (touches("arxiv.categories")) {
    const raw = result.arxiv.categories;
    if (!Array.isArray(raw) || raw.length > 200) invalid("arxiv.categories");
    const categories = raw.map(category => text(category, "arxiv.categories"));
    if (categories.some(category => !/^[a-zA-Z][a-zA-Z0-9.-]*$/.test(category)) || new Set(categories).size !== categories.length) invalid("arxiv.categories");
    result.arxiv.categories = categories;
    result.arxiv.category = categories[0] ?? "";
  }
  if (touches("arxiv.topics")) {
    if (!Array.isArray(result.arxiv.topics) || result.arxiv.topics.length > 100) invalid("arxiv.topics");
    const ids = new Set<string>();
    result.arxiv.topics = result.arxiv.topics.map(topic => {
      if (!topic || typeof topic !== "object" || Array.isArray(topic)) invalid("arxiv.topics");
      const id = text(topic.id, "arxiv.topics.id");
      if (!id || ids.has(id)) invalid("arxiv.topics.id");
      ids.add(id);
      bool(topic.detail, "arxiv.topics.detail");
      const rawDirections = Object.hasOwn(topic, "directions") ? topic.directions : normalizeTopic(topic).directions;
      if (!Array.isArray(rawDirections) || rawDirections.length > 1000) invalid("arxiv.topics.directions");
      const directionIds = new Set<string>();
      const directions = rawDirections.map(direction => {
        if (!direction || typeof direction !== "object") invalid("arxiv.topics.directions");
        const directionId = text(direction.id, "arxiv.topics.directions.id");
        if (!directionId || directionIds.has(directionId)) invalid("arxiv.topics.directions.id");
        directionIds.add(directionId);
        enumValue(direction.origin, ["manual", "migrated", "library"], "arxiv.topics.directions.origin");
        return { ...direction, id: directionId, text: text(direction.text, "arxiv.topics.directions.text", true).replace(/\s*\n+\s*/g, " ") };
      });
      return { ...topic, id, name: text(topic.name, "arxiv.topics.name"), tag: text(topic.tag, "arxiv.topics.tag"), directions, description: deriveTopicDescription(directions) };
    });
  }
  if (touches("arxiv.timezone")) {
    result.arxiv.timezone = text(result.arxiv.timezone, "arxiv.timezone");
    if (!result.arxiv.timezone) invalid("arxiv.timezone");
    try { new Intl.DateTimeFormat("en-US", { timeZone: result.arxiv.timezone }).format(); } catch { invalid("arxiv.timezone"); }
  }
  if (touches("email")) {
    const email = result.email;
    bool(email.enabled, "email.enabled"); enumValue(email.mode, Object.keys(getBusinessSetting('emailMode').options!), "email.mode");
    for (const field of ["to", "fromEmail"] as const) email[field] = text(email[field], `email.${field}`);
    for (const field of ["fromName", "apiKey", "hostedToken", "hostedBaseUrl"] as const) if (email[field] !== undefined) email[field] = text(email[field], `email.${field}`, false, field !== "fromName");
    if (email.hostedBaseUrl) email.hostedBaseUrl = endpoint(email.hostedBaseUrl, "email.hostedBaseUrl");
  }
  if (touches("embedding")) {
    const embedding = result.embedding;
    enumValue(embedding.mode, Object.keys(getBusinessSetting('embeddingMode').options!), "embedding.mode");
    for (const field of ["provider", "baseUrl", "apiKey", "model"] as const) embedding[field] = text(embedding[field], `embedding.${field}`);
    embedding.baseUrl = endpoint(embedding.baseUrl, "embedding.baseUrl");
    integer(embedding.dimension, "embedding.dimension", 1000000); bool(embedding.initialChoiceDone, "embedding.initialChoiceDone");
  }
  if (touches("output.dailyDir") || touches("output.papersDir")) {
    for (const field of ["dailyDir", "papersDir"] as const) {
      const validation = validateVaultRelativeDirectory(result.output[field]);
      if (!validation.ok || !validation.value) invalid(`output.${field}`);
      result.output[field] = validation.value;
    }
    if (vaultRelativeDirectoriesCollide(result.output.dailyDir, result.output.papersDir)) throw new Error("Daily and papers directories must be different");
  }
  if (touches("output.maxDailyPapers") && !isValidMaxDailyPapers(result.output.maxDailyPapers)) invalid("output.maxDailyPapers");
  if (touches("output.linkStyle") && result.output.linkStyle !== undefined) enumValue(result.output.linkStyle, Object.keys(getBusinessSetting('linkStyle').options!), "output.linkStyle");
  if (touches("output.summaryLanguage") && result.output.summaryLanguage !== undefined) enumValue(result.output.summaryLanguage, Object.keys(getBusinessSetting('summaryLanguage').options!), "output.summaryLanguage");
  if (touches("pdfParserSidecar")) {
    const sidecar = result.pdfParserSidecar;
    bool(sidecar.enabled, "pdfParserSidecar.enabled");
    sidecar.capabilitiesUrl = text(sidecar.capabilitiesUrl, "pdfParserSidecar.capabilitiesUrl");
    sidecar.parseUrl = text(sidecar.parseUrl, "pdfParserSidecar.parseUrl");
    const validation = validateLocalPdfParserSidecarConfig(result);
    if (!validation.ok) throw new Error(validation.reasons.join("; "));
  }
  if (touches("detailSelection")) {
    if (!result.detailSelection || !isDetailSelectionProfile(result.detailSelection.profile)) invalid("detailSelection.profile");
    result.detailSelection = sanitizeDetailSelection(result.detailSelection);
  }
  if (touches("advanced.logLevel")) enumValue(result.advanced.logLevel, Object.keys(getBusinessSetting('logLevel').options!), "advanced.logLevel");
  if (touches("schedule")) {
    const schedule = result.schedule;
    bool(schedule.enabled, "schedule.enabled");
    schedule.runAtLocal = text(schedule.runAtLocal, "schedule.runAtLocal");
    schedule.runUntilLocal = text(schedule.runUntilLocal, "schedule.runUntilLocal");
    integer(schedule.tickIntervalMin, "scheduler tick interval", 1440);
    const validation = schedule.enabled ? validateSchedulerConfig(result) : validateScheduleConfig(result);
    if (!validation.ok) throw new Error(validation.reasons.join("; "));
  }
  return result;
}
function invalid(field: string): never { throw new Error(`Invalid ${field}`); }
function text(value: unknown, field: string, multiline = false, trim = true): string {
  if (typeof value !== "string" || value.length > 20000 || (multiline ? /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/u : /[\u0000-\u001f\u007f]/u).test(value)) invalid(field);
  return trim ? value.trim() : value;
}
function bool(value: unknown, field: string): void { if (typeof value !== "boolean") invalid(field); }
function integer(value: unknown, field: string, max: number): void { if (typeof value !== "number" || !Number.isInteger(value) || value < 1 || value > max) invalid(field); }
function enumValue(value: unknown, options: readonly string[], field: string): void { if (typeof value !== "string" || !options.includes(value)) invalid(field); }
function endpoint(value: string, field: string): string {
  if (!value) return value;
  try { const url = new URL(value); if (!["http:", "https:"].includes(url.protocol) || !url.hostname || url.username || url.password || url.search || url.hash) invalid(field); } catch { invalid(field); }
  return value;
}
