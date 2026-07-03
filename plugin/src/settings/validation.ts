import type { PluginSettings } from "./types";
import { arxivCategories } from "./categories";
import { minutesFromHHMM } from "../utils/time";

export interface ValidationResult {
  ok: boolean;
  reasons: string[];
}

export function validateLlmConfig(settings: PluginSettings): ValidationResult {
  const reasons: string[] = [];
  if (!settings.llm.apiKey.trim()) reasons.push("LLM API Key is empty");
  if (!settings.llm.baseUrl.trim()) reasons.push("LLM Base URL is empty");
  if (!settings.llm.model.trim()) reasons.push("LLM Model is empty");
  return { ok: reasons.length === 0, reasons };
}

export function validateFilterConfig(settings: PluginSettings): ValidationResult {
  const llm = validateLlmConfig(settings);
  const reasons = [...llm.reasons];
  const categories = arxivCategories(settings.arxiv);
  if (categories.length === 0) {
    reasons.push("No arXiv categories configured");
  }
  if (
    settings.output.linkStyle &&
    settings.output.linkStyle !== "wikilink" &&
    settings.output.linkStyle !== "relative"
  ) {
    reasons.push(`Invalid link style: ${settings.output.linkStyle}`);
  }
  if (
    settings.output.summaryLanguage &&
    settings.output.summaryLanguage !== "zh" &&
    settings.output.summaryLanguage !== "en"
  ) {
    reasons.push(`Invalid summary language: ${settings.output.summaryLanguage}`);
  }
  const seenCategories = new Set<string>();
  for (const category of settings.arxiv.categories ?? []) {
    const trimmed = category.trim();
    if (!trimmed) {
      reasons.push("arXiv category is empty");
      continue;
    }
    if (seenCategories.has(trimmed)) {
      reasons.push(`Duplicate arXiv category: ${trimmed}`);
    }
    seenCategories.add(trimmed);
  }
  if (settings.arxiv.topics.length === 0) {
    reasons.push("No research topics defined");
  }
  const seenTags = new Set<string>();
  settings.arxiv.topics.forEach((topic, i) => {
    const label = `Topic ${i + 1}`;
    if (!topic.name.trim()) reasons.push(`${label} name is empty`);
    const tag = topic.tag.trim();
    if (!tag) {
      reasons.push(`${label} tag is empty`);
    } else if (seenTags.has(tag)) {
      reasons.push(`Duplicate topic tag: ${tag}`);
    } else {
      seenTags.add(tag);
    }
    if (!topic.description.trim()) reasons.push(`${label} description is empty`);
  });
  return { ok: reasons.length === 0, reasons };
}

export function validateScheduleConfig(settings: PluginSettings): ValidationResult {
  const reasons: string[] = [];
  let start: number | null = null;
  let end: number | null = null;
  try {
    start = minutesFromHHMM(settings.schedule.runAtLocal);
  } catch {
    reasons.push(`Invalid run window start: ${settings.schedule.runAtLocal}`);
  }
  try {
    end = minutesFromHHMM(settings.schedule.runUntilLocal);
  } catch {
    reasons.push(`Invalid run window end: ${settings.schedule.runUntilLocal}`);
  }
  if (start != null && end != null && start > end) {
    reasons.push(
      "Run window start must be earlier than or equal to the end; overnight windows are not supported",
    );
  }
  return { ok: reasons.length === 0, reasons };
}

export function validateSchedulerConfig(settings: PluginSettings): ValidationResult {
  const filter = validateFilterConfig(settings);
  const schedule = validateScheduleConfig(settings);
  return {
    ok: filter.ok && schedule.ok,
    reasons: [...filter.reasons, ...schedule.reasons],
  };
}
