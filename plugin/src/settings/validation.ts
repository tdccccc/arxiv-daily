import type { PluginSettings } from "./types";

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
