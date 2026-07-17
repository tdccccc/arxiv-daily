import type { PluginSettings } from "./types";
import { arxivCategories } from "./categories";
import { minutesFromHHMM } from "../utils/time";

export interface ValidationResult {
  ok: boolean;
  reasons: string[];
}

export interface VaultRelativeDirectoryValidation {
  ok: boolean;
  value?: string;
  reason?: string;
}

const WINDOWS_RESERVED_DEVICE_NAME_RE =
  /^(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)/iu;

/** A portable equality key for already-canonical vault-relative paths. */
export function portablePathCollisionKey(path: string): string {
  return path.normalize("NFC").toUpperCase().toLowerCase().normalize("NFC");
}

/** Return whether two paths collide on portable, case-insensitive filesystems. */
export function vaultRelativeDirectoriesCollide(a: string, b: string): boolean {
  return portablePathCollisionKey(a) === portablePathCollisionKey(b);
}

/** Canonicalize and validate a portable, vault-relative directory setting. */
export function validateVaultRelativeDirectory(
  input: unknown,
): VaultRelativeDirectoryValidation {
  if (typeof input !== "string") {
    return { ok: false, reason: "must be a string" };
  }
  const value = input.trim().replace(/\\/g, "/").normalize("NFC");
  if (!value) return { ok: false, reason: "must not be empty" };
  if (/[\u0000-\u001f\u007f]/u.test(value)) {
    return { ok: false, reason: "must not contain control characters" };
  }
  if (/^(?:\/|[a-z]:|\/\/)/i.test(value)) {
    return { ok: false, reason: "must be vault-relative" };
  }
  if (/^[a-z][a-z0-9+.-]*:/i.test(value)) {
    return { ok: false, reason: "must not be a URL or URI" };
  }
  if (/[<>:"|?*]/u.test(value)) {
    return { ok: false, reason: "contains characters invalid on portable filesystems" };
  }
  const segments = value.split("/");
  if (segments.some((segment) => !segment || segment === "." || segment === "..")) {
    return { ok: false, reason: "must not contain empty or dot path segments" };
  }
  if (segments.some((segment) => segment.toLowerCase() === ".obsidian")) {
    return { ok: false, reason: "must not target the .obsidian directory" };
  }
  if (segments.some((segment) => /[. ]$/u.test(segment))) {
    return { ok: false, reason: "path segments must not end in a dot or space" };
  }
  if (segments.some((segment) => WINDOWS_RESERVED_DEVICE_NAME_RE.test(segment))) {
    return { ok: false, reason: "must not contain Windows reserved device names" };
  }
  return { ok: true, value: segments.join("/") };
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
  const dailyDir = validateVaultRelativeDirectory(settings.output.dailyDir);
  const papersDir = validateVaultRelativeDirectory(settings.output.papersDir);
  if (!dailyDir.ok) reasons.push(`Invalid daily directory: ${dailyDir.reason}`);
  if (!papersDir.ok) reasons.push(`Invalid papers directory: ${papersDir.reason}`);
  if (
    dailyDir.ok &&
    papersDir.ok &&
    dailyDir.value &&
    papersDir.value &&
    vaultRelativeDirectoriesCollide(dailyDir.value, papersDir.value)
  ) {
    reasons.push("Daily and papers directories must be different");
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
