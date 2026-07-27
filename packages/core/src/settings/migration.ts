import { DEFAULT_SETTINGS } from "./defaults";
import type { ArxivSettings, EmailSettings, Topic } from "./types";
import { normalizeCategoryList } from "./categories";

function titleCase(slug: string): string {
  return slug
    .replace(/-/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function freshDefaults(): Topic[] {
  return DEFAULT_SETTINGS.arxiv.topics.map((t) => ({ ...t, id: crypto.randomUUID() }));
}

export function migrateArxivSettings(raw: unknown): ArxivSettings {
  const arxiv = (raw ?? {}) as Record<string, unknown>;

  const hasCategories = Array.isArray(arxiv.categories);
  const categories = hasCategories
    ? normalizeCategoryList(arxiv.categories, [])
    : normalizeCategoryList(
        arxiv.category,
        DEFAULT_SETTINGS.arxiv.categories,
      );
  const category = categories[0]
    ?? (typeof arxiv.category === "string" && arxiv.category.trim()
      ? arxiv.category.trim()
      : DEFAULT_SETTINGS.arxiv.category);
  const timezone =
    typeof arxiv.timezone === "string" && arxiv.timezone.length > 0
      ? arxiv.timezone
      : DEFAULT_SETTINGS.arxiv.timezone;

  if (Array.isArray(arxiv.topics) && arxiv.topics.length > 0) {
    return { category, categories, topics: arxiv.topics as Topic[], timezone };
  }

  const detailCategories = Array.isArray(arxiv.detailCategories)
    ? (arxiv.detailCategories as unknown[]).filter(
        (v): v is string => typeof v === "string",
      )
    : [];
  const displayMap =
    (arxiv.categoryDisplayMap as Record<string, string> | undefined) ?? {};

  const topics: Topic[] =
    detailCategories.length > 0
      ? detailCategories.map((tag) => ({
          id: crypto.randomUUID(),
          name: displayMap[tag] ?? titleCase(tag),
          tag,
          description: "",
          detail: true,
        }))
      : freshDefaults();

  return { category, categories, topics, timezone };
}

/** Soft-merge email settings so older data.json without `email` still loads. */
export function migrateEmailSettings(raw: unknown): EmailSettings {
  const defaults = DEFAULT_SETTINGS.email;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { ...defaults };
  }
  const email = raw as Record<string, unknown>;
  const mode =
    email.mode === "hosted" || email.mode === "self"
      ? email.mode
      : defaults.mode;
  return {
    enabled: typeof email.enabled === "boolean" ? email.enabled : defaults.enabled,
    mode,
    to: typeof email.to === "string" ? email.to : defaults.to,
    fromEmail: typeof email.fromEmail === "string" ? email.fromEmail : defaults.fromEmail,
    fromName:
      typeof email.fromName === "string"
        ? email.fromName
        : defaults.fromName ?? "",
    apiKey:
      typeof email.apiKey === "string" ? email.apiKey : defaults.apiKey ?? "",
    hostedToken:
      typeof email.hostedToken === "string"
        ? email.hostedToken
        : defaults.hostedToken ?? "",
    hostedBaseUrl:
      typeof email.hostedBaseUrl === "string"
        ? email.hostedBaseUrl
        : defaults.hostedBaseUrl ?? "",
  };
}
