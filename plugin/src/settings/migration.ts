import { DEFAULT_SETTINGS } from "./defaults";
import type { ArxivSettings, Topic } from "./types";
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

  const categories = normalizeCategoryList(
    arxiv.categories,
    normalizeCategoryList(arxiv.category, DEFAULT_SETTINGS.arxiv.categories),
  );
  const category = categories[0] ?? DEFAULT_SETTINGS.arxiv.category;
  const timezone =
    typeof arxiv.timezone === "string" ? arxiv.timezone : DEFAULT_SETTINGS.arxiv.timezone;

  if (Array.isArray(arxiv.topics) && arxiv.topics.length > 0) {
    return { category, categories, topics: arxiv.topics as Topic[], timezone };
  }

  const detailCategories = Array.isArray(arxiv.detailCategories)
    ? (arxiv.detailCategories as string[])
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
