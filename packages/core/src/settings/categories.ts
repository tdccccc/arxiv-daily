import type { ArxivSettings } from "./types";

export function normalizeCategoryList(
  raw: unknown,
  fallback: string[],
): string[] {
  const values = Array.isArray(raw) ? raw : typeof raw === "string" ? [raw] : [];
  const out: string[] = [];
  for (const value of values) {
    const category = String(value).trim();
    if (!category || out.includes(category)) continue;
    out.push(category);
  }
  return out.length > 0 ? out : fallback;
}

export function arxivCategories(arxiv: ArxivSettings): string[] {
  if (Array.isArray(arxiv.categories)) {
    return normalizeCategoryList(arxiv.categories, []);
  }
  return normalizeCategoryList(arxiv.category, ["astro-ph"]);
}

export function primaryArxivCategory(arxiv: ArxivSettings): string {
  return arxivCategories(arxiv)[0] ?? "astro-ph";
}

export function formatArxivCategories(arxiv: ArxivSettings): string {
  return arxivCategories(arxiv).join(", ");
}
