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
  return normalizeCategoryList(
    arxiv.categories,
    normalizeCategoryList(arxiv.category, ["astro-ph"]),
  );
}

export function primaryArxivCategory(arxiv: ArxivSettings): string {
  return arxivCategories(arxiv)[0];
}

export function formatArxivCategories(arxiv: ArxivSettings): string {
  return arxivCategories(arxiv).join(", ");
}
