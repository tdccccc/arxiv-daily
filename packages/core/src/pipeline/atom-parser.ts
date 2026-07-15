import type { MarkupParser } from "../core/adapters";
import type { PaperMeta } from "./arxiv-parser";

export interface AtomPaperMeta extends PaperMeta {
  published: string;
  updated: string;
  primaryCategory: string;
  categories: string[];
}

export function parseAtomAbstracts(
  xml: string,
  markupParser: MarkupParser,
): Map<string, string> {
  const out = new Map<string, string>();
  for (const paper of parseAtomPapers(xml, markupParser)) {
    if (paper.abstract) out.set(paper.id, paper.abstract);
  }
  return out;
}

/**
 * Parse an arXiv Atom API response into paper metadata.
 *
 * Strips trailing version suffix from <id> URLs ("2605.08080v1" → "2605.08080")
 * so callers can look up by the canonical id form.
 */
export function parseAtomPapers(
  xml: string,
  markupParser: MarkupParser,
): AtomPaperMeta[] {
  const out: AtomPaperMeta[] = [];
  const doc = markupParser.parseFromString(xml, "text/xml");
  const entries = Array.from(doc.querySelectorAll("entry")) as unknown as Element[];
  for (const entry of entries) {
    const idEl = entry.querySelector("id");
    if (!idEl) continue;
    const fullId = (idEl.textContent ?? "").trim();
    const m = /\/abs\/([^/?#]+?)(v\d+)?$/.exec(fullId);
    if (!m) continue;
    const baseId = m[1];
    if (!baseId) continue;
    const title = text(entry.querySelector("title"));
    const abstract = text(entry.querySelector("summary"));
    const authors = (
      Array.from(entry.querySelectorAll("author > name")) as unknown as Element[]
    )
      .map((el) => text(el))
      .filter(Boolean);
    const primaryCategoryElement = Array.from(entry.children).find((element) => {
      const name = element.localName.toLowerCase();
      return name === "primary_category" || name.endsWith(":primary_category");
    });
    const primaryCategory = primaryCategoryElement?.getAttribute("term")?.trim() ?? "";
    const categories = (
      Array.from(entry.querySelectorAll("category")) as unknown as Element[]
    )
      .map((el) => el.getAttribute("term")?.trim() ?? "")
      .filter((category): category is string => Boolean(category));
    out.push({
      id: baseId,
      title,
      authors: formatAuthors(authors),
      abstract,
      published: text(entry.querySelector("published")),
      updated: text(entry.querySelector("updated")),
      primaryCategory,
      categories,
    });
  }
  return out;
}

function text(el: Element | null): string {
  return (el?.textContent ?? "").replace(/\s+/g, " ").trim();
}

function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return "Unknown";
  const first = authors[0] ?? "Unknown";
  return authors.length > 1 ? `${first} et al.` : first;
}
