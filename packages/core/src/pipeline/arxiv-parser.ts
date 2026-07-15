import type { MarkupParser } from "../core/adapters";
export interface PaperMeta {
  id: string;
  title: string;
  authors: string;
  abstract: string;
}

export interface DateBucket {
  announceDate: string;
  papers: PaperMeta[];
}

const MONTHS: Record<string, number> = {
  jan: 1,
  january: 1,
  feb: 2,
  february: 2,
  mar: 3,
  march: 3,
  apr: 4,
  april: 4,
  may: 5,
  jun: 6,
  june: 6,
  jul: 7,
  july: 7,
  aug: 8,
  august: 8,
  sep: 9,
  sept: 9,
  september: 9,
  oct: 10,
  october: 10,
  nov: 11,
  november: 11,
  dec: 12,
  december: 12,
};
const ID_RE = /^(\d{4}\.\d{4,5})(?:v\d+)?$/;

function parseHeaderDate(headerText: string): string | null {
  const m = /(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})/.exec(headerText);
  if (!m) return null;
  const [, rawDay, rawMonth, rawYear] = m;
  if (!rawDay || !rawMonth || !rawYear) return null;
  const day = Number(rawDay);
  const month = MONTHS[rawMonth.toLowerCase()];
  const year = Number(rawYear);
  if (!month) return null;
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function stripUnsafeTags(html: string): string {
  return html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<link\b[^>]*\/?>/gi, "")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, "");
}

/**
 * Parse arXiv /list/<cat>/recent into per-day buckets.
 *
 * Layout (arXiv as of 2024+): one <dl id='articles'> per announce date,
 * containing the <h3>date</h3> header followed by alternating <dt>/<dd>
 * entries. Abstracts are not included in listings — only title, authors,
 * and arXiv id. Callers who need abstracts fetch /abs/<id> separately.
 */
export function parseRecent(html: string, markupParser: MarkupParser): DateBucket[] {
  const safe = stripUnsafeTags(html);
  const doc = markupParser.parseFromString(safe, "text/html");
  const buckets: DateBucket[] = [];
  const dls = Array.from(doc.querySelectorAll("dl#articles")) as unknown as Element[];
  for (const dl of dls) {
    const h3 = dl.querySelector("h3");
    const date = h3 ? parseHeaderDate(h3.textContent ?? "") : null;
    if (!date) continue;
    const dts = Array.from(dl.querySelectorAll("dt")) as unknown as Element[];
    const dds = Array.from(dl.querySelectorAll("dd")) as unknown as Element[];
    const pairs = Math.min(dts.length, dds.length);
    const papers: PaperMeta[] = [];
    for (let i = 0; i < pairs; i++) {
      const p = parsePaper(dts[i]!, dds[i]!);
      if (p) papers.push(p);
    }
    buckets.push({ announceDate: date, papers });
  }
  buckets.sort((a, b) => (a.announceDate > b.announceDate ? -1 : 1));
  return buckets;
}

function parsePaper(dt: Element, dd: Element): PaperMeta | null {
  const absLink = dt.querySelector('a[title="Abstract"]');
  if (!absLink) return null;
  const id = (absLink.textContent ?? "").replace("arXiv:", "").trim();
  if (!id) return null;
  if (!ID_RE.test(id)) {
    console.warn(`[arxiv-daily] arxiv-parser: invalid arXiv id in listing: ${id}`);
    return null;
  }

  const titleDiv = dd.querySelector(".list-title");
  const title = (titleDiv?.textContent ?? "")
    .replace(/^\s*Title:\s*/, "")
    .replace(/\s+/g, " ")
    .trim();

  const authorsDiv = dd.querySelector(".list-authors");
  let authors = "Unknown";
  if (authorsDiv) {
    const links = Array.from(authorsDiv.querySelectorAll("a"));
    if (links.length > 0) {
      const first = (links[0]!.textContent ?? "").trim();
      authors = links.length > 1 ? `${first} et al.` : first;
    } else {
      authors = (authorsDiv.textContent ?? "").replace(/^\s*Authors:\s*/, "").trim();
    }
  }

  const abstractP = dd.querySelector("p.mathjax");
  const abstract = (abstractP?.textContent ?? "").replace(/\s+/g, " ").trim();

  return { id, title, authors, abstract };
}
