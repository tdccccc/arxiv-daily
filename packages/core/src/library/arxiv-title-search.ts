import type { HttpClient } from "../core/adapters";
import { ArxivHttpError, coordinateArxivRequest, type ArxivRequestCoordinator } from "../pipeline/arxiv-fetcher";
import { modernArxivResources } from "../utils/arxiv";

/**
 * arXiv title search (identification v2, L2).
 *
 * Resolves a paper title to a canonical arXiv ID via the public arXiv Atom
 * API search endpoint. Acceptance is strict: the best match must clear a
 * normalized word-overlap threshold, and a second distinct candidate above
 * the same threshold means ambiguity — no ID is returned. The ID is the one in
 * the entry's own `<id>`; text elsewhere in the entry (summary, comment,
 * journal reference) cites other papers and is never read for IDs. The query
 * sends only the title text (public paper metadata) to arxiv.org.
 *
 * The request shares the spacing and the 429 cooldown of every other arXiv API
 * request. A search that could not be made or read throws; `arxivId: null` always
 * means arXiv was asked and nothing matched.
 */
export interface ArxivTitleSearchResult {
  arxivId: string | null;
  /** Matched entry title for diagnostics; undefined when nothing matched. */
  matchedTitle?: string;
}

const SEARCH_URL = "https://export.arxiv.org/api/query";
const MAX_RESULTS = 5;
const SEARCH_TIMEOUT_MS = 15_000;
/** Minimum word-overlap ratio for a candidate title to be accepted. */
const MIN_SIMILARITY = 0.75;

export async function searchArxivTitle(
  http: HttpClient,
  title: string,
  signal?: AbortSignal,
  /** The request coordinator to pace and cool down with; tests pass one with a fake clock. */
  coordinator?: ArxivRequestCoordinator,
): Promise<ArxivTitleSearchResult> {
  const query = title.trim().replace(/\s+/g, " ").replace(/["\\]/g, "");
  if (query.length < 4) return { arxivId: null };
  const url = `${SEARCH_URL}?search_query=ti:"${encodeURIComponent(query)}"&max_results=${MAX_RESULTS}`;
  const response = await coordinateArxivRequest(async () => {
    const answer = await http.request({
      url,
      method: "GET",
      headers: { Accept: "application/atom+xml" },
      timeoutMs: SEARCH_TIMEOUT_MS,
      signal,
    });
    if (answer.status < 200 || answer.status >= 300) {
      throw new ArxivHttpError(answer.status, url, answer.headers);
    }
    return answer;
  }, url, signal, coordinator);
  // A 200 that is not an Atom feed (a proxy or maintenance page) is no answer, not "no match".
  if (!/<feed[\s>]/.test(response.bodyText)) {
    throw new Error("arxiv title search returned something other than an Atom feed");
  }
  return matchBestTitle(query, response.bodyText);
}

export function matchBestTitle(query: string, atomXml: string): ArxivTitleSearchResult {
  const candidates = parseAtomEntries(atomXml);
  const queryWords = normalizeTitle(query);
  const scored = candidates
    .map((candidate) => ({
      candidate,
      similarity: titleSimilarity(queryWords, normalizeTitle(candidate.title)),
    }))
    .sort((left, right) => right.similarity - left.similarity);
  const best = scored[0];
  if (!best || best.similarity < MIN_SIMILARITY) return { arxivId: null };
  const second = scored[1];
  if (second && second.similarity >= MIN_SIMILARITY && second.candidate.id !== best.candidate.id) {
    return { arxivId: null };
  }
  return { arxivId: best.candidate.id, matchedTitle: best.candidate.title };
}

interface AtomEntry {
  id: string;
  title: string;
}

function parseAtomEntries(xml: string): AtomEntry[] {
  const entries: AtomEntry[] = [];
  let cursor = 0;
  while (cursor < xml.length) {
    const entryStart = xml.indexOf("<entry>", cursor);
    if (entryStart < 0) break;
    const entryEnd = xml.indexOf("</entry>", entryStart + "<entry>".length);
    if (entryEnd < 0) break;
    const body = xml.slice(entryStart + "<entry>".length, entryEnd);
    cursor = entryEnd + "</entry>".length;
    const idText = extractAtomElement(body, "id");
    const id = idText === undefined ? undefined : modernArxivResources(decodeXml(idText))?.id;
    const title = extractAtomElement(body, "title");
    if (id && title) entries.push({ id, title: decodeXml(title).replace(/\s+/g, " ").trim() });
  }
  return entries;
}

/** Text of the first `<name …>…</name>` element in `body`, if any. */
function extractAtomElement(body: string, name: string): string | undefined {
  let cursor = 0;
  while (cursor < body.length) {
    const open = body.indexOf(`<${name}`, cursor);
    if (open < 0) return undefined;
    if (!/[\s>]/.test(body.charAt(open + name.length + 1))) {
      // Another element whose name merely starts with `name`.
      cursor = open + 1;
      continue;
    }
    const openEnd = body.indexOf(">", open);
    if (openEnd < 0) return undefined;
    const close = body.indexOf(`</${name}>`, openEnd + 1);
    if (close < 0) return undefined;
    return body.slice(openEnd + 1, close);
  }
  return undefined;
}

function normalizeTitle(title: string): string[] {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter((word) => word.length > 0);
}

function titleSimilarity(left: string[], right: string[]): number {
  if (left.length === 0 || right.length === 0) return 0;
  const rightSet = new Set(right);
  const overlap = left.filter((word) => rightSet.has(word)).length;
  return overlap / Math.max(left.length, right.length);
}

function decodeXml(value: string): string {
  return value
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, "\"")
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, "&");
}
