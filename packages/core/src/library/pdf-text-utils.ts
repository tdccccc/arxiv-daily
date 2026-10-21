import { modernArxivResources } from "../utils/arxiv";

/** One arXiv ID candidate found in PDF text with its canonical form. */
export interface ArxivIdCandidate {
  /** Canonical arXiv ID (modern `YYYY.NNNNN` or legacy `cat/XXXXXXXX`). */
  canonicalId: string;
  raw: string;
}

const MODERN_ARXIV_ID_IN_TEXT_RE = /(?:^|[^0-9A-Za-z])arXiv\s{0,8}:?\s{0,8}(\d{4}\.\d{4,5}(?:v\d+)?)(?=$|[^0-9A-Za-z])/gi;
const LEGACY_ARXIV_ID_IN_TEXT_RE = /(?:^|[^0-9A-Za-z])arXiv\s{0,8}:?\s{0,8}([a-z-]{1,32}(?:\.[a-z-]{1,32}){0,4}\/\d{7})(?=$|[^0-9A-Za-z])/gi;
const LEGACY_ARXIV_ID_STRICT_RE = /^(?:arXiv\s{0,8}:?\s{0,8})?[a-z-]{1,32}(?:\.[a-z-]{1,32}){0,4}\/\d{7}$/i;
const ARXIV_URL_IN_TEXT_RE = /(?:^|[^0-9A-Za-z])(?:https?:\/\/)?(?:www\.)?arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})(?:v\d+)?(?=$|[^0-9A-Za-z])/gi;

/**
 * Extract arXiv IDs from free text (page headers, XMP identifiers). Returns
 * candidates in document order, deduplicated; callers must decide trust.
 * Reference lists can mention foreign arXiv IDs, so extraction alone never
 * asserts identity — the caller scopes the text region and the count.
 */
export function extractArxivIdsFromText(text: string): ArxivIdCandidate[] {
  // Each form is matched on its own; merge the matches back into text order.
  const found: Array<{ index: number; raw: string }> = [];
  for (const pattern of [MODERN_ARXIV_ID_IN_TEXT_RE, LEGACY_ARXIV_ID_IN_TEXT_RE, ARXIV_URL_IN_TEXT_RE]) {
    for (const match of text.matchAll(pattern)) found.push({ index: match.index, raw: match[1]! });
  }
  found.sort((left, right) => left.index - right.index);
  const seen = new Set<string>();
  const candidates: ArxivIdCandidate[] = [];
  for (const { raw } of found) {
    const canonicalId = canonicalArxivId(raw);
    if (!canonicalId || seen.has(canonicalId)) continue;
    seen.add(canonicalId);
    candidates.push({ canonicalId, raw });
  }
  return candidates;
}

/** Canonicalize a single arXiv ID string from text (modern or legacy). */
export function modernArxivIdFromText(text: string): string | undefined {
  return extractArxivIdsFromText(text)[0]?.canonicalId;
}

function canonicalArxivId(raw: string): string | undefined {
  const modern = modernArxivResources(raw);
  if (modern) return modern.id;
  const legacy = stripArxivPrefix(raw).trim().toLowerCase();
  if (LEGACY_ARXIV_ID_STRICT_RE.test(legacy)) return legacy;
  return undefined;
}

function stripArxivPrefix(raw: string): string {
  if (!raw.toLowerCase().startsWith("arxiv")) return raw;
  let index = 5;
  while (index < raw.length && raw.charCodeAt(index) <= 32) index += 1;
  if (raw.charCodeAt(index) === 58) {
    index += 1;
    while (index < raw.length && raw.charCodeAt(index) <= 32) index += 1;
  }
  return raw.slice(index);
}
