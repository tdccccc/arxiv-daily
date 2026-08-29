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
  const seen = new Set<string>();
  const candidates: ArxivIdCandidate[] = [];
  const push = (raw: string): void => {
    const canonicalId = canonicalArxivId(raw);
    if (!canonicalId || seen.has(canonicalId)) return;
    seen.add(canonicalId);
    candidates.push({ canonicalId, raw });
  };
  for (const match of text.matchAll(MODERN_ARXIV_ID_IN_TEXT_RE)) push(match[1]!);
  for (const match of text.matchAll(LEGACY_ARXIV_ID_IN_TEXT_RE)) push(match[1]!);
  for (const match of text.matchAll(ARXIV_URL_IN_TEXT_RE)) push(match[1]!);
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
