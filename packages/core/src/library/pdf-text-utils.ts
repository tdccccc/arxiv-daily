import { modernArxivResources } from "../utils/arxiv";

/** One arXiv ID candidate found in PDF text with its canonical form. */
export interface ArxivIdCandidate {
  /** Canonical arXiv ID (modern `YYYY.NNNNN` or legacy `cat/XXXXXXXX`). */
  canonicalId: string;
  raw: string;
}

const MODERN_ARXIV_ID_IN_TEXT_RE = /(?:^|[^0-9A-Za-z])arXiv\s*:?\s*(\d{4}\.\d{4,5}(?:v\d+)?)(?=$|[^0-9A-Za-z])/gi;
const LEGACY_ARXIV_ID_IN_TEXT_RE = /(?:^|[^0-9A-Za-z])arXiv\s*:?\s*([a-z-]+(?:\.[a-z-]+)*\/\d{7})(?=$|[^0-9A-Za-z])/gi;
const LEGACY_ARXIV_ID_STRICT_RE = /^(?:arXiv\s*:?\s*)?[a-z-]+(?:\.[a-z-]+)*\/\d{7}$/i;
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
  const legacy = raw.replace(/^arxiv\s*:?\s*/i, "").trim().toLowerCase();
  if (LEGACY_ARXIV_ID_STRICT_RE.test(legacy)) return legacy;
  return undefined;
}
