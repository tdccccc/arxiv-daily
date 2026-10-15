import { DISCOVERY_PROVENANCE_MARKER_PREFIX } from "./discovery-provenance-marker";
import {
  normalizePersonalNovelty,
  normalizePersonalNoveltyRecord,
  type PersonalNovelty,
  type PersonalNoveltyRecord,
} from "./personalized-novelty";

/** Newest marker version written; v1 (no titles) is still read and still written when there are none. */
export const PERSONAL_NOVELTY_MARKER_VERSION = 2 as const;
export const PERSONAL_NOVELTY_MARKER_PREFIX = "arxiv-daily-personal-novelty";
/** Display titles are carried by the marker only for a basis of at most this many papers. */
export const PERSONAL_NOVELTY_MARKER_MAX_TITLED_BASIS = 5 as const;
/** Each marker title is shortened (with an ellipsis) to at most this many code units. */
export const PERSONAL_NOVELTY_MARKER_MAX_TITLE_CODE_UNITS = 300 as const;
/**
 * Deterministic marker bound, measured on the two worst-case canonical payloads
 * (JSON escaping inflates a code unit at most 6x; base64url inflates by 4/3):
 *
 * - v1, no titles: a 1000-unit explanation of control characters plus the 40
 *   comparison-basis keys the read-back bound allows, the longest being the
 *   76-character `file:sha256:` form: 12_468 code units.
 * - v2, titles: the same explanation, at most 5 `file:sha256:` keys and 5
 *   titles of 300 control characters each: 20_808 code units.
 *
 * 21_000 covers both, so every valid marker renders and parses within bounds.
 */
export const PERSONAL_NOVELTY_MARKER_MAX_CODE_UNITS = 21_000 as const;

interface MarkerPayloadV1 {
  v: 1;
  d: string;
  a: string;
  n: PersonalNovelty;
}

interface MarkerPayloadV2 {
  v: 2;
  d: string;
  a: string;
  n: PersonalNovelty;
  /** Display titles aligned one-to-one with n.comparisonBasis. */
  t: string[];
}

export type MarkerPayload = MarkerPayloadV1 | MarkerPayloadV2;

export interface PaperOccurrencePersonalNovelty {
  arxivId: string;
  /** The display shape when the marker carried titles (v2), otherwise the minimal shape. */
  novelty: PersonalNoveltyRecord;
}

export type DailyReportPersonalNoveltyParseResult =
  | { kind: "valid"; occurrences: PaperOccurrencePersonalNovelty[] }
  | { kind: "invalid"; reason: string };

/**
 * Renders the novelty marker. Titles travel in the marker (v2) when the novelty
 * carries display titles and the basis has at most
 * PERSONAL_NOVELTY_MARKER_MAX_TITLED_BASIS papers; otherwise the line is the
 * titleless v1 form. Over-long titles are shortened here and nowhere else.
 */
export function renderPersonalNoveltyMarker(
  novelty: PersonalNoveltyRecord,
  arxivId: string,
  reportDate: string,
): string {
  const record = normalizePersonalNoveltyRecord(novelty);
  if (!record) throw new TypeError("personal novelty is malformed");
  if (!isCanonicalArxivId(arxivId)) throw new TypeError("personal novelty arXiv ID is malformed");
  if (!isReportDate(reportDate)) throw new TypeError("personal novelty report date is malformed");
  const minimal: PersonalNovelty = {
    differenceType: record.differenceType,
    comparisonBasis: [...record.comparisonBasis],
    evidenceDepth: record.evidenceDepth,
    explanation: record.explanation,
  };
  const titles = "comparisonBasisTitles" in record
    && record.comparisonBasis.length <= PERSONAL_NOVELTY_MARKER_MAX_TITLED_BASIS
    ? record.comparisonBasis.map((key) => capMarkerTitle(record.comparisonBasisTitles[key]!))
    : null;
  const payload: MarkerPayload = titles
    ? { v: 2, d: reportDate, a: arxivId, n: minimal, t: titles }
    : { v: 1, d: reportDate, a: arxivId, n: minimal };
  const encoded = encodeBase64Url(JSON.stringify(payload));
  const marker = `<!-- ${PERSONAL_NOVELTY_MARKER_PREFIX}:v${payload.v}:${encoded} -->`;
  if (marker.length > PERSONAL_NOVELTY_MARKER_MAX_CODE_UNITS) {
    throw new TypeError("personal novelty marker is too large");
  }
  return marker;
}

export function parsePersonalNoveltyMarker(line: string): MarkerPayload | null {
  if (line.length > PERSONAL_NOVELTY_MARKER_MAX_CODE_UNITS) return null;
  const match = /^<!-- arxiv-daily-personal-novelty:v([12]):([A-Za-z0-9_-]+) -->$/.exec(line);
  if (!match) return null;
  const version = match[1] === "2" ? 2 : 1;
  let payload: unknown;
  try {
    payload = JSON.parse(decodeBase64Url(match[2]!));
  } catch {
    return null;
  }
  if (!isExactDataObject(payload, version === 2 ? ["v", "d", "a", "n", "t"] : ["v", "d", "a", "n"])
    || payload.v !== version || !isReportDate(payload.d) || !isCanonicalArxivId(payload.a)) return null;
  const novelty = normalizePersonalNovelty(payload.n);
  if (!novelty) return null;
  let normalized: MarkerPayload;
  let source: PersonalNoveltyRecord = novelty;
  if (version === 2) {
    const titles = payload.t;
    if (!Array.isArray(titles) || titles.length !== novelty.comparisonBasis.length
      || titles.length > PERSONAL_NOVELTY_MARKER_MAX_TITLED_BASIS
      || !titles.every(isMarkerTitle)) return null;
    normalized = { v: 2, d: payload.d, a: payload.a, n: novelty, t: [...titles] };
    source = withTitles(novelty, normalized.t);
  } else {
    normalized = { v: 1, d: payload.d, a: payload.a, n: novelty };
  }
  try {
    if (renderPersonalNoveltyMarker(source, payload.a, payload.d) !== line) return null;
  } catch {
    return null;
  }
  return normalized;
}

/** The display shape a v2 marker stands for. */
export function markerPayloadNovelty(payload: MarkerPayload): PersonalNoveltyRecord {
  return payload.v === 2 ? withTitles(payload.n, payload.t) : payload.n;
}

function withTitles(novelty: PersonalNovelty, titles: readonly string[]): PersonalNoveltyRecord {
  return {
    ...novelty,
    comparisonBasisTitles: Object.fromEntries(novelty.comparisonBasis.map((key, index) => [key, titles[index]!])),
  };
}

function isMarkerTitle(value: unknown): value is string {
  return typeof value === "string" && value.length > 0
    && value.length <= PERSONAL_NOVELTY_MARKER_MAX_TITLE_CODE_UNITS && value.trim() === value;
}

/** Shortens to the cap with an ellipsis without splitting a surrogate pair; shorter titles pass unchanged. */
function capMarkerTitle(title: string): string {
  if (title.length <= PERSONAL_NOVELTY_MARKER_MAX_TITLE_CODE_UNITS) return title;
  let end = PERSONAL_NOVELTY_MARKER_MAX_TITLE_CODE_UNITS - 1;
  const last = title.charCodeAt(end - 1);
  if (last >= 0xd800 && last <= 0xdbff) end -= 1;
  return `${title.slice(0, end).trimEnd()}\u2026`;
}

/**
 * Parses the complete report projection for personal-novelty occurrences.
 * User edits are authoritative like all committed Markdown edits, but
 * structurally inconsistent novelty never mutates the derived index, and
 * invalid novelty never disables discovery-provenance projection (each marker
 * family parses independently).
 *
 * Placement: exactly one novelty marker per canonical paper block, on the
 * line right after the block heading, or on the following line when a
 * discovery-provenance marker occupies the first slot.
 */
export function parseDailyReportPersonalNovelty(
  markdown: string,
  reportDate: string,
): DailyReportPersonalNoveltyParseResult {
  if (!isReportDate(reportDate)) return { kind: "invalid", reason: "invalid report date" };
  const lines = markdown.split(/\r?\n/);
  const markerLineIndexes = lines.flatMap((line, index) =>
    line.startsWith(`<!-- ${PERSONAL_NOVELTY_MARKER_PREFIX}:`) ? [index] : []);
  if (markerLineIndexes.length === 0) return { kind: "valid", occurrences: [] };

  const blocks: Array<{ start: number; end: number }> = [];
  for (let index = 0; index < lines.length; index += 1) {
    if (!/^###\s+/.test(lines[index] ?? "")) continue;
    let end = lines.length;
    for (let scan = index + 1; scan < lines.length; scan += 1) {
      if (/^#{2,3}\s+/.test(lines[scan] ?? "")) { end = scan; break; }
    }
    blocks.push({ start: index, end });
  }

  const occurrences: PaperOccurrencePersonalNovelty[] = [];
  const seenIds = new Set<string>();
  const consumedMarkers = new Set<number>();
  for (const block of blocks) {
    const markerIndexes = markerLineIndexes.filter((index) => index > block.start && index < block.end);
    const arxivIds: string[] = [];
    for (let index = block.start + 1; index < block.end; index += 1) {
      const match = /^[-*]\s+\*\*arXiv\*\*[:：]\s*\[(\d{4}\.\d{4,5})\]\(https:\/\/arxiv\.org\/abs\/\1\)$/.exec(lines[index] ?? "");
      if (match) arxivIds.push(match[1]!);
    }
    if (markerIndexes.length === 0) continue;
    if (markerIndexes.length !== 1) {
      return { kind: "invalid", reason: "personal novelty marker count is invalid" };
    }
    const markerIndex = markerIndexes[0]!;
    const firstSlot = block.start + 1;
    const provenanceAtFirstSlot = lines[firstSlot]?.startsWith(
      `<!-- ${DISCOVERY_PROVENANCE_MARKER_PREFIX}:`,
    ) ?? false;
    // Canonical layouts only. With a discovery-provenance marker the visible
    // provenance line follows it, so the novelty marker sits on the third line
    // (heading / provenance-marker / provenance-visible / novelty-marker);
    // without provenance the novelty marker sits directly after the heading.
    // Swapped or scattered marker families are misplacement, so hostile
    // metadata cannot reorder markers into a canonical-looking layout.
    const blockHasProvenanceMarkers = lines
      .slice(firstSlot, block.end)
      .some((line) => line.startsWith(`<!-- ${DISCOVERY_PROVENANCE_MARKER_PREFIX}:`));
    const expectedSlot = blockHasProvenanceMarkers
      ? (provenanceAtFirstSlot ? block.start + 3 : -1)
      : firstSlot;
    if (markerIndex !== expectedSlot) {
      return { kind: "invalid", reason: "personal novelty marker placement is invalid" };
    }
    if (arxivIds.length !== 1) return { kind: "invalid", reason: "marked paper identity is ambiguous" };
    const payload = parsePersonalNoveltyMarker(lines[markerIndex] ?? "");
    if (!payload) return { kind: "invalid", reason: "personal novelty marker is malformed" };
    const arxivId = arxivIds[0]!;
    if (payload.d !== reportDate || payload.a !== arxivId) {
      return { kind: "invalid", reason: "personal novelty marker identity does not match its report occurrence" };
    }
    if (seenIds.has(arxivId)) return { kind: "invalid", reason: "duplicate marked paper occurrence" };
    seenIds.add(arxivId);
    consumedMarkers.add(markerIndex);
    occurrences.push({ arxivId, novelty: markerPayloadNovelty(payload) });
  }
  if (consumedMarkers.size !== markerLineIndexes.length) {
    return { kind: "invalid", reason: "personal novelty marker is outside a canonical paper block" };
  }
  return { kind: "valid", occurrences };
}

function encodeBase64Url(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function decodeBase64Url(value: string): string {
  if (value.length % 4 === 1) throw new TypeError("invalid base64url");
  const padded = value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - value.length % 4) % 4);
  const binary = atob(padded);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
}

function isExactDataObject(value: unknown, keys: string[]): value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype: unknown = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) return false;
  const ownKeys = Reflect.ownKeys(value);
  if (ownKeys.length !== keys.length || !keys.every((key) => ownKeys.includes(key))) return false;
  for (const key of ownKeys) {
    if (typeof key !== "string") return false;
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !("value" in descriptor) || !descriptor.enumerable) return false;
  }
  return true;
}

function isCanonicalArxivId(value: unknown): value is string {
  return typeof value === "string" && /^\d{4}\.\d{4,5}$/.test(value);
}

function isReportDate(value: unknown): value is string {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value);
}
