import { DISCOVERY_PROVENANCE_MARKER_PREFIX } from "./discovery-provenance-marker";
import {
  normalizePersonalNovelty,
  normalizePersonalNoveltyForMarker,
  type PersonalNovelty,
} from "./personalized-novelty";

export const PERSONAL_NOVELTY_MARKER_VERSION = 1 as const;
export const PERSONAL_NOVELTY_MARKER_PREFIX = "arxiv-daily-personal-novelty";
/**
 * Deterministic marker bound: the worst-case canonical payload keeps the
 * bounded explanation (at most PERSONAL_NOVELTY_MAX_EXPLANATION_CODE_UNITS
 * code units, inflating at most 6x under JSON escaping) plus the bounded
 * comparison basis (at most 40 canonical paperKeys) and fixed scaffolding;
 * base64url inflates by 4/3. The resulting worst-case line stays well under
 * 10_000 code units, so every valid marker renders and parses within bounds.
 */
export const PERSONAL_NOVELTY_MARKER_MAX_CODE_UNITS = 10_000 as const;

interface MarkerPayload {
  v: typeof PERSONAL_NOVELTY_MARKER_VERSION;
  d: string;
  a: string;
  n: PersonalNovelty;
}

export interface PaperOccurrencePersonalNovelty {
  arxivId: string;
  novelty: PersonalNovelty;
}

export type DailyReportPersonalNoveltyParseResult =
  | { kind: "valid"; occurrences: PaperOccurrencePersonalNovelty[] }
  | { kind: "invalid"; reason: string };

export function renderPersonalNoveltyMarker(
  novelty: PersonalNovelty,
  arxivId: string,
  reportDate: string,
): string {
  const normalized = normalizePersonalNoveltyForMarker(novelty);
  if (!normalized) throw new TypeError("personal novelty is malformed");
  if (!isCanonicalArxivId(arxivId)) throw new TypeError("personal novelty arXiv ID is malformed");
  if (!isReportDate(reportDate)) throw new TypeError("personal novelty report date is malformed");
  const encoded = encodeBase64Url(JSON.stringify({
    v: PERSONAL_NOVELTY_MARKER_VERSION,
    d: reportDate,
    a: arxivId,
    n: normalized,
  } satisfies MarkerPayload));
  const marker = `<!-- ${PERSONAL_NOVELTY_MARKER_PREFIX}:v1:${encoded} -->`;
  if (marker.length > PERSONAL_NOVELTY_MARKER_MAX_CODE_UNITS) {
    throw new TypeError("personal novelty marker is too large");
  }
  return marker;
}

export function parsePersonalNoveltyMarker(line: string): MarkerPayload | null {
  if (line.length > PERSONAL_NOVELTY_MARKER_MAX_CODE_UNITS) return null;
  const match = /^<!-- arxiv-daily-personal-novelty:v1:([A-Za-z0-9_-]+) -->$/.exec(line);
  if (!match) return null;
  let payload: unknown;
  try {
    payload = JSON.parse(decodeBase64Url(match[1]!));
  } catch {
    return null;
  }
  if (!isExactDataObject(payload, ["v", "d", "a", "n"])
    || payload.v !== 1 || !isReportDate(payload.d) || !isCanonicalArxivId(payload.a)) return null;
  const novelty = normalizePersonalNovelty(payload.n);
  if (!novelty) return null;
  const normalized: MarkerPayload = { v: 1, d: payload.d, a: payload.a, n: novelty };
  try {
    if (renderPersonalNoveltyMarker(novelty, payload.a, payload.d) !== line) return null;
  } catch {
    return null;
  }
  return normalized;
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
    occurrences.push({ arxivId, novelty: payload.n });
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

function isExactDataObject(value: unknown, keys: string[]): value is Record<string, any> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
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
