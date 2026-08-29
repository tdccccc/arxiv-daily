import {
  PERSONALIZED_FILTER_MAX_DIRECTIONS,
  PERSONALIZED_FILTER_MAX_ID_LENGTH,
  PERSONALIZED_FILTER_MAX_NAME_LENGTH,
  PERSONALIZED_FILTER_MAX_REPRESENTATIVES,
  PERSONALIZED_FILTER_MAX_TITLE_CODE_UNITS,
  type PaperDiscoveryProvenance,
} from "./personalized-paper-filter";

export const DISCOVERY_PROVENANCE_MARKER_VERSION = 1 as const;
export const DISCOVERY_PROVENANCE_MARKER_PREFIX = "arxiv-daily-discovery-provenance";
export const DISCOVERY_PROVENANCE_MARKER_MAX_CODE_UNITS = 48_000 as const;
export const DISCOVERY_PROVENANCE_MAX_MANUAL_TOPICS = 256 as const;
export const DISCOVERY_PROVENANCE_MAX_TOPIC_LENGTH = 120 as const;

interface MarkerPayload {
  v: typeof DISCOVERY_PROVENANCE_MARKER_VERSION;
  d: string;
  id: string;
  p: PaperDiscoveryProvenance;
}

export interface PaperOccurrenceDiscoveryProvenance {
  arxivId: string;
  provenance: PaperDiscoveryProvenance;
}

export type DailyReportDiscoveryProvenanceParseResult =
  | { kind: "valid"; occurrences: PaperOccurrenceDiscoveryProvenance[] }
  | { kind: "invalid"; reason: string };

export function normalizePaperDiscoveryProvenance(value: unknown): PaperDiscoveryProvenance | null {
  if (!isExactDataObject(value, ["manualTopicTags", "directions"])) return null;
  if (!isCanonicalTextArray(
    value.manualTopicTags,
    DISCOVERY_PROVENANCE_MAX_MANUAL_TOPICS,
    DISCOVERY_PROVENANCE_MAX_TOPIC_LENGTH,
  )) return null;
  if (!isOrdinaryDataArray(value.directions, PERSONALIZED_FILTER_MAX_DIRECTIONS)) return null;

  const directions: PaperDiscoveryProvenance["directions"] = [];
  for (const raw of value.directions) {
    if (!isExactDataObject(raw, ["id", "name", "representatives"])
      || !isBoundedText(raw.id, PERSONALIZED_FILTER_MAX_ID_LENGTH)
      || !isBoundedText(raw.name, PERSONALIZED_FILTER_MAX_NAME_LENGTH)
      || !isOrdinaryDataArray(raw.representatives, PERSONALIZED_FILTER_MAX_REPRESENTATIVES)
      || raw.representatives.length < 1) return null;
    const representatives: PaperDiscoveryProvenance["directions"][number]["representatives"] = [];
    for (const representative of raw.representatives) {
      if (!isExactDataObject(representative, ["paperKey", "title", "evidenceDepth"])
        || !/^arxiv:\d{4}\.\d{4,5}$/.test(representative.paperKey)
        || !isBoundedText(representative.title, PERSONALIZED_FILTER_MAX_TITLE_CODE_UNITS)
        || representative.evidenceDepth !== "metadata-and-abstract") return null;
      representatives.push({
        paperKey: representative.paperKey,
        title: representative.title,
        evidenceDepth: "metadata-and-abstract",
      });
    }
    if (!strictlySortedUnique(representatives.map(({ paperKey }) => paperKey))) return null;
    directions.push({ id: raw.id, name: raw.name, representatives });
  }
  if (!strictlySortedUnique(directions.map(({ id }) => id))) return null;
  if (value.manualTopicTags.length === 0 && directions.length === 0) return null;
  return { manualTopicTags: [...value.manualTopicTags], directions };
}

export function renderDiscoveryProvenanceMarker(
  value: PaperDiscoveryProvenance,
  arxivId: string,
  reportDate: string,
): string {
  const provenance = normalizePaperDiscoveryProvenance(value);
  if (!provenance) throw new TypeError("discovery provenance is malformed");
  if (!isCanonicalArxivId(arxivId)) throw new TypeError("discovery provenance arXiv ID is malformed");
  if (!isReportDate(reportDate)) throw new TypeError("discovery provenance report date is malformed");
  const encoded = encodeBase64Url(JSON.stringify({
    v: DISCOVERY_PROVENANCE_MARKER_VERSION,
    d: reportDate,
    id: arxivId,
    p: provenance,
  } satisfies MarkerPayload));
  const marker = `<!-- ${DISCOVERY_PROVENANCE_MARKER_PREFIX}:v1:${encoded} -->`;
  if (marker.length > DISCOVERY_PROVENANCE_MARKER_MAX_CODE_UNITS) {
    throw new TypeError("discovery provenance marker is too large");
  }
  return marker;
}

export function parseDiscoveryProvenanceMarker(line: string): MarkerPayload | null {
  if (line.length > DISCOVERY_PROVENANCE_MARKER_MAX_CODE_UNITS) return null;
  const match = /^<!-- arxiv-daily-discovery-provenance:v1:([A-Za-z0-9_-]+) -->$/.exec(line);
  if (!match) return null;
  let payload: unknown;
  try {
    payload = JSON.parse(decodeBase64Url(match[1]!));
  } catch {
    return null;
  }
  if (!isExactDataObject(payload, ["v", "d", "id", "p"])
    || payload.v !== 1 || !isReportDate(payload.d) || !isCanonicalArxivId(payload.id)) return null;
  const provenance = normalizePaperDiscoveryProvenance(payload.p);
  if (!provenance) return null;
  const normalized: MarkerPayload = { v: 1, d: payload.d, id: payload.id, p: provenance };
  try {
    if (renderDiscoveryProvenanceMarker(provenance, payload.id, payload.d) !== line) return null;
  } catch {
    return null;
  }
  return normalized;
}

/**
 * Parses the complete report projection. User edits are authoritative like all
 * committed Markdown edits, but structurally inconsistent provenance never
 * mutates the derived index.
 */
export function parseDailyReportDiscoveryProvenance(
  markdown: string,
  reportDate: string,
): DailyReportDiscoveryProvenanceParseResult {
  if (!isReportDate(reportDate)) return { kind: "invalid", reason: "invalid report date" };
  const lines = markdown.split(/\r?\n/);
  const markerLineIndexes = lines.flatMap((line, index) =>
    line.startsWith(`<!-- ${DISCOVERY_PROVENANCE_MARKER_PREFIX}:`) ? [index] : []);
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

  const occurrences: PaperOccurrenceDiscoveryProvenance[] = [];
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
    if (markerIndexes.length !== 1 || markerIndexes[0] !== block.start + 1) {
      return { kind: "invalid", reason: "provenance marker placement or count is invalid" };
    }
    if (arxivIds.length !== 1) return { kind: "invalid", reason: "marked paper identity is ambiguous" };
    const markerIndex = markerIndexes[0]!;
    const payload = parseDiscoveryProvenanceMarker(lines[markerIndex] ?? "");
    if (!payload) return { kind: "invalid", reason: "provenance marker is malformed" };
    const arxivId = arxivIds[0]!;
    if (payload.d !== reportDate || payload.id !== arxivId) {
      return { kind: "invalid", reason: "provenance marker identity does not match its report occurrence" };
    }
    if (seenIds.has(arxivId)) return { kind: "invalid", reason: "duplicate marked paper occurrence" };
    seenIds.add(arxivId);
    consumedMarkers.add(markerIndex);
    occurrences.push({ arxivId, provenance: payload.p });
  }
  if (consumedMarkers.size !== markerLineIndexes.length) {
    return { kind: "invalid", reason: "provenance marker is outside a canonical paper block" };
  }
  return { kind: "valid", occurrences };
}

/** Escape untrusted provenance metadata to literal, single-line Markdown text. */
export function escapeDiscoveryProvenancePlainText(value: string): string {
  return value.replace(/\s+/gu, " ").trim().replace(/[\\`*_{}\[\]()<>#+\-.!|>]/g, "\\$&");
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

function isOrdinaryDataArray(value: unknown, maxItems: number): value is any[] {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype
    || value.length > maxItems) return false;
  const ownKeys = Reflect.ownKeys(value);
  const expected = [...Array.from({ length: value.length }, (_, index) => String(index)), "length"];
  if (ownKeys.length !== expected.length || !expected.every((key) => ownKeys.includes(key))) return false;
  for (let index = 0; index < value.length; index += 1) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !("value" in descriptor) || !descriptor.enumerable) return false;
  }
  const lengthDescriptor = Object.getOwnPropertyDescriptor(value, "length");
  return Boolean(lengthDescriptor && "value" in lengthDescriptor && !lengthDescriptor.enumerable);
}

function isBoundedText(value: unknown, max: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= max && value.trim() === value;
}

function isCanonicalTextArray(value: unknown, maxItems: number, maxLength: number): value is string[] {
  return isOrdinaryDataArray(value, maxItems)
    && value.every((entry) => isBoundedText(entry, maxLength)) && strictlySortedUnique(value);
}

function isCanonicalArxivId(value: unknown): value is string {
  return typeof value === "string" && /^\d{4}\.\d{4,5}$/.test(value);
}

function isReportDate(value: unknown): value is string {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value);
}

function strictlySortedUnique(values: string[]): boolean {
  for (let index = 1; index < values.length; index += 1) {
    if (values[index - 1]! >= values[index]!) return false;
  }
  return true;
}
