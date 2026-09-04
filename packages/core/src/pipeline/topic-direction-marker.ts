import { DISCOVERY_PROVENANCE_MARKER_PREFIX } from "./discovery-provenance-marker";
import { PERSONAL_NOVELTY_MARKER_PREFIX } from "./personal-novelty-marker";
import {
  normalizeTopicDirectionHits,
  type TopicDirectionHit,
} from "./topic-direction-hits";

export const TOPIC_DIRECTION_MARKER_VERSION = 1 as const;
export const TOPIC_DIRECTION_MARKER_PREFIX = "arxiv-daily-topic-directions";
export const TOPIC_DIRECTION_MARKER_MAX_CODE_UNITS = 48_000 as const;

interface MarkerPayload {
  v: typeof TOPIC_DIRECTION_MARKER_VERSION;
  d: string;
  id: string;
  h: TopicDirectionHit[];
}

export interface PaperOccurrenceTopicDirections {
  arxivId: string;
  hits: TopicDirectionHit[];
}

export type DailyReportTopicDirectionParseResult =
  | { kind: "valid"; occurrences: PaperOccurrenceTopicDirections[] }
  | { kind: "invalid"; reason: string };

export function renderTopicDirectionMarker(
  hits: readonly TopicDirectionHit[],
  arxivId: string,
  reportDate: string,
): string {
  const normalized = normalizeTopicDirectionHits(hits);
  if (!normalized) throw new TypeError("topic directions are malformed");
  if (!isCanonicalArxivId(arxivId)) throw new TypeError("topic direction arXiv ID is malformed");
  if (!isReportDate(reportDate)) throw new TypeError("topic direction report date is malformed");
  const encoded = encodeBase64Url(JSON.stringify({
    v: TOPIC_DIRECTION_MARKER_VERSION,
    d: reportDate,
    id: arxivId,
    h: normalized,
  } satisfies MarkerPayload));
  const marker = `<!-- ${TOPIC_DIRECTION_MARKER_PREFIX}:v1:${encoded} -->`;
  if (marker.length > TOPIC_DIRECTION_MARKER_MAX_CODE_UNITS) {
    throw new TypeError("topic direction marker is too large");
  }
  return marker;
}

export function parseTopicDirectionMarker(line: string): MarkerPayload | null {
  if (line.length > TOPIC_DIRECTION_MARKER_MAX_CODE_UNITS) return null;
  const match = new RegExp(`^<!-- ${TOPIC_DIRECTION_MARKER_PREFIX}:v1:([A-Za-z0-9_-]+) -->$`)
    .exec(line);
  if (!match) return null;
  let payload: unknown;
  try {
    payload = JSON.parse(decodeBase64Url(match[1]!));
  } catch {
    return null;
  }
  if (!isExactDataObject(payload, ["v", "d", "id", "h"])
    || payload.v !== 1 || !isReportDate(payload.d) || !isCanonicalArxivId(payload.id)) return null;
  const hits = normalizeTopicDirectionHits(payload.h);
  if (!hits) return null;
  const normalized: MarkerPayload = { v: 1, d: payload.d, id: payload.id, h: hits };
  try {
    if (renderTopicDirectionMarker(hits, payload.id, payload.d) !== line) return null;
  } catch {
    return null;
  }
  return normalized;
}

/**
 * Parse the whole report's topic-direction markers.
 *
 * The marker family sits last among the three, after discovery provenance and
 * personal novelty, so adding it leaves both of those parsers' canonical slot
 * arithmetic untouched. Its own expected slot is therefore the first line of
 * the block plus two lines for each preceding family that is present.
 */
export function parseDailyReportTopicDirections(
  markdown: string,
  reportDate: string,
): DailyReportTopicDirectionParseResult {
  if (!isReportDate(reportDate)) return { kind: "invalid", reason: "invalid report date" };
  const lines = markdown.split(/\r?\n/);
  const markerLineIndexes = lines.flatMap((line, index) =>
    line.startsWith(`<!-- ${TOPIC_DIRECTION_MARKER_PREFIX}:`) ? [index] : []);
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

  const occurrences: PaperOccurrenceTopicDirections[] = [];
  const seenIds = new Set<string>();
  const consumedMarkers = new Set<number>();
  for (const block of blocks) {
    const markerIndexes = markerLineIndexes
      .filter((index) => index > block.start && index < block.end);
    if (markerIndexes.length === 0) continue;
    if (markerIndexes.length !== 1) {
      return { kind: "invalid", reason: "topic direction marker count is invalid" };
    }
    const preceding = [DISCOVERY_PROVENANCE_MARKER_PREFIX, PERSONAL_NOVELTY_MARKER_PREFIX]
      .filter((prefix) => lines.slice(block.start + 1, block.end)
        .some((line) => line.startsWith(`<!-- ${prefix}:`)));
    const markerIndex = markerIndexes[0]!;
    if (markerIndex !== block.start + 1 + preceding.length * 2) {
      return { kind: "invalid", reason: "topic direction marker placement is invalid" };
    }
    const arxivIds: string[] = [];
    for (let index = block.start + 1; index < block.end; index += 1) {
      const match = /^[-*]\s+\*\*arXiv\*\*[:：]\s*\[(\d{4}\.\d{4,5})\]\(https:\/\/arxiv\.org\/abs\/\1\)$/
        .exec(lines[index] ?? "");
      if (match) arxivIds.push(match[1]!);
    }
    if (arxivIds.length !== 1) return { kind: "invalid", reason: "marked paper identity is ambiguous" };
    const payload = parseTopicDirectionMarker(lines[markerIndex] ?? "");
    if (!payload) return { kind: "invalid", reason: "topic direction marker is malformed" };
    const arxivId = arxivIds[0]!;
    if (payload.d !== reportDate || payload.id !== arxivId) {
      return {
        kind: "invalid",
        reason: "topic direction marker identity does not match its report occurrence",
      };
    }
    if (seenIds.has(arxivId)) return { kind: "invalid", reason: "duplicate marked paper occurrence" };
    seenIds.add(arxivId);
    consumedMarkers.add(markerIndex);
    occurrences.push({ arxivId, hits: payload.h });
  }
  if (consumedMarkers.size !== markerLineIndexes.length) {
    return { kind: "invalid", reason: "topic direction marker is outside a canonical paper block" };
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
