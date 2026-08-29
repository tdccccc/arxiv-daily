/**
 * Reading candidates: durable snapshots of discovered papers the researcher
 * chose to keep for later reading (ADR 0004 steps 7–8).
 *
 * A candidate preserves identity, discovery source (direction / manual topic
 * with the triggering daily report), related prior works, and provisional
 * novelty evidence, plus an optional reading decision. The document is a
 * bypass store in the same spirit as the incremental suggestions: sharded by
 * the personal library scope / identification fingerprints, deletable and
 * rebuildable, and never the authoritative record of durable research
 * knowledge. Decisions are persisted here so a later phase can feed them back
 * into discovery without touching the confirmed interest profile.
 *
 * Snapshot fields are user-visible plain text and strictly bounded at decode
 * time; the document enforces a pending-candidate capacity and evicts the
 * oldest undecided entry first.
 */

import { paperKeyFromArxivId } from "../../services/paper-key";
import {
  PERSONAL_NOVELTY_DIFFERENCE_TYPES,
  PERSONAL_NOVELTY_MAX_EXPLANATION_CODE_UNITS,
  type PersonalNoveltyDifferenceType,
} from "../../pipeline/personalized-novelty";

export const READING_CANDIDATES_SCHEMA_VERSION = 1 as const;
export const READING_CANDIDATES_MAX_PENDING = 500 as const;

export const READING_CANDIDATE_MAX_TITLE_CODE_UNITS = 500 as const;
export const READING_CANDIDATE_MAX_AUTHORS_CODE_UNITS = 500 as const;
export const READING_CANDIDATE_MAX_TOPIC_CODE_UNITS = 200 as const;
export const READING_CANDIDATE_MAX_NOTE_CODE_UNITS = 500 as const;
export const READING_CANDIDATE_MAX_DIRECTIONS = 20 as const;
export const READING_CANDIDATE_MAX_TOPICS = 20 as const;
export const READING_CANDIDATE_MAX_RELATED_PRIOR_WORKS = 50 as const;
export const READING_CANDIDATE_MAX_COMPARISON_BASIS = 50 as const;

export type ReadingCandidateDecisionKind = "read-closely" | "skim" | "dismiss";

const READING_CANDIDATE_DECISION_KINDS: readonly ReadingCandidateDecisionKind[] = [
  "read-closely",
  "skim",
  "dismiss",
] as const;

export interface ReadingCandidateSource {
  kind: "manual" | "library" | "both";
  manualTopics: Array<{ tag: string; name?: string }>;
  directions: Array<{ id: string; name: string }>;
  reportPath: string;
  reportDate: string;
}

export interface ReadingCandidateRecord {
  paperKey: string;
  arxivId: string;
  title: string;
  authors: string;
  topic: string;
  source: ReadingCandidateSource;
  relatedPriorWorks: Array<{ paperKey: string; title: string }>;
  provisionalNovelty?: {
    differenceType: PersonalNoveltyDifferenceType;
    comparisonBasis: string[];
    evidenceDepth: "metadata-and-abstract";
    explanation: string;
  };
  savedAt: string;
  updatedAt: string;
  decision?: { kind: ReadingCandidateDecisionKind; at: string; note?: string };
}

export interface ReadingCandidatesDocument {
  schemaVersion: typeof READING_CANDIDATES_SCHEMA_VERSION;
  revision: number;
  scopeFingerprint: string;
  identificationFingerprint: string;
  updatedAt: string;
  candidates: Record<string, ReadingCandidateRecord>;
}

/** Host-neutral snapshot of a dashboard row carrying discovery provenance. */
export interface ReadingCandidateRowSnapshot {
  paperKey: string;
  arxivId: string;
  title: string;
  authors: string;
  topic: string;
  occurrenceProvenance?: {
    reportPath: string;
    reportDate: string;
    source: "manual" | "library" | "both";
    manualTopics: Array<{ tag: string; name?: string }>;
    directions: Array<{
      id: string;
      name: string;
      representatives: Array<{ paperKey: string; title: string }>;
    }>;
  };
  personalNovelty?: {
    differenceType: PersonalNoveltyDifferenceType;
    comparisonBasis: string[];
    evidenceDepth: "metadata-and-abstract";
    explanation: string;
  };
}

/**
 * Build a reading candidate from a dashboard row snapshot. Rows without
 * discovery provenance have no discoverable source and yield null; the result
 * must pass the strict record decoder or null is returned.
 */
export function readingCandidateFromRowSnapshot(
  snapshot: ReadingCandidateRowSnapshot,
  savedAt: string,
): ReadingCandidateRecord | null {
  const provenance = snapshot.occurrenceProvenance;
  if (!provenance) return null;
  const seen = new Set<string>();
  const relatedPriorWorks: Array<{ paperKey: string; title: string }> = [];
  for (const direction of provenance.directions) {
    for (const representative of direction.representatives) {
      if (seen.has(representative.paperKey)) continue;
      seen.add(representative.paperKey);
      if (relatedPriorWorks.length >= READING_CANDIDATE_MAX_RELATED_PRIOR_WORKS) break;
      relatedPriorWorks.push({ paperKey: representative.paperKey, title: representative.title });
    }
    if (relatedPriorWorks.length >= READING_CANDIDATE_MAX_RELATED_PRIOR_WORKS) break;
  }
  const record: ReadingCandidateRecord = {
    paperKey: snapshot.paperKey,
    arxivId: snapshot.arxivId,
    title: snapshot.title,
    authors: snapshot.authors,
    topic: snapshot.topic,
    source: {
      kind: provenance.source,
      manualTopics: provenance.manualTopics,
      directions: provenance.directions.map((direction) => ({
        id: direction.id,
        name: direction.name,
      })),
      reportPath: provenance.reportPath,
      reportDate: provenance.reportDate,
    },
    relatedPriorWorks,
    ...(snapshot.personalNovelty
      ? { provisionalNovelty: { ...snapshot.personalNovelty } }
      : {}),
    savedAt,
    updatedAt: savedAt,
  };
  return decodeReadingCandidateRecord(record);
}

export function emptyReadingCandidatesDocument(
  scopeFingerprint: string,
  identificationFingerprint: string,
  updatedAt: string,
): ReadingCandidatesDocument {
  return {
    schemaVersion: READING_CANDIDATES_SCHEMA_VERSION,
    revision: 0,
    scopeFingerprint,
    identificationFingerprint,
    updatedAt,
    candidates: {},
  };
}

/** Upsert one candidate snapshot; existing decisions survive a re-save. */
export function upsertReadingCandidate(
  document: ReadingCandidatesDocument,
  record: ReadingCandidateRecord,
  updatedAt: string,
): { document: ReadingCandidatesDocument; changed: boolean; evicted: string[] } {
  const decoded = decodeReadingCandidateRecord(record);
  if (!decoded) throw new Error("invalid reading candidate record");
  const existing = document.candidates[decoded.paperKey];
  const next: ReadingCandidatesDocument = {
    ...document,
    updatedAt,
    candidates: { ...document.candidates },
  };
  if (existing) {
    const refreshed = {
      ...decoded,
      savedAt: existing.savedAt,
      updatedAt,
      decision: existing.decision,
    };
    next.candidates[decoded.paperKey] = refreshed;
    return { document: next, changed: true, evicted: [] };
  }
  const evicted: string[] = [];
  const pendingKeys = Object.entries(next.candidates)
    .filter(([, value]) => !value.decision)
    .sort((left, right) => {
      const bySaved = left[1].savedAt.localeCompare(right[1].savedAt);
      return bySaved !== 0 ? bySaved : left[0].localeCompare(right[0]);
    });
  while (pendingKeys.length >= READING_CANDIDATES_MAX_PENDING) {
    const oldest = pendingKeys.shift();
    if (!oldest) break;
    delete next.candidates[oldest[0]];
    evicted.push(oldest[0]);
  }
  next.candidates[decoded.paperKey] = { ...decoded, updatedAt };
  return { document: next, changed: true, evicted };
}

/** Set or replace the reading decision for one candidate. */
export function decideReadingCandidate(
  document: ReadingCandidatesDocument,
  paperKey: string,
  kind: ReadingCandidateDecisionKind,
  at: string,
  note?: string,
): { document: ReadingCandidatesDocument; changed: boolean } {
  const existing = document.candidates[paperKey];
  if (!existing) return { document, changed: false };
  const trimmed = (note ?? "").trim();
  const next: ReadingCandidatesDocument = {
    ...document,
    updatedAt: at,
    candidates: { ...document.candidates },
  };
  next.candidates[paperKey] = {
    ...existing,
    updatedAt: at,
    decision: {
      kind,
      at,
      ...(trimmed ? { note: trimmed.slice(0, READING_CANDIDATE_MAX_NOTE_CODE_UNITS) } : {}),
    },
  };
  return { document: next, changed: true };
}

export function removeReadingCandidate(
  document: ReadingCandidatesDocument,
  paperKey: string,
  updatedAt: string,
): { document: ReadingCandidatesDocument; changed: boolean } {
  if (!Object.prototype.hasOwnProperty.call(document.candidates, paperKey)) {
    return { document, changed: false };
  }
  const candidates = { ...document.candidates };
  delete candidates[paperKey];
  return { document: { ...document, updatedAt, candidates }, changed: true };
}

/* ------------------------------------------------------------------ decode */

export function decodeReadingCandidateRecord(value: unknown): ReadingCandidateRecord | null {
  if (!isPlainObject(value)) return null;
  if (!hasAllowedKeys(value, [
    "paperKey", "arxivId", "title", "authors", "topic", "source",
    "relatedPriorWorks", "savedAt", "updatedAt",
  ], [
    "paperKey", "arxivId", "title", "authors", "topic", "source",
    "relatedPriorWorks", "provisionalNovelty", "savedAt", "updatedAt", "decision",
  ])) return null;
  if (typeof value.paperKey !== "string" || typeof value.arxivId !== "string") return null;
  let expectedKey: string;
  try {
    expectedKey = paperKeyFromArxivId(value.arxivId);
  } catch {
    return null;
  }
  if (value.paperKey !== expectedKey) return null;
  if (!isBoundedText(value.title, 1, READING_CANDIDATE_MAX_TITLE_CODE_UNITS)) return null;
  if (!isBoundedText(value.authors, 0, READING_CANDIDATE_MAX_AUTHORS_CODE_UNITS)) return null;
  if (!isBoundedText(value.topic, 0, READING_CANDIDATE_MAX_TOPIC_CODE_UNITS)) return null;
  if (!isIsoTimestamp(value.savedAt) || !isIsoTimestamp(value.updatedAt)) return null;
  const source = decodeReadingCandidateSource(value.source);
  if (!source) return null;
  const relatedPriorWorks = decodePriorWorks(value.relatedPriorWorks);
  if (!relatedPriorWorks) return null;
  const decision = value.decision === undefined ? undefined : decodeDecision(value.decision);
  if (value.decision !== undefined && !decision) return null;
  if (value.provisionalNovelty !== undefined) {
    const novelty = decodeProvisionalNovelty(value.provisionalNovelty);
    if (!novelty) return null;
    return {
      paperKey: value.paperKey,
      arxivId: value.arxivId,
      title: value.title,
      authors: value.authors,
      topic: value.topic,
      source,
      relatedPriorWorks,
      provisionalNovelty: novelty,
      savedAt: value.savedAt,
      updatedAt: value.updatedAt,
      ...(decision ? { decision } : {}),
    };
  }
  return {
    paperKey: value.paperKey,
    arxivId: value.arxivId,
    title: value.title,
    authors: value.authors,
    topic: value.topic,
    source,
    relatedPriorWorks,
    savedAt: value.savedAt,
    updatedAt: value.updatedAt,
    ...(decision ? { decision } : {}),
  };
}

export function decodeReadingCandidatesDocument(value: unknown): ReadingCandidatesDocument | null {
  if (!isPlainObject(value)) return null;
  if (!isExactObject(value, [
    "schemaVersion", "revision", "scopeFingerprint", "identificationFingerprint",
    "updatedAt", "candidates",
  ])) return null;
  if (value.schemaVersion !== READING_CANDIDATES_SCHEMA_VERSION) return null;
  if (!Number.isInteger(value.revision) || (value.revision as number) < 0) return null;
  if (typeof value.scopeFingerprint !== "string" || !value.scopeFingerprint) return null;
  if (typeof value.identificationFingerprint !== "string" || !value.identificationFingerprint) return null;
  if (!isIsoTimestamp(value.updatedAt)) return null;
  if (!isPlainObject(value.candidates)) return null;
  const candidates: Record<string, ReadingCandidateRecord> = {};
  for (const [paperKey, raw] of Object.entries(value.candidates as Record<string, unknown>)) {
    const record = decodeReadingCandidateRecord(raw);
    if (!record || record.paperKey !== paperKey) return null;
    candidates[paperKey] = record;
  }
  return {
    schemaVersion: READING_CANDIDATES_SCHEMA_VERSION,
    revision: value.revision as number,
    scopeFingerprint: value.scopeFingerprint,
    identificationFingerprint: value.identificationFingerprint,
    updatedAt: value.updatedAt,
    candidates,
  };
}

function decodeReadingCandidateSource(value: unknown): ReadingCandidateSource | null {
  if (!isPlainObject(value)) return null;
  if (!isExactObject(value, ["kind", "manualTopics", "directions", "reportPath", "reportDate"])) return null;
  if (value.kind !== "manual" && value.kind !== "library" && value.kind !== "both") return null;
  if (!Array.isArray(value.manualTopics) || value.manualTopics.length > READING_CANDIDATE_MAX_TOPICS) return null;
  const manualTopics: Array<{ tag: string; name?: string }> = [];
  for (const raw of value.manualTopics) {
    if (!isPlainObject(raw) || !hasAllowedKeys(raw, ["tag"], ["tag", "name"])) return null;
    if (!isBoundedText(raw.tag, 1, READING_CANDIDATE_MAX_TOPIC_CODE_UNITS)) return null;
    if (raw.name !== undefined && !isBoundedText(raw.name, 0, READING_CANDIDATE_MAX_TOPIC_CODE_UNITS)) return null;
    manualTopics.push(raw.name === undefined ? { tag: raw.tag } : { tag: raw.tag, name: raw.name });
  }
  if (!Array.isArray(value.directions) || value.directions.length > READING_CANDIDATE_MAX_DIRECTIONS) return null;
  const directions: Array<{ id: string; name: string }> = [];
  for (const raw of value.directions) {
    if (!isPlainObject(raw) || !isExactObject(raw, ["id", "name"])) return null;
    if (!isBoundedText(raw.id, 1, 200) || !isBoundedText(raw.name, 1, READING_CANDIDATE_MAX_TOPIC_CODE_UNITS)) return null;
    directions.push({ id: raw.id, name: raw.name });
  }
  if (!isBoundedText(value.reportPath, 1, 1_000)) return null;
  if (typeof value.reportDate !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value.reportDate)) return null;
  return { kind: value.kind, manualTopics, directions, reportPath: value.reportPath, reportDate: value.reportDate };
}

function decodePriorWorks(value: unknown): Array<{ paperKey: string; title: string }> | null {
  if (!Array.isArray(value) || value.length > READING_CANDIDATE_MAX_RELATED_PRIOR_WORKS) return null;
  const out: Array<{ paperKey: string; title: string }> = [];
  for (const raw of value) {
    if (!isPlainObject(raw) || !isExactObject(raw, ["paperKey", "title"])) return null;
    if (!isPaperKey(raw.paperKey)) return null;
    if (!isBoundedText(raw.title, 0, READING_CANDIDATE_MAX_TITLE_CODE_UNITS)) return null;
    out.push({ paperKey: raw.paperKey, title: raw.title });
  }
  return out;
}

function decodeProvisionalNovelty(value: unknown): NonNullable<ReadingCandidateRecord["provisionalNovelty"]> | null {
  if (!isPlainObject(value)) return null;
  if (!isExactObject(value, ["differenceType", "comparisonBasis", "evidenceDepth", "explanation"])) return null;
  if (!PERSONAL_NOVELTY_DIFFERENCE_TYPES.includes(value.differenceType as PersonalNoveltyDifferenceType)) return null;
  if (value.evidenceDepth !== "metadata-and-abstract") return null;
  if (!Array.isArray(value.comparisonBasis) || value.comparisonBasis.length > READING_CANDIDATE_MAX_COMPARISON_BASIS) return null;
  const comparisonBasis: string[] = [];
  for (const raw of value.comparisonBasis) {
    if (!isPaperKey(raw)) return null;
    if (comparisonBasis.includes(raw)) return null;
    comparisonBasis.push(raw);
  }
  if (!isBoundedText(value.explanation, 1, PERSONAL_NOVELTY_MAX_EXPLANATION_CODE_UNITS)) return null;
  return {
    differenceType: value.differenceType as PersonalNoveltyDifferenceType,
    comparisonBasis,
    evidenceDepth: "metadata-and-abstract",
    explanation: value.explanation,
  };
}

function decodeDecision(value: unknown): { kind: ReadingCandidateDecisionKind; at: string; note?: string } | null {
  if (!isPlainObject(value)) return null;
  if (!hasAllowedKeys(value, ["kind", "at"], ["kind", "at", "note"])) return null;
  if (!READING_CANDIDATE_DECISION_KINDS.includes(value.kind as ReadingCandidateDecisionKind)) return null;
  if (!isIsoTimestamp(value.at)) return null;
  if (value.note !== undefined && !isBoundedText(value.note, 0, READING_CANDIDATE_MAX_NOTE_CODE_UNITS)) return null;
  return value.note === undefined
    ? { kind: value.kind as ReadingCandidateDecisionKind, at: value.at }
    : { kind: value.kind as ReadingCandidateDecisionKind, at: value.at, note: value.note };
}

/* ----------------------------------------------------------------- helpers */

function isPaperKey(value: unknown): value is string {
  return typeof value === "string" && /^arxiv:\d{4}\.\d{4,5}$/.test(value);
}

function isBoundedText(value: unknown, min: number, max: number): value is string {
  return typeof value === "string" && value.trim().length >= min && value.length <= max;
}

function isIsoTimestamp(value: unknown): value is string {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?Z$/.test(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isExactObject(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every((key) => actual.includes(key));
}

function hasAllowedKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  allowed: readonly string[],
): boolean {
  const actual = Object.keys(value);
  return required.every((key) => actual.includes(key))
    && actual.every((key) => allowed.includes(key));
}
