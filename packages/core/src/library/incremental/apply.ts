/**
 * Applying persisted direction diff suggestions to the confirmed interest
 * profile. Style follows the review transaction flow: exact input objects,
 * strict document decode, monotonic timestamps, and a strict decode guard on
 * every produced profile. No input is mutated: the profile is decoded into a
 * fresh value first, and every produced profile is re-decoded before being
 * returned.
 *
 * - attach: papers join one existing direction's clusterMembers at the fixed
 *   membership confidence (0.9); papers may never join twice anywhere in the
 *   profile. Locked directions may accept attachments but never splits or
 *   merges.
 * - new: only produces a candidate draft for the existing confirmation flow;
 *   no direction is created here.
 * - split: papers leave the source direction (which must be terminal,
 *   unlocked, and hold those members) and a new derived direction is created
 *   with an id supplied by the caller.
 * - merge: mirrors the review merge semantics for two terminal unlocked
 *   directions, with the merged direction's text derived from the suggestion
 *   reason.
 */

import { sha256Hex } from "../../utils/digest";
import {
  PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS,
  PERSONAL_LIBRARY_MAX_DIRECTION_ANCESTRY_IDS,
  PERSONAL_LIBRARY_MAX_DIRECTIONS,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH,
  PERSONAL_LIBRARY_MAX_ID_LENGTH,
  PERSONAL_LIBRARY_MAX_NAME_LENGTH,
  PERSONAL_LIBRARY_MAX_PROFILE_ANCESTRY_IDS,
  PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS,
  PERSONAL_LIBRARY_MAX_REPRESENTATIVES,
  PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS,
  createPersonalLibraryRepresentativeSetFingerprint,
  decodePersonalLibraryInterestProfile,
  type PersonalLibraryClusterMember,
  type PersonalLibraryConfirmedDirection,
  type PersonalLibraryDirectionTimelineEvent,
  type PersonalLibraryInterestProfile,
  type PersonalLibraryRepresentativeEvidence,
} from "../personal-library-interest-profile";
import type { DirectionDiffSuggestion } from "./diff-suggestions";
import { decodeIncrementalSuggestion } from "./suggestions-store";

/** Fixed membership confidence applied when suggestions bring papers into a direction. */
export const SUGGESTION_MEMBER_CONFIDENCE = 0.9 as const;

/**
 * The split-derived direction's proposal lineage is a fixed marker: the strict
 * profile decoder requires non-empty proposalIds, and the direction was not
 * confirmed from any real proposal. The marker is inert to confirmation and
 * eligibility logic (no real proposal id can match a direction whose
 * candidateIds are empty).
 */
export const SPLIT_DERIVED_PROPOSAL_MARKER = "split-derived" as const;

export interface NewDirectionDraft {
  name: string;
  description: string;
  discoveryCues: string[];
  representativePaperKeys: string[];
  clusterMembers: PersonalLibraryClusterMember[];
}

export type IncrementalSuggestionsApplyErrorCode =
  | "invalid-input"
  | "invalid-document"
  | "not-found"
  | "conflict"
  | "direction-limit"
  | "lineage-limit";

export class IncrementalSuggestionsApplyError extends Error {
  constructor(
    message: string,
    readonly code: IncrementalSuggestionsApplyErrorCode,
    readonly details: Readonly<Record<string, unknown>> = {},
  ) {
    super(message);
    this.name = "IncrementalSuggestionsApplyError";
  }
}

export function applyAttachSuggestion(input: unknown): PersonalLibraryInterestProfile {
  const raw = exactInput(input, ["profile", "suggestion", "now"]);
  const profile = profileDocument(raw.profile);
  const suggestion = suggestionInput(raw.suggestion, "attach");
  const timestamp = canonicalDate(raw.now);
  const index = profile.directions.findIndex(({ id }) => id === suggestion.directionId);
  if (index < 0) fail("not-found", "direction was not found", { directionId: suggestion.directionId });
  const direction = profile.directions[index]!;
  if (direction.status === "merged") {
    fail("conflict", "merged directions cannot receive attachments", { directionId: direction.id });
  }
  // Papers may never be members of two directions (or twice in one).
  const memberKeys = new Set(
    profile.directions.flatMap(({ clusterMembers }) => clusterMembers.map(({ paperKey }) => paperKey)),
  );
  const duplicate = suggestion.paperKeys.find((paperKey) => memberKeys.has(paperKey));
  if (duplicate !== undefined) {
    fail("conflict", "papers are already members of a confirmed direction", { paperKey: duplicate });
  }
  const members = [
    ...direction.clusterMembers.map((member) => ({ ...member })),
    ...suggestion.paperKeys.map((paperKey) => ({ paperKey, confidence: SUGGESTION_MEMBER_CONFIDENCE })),
  ].sort(byPaperKey);
  const updatedAt = monotonicTimestamp(timestamp, direction.updatedAt);
  profile.directions[index] = {
    ...direction,
    clusterMembers: members,
    updatedAt,
    timeline: appendTimelineEvent(direction.timeline, { kind: "members-updated", at: updatedAt }),
  };
  return outputProfile(profile);
}

/**
 * Pure draft builder for a "new" suggestion: the direction is not created
 * here, the draft feeds the existing candidate confirmation flow instead.
 */
export function buildNewDirectionDraft(suggestion: DirectionDiffSuggestion): NewDirectionDraft {
  const decoded = decodeIncrementalSuggestion(suggestion);
  if (!decoded || decoded.kind !== "new") {
    fail("invalid-input", "suggestion must be a strictly valid new suggestion");
  }
  return {
    name: derivedName(decoded.reason),
    description: decoded.reason,
    discoveryCues: [],
    representativePaperKeys: decoded.paperKeys.slice(0, PERSONAL_LIBRARY_MAX_REPRESENTATIVES),
    clusterMembers: decoded.paperKeys.map((paperKey) => ({
      paperKey,
      confidence: SUGGESTION_MEMBER_CONFIDENCE,
    })),
  };
}

export function applySplitSuggestion(input: unknown): {
  profile: PersonalLibraryInterestProfile;
  newDirectionId: string;
} {
  const raw = exactInput(input, ["profile", "suggestion", "createId", "now"]);
  const profile = profileDocument(raw.profile);
  const suggestion = suggestionInput(raw.suggestion, "split");
  const createId = idFactory(raw.createId);
  const timestamp = canonicalDate(raw.now);
  const index = profile.directions.findIndex(({ id }) => id === suggestion.directionId);
  if (index < 0) fail("not-found", "direction was not found", { directionId: suggestion.directionId });
  const source = profile.directions[index]!;
  if (source.status === "merged") {
    fail("conflict", "merged directions do not participate in automatic splits", { directionId: source.id });
  }
  if (source.lockedAt !== undefined) {
    fail("conflict", "locked directions do not participate in automatic splits", { directionId: source.id });
  }
  const memberKeys = new Set(source.clusterMembers.map(({ paperKey }) => paperKey));
  const missing = suggestion.paperKeys.find((paperKey) => !memberKeys.has(paperKey));
  if (missing !== undefined) {
    fail("conflict", "split papers are not members of the direction", {
      directionId: source.id, paperKey: missing,
    });
  }
  const newDirectionId = createId("split");
  if (profile.directions.some(({ id }) => id === newDirectionId)) {
    fail("conflict", "directionId already exists", { directionId: newDirectionId });
  }
  if (profile.directions.length >= PERSONAL_LIBRARY_MAX_DIRECTIONS) {
    fail("direction-limit", "confirmed direction limit would be exceeded", { maximum: PERSONAL_LIBRARY_MAX_DIRECTIONS });
  }
  const removed = new Set(suggestion.paperKeys);
  const remaining = source.clusterMembers
    .filter(({ paperKey }) => !removed.has(paperKey))
    .sort(byPaperKey);
  const updatedAt = monotonicTimestamp(timestamp, source.updatedAt);
  profile.directions[index] = {
    ...source,
    clusterMembers: remaining,
    updatedAt,
    timeline: appendTimelineEvent(source.timeline, { kind: "split", at: updatedAt, sourceDirectionId: source.id }),
  };
  const derived = derivedDirection({
    id: newDirectionId,
    reason: suggestion.reason,
    paperKeys: suggestion.paperKeys,
    evidencePool: source.representatives,
    timestamp,
    timeline: [{ kind: "created", at: timestamp }],
    lineage: { proposalIds: [SPLIT_DERIVED_PROPOSAL_MARKER], candidateIds: [], directionIds: [] },
  });
  profile.directions.push(derived);
  profile.directions.sort(byId);
  return { profile: outputProfile(profile), newDirectionId };
}

export function applyMergeSuggestion(input: unknown): PersonalLibraryInterestProfile {
  const raw = exactInput(input, ["profile", "suggestion", "createId", "now"]);
  const profile = profileDocument(raw.profile);
  const suggestion = suggestionInput(raw.suggestion, "merge");
  const createId = idFactory(raw.createId);
  const timestamp = canonicalDate(raw.now);
  const sourceIds = suggestion.directionIds;
  const sources = sourceIds.map((id) => profile.directions.find((direction) => direction.id === id)
    ?? fail("not-found", "source direction was not found", { directionId: id }));
  if (sources.some(({ status }) => status === "merged")) {
    fail("conflict", "only terminal active or disabled directions can be merged");
  }
  if (sources.some(({ lockedAt }) => lockedAt !== undefined)) {
    fail("conflict", "locked directions do not participate in automatic merges");
  }
  const directionId = createId("merge");
  if (profile.directions.some(({ id }) => id === directionId)) {
    fail("conflict", "directionId already exists", { directionId });
  }
  if (profile.directions.length >= PERSONAL_LIBRARY_MAX_DIRECTIONS) {
    fail("direction-limit", "confirmed direction limit would be exceeded", { maximum: PERSONAL_LIBRARY_MAX_DIRECTIONS });
  }
  // Lineage semantics mirror mergePersonalLibraryConfirmedDirections.
  const proposalIds = canonicalUnion(...sources.map(({ lineage }) => lineage.proposalIds));
  const candidateIds = canonicalUnion(...sources.map(({ lineage }) => lineage.candidateIds));
  const directionIds = canonicalUnion(sourceIds, ...sources.map(({ lineage }) => lineage.directionIds));
  if (proposalIds.length > PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS) lineageLimit("proposalIds", proposalIds.length);
  if (candidateIds.length > PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS) lineageLimit("candidateIds", candidateIds.length);
  if (directionIds.length > PERSONAL_LIBRARY_MAX_DIRECTION_ANCESTRY_IDS) lineageLimit("directionIds", directionIds.length);
  const existingAncestry = profile.directions.reduce((total, direction) => (
    total + direction.lineage.directionIds.length
  ), 0);
  if (existingAncestry + directionIds.length > PERSONAL_LIBRARY_MAX_PROFILE_ANCESTRY_IDS) {
    lineageLimit("profileDirectionIds", existingAncestry + directionIds.length);
  }
  const targetUpdatedAt = monotonicTimestamp(timestamp, ...sources.map(({ updatedAt }) => updatedAt));
  const representatives = mergedRepresentatives(sources);
  const sourceSet = new Set(sourceIds);
  profile.directions = profile.directions.map((direction) => sourceSet.has(direction.id) ? {
    ...direction,
    status: "merged" as const,
    mergedIntoDirectionId: directionId,
    updatedAt: monotonicTimestamp(timestamp, direction.updatedAt),
  } : direction);
  profile.directions.push({
    id: directionId,
    status: "active",
    name: derivedName(suggestion.reason),
    description: suggestion.reason,
    discoveryCues: derivedCues(suggestion.reason),
    representatives,
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    // Mirrors the review merge transaction: the merged direction is a fresh
    // identity whose membership is re-established by later review activity.
    clusterMembers: [],
    timeline: [
      { kind: "created", at: timestamp },
      { kind: "merged", at: targetUpdatedAt, sourceDirectionIds: [...sourceIds] },
    ],
    lineage: { proposalIds, candidateIds, directionIds },
    createdAt: timestamp,
    updatedAt: targetUpdatedAt,
  });
  profile.directions.sort(byId);
  return outputProfile(profile);
}

function derivedDirection(input: {
  id: string;
  reason: string;
  paperKeys: readonly string[];
  evidencePool: readonly PersonalLibraryRepresentativeEvidence[];
  timestamp: string;
  timeline: PersonalLibraryDirectionTimelineEvent[];
  lineage: { proposalIds: string[]; candidateIds: string[]; directionIds: string[] };
}): PersonalLibraryConfirmedDirection {
  const representatives = representativesForPaperKeys(input.paperKeys, input.evidencePool);
  return {
    id: input.id,
    status: "active",
    name: derivedName(input.reason),
    description: input.reason,
    discoveryCues: derivedCues(input.reason),
    representatives,
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    clusterMembers: input.paperKeys.map((paperKey) => ({
      paperKey,
      confidence: SUGGESTION_MEMBER_CONFIDENCE,
    })),
    timeline: input.timeline,
    lineage: input.lineage,
    createdAt: input.timestamp,
    updatedAt: input.timestamp,
  };
}

function derivedName(reason: string): string {
  return reason.slice(0, PERSONAL_LIBRARY_MAX_NAME_LENGTH) || "New direction";
}

function derivedCues(reason: string): string[] {
  // The strict profile decoder requires at least one discovery cue; the
  // reason-derived cue is a draft placeholder for review.
  return [reason.slice(0, PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH)];
}

function representativesForPaperKeys(
  paperKeys: readonly string[],
  evidencePool: readonly PersonalLibraryRepresentativeEvidence[],
): PersonalLibraryRepresentativeEvidence[] {
  const byKey = new Map(evidencePool.map((entry) => [entry.paperKey, entry]));
  return paperKeys.slice(0, PERSONAL_LIBRARY_MAX_REPRESENTATIVES).map((paperKey) => ({
    ...(byKey.get(paperKey) ?? { paperKey, evidenceFingerprint: placeholderEvidenceFingerprint(paperKey) }),
  }));
}

/**
 * Split/new derived directions carry no catalog evidence yet. The placeholder
 * fingerprint is deterministic per paper key and is replaced with real
 * catalog evidence when the direction is reviewed and confirmed.
 */
function placeholderEvidenceFingerprint(paperKey: string): string {
  return `sha256:${sha256Hex(paperKey)}`;
}

function mergedRepresentatives(
  sources: readonly PersonalLibraryConfirmedDirection[],
): PersonalLibraryRepresentativeEvidence[] {
  const byKey = new Map<string, PersonalLibraryRepresentativeEvidence>();
  for (const source of sources) {
    for (const representative of source.representatives) {
      if (!byKey.has(representative.paperKey)) byKey.set(representative.paperKey, representative);
    }
  }
  return [...byKey.keys()].sort(codeUnitCompare)
    .slice(0, PERSONAL_LIBRARY_MAX_REPRESENTATIVES)
    .map((paperKey) => ({ ...byKey.get(paperKey)! }));
}

function profileDocument(value: unknown): PersonalLibraryInterestProfile {
  return decodePersonalLibraryInterestProfile(value)
    ?? fail("invalid-document", "profile must strictly decode");
}

function suggestionInput<K extends DirectionDiffSuggestion["kind"]>(
  value: unknown,
  kind: K,
): Extract<DirectionDiffSuggestion, { kind: K }> {
  const suggestion = decodeIncrementalSuggestion(value);
  if (!suggestion || suggestion.kind !== kind) {
    fail("invalid-input", `suggestion must be a strictly valid ${kind} suggestion`);
  }
  return suggestion as Extract<DirectionDiffSuggestion, { kind: K }>;
}

function idFactory(value: unknown): (kind: "split" | "merge") => string {
  if (typeof value !== "function") fail("invalid-input", "createId must be a function");
  return (kind) => {
    let id: unknown;
    try {
      id = (value as (kind: "split" | "merge") => string)(kind);
    } catch (caught) {
      fail("invalid-input", "createId threw while creating an id", { kind, cause: caught });
    }
    if (!isOpaqueId(id)) {
      fail("invalid-input", "createId must produce a valid opaque direction id", { kind, id });
    }
    return id;
  };
}

function outputProfile(value: PersonalLibraryInterestProfile): PersonalLibraryInterestProfile {
  return decodePersonalLibraryInterestProfile(value)
    ?? fail("invalid-document", "apply transaction produced an invalid profile");
}

function exactInput(
  value: unknown,
  required: readonly string[],
): Record<string, any> {
  if (!isPlainObject(value)) fail("invalid-input", "input must be an exact object");
  const keys = Object.keys(value);
  if (required.some((key) => !keys.includes(key))
    || keys.some((key) => !required.includes(key))) {
    fail("invalid-input", "input contains missing or unexpected fields");
  }
  return value;
}

function canonicalDate(value: unknown): string {
  try {
    if (!(value instanceof Date)) fail("invalid-input", "now must be a valid Date");
    const time = Date.prototype.getTime.call(value);
    if (!Number.isFinite(time)) fail("invalid-input", "now must be a valid Date");
    return new Date(time).toISOString();
  } catch (caught) {
    if (caught instanceof IncrementalSuggestionsApplyError) throw caught;
    fail("invalid-input", "now must be a valid Date");
  }
}

function monotonicTimestamp(candidate: string, ...existing: string[]): string {
  return [candidate, ...existing].reduce((latest, value) => Date.parse(value) > Date.parse(latest) ? value : latest);
}

function appendTimelineEvent(
  timeline: PersonalLibraryDirectionTimelineEvent[],
  event: PersonalLibraryDirectionTimelineEvent,
): PersonalLibraryDirectionTimelineEvent[] {
  const next = [...timeline, event];
  if (next.length <= PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS) return next;
  // Keep the created anchor event and the most recent events; drop the oldest non-anchor events.
  return [next[0]!, ...next.slice(next.length - (PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS - 1))];
}

function canonicalUnion(...sets: readonly (readonly string[])[]): string[] {
  return [...new Set(sets.flat())].sort(codeUnitCompare);
}

function lineageLimit(field: string, actual: number): never {
  fail("lineage-limit", `${field} lineage limit would be exceeded`, { field, actual });
}

function fail(
  code: IncrementalSuggestionsApplyErrorCode,
  message: string,
  details: Record<string, unknown> = {},
): never {
  throw new IncrementalSuggestionsApplyError(message, code, Object.freeze({ ...details }));
}

function isOpaqueId(value: unknown): value is string {
  return typeof value === "string"
    && value.length >= 1
    && value.length <= PERSONAL_LIBRARY_MAX_ID_LENGTH
    && /^[A-Za-z0-9._~-]+$/.test(value);
}

function byId(left: { id: string }, right: { id: string }): number {
  return codeUnitCompare(left.id, right.id);
}

function byPaperKey(left: { paperKey: string }, right: { paperKey: string }): number {
  return codeUnitCompare(left.paperKey, right.paperKey);
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function isPlainObject(value: unknown): value is Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}
