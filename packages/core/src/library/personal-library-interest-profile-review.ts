import {
  PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS,
  PERSONAL_LIBRARY_MAX_DIRECTION_ANCESTRY_IDS,
  PERSONAL_LIBRARY_MAX_DIRECTIONS,
  PERSONAL_LIBRARY_MAX_PROFILE_ANCESTRY_IDS,
  PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS,
  createPersonalLibraryCatalogInputManifestFingerprint,
  createPersonalLibraryPaperEvidenceFingerprint,
  createPersonalLibraryRepresentativeSetFingerprint,
  decodePersonalLibraryDirectionProposal,
  decodePersonalLibraryInterestProfile,
  type PersonalLibraryConfirmedDirection,
  type PersonalLibraryDirectionCandidate,
  type PersonalLibraryDirectionProposal,
  type PersonalLibraryInterestProfile,
  type PersonalLibraryRepresentativeEvidence,
} from "./personal-library-interest-profile";
import {
  decodePersonalLibraryCatalog,
  type PersonalLibraryCatalog,
} from "./personal-library-catalog";

export interface PersonalLibraryReviewedDirectionDraft {
  name: string;
  description: string;
  discoveryCues: string[];
  representativePaperKeys: string[];
}

export interface PersonalLibraryDirectionTextPatch {
  name?: string;
  description?: string;
  discoveryCues?: string[];
}

export type PersonalLibraryReviewErrorCode =
  | "invalid-input"
  | "invalid-document"
  | "incompatible-catalog"
  | "not-found"
  | "conflict"
  | "lineage-limit"
  | "direction-limit"
  | "merge-relationship"
  | "evidence-mismatch";

export class PersonalLibraryInterestProfileReviewError extends Error {
  constructor(
    message: string,
    readonly code: PersonalLibraryReviewErrorCode,
    readonly details: Readonly<Record<string, unknown>> = {},
  ) {
    super(message);
    this.name = "PersonalLibraryInterestProfileReviewError";
  }
}

export function updatePersonalLibraryDirectionCandidate(input: unknown): PersonalLibraryDirectionProposal {
  const raw = exactInput(input, ["proposal", "candidateId", "patch", "representativePaperKeys", "catalog"], [
    "representativePaperKeys", "catalog",
  ]);
  const proposal = proposalDocument(raw.proposal);
  const candidateId = opaqueId(raw.candidateId, "candidateId");
  const patch = textPatch(raw.patch);
  const index = proposal.candidates.findIndex(({ id }) => id === candidateId);
  if (index < 0) fail("not-found", "candidate was not found", { candidateId });
  const current = proposal.candidates[index]!;
  const representativePaperKeys = optionalRepresentativeKeys(raw.representativePaperKeys);
  if (representativePaperKeys !== undefined && raw.catalog === undefined) {
    fail("invalid-input", "catalog is required when representativePaperKeys are supplied");
  }
  if (representativePaperKeys === undefined && raw.catalog !== undefined) {
    fail("invalid-input", "catalog is only accepted with representativePaperKeys");
  }
  const representativeCatalog = representativePaperKeys === undefined
    ? undefined
    : compatibleCatalog(raw.catalog, proposal);
  const representatives = representativePaperKeys === undefined
    ? current.representatives
    : representativesFromCatalog(representativeCatalog!, representativePaperKeys);
  const updated = candidateFromReviewed(current.id, {
    name: patch.name ?? current.name,
    description: patch.description ?? current.description,
    discoveryCues: patch.discoveryCues ?? current.discoveryCues,
    representativePaperKeys: representatives.map(({ paperKey }) => paperKey),
  }, representatives, current.lineage.candidateIds);
  if (JSON.stringify(updated) === JSON.stringify(current)) return proposal;
  proposal.candidates[index] = updated;
  proposal.candidates.sort(byId);
  return outputProposal(proposal);
}

export function mergePersonalLibraryDirectionCandidates(input: unknown): PersonalLibraryDirectionProposal {
  const raw = exactInput(input, ["proposal", "sourceCandidateIds", "candidateId", "draft", "catalog"]);
  const proposal = proposalDocument(raw.proposal);
  const sourceIds = opaqueIdSet(raw.sourceCandidateIds, "sourceCandidateIds", 2, proposal.candidates.length);
  const candidateId = opaqueId(raw.candidateId, "candidateId");
  if (proposal.candidates.some(({ id }) => id === candidateId)) {
    fail("conflict", "candidateId already exists", { candidateId });
  }
  const sources = sourceIds.map((id) => proposal.candidates.find((candidate) => candidate.id === id)
    ?? fail("not-found", "source candidate was not found", { candidateId: id }));
  const lineage = canonicalUnion([candidateId], ...sources.map(({ lineage }) => lineage.candidateIds));
  if (lineage.length > PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS) lineageLimit("candidateIds", lineage.length);
  const draft = reviewedDraft(raw.draft);
  const catalog = compatibleCatalog(raw.catalog, proposal);
  const representatives = representativesFromCatalog(catalog, draft.representativePaperKeys);
  const merged = candidateFromReviewed(candidateId, draft, representatives, lineage);
  proposal.candidates = proposal.candidates.filter(({ id }) => !sourceIds.includes(id));
  proposal.candidates.push(merged);
  proposal.candidates.sort(byId);
  return outputProposal(proposal);
}

export function removePersonalLibraryDirectionCandidate(input: unknown): PersonalLibraryDirectionProposal {
  const raw = exactInput(input, ["proposal", "candidateId"]);
  const proposal = proposalDocument(raw.proposal);
  const candidateId = opaqueId(raw.candidateId, "candidateId");
  const next = proposal.candidates.filter(({ id }) => id !== candidateId);
  if (next.length === proposal.candidates.length) fail("not-found", "candidate was not found", { candidateId });
  proposal.candidates = next;
  return outputProposal(proposal);
}

export function confirmPersonalLibraryDirectionCandidate(input: unknown): {
  proposal: PersonalLibraryDirectionProposal;
  profile: PersonalLibraryInterestProfile;
} {
  const raw = exactInput(input, ["proposal", "profile", "catalog", "candidateId", "directionId", "status", "draft", "now"]);
  const proposal = proposalDocument(raw.proposal);
  const profile = profileDocument(raw.profile);
  compatibleDocuments(proposal, profile);
  const catalog = compatibleCatalog(raw.catalog, profile);
  verifyProposalCatalogManifest(proposal, catalog);
  const candidateId = opaqueId(raw.candidateId, "candidateId");
  const directionId = opaqueId(raw.directionId, "directionId");
  const status = activeOrDisabled(raw.status, "status");
  const draft = reviewedDraft(raw.draft);
  const representatives = representativesFromCatalog(catalog, draft.representativePaperKeys);
  const timestamp = canonicalDate(raw.now);
  const candidate = proposal.candidates.find(({ id }) => id === candidateId);
  const existingById = profile.directions.find(({ id }) => id === directionId);

  if (!candidate) {
    if (existingById && existingById.status !== "merged"
      && existingById.status === status
      && existingById.lineage.proposalIds.includes(proposal.proposalId)
      && existingById.lineage.candidateIds.includes(candidateId)
      && exactReviewedSemantics(existingById, draft, representatives)) {
      return { proposal, profile };
    }
    fail("not-found", "candidate was not found and no exact confirmed recovery exists", {
      candidateId, directionId,
    });
  }
  if (existingById && existingById.status !== "merged"
    && existingById.status === status
    && existingById.lineage.proposalIds.includes(proposal.proposalId)
    && candidate.lineage.candidateIds.every((id) => existingById.lineage.candidateIds.includes(id))
    && exactReviewedSemantics(existingById, draft, representatives)) {
    proposal.candidates = proposal.candidates.filter(({ id }) => id !== candidateId);
    return { proposal: outputProposal(proposal), profile };
  }
  const conflictingLineage = profile.directions.find((direction) => (
    direction.lineage.candidateIds.some((id) => candidate.lineage.candidateIds.includes(id))
  ));
  if (existingById || conflictingLineage) {
    fail("conflict", "direction ID or candidate lineage is already confirmed", {
      directionId,
      conflictingDirectionId: (existingById ?? conflictingLineage)?.id,
    });
  }
  if (profile.directions.length >= PERSONAL_LIBRARY_MAX_DIRECTIONS) {
    fail("direction-limit", "confirmed direction limit would be exceeded", { maximum: PERSONAL_LIBRARY_MAX_DIRECTIONS });
  }
  const direction: PersonalLibraryConfirmedDirection = {
    id: directionId,
    status,
    name: draft.name,
    description: draft.description,
    discoveryCues: [...draft.discoveryCues],
    representatives,
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    lineage: {
      proposalIds: [proposal.proposalId],
      candidateIds: [...candidate.lineage.candidateIds],
      directionIds: [],
    },
    createdAt: timestamp,
    updatedAt: timestamp,
  };
  proposal.candidates = proposal.candidates.filter(({ id }) => id !== candidateId);
  profile.directions.push(direction);
  profile.directions.sort(byId);
  return { proposal: outputProposal(proposal), profile: outputProfile(profile) };
}

export function updatePersonalLibraryConfirmedDirection(input: unknown): PersonalLibraryInterestProfile {
  const raw = exactInput(input, ["profile", "directionId", "patch", "representativePaperKeys", "catalog", "now"], [
    "representativePaperKeys", "catalog",
  ]);
  const profile = profileDocument(raw.profile);
  const directionId = opaqueId(raw.directionId, "directionId");
  const patch = textPatch(raw.patch);
  const timestamp = canonicalDate(raw.now);
  const index = profile.directions.findIndex(({ id }) => id === directionId);
  if (index < 0) fail("not-found", "direction was not found", { directionId });
  const current = profile.directions[index]!;
  if (current.status === "merged") fail("conflict", "merged directions cannot be edited", { directionId });
  const keys = optionalRepresentativeKeys(raw.representativePaperKeys);
  if (keys !== undefined && raw.catalog === undefined) fail("invalid-input", "catalog is required for representative changes");
  if (keys === undefined && raw.catalog !== undefined) fail("invalid-input", "catalog is only accepted with representative changes");
  const representatives = keys === undefined
    ? current.representatives
    : representativesFromCatalog(compatibleCatalog(raw.catalog, profile), keys);
  const next = {
    ...current,
    name: patch.name ?? current.name,
    description: patch.description ?? current.description,
    discoveryCues: patch.discoveryCues ?? current.discoveryCues,
    representatives,
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
  };
  if (sameDirectionSemantics(current, next)) return profile;
  next.updatedAt = monotonicTimestamp(timestamp, current.updatedAt);
  profile.directions[index] = next;
  return outputProfile(profile);
}

export function disablePersonalLibraryConfirmedDirection(input: unknown): PersonalLibraryInterestProfile {
  const raw = exactInput(input, ["profile", "directionId", "now"]);
  return changeStatus(profileDocument(raw.profile), opaqueId(raw.directionId, "directionId"), "disabled", canonicalDate(raw.now));
}

export function enablePersonalLibraryConfirmedDirection(input: unknown): PersonalLibraryInterestProfile {
  const raw = exactInput(input, ["profile", "directionId", "catalog", "now"]);
  const profile = profileDocument(raw.profile);
  const directionId = opaqueId(raw.directionId, "directionId");
  const catalog = compatibleCatalog(raw.catalog, profile);
  const direction = profile.directions.find(({ id }) => id === directionId);
  if (!direction) fail("not-found", "direction was not found", { directionId });
  if (direction.status === "merged") fail("conflict", "merged directions cannot be enabled", { directionId });
  verifyExistingEvidence(direction, catalog);
  return changeStatus(profile, directionId, "active", canonicalDate(raw.now));
}

export function mergePersonalLibraryConfirmedDirections(input: unknown): PersonalLibraryInterestProfile {
  const raw = exactInput(input, ["profile", "sourceDirectionIds", "directionId", "status", "draft", "catalog", "now"]);
  const profile = profileDocument(raw.profile);
  const sourceIds = opaqueIdSet(raw.sourceDirectionIds, "sourceDirectionIds", 2, profile.directions.length);
  const directionId = opaqueId(raw.directionId, "directionId");
  const status = activeOrDisabled(raw.status, "status");
  const timestamp = canonicalDate(raw.now);
  if (profile.directions.some(({ id }) => id === directionId)) fail("conflict", "directionId already exists", { directionId });
  const sources = sourceIds.map((id) => profile.directions.find((direction) => direction.id === id)
    ?? fail("not-found", "source direction was not found", { directionId: id }));
  if (sources.some(({ status: sourceStatus }) => sourceStatus === "merged")) {
    fail("conflict", "only terminal active or disabled directions can be merged");
  }
  if (profile.directions.length >= PERSONAL_LIBRARY_MAX_DIRECTIONS) {
    fail("direction-limit", "confirmed direction limit would be exceeded", { maximum: PERSONAL_LIBRARY_MAX_DIRECTIONS });
  }
  const draft = reviewedDraft(raw.draft);
  const representatives = representativesFromCatalog(compatibleCatalog(raw.catalog, profile), draft.representativePaperKeys);
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

  const sourceSet = new Set(sourceIds);
  profile.directions = profile.directions.map((direction) => sourceSet.has(direction.id) ? {
    ...direction,
    status: "merged" as const,
    mergedIntoDirectionId: directionId,
    updatedAt: monotonicTimestamp(timestamp, direction.updatedAt),
  } : direction);
  profile.directions.push({
    id: directionId,
    status,
    name: draft.name,
    description: draft.description,
    discoveryCues: [...draft.discoveryCues],
    representatives,
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    lineage: { proposalIds, candidateIds, directionIds },
    createdAt: timestamp,
    updatedAt: monotonicTimestamp(timestamp, ...sources.map(({ updatedAt }) => updatedAt)),
  });
  profile.directions.sort(byId);
  return outputProfile(profile);
}

export function removePersonalLibraryConfirmedDirection(input: unknown): PersonalLibraryInterestProfile {
  const raw = exactInput(input, ["profile", "directionId", "mode"]);
  const profile = profileDocument(raw.profile);
  const directionId = opaqueId(raw.directionId, "directionId");
  if (raw.mode !== "restrict" && raw.mode !== "cascade") fail("invalid-input", "mode must be restrict or cascade");
  if (!profile.directions.some(({ id }) => id === directionId)) fail("not-found", "direction was not found", { directionId });
  const component = mergeComponent(profile.directions, directionId);
  if (raw.mode === "restrict" && component.size !== 1) {
    fail("merge-relationship", "direction participates in a merge family", {
      directionId, relatedDirectionIds: [...component].sort(codeUnitCompare),
    });
  }
  profile.directions = profile.directions.filter(({ id }) => !component.has(id));
  return outputProfile(profile);
}

function changeStatus(
  profile: PersonalLibraryInterestProfile,
  directionId: string,
  status: "active" | "disabled",
  timestamp: string,
): PersonalLibraryInterestProfile {
  const index = profile.directions.findIndex(({ id }) => id === directionId);
  if (index < 0) fail("not-found", "direction was not found", { directionId });
  const current = profile.directions[index]!;
  if (current.status === "merged") fail("conflict", "merged direction status cannot change", { directionId });
  if (current.status === status) return profile;
  profile.directions[index] = { ...current, status, updatedAt: monotonicTimestamp(timestamp, current.updatedAt) };
  return outputProfile(profile);
}

function proposalDocument(value: unknown): PersonalLibraryDirectionProposal {
  return decodePersonalLibraryDirectionProposal(value)
    ?? fail("invalid-document", "proposal must strictly decode");
}

function profileDocument(value: unknown): PersonalLibraryInterestProfile {
  return decodePersonalLibraryInterestProfile(value)
    ?? fail("invalid-document", "profile must strictly decode");
}

function compatibleCatalog(
  value: unknown,
  document: Pick<PersonalLibraryInterestProfile, "scopeFingerprint" | "identificationFingerprint">,
): PersonalLibraryCatalog {
  const catalog = decodePersonalLibraryCatalog(value);
  if (!catalog) fail("invalid-document", "catalog must strictly decode");
  if (catalog.scopeFingerprint !== document.scopeFingerprint
    || catalog.identificationFingerprint !== document.identificationFingerprint) {
    fail("incompatible-catalog", "catalog identity is incompatible with the review document");
  }
  return catalog;
}

function compatibleDocuments(
  proposal: PersonalLibraryDirectionProposal,
  profile: PersonalLibraryInterestProfile,
): void {
  if (proposal.scopeFingerprint !== profile.scopeFingerprint
    || proposal.identificationFingerprint !== profile.identificationFingerprint) {
    fail("conflict", "proposal and profile identities are incompatible");
  }
}

function reviewedDraft(value: unknown): PersonalLibraryReviewedDirectionDraft {
  const raw = exactInput(value, ["name", "description", "discoveryCues", "representativePaperKeys"]);
  const candidate = {
    id: "validation",
    name: raw.name,
    description: raw.description,
    discoveryCues: raw.discoveryCues,
    representatives: [{ paperKey: "arxiv:2608.00001", evidenceFingerprint: `sha256:${"0".repeat(64)}` }],
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint([
      { paperKey: "arxiv:2608.00001", evidenceFingerprint: `sha256:${"0".repeat(64)}` },
    ]),
    lineage: { candidateIds: ["validation"] },
  };
  const decoded = decodeCandidateViaProposal(candidate);
  if (!decoded) fail("invalid-input", "reviewed draft text is invalid or noncanonical");
  return {
    name: decoded.name,
    description: decoded.description,
    discoveryCues: decoded.discoveryCues,
    representativePaperKeys: representativeKeys(raw.representativePaperKeys),
  };
}

function textPatch(value: unknown): PersonalLibraryDirectionTextPatch {
  if (!isPlainObject(value)) fail("invalid-input", "patch must be an object");
  const keys = Object.keys(value);
  if (keys.length === 0 || keys.some((key) => !["name", "description", "discoveryCues"].includes(key))) {
    fail("invalid-input", "patch must contain only one or more text fields");
  }
  const draft = reviewedDraft({
    name: value.name ?? "validation",
    description: value.description ?? "validation",
    discoveryCues: value.discoveryCues ?? ["validation"],
    representativePaperKeys: ["arxiv:2608.00001"],
  });
  return {
    ...(value.name !== undefined ? { name: draft.name } : {}),
    ...(value.description !== undefined ? { description: draft.description } : {}),
    ...(value.discoveryCues !== undefined ? { discoveryCues: draft.discoveryCues } : {}),
  };
}

function representativesFromCatalog(
  catalog: PersonalLibraryCatalog,
  paperKeys: string[],
): PersonalLibraryRepresentativeEvidence[] {
  return paperKeys.map((paperKey) => {
    const paper = catalog.papers[paperKey];
    if (!paper) fail("evidence-mismatch", "representative paper is absent from catalog", { paperKey });
    return { paperKey, evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(paper) };
  });
}

function verifyExistingEvidence(
  direction: PersonalLibraryConfirmedDirection,
  catalog: PersonalLibraryCatalog,
): void {
  for (const representative of direction.representatives) {
    const paper = catalog.papers[representative.paperKey];
    if (!paper || createPersonalLibraryPaperEvidenceFingerprint(paper) !== representative.evidenceFingerprint) {
      fail("evidence-mismatch", "existing representative evidence is missing or stale", {
        directionId: direction.id, paperKey: representative.paperKey,
      });
    }
  }
}

function candidateFromReviewed(
  id: string,
  draft: PersonalLibraryReviewedDirectionDraft,
  representatives: PersonalLibraryRepresentativeEvidence[],
  candidateIds: string[],
): PersonalLibraryDirectionCandidate {
  return {
    id,
    name: draft.name,
    description: draft.description,
    discoveryCues: [...draft.discoveryCues],
    representatives: representatives.map((entry) => ({ ...entry })),
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    lineage: { candidateIds: [...candidateIds] },
  };
}

function exactReviewedSemantics(
  direction: PersonalLibraryConfirmedDirection,
  draft: PersonalLibraryReviewedDirectionDraft,
  representatives: PersonalLibraryRepresentativeEvidence[],
): boolean {
  return direction.name === draft.name
    && direction.description === draft.description
    && JSON.stringify(direction.discoveryCues) === JSON.stringify(draft.discoveryCues)
    && JSON.stringify(direction.representatives) === JSON.stringify(representatives);
}

function sameDirectionSemantics(
  left: PersonalLibraryConfirmedDirection,
  right: PersonalLibraryConfirmedDirection,
): boolean {
  const strip = (direction: PersonalLibraryConfirmedDirection) => {
    const { updatedAt: _updatedAt, ...semantic } = direction;
    return semantic;
  };
  return JSON.stringify(strip(left)) === JSON.stringify(strip(right));
}

function outputProposal(value: PersonalLibraryDirectionProposal): PersonalLibraryDirectionProposal {
  return decodePersonalLibraryDirectionProposal(value)
    ?? fail("invalid-document", "review transaction produced an invalid proposal");
}

function outputProfile(value: PersonalLibraryInterestProfile): PersonalLibraryInterestProfile {
  return decodePersonalLibraryInterestProfile(value)
    ?? fail("invalid-document", "review transaction produced an invalid profile");
}

function decodeCandidateViaProposal(candidate: unknown): PersonalLibraryDirectionCandidate | null {
  const fingerprint = `sha256:${"0".repeat(64)}`;
  return decodePersonalLibraryDirectionProposal({
    schemaVersion: 2,
    revision: 0,
    proposalId: "validation",
    scopeFingerprint: fingerprint,
    identificationFingerprint: fingerprint,
    catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
      scopeFingerprint: fingerprint,
      identificationFingerprint: fingerprint,
      catalogInputPapers: [{ paperKey: "arxiv:2608.00001", evidenceFingerprint: fingerprint }],
    }),
    catalogInputPapers: [{ paperKey: "arxiv:2608.00001", evidenceFingerprint: fingerprint }],
    generationContractFingerprint: fingerprint,
    generatedAt: "2000-01-01T00:00:00.000Z",
    candidates: [candidate],
  })?.candidates[0] ?? null;
}

function representativeKeys(value: unknown): string[] {
  if (!Array.isArray(value)) fail("invalid-input", "representativePaperKeys must be an array");
  const evidence = `sha256:${"0".repeat(64)}`;
  try {
    const representatives = value.map((paperKey) => ({ paperKey, evidenceFingerprint: evidence }));
    return createPersonalLibraryRepresentativeSetFingerprint(representatives)
      ? representatives.map(({ paperKey }) => paperKey)
      : [];
  } catch {
    fail("invalid-input", "representativePaperKeys must be canonical, sorted, unique, and bounded");
  }
}

function optionalRepresentativeKeys(value: unknown): string[] | undefined {
  return value === undefined ? undefined : representativeKeys(value);
}

function opaqueId(value: unknown, field: string): string {
  if (typeof value !== "string") fail("invalid-input", `${field} must be a valid opaque ID`);
  const probe = decodeCandidateViaProposal({
    id: value,
    name: "validation",
    description: "validation",
    discoveryCues: ["validation"],
    representatives: [{ paperKey: "arxiv:2608.00001", evidenceFingerprint: `sha256:${"0".repeat(64)}` }],
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint([
      { paperKey: "arxiv:2608.00001", evidenceFingerprint: `sha256:${"0".repeat(64)}` },
    ]),
    lineage: { candidateIds: [value] },
  });
  if (!probe) fail("invalid-input", `${field} must be a valid opaque ID`, { field });
  return value;
}

function opaqueIdSet(value: unknown, field: string, minimum: number, maximum: number): string[] {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) {
    fail("invalid-input", `${field} has an invalid size`, { minimum, maximum });
  }
  const ids = value.map((id) => opaqueId(id, field));
  if (!isStrictlyOrderedUnique(ids)) fail("invalid-input", `${field} must be code-unit sorted and unique`);
  return ids;
}

function canonicalDate(value: unknown): string {
  try {
    if (!(value instanceof Date)) fail("invalid-input", "now must be a valid Date");
    const time = Date.prototype.getTime.call(value);
    if (!Number.isFinite(time)) fail("invalid-input", "now must be a valid Date");
    return new Date(time).toISOString();
  } catch (caught) {
    if (caught instanceof PersonalLibraryInterestProfileReviewError) throw caught;
    fail("invalid-input", "now must be a valid Date");
  }
}

function activeOrDisabled(value: unknown, field: string): "active" | "disabled" {
  if (value !== "active" && value !== "disabled") fail("invalid-input", `${field} must be active or disabled`);
  return value;
}

function exactInput(
  value: unknown,
  required: readonly string[],
  optional: readonly string[] = [],
): Record<string, any> {
  if (!isPlainObject(value)) fail("invalid-input", "input must be an exact object");
  const keys = Object.keys(value);
  if (required.some((key) => !optional.includes(key) && !keys.includes(key))
    || keys.some((key) => !required.includes(key))) {
    fail("invalid-input", "input contains missing or unexpected fields");
  }
  return value;
}

function mergeComponent(directions: PersonalLibraryConfirmedDirection[], start: string): Set<string> {
  const adjacency = new Map(directions.map(({ id }) => [id, new Set<string>()]));
  const connect = (left: string, right: string) => {
    adjacency.get(left)?.add(right);
    adjacency.get(right)?.add(left);
  };
  for (const direction of directions) {
    if (direction.status === "merged") connect(direction.id, direction.mergedIntoDirectionId);
    for (const ancestorId of direction.lineage.directionIds) connect(direction.id, ancestorId);
  }
  const found = new Set<string>();
  const pending = [start];
  while (pending.length > 0) {
    const id = pending.pop()!;
    if (found.has(id)) continue;
    found.add(id);
    for (const related of adjacency.get(id) ?? []) pending.push(related);
  }
  return found;
}

function canonicalUnion(...sets: readonly (readonly string[])[]): string[] {
  return [...new Set(sets.flat())].sort(codeUnitCompare);
}

function monotonicTimestamp(candidate: string, ...existing: string[]): string {
  return [candidate, ...existing].reduce((latest, value) => Date.parse(value) > Date.parse(latest) ? value : latest);
}

function verifyProposalCatalogManifest(
  proposal: PersonalLibraryDirectionProposal,
  catalog: PersonalLibraryCatalog,
): void {
  for (const selected of proposal.catalogInputPapers) {
    const paper = catalog.papers[selected.paperKey];
    if (!paper || createPersonalLibraryPaperEvidenceFingerprint(paper) !== selected.evidenceFingerprint) {
      fail("conflict", "proposal selected catalog evidence is stale", {
        proposalId: proposal.proposalId,
        paperKey: selected.paperKey,
      });
    }
  }
}

function lineageLimit(field: string, actual: number): never {
  fail("lineage-limit", `${field} lineage limit would be exceeded`, { field, actual });
}

function fail(code: PersonalLibraryReviewErrorCode, message: string, details: Record<string, unknown> = {}): never {
  throw new PersonalLibraryInterestProfileReviewError(message, code, Object.freeze({ ...details }));
}

function byId(left: { id: string }, right: { id: string }): number {
  return codeUnitCompare(left.id, right.id);
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function isStrictlyOrderedUnique(value: readonly string[]): boolean {
  return value.every((item, index) => index === 0 || codeUnitCompare(value[index - 1]!, item) < 0);
}

function isPlainObject(value: unknown): value is Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}
