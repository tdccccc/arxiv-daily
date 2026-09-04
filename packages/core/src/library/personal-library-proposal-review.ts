/**
 * Editing a direction *proposal* before it is accepted: rename a candidate,
 * merge two of them, discard one.
 *
 * This used to sit in `personal-library-interest-profile-review` next to the
 * operations that acted on confirmed directions inside the interest profile
 * document. That document retired with ADR 0012, and ADR 0014 sends an
 * accepted proposal into `settings.topics` instead, so only the proposal-side
 * half has a subject left. It is split out here rather than left in a module
 * named after a document that no longer exists.
 */
import {
  PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS,
  PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS,
  createPersonalLibraryCatalogInputManifestFingerprint,
  createPersonalLibraryPaperEvidenceFingerprint,
  createPersonalLibraryRepresentativeSetFingerprint,
  decodePersonalLibraryDirectionProposal,
  type PersonalLibraryClusterMember,
  type PersonalLibraryDirectionCandidate,
  type PersonalLibraryDirectionProposal,
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
  }, representatives, current.lineage.candidateIds, current.clusterMembers);
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
  const merged = candidateFromReviewed(candidateId, draft, representatives, lineage,
    unionClusterMembers(sources));
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

function proposalDocument(value: unknown): PersonalLibraryDirectionProposal {
  return decodePersonalLibraryDirectionProposal(value)
    ?? fail("invalid-document", "proposal must strictly decode");
}

function compatibleCatalog(
  value: unknown,
  document: Pick<PersonalLibraryDirectionProposal, "scopeFingerprint" | "identificationFingerprint">,
): PersonalLibraryCatalog {
  const catalog = decodePersonalLibraryCatalog(value);
  if (!catalog) fail("invalid-document", "catalog must strictly decode");
  if (catalog.scopeFingerprint !== document.scopeFingerprint
    || catalog.identificationFingerprint !== document.identificationFingerprint) {
    fail("incompatible-catalog", "catalog identity is incompatible with the review document");
  }
  return catalog;
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

function candidateFromReviewed(
  id: string,
  draft: PersonalLibraryReviewedDirectionDraft,
  representatives: PersonalLibraryRepresentativeEvidence[],
  candidateIds: string[],
  clusterMembers?: PersonalLibraryClusterMember[],
): PersonalLibraryDirectionCandidate {
  return {
    id,
    name: draft.name,
    description: draft.description,
    discoveryCues: [...draft.discoveryCues],
    representatives: representatives.map((entry) => ({ ...entry })),
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    lineage: { candidateIds: [...candidateIds] },
    ...(clusterMembers !== undefined ? { clusterMembers: clusterMembers.map((member) => ({ ...member })) } : {}),
  };
}

function unionClusterMembers(
  sources: readonly PersonalLibraryDirectionCandidate[],
): PersonalLibraryClusterMember[] | undefined {
  const byPaperKey = new Map<string, number>();
  for (const source of sources) {
    for (const member of source.clusterMembers ?? []) {
      const current = byPaperKey.get(member.paperKey);
      if (current === undefined || member.confidence > current) byPaperKey.set(member.paperKey, member.confidence);
    }
  }
  if (byPaperKey.size === 0) return undefined;
  return [...byPaperKey.entries()].map(([paperKey, confidence]) => ({ paperKey, confidence }));
}

function outputProposal(value: PersonalLibraryDirectionProposal): PersonalLibraryDirectionProposal {
  return decodePersonalLibraryDirectionProposal(value)
    ?? fail("invalid-document", "review transaction produced an invalid proposal");
}

function decodeCandidateViaProposal(candidate: unknown): PersonalLibraryDirectionCandidate | null {
  const fingerprint = `sha256:${"0".repeat(64)}`;
  return decodePersonalLibraryDirectionProposal({
    schemaVersion: 3,
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
