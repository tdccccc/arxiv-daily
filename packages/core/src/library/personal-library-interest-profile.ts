import {
  decodePersonalLibraryCatalog,
  type PersonalLibraryPaperRecord,
} from "./personal-library-catalog";
import { paperKeyFromArxivId } from "../services/paper-key";
import { sha256Hex } from "../utils/digest";

export const PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION = 2 as const;
export const PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION = 2 as const;
export const PERSONAL_LIBRARY_LEGACY_INTEREST_PROFILE_SCHEMA_VERSION = 1 as const;
export const PERSONAL_LIBRARY_MIN_PROPOSAL_CANDIDATES = 0 as const;
export const PERSONAL_LIBRARY_MAX_PROPOSAL_CANDIDATES = 12 as const;
export const PERSONAL_LIBRARY_MIN_REPRESENTATIVES = 1 as const;
export const PERSONAL_LIBRARY_MAX_REPRESENTATIVES = 5 as const;
export const PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS = 1_000 as const;
export const PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS = 12 as const;
export const PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS = 12 as const;
export const PERSONAL_LIBRARY_MAX_DIRECTIONS = 256 as const;
export const PERSONAL_LIBRARY_MAX_DIRECTION_ANCESTRY_IDS = 256 as const;
export const PERSONAL_LIBRARY_MAX_PROFILE_ANCESTRY_IDS = 256 as const;
export const PERSONAL_LIBRARY_MAX_ID_LENGTH = 128 as const;
export const PERSONAL_LIBRARY_MAX_NAME_LENGTH = 120 as const;
export const PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH = 1_000 as const;
export const PERSONAL_LIBRARY_MIN_DISCOVERY_CUES = 1 as const;
export const PERSONAL_LIBRARY_MAX_DISCOVERY_CUES = 12 as const;
export const PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH = 200 as const;
export const PERSONAL_LIBRARY_MAX_GENERATION_CONTRACT_LENGTH = 4_096 as const;

export interface PersonalLibraryRepresentativeEvidence {
  paperKey: string;
  evidenceFingerprint: string;
}

export interface PersonalLibraryDirectionCandidate {
  id: string;
  name: string;
  description: string;
  discoveryCues: string[];
  representatives: PersonalLibraryRepresentativeEvidence[];
  representativeSetFingerprint: string;
  /** Includes this candidate's id; other ids are retained historical source candidates. */
  lineage: { candidateIds: string[] };
}

export interface PersonalLibraryDirectionProposal {
  schemaVersion: typeof PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION;
  revision: number;
  proposalId: string;
  scopeFingerprint: string;
  identificationFingerprint: string;
  catalogInputFingerprint: string;
  catalogInputPapers: PersonalLibraryRepresentativeEvidence[];
  generationContractFingerprint: string;
  generatedAt: string;
  candidates: PersonalLibraryDirectionCandidate[];
}

interface PersonalLibraryConfirmedDirectionCommon {
  id: string;
  name: string;
  description: string;
  discoveryCues: string[];
  representatives: PersonalLibraryRepresentativeEvidence[];
  representativeSetFingerprint: string;
  lineage: {
    proposalIds: string[];
    candidateIds: string[];
    /** Retained merged ancestors whose merge chains terminate at this direction. */
    directionIds: string[];
  };
  createdAt: string;
  updatedAt: string;
}

export type PersonalLibraryConfirmedDirection =
  | (PersonalLibraryConfirmedDirectionCommon & { status: "active" | "disabled" })
  | (PersonalLibraryConfirmedDirectionCommon & {
      status: "merged";
      mergedIntoDirectionId: string;
    });

export interface PersonalLibraryInterestProfile {
  schemaVersion: typeof PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION;
  revision: number;
  scopeFingerprint: string;
  identificationFingerprint: string;
  updatedAt: string;
  directions: PersonalLibraryConfirmedDirection[];
}

export function createEmptyPersonalLibraryInterestProfile(
  scopeFingerprint: string,
  identificationFingerprint: string,
  now: Date = new Date(),
): PersonalLibraryInterestProfile {
  const profile: PersonalLibraryInterestProfile = {
    schemaVersion: PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION,
    revision: 0,
    scopeFingerprint,
    identificationFingerprint,
    updatedAt: now.toISOString(),
    directions: [],
  };
  const decoded = decodePersonalLibraryInterestProfile(profile);
  if (!decoded) throw new TypeError("cannot create empty personal library interest profile");
  return decoded;
}

export function isEmptyPersonalLibraryInterestProfile(
  value: unknown,
): value is PersonalLibraryInterestProfile {
  const decoded = decodePersonalLibraryInterestProfile(value);
  return decoded !== null && decoded.directions.length === 0;
}

export type PersonalLibraryEligibilityDocumentDiagnostic =
  | "profile-invalid"
  | "catalog-invalid"
  | "profile-scope-mismatch"
  | "profile-identification-mismatch";

export type PersonalLibraryDirectionStalenessReason =
  | "direction-disabled"
  | "direction-merged"
  | "representative-missing"
  | "representative-evidence-changed";

export interface PersonalLibraryDirectionStalenessDiagnostic {
  directionId: string;
  eligible: boolean;
  reasons: Array<{
    reason: PersonalLibraryDirectionStalenessReason;
    paperKey?: string;
  }>;
}

export interface PersonalLibraryEligibleDirection {
  id: string;
  name: string;
  description: string;
  discoveryCues: string[];
  representatives: PersonalLibraryRepresentativeEvidence[];
}

export interface PersonalLibraryInterestEligibility {
  documentDiagnostics: PersonalLibraryEligibilityDocumentDiagnostic[];
  eligibleDirections: PersonalLibraryEligibleDirection[];
  diagnostics: PersonalLibraryDirectionStalenessDiagnostic[];
}

export function createPersonalLibraryPaperEvidenceFingerprint(
  paper: PersonalLibraryPaperRecord,
): string {
  if (!isCanonicalCatalogPaper(paper)) {
    throw new TypeError("paper must be an exact canonical metadata-and-abstract arXiv catalog record");
  }
  return fingerprint({
    paperKey: paper.paperKey,
    source: paper.source,
    externalId: paper.externalId,
    title: paper.title,
    authors: [...paper.authors],
    abstract: paper.abstract,
    published: paper.published,
    updated: paper.updated,
    primaryCategory: paper.primaryCategory,
    // P2 defines categories as a unique set with primary-category membership.
    categories: [...paper.categories].sort(codeUnitCompare),
    evidenceDepth: paper.evidenceDepth,
  });
}

export function createPersonalLibraryCatalogInputManifest(
  papers: readonly PersonalLibraryPaperRecord[],
): PersonalLibraryRepresentativeEvidence[] {
  if (!Array.isArray(papers) || papers.length === 0
    || papers.length > PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS) {
    throw new TypeError("catalog input must contain a bounded explicit paper selection");
  }
  const manifest = papers.map((paper) => ({
    paperKey: paper.paperKey,
    evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(paper),
  })).sort((left, right) => codeUnitCompare(left.paperKey, right.paperKey));
  if (!isStrictlyOrderedUnique(manifest.map(({ paperKey }) => paperKey))) {
    throw new TypeError("selected catalog paper keys must be unique");
  }
  return manifest;
}

export function createPersonalLibraryCatalogInputManifestFingerprint(input: {
  scopeFingerprint: string;
  identificationFingerprint: string;
  catalogInputPapers: readonly PersonalLibraryRepresentativeEvidence[];
}): string {
  if (!isExactObject(input, ["scopeFingerprint", "identificationFingerprint", "catalogInputPapers"])
    || !isFingerprint(input.scopeFingerprint)
    || !isFingerprint(input.identificationFingerprint)) {
    throw new TypeError("catalog input manifest identity must be exact fingerprints");
  }
  const manifest = decodeCatalogInputManifest(input.catalogInputPapers);
  if (!manifest) throw new TypeError("catalog input manifest must be canonical and bounded");
  return fingerprint({
    scopeFingerprint: input.scopeFingerprint,
    identificationFingerprint: input.identificationFingerprint,
    papers: manifest,
  });
}

export function createPersonalLibraryCatalogInputFingerprint(input: {
  scopeFingerprint: string;
  identificationFingerprint: string;
  papers: readonly PersonalLibraryPaperRecord[];
}): string {
  if (!isExactObject(input, ["scopeFingerprint", "identificationFingerprint", "papers"])) {
    throw new TypeError("catalog input must be exact");
  }
  return createPersonalLibraryCatalogInputManifestFingerprint({
    scopeFingerprint: input.scopeFingerprint,
    identificationFingerprint: input.identificationFingerprint,
    catalogInputPapers: createPersonalLibraryCatalogInputManifest(input.papers),
  });
}

export function createPersonalLibraryRepresentativeSetFingerprint(
  representatives: readonly PersonalLibraryRepresentativeEvidence[],
): string {
  const decoded = decodeRepresentatives(representatives);
  if (!decoded) throw new TypeError("representatives must be canonical and bounded");
  return fingerprint({ representatives: decoded });
}

export function createPersonalLibraryGenerationContractFingerprint(contract: string): string {
  if (typeof contract !== "string" || contract.length === 0
    || contract.length > PERSONAL_LIBRARY_MAX_GENERATION_CONTRACT_LENGTH) {
    throw new TypeError("generation contract must be a bounded non-empty string");
  }
  return fingerprint({ contract });
}

export function decodePersonalLibraryDirectionProposal(
  value: unknown,
): PersonalLibraryDirectionProposal | null {
  if (!isExactObject(value, [
    "schemaVersion", "revision", "proposalId", "scopeFingerprint", "identificationFingerprint",
    "catalogInputFingerprint", "catalogInputPapers", "generationContractFingerprint", "generatedAt", "candidates",
  ]) || value.schemaVersion !== PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION
    || !isNonNegativeSafeInteger(value.revision)
    || !isOpaqueId(value.proposalId)
    || !isFingerprint(value.scopeFingerprint)
    || !isFingerprint(value.identificationFingerprint)
    || !isFingerprint(value.catalogInputFingerprint)
    || !isFingerprint(value.generationContractFingerprint)
    || !isCanonicalTimestamp(value.generatedAt)
    || !Array.isArray(value.candidates)
    || value.candidates.length < PERSONAL_LIBRARY_MIN_PROPOSAL_CANDIDATES
    || value.candidates.length > PERSONAL_LIBRARY_MAX_PROPOSAL_CANDIDATES) return null;

  const catalogInputPapers = decodeCatalogInputManifest(value.catalogInputPapers);
  if (!catalogInputPapers || createPersonalLibraryCatalogInputManifestFingerprint({
    scopeFingerprint: value.scopeFingerprint,
    identificationFingerprint: value.identificationFingerprint,
    catalogInputPapers,
  }) !== value.catalogInputFingerprint) return null;

  const candidates: PersonalLibraryDirectionCandidate[] = [];
  for (const raw of value.candidates) {
    const candidate = decodeCandidate(raw);
    if (!candidate) return null;
    candidates.push(candidate);
  }
  if (!isStrictlyOrderedUnique(candidates.map(({ id }) => id))) return null;
  return {
    schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
    revision: value.revision,
    proposalId: value.proposalId,
    scopeFingerprint: value.scopeFingerprint,
    identificationFingerprint: value.identificationFingerprint,
    catalogInputFingerprint: value.catalogInputFingerprint,
    catalogInputPapers,
    generationContractFingerprint: value.generationContractFingerprint,
    generatedAt: value.generatedAt,
    candidates,
  };
}

export function decodePersistedPersonalLibraryInterestProfile(
  value: unknown,
): PersonalLibraryInterestProfile | null {
  const profile = decodePersonalLibraryInterestProfile(value);
  if (!profile || profile.directions.some((direction) => direction.updatedAt > profile.updatedAt)) {
    return null;
  }
  return profile;
}

export function decodeDurablePersonalLibraryInterestProfile(
  value: unknown,
): PersonalLibraryInterestProfile | null {
  return decodePersistedPersonalLibraryInterestProfile(value)
    ?? migrateLegacyPersonalLibraryInterestProfile(value);
}

export function migrateLegacyPersonalLibraryInterestProfile(
  value: unknown,
): PersonalLibraryInterestProfile | null {
  if (!isExactObject(value, [
    "schemaVersion", "revision", "scopeFingerprint", "identificationFingerprint", "updatedAt",
    "directions",
  ]) || value.schemaVersion !== PERSONAL_LIBRARY_LEGACY_INTEREST_PROFILE_SCHEMA_VERSION
    || !Array.isArray(value.directions)) return null;
  const directions: unknown[] = [];
  for (const raw of value.directions) {
    if (!isPlainObject(raw) || !isExactObject(raw.lineage, ["proposalId", "candidateIds", "directionIds"])
      || !isOpaqueId(raw.lineage.proposalId)) return null;
    directions.push({
      ...raw,
      lineage: {
        proposalIds: [raw.lineage.proposalId],
        candidateIds: raw.lineage.candidateIds,
        directionIds: raw.lineage.directionIds,
      },
    });
  }
  const migrated = decodePersonalLibraryInterestProfile({
    ...value,
    schemaVersion: PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION,
    directions,
  });
  if (!migrated || migrated.directions.some((direction) => direction.updatedAt > migrated.updatedAt)) {
    return null;
  }
  return migrated;
}

export function decodePersonalLibraryInterestProfile(
  value: unknown,
): PersonalLibraryInterestProfile | null {
  if (!isExactObject(value, [
    "schemaVersion", "revision", "scopeFingerprint", "identificationFingerprint", "updatedAt",
    "directions",
  ]) || value.schemaVersion !== PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION
    || !isNonNegativeSafeInteger(value.revision)
    || !isFingerprint(value.scopeFingerprint)
    || !isFingerprint(value.identificationFingerprint)
    || !isCanonicalTimestamp(value.updatedAt)
    || !Array.isArray(value.directions)
    || value.directions.length > PERSONAL_LIBRARY_MAX_DIRECTIONS) return null;

  const directions: PersonalLibraryConfirmedDirection[] = [];
  let totalAncestryIds = 0;
  for (const raw of value.directions) {
    const direction = decodeDirection(raw);
    if (!direction) return null;
    totalAncestryIds += direction.lineage.directionIds.length;
    if (totalAncestryIds > PERSONAL_LIBRARY_MAX_PROFILE_ANCESTRY_IDS) return null;
    directions.push(direction);
  }
  if (!isStrictlyOrderedUnique(directions.map(({ id }) => id))) return null;
  const byId = new Map(directions.map((direction) => [direction.id, direction]));
  for (const direction of directions) {
    if (direction.status === "merged"
      && (!byId.has(direction.mergedIntoDirectionId)
        || direction.mergedIntoDirectionId === direction.id)) return null;
  }
  if (hasMergeCycle(directions)) return null;
  for (const direction of directions) {
    for (const ancestorId of direction.lineage.directionIds) {
      if (ancestorId === direction.id || !mergedChainTerminatesAt(ancestorId, direction.id, byId)) {
        return null;
      }
    }
  }
  return {
    schemaVersion: PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION,
    revision: value.revision,
    scopeFingerprint: value.scopeFingerprint,
    identificationFingerprint: value.identificationFingerprint,
    updatedAt: value.updatedAt,
    directions,
  };
}

export function evaluatePersonalLibraryInterestEligibility(
  profileValue: unknown,
  catalogValue: unknown,
): PersonalLibraryInterestEligibility {
  const profile = decodePersistedPersonalLibraryInterestProfile(profileValue);
  const catalog = decodePersonalLibraryCatalog(catalogValue);
  const documentDiagnostics: PersonalLibraryEligibilityDocumentDiagnostic[] = [];
  if (!profile) documentDiagnostics.push("profile-invalid");
  if (!catalog) documentDiagnostics.push("catalog-invalid");
  if (!profile || !catalog) {
    return { documentDiagnostics, eligibleDirections: [], diagnostics: [] };
  }
  if (profile.scopeFingerprint !== catalog.scopeFingerprint) {
    documentDiagnostics.push("profile-scope-mismatch");
  }
  if (profile.identificationFingerprint !== catalog.identificationFingerprint) {
    documentDiagnostics.push("profile-identification-mismatch");
  }
  const compatible = documentDiagnostics.length === 0;
  const eligibleDirections: PersonalLibraryEligibleDirection[] = [];
  const diagnostics: PersonalLibraryDirectionStalenessDiagnostic[] = [];

  for (const direction of profile.directions) {
    const reasons: PersonalLibraryDirectionStalenessDiagnostic["reasons"] = [];
    if (direction.status === "disabled") reasons.push({ reason: "direction-disabled" });
    if (direction.status === "merged") reasons.push({ reason: "direction-merged" });
    for (const representative of direction.representatives) {
      const paper = catalog.papers[representative.paperKey];
      if (!paper) {
        reasons.push({ reason: "representative-missing", paperKey: representative.paperKey });
      } else if (createPersonalLibraryPaperEvidenceFingerprint(paper)
        !== representative.evidenceFingerprint) {
        reasons.push({
          reason: "representative-evidence-changed",
          paperKey: representative.paperKey,
        });
      }
    }
    const eligible = compatible && direction.status === "active" && reasons.length === 0;
    diagnostics.push({ directionId: direction.id, eligible, reasons });
    if (eligible) {
      eligibleDirections.push({
        id: direction.id,
        name: direction.name,
        description: direction.description,
        discoveryCues: [...direction.discoveryCues],
        representatives: direction.representatives.map((entry) => ({ ...entry })),
      });
    }
  }
  return { documentDiagnostics, eligibleDirections, diagnostics };
}

function decodeCandidate(value: unknown): PersonalLibraryDirectionCandidate | null {
  if (!isExactObject(value, [
    "id", "name", "description", "discoveryCues", "representatives",
    "representativeSetFingerprint", "lineage",
  ]) || !isOpaqueId(value.id) || !isBoundedText(value.name, PERSONAL_LIBRARY_MAX_NAME_LENGTH)
    || !isBoundedText(value.description, PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH)
    || !isDiscoveryCues(value.discoveryCues)
    || !isFingerprint(value.representativeSetFingerprint)
    || !isExactObject(value.lineage, ["candidateIds"])
    || !isOpaqueIdArray(value.lineage.candidateIds, false, PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS)
    || !value.lineage.candidateIds.includes(value.id)) return null;
  const representatives = decodeRepresentatives(value.representatives);
  if (!representatives
    || createPersonalLibraryRepresentativeSetFingerprint(representatives)
      !== value.representativeSetFingerprint) return null;
  return {
    id: value.id,
    name: value.name,
    description: value.description,
    discoveryCues: [...value.discoveryCues],
    representatives,
    representativeSetFingerprint: value.representativeSetFingerprint,
    lineage: { candidateIds: [...value.lineage.candidateIds] },
  };
}

function decodeDirection(value: unknown): PersonalLibraryConfirmedDirection | null {
  if (!isPlainObject(value)) return null;
  const merged = value.status === "merged";
  const keys = [
    "id", "status", "name", "description", "discoveryCues", "representatives",
    "representativeSetFingerprint", "lineage", "createdAt", "updatedAt",
    ...(merged ? ["mergedIntoDirectionId"] : []),
  ];
  if (!isExactObject(value, keys)
    || (value.status !== "active" && value.status !== "disabled" && !merged)
    || !isOpaqueId(value.id)
    || !isBoundedText(value.name, PERSONAL_LIBRARY_MAX_NAME_LENGTH)
    || !isBoundedText(value.description, PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH)
    || !isDiscoveryCues(value.discoveryCues)
    || !isFingerprint(value.representativeSetFingerprint)
    || !isExactObject(value.lineage, ["proposalIds", "candidateIds", "directionIds"])
    || !isOpaqueIdArray(value.lineage.proposalIds, false, PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS)
    || !isOpaqueIdArray(value.lineage.candidateIds, true, PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS)
    || !isOpaqueIdArray(value.lineage.directionIds, true, PERSONAL_LIBRARY_MAX_DIRECTION_ANCESTRY_IDS)
    || !isCanonicalTimestamp(value.createdAt)
    || !isCanonicalTimestamp(value.updatedAt)
    || value.createdAt > value.updatedAt
    || (merged && !isOpaqueId(value.mergedIntoDirectionId))) return null;
  const representatives = decodeRepresentatives(value.representatives);
  if (!representatives
    || createPersonalLibraryRepresentativeSetFingerprint(representatives)
      !== value.representativeSetFingerprint) return null;
  const common: PersonalLibraryConfirmedDirectionCommon = {
    id: value.id,
    name: value.name,
    description: value.description,
    discoveryCues: [...value.discoveryCues],
    representatives,
    representativeSetFingerprint: value.representativeSetFingerprint,
    lineage: {
      proposalIds: [...value.lineage.proposalIds],
      candidateIds: [...value.lineage.candidateIds],
      directionIds: [...value.lineage.directionIds],
    },
    createdAt: value.createdAt,
    updatedAt: value.updatedAt,
  };
  return merged
    ? { ...common, status: "merged", mergedIntoDirectionId: value.mergedIntoDirectionId }
    : { ...common, status: value.status as "active" | "disabled" };
}

function decodeCatalogInputManifest(value: unknown): PersonalLibraryRepresentativeEvidence[] | null {
  if (!Array.isArray(value) || value.length === 0
    || value.length > PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS) return null;
  const manifest: PersonalLibraryRepresentativeEvidence[] = [];
  for (const raw of value) {
    if (!isExactObject(raw, ["paperKey", "evidenceFingerprint"])
      || !isCanonicalArxivPaperKey(raw.paperKey)
      || !isFingerprint(raw.evidenceFingerprint)) return null;
    manifest.push({ paperKey: raw.paperKey, evidenceFingerprint: raw.evidenceFingerprint });
  }
  return isStrictlyOrderedUnique(manifest.map(({ paperKey }) => paperKey)) ? manifest : null;
}

function decodeRepresentatives(value: unknown): PersonalLibraryRepresentativeEvidence[] | null {
  if (!Array.isArray(value)
    || value.length < PERSONAL_LIBRARY_MIN_REPRESENTATIVES
    || value.length > PERSONAL_LIBRARY_MAX_REPRESENTATIVES) return null;
  const representatives: PersonalLibraryRepresentativeEvidence[] = [];
  for (const raw of value) {
    if (!isExactObject(raw, ["paperKey", "evidenceFingerprint"])
      || !isCanonicalArxivPaperKey(raw.paperKey)
      || !isFingerprint(raw.evidenceFingerprint)) return null;
    representatives.push({ paperKey: raw.paperKey, evidenceFingerprint: raw.evidenceFingerprint });
  }
  return isStrictlyOrderedUnique(representatives.map(({ paperKey }) => paperKey))
    ? representatives
    : null;
}

function isCanonicalCatalogPaper(value: unknown): value is PersonalLibraryPaperRecord {
  if (!isExactObject(value, [
    "paperKey", "source", "externalId", "title", "authors", "abstract", "published", "updated",
    "primaryCategory", "categories", "evidenceDepth", "filePaths",
  ]) || !isCanonicalArxivPaperKey(value.paperKey)
    || value.source !== "arxiv" || typeof value.externalId !== "string") return false;
  let canonicalKey: string;
  try {
    canonicalKey = paperKeyFromArxivId(value.externalId);
  } catch {
    return false;
  }
  return canonicalKey === value.paperKey
    && value.externalId === value.paperKey.slice("arxiv:".length)
    && isNonEmptyString(value.title)
    && isNonEmptyStringArray(value.authors)
    && value.authors.length > 0
    && typeof value.abstract === "string"
    && isCanonicalTimestamp(value.published)
    && isCanonicalTimestamp(value.updated)
    && isNonEmptyString(value.primaryCategory)
    && isNonEmptyStringArray(value.categories)
    && value.categories.length > 0
    && new Set(value.categories).size === value.categories.length
    && value.categories.includes(value.primaryCategory)
    && value.evidenceDepth === "metadata-and-abstract"
    && isLogicalPathArray(value.filePaths);
}

function mergedChainTerminatesAt(
  ancestorId: string,
  directionId: string,
  byId: Map<string, PersonalLibraryConfirmedDirection>,
): boolean {
  let current = byId.get(ancestorId);
  const visited = new Set<string>();
  while (current?.status === "merged") {
    if (visited.has(current.id)) return false;
    visited.add(current.id);
    if (current.mergedIntoDirectionId === directionId) return true;
    current = byId.get(current.mergedIntoDirectionId);
  }
  return false;
}

function hasMergeCycle(directions: PersonalLibraryConfirmedDirection[]): boolean {
  const byId = new Map(directions.map((direction) => [direction.id, direction]));
  for (const start of directions) {
    const visited = new Set<string>();
    let current: PersonalLibraryConfirmedDirection | undefined = start;
    while (current?.status === "merged") {
      if (visited.has(current.id)) return true;
      visited.add(current.id);
      current = byId.get(current.mergedIntoDirectionId);
    }
  }
  return false;
}

function isCanonicalArxivPaperKey(value: unknown): value is string {
  if (typeof value !== "string" || !value.startsWith("arxiv:")) return false;
  try {
    return paperKeyFromArxivId(value.slice("arxiv:".length)) === value;
  } catch {
    return false;
  }
}

function isDiscoveryCues(value: unknown): value is string[] {
  return Array.isArray(value)
    && value.length >= PERSONAL_LIBRARY_MIN_DISCOVERY_CUES
    && value.length <= PERSONAL_LIBRARY_MAX_DISCOVERY_CUES
    && value.every((cue) => isBoundedText(cue, PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH))
    && isStrictlyOrderedUnique(value);
}

function isOpaqueIdArray(value: unknown, allowEmpty: boolean, maximum: number): value is string[] {
  return Array.isArray(value)
    && (allowEmpty || value.length > 0)
    && value.length <= maximum
    && value.every(isOpaqueId)
    && isStrictlyOrderedUnique(value);
}

function isOpaqueId(value: unknown): value is string {
  return typeof value === "string"
    && value.length >= 1
    && value.length <= PERSONAL_LIBRARY_MAX_ID_LENGTH
    && /^[A-Za-z0-9._~-]+$/.test(value);
}

function isBoundedText(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum
    && value.trim() === value;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isNonEmptyStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(isNonEmptyString);
}

function isLogicalPathArray(value: unknown): value is string[] {
  return Array.isArray(value)
    && value.length > 0
    && value.every(isLogicalRelativePath)
    && isStrictlyOrderedUnique(value);
}

function isLogicalRelativePath(value: unknown): value is string {
  return typeof value === "string"
    && value.length > 0
    && !value.includes("\\")
    && !value.includes("\0")
    && !value.startsWith("/")
    && !/^[A-Za-z]:/.test(value)
    && value.split("/").every((segment) => segment.length > 0 && segment !== "." && segment !== "..");
}

function isCanonicalTimestamp(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}

function isFingerprint(value: unknown): value is string {
  return typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
}

function fingerprint(value: unknown): string {
  return `sha256:${sha256Hex(JSON.stringify(value))}`;
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function isStrictlyOrderedUnique(value: readonly string[]): boolean {
  return value.every((item, index) => index === 0 || codeUnitCompare(value[index - 1]!, item) < 0);
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isPlainObject(value: unknown): value is Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isExactObject(value: unknown, keys: readonly string[]): value is Record<string, any> {
  if (!isPlainObject(value)) return false;
  const actual = Object.keys(value).sort(codeUnitCompare);
  const expected = [...keys].sort(codeUnitCompare);
  return actual.length === expected.length
    && actual.every((key, index) => key === expected[index]);
}
