import {
  decodePersonalLibraryCatalog,
  type PersonalLibraryPaperRecord,
} from "./personal-library-catalog";
import { paperKeyFromArxivId } from "../services/paper-key";
import { sha256Hex } from "../utils/digest";

export const PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION = 3 as const;
export const PERSONAL_LIBRARY_MIN_PROPOSAL_CANDIDATES = 0 as const;
export const PERSONAL_LIBRARY_MAX_PROPOSAL_CANDIDATES = 12 as const;
export const PERSONAL_LIBRARY_MIN_REPRESENTATIVES = 1 as const;
export const PERSONAL_LIBRARY_MAX_REPRESENTATIVES = 5 as const;
export const PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS = 1_000 as const;
export const PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS = 12 as const;
export const PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS = 12 as const;
export const PERSONAL_LIBRARY_MAX_ID_LENGTH = 128 as const;
export const PERSONAL_LIBRARY_MAX_NAME_LENGTH = 120 as const;
export const PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH = 1_000 as const;
export const PERSONAL_LIBRARY_MIN_DISCOVERY_CUES = 1 as const;
export const PERSONAL_LIBRARY_MAX_DISCOVERY_CUES = 12 as const;
export const PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH = 200 as const;
export const PERSONAL_LIBRARY_MAX_GENERATION_CONTRACT_LENGTH = 4_096 as const;
export const PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS = 512 as const;

export interface PersonalLibraryRepresentativeEvidence {
  paperKey: string;
  evidenceFingerprint: string;
}

export interface PersonalLibraryClusterMember {
  paperKey: string;
  confidence: number;
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
  /** Cluster membership confidence produced by the clustering proposer; absent in legacy candidates. */
  clusterMembers?: PersonalLibraryClusterMember[];
}

/**
 * A research direction needs more than one paper behind it: a single
 * representative is a paper, not a direction. Candidates below this bar are
 * still proposed and still confirmable one by one, but they are marked and
 * left unselected when a group is accepted at once, so including one stays a
 * deliberate act (ADR 0009 §3).
 */
export const PERSONAL_LIBRARY_MIN_UNMARKED_REPRESENTATIVES = 2 as const;

export function isThinEvidenceDirectionCandidate(
  candidate: Pick<PersonalLibraryDirectionCandidate, "representatives">,
): boolean {
  return candidate.representatives.length < PERSONAL_LIBRARY_MIN_UNMARKED_REPRESENTATIVES;
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
  ])
    // Schema 2 proposals remain readable: candidates gained only the optional clusterMembers field in v3.
    || (value.schemaVersion !== PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION && value.schemaVersion !== 2)
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

function decodeCandidate(value: unknown): PersonalLibraryDirectionCandidate | null {
  const hasClusterMembers = isPlainObject(value) && Object.hasOwn(value, "clusterMembers");
  const keys = [
    "id", "name", "description", "discoveryCues", "representatives",
    "representativeSetFingerprint", "lineage",
    ...(hasClusterMembers ? ["clusterMembers"] : []),
  ];
  if (!isExactObject(value, keys)
    || !isOpaqueId(value.id) || !isBoundedText(value.name, PERSONAL_LIBRARY_MAX_NAME_LENGTH)
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
  let clusterMembers: PersonalLibraryClusterMember[] | undefined;
  if (hasClusterMembers) {
    const decoded = decodeClusterMembers(value.clusterMembers);
    if (!decoded) return null;
    clusterMembers = decoded;
  }
  return {
    id: value.id,
    name: value.name,
    description: value.description,
    discoveryCues: [...value.discoveryCues],
    representatives,
    representativeSetFingerprint: value.representativeSetFingerprint,
    lineage: { candidateIds: [...value.lineage.candidateIds] },
    ...(clusterMembers !== undefined ? { clusterMembers } : {}),
  };
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

function decodeClusterMembers(value: unknown): PersonalLibraryClusterMember[] | null {
  if (!Array.isArray(value) || value.length > PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS) return null;
  const members: PersonalLibraryClusterMember[] = [];
  const seen = new Set<string>();
  for (const raw of value) {
    if (!isExactObject(raw, ["paperKey", "confidence"])
      || typeof raw.paperKey !== "string"
      || raw.paperKey.length === 0
      || raw.paperKey.length > PERSONAL_LIBRARY_MAX_ID_LENGTH
      || typeof raw.confidence !== "number"
      || !Number.isFinite(raw.confidence)
      || raw.confidence < 0
      || raw.confidence > 1
      || seen.has(raw.paperKey)) return null;
    seen.add(raw.paperKey);
    members.push({ paperKey: raw.paperKey, confidence: raw.confidence });
  }
  return members;
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
