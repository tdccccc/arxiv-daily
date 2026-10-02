import { describe, expect, it } from "vitest";
import {
  PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS,
  PERSONAL_LIBRARY_MAX_DIRECTIONS,
  PERSONAL_LIBRARY_MAX_PROPOSAL_CANDIDATES,
  PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS,
  PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS,
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
  createPersonalLibraryCatalogInputFingerprint,
  createPersonalLibraryCatalogInputManifest,
  createPersonalLibraryGenerationContractFingerprint,
  createPersonalLibraryPaperEvidenceFingerprint,
  createPersonalLibraryRepresentativeSetFingerprint,
  decodeDurablePersonalLibraryInterestProfile,
  decodePersonalLibraryDirectionProposal,
  decodePersonalLibraryInterestProfile,
  decodePersistedPersonalLibraryInterestProfile,
  evaluatePersonalLibraryInterestEligibility,
  type PersonalLibraryConfirmedDirection,
  type PersonalLibraryDirectionProposal,
  type PersonalLibraryInterestProfile,
  type PersonalLibraryRepresentativeEvidence,
} from "../src/library/personal-library-interest-profile";
import type {
  PersonalLibraryCatalog,
  PersonalLibraryPaperRecord,
} from "../src/library/personal-library-catalog";

const scopeFingerprint = `sha256:${"a".repeat(64)}`;
const identificationFingerprint = `sha256:${"b".repeat(64)}`;
const now = "2026-08-03T12:00:00.000Z";

function paper(externalId: string, overrides: Partial<PersonalLibraryPaperRecord> = {}): PersonalLibraryPaperRecord {
  return {
    paperKey: `arxiv:${externalId}`,
    source: "arxiv",
    externalId,
    title: `Paper ${externalId}`,
    authors: ["A. Author", "B. Author"],
    abstract: `Abstract ${externalId}`,
    published: "2026-08-01T00:00:00.000Z",
    updated: "2026-08-02T00:00:00.000Z",
    primaryCategory: "cs.AI",
    categories: ["cs.AI", "cs.LG"],
    evidenceDepth: "metadata-and-abstract",
    filePaths: [`papers/${externalId}.pdf`],
    ...overrides,
  };
}

function catalog(entries = [paper("2608.00001"), paper("2608.00002")]): PersonalLibraryCatalog {
  return {
    schemaVersion: 1,
    revision: 7,
    scopeFingerprint,
    identificationFingerprint,
    updatedAt: now,
    lastScan: null,
    files: Object.fromEntries(entries.map((entry, index) => [entry.filePaths[0]!, {
      path: entry.filePaths[0]!,
      status: "ready" as const,
      observationFingerprint: `sha256:${String(index % 10).repeat(64)}`,
      paperKey: entry.paperKey,
      arxivId: entry.externalId,
      updatedAt: now,
    }])),
    papers: Object.fromEntries(entries.map((entry) => [entry.paperKey, entry])),
  };
}

function representative(entry = paper("2608.00001")): PersonalLibraryRepresentativeEvidence {
  return { paperKey: entry.paperKey, evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(entry) };
}

function proposal(): PersonalLibraryDirectionProposal {
  const representatives = [representative()];
  return {
    schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
    revision: 0,
    proposalId: "proposal.1",
    scopeFingerprint,
    identificationFingerprint,
    catalogInputFingerprint: createPersonalLibraryCatalogInputFingerprint({
      scopeFingerprint,
      identificationFingerprint,
      papers: Object.values(catalog().papers),
    }),
    catalogInputPapers: createPersonalLibraryCatalogInputManifest(Object.values(catalog().papers)),
    generationContractFingerprint: createPersonalLibraryGenerationContractFingerprint("contract-v1"),
    generatedAt: now,
    topics: [{ id: "topic.1", suggestedName: "Efficient language models", directions: [{
      id: "candidate.1",
      text: "Methods that reduce language model inference cost.",
      discoveryCues: ["efficient inference", "model compression"],
      representatives,
      representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
      lineage: { candidateIds: ["candidate.1", "historical.1"] },
    }] }],
  };
}

describe("traceable existing coverage", () => {
  function covered() {
    const original = proposal();
    const paperKeys = original.catalogInputPapers.map(({ paperKey }) => paperKey);
    return {
      ...original, topics: [], coveredPaperKeys: paperKeys,
      coverageEvidence: [{ topicId: "existing", directionId: "direction", directionText: "Research agents", paperKeys }],
    };
  }

  it("round-trips the exact direction text and paper membership underlying coverage", () => {
    const input = covered();
    expect(decodePersonalLibraryDirectionProposal(input)).toEqual(input);
  });

  it.each(["missing", "duplicate", "unknown", "text", "id"])("rejects %s coverage evidence", (kind) => {
    const input = covered();
    const item = input.coverageEvidence[0]!;
    if (kind === "missing") item.paperKeys = [item.paperKeys[0]!];
    if (kind === "duplicate") input.coverageEvidence.push({ ...item });
    if (kind === "unknown") item.paperKeys = ["arxiv:2608.99999"];
    if (kind === "text") item.directionText = "";
    if (kind === "id") item.directionId = "invalid id";
    expect(decodePersonalLibraryDirectionProposal(input)).toBeNull();
  });
});

function direction(
  id: string,
  status: "active" | "disabled" | "merged" = "active",
  options: {
    entry?: PersonalLibraryPaperRecord;
    target?: string;
    ancestors?: string[];
  } = {},
): PersonalLibraryConfirmedDirection {
  const representatives = [representative(options.entry)];
  const common = {
    id,
    name: `Direction ${id}`,
    description: "A researcher-confirmed direction.",
    discoveryCues: ["cue one", "cue two"],
    representatives,
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    clusterMembers: [],
    timeline: [{ kind: "created" as const, at: now }],
    lineage: {
      proposalIds: ["proposal.1"],
      candidateIds: ["candidate.1"],
      directionIds: options.ancestors ?? [],
    },
    createdAt: now,
    updatedAt: now,
  };
  return status === "merged"
    ? { ...common, status, mergedIntoDirectionId: options.target! }
    : { ...common, status };
}

function profile(directions: PersonalLibraryConfirmedDirection[] = [direction("direction.1")]): PersonalLibraryInterestProfile {
  return {
    schemaVersion: 3,
    revision: 3,
    scopeFingerprint,
    identificationFingerprint,
    updatedAt: now,
    directions,
  };
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function expectInvalidRevisions(document: Record<string, unknown>, decode: (value: unknown) => unknown): void {
  for (const revision of [-1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
    expect(decode({ ...document, revision })).toBeNull();
  }
}

describe("personal library fingerprints", () => {
  it("uses explicit bounded unique selection without mutating order", () => {
    const first = paper("2608.00001");
    const second = paper("2608.00002");
    const selected = [second, first];
    const input = { scopeFingerprint, identificationFingerprint, papers: selected };
    expect(createPersonalLibraryCatalogInputFingerprint(input)).toBe(
      createPersonalLibraryCatalogInputFingerprint({ ...input, papers: [first, second] }),
    );
    expect(selected).toEqual([second, first]);
    expect(() => createPersonalLibraryCatalogInputFingerprint({ ...input, papers: [first, first] }))
      .toThrow(/unique/);
    expect(() => createPersonalLibraryCatalogInputFingerprint({
      ...input,
      papers: Array.from({ length: PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS + 1 }, (_, index) => (
        paper(`26${String(index).padStart(2, "0")}.${String(index).padStart(5, "0")}`)
      )),
    })).toThrow(/bounded/);
  });

  it("fingerprints metadata and abstract, preserves author order, treats categories as a set, and excludes paths", () => {
    const original = paper("2608.00001");
    expect(createPersonalLibraryPaperEvidenceFingerprint({ ...original, filePaths: ["moved/paper.pdf"] }))
      .toBe(createPersonalLibraryPaperEvidenceFingerprint(original));
    expect(createPersonalLibraryPaperEvidenceFingerprint({ ...original, categories: ["cs.LG", "cs.AI"] }))
      .toBe(createPersonalLibraryPaperEvidenceFingerprint(original));
    expect(createPersonalLibraryPaperEvidenceFingerprint({ ...original, authors: [...original.authors].reverse() }))
      .not.toBe(createPersonalLibraryPaperEvidenceFingerprint(original));
    expect(createPersonalLibraryPaperEvidenceFingerprint({ ...original, abstract: "Changed." }))
      .not.toBe(createPersonalLibraryPaperEvidenceFingerprint(original));
  });

  it("fingerprints a fallback record on its indexed evidence and excludes paths", () => {
    const original = {
      paperKey: `file:sha256:${"a".repeat(64)}`,
      source: "file" as const,
      title: "A Local Paper",
      abstract: "The abstract the index read.",
      evidenceDepth: "metadata-and-abstract" as const,
      filePaths: ["library/local.pdf"],
    };
    // Renaming the file is not a new paper: identity is the content hash.
    expect(createPersonalLibraryPaperEvidenceFingerprint({ ...original, filePaths: ["moved/local.pdf"] }))
      .toBe(createPersonalLibraryPaperEvidenceFingerprint(original));
    // Re-extracting different evidence for the same file is a new fingerprint,
    // which is what makes a stale proposal detectable.
    expect(createPersonalLibraryPaperEvidenceFingerprint({ ...original, title: "Another Title" }))
      .not.toBe(createPersonalLibraryPaperEvidenceFingerprint(original));
    expect(createPersonalLibraryPaperEvidenceFingerprint({ ...original, abstract: "Changed." }))
      .not.toBe(createPersonalLibraryPaperEvidenceFingerprint(original));
  });

  it("rejects fallback records that are not canonical", () => {
    const original = {
      paperKey: `file:sha256:${"a".repeat(64)}`,
      source: "file" as const,
      title: "A Local Paper",
      abstract: "The abstract the index read.",
      evidenceDepth: "metadata-and-abstract" as const,
      filePaths: ["library/local.pdf"],
    };
    // A key that is not the content-addressed form is not a fallback record,
    // and it is not an arXiv record either — it must not fingerprint at all.
    expect(() => createPersonalLibraryPaperEvidenceFingerprint({ ...original, paperKey: "file:sha256:short" }))
      .toThrow(/exact canonical/);
    expect(() => createPersonalLibraryPaperEvidenceFingerprint({ ...original, title: "" }))
      .toThrow(/exact canonical/);
  });

  it("strictly validates direct paper records while excluding valid file paths from evidence", () => {
    const extra = { ...paper("2608.00001"), unexpected: true };
    expect(() => createPersonalLibraryPaperEvidenceFingerprint(extra)).toThrow(/exact canonical/);
    expect(() => createPersonalLibraryPaperEvidenceFingerprint(paper("2608.00001", { authors: [] })))
      .toThrow(/exact canonical/);
    expect(() => createPersonalLibraryPaperEvidenceFingerprint(paper("2608.00001", { categories: [] })))
      .toThrow(/exact canonical/);
    expect(() => createPersonalLibraryPaperEvidenceFingerprint(paper("2608.00001", { filePaths: ["../escape.pdf"] })))
      .toThrow(/exact canonical/);
    expect(() => createPersonalLibraryPaperEvidenceFingerprint(paper("2608.00001", { filePaths: ["b.pdf", "a.pdf"] })))
      .toThrow(/exact canonical/);
  });
});

describe("reviewed proposal contract", () => {
  it("round-trips covered evidence and explicit target IDs using schema 6", () => {
    const value = proposal();
    value.coveredPaperKeys = ["arxiv:2608.00002"];
    value.topics[0]!.targetTopicId = "settings-topic.1";
    expect(decodePersonalLibraryDirectionProposal(value)).toEqual({ ...value, schemaVersion: 6 });
    expect(decodePersonalLibraryDirectionProposal({ ...value, schemaVersion: 5 })).toBeNull();
  });

  it("accepts a fully covered empty proposal and preserves omitted optional fields", () => {
    const value = { ...proposal(), topics: [], coveredPaperKeys: ["arxiv:2608.00001", "arxiv:2608.00002"] };
    expect(decodePersonalLibraryDirectionProposal(value)).toEqual(value);
    expect(decodePersonalLibraryDirectionProposal(proposal())).toEqual(proposal());
    expect(decodePersonalLibraryDirectionProposal(proposal())).not.toHaveProperty("coveredPaperKeys");
    expect(decodePersonalLibraryDirectionProposal(proposal())!.topics[0]).not.toHaveProperty("targetTopicId");
  });

  it.each([
    null, "arxiv:2608.00001", [17], ["arxiv:2608.00001v2"], ["not-a-paper"],
    ["arxiv:2608.00001", "arxiv:2608.00001"], ["arxiv:2608.00002", "arxiv:2608.00001"], ["arxiv:2608.00099"],
  ].map((coveredPaperKeys) => ({ coveredPaperKeys })))(
    "rejects noncanonical, duplicate, unordered or out-of-manifest coverage $coveredPaperKeys", ({ coveredPaperKeys }) => {
      expect(decodePersonalLibraryDirectionProposal({ ...proposal(), topics: [], coveredPaperKeys })).toBeNull();
    },
  );

  it("rejects covered papers that also occur in candidate evidence", () => {
    const value = proposal();
    expect(decodePersonalLibraryDirectionProposal({ ...value, coveredPaperKeys: ["arxiv:2608.00001"] })).toBeNull();
    value.topics[0]!.directions[0]!.clusterMembers = [
      { paperKey: "arxiv:2608.00001", confidence: 1 },
      { paperKey: "arxiv:2608.00002", confidence: 0.5 },
    ];
    expect(decodePersonalLibraryDirectionProposal({ ...value, coveredPaperKeys: ["arxiv:2608.00002"] })).toBeNull();
  });

  it.each([null, "", " ", "id with spaces", "x".repeat(129)])("rejects malformed persisted target IDs %j", (targetTopicId) => {
    const value = proposal();
    expect(decodePersonalLibraryDirectionProposal({ ...value, topics: [{ ...value.topics[0], targetTopicId }] })).toBeNull();
  });

  it("bounds candidate count across all proposal topics", () => {
    const value = proposal();
    const template = value.topics[0]!.directions[0]!;
    value.topics = [0, 1].map((topicIndex) => ({
      id: `topic.${topicIndex}`, suggestedName: `Topic ${topicIndex}`,
      directions: Array.from({ length: 6 }, (_, index) => {
        const id = `candidate.${topicIndex}.${index}`;
        return { ...template, id, lineage: { candidateIds: [id] } };
      }),
    }));
    expect(decodePersonalLibraryDirectionProposal(value)).toEqual(value);
    value.topics[1]!.directions.push({ ...template, id: "candidate.1.6", lineage: { candidateIds: ["candidate.1.6"] } });
    expect(decodePersonalLibraryDirectionProposal(value)).toBeNull();
  });
});
