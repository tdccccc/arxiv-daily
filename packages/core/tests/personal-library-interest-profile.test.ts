import { describe, expect, it } from "vitest";
import {
  PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS,
  PERSONAL_LIBRARY_MAX_DIRECTIONS,
  PERSONAL_LIBRARY_MAX_PROPOSAL_CANDIDATES,
  PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS,
  PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS,
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
    schemaVersion: 3,
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
    candidates: [{
      id: "candidate.1",
      name: "Efficient language models",
      description: "Methods that reduce inference cost.",
      discoveryCues: ["efficient inference", "model compression"],
      representatives,
      representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
      lineage: { candidateIds: ["candidate.1", "historical.1"] },
    }],
  };
}

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

describe("personal library proposal and profile contracts", () => {
  it("strictly decodes separate exact current-schema documents", () => {
    expect(decodePersonalLibraryDirectionProposal(proposal())).toEqual(proposal());
    expect(decodePersonalLibraryInterestProfile(profile())).toEqual(profile());
    expect(decodePersonalLibraryDirectionProposal({ ...proposal(), schemaVersion: 1 })).toBeNull();
    expect(decodePersonalLibraryInterestProfile({ ...profile(), schemaVersion: 4 })).toBeNull();
    expect(decodePersonalLibraryDirectionProposal({ ...proposal(), unexpected: true })).toBeNull();
    expect(decodePersonalLibraryInterestProfile({ ...profile(), unexpected: true })).toBeNull();
    expect(decodePersonalLibraryDirectionProposal({ ...proposal(), generatedAt: "2026-08-03" })).toBeNull();
    expect(decodePersonalLibraryInterestProfile({ ...profile(), updatedAt: "2026-08-03" })).toBeNull();
    expectInvalidRevisions(proposal() as unknown as Record<string, unknown>, decodePersonalLibraryDirectionProposal);
    expectInvalidRevisions(profile() as unknown as Record<string, unknown>, decodePersonalLibraryInterestProfile);
  });

  it("requires an exact canonical bounded proposal input manifest and matching fingerprint", () => {
    const missing = clone(proposal()) as unknown as Record<string, any>;
    delete missing.catalogInputPapers;
    expect(decodePersonalLibraryDirectionProposal(missing)).toBeNull();
    const empty = clone(proposal());
    empty.catalogInputPapers = [];
    expect(decodePersonalLibraryDirectionProposal(empty)).toBeNull();
    const unsorted = clone(proposal());
    unsorted.catalogInputPapers.reverse();
    expect(decodePersonalLibraryDirectionProposal(unsorted)).toBeNull();
    const duplicate = clone(proposal());
    duplicate.catalogInputPapers = [duplicate.catalogInputPapers[0]!, duplicate.catalogInputPapers[0]!];
    expect(decodePersonalLibraryDirectionProposal(duplicate)).toBeNull();
    const tampered = clone(proposal());
    tampered.catalogInputPapers[0]!.evidenceFingerprint = `sha256:${"f".repeat(64)}`;
    expect(decodePersonalLibraryDirectionProposal(tampered)).toBeNull();
    const legacy = clone(proposal()) as unknown as Record<string, any>;
    legacy.schemaVersion = 1;
    delete legacy.catalogInputPapers;
    expect(decodePersonalLibraryDirectionProposal(legacy)).toBeNull();
  });

  it("rejects nested extra keys, wrong status shapes, and noncanonical timestamps", () => {
    const candidateExtra = clone(proposal()) as unknown as Record<string, any>;
    candidateExtra.candidates[0].lineage.unexpected = true;
    expect(decodePersonalLibraryDirectionProposal(candidateExtra)).toBeNull();

    const representativeExtra = clone(profile()) as unknown as Record<string, any>;
    representativeExtra.directions[0].representatives[0].unexpected = true;
    expect(decodePersonalLibraryInterestProfile(representativeExtra)).toBeNull();

    const activeWithTarget = clone(profile()) as unknown as Record<string, any>;
    activeWithTarget.directions[0].mergedIntoDirectionId = "direction.2";
    expect(decodePersonalLibraryInterestProfile(activeWithTarget)).toBeNull();

    const badTimestamp = clone(profile()) as unknown as Record<string, any>;
    badTimestamp.directions[0].createdAt = "2026-08-03";
    expect(decodePersonalLibraryInterestProfile(badTimestamp)).toBeNull();
  });

  it("enforces candidate, representative, ordering, duplicate, and lineage bounds", () => {
    expect(decodePersonalLibraryDirectionProposal({ ...proposal(), candidates: [] }))
      .toEqual({ ...proposal(), candidates: [] });
    const tooMany = proposal();
    tooMany.candidates = Array.from({ length: PERSONAL_LIBRARY_MAX_PROPOSAL_CANDIDATES + 1 }, (_, index) => ({
      ...proposal().candidates[0]!, id: `candidate.${String(index).padStart(2, "0")}`,
      lineage: { candidateIds: [`candidate.${String(index).padStart(2, "0")}`] },
    }));
    expect(decodePersonalLibraryDirectionProposal(tooMany)).toBeNull();

    const malformed = proposal();
    malformed.candidates[0]!.discoveryCues = ["z", "a"];
    expect(decodePersonalLibraryDirectionProposal(malformed)).toBeNull();
    malformed.candidates[0]!.discoveryCues = ["a", "a"];
    expect(decodePersonalLibraryDirectionProposal(malformed)).toBeNull();

    const missingSelf = proposal();
    missingSelf.candidates[0]!.lineage.candidateIds = ["historical.1"];
    expect(decodePersonalLibraryDirectionProposal(missingSelf)).toBeNull();
    const historicalAllowed = proposal();
    historicalAllowed.candidates[0]!.lineage.candidateIds = ["candidate.1", "removed.from.current.proposal"];
    expect(decodePersonalLibraryDirectionProposal(historicalAllowed)).not.toBeNull();

    const lineageOverflow = proposal();
    lineageOverflow.candidates[0]!.lineage.candidateIds = Array.from(
      { length: PERSONAL_LIBRARY_MAX_CANDIDATE_LINEAGE_IDS + 1 },
      (_, index) => index === 0 ? "candidate.1" : `historical.${String(index).padStart(2, "0")}`,
    );
    expect(decodePersonalLibraryDirectionProposal(lineageOverflow)).toBeNull();

    const candidateOrder = proposal();
    const second = clone(candidateOrder.candidates[0]!);
    second.id = "candidate.0";
    second.lineage.candidateIds = ["candidate.0"];
    candidateOrder.candidates.push(second);
    expect(decodePersonalLibraryDirectionProposal(candidateOrder)).toBeNull();
    second.id = "candidate.1";
    second.lineage.candidateIds = ["candidate.1"];
    expect(decodePersonalLibraryDirectionProposal(candidateOrder)).toBeNull();
  });

  it("rejects representative ordering, duplicates, noncanonical keys, and fingerprint tampering", () => {
    const raw = proposal();
    const second = representative(paper("2608.00002"));
    raw.candidates[0]!.representatives = [second, representative()];
    raw.candidates[0]!.representativeSetFingerprint = createPersonalLibraryRepresentativeSetFingerprint([
      representative(), second,
    ]);
    expect(decodePersonalLibraryDirectionProposal(raw)).toBeNull();

    const duplicate = proposal();
    duplicate.candidates[0]!.representatives = [representative(), representative()];
    expect(decodePersonalLibraryDirectionProposal(duplicate)).toBeNull();
    const noncanonical = proposal();
    noncanonical.candidates[0]!.representatives[0]!.paperKey = "arxiv:2608.00001v2";
    expect(decodePersonalLibraryDirectionProposal(noncanonical)).toBeNull();
    const tampered = proposal();
    tampered.candidates[0]!.representativeSetFingerprint = `sha256:${"f".repeat(64)}`;
    expect(decodePersonalLibraryDirectionProposal(tampered)).toBeNull();
  });

  it("validates retained merge graphs and exact merged ancestry", () => {
    const valid = profile([
      direction("direction.1", "merged", { target: "direction.2" }),
      direction("direction.2", "merged", { target: "direction.3" }),
      direction("direction.3", "active", { ancestors: ["direction.1", "direction.2"] }),
    ]);
    expect(decodePersonalLibraryInterestProfile(valid)).toEqual(valid);
    expect(decodePersonalLibraryInterestProfile(profile([
      direction("direction.1", "merged", { target: "missing" }),
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(profile([
      direction("direction.1", "merged", { target: "direction.1" }),
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(profile([
      direction("direction.1", "merged", { target: "direction.2" }),
      direction("direction.2", "merged", { target: "direction.1" }),
    ]))).toBeNull();

    const danglingAncestry = profile([direction("direction.1", "active", { ancestors: ["missing"] })]);
    expect(decodePersonalLibraryInterestProfile(danglingAncestry)).toBeNull();
    const unrelated = profile([
      direction("direction.1"),
      direction("direction.2", "active", { ancestors: ["direction.1"] }),
    ]);
    expect(decodePersonalLibraryInterestProfile(unrelated)).toBeNull();
    const descendant = profile([
      direction("direction.1", "active", { ancestors: ["direction.2"] }),
      direction("direction.2", "merged", { target: "direction.1" }),
    ]);
    descendant.directions[1]!.lineage.directionIds = ["direction.1"];
    expect(decodePersonalLibraryInterestProfile(descendant)).toBeNull();
  });

  it("strictly migrates exact legacy v1 profiles while current decode remains v3-only", () => {
    const legacy = clone(profile()) as unknown as Record<string, any>;
    legacy.schemaVersion = 1;
    legacy.directions[0].lineage = {
      proposalId: "proposal.1", candidateIds: ["candidate.1"], directionIds: [],
    };
    expect(decodePersonalLibraryInterestProfile(legacy)).toBeNull();
    expect(decodeDurablePersonalLibraryInterestProfile(legacy)).toEqual(profile());
    const malformed = clone(legacy);
    malformed.directions[0].lineage.unexpected = true;
    expect(decodeDurablePersonalLibraryInterestProfile(malformed)).toBeNull();
    const wrongShape = clone(legacy);
    wrongShape.directions[0].lineage.proposalId = " padded ";
    expect(decodeDurablePersonalLibraryInterestProfile(wrongShape)).toBeNull();
    const oldInvalidTimestamp = clone(legacy);
    oldInvalidTimestamp.directions[0].updatedAt = "2030-01-01T00:00:00.000Z";
    expect(decodeDurablePersonalLibraryInterestProfile(oldInvalidTimestamp)).toBeNull();
  });

  it("allows transient chronology but rejects it as persisted or eligible profile state", () => {
    const transient = profile();
    transient.directions[0]!.updatedAt = "2026-08-03T13:00:00.000Z";
    expect(decodePersonalLibraryInterestProfile(transient)).toEqual(transient);
    expect(decodePersistedPersonalLibraryInterestProfile(transient)).toBeNull();
    expect(decodeDurablePersonalLibraryInterestProfile(transient)).toBeNull();
    expect(evaluatePersonalLibraryInterestEligibility(transient, catalog())).toEqual({
      documentDiagnostics: ["profile-invalid"], eligibleDirections: [], diagnostics: [],
    });
  });

  it("requires canonical non-empty bounded confirmed proposal lineage", () => {
    const singular = profile();
    expect(decodePersonalLibraryInterestProfile(singular)).toEqual(singular);
    const legacy = clone(profile()) as unknown as Record<string, any>;
    legacy.directions[0].lineage = {
      proposalId: "proposal.1", candidateIds: ["candidate.1"], directionIds: [],
    };
    expect(decodePersonalLibraryInterestProfile(legacy)).toBeNull();
    const empty = clone(profile());
    empty.directions[0]!.lineage.proposalIds = [];
    expect(decodePersonalLibraryInterestProfile(empty)).toBeNull();
    const unsorted = clone(profile());
    unsorted.directions[0]!.lineage.proposalIds = ["proposal.2", "proposal.1"];
    expect(decodePersonalLibraryInterestProfile(unsorted)).toBeNull();
    const duplicate = clone(profile());
    duplicate.directions[0]!.lineage.proposalIds = ["proposal.1", "proposal.1"];
    expect(decodePersonalLibraryInterestProfile(duplicate)).toBeNull();
    const overflow = clone(profile());
    overflow.directions[0]!.lineage.proposalIds = Array.from(
      { length: PERSONAL_LIBRARY_MAX_PROPOSAL_LINEAGE_IDS + 1 },
      (_, index) => `proposal.${String(index).padStart(2, "0")}`,
    );
    expect(decodePersonalLibraryInterestProfile(overflow)).toBeNull();
  });

  it("bounds profile directions and rejects direction ordering and duplicates", () => {
    const overflow = profile(Array.from({ length: PERSONAL_LIBRARY_MAX_DIRECTIONS + 1 }, (_, index) => (
      direction(`direction.${String(index).padStart(3, "0")}`)
    )));
    expect(decodePersonalLibraryInterestProfile(overflow)).toBeNull();
    expect(decodePersonalLibraryInterestProfile(profile([
      direction("direction.2"), direction("direction.1"),
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(profile([
      direction("direction.1"), direction("direction.1"),
    ]))).toBeNull();
  });

  it("does not repair input and safely copies prototype-sensitive IDs", () => {
    const raw = proposal() as unknown as Record<string, any>;
    raw.candidates[0].id = "__proto__";
    raw.candidates[0].lineage.candidateIds = ["__proto__", "historical.1"];
    const before = JSON.stringify(raw);
    const decoded = decodePersonalLibraryDirectionProposal(raw);
    expect(decoded?.candidates[0]?.id).toBe("__proto__");
    expect(JSON.stringify(raw)).toBe(before);
    decoded!.candidates[0]!.discoveryCues[0] = "changed";
    expect(raw.candidates[0].discoveryCues[0]).toBe("efficient inference");
  });

  it("keeps proposal candidates structurally non-authoritative", () => {
    const candidate = proposal().candidates[0] as unknown as Record<string, unknown>;
    expect(candidate.status).toBeUndefined();
    expect(candidate.eligible).toBeUndefined();
    expect(Object.keys(candidate)).not.toContain("confirmed");
  });
});

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

describe("personal library eligibility", () => {
  it("admits only compatible active directions and ignores path changes and unrelated additions", () => {
    const original = paper("2608.00001");
    const current = catalog([{ ...original, filePaths: ["moved/paper.pdf"] }, paper("2608.99999")]);
    expect(evaluatePersonalLibraryInterestEligibility(profile([direction("direction.1", "active", { entry: original })]), current))
      .toEqual({
        documentDiagnostics: [],
        eligibleDirections: [{
          id: "direction.1",
          name: "Direction direction.1",
          description: "A researcher-confirmed direction.",
          discoveryCues: ["cue one", "cue two"],
          representatives: [representative(original)],
        }],
        diagnostics: [{ directionId: "direction.1", eligible: true, reasons: [] }],
      });
  });

  it("explicitly diagnoses invalid profile and invalid/future/broken catalogs", () => {
    const malformedProfile = { ...profile(), schemaVersion: 4 };
    expect(evaluatePersonalLibraryInterestEligibility(malformedProfile, catalog())).toEqual({
      documentDiagnostics: ["profile-invalid"], eligibleDirections: [], diagnostics: [],
    });
    expect(evaluatePersonalLibraryInterestEligibility(profile(), { ...catalog(), schemaVersion: 2 })).toEqual({
      documentDiagnostics: ["catalog-invalid"], eligibleDirections: [], diagnostics: [],
    });
    const broken = clone(catalog());
    broken.files = {};
    expect(evaluatePersonalLibraryInterestEligibility(profile(), broken)).toEqual({
      documentDiagnostics: ["catalog-invalid"], eligibleDirections: [], diagnostics: [],
    });
    const catalogExtra = { ...catalog(), unexpected: true };
    expect(evaluatePersonalLibraryInterestEligibility(profile(), catalogExtra)).toEqual({
      documentDiagnostics: ["catalog-invalid"], eligibleDirections: [], diagnostics: [],
    });
    expect(evaluatePersonalLibraryInterestEligibility(proposal(), null)).toEqual({
      documentDiagnostics: ["profile-invalid", "catalog-invalid"],
      eligibleDirections: [], diagnostics: [],
    });
  });

  it("diagnoses valid scope and identification mismatches deterministically with zero eligibility", () => {
    const mismatchedCatalog = {
      ...catalog(),
      scopeFingerprint: `sha256:${"c".repeat(64)}`,
      identificationFingerprint: `sha256:${"d".repeat(64)}`,
    };
    const expected = {
      documentDiagnostics: ["profile-scope-mismatch", "profile-identification-mismatch"],
      eligibleDirections: [],
      diagnostics: [{ directionId: "direction.1", eligible: false, reasons: [] }],
    };
    expect(evaluatePersonalLibraryInterestEligibility(profile(), mismatchedCatalog)).toEqual(expected);
    expect(evaluatePersonalLibraryInterestEligibility(profile(), mismatchedCatalog)).toEqual(expected);
  });

  it("deterministically diagnoses disabled, merged, missing, and changed representatives", () => {
    const original = paper("2608.00001");
    const confirmed = profile([
      direction("active", "active", { entry: original }),
      direction("disabled", "disabled", { entry: original }),
      direction("merged", "merged", { entry: original, target: "active" }),
      direction("missing", "active", { entry: paper("2608.00003") }),
    ]);
    const current = catalog([{ ...original, abstract: "Changed evidence." }]);
    const result = evaluatePersonalLibraryInterestEligibility(confirmed, current);
    expect(result).toEqual({
      documentDiagnostics: [], eligibleDirections: [], diagnostics: [
        { directionId: "active", eligible: false, reasons: [
          { reason: "representative-evidence-changed", paperKey: original.paperKey },
        ] },
        { directionId: "disabled", eligible: false, reasons: [
          { reason: "direction-disabled" },
          { reason: "representative-evidence-changed", paperKey: original.paperKey },
        ] },
        { directionId: "merged", eligible: false, reasons: [
          { reason: "direction-merged" },
          { reason: "representative-evidence-changed", paperKey: original.paperKey },
        ] },
        { directionId: "missing", eligible: false, reasons: [
          { reason: "representative-missing", paperKey: "arxiv:2608.00003" },
        ] },
      ],
    });
    expect(evaluatePersonalLibraryInterestEligibility(confirmed, current)).toEqual(result);
  });
});
