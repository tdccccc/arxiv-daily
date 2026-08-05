import { describe, expect, it } from "vitest";
import type { PersonalLibraryCatalog, PersonalLibraryPaperRecord } from "../src/library/personal-library-catalog";
import {
  createEmptyPersonalLibraryInterestProfile,
  createPersonalLibraryCatalogInputManifest,
  createPersonalLibraryCatalogInputManifestFingerprint,
  createPersonalLibraryGenerationContractFingerprint,
  createPersonalLibraryPaperEvidenceFingerprint,
  createPersonalLibraryRepresentativeSetFingerprint,
  decodePersonalLibraryDirectionProposal,
  decodePersonalLibraryInterestProfile,
  evaluatePersonalLibraryInterestEligibility,
  type PersonalLibraryConfirmedDirection,
  type PersonalLibraryDirectionProposal,
  type PersonalLibraryInterestProfile,
} from "../src/library/personal-library-interest-profile";
import {
  confirmPersonalLibraryDirectionCandidate,
  disablePersonalLibraryConfirmedDirection,
  enablePersonalLibraryConfirmedDirection,
  mergePersonalLibraryConfirmedDirections,
  mergePersonalLibraryDirectionCandidates,
  removePersonalLibraryConfirmedDirection,
  removePersonalLibraryDirectionCandidate,
  updatePersonalLibraryConfirmedDirection,
  updatePersonalLibraryDirectionCandidate,
} from "../src/library/personal-library-interest-profile-review";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const t0 = "2026-08-03T10:00:00.000Z";
const t1 = new Date("2026-08-03T11:00:00.000Z");
const t2 = new Date("2026-08-03T12:00:00.000Z");

function paper(id: string, overrides: Partial<PersonalLibraryPaperRecord> = {}): PersonalLibraryPaperRecord {
  return {
    paperKey: `arxiv:${id}`, source: "arxiv", externalId: id, title: `Paper ${id}`,
    authors: ["Researcher"], abstract: `Abstract ${id}`,
    published: "2026-08-01T00:00:00.000Z", updated: "2026-08-02T00:00:00.000Z",
    primaryCategory: "cs.AI", categories: ["cs.AI"], evidenceDepth: "metadata-and-abstract",
    filePaths: [`papers/${id}.pdf`], ...overrides,
  };
}

function catalog(entries = [paper("2608.00001"), paper("2608.00002"), paper("2608.00003")]): PersonalLibraryCatalog {
  return {
    schemaVersion: 1, revision: 1, scopeFingerprint: scope, identificationFingerprint: identification,
    updatedAt: t0, lastScan: null,
    files: Object.fromEntries(entries.map((entry, index) => [entry.filePaths[0]!, {
      path: entry.filePaths[0]!, status: "ready" as const,
      observationFingerprint: `sha256:${String((index + 1) % 10).repeat(64)}`,
      paperKey: entry.paperKey, arxivId: entry.externalId, updatedAt: t0,
    }])),
    papers: Object.fromEntries(entries.map((entry) => [entry.paperKey, entry])),
  };
}

function candidate(id: string, paperKey = "arxiv:2608.00001") {
  const entry = catalog().papers[paperKey]!;
  const representatives = [{ paperKey, evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(entry) }];
  return {
    id, name: `Candidate ${id}`, description: `Description ${id}`, discoveryCues: [`cue ${id}`],
    representatives, representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    lineage: { candidateIds: [id] },
  };
}

function proposal(ids = ["candidate.1", "candidate.2"]): PersonalLibraryDirectionProposal {
  const catalogInputPapers = createPersonalLibraryCatalogInputManifest([
    catalog().papers["arxiv:2608.00001"]!,
  ]);
  return {
    schemaVersion: 3, revision: 7, proposalId: "proposal.1", scopeFingerprint: scope,
    identificationFingerprint: identification,
    catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
      scopeFingerprint: scope, identificationFingerprint: identification, catalogInputPapers,
    }),
    catalogInputPapers,
    generationContractFingerprint: createPersonalLibraryGenerationContractFingerprint("review-test"),
    generatedAt: t0, candidates: ids.map((id, index) => candidate(id, `arxiv:2608.0000${index + 1}`)),
  };
}

function draft(paperKeys = ["arxiv:2608.00001"], name = "Reviewed direction") {
  return {
    name, description: "Researcher reviewed description.",
    discoveryCues: ["reviewed cue", "second cue"], representativePaperKeys: paperKeys,
  };
}

function profile(directions: PersonalLibraryConfirmedDirection[] = []): PersonalLibraryInterestProfile {
  return { ...createEmptyPersonalLibraryInterestProfile(scope, identification, new Date(t0)), revision: 4, directions };
}

function direction(id: string, proposalId: string, candidateId: string, status: "active" | "disabled" = "active") {
  const representative = {
    paperKey: "arxiv:2608.00001",
    evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(catalog().papers["arxiv:2608.00001"]!),
  };
  return {
    id, status, name: `Direction ${id}`, description: "Confirmed.", discoveryCues: ["confirmed cue"],
    representatives: [representative],
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint([representative]),
    clusterMembers: [],
    timeline: [{ kind: "created" as const, at: t0 }],
    lineage: { proposalIds: [proposalId], candidateIds: [candidateId], directionIds: [] },
    createdAt: t0, updatedAt: t0,
  } satisfies PersonalLibraryConfirmedDirection;
}

function frozen<T>(value: T): T {
  const visit = (item: any): any => {
    if (item && typeof item === "object") {
      Object.values(item).forEach(visit);
      Object.freeze(item);
    }
    return item;
  };
  return visit(value);
}

describe("candidate review transactions", () => {
  it("updates text locally, updates representatives from strict compatible evidence, and never confirms", () => {
    const original = frozen(proposal());
    const text = updatePersonalLibraryDirectionCandidate({
      proposal: original, candidateId: "candidate.1", patch: { name: "Corrected" },
    });
    expect(text.candidates[0]).toMatchObject({ id: "candidate.1", name: "Corrected" });
    expect((text.candidates[0] as any).status).toBeUndefined();
    expect(original.candidates[0]!.name).toBe("Candidate candidate.1");
    const reps = updatePersonalLibraryDirectionCandidate({
      proposal: text, candidateId: "candidate.1", patch: { description: "Corrected detail" },
      representativePaperKeys: ["arxiv:2608.00003"], catalog: catalog(),
    });
    expect(reps.candidates[0]!.representatives).toEqual([{
      paperKey: "arxiv:2608.00003",
      evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(catalog().papers["arxiv:2608.00003"]!),
    }]);
    expect(reps.catalogInputFingerprint).toBe(original.catalogInputFingerprint);
    expect(reps.catalogInputPapers).toEqual(original.catalogInputPapers);
    expect(decodePersonalLibraryDirectionProposal(reps)).toEqual(reps);
  });

  it("merges at least two candidates with fresh ID, reviewed draft, and complete bounded lineage", () => {
    const merged = mergePersonalLibraryDirectionCandidates({
      proposal: frozen(proposal()), sourceCandidateIds: ["candidate.1", "candidate.2"],
      candidateId: "candidate.3", draft: draft(["arxiv:2608.00002"]), catalog: catalog(),
    });
    expect(merged.candidates).toHaveLength(1);
    expect(merged.candidates[0]).toMatchObject({
      id: "candidate.3", name: "Reviewed direction",
      lineage: { candidateIds: ["candidate.1", "candidate.2", "candidate.3"] },
    });
    expect(() => mergePersonalLibraryDirectionCandidates({
      proposal: proposal(), sourceCandidateIds: ["candidate.1"], candidateId: "candidate.3",
      draft: draft(), catalog: catalog(),
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
  });

  it("removes any candidate including the final candidate and rejects repair-like inputs", () => {
    const result = removePersonalLibraryDirectionCandidate({ proposal: proposal(["candidate.1"]), candidateId: "candidate.1" });
    expect(result.candidates).toEqual([]);
    expect(() => updatePersonalLibraryDirectionCandidate({
      proposal: proposal(), candidateId: "candidate.1", patch: { name: " padded " },
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
    expect(() => mergePersonalLibraryDirectionCandidates({
      proposal: proposal(), sourceCandidateIds: ["candidate.2", "candidate.1"], candidateId: "candidate.3",
      draft: draft(), catalog: catalog(),
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
  });
});

describe("confirmation authority transaction", () => {
  it("is the only candidate-to-profile operation and preserves document revision/timestamps", () => {
    const originalProposal = frozen(proposal());
    const originalProfile = frozen(profile());
    const result = confirmPersonalLibraryDirectionCandidate({
      proposal: originalProposal, profile: originalProfile, catalog: frozen(catalog()),
      candidateId: "candidate.1", directionId: "direction.1", status: "active", draft: draft(), now: t1,
    });
    expect(result.proposal).toMatchObject({ revision: 7, generatedAt: t0 });
    expect(result.proposal.candidates.map(({ id }) => id)).toEqual(["candidate.2"]);
    expect(result.profile).toMatchObject({ revision: 4, updatedAt: t0 });
    expect(result.profile.directions[0]).toMatchObject({
      id: "direction.1", status: "active",
      lineage: { proposalIds: ["proposal.1"], candidateIds: ["candidate.1"], directionIds: [] },
      createdAt: t1.toISOString(), updatedAt: t1.toISOString(),
    });
    expect(decodePersonalLibraryInterestProfile(result.profile)).toEqual(result.profile);
    expect(originalProfile.directions).toEqual([]);
  });

  it("supports exact recovery before or after candidate consumption and rejects conflicts", () => {
    const first = confirmPersonalLibraryDirectionCandidate({
      proposal: proposal(), profile: profile(), catalog: catalog(), candidateId: "candidate.1",
      directionId: "direction.1", status: "active", draft: draft(), now: t1,
    });
    const beforeConsumption = confirmPersonalLibraryDirectionCandidate({
      proposal: proposal(), profile: first.profile, catalog: catalog(), candidateId: "candidate.1",
      directionId: "direction.1", status: "active", draft: draft(), now: t2,
    });
    expect(beforeConsumption.profile).toEqual(first.profile);
    expect(beforeConsumption.proposal.candidates.map(({ id }) => id)).toEqual(["candidate.2"]);
    const afterConsumption = confirmPersonalLibraryDirectionCandidate({
      proposal: first.proposal, profile: first.profile, catalog: catalog(), candidateId: "candidate.1",
      directionId: "direction.1", status: "active", draft: draft(), now: t2,
    });
    expect(afterConsumption).toEqual(first);
    expect(() => confirmPersonalLibraryDirectionCandidate({
      proposal: proposal(), profile: first.profile, catalog: catalog(), candidateId: "candidate.1",
      directionId: "direction.1", status: "active", draft: draft([], "Different"), now: t2,
    })).toThrow();
    expect(() => confirmPersonalLibraryDirectionCandidate({
      proposal: proposal(), profile: first.profile, catalog: catalog(), candidateId: "candidate.1",
      directionId: "direction.other", status: "active", draft: draft(), now: t2,
    })).toThrow(expect.objectContaining({ code: "conflict" }));
  });

  it("binds only selected manifest evidence and tolerates unrelated catalog evolution", () => {
    const base = { proposal: proposal(), profile: profile(), candidateId: "candidate.1",
      directionId: "direction.1", status: "active", draft: draft(), now: t1 };
    expect(() => confirmPersonalLibraryDirectionCandidate({
      ...base, catalog: catalog([
        paper("2608.00001", { abstract: "changed" }), paper("2608.00002"), paper("2608.00003"),
      ]),
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    expect(() => confirmPersonalLibraryDirectionCandidate({
      ...base, catalog: catalog([paper("2608.00002"), paper("2608.00003")]),
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    expect(() => confirmPersonalLibraryDirectionCandidate({
      ...base, catalog: catalog([...Object.values(catalog().papers), paper("2608.00004")]),
    })).not.toThrow();
    expect(() => confirmPersonalLibraryDirectionCandidate({
      ...base, catalog: catalog([
        paper("2608.00001"), paper("2608.00002", { abstract: "unrelated change" }), paper("2608.00003"),
      ]),
    })).not.toThrow();
  });

  it("confirms against a small manifest when the current catalog exceeds 1000 papers", () => {
    const entries = [paper("2608.00001"), ...Array.from({ length: 1001 }, (_, index) => (
      paper(`25${String(Math.floor(index / 100000)).padStart(2, "0")}.${String(index).padStart(5, "0")}`)
    ))];
    const result = confirmPersonalLibraryDirectionCandidate({
      proposal: proposal(), profile: profile(), catalog: catalog(entries), candidateId: "candidate.1",
      directionId: "direction.1", status: "active", draft: draft(), now: t1,
    });
    expect(result.profile.directions).toHaveLength(1);
  });

  it("rejects invalid dates, IDs, incompatible catalogs, and missing representatives", () => {
    const base = { proposal: proposal(), profile: profile(), catalog: catalog(), candidateId: "candidate.1",
      directionId: "direction.1", status: "active", draft: draft(), now: t1 };
    expect(() => confirmPersonalLibraryDirectionCandidate({ ...base, now: new Date(Number.NaN) }))
      .toThrow(expect.objectContaining({ code: "invalid-input" }));
    const hostileDate = new Date(t1);
    hostileDate.getTime = () => Number.NaN;
    hostileDate.toISOString = () => { throw new Error("overridden"); };
    const accepted = confirmPersonalLibraryDirectionCandidate({ ...base, now: hostileDate });
    expect(accepted.profile.directions[0]!.createdAt).toBe(t1.toISOString());
    expect(() => confirmPersonalLibraryDirectionCandidate({ ...base, directionId: " bad " }))
      .toThrow(expect.objectContaining({ code: "invalid-input" }));
    expect(() => confirmPersonalLibraryDirectionCandidate({
      ...base, catalog: { ...catalog(), scopeFingerprint: `sha256:${"c".repeat(64)}` },
    })).toThrow(expect.objectContaining({ code: "incompatible-catalog" }));
    expect(() => confirmPersonalLibraryDirectionCandidate({ ...base, draft: draft(["arxiv:2608.99999"]) }))
      .toThrow(expect.objectContaining({ code: "evidence-mismatch" }));
  });
});

describe("confirmed direction review transactions", () => {
  it("edits text or reviewed representatives while preserving status, creation, lineage, and monotonic time", () => {
    const original = profile([direction("direction.1", "proposal.1", "candidate.1")]);
    const text = updatePersonalLibraryConfirmedDirection({
      profile: frozen(original), directionId: "direction.1", patch: { name: "Edited" },
      now: new Date("2020-01-01T00:00:00.000Z"),
    });
    expect(text.directions[0]).toMatchObject({
      name: "Edited", status: "active", createdAt: t0, updatedAt: t0,
      lineage: original.directions[0]!.lineage,
    });
    const repaired = updatePersonalLibraryConfirmedDirection({
      profile: text, directionId: "direction.1", patch: { discoveryCues: ["repaired cue"] },
      representativePaperKeys: ["arxiv:2608.00002"], catalog: catalog(), now: t2,
    });
    expect(repaired.directions[0]!.representatives[0]!.paperKey).toBe("arxiv:2608.00002");
    expect(repaired.directions[0]!.updatedAt).toBe(t2.toISOString());
    expect(repaired.directions[0]!.status).toBe("active");
  });

  it("disables without catalog; enables only exact compatible existing evidence; no-ops preserve time", () => {
    const active = profile([direction("direction.1", "proposal.1", "candidate.1")]);
    expect(disablePersonalLibraryConfirmedDirection({ profile: active, directionId: "direction.1", now: t1 }))
      .toMatchObject({ directions: [expect.objectContaining({ status: "disabled", updatedAt: t1.toISOString() })] });
    const disabled = disablePersonalLibraryConfirmedDirection({ profile: active, directionId: "direction.1", now: t1 });
    const noOp = disablePersonalLibraryConfirmedDirection({ profile: disabled, directionId: "direction.1", now: t2 });
    expect(noOp).toEqual(disabled);
    const enabled = enablePersonalLibraryConfirmedDirection({ profile: disabled, directionId: "direction.1", catalog: catalog(), now: t2 });
    expect(enabled.directions[0]).toMatchObject({ status: "active", updatedAt: t2.toISOString() });
    const staleCatalog = catalog([paper("2608.00001", { abstract: "changed" })]);
    expect(() => enablePersonalLibraryConfirmedDirection({ profile: disabled, directionId: "direction.1", catalog: staleCatalog, now: t2 }))
      .toThrow(expect.objectContaining({ code: "evidence-mismatch" }));
  });

  it("merges terminal directions across proposals without lineage loss and retains direct sources", () => {
    const original = frozen(profile([
      direction("direction.1", "proposal.1", "candidate.1"),
      {
        ...direction("direction.2", "proposal.2", "candidate.2", "disabled"),
        createdAt: "2026-08-03T09:00:00.000Z",
        timeline: [{ kind: "created", at: "2026-08-03T09:00:00.000Z" }],
      },
    ]));
    const merged = mergePersonalLibraryConfirmedDirections({
      profile: original, sourceDirectionIds: ["direction.1", "direction.2"], directionId: "direction.3",
      status: "disabled", draft: draft(["arxiv:2608.00003"], "Merged"), catalog: catalog(), now: t1,
    });
    expect(merged.directions.slice(0, 2)).toEqual([
      expect.objectContaining({ id: "direction.1", status: "merged", mergedIntoDirectionId: "direction.3", createdAt: t0 }),
      expect.objectContaining({ id: "direction.2", status: "merged", mergedIntoDirectionId: "direction.3", createdAt: "2026-08-03T09:00:00.000Z" }),
    ]);
    expect(merged.directions[2]).toMatchObject({
      id: "direction.3", status: "disabled", createdAt: t1.toISOString(),
      lineage: {
        proposalIds: ["proposal.1", "proposal.2"], candidateIds: ["candidate.1", "candidate.2"],
        directionIds: ["direction.1", "direction.2"],
      },
    });
    expect(decodePersonalLibraryInterestProfile(merged)).toEqual(merged);
    expect(() => mergePersonalLibraryConfirmedDirections({
      profile: merged, sourceDirectionIds: ["direction.1", "direction.3"], directionId: "direction.4",
      status: "active", draft: draft(), catalog: catalog(), now: t2,
    })).toThrow(expect.objectContaining({ code: "conflict" }));

    const newerSource = {
      ...direction("direction.4", "proposal.4", "candidate.4"),
      createdAt: "2026-08-03T13:00:00.000Z", updatedAt: "2026-08-03T13:00:00.000Z",
      timeline: [{ kind: "created", at: "2026-08-03T13:00:00.000Z" }],
    };
    const chainBase = { ...merged, updatedAt: t0, directions: [...merged.directions, newerSource].sort((a, b) => a.id.localeCompare(b.id)) };
    const chained = mergePersonalLibraryConfirmedDirections({
      profile: chainBase, sourceDirectionIds: ["direction.3", "direction.4"], directionId: "direction.5",
      status: "active", draft: draft(), catalog: catalog(), now: new Date("2020-01-01T00:00:00.000Z"),
    });
    const target = chained.directions.find(({ id }) => id === "direction.5")!;
    expect(target.createdAt).toBe("2020-01-01T00:00:00.000Z");
    expect(target.updatedAt).toBe("2026-08-03T13:00:00.000Z");
    expect(chained.directions.find(({ id }) => id === "direction.4")!.createdAt).toBe("2026-08-03T13:00:00.000Z");
  });

  it("removes standalone directions in restrict mode or complete undirected merge families by cascade", () => {
    const standalone = profile([direction("direction.1", "proposal.1", "candidate.1")]);
    expect(removePersonalLibraryConfirmedDirection({ profile: standalone, directionId: "direction.1", mode: "restrict" }).directions)
      .toEqual([]);
    const family = mergePersonalLibraryConfirmedDirections({
      profile: profile([
        direction("direction.1", "proposal.1", "candidate.1"),
        direction("direction.2", "proposal.2", "candidate.2"),
      ]), sourceDirectionIds: ["direction.1", "direction.2"], directionId: "direction.3",
      status: "active", draft: draft(), catalog: catalog(), now: t1,
    });
    expect(() => removePersonalLibraryConfirmedDirection({ profile: family, directionId: "direction.1", mode: "restrict" }))
      .toThrow(expect.objectContaining({ code: "merge-relationship" }));
    expect(removePersonalLibraryConfirmedDirection({ profile: family, directionId: "direction.2", mode: "cascade" }).directions)
      .toEqual([]);
  });

  it("preserves eligibility boundaries across disable, stale evidence, explicit repair, and enable", () => {
    const active = profile([direction("direction.1", "proposal.1", "candidate.1")]);
    expect(evaluatePersonalLibraryInterestEligibility(active, catalog()).eligibleDirections).toHaveLength(1);
    const disabled = disablePersonalLibraryConfirmedDirection({ profile: active, directionId: "direction.1", now: t1 });
    expect(evaluatePersonalLibraryInterestEligibility(disabled, catalog()).eligibleDirections).toEqual([]);
    const stale = catalog([paper("2608.00001", { abstract: "changed" })]);
    expect(() => enablePersonalLibraryConfirmedDirection({ profile: disabled, directionId: "direction.1", catalog: stale, now: t2 })).toThrow();
    const textOnly = updatePersonalLibraryConfirmedDirection({
      profile: disabled, directionId: "direction.1", patch: { description: "Text alone cannot refresh evidence." },
      now: t2,
    });
    expect(() => enablePersonalLibraryConfirmedDirection({ profile: textOnly, directionId: "direction.1", catalog: stale, now: t2 }))
      .toThrow(expect.objectContaining({ code: "evidence-mismatch" }));
    const repaired = updatePersonalLibraryConfirmedDirection({
      profile: textOnly, directionId: "direction.1", patch: { description: "Reviewed changed evidence." },
      representativePaperKeys: ["arxiv:2608.00001"], catalog: stale, now: t2,
    });
    const enabled = enablePersonalLibraryConfirmedDirection({ profile: repaired, directionId: "direction.1", catalog: stale, now: t2 });
    expect(evaluatePersonalLibraryInterestEligibility({ ...enabled, updatedAt: t2.toISOString() }, stale)
      .eligibleDirections).toHaveLength(1);
    expect(removePersonalLibraryConfirmedDirection({ profile: enabled, directionId: "direction.1", mode: "restrict" }).directions)
      .toEqual([]);
  });
});
