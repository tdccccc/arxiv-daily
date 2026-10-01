import { describe, expect, it } from "vitest";
import type { PersonalLibraryCatalog, PersonalLibraryPaperRecord } from "../src/library/personal-library-catalog";
import {
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
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
} from "../src/library/personal-library-interest-profile";
import {
  confirmPersonalLibraryDirectionCandidate,
  confirmPersonalLibraryDirectionCandidates,
  disablePersonalLibraryConfirmedDirection,
  enablePersonalLibraryConfirmedDirection,
  mergePersonalLibraryConfirmedDirections,
  mergePersonalLibraryDirectionCandidates,
  movePersonalLibraryDirectionCandidate,
  removePersonalLibraryConfirmedDirection,
  removePersonalLibraryDirectionCandidate,
  updatePersonalLibraryConfirmedDirection,
  updatePersonalLibraryDirectionCandidate,
} from "../src/library/personal-library-proposal-review";

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
    id, text: `Candidate direction ${id}`, discoveryCues: [`cue ${id}`],
    representatives, representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    lineage: { candidateIds: [id] },
  };
}

function proposal(ids = ["candidate.1", "candidate.2"]): PersonalLibraryDirectionProposal {
  const catalogInputPapers = createPersonalLibraryCatalogInputManifest([
    catalog().papers["arxiv:2608.00001"]!,
  ]);
  return {
    schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION, revision: 7, proposalId: "proposal.1", scopeFingerprint: scope,
    identificationFingerprint: identification,
    catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
      scopeFingerprint: scope, identificationFingerprint: identification, catalogInputPapers,
    }),
    catalogInputPapers,
    generationContractFingerprint: createPersonalLibraryGenerationContractFingerprint("review-test"),
    generatedAt: t0,
    topics: [{
      id: "topic.1",
      suggestedName: "Reviewed topic",
      directions: ids.map((id, index) => candidate(id, `arxiv:2608.0000${index + 1}`)),
    }],
  };
}

function draft(paperKeys = ["arxiv:2608.00001"], text = "Reviewed direction") {
  return {
    text, discoveryCues: ["reviewed cue", "second cue"], representativePaperKeys: paperKeys,
  };
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
  function localProposal() {
    const original = proposal(["candidate.1"]);
    const entries = [1, 2].map((number) => ({
      paperKey: `file:sha256:${String(number).repeat(64)}`,
      evidenceFingerprint: `sha256:${String(number + 2).repeat(64)}`,
    }));
    original.catalogInputPapers = entries;
    original.catalogInputFingerprint = createPersonalLibraryCatalogInputManifestFingerprint({
      scopeFingerprint: scope, identificationFingerprint: identification, catalogInputPapers: entries,
    });
    original.topics[0]!.directions[0]!.representatives = [entries[0]!];
    original.topics[0]!.directions[0]!.representativeSetFingerprint = createPersonalLibraryRepresentativeSetFingerprint([entries[0]!]);
    return original;
  }

  it("saves local-file direction edits and resolves reviewed representatives from known proposal evidence", () => {
    const original = frozen(localProposal());
    expect(decodePersonalLibraryDirectionProposal(original)).not.toBeNull();
    const result = updatePersonalLibraryDirectionCandidate({
      proposal: original, candidateId: "candidate.1", patch: { text: "Reviewed local research" },
      representativePaperKeys: [original.catalogInputPapers[1]!.paperKey], catalog: catalog([]),
    });
    expect(result.topics[0]!.directions[0]!.text).toBe("Reviewed local research");
    expect(result.topics[0]!.directions[0]!.representatives).toEqual([original.catalogInputPapers[1]]);
    expect(decodePersonalLibraryDirectionProposal(result)).not.toBeNull();
  });

  it("rejects unknown local-file evidence and another library's catalog", () => {
    const original = localProposal();
    expect(() => updatePersonalLibraryDirectionCandidate({
      proposal: original, candidateId: "candidate.1", patch: { text: "Reviewed local research" },
      representativePaperKeys: [`file:sha256:${"9".repeat(64)}`], catalog: catalog([]),
    })).toThrow(expect.objectContaining({ code: "evidence-mismatch" }));
    expect(() => updatePersonalLibraryDirectionCandidate({
      proposal: original, candidateId: "candidate.1", patch: { text: "Reviewed local research" },
      representativePaperKeys: [original.catalogInputPapers[0]!.paperKey],
      catalog: { ...catalog([]), scopeFingerprint: `sha256:${"9".repeat(64)}` },
    })).toThrow(expect.objectContaining({ code: "incompatible-catalog" }));
  });

  it("saves mixed arXiv and local-file representatives without replacing either evidence identity", () => {
    const original = localProposal();
    const arxivPaper = catalog().papers["arxiv:2608.00001"]!;
    const localEvidence = original.catalogInputPapers[0]!;
    const result = updatePersonalLibraryDirectionCandidate({
      proposal: original, candidateId: "candidate.1", patch: { text: "Mixed-source research methods" },
      representativePaperKeys: [arxivPaper.paperKey, localEvidence.paperKey], catalog: catalog(),
    });
    expect(result.topics[0]!.directions[0]!.representatives).toEqual([
      { paperKey: arxivPaper.paperKey, evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(arxivPaper) },
      localEvidence,
    ]);
    expect(decodePersonalLibraryDirectionProposal(result)).not.toBeNull();
  });

  it("updates text locally, updates representatives from strict compatible evidence, and never confirms", () => {
    const original = frozen(proposal());
    const text = updatePersonalLibraryDirectionCandidate({
      proposal: original, candidateId: "candidate.1", patch: { text: "Corrected direction line" },
    });
    expect(text.topics[0]!.directions[0]).toMatchObject({ id: "candidate.1", text: "Corrected direction line" });
    expect((text.topics[0]!.directions[0] as any).status).toBeUndefined();
    expect(original.topics[0]!.directions[0]!.text).toBe("Candidate direction candidate.1");
    const reps = updatePersonalLibraryDirectionCandidate({
      proposal: text, candidateId: "candidate.1", patch: { discoveryCues: ["corrected cue"] },
      representativePaperKeys: ["arxiv:2608.00003"], catalog: catalog(),
    });
    expect(reps.topics[0]!.directions[0]!.representatives).toEqual([{
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
    expect(merged.topics[0]!.directions).toHaveLength(1);
    expect(merged.topics[0]!.directions[0]).toMatchObject({
      id: "candidate.3", text: "Reviewed direction",
      lineage: { candidateIds: ["candidate.1", "candidate.2", "candidate.3"] },
    });
    expect(() => mergePersonalLibraryDirectionCandidates({
      proposal: proposal(), sourceCandidateIds: ["candidate.1"], candidateId: "candidate.3",
      draft: draft(), catalog: catalog(),
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
  });

  it("removes any candidate including the final candidate and rejects repair-like inputs", () => {
    const result = removePersonalLibraryDirectionCandidate({ proposal: proposal(["candidate.1"]), candidateId: "candidate.1" });
    expect(result.topics).toEqual([]);
    expect(() => updatePersonalLibraryDirectionCandidate({
      proposal: proposal(), candidateId: "candidate.1", patch: { text: " padded " },
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
    expect(() => mergePersonalLibraryDirectionCandidates({
      proposal: proposal(), sourceCandidateIds: ["candidate.2", "candidate.1"], candidateId: "candidate.3",
      draft: draft(), catalog: catalog(),
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
  });
});

describe("moving one proposed direction", () => {
  it("moves only the chosen candidate and preserves its complete evidence", () => {
    const original = proposal();
    original.topics[0]!.targetTopicId = "settings.original";
    original.topics[0]!.directions[0]!.clusterMembers = [
      { paperKey: "arxiv:2608.00001", confidence: 0.95 },
      { paperKey: "arxiv:2608.00003", confidence: 0.8 },
    ];
    const before = structuredClone(original);

    const result = movePersonalLibraryDirectionCandidate({
      proposal: frozen(original), candidateId: "candidate.1", targetTopicId: "settings.destination",
      suggestedName: "Destination topic", topicId: "topic.0",
    });

    expect(result.topics).toEqual([
      {
        id: "topic.0", suggestedName: "Destination topic", targetTopicId: "settings.destination",
        directions: [before.topics[0]!.directions[0]],
      },
      { ...before.topics[0], directions: [before.topics[0]!.directions[1]] },
    ]);
    expect(result).toEqual({ ...before, topics: result.topics });
    expect(decodePersonalLibraryDirectionProposal(result)).toEqual(result);
    expect(original).toEqual(before);
  });

  it("removes the source topic when its last candidate moves", () => {
    const original = frozen(proposal(["candidate.1"]));
    const result = movePersonalLibraryDirectionCandidate({
      proposal: original, candidateId: "candidate.1", targetTopicId: "settings.destination",
      suggestedName: "Destination topic", topicId: "topic.2",
    });

    expect(result.topics).toEqual([{
      id: "topic.2", suggestedName: "Destination topic", targetTopicId: "settings.destination",
      directions: original.topics[0]!.directions,
    }]);
    expect(original.topics[0]!.id).toBe("topic.1");
  });

  it("removes the existing target when the researcher explicitly chooses a new topic", () => {
    const original = proposal();
    original.topics[0]!.targetTopicId = "settings.original";
    const result = movePersonalLibraryDirectionCandidate({
      proposal: frozen(original), candidateId: "candidate.1", targetTopicId: null,
      suggestedName: "New topic", topicId: "topic.2",
    });

    expect(result.topics[1]).toEqual({
      id: "topic.2", suggestedName: "New topic", directions: [original.topics[0]!.directions[0]],
    });
    expect(result.topics[1]).not.toHaveProperty("targetTopicId");
    expect(result.topics[0]!.targetTopicId).toBe("settings.original");
  });

  it("gives an explicit new topic a fresh identity even when the suggested name is unchanged", () => {
    const original = frozen(proposal());
    const result = movePersonalLibraryDirectionCandidate({
      proposal: original, candidateId: "candidate.1", targetTopicId: null,
      suggestedName: "Reviewed topic", topicId: "topic.2",
    });

    expect(result.topics.map(({ id, directions }) => ({ id, candidateIds: directions.map(({ id }) => id) })))
      .toEqual([
        { id: "topic.1", candidateIds: ["candidate.2"] },
        { id: "topic.2", candidateIds: ["candidate.1"] },
      ]);
  });

  it("leaves an unchanged existing target and name alone without allocating a topic", () => {
    const original = proposal();
    original.topics[0]!.targetTopicId = "settings.original";
    const result = movePersonalLibraryDirectionCandidate({
      proposal: frozen(original), candidateId: "candidate.1", targetTopicId: "settings.original",
      suggestedName: "Reviewed topic", topicId: "topic.1",
    });

    expect(result).toEqual(original);
  });

  it("rejects an occupied proposal topic identity before moving anything", () => {
    const original = frozen(proposal(["candidate.1"]));
    expect(() => movePersonalLibraryDirectionCandidate({
      proposal: original, candidateId: "candidate.1", targetTopicId: "settings.destination",
      suggestedName: "Destination topic", topicId: "topic.1",
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    expect(original.topics[0]!.directions.map(({ id }) => id)).toEqual(["candidate.1"]);
  });

  it("rejects a candidate that is no longer in the proposal", () => {
    expect(() => movePersonalLibraryDirectionCandidate({
      proposal: frozen(proposal()), candidateId: "candidate.missing", targetTopicId: null,
      suggestedName: "New topic", topicId: "topic.2",
    })).toThrow(expect.objectContaining({ code: "not-found" }));
  });

  it.each([null, 1, "", "   ", "x".repeat(121), "Two\nlines", "Two\rlines"])(
    "rejects an invalid destination name %j",
    (suggestedName) => {
      expect(() => movePersonalLibraryDirectionCandidate({
        proposal: frozen(proposal()), candidateId: "candidate.1", targetTopicId: null,
        suggestedName, topicId: "topic.2",
      })).toThrow(expect.objectContaining({ code: "invalid-input" }));
    },
  );

  it.each([
    { targetTopicId: undefined },
    { targetTopicId: "" },
    { targetTopicId: "invalid id" },
    { topicId: "" },
    { topicId: "invalid id" },
    { unexpected: true },
  ])("rejects invalid or extra move fields %j", (patch) => {
    expect(() => movePersonalLibraryDirectionCandidate({
      proposal: frozen(proposal()), candidateId: "candidate.1", targetTopicId: null,
      suggestedName: "New topic", topicId: "topic.2", ...patch,
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
  });

  it("allows splitting all twelve candidates without losing any or exceeding topic bounds", () => {
    const original = proposal(["candidate.00"]);
    original.topics[0]!.directions = Array.from({ length: 12 }, (_, index) =>
      candidate(`candidate.${String(index).padStart(2, "0")}`));
    let result = frozen(original);
    for (let index = 0; index < 11; index += 1) {
      result = movePersonalLibraryDirectionCandidate({
        proposal: result, candidateId: `candidate.${String(index).padStart(2, "0")}`,
        targetTopicId: null, suggestedName: `New topic ${index}`, topicId: `new-topic.${index}`,
      });
    }

    expect(result.topics).toHaveLength(12);
    expect(result.topics.every(({ directions }) => directions.length === 1)).toBe(true);
    expect(result.topics.flatMap(({ directions }) => directions).map(({ id }) => id).sort())
      .toEqual(original.topics[0]!.directions.map(({ id }) => id));
    expect(decodePersonalLibraryDirectionProposal(result)).toEqual(result);
    expect(original.topics).toHaveLength(1);
    expect(original.topics[0]!.directions).toHaveLength(12);
  });
});
