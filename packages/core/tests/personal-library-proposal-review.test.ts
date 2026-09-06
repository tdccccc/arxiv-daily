import { describe, expect, it } from "vitest";
import type { PersonalLibraryCatalog, PersonalLibraryPaperRecord } from "../src/library/personal-library-catalog";
import {
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
    schemaVersion: 5, revision: 7, proposalId: "proposal.1", scopeFingerprint: scope,
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
