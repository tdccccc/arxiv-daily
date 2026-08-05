import { describe, expect, it } from "vitest";
import type {
  PersonalLibraryCatalog,
  PersonalLibraryPaperRecord,
} from "../src/library/personal-library-catalog";
import {
  PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS,
  PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS,
  createEmptyPersonalLibraryInterestProfile,
  createPersonalLibraryCatalogInputManifest,
  createPersonalLibraryCatalogInputManifestFingerprint,
  createPersonalLibraryGenerationContractFingerprint,
  createPersonalLibraryPaperEvidenceFingerprint,
  createPersonalLibraryRepresentativeSetFingerprint,
  decodeDurablePersonalLibraryInterestProfile,
  decodePersonalLibraryDirectionProposal,
  decodePersonalLibraryInterestProfile,
  decodePersistedPersonalLibraryInterestProfile,
  type PersonalLibraryClusterMember,
  type PersonalLibraryConfirmedDirection,
  type PersonalLibraryDirectionProposal,
  type PersonalLibraryDirectionTimelineEvent,
  type PersonalLibraryInterestProfile,
  type PersonalLibraryRepresentativeEvidence,
} from "../src/library/personal-library-interest-profile";
import {
  confirmPersonalLibraryDirectionCandidate,
  mergePersonalLibraryConfirmedDirections,
  mergePersonalLibraryDirectionCandidates,
  removePersonalLibraryConfirmedDirection,
  updatePersonalLibraryConfirmedDirection,
  updatePersonalLibraryDirectionCandidate,
} from "../src/library/personal-library-interest-profile-review";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const t0 = "2026-08-03T10:00:00.000Z";
const t1 = "2026-08-03T11:00:00.000Z";
const t2 = "2026-08-03T12:00:00.000Z";
const later = "2026-08-03T13:00:00.000Z";

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

function members(): PersonalLibraryClusterMember[] {
  return [
    { paperKey: "arxiv:2608.00001", confidence: 0.92 },
    { paperKey: "arxiv:2608.00002", confidence: 0.71 },
  ];
}

function candidate(
  id: string,
  paperKey = "arxiv:2608.00001",
  clusterMembers: PersonalLibraryClusterMember[] | undefined = undefined,
) {
  const entry = catalog().papers[paperKey]!;
  const representatives = [{ paperKey, evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(entry) }];
  return {
    id, name: `Candidate ${id}`, description: `Description ${id}`, discoveryCues: [`cue ${id}`],
    representatives, representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    lineage: { candidateIds: [id] },
    ...(clusterMembers !== undefined ? { clusterMembers } : {}),
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
    generationContractFingerprint: createPersonalLibraryGenerationContractFingerprint("timeline-test"),
    generatedAt: t0, candidates: ids.map((id, index) => candidate(id, `arxiv:2608.0000${index + 1}`)),
  };
}

function draft(paperKeys = ["arxiv:2608.00001"], name = "Reviewed direction") {
  return {
    name, description: "Researcher reviewed description.",
    discoveryCues: ["reviewed cue", "second cue"], representativePaperKeys: paperKeys,
  };
}

function direction(
  id: string,
  status: "active" | "disabled" | "merged" = "active",
  options: {
    createdAt?: string;
    updatedAt?: string;
    timeline?: PersonalLibraryDirectionTimelineEvent[];
    clusterMembers?: PersonalLibraryClusterMember[];
    target?: string;
  } = {},
): PersonalLibraryConfirmedDirection {
  const createdAt = options.createdAt ?? t0;
  const representative: PersonalLibraryRepresentativeEvidence = {
    paperKey: "arxiv:2608.00001",
    evidenceFingerprint: createPersonalLibraryPaperEvidenceFingerprint(catalog().papers["arxiv:2608.00001"]!),
  };
  const common = {
    id, status, name: `Direction ${id}`, description: "Confirmed.", discoveryCues: ["confirmed cue"],
    representatives: [representative],
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint([representative]),
    clusterMembers: options.clusterMembers ?? [],
    timeline: options.timeline ?? [{ kind: "created" as const, at: createdAt }],
    lineage: { proposalIds: ["proposal.1"], candidateIds: [`candidate.${id}`], directionIds: [] },
    createdAt, updatedAt: options.updatedAt ?? createdAt,
  };
  return status === "merged"
    ? { ...common, status, mergedIntoDirectionId: options.target! }
    : { ...common, status };
}

function profile(directions: PersonalLibraryConfirmedDirection[] = []): PersonalLibraryInterestProfile {
  return { ...createEmptyPersonalLibraryInterestProfile(scope, identification, new Date(t0)), revision: 4, directions };
}

function v2Document(profileValue: PersonalLibraryInterestProfile): Record<string, any> {
  const document = JSON.parse(JSON.stringify(profileValue)) as Record<string, any>;
  document.schemaVersion = 2;
  document.directions = document.directions.map((raw: Record<string, any>) => {
    const { clusterMembers: _clusterMembers, timeline: _timeline, ...rest } = raw;
    return rest;
  });
  return document;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function createdTimeline(at: string): PersonalLibraryDirectionTimelineEvent[] {
  return [{ kind: "created", at }];
}

function editedTimeline(...ats: string[]): PersonalLibraryDirectionTimelineEvent[] {
  return ats.map((at) => ({ kind: "edited" as const, at }));
}

describe("schema v2 to v3 migration compatibility", () => {
  it("decodes exact v2 profiles with clusterMembers and created timeline defaults preserving semantics", () => {
    const v2 = v2Document(profile([
      direction("direction.1"),
      direction("direction.2", "disabled", { createdAt: "2026-08-03T09:00:00.000Z" }),
    ]));
    const expected = profile([
      direction("direction.1"),
      direction("direction.2", "disabled", { createdAt: "2026-08-03T09:00:00.000Z" }),
    ]);
    expect(decodePersistedPersonalLibraryInterestProfile(v2)).toEqual(expected);
    expect(decodeDurablePersonalLibraryInterestProfile(v2)).toEqual(expected);
    expect(decodePersonalLibraryInterestProfile(v2)).toBeNull();
    expect(decodePersonalLibraryInterestProfile(expected)).toEqual(expected);
  });

  it("retains v2 merge graphs, chronology, and rejection of malformed v2 envelopes", () => {
    const chain = profile([
      direction("direction.1", "merged", { target: "direction.2" }),
      direction("direction.2", "active"),
    ]);
    expect(decodePersistedPersonalLibraryInterestProfile(v2Document(chain))).toEqual(chain);
    const staleChronology = v2Document(profile([direction("direction.1")]));
    staleChronology.directions[0].updatedAt = later;
    expect(decodePersistedPersonalLibraryInterestProfile(staleChronology)).toBeNull();
    expect(decodeDurablePersonalLibraryInterestProfile(staleChronology)).toBeNull();
    const extraKey = { ...v2Document(profile([direction("direction.1")])), unexpected: true };
    expect(decodePersistedPersonalLibraryInterestProfile(extraKey)).toBeNull();
  });

  it("migrates legacy v1 profiles with v3 defaults through the durable chain", () => {
    const v1 = v2Document(profile([direction("direction.1")]));
    v1.schemaVersion = 1;
    v1.directions[0].lineage = {
      proposalId: "proposal.1", candidateIds: ["candidate.1"], directionIds: [],
    };
    const migrated = decodeDurablePersonalLibraryInterestProfile(v1);
    expect(migrated?.schemaVersion).toBe(3);
    expect(migrated?.directions[0]).toMatchObject({
      clusterMembers: [],
      timeline: [{ kind: "created", at: t0 }],
      createdAt: t0,
      lineage: { proposalIds: ["proposal.1"], candidateIds: ["candidate.1"], directionIds: [] },
    });
  });

  it("normalizes schema 2 proposals to schema 3 while preserving candidates", () => {
    const v2 = JSON.parse(JSON.stringify(proposal())) as Record<string, any>;
    v2.schemaVersion = 2;
    const decoded = decodePersonalLibraryDirectionProposal(v2);
    expect(decoded?.schemaVersion).toBe(3);
    expect(decoded?.candidates).toEqual(proposal().candidates);
    expect(Object.hasOwn(decoded!.candidates[0]!, "clusterMembers")).toBe(false);
    expect(decodePersonalLibraryDirectionProposal({ ...v2, schemaVersion: 1 })).toBeNull();
  });
});

describe("v3 strict direction decoding", () => {
  it("round-trips exact v3 profiles with clusterMembers and timelines", () => {
    const exact = profile([direction("direction.1", "active", {
      clusterMembers: [{ paperKey: "internal:doc-42", confidence: 0.5 }, ...members()],
      timeline: [...createdTimeline(t0), ...editedTimeline(t1, t2)],
    })]);
    expect(decodePersonalLibraryInterestProfile(exact)).toEqual(exact);
    expect(decodePersistedPersonalLibraryInterestProfile(exact)).toEqual(exact);
    const withMergedAndRemoved = profile([direction("direction.1", "active", {
      timeline: [
        { kind: "created", at: t0 },
        { kind: "merged", at: t1, sourceDirectionIds: ["direction.0", "direction.2"] },
        { kind: "members-updated", at: t1 },
        { kind: "removed", at: t2, mode: "restrict" },
      ],
    })]);
    expect(decodePersonalLibraryInterestProfile(withMergedAndRemoved)).toEqual(withMergedAndRemoved);
  });

  it("rejects invalid clusterMembers on confirmed directions", () => {
    const base = profile([direction("direction.1")]);
    const withMembers = (clusterMembers: unknown) => ({
      ...clone(base),
      directions: [{ ...clone(base.directions[0]), clusterMembers }],
    });
    expect(decodePersonalLibraryInterestProfile(withMembers([{ paperKey: "arxiv:2608.00001", confidence: 1.01 }]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withMembers([{ paperKey: "arxiv:2608.00001", confidence: -0.1 }]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withMembers([{ paperKey: "arxiv:2608.00001", confidence: Number.NaN }]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withMembers([{ paperKey: "arxiv:2608.00001", confidence: Number.POSITIVE_INFINITY }]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withMembers([
      { paperKey: "arxiv:2608.00001", confidence: 0.5 },
      { paperKey: "arxiv:2608.00001", confidence: 0.6 },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withMembers("not-an-array"))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withMembers([{ confidence: 0.5 }]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withMembers([{ paperKey: "", confidence: 0.5 }]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withMembers([{ paperKey: "arxiv:2608.00001", confidence: 0.5, extra: true }]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withMembers(
      Array.from({ length: PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS + 1 }, (_, index) => ({
        paperKey: `paper.${String(index).padStart(4, "0")}`, confidence: 0.5,
      })),
    ))).toBeNull();
    const missing = clone(base);
    delete missing.directions[0]!.clusterMembers;
    expect(decodePersonalLibraryInterestProfile(missing)).toBeNull();
  });

  it("rejects invalid timelines on confirmed directions", () => {
    const base = profile([direction("direction.1")]);
    const withTimeline = (timeline: unknown) => ({
      ...clone(base),
      directions: [{ ...clone(base.directions[0]), timeline }],
    });
    const missing = clone(base);
    delete missing.directions[0]!.timeline;
    expect(decodePersonalLibraryInterestProfile(missing)).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([{ kind: "edited", at: t1 }]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([{ kind: "created", at: t1 }]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "created", at: t1 },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "renamed", at: t1 },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "edited", at: "2026-08-03" },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "edited", at: t0 }, { kind: "edited", at: t1 },
    ]))).toEqual(withTimeline([
      { kind: "created", at: t0 }, { kind: "edited", at: t0 }, { kind: "edited", at: t1 },
    ]));
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "edited", at: t2 }, { kind: "edited", at: t1 },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "edited", at: t0, extra: true },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline(
      Array.from({ length: PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS + 1 }, (_, index) => index === 0
        ? { kind: "created", at: t0 }
        : { kind: "edited", at: new Date(Date.parse(t0) + index * 60_000).toISOString() }),
    ))).toBeNull();
  });

  it("validates merged and removed event payloads", () => {
    const base = profile([direction("direction.1")]);
    const withTimeline = (timeline: unknown) => ({
      ...clone(base),
      directions: [{ ...clone(base.directions[0]), timeline }],
    });
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "merged", at: t1 },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "merged", at: t1, sourceDirectionIds: [] },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "merged", at: t1, sourceDirectionIds: ["direction.2", "direction.1"] },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "merged", at: t1, sourceDirectionIds: ["direction.1", "direction.1"] },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "removed", at: t1, mode: "archive" },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "removed", at: t1, mode: "restrict", extra: true },
    ]))).toBeNull();
  });

  it("enforces candidate clusterMembers bounds and preserves absence for legacy candidates", () => {
    const withMembers = [...members(), { paperKey: "arxiv:2608.00003", confidence: 0.3 }];
    const rich = proposal();
    rich.candidates[0]!.clusterMembers = withMembers;
    const decoded = decodePersonalLibraryDirectionProposal(rich);
    expect(decoded?.candidates[0]?.clusterMembers).toEqual(withMembers);
    const plain = proposal();
    expect(Object.hasOwn(decoded!.candidates[0]!, "clusterMembers")).toBe(true);
    expect(Object.hasOwn(decodePersonalLibraryDirectionProposal(plain)!.candidates[0]!, "clusterMembers")).toBe(false);
    const duplicate = proposal();
    duplicate.candidates[0]!.clusterMembers = [members()[0]!, members()[0]!];
    expect(decodePersonalLibraryDirectionProposal(duplicate)).toBeNull();
    const excessive = proposal();
    excessive.candidates[0]!.clusterMembers = Array.from(
      { length: PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS + 1 },
      (_, index) => ({ paperKey: `paper.${String(index).padStart(4, "0")}`, confidence: 0.5 }),
    );
    expect(decodePersonalLibraryDirectionProposal(excessive)).toBeNull();
  });
});

describe("review workflow timeline writes", () => {
  it("confirms candidate clusterMembers and the created event onto the direction", () => {
    const rich = proposal();
    rich.candidates[0]!.clusterMembers = members();
    const result = confirmPersonalLibraryDirectionCandidate({
      proposal: rich, profile: profile(), catalog: catalog(), candidateId: "candidate.1",
      directionId: "direction.1", status: "active", draft: draft(), now: new Date(t1),
    });
    expect(result.profile.directions[0]!.clusterMembers).toEqual(members());
    expect(result.profile.directions[0]!.timeline).toEqual([{ kind: "created", at: t1 }]);
    expect(decodePersonalLibraryInterestProfile(result.profile)).toEqual(result.profile);

    const plain = confirmPersonalLibraryDirectionCandidate({
      proposal: proposal(), profile: profile(), catalog: catalog(), candidateId: "candidate.1",
      directionId: "direction.2", status: "active", draft: draft(), now: new Date(t1),
    });
    expect(plain.profile.directions[0]!.clusterMembers).toEqual([]);
  });

  it("keeps clusterMembers across candidate text edits and merges candidates without member loss", () => {
    const rich = proposal();
    rich.candidates[0]!.clusterMembers = members();
    rich.candidates[1]!.clusterMembers = [
      { paperKey: "arxiv:2608.00001", confidence: 0.6 },
      { paperKey: "arxiv:2608.00003", confidence: 0.8 },
    ];
    const edited = updatePersonalLibraryDirectionCandidate({
      proposal: rich, candidateId: "candidate.1", patch: { name: "Corrected" },
    });
    expect(edited.candidates[0]!.clusterMembers).toEqual(members());
    expect(edited.candidates[0]!.name).toBe("Corrected");
    const merged = mergePersonalLibraryDirectionCandidates({
      proposal: rich, sourceCandidateIds: ["candidate.1", "candidate.2"], candidateId: "candidate.3",
      draft: draft(["arxiv:2608.00002"]), catalog: catalog(),
    });
    expect(merged.candidates[0]!.clusterMembers).toEqual([
      { paperKey: "arxiv:2608.00001", confidence: 0.92 },
      { paperKey: "arxiv:2608.00002", confidence: 0.71 },
      { paperKey: "arxiv:2608.00003", confidence: 0.8 },
    ]);
    expect(decodePersonalLibraryDirectionProposal(merged)).toEqual(merged);
  });

  it("appends edited events with monotonic at and skips no-op edits", () => {
    const base = profile([direction("direction.1")]);
    const edited = updatePersonalLibraryConfirmedDirection({
      profile: base, directionId: "direction.1", patch: { name: "Edited" }, now: new Date(t1),
    });
    expect(edited.directions[0]!.timeline).toEqual([
      { kind: "created", at: t0 }, { kind: "edited", at: t1 },
    ]);
    expect(edited.directions[0]!.updatedAt).toBe(t1);
    const backward = updatePersonalLibraryConfirmedDirection({
      profile: base, directionId: "direction.1", patch: { name: "Backward clock" },
      now: new Date("2020-01-01T00:00:00.000Z"),
    });
    expect(backward.directions[0]!.timeline).toEqual([
      { kind: "created", at: t0 }, { kind: "edited", at: t0 },
    ]);
    expect(backward.directions[0]!.updatedAt).toBe(t0);
    const noOp = updatePersonalLibraryConfirmedDirection({
      profile: base, directionId: "direction.1", patch: { name: "Direction direction.1" },
      now: new Date(t2),
    });
    expect(noOp.directions[0]!.timeline).toEqual(createdTimeline(t0));
    expect(noOp.directions[0]!.updatedAt).toBe(t0);
  });

  it("writes the merged event with source ids on the target and leaves source timelines untouched", () => {
    const sources = profile([
      direction("direction.1"),
      direction("direction.2", "disabled", { createdAt: "2026-08-03T09:00:00.000Z" }),
    ]);
    const merged = mergePersonalLibraryConfirmedDirections({
      profile: sources, sourceDirectionIds: ["direction.1", "direction.2"], directionId: "direction.3",
      status: "active", draft: draft(["arxiv:2608.00003"], "Merged"), catalog: catalog(), now: new Date(t1),
    });
    const target = merged.directions.find(({ id }) => id === "direction.3")!;
    expect(target.timeline).toEqual([
      { kind: "created", at: t1 },
      { kind: "merged", at: t1, sourceDirectionIds: ["direction.1", "direction.2"] },
    ]);
    expect(target.clusterMembers).toEqual([]);
    expect(target.updatedAt).toBe(t1);
    for (const sourceId of ["direction.1", "direction.2"]) {
      const source = merged.directions.find(({ id }) => id === sourceId)!;
      expect(source.timeline).toEqual(createdTimeline(source.createdAt));
      expect(source.status).toBe("merged");
    }
    expect(decodePersonalLibraryInterestProfile(merged)).toEqual(merged);

    const futureSources = profile([
      direction("direction.4", "active", {
        createdAt: later, updatedAt: later, timeline: createdTimeline(later),
      }),
      direction("direction.6", "active", {
        createdAt: t1, updatedAt: t1, timeline: createdTimeline(t1),
      }),
    ]);
    const chained = mergePersonalLibraryConfirmedDirections({
      profile: futureSources, sourceDirectionIds: ["direction.4", "direction.6"], directionId: "direction.5",
      status: "active", draft: draft(), catalog: catalog(), now: new Date(t1),
    });
    const targetTwo = chained.directions.find(({ id }) => id === "direction.5")!;
    expect(targetTwo.timeline).toEqual([
      { kind: "created", at: t1 },
      { kind: "merged", at: later, sourceDirectionIds: ["direction.4", "direction.6"] },
    ]);
    expect(decodePersonalLibraryInterestProfile(chained)).toEqual(chained);
  });

  it("removes directions after closing their timelines and accepts an optional now", () => {
    const standalone = profile([direction("direction.1")]);
    const removed = removePersonalLibraryConfirmedDirection({
      profile: standalone, directionId: "direction.1", mode: "restrict", now: new Date(t2),
    });
    expect(removed.directions).toEqual([]);
    expect(decodePersonalLibraryInterestProfile(removed)).toEqual(removed);
    expect(removePersonalLibraryConfirmedDirection({
      profile: standalone, directionId: "direction.1", mode: "restrict",
    }).directions).toEqual([]);
    const family = mergePersonalLibraryConfirmedDirections({
      profile: profile([
        direction("direction.1"), direction("direction.2"),
      ]), sourceDirectionIds: ["direction.1", "direction.2"], directionId: "direction.3",
      status: "active", draft: draft(), catalog: catalog(), now: new Date(t1),
    });
    expect(removePersonalLibraryConfirmedDirection({
      profile: family, directionId: "direction.1", mode: "cascade", now: new Date(t2),
    }).directions).toEqual([]);
  });

  it("caps timelines at the event limit, dropping the oldest non-created events", () => {
    const at = (offsetMinutes: number) => new Date(Date.parse(t0) + offsetMinutes * 60_000).toISOString();
    const fullTimeline: PersonalLibraryDirectionTimelineEvent[] = [
      { kind: "created", at: t0 },
      ...Array.from({ length: PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS - 1 }, (_, index) => ({
        kind: "edited" as const, at: at(index + 1),
      })),
    ];
    const full = profile([direction("direction.1", "active", {
      updatedAt: at(PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS - 1),
      timeline: fullTimeline,
    })]);
    expect(decodePersonalLibraryInterestProfile(full)).toEqual(full);
    const overflow = updatePersonalLibraryConfirmedDirection({
      profile: full, directionId: "direction.1", patch: { name: "Overflow" },
      now: new Date(at(PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS)),
    });
    const timeline = overflow.directions[0]!.timeline;
    expect(timeline).toHaveLength(PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS);
    expect(timeline[0]).toEqual({ kind: "created", at: t0 });
    expect(timeline[1]).toEqual({ kind: "edited", at: at(2) });
    expect(timeline[timeline.length - 1]).toEqual({
      kind: "edited", at: at(PERSONAL_LIBRARY_MAX_TIMELINE_EVENTS),
    });
    expect(decodePersonalLibraryInterestProfile(overflow)).toEqual(overflow);
    const removed = removePersonalLibraryConfirmedDirection({
      profile: overflow, directionId: "direction.1", mode: "restrict", now: new Date(at(65)),
    });
    expect(removed.directions).toEqual([]);
  });
});
