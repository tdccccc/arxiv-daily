import { describe, expect, it } from "vitest";
import type {
  PersonalLibraryCatalog,
  PersonalLibraryPaperRecord,
} from "../src/library/personal-library-catalog";
import {
  createEmptyPersonalLibraryInterestProfile,
  createPersonalLibraryPaperEvidenceFingerprint,
  createPersonalLibraryRepresentativeSetFingerprint,
  decodePersonalLibraryInterestProfile,
  type PersonalLibraryClusterMember,
  type PersonalLibraryConfirmedDirection,
  type PersonalLibraryDirectionTimelineEvent,
  type PersonalLibraryInterestProfile,
  type PersonalLibraryRepresentativeEvidence,
} from "../src/library/personal-library-interest-profile";
import {
  lockPersonalLibraryConfirmedDirection,
  unlockPersonalLibraryConfirmedDirection,
  updatePersonalLibraryConfirmedDirection,
} from "../src/library/personal-library-interest-profile-review";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const t0 = "2026-08-03T10:00:00.000Z";
const t1 = "2026-08-03T11:00:00.000Z";
const t2 = "2026-08-03T12:00:00.000Z";

function paper(id: string): PersonalLibraryPaperRecord {
  return {
    paperKey: `arxiv:${id}`, source: "arxiv", externalId: id, title: `Paper ${id}`,
    authors: ["Researcher"], abstract: `Abstract ${id}`,
    published: "2026-08-01T00:00:00.000Z", updated: "2026-08-02T00:00:00.000Z",
    primaryCategory: "cs.AI", categories: ["cs.AI"], evidenceDepth: "metadata-and-abstract",
    filePaths: [`papers/${id}.pdf`],
  };
}

function catalog(): PersonalLibraryCatalog {
  const entry = paper("2608.00001");
  return {
    schemaVersion: 1, revision: 1, scopeFingerprint: scope, identificationFingerprint: identification,
    updatedAt: t0, lastScan: null,
    files: { [entry.filePaths[0]!]: {
      path: entry.filePaths[0]!, status: "ready" as const,
      observationFingerprint: `sha256:${"1".repeat(64)}`,
      paperKey: entry.paperKey, arxivId: entry.externalId, updatedAt: t0,
    } },
    papers: { [entry.paperKey]: entry },
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
    lockedAt?: string;
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
    ...(options.lockedAt !== undefined ? { lockedAt: options.lockedAt } : {}),
  };
  return status === "merged"
    ? { ...common, status, mergedIntoDirectionId: options.target! }
    : { ...common, status };
}

function profile(directions: PersonalLibraryConfirmedDirection[] = []): PersonalLibraryInterestProfile {
  return { ...createEmptyPersonalLibraryInterestProfile(scope, identification, new Date(t0)), revision: 4, directions };
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

describe("lockedAt decoding", () => {
  it("decodes v3 directions without lockedAt for backward compatibility", () => {
    const exact = profile([direction("direction.1")]);
    expect(decodePersonalLibraryInterestProfile(exact)).toEqual(exact);
    expect(Object.hasOwn(exact.directions[0]!, "lockedAt")).toBe(false);
    const disabled = profile([direction("direction.1", "disabled")]);
    expect(decodePersonalLibraryInterestProfile(disabled)).toEqual(disabled);
  });

  it("round-trips a canonical lockedAt and rejects invalid lockedAt values", () => {
    const locked = profile([direction("direction.1", "active", {
      updatedAt: t1,
      timeline: [{ kind: "created", at: t0 }, { kind: "locked", at: t1 }],
      lockedAt: t1,
    })]);
    expect(decodePersonalLibraryInterestProfile(locked)).toEqual(locked);

    const base = profile([direction("direction.1")]);
    const withLockedAt = (lockedAt: unknown) => ({
      ...clone(base),
      directions: [{ ...clone(base.directions[0]), lockedAt }],
    });
    expect(decodePersonalLibraryInterestProfile(withLockedAt(""))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withLockedAt("2026-08-03"))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withLockedAt("not-a-timestamp"))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withLockedAt(12345))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withLockedAt(null))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withLockedAt("2026-08-02T00:00:00.000Z"))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withLockedAt({ at: t1 }))).toBeNull();
    const extra = { ...clone(base), directions: [{ ...clone(base.directions[0]), lockedAt: t1, extra: true }] };
    expect(decodePersonalLibraryInterestProfile(extra)).toBeNull();
  });
});

describe("split timeline event decoding", () => {
  it("decodes valid split events carrying an opaque source direction id", () => {
    const base = profile([direction("direction.1")]);
    const withTimeline = (timeline: unknown) => ({
      ...clone(base),
      directions: [{ ...clone(base.directions[0]), timeline }],
    });
    const split = withTimeline([
      { kind: "created", at: t0 },
      { kind: "split", at: t1, sourceDirectionId: "direction.2" },
      { kind: "members-updated", at: t1 },
    ]);
    expect(decodePersonalLibraryInterestProfile(split)).toEqual(split);
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "split", at: t1, sourceDirectionId: "proposal.1.candidate.7" },
    ]))).not.toBeNull();
  });

  it("rejects split events with missing or malformed sourceDirectionId", () => {
    const base = profile([direction("direction.1")]);
    const withTimeline = (timeline: unknown) => ({
      ...clone(base),
      directions: [{ ...clone(base.directions[0]), timeline }],
    });
    const sourceDirectionId = "d".repeat(257);
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "split", at: t1 },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "split", at: t1, sourceDirectionId: "" },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "split", at: t1, sourceDirectionId: [] },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "split", at: t1, sourceDirectionId: 42 },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "split", at: t1, sourceDirectionId },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "split", at: t1, sourceDirectionId: "direction 2" },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "split", at: t1, sourceDirectionId: "direction.2", extra: true },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "split", at: t0, sourceDirectionId: "direction.2" },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "split", at: t0, sourceDirectionId: "direction.2" },
      { kind: "split", at: t1, sourceDirectionId: "direction.3" },
      { kind: "split", at: t0, sourceDirectionId: "direction.4" },
    ]))).toBeNull();
  });

  it("decodes locked and unlocked events under the monotonic at rule", () => {
    const base = profile([direction("direction.1")]);
    const withTimeline = (timeline: unknown) => ({
      ...clone(base),
      directions: [{ ...clone(base.directions[0]), timeline }],
    });
    const cycle = withTimeline([
      { kind: "created", at: t0 },
      { kind: "locked", at: t1 },
      { kind: "edited", at: t1 },
      { kind: "unlocked", at: t2 },
    ]);
    expect(decodePersonalLibraryInterestProfile(cycle)).toEqual(cycle);
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "locked", at: t1, extra: true },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "unlocked", at: t2 }, { kind: "locked", at: t1 },
    ]))).toBeNull();
    expect(decodePersonalLibraryInterestProfile(withTimeline([
      { kind: "created", at: t0 }, { kind: "unlocked", at: "2026-08-03" },
    ]))).toBeNull();
  });
});

describe("lock and unlock mutations", () => {
  it("locks a direction, writing lockedAt and a locked event with a synced monotonic updatedAt", () => {
    const base = profile([direction("direction.1")]);
    const locked = lockPersonalLibraryConfirmedDirection({
      profile: base, directionId: "direction.1", now: new Date(t1),
    });
    expect(locked.directions[0]!.lockedAt).toBe(t1);
    expect(locked.directions[0]!.updatedAt).toBe(t1);
    expect(locked.directions[0]!.timeline).toEqual([
      { kind: "created", at: t0 }, { kind: "locked", at: t1 },
    ]);
    expect(decodePersonalLibraryInterestProfile(locked)).toEqual(locked);
  });

  it("keeps lockedAt monotonic against backward clocks and defaults now to updatedAt", () => {
    const base = profile([direction("direction.1")]);
    const backward = lockPersonalLibraryConfirmedDirection({
      profile: base, directionId: "direction.1", now: new Date("2020-01-01T00:00:00.000Z"),
    });
    expect(backward.directions[0]!.lockedAt).toBe(t0);
    expect(backward.directions[0]!.updatedAt).toBe(t0);
    expect(backward.directions[0]!.timeline).toEqual([
      { kind: "created", at: t0 }, { kind: "locked", at: t0 },
    ]);
    const noNow = lockPersonalLibraryConfirmedDirection({
      profile: base, directionId: "direction.1",
    });
    expect(noNow.directions[0]!.lockedAt).toBe(t0);
    expect(noNow.directions[0]!.timeline).toEqual([
      { kind: "created", at: t0 }, { kind: "locked", at: t0 },
    ]);
  });

  it("unlocks a direction, clearing lockedAt and appending an unlocked event", () => {
    const locked = profile([direction("direction.1", "active", {
      updatedAt: t1,
      timeline: [{ kind: "created", at: t0 }, { kind: "locked", at: t1 }],
      lockedAt: t1,
    })]);
    const unlocked = unlockPersonalLibraryConfirmedDirection({
      profile: locked, directionId: "direction.1", now: new Date(t2),
    });
    expect(Object.hasOwn(unlocked.directions[0]!, "lockedAt")).toBe(false);
    expect(unlocked.directions[0]!.updatedAt).toBe(t2);
    expect(unlocked.directions[0]!.timeline).toEqual([
      { kind: "created", at: t0 },
      { kind: "locked", at: t1 },
      { kind: "unlocked", at: t2 },
    ]);
    expect(decodePersonalLibraryInterestProfile(unlocked)).toEqual(unlocked);
  });

  it("supports a lock/unlock round trip and keeps lockedAt across unrelated edits", () => {
    const base = profile([direction("direction.1")]);
    const locked = lockPersonalLibraryConfirmedDirection({
      profile: base, directionId: "direction.1", now: new Date(t1),
    });
    const edited = updatePersonalLibraryConfirmedDirection({
      profile: locked, directionId: "direction.1", patch: { name: "Edited while locked" },
      now: new Date(t2),
    });
    expect(edited.directions[0]!.lockedAt).toBe(t1);
    expect(edited.directions[0]!.updatedAt).toBe(t2);
    const cycle = unlockPersonalLibraryConfirmedDirection({
      profile: edited, directionId: "direction.1", now: new Date(t2),
    });
    expect(Object.hasOwn(cycle.directions[0]!, "lockedAt")).toBe(false);
    expect(cycle.directions[0]!.timeline).toEqual([
      { kind: "created", at: t0 },
      { kind: "locked", at: t1 },
      { kind: "edited", at: t2 },
      { kind: "unlocked", at: t2 },
    ]);
    expect(decodePersonalLibraryInterestProfile(cycle)).toEqual(cycle);
    const relocked = lockPersonalLibraryConfirmedDirection({
      profile: cycle, directionId: "direction.1", now: new Date(t2),
    });
    expect(relocked.directions[0]!.lockedAt).toBe(t2);
    expect(relocked.directions[0]!.timeline.at(-1)).toEqual({ kind: "locked", at: t2 });
  });

  it("rejects duplicate locks, unlocks of unlocked directions, missing directions, and merged directions", () => {
    const locked = profile([direction("direction.1", "active", {
      updatedAt: t1,
      timeline: [{ kind: "created", at: t0 }, { kind: "locked", at: t1 }],
      lockedAt: t1,
    })]);
    expect(() => lockPersonalLibraryConfirmedDirection({
      profile: locked, directionId: "direction.1", now: new Date(t2),
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    expect(() => unlockPersonalLibraryConfirmedDirection({
      profile: profile([direction("direction.1")]), directionId: "direction.1", now: new Date(t2),
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    expect(() => lockPersonalLibraryConfirmedDirection({
      profile: profile(), directionId: "missing.1", now: new Date(t1),
    })).toThrow(expect.objectContaining({ code: "not-found" }));
    expect(() => unlockPersonalLibraryConfirmedDirection({
      profile: profile(), directionId: "missing.1", now: new Date(t1),
    })).toThrow(expect.objectContaining({ code: "not-found" }));
    const merged = profile([
      direction("direction.1", "merged", { target: "direction.2" }),
      direction("direction.2"),
    ]);
    expect(() => lockPersonalLibraryConfirmedDirection({
      profile: merged, directionId: "direction.1", now: new Date(t1),
    })).toThrow(expect.objectContaining({ code: "conflict" }));
    expect(() => unlockPersonalLibraryConfirmedDirection({
      profile: merged, directionId: "direction.1", now: new Date(t1),
    })).toThrow(expect.objectContaining({ code: "conflict" }));
  });

  it("rejects non-exact inputs and never mutates the input profile", () => {
    const base = profile([direction("direction.1")]);
    expect(() => lockPersonalLibraryConfirmedDirection({
      profile: base, directionId: "direction.1", unexpected: true,
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
    expect(() => unlockPersonalLibraryConfirmedDirection({
      profile: base, directionId: "direction.1", now: "2026-08-03T11:00:00.000Z",
    })).toThrow(expect.objectContaining({ code: "invalid-input" }));
    expect(base.directions[0]!.lockedAt).toBeUndefined();
    expect(base.directions[0]!.timeline).toEqual([{ kind: "created", at: t0 }]);
  });
});
