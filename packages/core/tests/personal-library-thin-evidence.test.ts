import { describe, expect, it } from "vitest";
import {
  isThinEvidenceDirectionCandidate,
  type PersonalLibraryClusterMember,
  type PersonalLibraryRepresentativeEvidence,
} from "../src/library/personal-library-interest-profile";

function representatives(...paperKeys: string[]): PersonalLibraryRepresentativeEvidence[] {
  return paperKeys.map((paperKey) => ({
    paperKey,
    evidenceFingerprint: `sha256:${"a".repeat(64)}`,
  }));
}

function members(...paperKeys: string[]): PersonalLibraryClusterMember[] {
  return paperKeys.map((paperKey) => ({ paperKey, confidence: 1 }));
}

const paperA = "arxiv:2609.00001";
const paperB = "arxiv:2609.00002";

describe("isThinEvidenceDirectionCandidate", () => {
  it("counts complete evidence coverage when only one representative is displayed", () => {
    const candidate = {
      representatives: representatives(paperA),
      clusterMembers: members(paperA, paperB),
    };
    expect(isThinEvidenceDirectionCandidate(candidate)).toBe(false);
  });

  it("does not count repeated members as independent evidence", () => {
    const candidate = {
      representatives: representatives(paperA),
      clusterMembers: members(paperA, paperA),
    };
    expect(isThinEvidenceDirectionCandidate(candidate)).toBe(true);
  });

  it("keeps multiple distinct members sufficient even when some are repeated", () => {
    const candidate = {
      representatives: representatives(paperA),
      clusterMembers: members(paperA, paperA, paperB),
    };
    expect(isThinEvidenceDirectionCandidate(candidate)).toBe(false);
  });

  it("uses a nonempty member list as authority instead of inflating it with representatives", () => {
    const candidate = {
      representatives: representatives(paperA, paperB),
      clusterMembers: members(paperA),
    };
    expect(isThinEvidenceDirectionCandidate(candidate)).toBe(true);
  });

  it.each([
    { label: "no papers", keys: [], thin: true },
    { label: "one paper", keys: [paperA], thin: true },
    { label: "repeated paper", keys: [paperA, paperA], thin: true },
    { label: "two papers", keys: [paperA, paperB], thin: false },
    { label: "two papers with repetition", keys: [paperA, paperA, paperB], thin: false },
  ])("falls back to unique representatives for legacy $label", ({ keys, thin }) => {
    expect(isThinEvidenceDirectionCandidate({ representatives: representatives(...keys) })).toBe(thin);
    const emptyMembership = { representatives: representatives(...keys), clusterMembers: [] };
    expect(isThinEvidenceDirectionCandidate(emptyMembership)).toBe(thin);
  });
});
