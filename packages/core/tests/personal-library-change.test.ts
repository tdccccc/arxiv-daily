import { describe, expect, it } from "vitest";
import {
  LIBRARY_CHANGE_NUDGE_COUNT,
  listUnaccountedLibraryPapers,
  shouldNudgeLibraryChange,
} from "../src/library/personal-library-change";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  type FullTextKnowledgeBaseManifest,
  type FullTextPaperKnowledgeRecord,
} from "../src/library/fulltext/knowledge-base";
import {
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
  createPersonalLibraryCatalogInputManifestFingerprint,
  createPersonalLibraryGenerationContractFingerprint,
  type PersonalLibraryDirectionProposal,
} from "../src/library/personal-library-interest-profile";

const scope = `sha256:${"a".repeat(64)}`;
const identification = `sha256:${"b".repeat(64)}`;
const evidence = `sha256:${"e".repeat(64)}`;
const time = "2026-10-15T00:00:00.000Z";

function arxivKey(month: string, serial: number): string {
  return `arxiv:${month}.${String(serial).padStart(5, "0")}`;
}

function fileKey(index: number): string {
  return `file:sha256:${index.toString(16).padStart(64, "0")}`;
}

function record(paperKey: string, overrides: Partial<FullTextPaperKnowledgeRecord> = {}): FullTextPaperKnowledgeRecord {
  return {
    paperKey, status: "ready", modelId: "fake", dimension: 8, filePaths: [`papers/${paperKey.replace(/\W/g, "_")}.pdf`],
    observationFingerprints: [evidence], chunkCount: 3, updatedAt: time, ...overrides,
  };
}

function manifestOf(records: FullTextPaperKnowledgeRecord[]): FullTextKnowledgeBaseManifest {
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION, revision: 1,
    scopeFingerprint: scope, identificationFingerprint: identification,
    modelId: "fake", dimension: 8, updatedAt: time,
    papers: Object.fromEntries(records.map((entry) => [entry.paperKey, entry])),
  };
}

function proposalOf(
  inputKeys: string[],
  extra: Partial<PersonalLibraryDirectionProposal> = {},
): PersonalLibraryDirectionProposal {
  const catalogInputPapers = [...inputKeys].sort().map((paperKey) => ({ paperKey, evidenceFingerprint: evidence }));
  return {
    schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION, revision: 0, proposalId: "proposal.1",
    scopeFingerprint: scope, identificationFingerprint: identification,
    catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
      scopeFingerprint: scope, identificationFingerprint: identification, catalogInputPapers,
    }),
    catalogInputPapers,
    generationContractFingerprint: createPersonalLibraryGenerationContractFingerprint("change-test"),
    generatedAt: time, topics: [], ...extra,
  };
}

describe("library change nudge constants", () => {
  it("nudges from three unaccounted papers upward and never below", () => {
    expect(LIBRARY_CHANGE_NUDGE_COUNT).toBe(3);
    expect([0, 1, 2].map(shouldNudgeLibraryChange)).toEqual([false, false, false]);
    expect([3, 4, 100].map(shouldNudgeLibraryChange)).toEqual([true, true, true]);
  });
});

describe("listUnaccountedLibraryPapers", () => {
  const known = [arxivKey("2610", 1), arxivKey("2610", 2), fileKey(1)];

  it("returns null when there is no proposal", () => {
    expect(listUnaccountedLibraryPapers({ manifest: manifestOf(known.map((key) => record(key))), proposal: null })).toBeNull();
  });

  it("returns an empty list when every ready paper was seen by the proposal", () => {
    const manifest = manifestOf(known.map((key) => record(key)));
    expect(listUnaccountedLibraryPapers({ manifest, proposal: proposalOf(known) })).toEqual([]);
  });

  it("treats ungrouped and covered input papers of an old proposal (no field) as accounted", () => {
    // The old proposal knows these three only through catalogInputPapers: one is
    // covered, one is in a direction, one is ungrouped. None may look new.
    const manifest = manifestOf(known.map((key) => record(key)));
    const proposal = proposalOf(known, { coveredPaperKeys: [known[0]!] });
    expect(listUnaccountedLibraryPapers({ manifest, proposal })).toEqual([]);
  });

  it("lists new ready papers, newest first, and ignores failed and zero-chunk ones", () => {
    const added = [arxivKey("2611", 7), arxivKey("2610", 9), fileKey(2)];
    const manifest = manifestOf([
      ...known.map((key) => record(key)),
      ...added.map((key) => record(key)),
      record(arxivKey("2611", 8), { status: "failed", error: "boom", chunkCount: 0 }),
      record(arxivKey("2611", 9), { chunkCount: 0 }),
    ]);
    expect(listUnaccountedLibraryPapers({ manifest, proposal: proposalOf(known) })).toEqual([
      arxivKey("2611", 7), arxivKey("2610", 9), fileKey(2),
    ]);
  });

  it("does not nudge for two new papers and does for three", () => {
    const base = known.map((key) => record(key));
    const two = listUnaccountedLibraryPapers({
      manifest: manifestOf([...base, record(arxivKey("2611", 1)), record(arxivKey("2611", 2))]),
      proposal: proposalOf(known),
    })!;
    const three = listUnaccountedLibraryPapers({
      manifest: manifestOf([...base, record(arxivKey("2611", 1)), record(arxivKey("2611", 2)), record(arxivKey("2611", 3))]),
      proposal: proposalOf(known),
    })!;
    expect([two.length, shouldNudgeLibraryChange(two.length)]).toEqual([2, false]);
    expect([three.length, shouldNudgeLibraryChange(three.length)]).toEqual([3, true]);
  });

  it("uses the recorded accounted set instead of only the input papers", () => {
    const earlier = fileKey(5);
    const manifest = manifestOf([...known.map((key) => record(key)), record(earlier)]);
    const withoutField = proposalOf(known);
    expect(listUnaccountedLibraryPapers({ manifest, proposal: withoutField })).toEqual([earlier]);
    const withField = proposalOf(known, { accountedPaperKeys: [...known, earlier].sort() });
    expect(listUnaccountedLibraryPapers({ manifest, proposal: withField })).toEqual([]);
  });

  it("does not count a same-title duplicate that the first scan merged away", () => {
    const title = "A Sufficiently Long Identifying Paper Title For Merge";
    const kept = fileKey(3);
    const dropped = fileKey(4);
    const manifest = manifestOf([
      ...known.map((key) => record(key)),
      record(kept, { title }), record(dropped, { title: title.toUpperCase() }),
    ]);
    // The scan kept `kept`, so the proposal records `kept` as input (and `dropped` as accounted).
    const proposal = proposalOf([...known, kept]);
    expect(listUnaccountedLibraryPapers({ manifest, proposal })).toEqual([]);
  });

  it("does not let a later same-title file replace the paper the scan kept", () => {
    const title = "Another Sufficiently Long Identifying Title For Merge";
    // fileKey(9) sorts after fileKey(3); a scan keeping fileKey(9) must still cover a new
    // fileKey(3) copy so a re-download does not look like a new paper.
    const keptEarlier = fileKey(9);
    const newCopy = fileKey(3);
    const manifest = manifestOf([
      ...known.map((key) => record(key)), record(keptEarlier, { title }), record(newCopy, { title }),
    ]);
    expect(listUnaccountedLibraryPapers({ manifest, proposal: proposalOf([...known, keptEarlier]) })).toEqual([]);
  });

  it("does not merge short or untitled files and never merges arXiv papers", () => {
    const manifest = manifestOf([
      ...known.map((key) => record(key)),
      record(fileKey(6), { title: "Introduction" }), record(fileKey(7), { title: "Introduction" }),
      record(fileKey(8)), record(fileKey(10)),
    ]);
    expect(listUnaccountedLibraryPapers({ manifest, proposal: proposalOf(known) })).toEqual([
      fileKey(10), fileKey(6), fileKey(7), fileKey(8),
    ].sort((a, b) => a.localeCompare(b)));
  });

  it("selects only the newest 1000 papers, as the first scan did", () => {
    const old = Array.from({ length: 1000 }, (_, index) => arxivKey("2501", index + 1));
    const fresh = [arxivKey("2611", 1), arxivKey("2611", 2)];
    // The proposal saw the 1000 old papers. Two fresh papers push the two oldest out of
    // the newest-1000 window; only the fresh ones are new, the displaced ones are not.
    const manifest = manifestOf([...old, ...fresh].map((key) => record(key)));
    expect(listUnaccountedLibraryPapers({ manifest, proposal: proposalOf(old) })).toEqual(fresh);
  });

  it("does not list papers beyond the window even though no scan saw them", () => {
    const window = Array.from({ length: 1000 }, (_, index) => arxivKey("2610", index + 1));
    const beyond = [arxivKey("2401", 1), arxivKey("2401", 2), arxivKey("2401", 3)];
    const manifest = manifestOf([...window, ...beyond].map((key) => record(key)));
    expect(listUnaccountedLibraryPapers({ manifest, proposal: proposalOf(window) })).toEqual([]);
  });
  it("leaves out papers the scan cannot use when the catalog is given", () => {
    // A file with neither catalog metadata nor an indexed title makes a scan fail
    // with catalog-invalid, so counting it would nudge towards an update that errors.
    const usable = fileKey(20);
    const unusable = fileKey(21);
    const inCatalog = arxivKey("2611", 4);
    const missingFromCatalog = arxivKey("2611", 5);
    const manifest = manifestOf([
      ...known.map((key) => record(key)),
      record(usable, { title: "A usable local report" }), record(unusable),
      record(inCatalog), record(missingFromCatalog),
    ]);
    const catalog = { papers: { [inCatalog]: {} } as never };
    expect(listUnaccountedLibraryPapers({ manifest, proposal: proposalOf(known), catalog })).toEqual([inCatalog, usable]);
    // Without the catalog nothing can be judged, so the older behavior is unchanged.
    expect(listUnaccountedLibraryPapers({ manifest, proposal: proposalOf(known) })).toHaveLength(4);
  });
});
