import { describe, expect, it } from "vitest";
import {
  READING_CANDIDATES_MAX_PENDING,
  READING_CANDIDATES_SCHEMA_VERSION,
  decideReadingCandidate,
  decodeReadingCandidateRecord,
  decodeReadingCandidatesDocument,
  emptyReadingCandidatesDocument,
  readingCandidateFromRowSnapshot,
  removeReadingCandidate,
  upsertReadingCandidate,
  type ReadingCandidateRecord,
  type ReadingCandidateRowSnapshot,
} from "../src/library/reading-candidates/reading-candidates";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const NOW = "2026-08-13T00:00:00.000Z";

function candidate(index: number, overrides: Partial<ReadingCandidateRecord> = {}): ReadingCandidateRecord {
  return {
    paperKey: `arxiv:2608.${String(index).padStart(5, "0")}`,
    arxivId: `2608.${String(index).padStart(5, "0")}`,
    title: `Candidate title ${index}`,
    authors: "A. Author, B. Author",
    topic: "astrophysics",
    source: {
      kind: "library",
      manualTopics: [],
      directions: [{ id: "direction-1", name: "Cosmology" }],
      reportPath: "arxiv-daily/daily/2026-08-12.md",
      reportDate: "2026-08-12",
    },
    relatedPriorWorks: [
      { paperKey: "arxiv:2305.00001", title: "Prior survey paper" },
    ],
    provisionalNovelty: {
      differenceType: "new-dataset",
      comparisonBasis: ["arxiv:2305.00001"],
      evidenceDepth: "metadata-and-abstract",
      explanation: "Adds a new dataset relative to the prior survey.",
    },
    savedAt: NOW,
    updatedAt: NOW,
    ...overrides,
  };
}

function document(entries: ReadingCandidateRecord[] = []): ReturnType<typeof emptyReadingCandidatesDocument> {
  const doc = emptyReadingCandidatesDocument(SCOPE, IDENTIFICATION, NOW);
  for (const entry of entries) doc.candidates[entry.paperKey] = entry;
  return doc;
}

describe("reading candidates document operations", () => {
  it("upserts a candidate keyed by paperKey without losing existing decisions", () => {
    const saved = upsertReadingCandidate(document(), candidate(1), NOW);
    expect(saved.changed).toBe(true);
    expect(saved.document.candidates["arxiv:2608.00001"]?.title).toBe("Candidate title 1");

    const decided = decideReadingCandidate(saved.document, "arxiv:2608.00001", "skim", NOW).document;
    const refreshed = upsertReadingCandidate(
      decided,
      candidate(1, { title: "Refreshed title 1" }),
      "2026-08-13T01:00:00.000Z",
    );
    expect(refreshed.changed).toBe(true);
    expect(refreshed.document.candidates["arxiv:2608.00001"]?.title).toBe("Refreshed title 1");
    expect(refreshed.document.candidates["arxiv:2608.00001"]?.decision?.kind).toBe("skim");
  });

  it("evicts the oldest undecided candidate when pending capacity is exceeded", () => {
    let doc = document();
    const base = Date.parse("2026-01-01T00:00:00.000Z");
    for (let index = 0; index < READING_CANDIDATES_MAX_PENDING + 1; index += 1) {
      const at = new Date(base + index * 1000).toISOString();
      const result = upsertReadingCandidate(
        doc,
        candidate(index + 1, { savedAt: at, updatedAt: at }),
        at,
      );
      doc = result.document;
    }
    const keys = Object.keys(doc.candidates);
    expect(keys.length).toBe(READING_CANDIDATES_MAX_PENDING);
    // Oldest saved (index 1) evicted; newest retained.
    expect(doc.candidates["arxiv:2608.00001"]).toBeUndefined();
    expect(doc.candidates[`arxiv:2608.${String(READING_CANDIDATES_MAX_PENDING + 1).padStart(5, "0")}`]).toBeDefined();
  });

  it("keeps decided candidates beyond the pending capacity", () => {
    let doc = document();
    const base = Date.parse("2026-01-01T00:00:00.000Z");
    const decidedAt = new Date(base).toISOString();
    doc = decideReadingCandidate(
      upsertReadingCandidate(doc, candidate(1, { savedAt: decidedAt, updatedAt: decidedAt }), decidedAt).document,
      "arxiv:2608.00001",
      "read-closely",
      decidedAt,
    ).document;
    for (let index = 0; index < READING_CANDIDATES_MAX_PENDING + 5; index += 1) {
      const at = new Date(base + (index + 1) * 1000).toISOString();
      doc = upsertReadingCandidate(doc, candidate(index + 1000, { savedAt: at, updatedAt: at }), at).document;
    }
    expect(doc.candidates["arxiv:2608.00001"]?.decision?.kind).toBe("read-closely");
  });

  it("decides and removes candidates, ignoring unknown paperKeys", () => {
    const doc = upsertReadingCandidate(document(), candidate(1), NOW).document;
    expect(decideReadingCandidate(doc, "arxiv:9999.00001", "dismiss", NOW).changed).toBe(false);
    const decided = decideReadingCandidate(doc, "arxiv:2608.00001", "dismiss", NOW, "  not relevant  ");
    expect(decided.changed).toBe(true);
    expect(decided.document.candidates["arxiv:2608.00001"]?.decision).toEqual({
      kind: "dismiss",
      at: NOW,
      note: "not relevant",
    });
    expect(removeReadingCandidate(decided.document, "arxiv:9999.00001").changed).toBe(false);
    expect(removeReadingCandidate(decided.document, "arxiv:2608.00001").changed).toBe(true);
    expect(Object.keys(removeReadingCandidate(decided.document, "arxiv:2608.00001").document.candidates)).toHaveLength(0);
  });

  it("builds a candidate from a dashboard row snapshot and refuses rows without provenance", () => {
    const snapshot: ReadingCandidateRowSnapshot = {
      paperKey: "arxiv:2608.00001",
      arxivId: "2608.00001",
      title: "A new survey",
      authors: "A. Author",
      topic: "astrophysics",
      occurrenceProvenance: {
        reportPath: "arxiv-daily/daily/2026-08-12.md",
        reportDate: "2026-08-12",
        source: "both",
        manualTopics: [{ tag: "cosmology" }],
        directions: [
          {
            id: "direction-1",
            name: "Cosmology",
            representatives: [
              { paperKey: "arxiv:2305.00001", title: "Prior survey" },
              { paperKey: "arxiv:2201.00001", title: "Older survey" },
            ],
          },
        ],
      },
      personalNovelty: {
        differenceType: "new-dataset",
        comparisonBasis: ["arxiv:2305.00001"],
        evidenceDepth: "metadata-and-abstract",
        explanation: "New dataset.",
      },
    };
    const record = readingCandidateFromRowSnapshot(snapshot, NOW);
    expect(record).toMatchObject({
      paperKey: "arxiv:2608.00001",
      source: { kind: "both", reportDate: "2026-08-12" },
      relatedPriorWorks: [
        { paperKey: "arxiv:2305.00001", title: "Prior survey" },
        { paperKey: "arxiv:2201.00001", title: "Older survey" },
      ],
    });
    expect(record?.provisionalNovelty?.differenceType).toBe("new-dataset");
    expect(readingCandidateFromRowSnapshot({ ...snapshot, occurrenceProvenance: undefined }, NOW)).toBeNull();
    expect(readingCandidateFromRowSnapshot({
      ...snapshot,
      title: "x".repeat(501),
    }, NOW)).toBeNull();
  });

  it("rejects invalid records and documents at decode time", () => {
    expect(decodeReadingCandidateRecord(candidate(1))).toBeDefined();
    expect(decodeReadingCandidateRecord({ ...candidate(1), paperKey: "nope" })).toBeNull();
    expect(decodeReadingCandidateRecord({ ...candidate(1), arxivId: "other" })).toBeNull();
    expect(decodeReadingCandidateRecord({ ...candidate(1), title: "x".repeat(501) })).toBeNull();
    expect(decodeReadingCandidateRecord({
      ...candidate(1),
      provisionalNovelty: { ...candidate(1).provisionalNovelty!, differenceType: "made-up" },
    })).toBeNull();
    const valid = document([candidate(1)]);
    expect(decodeReadingCandidatesDocument(valid)?.candidates["arxiv:2608.00001"]).toBeDefined();
    expect(decodeReadingCandidatesDocument({ ...valid, schemaVersion: 99 })).toBeNull();
    expect(decodeReadingCandidatesDocument({ ...valid, candidates: { bad: "x" } })).toBeNull();
    expect(decodeReadingCandidatesDocument(valid)).toMatchObject({ schemaVersion: READING_CANDIDATES_SCHEMA_VERSION });
  });
});
