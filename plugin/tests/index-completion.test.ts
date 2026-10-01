import { describe, expect, it } from "vitest";
import type { FullTextIndexRunSummary } from "@arxiv-daily/core";
import { describeFullTextIndexCompletion, type FullTextIndexLibraryContext } from "../src/library/index-completion";

function libraryContext(overrides: Partial<FullTextIndexLibraryContext> = {}): FullTextIndexLibraryContext {
  return {
    totalFiles: 0,
    readyPapers: 0,
    unresolvedFallbackFiles: 0,
    metadataFetchFailures: 0,
    scannedBeforeIndexing: false,
    ...overrides,
  };
}

function summary(overrides: Partial<FullTextIndexRunSummary> = {}): FullTextIndexRunSummary {
  return {
    indexed: 3,
    reused: 1,
    titlesRefreshed: 0,
    failed: 0,
    pruned: 0,
    outcomes: [],
    manifestRevision: 7,
    manifestUpdatedAt: "2026-09-01T00:00:00.000Z",
    searchablePapers: 4,
    ...overrides,
  };
}

const NETWORK_FAILURE_MESSAGE =
  "Couldn't download the embedding model (network problem): arXiv Daily could not "
  + "reach Hugging Face to download the local embedding model. Check your internet "
  + "connection and try indexing again; the underlying error is logged to the "
  + "developer console.";

describe("describeFullTextIndexCompletion", () => {
  it("reports the ordinary counts when nothing failed", () => {
    expect(describeFullTextIndexCompletion(summary())).toBe(
      "full-text index — 3 indexed, 1 reused, 0 failed, 0 pruned.",
    );
  });

  it("appends the optional suffix only on the ordinary path", () => {
    expect(
      describeFullTextIndexCompletion(summary(), { onCompletionSuffix: "Search from the Dashboard." }),
    ).toBe("full-text index — 3 indexed, 1 reused, 0 failed, 0 pruned. Search from the Dashboard.");
  });

  it("mentions refreshed titles when any were refreshed", () => {
    expect(describeFullTextIndexCompletion(summary({ titlesRefreshed: 2 }))).toBe(
      "full-text index — 3 indexed, 1 reused, 0 failed, 0 pruned, 2 titles refreshed.",
    );
  });

  it("reports an ordinary per-paper failure as a plain count, not the network message", () => {
    const withOrdinaryFailure = summary({
      failed: 1,
      outcomes: [{ paperKey: "file:abc", status: "failed", error: "could not parse PDF" }],
    });
    expect(describeFullTextIndexCompletion(withOrdinaryFailure)).toBe(
      "full-text index — 3 indexed, 1 reused, 1 failed, 0 pruned.",
    );
  });

  it("replaces the generic count with a clear network-problem message when the model could not download", () => {
    const networkFailure = summary({
      indexed: 0,
      reused: 0,
      failed: 4,
      outcomes: [
        { paperKey: "file:a", status: "failed", error: NETWORK_FAILURE_MESSAGE },
        { paperKey: "file:b", status: "failed", error: NETWORK_FAILURE_MESSAGE },
      ],
    });
    const text = describeFullTextIndexCompletion(networkFailure);
    expect(text).toContain("couldn't download the embedding model (network problem)");
    expect(text).toContain("developer console");
    expect(text).not.toContain("4 failed");
  });

  it("ignores the optional suffix on the network-failure path", () => {
    const networkFailure = summary({
      failed: 1,
      outcomes: [{ paperKey: "file:a", status: "failed", error: NETWORK_FAILURE_MESSAGE }],
    });
    const text = describeFullTextIndexCompletion(networkFailure, {
      onCompletionSuffix: "Search from the Dashboard.",
    });
    expect(text).not.toContain("Search from the Dashboard.");
  });

  it("does not promise search when all attempted PDFs failed", () => {
    const text = describeFullTextIndexCompletion(summary({
      indexed: 0, reused: 0, searchablePapers: 0, failed: 1,
      outcomes: [{ paperKey: "file:a", status: "failed", error: "could not parse PDF" }],
    }), {
      onCompletionSuffix: "Search from the Dashboard.",
      libraryContext: libraryContext({ totalFiles: 1, unresolvedFallbackFiles: 1, scannedBeforeIndexing: true }),
    });
    expect(text).toContain("Nothing is searchable yet");
    expect(text).toContain("could not be read or embedded");
    expect(text).not.toContain("Full-text search works");
    expect(text).not.toContain("Search from the Dashboard");
  });

  describe("zero indexable units", () => {
    it("says the folder has no PDFs when the catalog has no files at all", () => {
      const empty = summary({ indexed: 0, reused: 0, outcomes: [] });
      const text = describeFullTextIndexCompletion(empty, {
        libraryContext: libraryContext({ totalFiles: 0 }),
      });
      expect(text).toBe(
        "no PDFs found in the personal library folder. Add PDFs to the folder, then use Scan library before building the index again.",
      );
    });

    it("says files exist but none could be read when the catalog has files but no indexable units", () => {
      const empty = summary({ indexed: 0, reused: 0, outcomes: [] });
      const text = describeFullTextIndexCompletion(empty, {
        libraryContext: libraryContext({ totalFiles: 5 }),
      });
      expect(text).toContain("found files in the personal library folder");
      expect(text).toContain("none of them could be read");
    });

    it("does not special-case zero outcomes without a library context (backward compatible)", () => {
      const empty = summary({ indexed: 0, reused: 0, outcomes: [] });
      expect(describeFullTextIndexCompletion(empty)).toBe(
        "full-text index — 0 indexed, 0 reused, 0 failed, 0 pruned.",
      );
    });

    it("leaves an ordinary non-zero result alone even when files/papers happen to be zero", () => {
      // outcomes.length > 0 (some unit was attempted) must take the normal path,
      // regardless of what the catalog counts say.
      const nonEmpty = summary({
        indexed: 1,
        reused: 0,
        outcomes: [{ paperKey: "file:a", status: "indexed", chunkCount: 3 }],
      });
      const text = describeFullTextIndexCompletion(nonEmpty, {
        libraryContext: libraryContext({ totalFiles: 0, readyPapers: 0 }),
      });
      expect(text).toBe("full-text index — 1 indexed, 0 reused, 0 failed, 0 pruned.");
    });
  });

  describe("zero recognized arXiv papers", () => {
    it("explains that full-text search still works but directions need arXiv papers", () => {
      const fallbackOnly = summary({
        indexed: 3,
        reused: 0,
        outcomes: [{ paperKey: "file:sha256:a", status: "indexed", chunkCount: 2 }],
      });
      const text = describeFullTextIndexCompletion(fallbackOnly, {
        libraryContext: libraryContext({
          totalFiles: 3,
          readyPapers: 0,
          unresolvedFallbackFiles: 3,
          scannedBeforeIndexing: true,
        }),
      });
      expect(text).toContain("Full-text search works for these files");
      expect(text).toContain("research directions and daily-report steering need papers recognized as arXiv papers");
      expect(text).not.toContain("offline");
    });

    it("only mentions a possible offline scan when the scan recorded a metadata-fetch failure", () => {
      const fallbackOnly = summary({
        indexed: 3,
        reused: 0,
        outcomes: [{ paperKey: "file:sha256:a", status: "indexed", chunkCount: 2 }],
      });
      const text = describeFullTextIndexCompletion(fallbackOnly, {
        libraryContext: libraryContext({
          totalFiles: 3,
          readyPapers: 0,
          unresolvedFallbackFiles: 3,
          scannedBeforeIndexing: true,
          metadataFetchFailures: 2,
        }),
      });
      expect(text).toContain("Some arXiv lookups failed");
      expect(text).toContain("offline during the scan");
    });

    it("does not overclaim an offline scan when files are merely unresolved (not a known metadata-fetch failure)", () => {
      const fallbackOnly = summary({
        indexed: 3,
        reused: 0,
        outcomes: [{ paperKey: "file:sha256:a", status: "indexed", chunkCount: 2 }],
      });
      const text = describeFullTextIndexCompletion(fallbackOnly, {
        libraryContext: libraryContext({
          totalFiles: 3,
          readyPapers: 0,
          unresolvedFallbackFiles: 3,
          scannedBeforeIndexing: true,
          metadataFetchFailures: 0,
        }),
      });
      expect(text).not.toContain("offline");
    });

    it("says nothing extra when this run did not scan (an already-scanned library with no recognized papers)", () => {
      const fallbackOnly = summary({
        indexed: 3,
        reused: 0,
        outcomes: [{ paperKey: "file:sha256:a", status: "indexed", chunkCount: 2 }],
      });
      const text = describeFullTextIndexCompletion(fallbackOnly, {
        libraryContext: libraryContext({
          totalFiles: 3,
          readyPapers: 0,
          unresolvedFallbackFiles: 3,
          scannedBeforeIndexing: false,
        }),
      });
      expect(text).toBe("full-text index — 3 indexed, 0 reused, 0 failed, 0 pruned.");
    });

    it("says nothing extra once some recognized arXiv papers exist", () => {
      const mixed = summary({
        indexed: 4,
        reused: 0,
        outcomes: [{ paperKey: "arxiv:2601.00001", status: "indexed", chunkCount: 2 }],
      });
      const text = describeFullTextIndexCompletion(mixed, {
        libraryContext: libraryContext({
          totalFiles: 4,
          readyPapers: 1,
          unresolvedFallbackFiles: 3,
          scannedBeforeIndexing: true,
        }),
      });
      expect(text).toBe("full-text index — 4 indexed, 0 reused, 0 failed, 0 pruned.");
    });
  });
});
