import { describe, expect, it } from "vitest";
import type { FullTextIndexRunSummary } from "@arxiv-daily/core";
import { describeFullTextIndexCompletion } from "../src/library/index-completion";

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
});
