import { describe, expect, it, vi } from "vitest";
import { ArxivLibraryMetadataResolver } from "../src/library/arxiv-library-metadata-resolver";
import type { AtomPaperMeta } from "../src/pipeline/atom-parser";

const paper: AtomPaperMeta = {
  id: "2608.00001",
  title: "Paper",
  authors: "A. Author et al.",
  authorNames: ["A. Author", "B. Author"],
  abstract: "Abstract.",
  published: "2026-08-01T00:00:00Z",
  updated: "2026-08-02T00:00:00Z",
  primaryCategory: "cs.AI",
  categories: ["cs.AI", "cs.LG"],
};

describe("ArxivLibraryMetadataResolver", () => {
  it("maps Atom metadata to canonical catalog metadata and preserves all authors", async () => {
    const fetchMetadataByIds = vi.fn(async () => new Map([[paper.id, paper]]));
    const resolver = new ArxivLibraryMetadataResolver({ fetchMetadataByIds });
    const controller = new AbortController();

    const resolved = await resolver.resolve([paper.id], controller.signal);

    expect(fetchMetadataByIds).toHaveBeenCalledWith([paper.id], controller.signal);
    expect(resolved.get(paper.id)).toEqual({
      arxivId: paper.id,
      title: paper.title,
      authors: ["A. Author", "B. Author"],
      abstract: paper.abstract,
      published: "2026-08-01T00:00:00.000Z",
      updated: "2026-08-02T00:00:00.000Z",
      primaryCategory: paper.primaryCategory,
      categories: paper.categories,
    });
  });

  it("preserves partial omission without inventing records", async () => {
    const resolver = new ArxivLibraryMetadataResolver({
      fetchMetadataByIds: vi.fn(async () => new Map([[paper.id, paper]])),
    });

    const resolved = await resolver.resolve([paper.id, "2608.00002"]);

    expect([...resolved.keys()]).toEqual([paper.id]);
  });
});
