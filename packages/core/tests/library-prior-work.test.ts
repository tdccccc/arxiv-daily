import { describe, expect, it, vi } from "vitest";
import {
  createLibraryPriorWork as createLibraryPriorWorkFromIndex,
  LibraryPriorWorkUnavailableError,
} from "../src";
import { createLibraryPriorWork, type LibraryPriorWorkSearchRequest } from "../src/library/prior-work";
import type { PersonalLibraryPaperRecord } from "../src/library/personal-library-catalog";
import { MAX_BM25_QUERY_TERMS } from "../src/library/fulltext/generation-bm25-index";
import { tokenizeUnicode } from "../src/library/fulltext/bm25-retrieval";
import type { FullTextPaperKnowledgeRecord } from "../src/library/fulltext/knowledge-base";
import { RunCancelledError } from "../src/services/cancellation";

const key = (index: number): string => `arxiv:2501.${String(index).padStart(5, "0")}`;
const record = (index: number, overrides: Partial<PersonalLibraryPaperRecord> = {}): PersonalLibraryPaperRecord => ({
  paperKey: key(index),
  source: "arxiv",
  externalId: `2501.${String(index).padStart(5, "0")}`,
  title: `Library paper ${index}`,
  authors: ["L. Reader"],
  abstract: `Library abstract ${index}`,
  published: "2025-01-02",
  updated: "2025-01-03",
  primaryCategory: "astro-ph.GA",
  categories: ["astro-ph.GA"],
  evidenceDepth: "metadata-and-abstract",
  filePaths: [`library/${index}.pdf`],
  ...overrides,
});
const query = { paperKey: "arxiv:2609.00001", title: "A new galaxy survey", abstract: "We map galaxies." };

function setup(
  papers: PersonalLibraryPaperRecord[],
  matches: string[],
  overrides: { assertAuthorized?: () => void | Promise<void> } = {},
) {
  const catalog = { load: vi.fn(async () => ({ papers: Object.fromEntries(papers.map((paper) => [paper.paperKey, paper])) })) };
  const search = vi.fn(async (_request: LibraryPriorWorkSearchRequest) =>
    matches.map((paperKey) => ({ paperKey })));
  const assertAuthorized = vi.fn(overrides.assertAuthorized ?? (async () => {}));
  const port = createLibraryPriorWork({ catalog, search, assertAuthorized, mode: "lexical" });
  return { port, catalog, search, assertAuthorized };
}

describe("createLibraryPriorWork", () => {
  it("is exported from the package index", () => {
    expect(createLibraryPriorWorkFromIndex).toBe(createLibraryPriorWork);
  });

  it("joins matches with the catalog in search order and returns the metadata the novelty contract needs", async () => {
    const { port } = setup([record(1), record(2)], [key(2), key(1)]);
    await expect(port.findSimilar(query, 5)).resolves.toEqual([
      {
        paperKey: key(2), title: "Library paper 2", authors: ["L. Reader"],
        abstract: "Library abstract 2", published: "2025-01-02", categories: ["astro-ph.GA"],
      },
      expect.objectContaining({ paperKey: key(1) }),
    ]);
  });

  it("searches with the title and abstract, the configured mode and the signal, over-fetching candidates", async () => {
    const { port, search } = setup([record(1)], [key(1)]);
    const controller = new AbortController();
    await port.findSimilar(query, 5, controller.signal);
    expect(search).toHaveBeenCalledTimes(1);
    const { queryText, limit, mode, signal } = search.mock.calls[0]![0];
    expect(queryText).toContain(query.title);
    expect(queryText).toContain(query.abstract);
    // Three candidates per wanted paper plus the query paper itself.
    expect([limit, mode, signal]).toEqual([16, "lexical", controller.signal]);
  });

  it("gives the search a lexical query within the index term limit, title terms first, and the full text for embedding", async () => {
    const { port, search } = setup([record(1)], [key(1)]);
    const longAbstract = Array.from({ length: 150 }, (_, index) => `word${index}`).join(" ");
    await port.findSimilar({ ...query, title: "Galaxy lensing", abstract: longAbstract }, 5);
    const request = search.mock.calls[0]![0];
    expect(request.queryText).toBe(`Galaxy lensing\n${longAbstract}`);
    const lexicalTerms = tokenizeUnicode(request.lexicalQueryText);
    expect(new Set(lexicalTerms).size).toBe(MAX_BM25_QUERY_TERMS);
    expect(lexicalTerms.slice(0, 3)).toEqual(["galaxy", "lensing", "word0"]);
  });

  it("still fills the basis when over-fetched hits turn out unusable", async () => {
    const usable = [1, 2, 3, 4, 5].map((index) => record(index));
    const unusable = [10, 11, 12, 13, 14, 15].map((index) => record(index, { source: "other" as "arxiv" }));
    const { port } = setup(
      [...unusable, ...usable],
      [...unusable.map(({ paperKey }) => paperKey), ...usable.map(({ paperKey }) => paperKey)],
    );
    const found = await port.findSimilar(query, 5);
    expect(found.map(({ paperKey }) => paperKey)).toEqual(usable.map(({ paperKey }) => paperKey));
  });

  it("excludes the query paper itself and still fills the basis from the extra match", async () => {
    const own = { ...record(9), paperKey: query.paperKey };
    const papers = [own, ...[1, 2, 3, 4, 5].map((index) => record(index))];
    const { port } = setup(papers, [query.paperKey, key(1), key(2), key(3), key(4), key(5)]);
    const found = await port.findSimilar(query, 5);
    expect(found.map(({ paperKey }) => paperKey)).toEqual([key(1), key(2), key(3), key(4), key(5)]);
  });

  it("returns at most five papers even when asked for more or when more match", async () => {
    const papers = [1, 2, 3, 4, 5, 6, 7, 8].map((index) => record(index));
    const { port, search } = setup(papers, papers.map(({ paperKey }) => paperKey));
    expect(await port.findSimilar(query, 20)).toHaveLength(5);
    expect(search.mock.calls[0]![0].limit).toBe(16);
  });

  it("drops matches that have no usable catalog record instead of truncating or guessing", async () => {
    const papers = [
      record(1),
      record(2, { abstract: "" }),
      record(3, { title: "   " }),
      record(4, { title: "T".repeat(2001) }),
      record(5, { abstract: "A".repeat(6001) }),
      record(6, { authors: [] }),
      record(7, { source: "other" as "arxiv" }),
      record(8),
    ];
    const { port } = setup(papers, [key(1), key(2), key(3), key(4), key(5), key(6), key(7), key(99), key(8)]);
    const found = await port.findSimilar(query, 5);
    expect(found.map(({ paperKey }) => paperKey)).toEqual([key(1), key(8)]);
  });

  it("returns an empty basis when the index finds nothing", async () => {
    const { port } = setup([record(1)], []);
    await expect(port.findSimilar(query, 5)).resolves.toEqual([]);
  });

  it("checks authorization before every lookup", async () => {
    const { port, assertAuthorized } = setup([record(1)], [key(1)]);
    await port.findSimilar(query, 5);
    await port.findSimilar({ ...query, paperKey: "arxiv:2609.00002" }, 5);
    expect(assertAuthorized).toHaveBeenCalledTimes(2);
  });

  it("reports itself unavailable, without searching or reading the catalog, when authorization fails", async () => {
    const { port, search, catalog } = setup([record(1)], [key(1)], {
      assertAuthorized: async () => { throw new Error("library processing consent was revoked"); },
    });
    await expect(port.findSimilar(query, 5)).rejects.toBeInstanceOf(LibraryPriorWorkUnavailableError);
    expect(search).not.toHaveBeenCalled();
    expect(catalog.load).not.toHaveBeenCalled();
  });

  it("lets cancellation through the authorization check unchanged", async () => {
    const { port, search } = setup([record(1)], [key(1)], {
      assertAuthorized: () => { throw new RunCancelledError(); },
    });
    await expect(port.findSimilar(query, 5)).rejects.toBeInstanceOf(RunCancelledError);
    expect(search).not.toHaveBeenCalled();
  });

  it("propagates search failures as ordinary errors, not as unavailability", async () => {
    const { port, search } = setup([record(1)], [key(1)]);
    search.mockRejectedValueOnce(new Error("index read failed"));
    const failure = await port.findSimilar(query, 5).catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(Error);
    expect(failure).not.toBeInstanceOf(LibraryPriorWorkUnavailableError);
  });
});

describe("createLibraryPriorWork with knowledge-base file documents", () => {
  const fileKey = (digit: string): string => `file:sha256:${digit.repeat(64)}`;
  const fileRecord = (
    digit: string,
    overrides: Partial<FullTextPaperKnowledgeRecord> = {},
  ): FullTextPaperKnowledgeRecord => ({
    paperKey: fileKey(digit),
    status: "ready",
    modelId: "model-a",
    dimension: 2,
    title: `Library document ${digit}`,
    abstract: `Extracted abstract ${digit}`,
    filePaths: [`library/${digit}.pdf`],
    observationFingerprints: [],
    chunkCount: 3,
    updatedAt: "2026-10-15T00:00:00.000Z",
    ...overrides,
  });

  function setupFiles(
    records: FullTextPaperKnowledgeRecord[],
    matches: string[],
    papers: PersonalLibraryPaperRecord[] = [],
  ) {
    const catalog = { load: vi.fn(async () => ({ papers: Object.fromEntries(papers.map((paper) => [paper.paperKey, paper])) })) };
    const search = vi.fn(async (_request: LibraryPriorWorkSearchRequest) =>
      matches.map((paperKey) => ({ paperKey })));
    const knowledgeBase = {
      loadManifest: vi.fn(async () => ({
        papers: Object.fromEntries(records.map((entry) => [entry.paperKey, entry])),
      })),
    };
    const port = createLibraryPriorWork({
      catalog, search, knowledgeBase, assertAuthorized: async () => {}, mode: "lexical",
    });
    return { port, search, knowledgeBase, catalog };
  }

  it("turns a ready file document with title and abstract into a basis paper with only those fields", async () => {
    const { port } = setupFiles([fileRecord("a")], [fileKey("a")]);
    await expect(port.findSimilar(query, 5)).resolves.toEqual([
      { paperKey: fileKey("a"), title: "Library document a", abstract: "Extracted abstract a" },
    ]);
  });

  it("keeps hit order across arXiv and file documents", async () => {
    const { port } = setupFiles(
      [fileRecord("a"), fileRecord("b")],
      [fileKey("a"), key(1), fileKey("b")],
      [record(1)],
    );
    const found = await port.findSimilar(query, 5);
    expect(found.map(({ paperKey }) => paperKey)).toEqual([fileKey("a"), key(1), fileKey("b")]);
  });

  it("drops failed, title-less, abstract-less, blank and out-of-bound file documents", async () => {
    const records = [
      fileRecord("1"),
      fileRecord("2", { status: "failed" }),
      fileRecord("3", { abstract: undefined }),
      fileRecord("4", { abstract: "" }),
      fileRecord("5", { title: undefined }),
      fileRecord("6", { title: " padded " }),
      fileRecord("7", { title: "T".repeat(2001) }),
      fileRecord("8", { abstract: "A".repeat(6001) }),
      fileRecord("9"),
    ];
    const { port } = setupFiles(records, records.map(({ paperKey }) => paperKey).concat(fileKey("f")));
    const found = await port.findSimilar(query, 5);
    expect(found.map(({ paperKey }) => paperKey)).toEqual([fileKey("1"), fileKey("9")]);
  });

  it("drops a file document whose normalized title equals the digest paper's title", async () => {
    const same = fileRecord("a", { title: "  A New   Galaxy-Survey! " });
    const { port } = setupFiles([same, fileRecord("b")], [same.paperKey, fileKey("b")]);
    const found = await port.findSimilar({ ...query, title: "A new galaxy survey" }, 5);
    expect(found.map(({ paperKey }) => paperKey)).toEqual([fileKey("b")]);
  });

  it("does not drop a file document that merely shares words with the digest title", async () => {
    const { port } = setupFiles([fileRecord("a", { title: "A new galaxy survey of clusters" })], [fileKey("a")]);
    expect(await port.findSimilar({ ...query, title: "A new galaxy survey" }, 5)).toHaveLength(1);
  });

  it("loads the manifest lazily, once per lookup, and not at all without file hits", async () => {
    const withFiles = setupFiles([fileRecord("a"), fileRecord("b")], [fileKey("a"), fileKey("b"), key(1)], [record(1)]);
    await withFiles.port.findSimilar(query, 5);
    expect(withFiles.knowledgeBase.loadManifest).toHaveBeenCalledTimes(1);
    const arxivOnly = setupFiles([fileRecord("a")], [key(1), key(2)], [record(1), record(2)]);
    await arxivOnly.port.findSimilar(query, 5);
    expect(arxivOnly.knowledgeBase.loadManifest).not.toHaveBeenCalled();
    const none = setupFiles([fileRecord("a")], []);
    await none.port.findSimilar(query, 5);
    expect(none.knowledgeBase.loadManifest).not.toHaveBeenCalled();
  });

  it("drops file hits quietly when no knowledge base is wired", async () => {
    const { port } = setup([record(1)], [fileKey("a"), key(1)]);
    const found = await port.findSimilar(query, 5);
    expect(found.map(({ paperKey }) => paperKey)).toEqual([key(1)]);
  });

  it("fills the basis from file documents when they outrank arXiv papers", async () => {
    const files = ["1", "2", "3", "4", "5", "6"].map((digit) => fileRecord(digit));
    const { port } = setupFiles(files, files.map(({ paperKey }) => paperKey));
    expect(await port.findSimilar(query, 5)).toHaveLength(5);
  });

  it("lets a manifest read failure fail the lookup instead of guessing", async () => {
    const { port, knowledgeBase } = setupFiles([fileRecord("a")], [fileKey("a")]);
    knowledgeBase.loadManifest.mockRejectedValueOnce(new Error("manifest unreadable"));
    await expect(port.findSimilar(query, 5)).rejects.toThrow("manifest unreadable");
  });
});
