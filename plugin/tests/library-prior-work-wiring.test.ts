import { describe, expect, it, vi } from "vitest";
import {
  LibraryPriorWorkUnavailableError,
  type EmbeddingModel,
  type LibraryConnectionStatus,
  type PersonalLibraryPaperRecord,
} from "@arxiv-daily/core";
import {
  wireLibraryPriorWork,
  type LibraryPriorWorkWiringDeps,
} from "../src/library/prior-work-wiring";

const authorized: LibraryConnectionStatus = {
  kind: "authorized",
  rootLabel: "papers",
  grantedAt: "2026-10-01T00:00:00.000Z",
};
const query = { paperKey: "arxiv:2609.00001", title: "A new galaxy survey", abstract: "We map galaxies." };

const record = (index: number): PersonalLibraryPaperRecord => ({
  paperKey: `arxiv:2501.0000${index}`,
  source: "arxiv",
  externalId: `2501.0000${index}`,
  title: `Library paper ${index}`,
  authors: ["L. Reader"],
  abstract: `Library abstract ${index}`,
  published: "2025-01-02",
  updated: "2025-01-03",
  primaryCategory: "astro-ph.GA",
  categories: ["astro-ph.GA"],
  evidenceDepth: "metadata-and-abstract",
  filePaths: [`library/${index}.pdf`],
});

function setup(overrides: Partial<LibraryPriorWorkWiringDeps> = {}) {
  const embed = vi.fn(async () => [new Float32Array([1])]);
  const embedding: EmbeddingModel = { modelId: "model", dimension: 1, prefixPolicy: "none", embed };
  const papers = Object.fromEntries([record(1), record(2)].map((paper) => [paper.paperKey, paper]));
  const loadCatalog = vi.fn(async () => ({ papers }));
  const openSearchSource = vi.fn(async () => ({
    store: { tag: "store" },
    sourceManifest: { tag: "manifest" },
    generationStore: { tag: "generation" },
    embedding,
  }) as never);
  const searchCore = vi.fn(async (_input: unknown) => [{ paperKey: "arxiv:2501.00002" }, { paperKey: "arxiv:2501.00001" }]);
  const getStatus = vi.fn((): LibraryConnectionStatus => authorized);
  const assertRemoteEmbeddingReady = vi.fn();
  const deps: LibraryPriorWorkWiringDeps = {
    getStatus,
    embeddingMode: "local",
    assertRemoteEmbeddingReady,
    loadCatalog,
    openSearchSource,
    searchCore: searchCore as never,
    ...overrides,
  };
  return { deps, embed, embedding, loadCatalog, openSearchSource, searchCore, getStatus, assertRemoteEmbeddingReady };
}

describe("wireLibraryPriorWork", () => {
  it.each<LibraryConnectionStatus>([
    { kind: "disconnected" },
    { kind: "authorization-required", rootLabel: "papers" },
    { kind: "authorization-invalidated", rootLabel: "papers" },
  ])("supplies no port while the library is %j", (status) => {
    const { deps, openSearchSource } = setup({ getStatus: () => status });
    expect(wireLibraryPriorWork(deps)).toBeNull();
    expect(openSearchSource).not.toHaveBeenCalled();
  });

  it("searches lexically with a local embedding model and never embeds", async () => {
    const { deps, embed, searchCore, assertRemoteEmbeddingReady } = setup();
    const port = wireLibraryPriorWork(deps)!;
    const basis = await port.findSimilar(query, 5);

    expect(basis.map(({ paperKey }) => paperKey)).toEqual(["arxiv:2501.00002", "arxiv:2501.00001"]);
    expect(searchCore).toHaveBeenCalledTimes(1);
    const input = searchCore.mock.calls[0]![0] as {
      mode: string; queryText: string; lexicalQueryText: string; limit: number; embedding: EmbeddingModel;
      titles: ReadonlyMap<string, string>; store: unknown; generationStore: unknown;
    };
    expect(input.mode).toBe("lexical");
    expect(input.queryText).toBe(`${query.title}\n${query.abstract}`);
    expect(input.lexicalQueryText).toBe(input.queryText);
    expect(input.limit).toBe(16);
    expect(input.store).toEqual({ tag: "store" });
    expect(input.generationStore).toEqual({ tag: "generation" });
    expect(input.titles.get("arxiv:2501.00001")).toBe("Library paper 1");
    // Lexical search must not be able to load the embedding model, even by mistake.
    expect(input.embedding.modelId).toBe("model");
    await expect(input.embedding.embed(["x"])).rejects.toThrow();
    expect(embed).not.toHaveBeenCalled();
    expect(assertRemoteEmbeddingReady).not.toHaveBeenCalled();
  });

  it("resolves file: hits through the manifest the search was bound to, without reading it again", async () => {
    const fileKey = `file:sha256:${"ab".repeat(32)}`;
    const sourceManifest = {
      papers: {
        [fileKey]: {
          paperKey: fileKey, status: "ready", title: "My library PDF", abstract: "Extracted abstract.",
          filePaths: ["library/x.pdf"],
        },
        [`file:sha256:${"cd".repeat(32)}`]: {
          paperKey: `file:sha256:${"cd".repeat(32)}`, status: "failed", title: "Broken", abstract: "x",
          filePaths: [],
        },
      },
    };
    const loadManifest = vi.fn();
    const base = setup();
    const { deps, searchCore, openSearchSource } = setup({
      openSearchSource: vi.fn(async () => ({
        store: { loadManifest }, sourceManifest, generationStore: {}, embedding: base.embedding,
      }) as never),
    });
    searchCore.mockResolvedValueOnce([
      { paperKey: fileKey }, { paperKey: `file:sha256:${"cd".repeat(32)}` }, { paperKey: "arxiv:2501.00001" },
    ]);
    const port = wireLibraryPriorWork(deps)!;
    const basis = await port.findSimilar(query, 5);
    expect(basis).toEqual([
      { paperKey: fileKey, title: "My library PDF", abstract: "Extracted abstract." },
      expect.objectContaining({ paperKey: "arxiv:2501.00001" }),
    ]);
    expect(deps.openSearchSource).toHaveBeenCalledTimes(1);
    expect(loadManifest).not.toHaveBeenCalled();
    void openSearchSource;
  });

  it("falls back to the store when the search source carries no manifest snapshot", async () => {
    const fileKey = `file:sha256:${"ab".repeat(32)}`;
    const loadManifest = vi.fn(async () => ({
      papers: { [fileKey]: { paperKey: fileKey, status: "ready", title: "T", abstract: "A", filePaths: [] } },
    }));
    const base = setup();
    const { deps, searchCore } = setup({
      openSearchSource: vi.fn(async () => ({
        store: { loadManifest }, generationStore: {}, embedding: base.embedding,
      }) as never),
    });
    searchCore.mockResolvedValueOnce([{ paperKey: fileKey }]);
    const basis = await wireLibraryPriorWork(deps)!.findSimilar(query, 5);
    expect(basis).toEqual([{ paperKey: fileKey, title: "T", abstract: "A" }]);
    expect(loadManifest).toHaveBeenCalledTimes(1);
  });

  it("hands the core search an abstract cut to the index term limit while keeping the full query text", async () => {
    const { deps, searchCore } = setup();
    const port = wireLibraryPriorWork(deps)!;
    const abstract = Array.from({ length: 120 }, (_, index) => `word${index}`).join(" ");
    await port.findSimilar({ ...query, abstract }, 5);
    const input = searchCore.mock.calls[0]![0] as { queryText: string; lexicalQueryText: string };
    expect(input.queryText).toBe(`${query.title}\n${abstract}`);
    expect(new Set(input.lexicalQueryText.split(/\s+/)).size).toBe(64);
  });

  it("searches hybrid with the real embedding model for a complete remote configuration", async () => {
    const { deps, embedding, searchCore, assertRemoteEmbeddingReady } = setup({ embeddingMode: "remote" });
    const port = wireLibraryPriorWork(deps)!;
    await port.findSimilar(query, 5);

    const input = searchCore.mock.calls[0]![0] as { mode: string; embedding: EmbeddingModel };
    expect(input.mode).toBe("hybrid");
    expect(input.embedding).toBe(embedding);
    expect(assertRemoteEmbeddingReady).toHaveBeenCalled();
  });

  it("is unavailable for an incomplete remote configuration without searching", async () => {
    const { deps, searchCore, openSearchSource } = setup({
      embeddingMode: "remote",
      assertRemoteEmbeddingReady: () => {
        throw new Error("Remote embedding configuration incomplete: model is empty");
      },
    });
    const port = wireLibraryPriorWork(deps)!;
    await expect(port.findSimilar(query, 5)).rejects.toBeInstanceOf(LibraryPriorWorkUnavailableError);
    expect(searchCore).not.toHaveBeenCalled();
    expect(openSearchSource).not.toHaveBeenCalled();
  });

  it("re-checks the live status on every lookup so lost consent stops the run midway", async () => {
    let status: LibraryConnectionStatus = authorized;
    const { deps, searchCore } = setup({ getStatus: () => status });
    const port = wireLibraryPriorWork(deps)!;
    await port.findSimilar(query, 5);
    expect(searchCore).toHaveBeenCalledTimes(1);

    status = { kind: "authorization-invalidated", rootLabel: "papers" };
    await expect(port.findSimilar(query, 5)).rejects.toBeInstanceOf(LibraryPriorWorkUnavailableError);
    status = { kind: "disconnected" };
    await expect(port.findSimilar(query, 5)).rejects.toBeInstanceOf(LibraryPriorWorkUnavailableError);
    expect(searchCore).toHaveBeenCalledTimes(1);
  });

  it("passes the abort signal through to the search", async () => {
    const { deps, searchCore } = setup();
    const controller = new AbortController();
    await wireLibraryPriorWork(deps)!.findSimilar(query, 5, controller.signal);
    expect((searchCore.mock.calls[0]![0] as { signal?: AbortSignal }).signal).toBe(controller.signal);
  });

  it("loads the catalog once per port and reuses it across lookups", async () => {
    const { deps, loadCatalog } = setup();
    const port = wireLibraryPriorWork(deps)!;
    await port.findSimilar(query, 5);
    await port.findSimilar({ ...query, paperKey: "arxiv:2609.00002" }, 5);
    expect(loadCatalog).toHaveBeenCalledTimes(1);
  });

  it("retries the catalog load after a failed one", async () => {
    const { deps, loadCatalog } = setup();
    loadCatalog.mockRejectedValueOnce(new Error("disk"));
    const port = wireLibraryPriorWork(deps)!;
    await expect(port.findSimilar(query, 5)).rejects.toThrow("disk");
    await expect(port.findSimilar(query, 5)).resolves.toHaveLength(2);
    expect(loadCatalog).toHaveBeenCalledTimes(2);
  });
});
