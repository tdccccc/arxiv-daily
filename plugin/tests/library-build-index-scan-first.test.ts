import "obsidian";
import { Notice } from "obsidian";
import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  OperationRegistry,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
  type StorageAdapter,
} from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";
import { LibraryIndexStatusStore } from "../src/library/index-status";
import { createLibraryConnection } from "../src/library/connection";

/**
 * "Build index scans the library first when it has never been scanned"
 * (0.5.0): `indexPersonalLibraryFullText` used to only ever read the stored
 * catalog (`main.ts`'s old behavior, see the P8/review investigation at
 * `/tmp/arxiv-library-firstrun.md`), so a library that was never scanned
 * always indexed zero units with no explanation. It now scans first — as the
 * first phase of the same cancellable operation — exactly when the stored
 * catalog has never been scanned for the connected folder (`lastScan ===
 * null`, which `PersonalLibraryCatalogStore.load` already produces for a
 * missing, corrupt, or different-folder/extension-set catalog).
 *
 * These tests use the real `PersonalLibraryCatalogStore` (real in-memory
 * storage, not mocked) so the scan really walks a fake folder and really
 * writes a catalog — only the full-text knowledge base store, the embedding
 * model and the post-index direction update are stubbed. The metadata-failure
 * regression supplies a readable document parser to prove that arXiv lookup
 * failure does not exclude local PDF text from indexing. Other scenarios use
 * empty or non-PDF inventories to isolate scan lifecycle behavior.
 */

function makeStorage() {
  const files = new Map<string, string>();
  const directories = new Set<string>();
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/"),
    exists: async (path) => files.has(path) || directories.has(path),
    readText: async (path) => {
      const content = files.get(path);
      if (content === undefined) throw new Error(`missing ${path}`);
      return content;
    },
    writeText: async (path, content) => { files.set(path, content); },
    writeTextAtomic: async (path, content) => { files.set(path, content); },
    mkdir: async (path) => { directories.add(path); },
    rename: async (from, to) => {
      const content = files.get(from);
      if (content === undefined) throw new Error(`missing ${from}`);
      files.set(to, content);
      files.delete(from);
    },
    remove: async (path) => { files.delete(path); directories.delete(path); },
  };
  return { files, storage };
}

function knowledgeBaseStore(scopeFingerprint: string, identificationFingerprint: string) {
  const manifest = {
    schemaVersion: 1 as const,
    revision: 0,
    scopeFingerprint,
    identificationFingerprint,
    modelId: "",
    dimension: 0,
    updatedAt: "2026-09-01T00:00:00.000Z",
    papers: {},
  };
  let current = manifest;
  return {
    loadManifest: vi.fn(async () => structuredClone(current)),
    replaceManifest: vi.fn(async (next: typeof manifest) => {
      current = { ...structuredClone(next), revision: current.revision + 1, updatedAt: "2026-09-01T00:01:00.000Z" };
      return structuredClone(current);
    }),
    loadPaper: vi.fn(async () => null),
    savePaper: vi.fn(async () => undefined),
    removePaper: vi.fn(async () => undefined),
    removeAll: vi.fn(async () => undefined),
    paths: {
      directory: "legacy",
      manifest: {
        directory: "legacy",
        documentPath: "legacy/manifest.json",
        backupPath: "legacy/manifest.json.backup",
      },
      papersDirectory: "legacy/papers",
    },
  };
}

function fixture(entries: Array<{ path: string; size: number; mtimeMs: number }> = []) {
  const { storage } = makeStorage();
  const settings = structuredClone(DEFAULT_SETTINGS);
  const connection = createLibraryConnection("/private/library", "1:2");
  const scopeFingerprint = createPersonalLibraryScopeFingerprint({
    rootIdentity: connection.rootIdentity,
    eligibleExtensions: connection.eligibleExtensions,
  });
  const identificationFingerprint = createPersonalLibraryIdentificationFingerprint(
    connection.eligibleExtensions,
  );
  const operations = new OperationRegistry();
  const fetchMetadataByIds = vi.fn(async () => new Map());
  const source = {
    canonicalRoot: connection.selectedRoot,
    rootIdentity: connection.rootIdentity,
    inventory: vi.fn(async () => ({
      entries: entries.map((entry) => ({ ...entry, type: "file" as const })),
      truncated: false,
    })),
    readBinary: vi.fn(async (): Promise<ArrayBuffer> => {
      throw new Error("must not read PDF bytes in this fixture");
    }),
  };
  const legacy = knowledgeBaseStore(scopeFingerprint, identificationFingerprint);
  const embeddingModel = {
    modelId: "fixture-model",
    dimension: 2,
    prefixPolicy: "none" as const,
    embed: vi.fn(async (texts: string[]) => texts.map(() => new Float32Array([0, 0]))),
  };
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  const internals = plugin as unknown as Record<string, any>;
  Object.assign(plugin, {
    settings,
    logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn(), setSensitiveValues: vi.fn() },
    host: { storage, http: {}, markupParser: {} },
    progress: {
      setTask: vi.fn(),
      setComplete: vi.fn(),
      setError: vi.fn(),
      setIdle: vi.fn(),
      setDisabled: vi.fn(),
    },
    operations,
    libraryIndexStatus: new LibraryIndexStatusStore(),
    libraryConnection: connection,
    librarySource: source,
    libraryIndexedPapers: [],
    libraryConnectionRevision: 0,
    libraryOutputRevision: 0,
    librarySelectionRevision: 0,
    libraryMutationQueue: Promise.resolve(),
    buildArxivFetcher: vi.fn(() => ({ fetchMetadataByIds })),
  });
  internals.buildFullTextKnowledgeBaseStore = vi.fn(() => legacy);
  internals.buildEmbeddingModel = vi.fn(() => embeddingModel);

  return { plugin, internals, source, fetchMetadataByIds, legacy };
}

function arxivPaper(id: string) {
  return {
    title: `Paper ${id}`, authorNames: ["A. Author"], abstract: `Abstract of ${id}.`,
    published: "2026-08-01T00:00:00Z", updated: "2026-08-02T00:00:00Z",
    primaryCategory: "cs.AI", categories: ["cs.AI"],
  };
}

describe("Build index scans a library that changed since its last scan", () => {
  function changedFixture() {
    const entries = [{ path: "2601.00001.pdf", size: 100, mtimeMs: 12 }];
    const runtime = fixture(entries);
    runtime.fetchMetadataByIds.mockImplementation(async (ids: string[]) => new Map(
      ids.map((id) => [id, arxivPaper(id)] as const),
    ));
    runtime.source.readBinary.mockResolvedValue(new TextEncoder().encode("readable PDF fixture").buffer);
    runtime.internals.buildFullTextExtractor = vi.fn(() => ({
      provenance: { id: "fixture-extractor", version: "1" },
      extractPdfText: async () => ({
        pages: ["Scientific paper title\nAbstract\n" + "Readable scientific paper content. ".repeat(30)],
      }),
    }));
    return { ...runtime, entries };
  }
  const idsFetched = (fetch: ReturnType<typeof vi.fn>) => fetch.mock.calls.map(([ids]) => ids);

  it("picks up PDFs added after the last scan with only Index, asking arXiv only about them", async () => {
    const { plugin, entries, fetchMetadataByIds } = changedFixture();
    await plugin.scanPersonalLibrary();
    await plugin.indexPersonalLibraryFullText();
    expect(idsFetched(fetchMetadataByIds)).toEqual([["2601.00001"]]);

    entries.push(
      { path: "2601.00002.pdf", size: 100, mtimeMs: 13 },
      { path: "sub/2601.00003.pdf", size: 100, mtimeMs: 14 },
    );
    const summary = await plugin.indexPersonalLibraryFullText();

    expect(idsFetched(fetchMetadataByIds)).toEqual([["2601.00001"], ["2601.00002", "2601.00003"]]);
    // The stubbed knowledge-base store keeps nothing between runs, so every paper is indexed again.
    expect(summary).toMatchObject({ indexed: 3, failed: 0 });
    expect(Object.keys(plugin.getPersonalLibraryProfileSnapshot().catalog!.papers).sort())
      .toEqual(["arxiv:2601.00001", "arxiv:2601.00002", "arxiv:2601.00003"]);
    expect(plugin.getLastFullTextIndexLibraryContext()).toMatchObject({ scannedBeforeIndexing: true, readyPapers: 3 });
  });

  it("nudges after Index alone when enough new PDFs were added to the folder", async () => {
    const { plugin, internals, entries } = changedFixture();
    entries.length = 0;
    await plugin.scanPersonalLibrary();
    internals.libraryProposal = {
      revision: 1, proposalId: "proposal.1",
      scopeFingerprint: createPersonalLibraryScopeFingerprint({
        rootIdentity: internals.libraryConnection.rootIdentity,
        eligibleExtensions: internals.libraryConnection.eligibleExtensions,
      }),
      catalogInputPapers: [],
    };
    Notice.calls = [];
    entries.push(
      { path: "2601.00001.pdf", size: 100, mtimeMs: 12 },
      { path: "2601.00002.pdf", size: 100, mtimeMs: 13 },
      { path: "2601.00003.pdf", size: 100, mtimeMs: 14 },
    );

    await plugin.indexPersonalLibraryFullText();

    expect(plugin.getPersonalLibraryProfileSnapshot().newPaperKeys).toHaveLength(3);
    const nudges = Notice.calls.filter(({ message }) => /direction suggestions/.test(message));
    expect(nudges).toHaveLength(1);
    expect(nudges[0]!.message).toContain("3 new papers in your library");
  });

  it("scans for removed and renamed PDFs too", async () => {
    const { plugin, entries, fetchMetadataByIds } = changedFixture();
    entries.push({ path: "2601.00002.pdf", size: 100, mtimeMs: 13 });
    await plugin.scanPersonalLibrary();
    expect(idsFetched(fetchMetadataByIds)).toEqual([["2601.00001", "2601.00002"]]);

    entries.splice(0, 1);
    entries[0] = { path: "moved/2601.00002.pdf", size: 100, mtimeMs: 13 };
    await plugin.indexPersonalLibraryFullText();

    // Nothing new to look up: the renamed paper is already in the catalog.
    expect(idsFetched(fetchMetadataByIds)).toEqual([["2601.00001", "2601.00002"]]);
    const catalog = plugin.getPersonalLibraryProfileSnapshot().catalog!;
    expect(Object.keys(catalog.files)).toEqual(["moved/2601.00002.pdf"]);
    expect(Object.keys(catalog.papers)).toEqual(["arxiv:2601.00002"]);
  });

  it("does not scan, or contact arXiv, when the folder has not changed", async () => {
    const { plugin, source, internals, fetchMetadataByIds } = changedFixture();
    await plugin.scanPersonalLibrary();
    await plugin.indexPersonalLibraryFullText();
    const catalogRevision = plugin.getPersonalLibraryProfileSnapshot().catalog!.revision;
    const listings = source.inventory.mock.calls.length;
    const scanFolder = vi.spyOn(internals, "scanPersonalLibraryFolder");

    await plugin.indexPersonalLibraryFullText();

    expect(source.inventory.mock.calls.length).toBe(listings + 1);
    expect(scanFolder).not.toHaveBeenCalled();
    expect(idsFetched(fetchMetadataByIds)).toEqual([["2601.00001"]]);
    expect(plugin.getPersonalLibraryProfileSnapshot().catalog!.revision).toBe(catalogRevision);
    expect(plugin.getLastFullTextIndexLibraryContext()).toMatchObject({ scannedBeforeIndexing: false });
  });

  it("indexes the stored catalog as before when the folder cannot be listed for the change check", async () => {
    const { plugin, source, internals, fetchMetadataByIds } = changedFixture();
    await plugin.scanPersonalLibrary();
    source.inventory.mockRejectedValueOnce(new Error("folder unreachable"));
    const scanFolder = vi.spyOn(internals, "scanPersonalLibraryFolder");

    await expect(plugin.indexPersonalLibraryFullText()).resolves.toMatchObject({ indexed: 1 });

    expect(scanFolder).not.toHaveBeenCalled();
    expect(idsFetched(fetchMetadataByIds)).toEqual([["2601.00001"]]);
  });
});

describe("Build index scans a never-scanned library first", () => {
  it("shows local model preparation in the active library operation", async () => {
    const { plugin, internals } = fixture([]);
    const embedding = internals.buildEmbeddingModel();
    internals.buildEmbeddingModel.mockImplementation((options: {
      signal: AbortSignal;
      onProgress: (progress: { phase: string; message: string; progress?: number }) => void;
    }) => {
      expect(options?.signal).toBeInstanceOf(AbortSignal);
      expect(options?.onProgress).toBeTypeOf("function");
      options.onProgress({ phase: "loading", message: "Loading local model files", progress: 42 });
      expect(plugin.libraryIndexStatus.snapshot().activity?.phase).toContain("Loading local model files");
      expect(plugin.libraryIndexStatus.snapshot().activity?.phase).toContain("42%");
      options.onProgress({ phase: "ready", message: "Model ready" });
      expect(plugin.libraryIndexStatus.snapshot().activity?.phase).toBe("extracting and embedding titles and abstracts");
      return embedding;
    });
    await plugin.indexPersonalLibraryFullText();
  });

  it("indexes readable PDFs even when their arXiv metadata lookup fails", async () => {
    const { plugin, source, internals, fetchMetadataByIds } = fixture([
      { path: "2601.00001.pdf", size: 100, mtimeMs: 12 },
    ]);
    fetchMetadataByIds.mockRejectedValueOnce(new Error("arXiv unavailable"));
    source.readBinary.mockResolvedValue(new TextEncoder().encode("readable PDF fixture").buffer);
    internals.buildFullTextExtractor = vi.fn(() => ({
      provenance: { id: "fixture-extractor", version: "1" },
      extractPdfText: async () => ({
        pages: ["Scientific paper title\nAbstract\n" + "Readable scientific paper content. ".repeat(30)],
      }),
    }));

    const summary = await plugin.indexPersonalLibraryFullText();

    expect(fetchMetadataByIds).toHaveBeenCalledTimes(1);
    expect(summary.indexed).toBe(1);
    expect(summary.outcomes[0]?.paperKey).toMatch(/^file:/);
    expect(plugin.getPersonalLibraryProfileSnapshot().indexedPapers).toEqual([
      expect.objectContaining({ paperKey: summary.outcomes[0]?.paperKey }),
    ]);
    expect(plugin.getLastFullTextIndexLibraryContext()).toMatchObject({
      readyPapers: 0, metadataFetchFailures: 1, unresolvedFallbackFiles: 1,
    });
  });

  it("scans an empty, never-scanned folder before indexing, and explains the zero-paper result", async () => {
    const { plugin, source } = fixture([]);

    const summary = await plugin.indexPersonalLibraryFullText();

    expect(source.inventory).toHaveBeenCalledTimes(1);
    expect(summary.outcomes).toEqual([]);
    const context = plugin.getLastFullTextIndexLibraryContext();
    expect(context).toEqual({
      totalFiles: 0,
      readyPapers: 0,
      unresolvedFallbackFiles: 0,
      metadataFetchFailures: 0,
      scannedBeforeIndexing: true,
    });
  });

  it("scans a folder with only non-PDF files, finds zero indexable units, and still records the scan", async () => {
    const { plugin, source } = fixture([{ path: "draft.md", size: 5, mtimeMs: 12 }]);

    const summary = await plugin.indexPersonalLibraryFullText();

    expect(source.inventory).toHaveBeenCalledTimes(1);
    expect(summary.outcomes).toEqual([]);
    const context = plugin.getLastFullTextIndexLibraryContext();
    // The file exists in the catalog (as "unrelated": unsupported file type)
    // but contributes no indexable unit — a different zero-unit explanation
    // than an altogether empty folder.
    expect(context).toMatchObject({ totalFiles: 1, readyPapers: 0, unresolvedFallbackFiles: 0 });
  });

  it("does not scan again once the library has already been scanned", async () => {
    const { plugin, source } = fixture([]);
    await plugin.scanPersonalLibrary();
    expect(source.inventory).toHaveBeenCalledTimes(1);

    const summary = await plugin.indexPersonalLibraryFullText();

    // The index build lists the folder locally to see whether it changed, but
    // an unchanged folder is not scanned again (no catalog write, no network).
    expect(source.inventory).toHaveBeenCalledTimes(2);
    expect(summary.outcomes).toEqual([]);
    expect(plugin.getLastFullTextIndexLibraryContext()).toMatchObject({ scannedBeforeIndexing: false });
  });

  it("scans again when the saved catalog belongs to another root", async () => {
    const { plugin, source, internals } = fixture([]);
    await plugin.scanPersonalLibrary();
    internals.libraryConnection = createLibraryConnection("/private/other-library", "3:4");
    source.canonicalRoot = "/private/other-library";
    source.rootIdentity = "3:4";
    internals.buildFullTextKnowledgeBaseStore = vi.fn(() => knowledgeBaseStore(
      createPersonalLibraryScopeFingerprint({ rootIdentity: "3:4", eligibleExtensions: [".pdf"] }),
      createPersonalLibraryIdentificationFingerprint([".pdf"]),
    ));
    await plugin.indexPersonalLibraryFullText();
    expect(source.inventory).toHaveBeenCalledTimes(2);
    expect(plugin.getLastFullTextIndexLibraryContext()?.scannedBeforeIndexing).toBe(true);
  });

  it("does not index when cancelled just after the scan catalog is saved", async () => {
    const { plugin, internals } = fixture([]);
    const originalWrite = internals.host.storage.writeTextAtomic;
    internals.host.storage.writeTextAtomic = async (...args: unknown[]) => {
      await originalWrite(...args);
      plugin.cancelPersonalLibraryIndexing();
    };
    await expect(plugin.indexPersonalLibraryFullText()).rejects.toMatchObject({ name: "RunCancelledError", message: "cancelled from settings" });
    expect(internals.buildFullTextKnowledgeBaseStore).not.toHaveBeenCalled();
    expect(plugin.operations.snapshot()).toEqual([]);
  });

  it("is cancellable while scanning, and never starts indexing", async () => {
    const { plugin, source, internals } = fixture([]);
    source.inventory.mockImplementationOnce(({ signal }: { signal?: AbortSignal } = {}) =>
      new Promise((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
      }),
    );

    const run = plugin.indexPersonalLibraryFullText();
    await vi.waitFor(() => expect(source.inventory).toHaveBeenCalledTimes(1));
    expect(plugin.libraryIndexStatus.snapshot().activity?.phase).toBe("scanning the library folder");

    expect(plugin.cancelPersonalLibraryIndexing()).toBe(true);
    await expect(run).rejects.toMatchObject({ name: "RunCancelledError", message: "cancelled from settings" });

    expect(internals.buildFullTextKnowledgeBaseStore).not.toHaveBeenCalled();
    expect(plugin.libraryIndexStatus.snapshot().activity).toBeUndefined();
    expect(plugin.operations.snapshot()).toEqual([]);
  });

  it("surfaces a scan failure clearly and never starts indexing", async () => {
    const { plugin, internals } = fixture([]);
    internals.librarySource.inventory.mockRejectedValueOnce(new Error("folder unreachable"));

    await expect(plugin.indexPersonalLibraryFullText()).rejects.toThrow(
      "Could not scan the library folder before indexing: folder unreachable",
    );

    expect(internals.buildFullTextKnowledgeBaseStore).not.toHaveBeenCalled();
    expect(plugin.libraryIndexStatus.snapshot().activity).toBeUndefined();
    expect(plugin.operations.snapshot()).toEqual([]);
  });

  it("rejects a concurrent manual scan while the build's own scan is in flight", async () => {
    const { plugin, source } = fixture([]);
    let finishInventory!: () => void;
    source.inventory.mockImplementationOnce(() => new Promise((resolve) => {
      finishInventory = () => resolve({ entries: [], truncated: false });
    }));

    const build = plugin.indexPersonalLibraryFullText();
    await vi.waitFor(() => expect(source.inventory).toHaveBeenCalledTimes(1));

    await expect(plugin.scanPersonalLibrary()).rejects.toThrow("already active");

    finishInventory();
    await build;
  });

  it("rejects a concurrent build while a manual scan is in flight", async () => {
    const { plugin, source } = fixture([]);
    let finishInventory!: () => void;
    source.inventory.mockImplementationOnce(() => new Promise((resolve) => {
      finishInventory = () => resolve({ entries: [], truncated: false });
    }));

    const scan = plugin.scanPersonalLibrary();
    await vi.waitFor(() => expect(source.inventory).toHaveBeenCalledTimes(1));

    await expect(plugin.indexPersonalLibraryFullText()).rejects.toThrow("already active");

    finishInventory();
    await scan;
  });
});
