import { describe, expect, it, vi } from "vitest";
import { deflateSync } from "node:zlib";
import {
  DEFAULT_SETTINGS,
  OperationRegistry,
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
  derivePersonalLibraryCatalogPaths,
  type StorageAdapter,
} from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";
import { createLibraryConnection } from "../src/library/connection";

function makeStorage(initial: Record<string, string> = {}) {
  const files = new Map(Object.entries(initial));
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
    writeTextAtomic: vi.fn(async (path, content) => { files.set(path, content); }),
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

function metadata(arxivId: string) {
  return {
    id: arxivId,
    title: `Paper ${arxivId}`,
    authorNames: ["Ada Researcher"],
    abstract: "Abstract",
    published: "2026-01-01T00:00:00.000Z",
    updated: "2026-01-02T00:00:00.000Z",
    primaryCategory: "cs.LG",
    categories: ["cs.LG"],
  };
}

function bytesOf(parts: Array<string | Uint8Array>): Uint8Array {
  const chunks = parts.map((part) =>
    typeof part === "string" ? new TextEncoder().encode(part) : part);
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function identificationPdf(streamText: string, title?: string): Uint8Array {
  const compressed = deflateSync(streamText);
  return bytesOf([
    "%PDF-1.4\n",
    `1 0 obj\n<< /Length ${compressed.length} /Filter /FlateDecode >>\nstream\n`,
    compressed,
    "\nendstream\nendobj\n",
    title ? `2 0 obj\n<< /Title (${title}) >>\nendobj\n` : "",
    "trailer\n<< /Info 2 0 R >>\n%%EOF\n",
  ]);
}

function makePlugin(storage = makeStorage().storage) {
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  const operations = new OperationRegistry();
  const fetchMetadataByIds = vi.fn(async (ids: string[]) =>
    new Map(ids.map((id) => [id, metadata(id)])),
  );
  const source = {
    canonicalRoot: "/private/library",
    rootIdentity: "1:2",
    inventory: vi.fn().mockResolvedValue({
      entries: [
        { path: "papers/2601.01234.pdf", type: "file", size: 100, mtimeMs: 10 },
        { path: "papers/notes.pdf", type: "file", size: 20, mtimeMs: 11 },
        { path: "draft.md", type: "file", size: 5, mtimeMs: 12 },
      ],
      truncated: false,
    }),
    readBinary: vi.fn(async (_path: string, _options?: object): Promise<ArrayBuffer> => {
      throw new Error("must not read PDF bytes");
    }),
  };
  Object.assign(plugin, {
    settings: structuredClone(DEFAULT_SETTINGS),
    logger: { warn: vi.fn(), error: vi.fn(), setSensitiveValues: vi.fn() },
    host: { storage, http: {}, markupParser: {} },
    progress: {
      setTask: vi.fn(),
      setComplete: vi.fn(),
      setError: vi.fn(),
      setIdle: vi.fn(),
      setDisabled: vi.fn(),
    },
    operations,
    libraryConnection: createLibraryConnection("/private/library", "1:2"),
    librarySource: source,
    libraryConnectionRevision: 0,
    libraryOutputRevision: 0,
    librarySelectionRevision: 0,
    libraryMutationQueue: Promise.resolve(),
    buildArxivFetcher: vi.fn(() => ({ fetchMetadataByIds })),
  });
  return {
    plugin,
    internals: plugin as unknown as Record<string, any>,
    source,
    fetchMetadataByIds,
    storage,
  };
}

describe("personal library scan lifecycle", () => {
  it("scans without model authorization, resolves only canonical IDs, and atomically reloads", async () => {
    const { plugin, internals, source, fetchMetadataByIds, storage } = makePlugin();
    plugin.settings.llm.apiKey = "";
    plugin.settings.llm.model = "";

    const scanned = await plugin.scanPersonalLibrary();

    expect(scanned.revision).toBe(1);
    expect(scanned.lastScan).toEqual({
      ready: 1,
      papers: 1,
      unresolved: 1,
      unrelated: 1,
      failed: 0,
      truncated: false,
    });
    expect(fetchMetadataByIds).toHaveBeenCalledWith(["2601.01234"], expect.any(AbortSignal));
    // Identification v2 attempts PDF-evidence identification for unrecognized
    // filenames; a read failure keeps the file unresolved instead of failing.
    expect(source.readBinary).toHaveBeenCalledWith("papers/notes.pdf", expect.any(Object));
    expect(storage.writeTextAtomic).toHaveBeenCalled();
    expect(internals.buildArxivFetcher).toHaveBeenCalledTimes(1);

    internals.libraryCatalog = null;
    await expect(plugin.reloadPersonalLibraryCatalog()).resolves.toEqual(scanned);
  });

  it("continues title search when PDF evidence contains an unsupported legacy arXiv ID", async () => {
    const { plugin, internals, source, fetchMetadataByIds } = makePlugin();
    const title = "Legacy Evidence Replacement Paper";
    const pdf = identificationPdf(
      "(arXiv:astro-ph/0609591 [astro-ph] 27 Sep 2006) Tj",
      title,
    );
    source.inventory.mockResolvedValue({
      entries: [{ path: "papers/legacy.pdf", type: "file", size: pdf.byteLength, mtimeMs: 10 }],
      truncated: false,
    });
    source.readBinary.mockResolvedValue(pdf.buffer);
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: `<feed><entry><id>https://arxiv.org/abs/2403.19236v1</id><title>${title}</title></entry></feed>`,
    }));
    internals.host.http = { request };

    const scanned = await plugin.scanPersonalLibrary();

    expect(request).toHaveBeenCalledWith(expect.objectContaining({
      url: expect.stringContaining("search_query=ti:"),
      signal: expect.any(AbortSignal),
    }));
    expect(fetchMetadataByIds).toHaveBeenCalledWith(["2403.19236"], expect.any(AbortSignal));
    expect(scanned.files["papers/legacy.pdf"]).toMatchObject({
      status: "ready",
      paperKey: "arxiv:2403.19236",
      arxivId: "2403.19236",
    });
    expect(scanned.papers["arxiv:2403.19236"]).toBeDefined();
  });

  it("keeps a legacy-only PDF unresolved without aborting the scan", async () => {
    const { plugin, internals, source, fetchMetadataByIds } = makePlugin();
    const pdf = identificationPdf("(arXiv:astro-ph/0609591 [astro-ph] 27 Sep 2006) Tj");
    source.inventory.mockResolvedValue({
      entries: [{ path: "papers/legacy.pdf", type: "file", size: pdf.byteLength, mtimeMs: 10 }],
      truncated: false,
    });
    source.readBinary.mockResolvedValue(pdf.buffer);
    const request = vi.fn();
    internals.host.http = { request };

    const scanned = await plugin.scanPersonalLibrary();

    expect(request).not.toHaveBeenCalled();
    expect(fetchMetadataByIds).not.toHaveBeenCalled();
    expect(scanned.lastScan).toEqual({
      ready: 0,
      papers: 0,
      unresolved: 1,
      unrelated: 0,
      failed: 0,
      truncated: false,
    });
    expect(scanned.files["papers/legacy.pdf"]).toMatchObject({ status: "unresolved" });
  });

  it("invalidates captured personalized discovery before catalog promotion and reload installs", async () => {
    const { plugin, internals, storage, source } = makePlugin();
    const controllers = new Map<object, AbortController>();
    const promoted = new AbortController();
    controllers.set({}, promoted);
    internals.personalizedDailyRunControllers = controllers;
    let releaseWrite!: () => void;
    vi.mocked(storage.writeTextAtomic!).mockImplementationOnce(async () => {
      await new Promise<void>((resolve) => { releaseWrite = resolve; });
    });

    const scan = plugin.scanPersonalLibrary();
    await vi.waitFor(() => expect(storage.writeTextAtomic).toHaveBeenCalled());
    expect(promoted.signal.aborted).toBe(true);
    releaseWrite();
    await scan;

    const reloaded = new AbortController();
    controllers.set({}, reloaded);
    const reload = plugin.reloadPersonalLibraryCatalog();
    expect(reloaded.signal.aborted).toBe(true);
    await reload;
    // A scan may read PDF evidence for unrecognized files; the reload itself
    // must not re-read PDFs (unrecognized files reuse their observations).
    expect(source.readBinary).toHaveBeenCalledTimes(1);
  });

  it("rejects a duplicate scan operation", async () => {
    const { plugin, source } = makePlugin();
    let finishInventory!: () => void;
    source.inventory.mockImplementationOnce(() => new Promise((resolve) => {
      finishInventory = () => resolve({ entries: [], truncated: false });
    }));

    const first = plugin.scanPersonalLibrary();
    await vi.waitFor(() => expect(source.inventory).toHaveBeenCalledTimes(1));
    await expect(plugin.scanPersonalLibrary()).rejects.toThrow("already active");
    finishInventory();
    await first;
  });

  it("cancels only active scans when the selected root changes", async () => {
    const { plugin, internals, source } = makePlugin();
    const unrelated = plugin.operations.begin("detail-summary", "detail", "2601.00001");
    source.inventory.mockImplementationOnce(({ signal }: { signal?: AbortSignal } = {}) =>
      new Promise((_resolve, reject) => signal?.addEventListener("abort", () => reject(signal.reason), { once: true })),
    );
    internals.libraryDirectoryPicker = {
      select: vi.fn().mockResolvedValue({ kind: "selected", path: "/next" }),
    };
    internals.openLibrarySource = vi.fn().mockResolvedValue({
      canonicalRoot: "/next",
      rootIdentity: "1:3",
      inventory: vi.fn(),
      readBinary: vi.fn(),
    });
    internals.saveData = vi.fn().mockResolvedValue(undefined);

    const scan = plugin.scanPersonalLibrary();
    await vi.waitFor(() => expect(source.inventory).toHaveBeenCalled());
    await expect(plugin.selectLibraryRoot()).resolves.toBe("selected");
    await expect(scan).rejects.toBe("library folder changed");
    expect(plugin.operations.find("detail-summary", "2601.00001")?.cancellationRequested).toBe(false);
    unrelated.finish();
  });

  it("returns null while disconnected and surfaces corrupt durable catalogs", async () => {
    const disconnected = makePlugin();
    disconnected.internals.libraryConnection = undefined;
    await expect(disconnected.plugin.reloadPersonalLibraryCatalog()).resolves.toBeNull();

    const { documentPath } = derivePersonalLibraryCatalogPaths(
      disconnected.storage,
      disconnected.plugin.settings.output,
    );
    const corruptStorage = makeStorage({ [documentPath]: "{broken" }).storage;
    const connected = makePlugin(corruptStorage);
    await expect(connected.plugin.reloadPersonalLibraryCatalog()).rejects.toThrow(
      "cannot load unreadable personal library catalog",
    );
    expect(connected.plugin.getPersonalLibraryCatalog()).toBeNull();
  });

  it("restores the connection revision when selecting a new root fails to persist", async () => {
    const { plugin, internals } = makePlugin();
    const previousConnection = internals.libraryConnection;
    internals.libraryConnectionRevision = 7;
    internals.libraryDirectoryPicker = {
      select: vi.fn().mockResolvedValue({ kind: "selected", path: "/next" }),
    };
    internals.openLibrarySource = vi.fn().mockResolvedValue({
      canonicalRoot: "/next",
      rootIdentity: "1:3",
      inventory: vi.fn(),
      readBinary: vi.fn(),
    });
    internals.saveData = vi.fn().mockRejectedValue(new Error("disk full"));

    await expect(plugin.selectLibraryRoot()).rejects.toThrow("disk full");

    expect(internals.libraryConnection).toBe(previousConnection);
    expect(internals.libraryConnectionRevision).toBe(7);
  });

  it("treats cancellation during the final atomic write as committed after the write succeeds", async () => {
    const { plugin, storage, files } = (() => {
      const fixture = makeStorage();
      return { ...makePlugin(fixture.storage), files: fixture.files };
    })();
    const { documentPath } = derivePersonalLibraryCatalogPaths(
      storage,
      plugin.settings.output,
    );
    const atomicWrite = vi.mocked(storage.writeTextAtomic!);
    let releaseWrite!: () => void;
    atomicWrite.mockImplementationOnce(async (path, content) => {
      await new Promise<void>((resolve) => { releaseWrite = resolve; });
      files.set(path, content);
    });

    const scan = plugin.scanPersonalLibrary();
    await vi.waitFor(() => expect(atomicWrite).toHaveBeenCalledWith(
      documentPath,
      expect.any(String),
    ));
    const active = plugin.operations.snapshot().find((item) =>
      item.kind === "personal-library-scan"
    )!;
    plugin.operations.cancel(active.id, "cancelled during final write");
    releaseWrite();

    const committed = await scan;
    expect(committed.revision).toBe(1);
    expect(plugin.getPersonalLibraryCatalog()).toEqual(committed);
    expect(JSON.parse(files.get(documentPath)!)).toEqual(committed);
  });

  it("cancels a scan queued for commit before output reload and keeps the reloaded catalog", async () => {
    const fixture = makeStorage();
    const { plugin, internals, storage } = makePlugin(fixture.storage);
    const oldOutput = structuredClone(plugin.settings.output);
    const nextOutput = {
      ...structuredClone(plugin.settings.output),
      dailyDir: "next/daily",
      papersDir: "next/papers",
    };
    const connection = internals.libraryConnection as ReturnType<typeof createLibraryConnection>;
    const scopeFingerprint = createPersonalLibraryScopeFingerprint({
      rootIdentity: connection.rootIdentity,
      eligibleExtensions: connection.eligibleExtensions,
    });
    const identificationFingerprint = createPersonalLibraryIdentificationFingerprint(
      connection.eligibleExtensions,
    );
    const reloaded = createEmptyPersonalLibraryCatalog(
      scopeFingerprint,
      identificationFingerprint,
      new Date("2026-08-03T00:00:00.000Z"),
    );
    reloaded.revision = 9;
    const nextPath = derivePersonalLibraryCatalogPaths(storage, nextOutput).documentPath;
    fixture.files.set(nextPath, `${JSON.stringify(reloaded)}\n`);

    let releaseMutation!: () => void;
    internals.libraryMutationQueue = new Promise<void>((resolve) => {
      releaseMutation = resolve;
    });
    internals.scheduler = {
      replaceStore: vi.fn(),
      replaceRunHistory: vi.fn(),
    };
    plugin.settings.output = nextOutput;

    const scan = plugin.scanPersonalLibrary();
    await vi.waitFor(() => expect(
      plugin.operations.snapshot().some((item) => item.kind === "personal-library-scan"),
    ).toBe(true));
    const reload = plugin.reloadStateStoreForOutputPaths();
    await vi.waitFor(() => expect(
      plugin.operations.snapshot().find((item) => item.kind === "personal-library-scan")
        ?.cancellationRequested,
    ).toBe(true));
    releaseMutation();

    await expect(scan).rejects.toBe("output paths changed");
    await expect(reload).resolves.toBeUndefined();
    expect(plugin.getPersonalLibraryCatalog()).toEqual(reloaded);
    const oldPath = derivePersonalLibraryCatalogPaths(storage, oldOutput).documentPath;
    expect(fixture.files.has(oldPath)).toBe(false);
    expect(JSON.parse(fixture.files.get(nextPath)!)).toEqual(reloaded);
  });
});

  it("trusts a title search that disagrees with a direct arXiv ID (reference-list misidentification)", async () => {
    // Chen2025-style: the reference list ("… arXiv:0912.0201 …" = LSST
    // Science Book) sits in a content-stream header; the document title
    // search resolves to the real paper, which must win.
    const { plugin, internals, source, fetchMetadataByIds } = makePlugin();
    const title = "LSTM-MDNz: Estimating Quasar Photometric Redshifts with an LSTM-Augmented Mixture Density Network";
    const pdf = identificationPdf(
      "(arXiv:0912.0201 [astro-ph.IM] 1 Dec 2009) Tj",
      title,
    );
    source.inventory.mockResolvedValue({
      entries: [{ path: "papers/chen.pdf", type: "file", size: pdf.byteLength, mtimeMs: 10 }],
      truncated: false,
    });
    source.readBinary.mockResolvedValue(pdf.buffer);
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: `<feed><entry><id>https://arxiv.org/abs/2512.16010v1</id><title>${title}</title></entry></feed>`,
    }));
    internals.host.http = { request };

    const scanned = await plugin.scanPersonalLibrary();

    expect(request).toHaveBeenCalledWith(expect.objectContaining({
      url: expect.stringContaining("search_query=ti:"),
      signal: expect.any(AbortSignal),
    }));
    expect(fetchMetadataByIds).toHaveBeenCalledWith(["2512.16010"], expect.any(AbortSignal));
    expect(scanned.files["papers/chen.pdf"]).toMatchObject({
      status: "ready",
      paperKey: "arxiv:2512.16010",
      arxivId: "2512.16010",
    });
  });

  it("keeps a direct arXiv ID when the title search agrees or finds nothing", async () => {
    const { plugin, internals, source, fetchMetadataByIds } = makePlugin();
    const title = "The Cluster Mass Calibration Project";
    const pdf = identificationPdf(
      "(arXiv:2302.05010v2 [astro-ph.CO] 10 Feb 2023) Tj",
      title,
    );
    source.inventory.mockResolvedValue({
      entries: [{ path: "papers/cluster.pdf", type: "file", size: pdf.byteLength, mtimeMs: 10 }],
      truncated: false,
    });
    source.readBinary.mockResolvedValue(pdf.buffer);
    // Title search returns the same paper: the direct ID stands.
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: `<feed><entry><id>https://arxiv.org/abs/2302.05010v2</id><title>${title}</title></entry></feed>`,
    }));
    internals.host.http = { request };

    const scanned = await plugin.scanPersonalLibrary();

    expect(request).toHaveBeenCalled();
    expect(scanned.files["papers/cluster.pdf"]).toMatchObject({
      status: "ready",
      paperKey: "arxiv:2302.05010",
      arxivId: "2302.05010",
    });

    // A failed title search (garbage document title) must not demote the
    // real paper: the direct ID stands.
    const { plugin: plugin2, internals: internals2, source: source2, fetchMetadataByIds: fetch2 } = makePlugin();
    const pdf2 = identificationPdf(
      "(arXiv:1601.00621v1 [astro-ph.CO] 5 Jan 2016) Tj",
      "Graphics produced by IDL",
    );
    source2.inventory.mockResolvedValue({
      entries: [{ path: "papers/redmapper.pdf", type: "file", size: pdf2.byteLength, mtimeMs: 10 }],
      truncated: false,
    });
    source2.readBinary.mockResolvedValue(pdf2.buffer);
    const request2 = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: "<feed></feed>",
    }));
    internals2.host.http = { request: request2 };

    const scanned2 = await plugin2.scanPersonalLibrary();

    expect(request2).toHaveBeenCalled();
    expect(scanned2.files["papers/redmapper.pdf"]).toMatchObject({
      status: "ready",
      paperKey: "arxiv:1601.00621",
      arxivId: "1601.00621",
    });
  });
