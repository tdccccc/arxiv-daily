import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  FullTextGenerationIndexStore,
  GENERATION_DESCRIPTOR_FORMAT_VERSION,
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
  OperationRegistry,
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
  type EmbeddingModel,
  type FullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBaseStore,
  type GenerationDescriptor,
  type StorageAdapter,
} from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";
import { createLibraryConnection } from "../src/library/connection";

function memoryStorage() {
  const text = new Map<string, string>();
  const binary = new Map<string, Uint8Array>();
  const directories = new Set<string>();
  const normalizePath = (path: string) => path
    .replace(/\\/g, "/")
    .replace(/\/+/g, "/")
    .replace(/^\/+|\/+$/g, "");
  const storage: StorageAdapter = {
    normalizePath,
    readText: async (path) => {
      const value = text.get(normalizePath(path));
      if (value === undefined) throw new Error(`missing ${path}`);
      return value;
    },
    writeText: async (path, value) => { text.set(normalizePath(path), value); },
    writeTextAtomic: async (path, value) => { text.set(normalizePath(path), value); },
    createTextExclusive: async (path, value) => {
      const normalized = normalizePath(path);
      if (text.has(normalized) || binary.has(normalized) || directories.has(normalized)) return false;
      text.set(normalized, value);
      return true;
    },
    exists: async (path) => {
      const normalized = normalizePath(path);
      return text.has(normalized) || binary.has(normalized) || directories.has(normalized);
    },
    mkdir: async (path) => { directories.add(normalizePath(path)); },
    remove: async (path) => {
      const normalized = normalizePath(path);
      const prefix = `${normalized}/`;
      for (const key of [...text.keys()]) {
        if (key === normalized || key.startsWith(prefix)) text.delete(key);
      }
      for (const key of [...binary.keys()]) {
        if (key === normalized || key.startsWith(prefix)) binary.delete(key);
      }
      for (const key of [...directories]) {
        if (key === normalized || key.startsWith(prefix)) directories.delete(key);
      }
    },
    list: async (dir) => {
      const normalized = normalizePath(dir);
      if (!directories.has(normalized)) throw new Error(`missing ${dir}`);
      const prefix = `${normalized}/`;
      const entries = new Map<string, "file" | "folder">();
      for (const path of [...text.keys(), ...binary.keys(), ...directories]) {
        if (!path.startsWith(prefix)) continue;
        const remainder = path.slice(prefix.length);
        if (!remainder) continue;
        const child = remainder.split("/")[0]!;
        const childPath = `${normalized}/${child}`;
        entries.set(
          childPath,
          remainder.includes("/") || directories.has(childPath) ? "folder" : "file",
        );
      }
      return [...entries].map(([path, type]) => ({ path, type }));
    },
    rename: async (from, to) => {
      const source = normalizePath(from);
      const value = text.get(source);
      if (value === undefined) throw new Error(`missing ${from}`);
      text.delete(source);
      text.set(normalizePath(to), value);
    },
    readBinary: async (path) => {
      const value = binary.get(normalizePath(path));
      if (value === undefined) throw new Error(`missing ${path}`);
      return value.slice().buffer;
    },
    writeBinary: async (path, value) => {
      binary.set(normalizePath(path), new Uint8Array(value).slice());
    },
  };
  return { storage, text };
}

function limitedStorage(storage: StorageAdapter): StorageAdapter {
  const limited = { ...storage };
  delete limited.createTextExclusive;
  return limited;
}

function emptyGeneration(
  scopeFingerprint: string,
  identificationFingerprint: string,
): GenerationDescriptor {
  return {
    formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION,
    schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION,
    generationId: "gen-plugin-preflight",
    sourceRevision: 7,
    scopeFingerprint,
    identificationFingerprint,
    modelId: "fixture-model",
    dimension: 2,
    corpusMean: [0, 0],
    corpusStats: {
      indexedPaperCount: 0,
      chunkCount: 0,
      totalLexicalTokenCount: 0,
      avgdl: 0,
      totalLexicalTokenCountWithHanSingles: 0,
      avgdlWithHanSingles: 0,
    },
    lexicalCapability: "none",
    lexicalRouting: Array.from({ length: 256 }, () => []),
    indexDerivation: {
      builderVersion: 1,
      denseCenteringVersion: 1,
      tokenizerVersion: 1,
      postingsVersion: 1,
    },
    objects: [],
  };
}

function knowledgeBase(
  scopeFingerprint: string,
  identificationFingerprint: string,
) {
  const initial: FullTextKnowledgeBaseManifest = {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    revision: 0,
    scopeFingerprint,
    identificationFingerprint,
    modelId: "",
    dimension: 0,
    updatedAt: "2026-08-18T00:00:00.000Z",
    papers: {},
  };
  let current = initial;
  const loadManifest = vi.fn(async () => structuredClone(current));
  const replaceManifest = vi.fn(async (next: FullTextKnowledgeBaseManifest) => {
    current = {
      ...structuredClone(next),
      revision: current.revision + 1,
      updatedAt: "2026-08-18T00:01:00.000Z",
    };
    return structuredClone(current);
  });
  const store: FullTextKnowledgeBaseStore = {
    paths: {
      directory: "legacy",
      manifest: {
        directory: "legacy",
        documentPath: "legacy/manifest.json",
        backupPath: "legacy/manifest.json.backup",
      },
      papersDirectory: "legacy/papers",
    },
    loadManifest,
    replaceManifest,
    loadPaper: vi.fn(async () => null),
    savePaper: vi.fn(async () => undefined),
    removePaper: vi.fn(async () => undefined),
    removeAll: vi.fn(async () => undefined),
  };
  return { store, loadManifest, replaceManifest };
}

function fixture(storage: StorageAdapter) {
  const settings = structuredClone(DEFAULT_SETTINGS);
  const connection = createLibraryConnection("/private/library", "1:2");
  const scopeFingerprint = createPersonalLibraryScopeFingerprint({
    rootIdentity: connection.rootIdentity,
    eligibleExtensions: connection.eligibleExtensions,
  });
  const identificationFingerprint = createPersonalLibraryIdentificationFingerprint(
    connection.eligibleExtensions,
  );
  const catalog = createEmptyPersonalLibraryCatalog(
    scopeFingerprint,
    identificationFingerprint,
    new Date("2026-08-18T00:00:00.000Z"),
  );
  const legacy = knowledgeBase(scopeFingerprint, identificationFingerprint);
  const model: EmbeddingModel = {
    modelId: "fixture-model",
    dimension: 2,
    prefixPolicy: "none",
    embed: vi.fn(async (texts) => texts.map(() => new Float32Array([0, 0]))),
  };
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  const internals = plugin as unknown as Record<string, any>;
  Object.assign(plugin, {
    settings,
    logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    host: { storage, http: {}, markupParser: {} },
    progress: {
      setTask: vi.fn(),
      setComplete: vi.fn(),
      setError: vi.fn(),
    },
    operations: new OperationRegistry(),
    libraryConnection: connection,
    librarySource: {
      canonicalRoot: connection.selectedRoot,
      rootIdentity: connection.rootIdentity,
      inventory: vi.fn(async () => ({ entries: [], truncated: false })),
      readBinary: vi.fn(async () => new ArrayBuffer(0)),
    },
    libraryConnectionRevision: 0,
    libraryOutputRevision: 0,
    librarySelectionRevision: 0,
  });
  internals.buildPersonalLibraryCatalogStore = vi.fn(() => ({
    load: vi.fn(async () => structuredClone(catalog)),
  }));
  internals.buildFullTextKnowledgeBaseStore = vi.fn(() => legacy.store);
  internals.buildEmbeddingModel = vi.fn(() => model);
  internals.runIncrementalDirectionUpdateAfterIndex = vi.fn(async () => undefined);
  return {
    plugin,
    internals,
    legacy,
    connection,
    scopeFingerprint,
    identificationFingerprint,
  };
}

describe("personal library full-text index lifecycle", () => {
  it("does not probe a local parser sidecar while it is disabled", async () => {
    const memory = memoryStorage();
    const runtime = fixture(memory.storage);
    const request = vi.fn();
    runtime.internals.host.http = { request };

    const configured = await runtime.internals.buildFullTextDocumentParser();

    expect(configured.parser?.provenance).toEqual({ id: "obsidian-pdfjs", version: "1" });
    expect(configured.parserSelector).toBeUndefined();
    expect(request).not.toHaveBeenCalled();
  });

  it("falls back to PDF.js when an enabled local sidecar cannot be probed", async () => {
    const memory = memoryStorage();
    const runtime = fixture(memory.storage);
    runtime.internals.settings.pdfParserSidecar.enabled = true;
    const failure = new Error("connection refused");
    const request = vi.fn(async () => { throw failure; });
    runtime.internals.host.http = { request };

    const configured = await runtime.internals.buildFullTextDocumentParser();

    expect(request).toHaveBeenCalledWith(expect.objectContaining({
      method: "GET",
      url: "http://127.0.0.1:5001/v1/capabilities",
    }));
    expect(configured.parser?.provenance).toEqual({ id: "obsidian-pdfjs", version: "1" });
    expect(configured.parserSelector).toBeUndefined();
    expect(runtime.internals.logger.warn).toHaveBeenCalledWith(
      "fulltext: local PDF parser sidecar probe failed; using PDF.js",
      expect.anything(),
    );
  });

  it("stops an active full-text index when local sidecar settings change", () => {
    const memory = memoryStorage();
    const runtime = fixture(memory.storage);
    const operation = runtime.internals.operations.begin(
      "personal-library-fulltext-index",
      "Personal library full-text index",
      runtime.scopeFingerprint,
    );

    runtime.internals.preparePdfParserSidecarSettingsChange([
      "pdfParserSidecar.parseUrl",
    ]);

    expect(operation.signal.aborted).toBe(true);
    expect(operation.signal.reason).toBe("local PDF parser sidecar settings changed");
  });

  it("revalidates a current indexed PDF before opening its evidence page", async () => {
    const memory = memoryStorage();
    const runtime = fixture(memory.storage);
    const connection = runtime.connection;
    connection.selectedRoot = "/vault/library";
    const openLinkText = vi.fn(async () => undefined);
    runtime.internals.app = {
      workspace: { openLinkText },
      vault: { adapter: { getBasePath: () => "/vault" } },
    };
    runtime.internals.librarySource = {
      canonicalRoot: connection.selectedRoot,
      rootIdentity: connection.rootIdentity,
      inventory: vi.fn(async () => ({ entries: [], truncated: false })),
      readBinary: vi.fn(async () => new ArrayBuffer(1)),
    };
    runtime.legacy.loadManifest.mockResolvedValue({
      papers: {
        "arxiv:2607.00001": { filePaths: ["papers/evidence.pdf"] },
      },
    });

    await expect(runtime.plugin.openPersonalLibraryFullTextEvidence({
      paperKey: "arxiv:2607.00001",
      filePath: "papers/evidence.pdf",
    })).resolves.toBe("page-targeted");

    expect(runtime.internals.librarySource.readBinary).toHaveBeenCalledWith(
      "papers/evidence.pdf",
      { start: 0, end: 1, maxBytes: Number.MAX_SAFE_INTEGER },
    );
    expect(openLinkText).toHaveBeenCalledWith("library/papers/evidence.pdf", "", false);
  });

  it("opens a vault PDF even when getBasePath must keep its adapter this", async () => {
    const memory = memoryStorage();
    const runtime = fixture(memory.storage);
    const connection = runtime.connection;
    connection.selectedRoot = "/vault/library";
    const openLinkText = vi.fn(async () => undefined);
    runtime.internals.app = {
      workspace: { openLinkText },
      vault: {
        adapter: {
          basePath: "/vault",
          getBasePath() {
            return this.basePath;
          },
        },
      },
    };
    runtime.internals.librarySource = {
      canonicalRoot: connection.selectedRoot,
      rootIdentity: connection.rootIdentity,
      inventory: vi.fn(async () => ({ entries: [], truncated: false })),
      readBinary: vi.fn(async () => new ArrayBuffer(1)),
    };
    runtime.legacy.loadManifest.mockResolvedValue({
      papers: {
        "arxiv:2607.00001": { filePaths: ["papers/evidence.pdf"] },
      },
    });

    await expect(runtime.plugin.openPersonalLibraryFullTextEvidence({
      paperKey: "arxiv:2607.00001",
      filePath: "papers/evidence.pdf",
    })).resolves.toBe("page-targeted");

    expect(openLinkText).toHaveBeenCalledWith("library/papers/evidence.pdf", "", false);
  });

  it("uses one manifest snapshot for full-text search orchestration", async () => {
    const memory = memoryStorage();
    const runtime = fixture(memory.storage);

    await expect(runtime.plugin.searchPersonalLibraryFullText("alpha")).resolves.toEqual([]);

    expect(runtime.legacy.loadManifest).toHaveBeenCalledTimes(1);
  });

  it("rejects an incompatible generation before a capable host touches the legacy manifest", async () => {
    const memory = memoryStorage();
    const runtime = fixture(memory.storage);
    const generations = new FullTextGenerationIndexStore(
      memory.storage,
      runtime.internals.settings.output,
      runtime.scopeFingerprint,
      runtime.identificationFingerprint,
    );
    memory.text.set(generations.paths.currentPath, JSON.stringify({
      formatVersion: 2,
      schemaVersion: 2,
    }));

    await expect(runtime.plugin.indexPersonalLibraryFullText()).rejects.toMatchObject({
      code: "incompatible",
    });

    expect(runtime.legacy.loadManifest).not.toHaveBeenCalled();
    expect(runtime.legacy.replaceManifest).not.toHaveBeenCalled();
  });

  it("fails closed before a legacy manifest read or write when cutover exists but the host lost generation capabilities", async () => {
    const memory = memoryStorage();
    const seeded = fixture(memory.storage);
    const generations = new FullTextGenerationIndexStore(
      memory.storage,
      seeded.internals.settings.output,
      seeded.scopeFingerprint,
      seeded.identificationFingerprint,
    );
    await generations.stageAndPromote({
      descriptor: emptyGeneration(seeded.scopeFingerprint, seeded.identificationFingerprint),
      objects: [],
      writerToken: `writer-plugin-preflight-${"a".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 7,
    });
    expect(memory.text.has(generations.paths.currentPath)).toBe(true);
    expect(memory.text.has(generations.paths.cutoverMarkerPath)).toBe(true);

    const runtime = fixture(limitedStorage(memory.storage));
    await expect(runtime.plugin.indexPersonalLibraryFullText()).rejects.toMatchObject({
      code: "capability-unsupported",
    });

    expect(runtime.legacy.loadManifest).not.toHaveBeenCalled();
    expect(runtime.legacy.replaceManifest).not.toHaveBeenCalled();
    expect(runtime.internals.runIncrementalDirectionUpdateAfterIndex).not.toHaveBeenCalled();
  });

  it("rejects fallback admission when cutover wins after preflight", async () => {
    const memory = memoryStorage();
    const runtime = fixture(limitedStorage(memory.storage));
    const generations = new FullTextGenerationIndexStore(
      memory.storage,
      runtime.internals.settings.output,
      runtime.scopeFingerprint,
      runtime.identificationFingerprint,
    );
    let cutoverEstablished = false;
    runtime.internals.buildFullTextGenerationIndexStore = vi.fn(() => ({
      openCurrent: vi.fn(async () => {
        const observed = await generations.openCurrent();
        if (!cutoverEstablished) {
          cutoverEstablished = true;
          await generations.stageAndPromote({
            descriptor: emptyGeneration(runtime.scopeFingerprint, runtime.identificationFingerprint),
            objects: [],
            writerToken: `writer-preflight-race-${"a".repeat(32)}`,
            expectedCurrent: null,
            sourceCurrentRevision: () => 7,
          });
        }
        return observed;
      }),
      acquireLegacyMigrationLease: (token: string) => generations.acquireLegacyMigrationLease(token),
    }));

    await expect(runtime.plugin.indexPersonalLibraryFullText()).rejects.toMatchObject({
      code: "capability-unsupported",
    });

    expect(runtime.legacy.loadManifest).not.toHaveBeenCalled();
    expect(runtime.legacy.replaceManifest).not.toHaveBeenCalled();
  });

  it("keeps the legacy migration fallback when the unsupported host has never cut over", async () => {
    const memory = memoryStorage();
    const runtime = fixture(limitedStorage(memory.storage));

    await expect(runtime.plugin.indexPersonalLibraryFullText()).resolves.toMatchObject({
      manifestRevision: 1,
      indexed: 0,
      reused: 0,
      failed: 0,
      pruned: 0,
    });

    expect(runtime.legacy.loadManifest).toHaveBeenCalledTimes(1);
    expect(runtime.legacy.replaceManifest).toHaveBeenCalledTimes(1);
    expect(runtime.internals.runIncrementalDirectionUpdateAfterIndex).toHaveBeenCalledTimes(1);
    expect(runtime.internals.logger.warn).toHaveBeenCalledWith(
      "fulltext: immutable generation cutover is unavailable on this host; retaining the legacy migration fallback",
    );
  });

  it("holds and validates a legacy migration lease around the manifest commit", async () => {
    const memory = memoryStorage();
    const runtime = fixture(limitedStorage(memory.storage));
    const lifecycle: string[] = [];
    const assertOwned = vi.fn(async () => undefined);
    const release = vi.fn(async () => { lifecycle.push("release"); });
    const acquireLegacyMigrationLease = vi.fn(async () => ({ assertOwned, release }));
    runtime.internals.runIncrementalDirectionUpdateAfterIndex = vi.fn(async () => {
      lifecycle.push("direction");
    });
    runtime.internals.buildFullTextGenerationIndexStore = vi.fn(() => ({
      openCurrent: vi.fn(async () => null),
      acquireLegacyMigrationLease,
    }));

    await expect(runtime.plugin.indexPersonalLibraryFullText()).resolves.toMatchObject({
      manifestRevision: 1,
    });

    expect(acquireLegacyMigrationLease).toHaveBeenCalledWith(
      expect.stringMatching(/^writer-[a-f0-9]{32}$/),
    );
    expect(assertOwned).toHaveBeenCalledTimes(2);
    expect(release).toHaveBeenCalledTimes(1);
    expect(lifecycle).toEqual(["release", "direction"]);
  });

  it("preserves the indexing error when legacy lease release also fails", async () => {
    const memory = memoryStorage();
    const runtime = fixture(limitedStorage(memory.storage));
    const manifestFailure = new Error("manifest commit failed");
    const releaseFailure = new Error("legacy lease release failed");
    const release = vi.fn(async () => { throw releaseFailure; });
    runtime.legacy.replaceManifest.mockRejectedValueOnce(manifestFailure);
    runtime.internals.buildFullTextGenerationIndexStore = vi.fn(() => ({
      openCurrent: vi.fn(async () => null),
      acquireLegacyMigrationLease: vi.fn(async () => ({
        assertOwned: vi.fn(async () => undefined),
        release,
      })),
    }));

    await expect(runtime.plugin.indexPersonalLibraryFullText()).rejects.toBe(manifestFailure);

    expect(release).toHaveBeenCalledTimes(1);
    expect(runtime.internals.logger.warn).toHaveBeenCalledWith(
      "fulltext: failed to release the legacy migration lease after indexing failed",
      releaseFailure,
    );
  });

  it("fails the command when a successful fallback cannot release its lease", async () => {
    const memory = memoryStorage();
    const runtime = fixture(limitedStorage(memory.storage));
    const releaseFailure = new Error("legacy lease release uncertain");
    runtime.internals.buildFullTextGenerationIndexStore = vi.fn(() => ({
      openCurrent: vi.fn(async () => null),
      acquireLegacyMigrationLease: vi.fn(async () => ({
        assertOwned: vi.fn(async () => undefined),
        release: vi.fn(async () => { throw releaseFailure; }),
      })),
    }));

    await expect(runtime.plugin.indexPersonalLibraryFullText()).rejects.toBe(releaseFailure);

    expect(runtime.internals.progress.setComplete).not.toHaveBeenCalled();
    expect(runtime.internals.progress.setError).toHaveBeenCalledWith(
      "Personal library full-text indexing failed",
    );
  });

  it("blocks first cutover on a shared backing store while fallback commits", async () => {
    const memory = memoryStorage();
    const runtime = fixture(limitedStorage(memory.storage));
    const generations = new FullTextGenerationIndexStore(
      memory.storage,
      runtime.internals.settings.output,
      runtime.scopeFingerprint,
      runtime.identificationFingerprint,
    );
    runtime.legacy.replaceManifest.mockImplementationOnce(async (next) => {
      await expect(generations.stageAndPromote({
        descriptor: emptyGeneration(runtime.scopeFingerprint, runtime.identificationFingerprint),
        objects: [],
        writerToken: `writer-cutover-race-${"d".repeat(32)}`,
        expectedCurrent: null,
        sourceCurrentRevision: () => 7,
      })).rejects.toMatchObject({ code: "concurrent" });
      return {
        ...structuredClone(next),
        revision: 1,
        updatedAt: "2026-08-18T00:01:00.000Z",
      };
    });

    await expect(runtime.plugin.indexPersonalLibraryFullText()).resolves.toMatchObject({
      manifestRevision: 1,
    });
    expect(memory.text.has(generations.paths.cutoverMarkerPath)).toBe(false);

    await expect(generations.stageAndPromote({
      descriptor: emptyGeneration(runtime.scopeFingerprint, runtime.identificationFingerprint),
      objects: [],
      writerToken: `writer-cutover-after-${"e".repeat(32)}`,
      expectedCurrent: null,
      sourceCurrentRevision: () => 7,
    })).resolves.toMatchObject({ descriptor: { generationId: "gen-plugin-preflight" } });
  });

  it("preserves the post-commit direction update when generation synchronization fails", async () => {
    const memory = memoryStorage();
    const runtime = fixture(memory.storage);
    const stageAndPromote = vi.fn(async () => { throw new Error("generation promotion failed"); });
    runtime.internals.buildFullTextGenerationIndexStore = vi.fn(() => ({
      openCurrent: vi.fn(async () => null),
      stageAndPromote,
    }));

    await expect(runtime.plugin.indexPersonalLibraryFullText())
      .rejects.toThrow("generation promotion failed");

    expect(runtime.legacy.replaceManifest).toHaveBeenCalledTimes(1);
    expect(runtime.internals.runIncrementalDirectionUpdateAfterIndex).toHaveBeenCalledTimes(1);
    expect(stageAndPromote.mock.invocationCallOrder[0]).toBeLessThan(
      runtime.internals.runIncrementalDirectionUpdateAfterIndex.mock.invocationCallOrder[0],
    );
  });

  it("runs generation maintenance only through an explicit host quiet-period gate", async () => {
    const memory = memoryStorage();
    const runtime = fixture(memory.storage);
    runtime.internals.scheduler = {
      activeRuns: vi.fn(() => []),
      stop: vi.fn(),
      start: vi.fn(),
    };
    const generations = new FullTextGenerationIndexStore(
      memory.storage,
      runtime.internals.settings.output,
      runtime.scopeFingerprint,
      runtime.identificationFingerprint,
    );
    runtime.internals.buildFullTextGenerationIndexStore = vi.fn(() => generations);
    runtime.internals.operations.beginFullTextMaintenanceTransition = vi.fn(() => vi.fn());

    await expect(runtime.plugin.maintainPersonalLibraryFullTextGenerations()).resolves.toMatchObject({
      promotionClaim: "absent",
      removedGenerationIds: [],
    });
    expect(runtime.internals.operations.beginFullTextMaintenanceTransition).toHaveBeenCalledTimes(1);
  });
});
