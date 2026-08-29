import { describe, expect, it, vi } from "vitest";
import ArxivDailyPlugin from "../main.ts";
import {
  DEFAULT_SETTINGS,
  ReadingCandidatesStore,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
  type StorageAdapter,
  type ReadingCandidateRowSnapshot,
} from "@arxiv-daily/core";
import type { PersistedLibraryConnection } from "../src/library/connection";

const connection: PersistedLibraryConnection = {
  schemaVersion: 1,
  selectedRoot: "/home/user/library",
  rootIdentity: "dev:ino:1",
  eligibleExtensions: [".pdf"],
  processingDepth: "metadata-and-abstract",
};

function snapshot(index: number, overrides: Partial<ReadingCandidateRowSnapshot> = {}): ReadingCandidateRowSnapshot {
  return {
    paperKey: `arxiv:2608.${String(index).padStart(5, "0")}`,
    arxivId: `2608.${String(index).padStart(5, "0")}`,
    title: `New paper ${index}`,
    authors: "A. Author",
    topic: "astrophysics",
    occurrenceProvenance: {
      reportPath: "arxiv-daily/daily/2026-08-12.md",
      reportDate: "2026-08-12",
      source: "library",
      manualTopics: [],
      directions: [{
        id: "direction-1",
        name: "Cosmology",
        representatives: [{ paperKey: "arxiv:2305.00001", title: "Prior survey" }],
      }],
    },
    ...overrides,
  };
}

function makeStorage() {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const storage: StorageAdapter = {
    normalizePath: (path: string) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""),
    readText: async (path) => {
      if (!(path in files)) throw new Error(`unreadable ${path}`);
      return files[path]!;
    },
    writeText: async (path, content) => { files[path] = content; },
    writeTextAtomic: async (path, content) => { files[path] = content; },
    exists: async (path) => path in files || dirs.has(path),
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => { delete files[path]; dirs.delete(path); },
    rename: async (from, to) => { files[to] = files[from]!; delete files[from]; },
  };
  return { files, storage };
}

function makePlugin() {
  const { files, storage } = makeStorage();
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  Object.assign(plugin, {
    settings: structuredClone(DEFAULT_SETTINGS),
    host: { storage },
    logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), notice: vi.fn() },
    libraryConnection: connection,
  });
  return { plugin, files, storage };
}

function persistedPath(files: Record<string, string>): string {
  const path = Object.keys(files).find((key) => key.endsWith("reading-candidates.json"));
  if (!path) throw new Error("reading candidates document was not persisted");
  return path;
}

describe("reading candidates plugin lifecycle", () => {
  it("saves a dashboard row snapshot through the durable store", async () => {
    const { plugin, files } = makePlugin();
    await expect(plugin.saveReadingCandidateForRow(snapshot(1))).resolves.toBe("saved");
    expect(files[persistedPath(files)]).toContain("New paper 1");
    expect(plugin.getReadingCandidates()?.candidates["arxiv:2608.00001"]?.source.kind).toBe("library");
  });

  it("refuses rows without provenance and misses without a connection", async () => {
    const { plugin } = makePlugin();
    await expect(plugin.saveReadingCandidateForRow({
      ...snapshot(1),
      occurrenceProvenance: undefined,
    })).resolves.toBe("missing-source");

    const disconnected = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
    Object.assign(disconnected, {
      settings: structuredClone(DEFAULT_SETTINGS),
      logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), notice: vi.fn() },
      libraryConnection: null,
    });
    await expect(disconnected.saveReadingCandidateForRow(snapshot(1))).resolves.toBe("unavailable");
  });

  it("persists decisions durably and removes candidates", async () => {
    const { plugin, files, storage } = makePlugin();
    await plugin.saveReadingCandidateForRow(snapshot(1));
    await expect(plugin.decideReadingCandidateForReview("arxiv:2608.00001", "skim", "maybe")).resolves.toBe(true);
    expect(plugin.getReadingCandidates()?.candidates["arxiv:2608.00001"]?.decision?.kind).toBe("skim");

    // A fresh store over the same adapter reads the decision back from disk.
    const fresh = new ReadingCandidatesStore(
      storage,
      DEFAULT_SETTINGS.output,
      createPersonalLibraryScopeFingerprint({
        rootIdentity: connection.rootIdentity,
        eligibleExtensions: connection.eligibleExtensions,
      }),
      createPersonalLibraryIdentificationFingerprint(connection.eligibleExtensions),
    );
    const reloaded = await fresh.load();
    expect(reloaded.candidates["arxiv:2608.00001"]?.decision).toMatchObject({
      kind: "skim",
      note: "maybe",
    });

    await expect(plugin.removeReadingCandidateForReview("arxiv:2608.00001")).resolves.toBe(true);
    expect(plugin.getReadingCandidates()?.candidates["arxiv:2608.00001"]).toBeUndefined();
    expect(files[persistedPath(files)]).not.toContain("New paper 1");
  });
});
