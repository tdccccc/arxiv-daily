import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import {
  PERSONAL_LIBRARY_CATALOG_SCHEMA_VERSION,
  PersonalLibraryCatalogStore,
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
  decodePersonalLibraryCatalog,
  derivePersonalLibraryCatalogPaths,
  type PersonalLibraryCatalog,
} from "../src/library/personal-library-catalog";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

const documentPath = "arxiv-daily/.index/personal-library-catalog.json";
const backupPath = `${documentPath}.backup`;
const scopeFingerprint = createPersonalLibraryScopeFingerprint({
  rootIdentity: "42:1001",
  eligibleExtensions: [".PDF", ".pdf"],
});
const identificationFingerprint = createPersonalLibraryIdentificationFingerprint([".pdf"]);
const firstNow = new Date("2026-08-03T12:00:00.000Z");
const secondNow = new Date("2026-08-03T13:00:00.000Z");

function makeStorage(options: { rejectExistingRenameTarget?: boolean } = {}) {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const rename = vi.fn(async (from: string, to: string) => {
    if (!(from in files)) throw new Error(`missing ${from}`);
    if (options.rejectExistingRenameTarget && (to in files || dirs.has(to))) {
      throw new Error(`destination exists: ${to}`);
    }
    files[to] = files[from]!;
    delete files[from];
  });
  const writeTextAtomic = vi.fn(async (path: string, content: string) => {
    files[path] = content;
  });
  const readText = vi.fn(async (path: string) => {
    if (!(path in files)) throw new Error(`missing ${path}`);
    return files[path]!;
  });
  const exists = vi.fn(async (path: string) => path in files || dirs.has(path));
  const storage: StorageAdapter = {
    normalizePath: (path) => path
      .replace(/\\/g, "/")
      .replace(/\/+/g, "/")
      .replace(/^\/+|\/+$/g, ""),
    readText,
    writeText: async (path, content) => { files[path] = content; },
    writeTextAtomic,
    exists,
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => { delete files[path]; dirs.delete(path); },
    rename,
  };
  return { files, dirs, storage, exists, readText, rename, writeTextAtomic };
}

function populatedCatalog(now = firstNow): PersonalLibraryCatalog {
  return {
    ...createEmptyPersonalLibraryCatalog(scopeFingerprint, identificationFingerprint, now),
    lastScan: {
      ready: 1,
      unresolved: 1,
      unrelated: 1,
      failed: 1,
      papers: 1,
      truncated: false,
    },
    files: {
      "papers/2608.00001v2.pdf": {
        path: "papers/2608.00001v2.pdf",
        status: "ready",
        observationFingerprint: `sha256:${"1".repeat(64)}`,
        paperKey: "arxiv:2608.00001",
        arxivId: "2608.00001",
        updatedAt: now.toISOString(),
      },
      "papers/unknown.pdf": {
        path: "papers/unknown.pdf",
        status: "unresolved",
        observationFingerprint: `sha256:${"2".repeat(64)}`,
        reason: "unrecognized-filename",
        updatedAt: now.toISOString(),
      },
      "notes/draft.md": {
        path: "notes/draft.md",
        status: "unrelated",
        observationFingerprint: `sha256:${"3".repeat(64)}`,
        reason: "unsupported-file-type",
        updatedAt: now.toISOString(),
      },
      "papers/2608.00002.pdf": {
        path: "papers/2608.00002.pdf",
        status: "failed",
        observationFingerprint: `sha256:${"4".repeat(64)}`,
        reason: "metadata-fetch-failed",
        arxivId: "2608.00002",
        updatedAt: now.toISOString(),
      },
    },
    papers: {
      "arxiv:2608.00001": {
        paperKey: "arxiv:2608.00001",
        source: "arxiv",
        externalId: "2608.00001",
        title: "A useful paper",
        authors: ["A. Author", "B. Author"],
        abstract: "An abstract.",
        published: "2026-08-01T00:00:00.000Z",
        updated: "2026-08-02T00:00:00.000Z",
        primaryCategory: "cs.AI",
        categories: ["cs.AI"],
        evidenceDepth: "metadata-and-abstract",
        filePaths: ["papers/2608.00001v2.pdf"],
      },
    },
  };
}

function makeStore(storage: StorageAdapter, now = () => secondNow, onWarning = vi.fn()) {
  return new PersonalLibraryCatalogStore(storage, DEFAULT_SETTINGS.output, { now, onWarning });
}

describe("personal library catalog schema", () => {
  it("derives a separate path under the configured index root", () => {
    const { storage } = makeStorage();
    expect(derivePersonalLibraryCatalogPaths(storage, DEFAULT_SETTINGS.output)).toEqual({
      directory: "arxiv-daily/.index",
      documentPath,
      backupPath,
    });
  });

  it("creates stable scope and identification fingerprints", () => {
    expect(createPersonalLibraryScopeFingerprint({
      rootIdentity: "42:1001",
      eligibleExtensions: [".pdf"],
    })).toBe(scopeFingerprint);
    expect(createPersonalLibraryIdentificationFingerprint([".PDF", ".pdf"]))
      .toBe(identificationFingerprint);
    expect(() => createPersonalLibraryIdentificationFingerprint(["pdf"]))
      .toThrow(/invalid eligible extension/);
  });

  it("strictly decodes a complete catalog", () => {
    const catalog = populatedCatalog();
    expect(decodePersonalLibraryCatalog(catalog)).toEqual(catalog);
    expect(decodePersonalLibraryCatalog({ ...catalog, schemaVersion: 2 })).toBeNull();
    expect(decodePersonalLibraryCatalog({ ...catalog, unexpected: true })).toBeNull();
  });

  it("rejects unsafe paths, noncanonical IDs, and broken file-paper membership", () => {
    const unsafe = populatedCatalog();
    unsafe.files["../escape.pdf"] = {
      ...unsafe.files["papers/unknown.pdf"]!,
      path: "../escape.pdf",
    };
    expect(decodePersonalLibraryCatalog(unsafe)).toBeNull();

    const versioned = populatedCatalog();
    const ready = versioned.files["papers/2608.00001v2.pdf"]!;
    if (ready.status === "ready") ready.arxivId = "2608.00001v2";
    expect(decodePersonalLibraryCatalog(versioned)).toBeNull();

    const detached = populatedCatalog();
    detached.papers["arxiv:2608.00001"]!.filePaths = ["papers/unknown.pdf"];
    expect(decodePersonalLibraryCatalog(detached)).toBeNull();
  });

  it("preserves prototype-sensitive logical paths as ordinary record keys", () => {
    const catalog = populatedCatalog();
    const raw = JSON.parse(JSON.stringify(catalog)) as Record<string, any>;
    const prototypeRecord = {
      path: "__proto__",
      status: "unresolved",
      observationFingerprint: `sha256:${"5".repeat(64)}`,
      reason: "unrecognized-filename",
      updatedAt: firstNow.toISOString(),
    };
    Object.defineProperty(raw.files, "__proto__", {
      value: prototypeRecord,
      enumerable: true,
      configurable: true,
      writable: true,
    });

    const decoded = decodePersonalLibraryCatalog(raw);
    expect(decoded).not.toBeNull();
    expect(Object.hasOwn(decoded!.files, "__proto__")).toBe(true);
    expect(decoded!.files.__proto__).toEqual(prototypeRecord);
  });

  it("requires canonical timestamps and internally consistent categories", () => {
    const dateOnly = populatedCatalog();
    dateOnly.updatedAt = "2026-08-03";
    expect(decodePersonalLibraryCatalog(dateOnly)).toBeNull();

    const inconsistent = populatedCatalog();
    inconsistent.papers["arxiv:2608.00001"]!.categories = ["cs.CL"];
    expect(decodePersonalLibraryCatalog(inconsistent)).toBeNull();

    const duplicate = populatedCatalog();
    duplicate.papers["arxiv:2608.00001"]!.categories = ["cs.AI", "cs.AI"];
    expect(decodePersonalLibraryCatalog(duplicate)).toBeNull();
  });
});

describe("PersonalLibraryCatalogStore", () => {
  it("returns an empty current-schema catalog when no durable state exists", async () => {
    const { storage } = makeStorage();
    const catalog = await makeStore(storage).load(scopeFingerprint, identificationFingerprint);
    expect(catalog).toEqual({
      schemaVersion: PERSONAL_LIBRARY_CATALOG_SCHEMA_VERSION,
      revision: 0,
      scopeFingerprint,
      identificationFingerprint,
      updatedAt: secondNow.toISOString(),
      lastScan: null,
      files: {},
      papers: {},
    });
  });

  it("persists and reloads a validated whole catalog with a semantic revision", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    const saved = await store.replace(populatedCatalog());

    expect(saved.revision).toBe(1);
    expect(saved.updatedAt).toBe(secondNow.toISOString());
    expect(files[documentPath]).toBe(`${JSON.stringify(saved, null, 2)}\n`);
    expect(files[backupPath]).toBe(files[documentPath]);
    await expect(store.load(scopeFingerprint, identificationFingerprint)).resolves.toEqual(saved);
  });

  it("does not write or increment revision for semantically unchanged state", async () => {
    const { files, storage, writeTextAtomic } = makeStorage();
    const store = makeStore(storage);
    const first = await store.replace(populatedCatalog());
    const content = files[documentPath];
    writeTextAtomic.mockClear();
    const second = await store.replace({
      ...first,
      revision: 999,
      updatedAt: "2030-01-01T00:00:00.000Z",
      files: Object.fromEntries(Object.entries(first.files).reverse()),
      papers: Object.fromEntries(Object.entries(first.papers).reverse()),
    });

    expect(second).toEqual(first);
    expect(files[documentPath]).toBe(content);
    expect(writeTextAtomic).not.toHaveBeenCalled();
  });

  it("refuses catalogs from another library scope", async () => {
    const { storage } = makeStorage();
    const store = makeStore(storage);
    await store.replace(populatedCatalog());
    const otherScope = createPersonalLibraryScopeFingerprint({
      rootIdentity: "42:2002",
      eligibleExtensions: [".pdf"],
    });

    await expect(store.load(otherScope, identificationFingerprint))
      .rejects.toThrow(/cannot load unreadable/);
    const replacement = populatedCatalog();
    replacement.scopeFingerprint = otherScope;
    await expect(store.replace(replacement)).rejects.toThrow(/cannot mutate unreadable/);
  });

  it("refuses revision overflow without changing the primary", async () => {
    const { files, storage } = makeStorage();
    const catalog = populatedCatalog();
    catalog.revision = Number.MAX_SAFE_INTEGER;
    files[documentPath] = `${JSON.stringify(catalog, null, 2)}\n`;
    const changed = populatedCatalog();
    changed.lastScan = { ...changed.lastScan!, truncated: true };
    const previous = files[documentPath];

    await expect(makeStore(storage).replace(changed)).rejects.toThrow(/revision is exhausted/);
    expect(files[documentPath]).toBe(previous);
  });

  it("recovers from a valid backup and warns", async () => {
    const { files, storage } = makeStorage();
    const store = makeStore(storage);
    const saved = await store.replace(populatedCatalog());
    files[backupPath] = files[documentPath]!;
    files[documentPath] = "corrupt";
    const warning = vi.fn();

    await expect(makeStore(storage, () => secondNow, warning)
      .load(scopeFingerprint, identificationFingerprint)).resolves.toEqual(saved);
    expect(JSON.parse(files[documentPath]!)).toEqual(saved);
    expect(warning.mock.calls.some(([message]) => String(message).includes("recovered from backup")))
      .toBe(true);
  });

  it("fails closed when durable state is corrupt and no valid backup exists", async () => {
    const { files, storage } = makeStorage();
    files[documentPath] = JSON.stringify({ schemaVersion: 999 });
    files[backupPath] = "not-json";

    await expect(makeStore(storage).load(scopeFingerprint, identificationFingerprint))
      .rejects.toThrow(/cannot load unreadable/);
    await expect(makeStore(storage).replace(populatedCatalog()))
      .rejects.toThrow(/cannot mutate unreadable/);
    expect(files[documentPath]).toBe(JSON.stringify({ schemaVersion: 999 }));
  });

  it("serializes same-path replacements across store instances", async () => {
    const { storage } = makeStorage();
    const firstStore = makeStore(storage);
    const secondStore = makeStore(storage);
    const first = populatedCatalog();
    const second = populatedCatalog();
    second.files["papers/unknown.pdf"] = {
      ...second.files["papers/unknown.pdf"]!,
      observationFingerprint: `sha256:${"9".repeat(64)}`,
    };

    const [firstSaved, secondSaved] = await Promise.all([
      firstStore.replace(first),
      secondStore.replace(second),
    ]);

    expect(firstSaved.revision).toBe(1);
    expect(secondSaved.revision).toBe(2);
    await expect(firstStore.load(scopeFingerprint, identificationFingerprint))
      .resolves.toEqual(secondSaved);
  });

  it("rotates a valid backup without exposing store-owned temp files", async () => {
    const { files, storage } = makeStorage({ rejectExistingRenameTarget: true });
    const store = makeStore(storage);
    const first = await store.replace(populatedCatalog());
    const changed = populatedCatalog();
    changed.lastScan = { ...changed.lastScan!, truncated: true };
    const second = await store.replace(changed);

    expect(second.revision).toBe(2);
    expect(JSON.parse(files[backupPath]!)).toEqual(first);
    expect(files[`${documentPath}.tmp`]).toBeUndefined();
    expect(files[`${backupPath}.tmp`]).toBeUndefined();
  });

  it("restores the previous primary when promotion fails", async () => {
    const { files, storage, writeTextAtomic } = makeStorage();
    const store = makeStore(storage);
    const first = await store.replace(populatedCatalog());
    const previous = files[documentPath]!;
    const changed = populatedCatalog();
    changed.lastScan = { ...changed.lastScan!, truncated: true };
    writeTextAtomic.mockImplementationOnce(async (path, content) => {
      files[path] = content;
    });
    writeTextAtomic.mockImplementationOnce(async () => {
      throw new Error("injected promotion failure");
    });
    writeTextAtomic.mockImplementationOnce(async (path, content) => {
      files[path] = content;
    });

    await expect(store.replace(changed)).rejects.toThrow(/failed to save/);
    expect(files[documentPath]).toBe(previous);
    await expect(store.load(scopeFingerprint, identificationFingerprint)).resolves.toEqual(first);
  });
});
