import { describe, expect, it, vi } from "vitest";
import {
  createEmptyPersonalLibraryCatalog,
  decodePersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
} from "../src/library/personal-library-catalog";
import {
  createLibraryFileObservationFingerprint,
  identifyModernArxivIdFromFilename,
  reconcilePersonalLibraryCatalog,
  type PersonalLibraryMetadataResolver,
  type PersonalLibraryResolvedMetadata,
} from "../src/library/personal-library-reconciliation";
import type { LibraryInventory, LibrarySourceEntry } from "../src/library/scoped-library-source";
import { isCancellationError } from "../src/services/cancellation";

const scopeFingerprint = createPersonalLibraryScopeFingerprint({
  rootIdentity: "1:2",
  eligibleExtensions: [".pdf"],
});
const identificationFingerprint = createPersonalLibraryIdentificationFingerprint([".pdf"]);
const now = new Date("2026-08-03T12:00:00.000Z");

function file(path: string, size = 100, mtimeMs = 1_754_224_800_000): LibrarySourceEntry {
  return { path, type: "file", size, mtimeMs };
}

function metadata(arxivId: string): PersonalLibraryResolvedMetadata {
  return {
    arxivId,
    title: `Paper ${arxivId}`,
    authors: ["A. Author"],
    abstract: "Abstract.",
    published: "2026-08-01T00:00:00.000Z",
    updated: "2026-08-02T00:00:00.000Z",
    primaryCategory: "cs.AI",
    categories: ["cs.AI"],
  };
}

function resolver(
  values: PersonalLibraryResolvedMetadata[],
): PersonalLibraryMetadataResolver & { resolve: ReturnType<typeof vi.fn> } {
  return {
    resolve: vi.fn(async (ids: string[]) => new Map(
      values.filter((value) => ids.includes(value.arxivId)).map((value) => [value.arxivId, value]),
    )),
  };
}

function emptyCatalog() {
  return createEmptyPersonalLibraryCatalog(scopeFingerprint, identificationFingerprint, now);
}

async function reconcile(
  inventory: LibraryInventory,
  metadataValues: PersonalLibraryResolvedMetadata[] = [],
  current = emptyCatalog(),
  metadataResolver = resolver(metadataValues),
) {
  return await reconcilePersonalLibraryCatalog({
    current,
    inventory,
    eligibleExtensions: [".pdf"],
    resolver: metadataResolver,
    now,
  });
}

describe("identifyModernArxivIdFromFilename", () => {
  it.each([
    ["2608.00001.pdf", "2608.00001"],
    ["papers/arxiv-2608.00001v3-final.PDF", "2608.00001"],
    ["papers/[2608.12345] title.pdf", "2608.12345"],
  ])("identifies %s", (path, expected) => {
    expect(identifyModernArxivIdFromFilename(path)).toBe(expected);
  });

  it.each([
    "paper.pdf",
    "hep-th-9901001.pdf",
    "2608.00001-and-2608.00002.pdf",
    "2608.00001version.pdf",
    "2608.00001v2v3.pdf",
    "x12608.00001.pdf",
  ])("leaves %s unresolved", (path) => {
    expect(identifyModernArxivIdFromFilename(path)).toBeNull();
  });
});

describe("reconcilePersonalLibraryCatalog", () => {
  it("isolates unrelated, unresolved, and missing metadata while keeping ready papers", async () => {
    const result = await reconcile({
      entries: [
        file("papers/2608.00001v2.pdf"),
        file("papers/2608.00002.pdf"),
        file("papers/unknown.pdf"),
        file("notes/draft.md"),
        { path: "folders", type: "folder" },
        { path: "linked", type: "ignored", ignoredReason: "symbolic-link" },
      ],
      truncated: false,
    }, [metadata("2608.00001")]);

    expect(result.resolvedArxivIds).toEqual(["2608.00001", "2608.00002"]);
    expect(result.catalog.lastScan).toEqual({
      ready: 1,
      unresolved: 1,
      unrelated: 1,
      failed: 1,
      papers: 1,
      truncated: false,
    });
    expect(result.catalog.files["papers/2608.00001v2.pdf"]).toMatchObject({
      status: "ready",
      paperKey: "arxiv:2608.00001",
    });
    expect(result.catalog.files["papers/2608.00002.pdf"]).toMatchObject({
      status: "failed",
      reason: "metadata-unavailable",
    });
    expect(result.catalog.papers["arxiv:2608.00001"]?.filePaths)
      .toEqual(["papers/2608.00001v2.pdf"]);
    expect(decodePersonalLibraryCatalog(result.catalog)).toEqual(result.catalog);
  });

  it("maps duplicate PDFs to one paper and resolves the ID once", async () => {
    const metadataResolver = resolver([metadata("2608.00001")]);
    const result = await reconcile({
      entries: [file("a/2608.00001.pdf"), file("b/2608.00001v2.pdf")],
      truncated: false,
    }, [], emptyCatalog(), metadataResolver);

    expect(metadataResolver.resolve).toHaveBeenCalledWith(["2608.00001"], undefined);
    expect(Object.keys(result.catalog.papers)).toEqual(["arxiv:2608.00001"]);
    expect(result.catalog.papers["arxiv:2608.00001"]?.filePaths)
      .toEqual(["a/2608.00001.pdf", "b/2608.00001v2.pdf"]);
  });

  it("reuses unchanged records without resolving metadata again", async () => {
    const first = await reconcile({ entries: [file("2608.00001.pdf")], truncated: false }, [
      metadata("2608.00001"),
    ]);
    const metadataResolver = resolver([]);
    const second = await reconcile(
      { entries: [file("2608.00001.pdf")], truncated: false },
      [],
      first.catalog,
      metadataResolver,
    );

    expect(second.reusedFileCount).toBe(1);
    expect(second.resolvedArxivIds).toEqual([]);
    expect(metadataResolver.resolve).not.toHaveBeenCalled();
    expect(second.catalog.papers).toEqual(first.catalog.papers);
  });

  it("reprocesses changed files but preserves prior usable metadata on resolver failure", async () => {
    const first = await reconcile({ entries: [file("2608.00001.pdf")], truncated: false }, [
      metadata("2608.00001"),
    ]);
    const failingResolver: PersonalLibraryMetadataResolver = {
      resolve: vi.fn(async () => { throw new Error("offline"); }),
    };
    const second = await reconcile(
      { entries: [file("2608.00001.pdf", 101)], truncated: false },
      [],
      first.catalog,
      failingResolver,
    );

    expect(second.catalog.files["2608.00001.pdf"]?.status).toBe("ready");
    expect(second.catalog.papers).toEqual(first.catalog.papers);
  });

  it("removes missing files only after a complete inventory", async () => {
    const first = await reconcile({
      entries: [file("2608.00001.pdf"), file("unknown.pdf")],
      truncated: false,
    }, [metadata("2608.00001")]);
    const complete = await reconcile({ entries: [], truncated: false }, [], first.catalog);
    const truncated = await reconcile({ entries: [], truncated: true }, [], first.catalog);

    expect(complete.catalog.files).toEqual({});
    expect(complete.catalog.papers).toEqual({});
    expect(truncated.catalog.files).toEqual(first.catalog.files);
    expect(truncated.catalog.papers).toEqual(first.catalog.papers);
  });

  it("retries failed files and preserves cancellation without returning a catalog", async () => {
    const first = await reconcile({ entries: [file("2608.00001.pdf")], truncated: false });
    expect(first.catalog.files["2608.00001.pdf"]?.status).toBe("failed");
    const metadataResolver = resolver([metadata("2608.00001")]);
    const retried = await reconcile(
      { entries: [file("2608.00001.pdf")], truncated: false },
      [],
      first.catalog,
      metadataResolver,
    );
    expect(retried.catalog.files["2608.00001.pdf"]?.status).toBe("ready");

    const controller = new AbortController();
    controller.abort("stop");
    await expect(reconcilePersonalLibraryCatalog({
      current: retried.catalog,
      inventory: { entries: [], truncated: false },
      eligibleExtensions: [".pdf"],
      resolver: metadataResolver,
      signal: controller.signal,
    })).rejects.toSatisfy(isCancellationError);
  });

  it("preserves prototype-sensitive paths during complete and truncated scans", async () => {
    const prototypePaths = ["__proto__", "constructor", "toString"];
    const first = await reconcile({
      entries: prototypePaths.map((path) => file(path)),
      truncated: false,
    });
    for (const path of prototypePaths) {
      expect(Object.hasOwn(first.catalog.files, path)).toBe(true);
      expect(first.catalog.files[path]).toMatchObject({ path, status: "unrelated" });
    }

    const truncated = await reconcile({ entries: [], truncated: true }, [], first.catalog);
    for (const path of prototypePaths) {
      expect(Object.hasOwn(truncated.catalog.files, path)).toBe(true);
    }
  });

  it("isolates metadata that cannot produce a strictly decodable catalog", async () => {
    const invalid = metadata("2608.00001");
    invalid.published = "2026-08-01";
    invalid.categories = ["cs.AI", "cs.AI"];
    const result = await reconcile({
      entries: [file("2608.00001.pdf")],
      truncated: false,
    }, [invalid]);

    expect(result.catalog.files["2608.00001.pdf"]).toMatchObject({
      status: "failed",
      reason: "metadata-unavailable",
    });
    expect(result.catalog.papers).toEqual({});
  });

  it("binds observations to path, metadata, and identification strategy", () => {
    const entry = file("2608.00001.pdf");
    expect(createLibraryFileObservationFingerprint(entry, identificationFingerprint))
      .not.toBe(createLibraryFileObservationFingerprint({ ...entry, size: 101 }, identificationFingerprint));
    expect(createLibraryFileObservationFingerprint(entry, identificationFingerprint))
      .not.toBe(createLibraryFileObservationFingerprint({ ...entry, path: "copy.pdf" }, identificationFingerprint));
  });
});

describe("content-based file identification (strategy v2)", () => {
  it("identifies unresolved files through the injected identifier", async () => {
    const identifyFile = { identify: vi.fn(async (path: string) =>
      path.endsWith("Wadekar2023.pdf") ? "2001.04385" : null) };
    const result = await reconcilePersonalLibraryCatalog({
      current: emptyCatalog(),
      inventory: { entries: [file("Wadekar2023.pdf")], truncated: false },
      eligibleExtensions: [".pdf"],
      resolver: resolver([metadata("2001.04385")]),
      identifyFile,
      now,
    });
    expect(identifyFile.identify).toHaveBeenCalledWith("Wadekar2023.pdf", undefined, 100);
    const record = Object.values(result.catalog.files)[0]!;
    expect(record.status).toBe("ready");
    expect(result.catalog.papers["arxiv:2001.04385"]).toBeDefined();
  });

  it("keeps files unresolved when content identification fails or throws", async () => {
    const identifyFile = { identify: vi.fn(async () => {
      throw new Error("read failed");
    }) };
    const result = await reconcilePersonalLibraryCatalog({
      current: emptyCatalog(),
      inventory: { entries: [file("scan.pdf")], truncated: false },
      eligibleExtensions: [".pdf"],
      resolver: resolver([]),
      identifyFile,
      now,
    });
    const record = Object.values(result.catalog.files)[0]!;
    expect(record.status).toBe("unresolved");
  });

  it("never calls the identifier for files whose names already carry an arXiv ID", async () => {
    const identifyFile = { identify: vi.fn(async () => "2402.18634") };
    const result = await reconcilePersonalLibraryCatalog({
      current: emptyCatalog(),
      inventory: { entries: [file("2402.18634v2.pdf")], truncated: false },
      eligibleExtensions: [".pdf"],
      resolver: resolver([metadata("2402.18634")]),
      identifyFile,
      now,
    });
    expect(identifyFile.identify).not.toHaveBeenCalled();
    const record = Object.values(result.catalog.files)[0]!;
    expect(record.status).toBe("ready");
  });
});
