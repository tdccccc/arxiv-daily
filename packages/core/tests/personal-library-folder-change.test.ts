import { describe, expect, it, vi } from "vitest";
import {
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
} from "../src/library/personal-library-catalog";
import {
  detectPersonalLibraryFolderChange,
  reconcilePersonalLibraryCatalog,
  type PersonalLibraryMetadataResolver,
  type PersonalLibraryResolvedMetadata,
} from "../src/library/personal-library-reconciliation";
import type { LibraryInventory, LibrarySourceEntry } from "../src/library/scoped-library-source";

const scopeFingerprint = createPersonalLibraryScopeFingerprint({ rootIdentity: "1:2", eligibleExtensions: [".pdf"] });
const identificationFingerprint = createPersonalLibraryIdentificationFingerprint([".pdf"]);
const now = new Date("2026-08-03T12:00:00.000Z");

function file(path: string, size = 100, mtimeMs = 1_754_224_800_000): LibrarySourceEntry {
  return { path, type: "file", size, mtimeMs };
}

function metadata(arxivId: string): PersonalLibraryResolvedMetadata {
  return {
    arxivId, title: `Paper ${arxivId}`, authors: ["A. Author"], abstract: "Abstract.",
    published: "2026-08-01T00:00:00.000Z", updated: "2026-08-02T00:00:00.000Z",
    primaryCategory: "cs.AI", categories: ["cs.AI"],
  };
}

function resolver(values: PersonalLibraryResolvedMetadata[]): PersonalLibraryMetadataResolver & { resolve: ReturnType<typeof vi.fn> } {
  return {
    resolve: vi.fn(async (ids: string[]) => new Map(
      values.filter((value) => ids.includes(value.arxivId)).map((value) => [value.arxivId, value]),
    )),
  };
}

async function scanned(inventory: LibraryInventory, values: PersonalLibraryResolvedMetadata[] = []) {
  const result = await reconcilePersonalLibraryCatalog({
    current: createEmptyPersonalLibraryCatalog(scopeFingerprint, identificationFingerprint, now),
    inventory, eligibleExtensions: [".pdf"], resolver: resolver(values), now,
  });
  return result.catalog;
}

const complete = (...entries: LibrarySourceEntry[]): LibraryInventory => ({ entries, truncated: false });

describe("detectPersonalLibraryFolderChange", () => {
  it("treats a catalog that was never scanned as changed, whatever the folder holds", () => {
    const catalog = createEmptyPersonalLibraryCatalog(scopeFingerprint, identificationFingerprint, now);
    expect(detectPersonalLibraryFolderChange({ catalog, inventory: complete(), eligibleExtensions: [".pdf"] }))
      .toMatchObject({ changed: true, neverScanned: true });
  });

  it("reports an unchanged folder as unchanged, including failed and unrecognized files", async () => {
    const inventory = complete(file("2608.00001.pdf"), file("notes/unknown.pdf"), file("2608.00009.pdf"), file("draft.md"));
    const catalog = await scanned(inventory, [metadata("2608.00001")]);
    expect(Object.values(catalog.files).map((record) => record.status).sort())
      .toEqual(["failed", "ready", "unrelated", "unresolved"]);
    expect(detectPersonalLibraryFolderChange({ catalog, inventory, eligibleExtensions: [".pdf"] }))
      .toEqual({ changed: false, neverScanned: false, added: [], removed: [], modified: [] });
  });

  it("reports added, removed and modified eligible files", async () => {
    const catalog = await scanned(complete(file("a/2608.00001.pdf"), file("2608.00002.pdf"), file("2608.00003.pdf")), [
      metadata("2608.00001"), metadata("2608.00002"), metadata("2608.00003"),
    ]);
    const change = detectPersonalLibraryFolderChange({
      catalog, eligibleExtensions: [".pdf"],
      inventory: complete(file("a/2608.00001.pdf"), file("2608.00002.pdf", 101), file("2608.00004.pdf")),
    });
    expect(change).toEqual({
      changed: true, neverScanned: false,
      added: ["2608.00004.pdf"], removed: ["2608.00003.pdf"], modified: ["2608.00002.pdf"],
    });
  });

  it("sees a renamed or moved file as one removal plus one addition", async () => {
    const catalog = await scanned(complete(file("2608.00001.pdf")), [metadata("2608.00001")]);
    expect(detectPersonalLibraryFolderChange({
      catalog, eligibleExtensions: [".pdf"], inventory: complete(file("moved/2608.00001.pdf")),
    })).toMatchObject({ changed: true, added: ["moved/2608.00001.pdf"], removed: ["2608.00001.pdf"] });
  });

  it("ignores files that can never be library papers", async () => {
    const catalog = await scanned(complete(file("2608.00001.pdf"), file("draft.md")), [metadata("2608.00001")]);
    expect(detectPersonalLibraryFolderChange({
      catalog, eligibleExtensions: [".pdf"],
      inventory: complete(file("2608.00001.pdf"), file("other.docx"), file("sub", 0), { path: "sub", type: "folder" }),
    }).changed).toBe(false);
    expect(detectPersonalLibraryFolderChange({
      catalog, eligibleExtensions: [".pdf"], inventory: complete(file("2608.00001.pdf")),
    }).changed).toBe(false);
  });

  it("cannot infer removals from a truncated listing, but still sees additions", async () => {
    const catalog = await scanned(complete(file("2608.00001.pdf"), file("2608.00002.pdf")), [
      metadata("2608.00001"), metadata("2608.00002"),
    ]);
    expect(detectPersonalLibraryFolderChange({
      catalog, eligibleExtensions: [".pdf"], inventory: { entries: [file("2608.00001.pdf")], truncated: true },
    }).changed).toBe(false);
    expect(detectPersonalLibraryFolderChange({
      catalog, eligibleExtensions: [".pdf"],
      inventory: { entries: [file("2608.00001.pdf"), file("2608.00005.pdf")], truncated: true },
    })).toMatchObject({ changed: true, added: ["2608.00005.pdf"], removed: [] });
  });
});

describe("a scan after a folder change only asks arXiv about papers the catalog lacks", () => {
  async function rescan(current: Awaited<ReturnType<typeof scanned>>, inventory: LibraryInventory, values: PersonalLibraryResolvedMetadata[]) {
    const metadataResolver = resolver(values);
    const result = await reconcilePersonalLibraryCatalog({
      current, inventory, eligibleExtensions: [".pdf"], resolver: metadataResolver, now,
    });
    return { result, metadataResolver };
  }

  it("asks only about the new paper and keeps the others from the catalog", async () => {
    const first = await scanned(complete(file("2608.00001.pdf"), file("2608.00002.pdf")), [metadata("2608.00001"), metadata("2608.00002")]);
    const { result, metadataResolver } = await rescan(
      first, complete(file("2608.00001.pdf"), file("2608.00002.pdf"), file("2608.00003.pdf")), [metadata("2608.00003")],
    );
    expect(metadataResolver.resolve).toHaveBeenCalledOnce();
    expect(metadataResolver.resolve.mock.calls[0]![0]).toEqual(["2608.00003"]);
    expect(Object.keys(result.catalog.papers).sort()).toEqual(["arxiv:2608.00001", "arxiv:2608.00002", "arxiv:2608.00003"]);
  });

  it("does not ask arXiv again for a paper that was only renamed, moved or touched", async () => {
    const first = await scanned(complete(file("2608.00001.pdf"), file("2608.00002.pdf")), [metadata("2608.00001"), metadata("2608.00002")]);
    const { result, metadataResolver } = await rescan(
      first, complete(file("shelf/2608.00001.pdf"), file("2608.00002.pdf", 100, 1_754_999_999_999)), [],
    );
    expect(metadataResolver.resolve).not.toHaveBeenCalled();
    expect(result.catalog.files["shelf/2608.00001.pdf"]?.status).toBe("ready");
    expect(result.catalog.files["2608.00002.pdf"]?.status).toBe("ready");
    expect(result.catalog.files["2608.00001.pdf"]).toBeUndefined();
    expect(result.catalog.papers["arxiv:2608.00001"]?.filePaths).toEqual(["shelf/2608.00001.pdf"]);
    expect(result.catalog.papers["arxiv:2608.00001"]?.title).toBe("Paper 2608.00001");
  });

  it("still asks about a changed file that now identifies as a different paper", async () => {
    const first = await scanned(complete(file("2608.00001.pdf")), [metadata("2608.00001")]);
    const { result, metadataResolver } = await rescan(
      first, complete(file("2608.00002.pdf")), [metadata("2608.00002")],
    );
    expect(metadataResolver.resolve.mock.calls[0]![0]).toEqual(["2608.00002"]);
    expect(Object.keys(result.catalog.papers)).toEqual(["arxiv:2608.00002"]);
  });
});
