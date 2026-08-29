import { describe, expect, it, vi } from "vitest";
import {
  openLibraryPdfAtPage,
  resolveLibraryPdfOpenTarget,
} from "../src/library/pdf-opener";

describe("resolveLibraryPdfOpenTarget", () => {
  it("uses Obsidian's vault-relative PDF path when the selected library is inside the vault", () => {
    expect(resolveLibraryPdfOpenTarget({
      canonicalRoot: "/vault/library",
      logicalPath: "papers/evidence paper.pdf",
      vaultRoot: "/vault",
    })).toEqual({ kind: "vault", path: "library/papers/evidence paper.pdf" });
  });

  it("keeps an explicit page fragment when a caller still requests one", () => {
    expect(resolveLibraryPdfOpenTarget({
      canonicalRoot: "/vault/library",
      logicalPath: "papers/evidence paper.pdf",
      page: 7,
      vaultRoot: "/vault",
    })).toEqual({ kind: "vault", path: "library/papers/evidence paper.pdf#page=7" });
  });

  it("uses an encoded external file URL without a page fragment by default", () => {
    expect(resolveLibraryPdfOpenTarget({
      canonicalRoot: "/private/library",
      logicalPath: "papers/evidence paper.pdf",
      vaultRoot: "/vault",
    })).toEqual({
      kind: "external",
      url: "file:///private/library/papers/evidence%20paper.pdf",
    });
  });

  it("keeps an encoded external file URL page fragment when a caller still requests one", () => {
    expect(resolveLibraryPdfOpenTarget({
      canonicalRoot: "/private/library",
      logicalPath: "papers/evidence paper.pdf",
      page: 7,
      vaultRoot: "/vault",
    })).toEqual({
      kind: "external",
      url: "file:///private/library/papers/evidence%20paper.pdf#page=7",
    });
  });

  it("normalizes Windows drive paths before deriving vault and external targets", () => {
    expect(resolveLibraryPdfOpenTarget({
      canonicalRoot: "C:\\Vault\\library",
      logicalPath: "papers/evidence.pdf",
      page: 3,
      vaultRoot: "c:\\vault",
    })).toEqual({ kind: "vault", path: "library/papers/evidence.pdf#page=3" });
    expect(resolveLibraryPdfOpenTarget({
      canonicalRoot: "C:\\private\\library",
      logicalPath: "papers/evidence.pdf",
      page: 3,
      vaultRoot: "C:\\Vault",
    })).toEqual({
      kind: "external",
      url: "file:///C:/private/library/papers/evidence.pdf#page=3",
    });
  });

  it.each(["", "../outside.pdf", "/absolute.pdf", "nested\\windows.pdf", "C:/drive.pdf"])(
    "rejects unsafe selected-library path %s",
    (logicalPath) => {
      expect(() => resolveLibraryPdfOpenTarget({
        canonicalRoot: "/private/library",
        logicalPath,
        page: 1,
      })).toThrow(/safe relative path/);
    },
  );
});

describe("openLibraryPdfAtPage", () => {
  it("falls back to opening the vault PDF without a page subpath when the host rejects page navigation", async () => {
    const openLinkText = vi.fn()
      .mockRejectedValueOnce(new Error("page subpaths unavailable"))
      .mockResolvedValueOnce(undefined);

    const result = await openLibraryPdfAtPage({
      target: { kind: "vault", path: "library/papers/evidence.pdf#page=7" },
      app: { workspace: { openLinkText } } as any,
    });

    expect(result).toBe("file-fallback");
    expect(openLinkText).toHaveBeenNthCalledWith(1, "library/papers/evidence.pdf#page=7", "", false);
    expect(openLinkText).toHaveBeenNthCalledWith(2, "library/papers/evidence.pdf", "", false);
  });
});
