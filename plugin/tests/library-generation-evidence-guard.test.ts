import { describe, expect, it } from "vitest";
import {
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
  type PersonalLibraryCatalog,
} from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";

/**
 * The guard that refuses to finish a generation whose evidence changed
 * underneath it. It fingerprints what the run is based on, so it has to
 * describe every library the run can start from — including one whose files
 * carry no arXiv identity, where the catalog is empty and the index holds
 * everything. Describing that as "no selection at all" made the guard throw
 * before generation began.
 */
const scopeFingerprint = createPersonalLibraryScopeFingerprint({
  rootIdentity: "1:2",
  eligibleExtensions: [".pdf"],
});
const identificationFingerprint = createPersonalLibraryIdentificationFingerprint([".pdf"]);

function emptyCatalog(): PersonalLibraryCatalog {
  return createEmptyPersonalLibraryCatalog(scopeFingerprint, identificationFingerprint);
}

function pluginWith(indexedPapers: Array<{ paperKey: string; title: string }>) {
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  Object.assign(plugin, { libraryIndexedPapers: indexedPapers });
  return plugin as unknown as {
    selectedCatalogFingerprint(catalog: PersonalLibraryCatalog): string;
  };
}

const paper = (index: number) => ({
  paperKey: `file:sha256:${String(index).padStart(64, "0")}`,
  title: `Local Paper ${index}`,
});

describe("generation evidence guard", () => {
  it("describes a library whose catalog is empty and whose index is not", () => {
    const plugin = pluginWith([paper(1), paper(2)]);
    expect(() => plugin.selectedCatalogFingerprint(emptyCatalog())).not.toThrow();
    expect(plugin.selectedCatalogFingerprint(emptyCatalog())).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it("is stable for the same evidence", () => {
    const first = pluginWith([paper(1), paper(2)]).selectedCatalogFingerprint(emptyCatalog());
    const second = pluginWith([paper(1), paper(2)]).selectedCatalogFingerprint(emptyCatalog());
    expect(first).toBe(second);
  });

  it("changes when an indexed paper joins, leaves, or is renamed", () => {
    const base = pluginWith([paper(1), paper(2)]).selectedCatalogFingerprint(emptyCatalog());
    // Without covering the index the guard would sleep through exactly the
    // evidence a non-arXiv library runs on.
    expect(pluginWith([paper(1), paper(2), paper(3)]).selectedCatalogFingerprint(emptyCatalog()))
      .not.toBe(base);
    expect(pluginWith([paper(1)]).selectedCatalogFingerprint(emptyCatalog())).not.toBe(base);
    expect(pluginWith([{ ...paper(1), title: "Retitled" }, paper(2)])
      .selectedCatalogFingerprint(emptyCatalog())).not.toBe(base);
  });

  it("distinguishes an empty library from one with indexed papers", () => {
    expect(pluginWith([]).selectedCatalogFingerprint(emptyCatalog()))
      .not.toBe(pluginWith([paper(1)]).selectedCatalogFingerprint(emptyCatalog()));
  });
});
