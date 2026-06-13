import { describe, expect, it, vi } from "vitest";
import {
  BibtexService,
  extractArxivIdFromMarkdown,
  parseBibtexKey,
  rewriteBibtexKey,
} from "../src/services/bibtex";
import { Logger } from "../src/services/logger";
import { PaperIndexStore } from "../src/services/paper-index";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { StorageAdapter } from "../src/core/adapters";

function makeStorage(initialFiles: Record<string, string> = {}) {
  const files: Record<string, string> = { ...initialFiles };
  const dirs = new Set<string>();
  const storage = {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/");
    },
    async readText(path: string) {
      return files[path];
    },
    async writeText(path: string, content: string) {
      files[path] = content;
    },
    async exists(path: string) {
      return path in files || dirs.has(path);
    },
    async mkdir(path: string) {
      dirs.add(path);
    },
    async rename(from: string, to: string) {
      files[to] = files[from];
      delete files[from];
    },
    async remove(path: string) {
      delete files[path];
      dirs.delete(path);
    },
  } satisfies StorageAdapter;
  return { files, storage };
}

describe("BibTeX service", () => {
  it("parses citation keys from BibTeX entries", () => {
    expect(parseBibtexKey("@article{Smith2026,\n title={T}\n}")).toBe(
      "Smith2026",
    );
    expect(parseBibtexKey("@misc{2606.12345,\n title={T}\n}")).toBe(
      "2606.12345",
    );
    expect(parseBibtexKey("not bibtex")).toBeNull();
    expect(rewriteBibtexKey("@article{OldKey,\n title={T}\n}", "NewKey")).toBe(
      "@article{NewKey,\n title={T}\n}",
    );
  });

  it("extracts arXiv IDs from markdown frontmatter and body", () => {
    expect(
      extractArxivIdFromMarkdown(
        [
          "---",
          'arxiv_id: "2606.12345"',
          "---",
          "",
          "Body",
        ].join("\n"),
      ),
    ).toBe("2606.12345");
    expect(
      extractArxivIdFromMarkdown(
        "See [paper](https://arxiv.org/abs/2606.54321v2) for details.",
      ),
    ).toBe("2606.54321");
    expect(extractArxivIdFromMarkdown("Related arXiv:2606.99999v1")).toBe(
      "2606.99999",
    );
    expect(extractArxivIdFromMarkdown("No paper here")).toBeNull();
  });

  it("fetches BibTeX and stores citationKey for indexed papers", async () => {
    const { files, storage } = makeStorage();
    const store = new PaperIndexStore(
      storage,
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-13T00:00:00.000Z"),
    );
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A",
      date: "2026-06-13",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
    });
    const service = new BibtexService({
      fetcher: {
        fetchBibtex: vi
          .fn()
          .mockResolvedValue("@article{Smith2026,\n title={A paper}\n}\n"),
      },
      paperIndex: store,
      logger: new Logger("error"),
    });

    const result = await service.fetchAndStore(
      "https://arxiv.org/abs/2606.12345v2",
    );

    expect(result).toMatchObject({
      kind: "done",
      arxivId: "2606.12345",
      citationKey: "Smith2026",
      entryUpdated: true,
    });
    const saved = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(saved.papers["2606.12345"].citationKey).toBe("Smith2026");
  });

  it("returns BibTeX even when the paper is not indexed", async () => {
    const { storage } = makeStorage();
    const store = new PaperIndexStore(
      storage,
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-13T00:00:00.000Z"),
    );
    const service = new BibtexService({
      fetcher: {
        fetchBibtex: vi.fn().mockResolvedValue("@misc{Key2026,\n}\n"),
      },
      paperIndex: store,
      logger: new Logger("error"),
    });

    const result = await service.fetchAndStore("2606.54321");

    expect(result).toMatchObject({
      kind: "done",
      arxivId: "2606.54321",
      citationKey: "Key2026",
      entryUpdated: false,
    });
  });

  it("rejects invalid ids and invalid BibTeX responses", async () => {
    const service = new BibtexService({
      fetcher: {
        fetchBibtex: vi.fn().mockResolvedValue("not bibtex"),
      },
      logger: new Logger("error"),
    });

    await expect(service.fetchAndStore("not-an-id")).resolves.toMatchObject({
      kind: "invalid_id",
    });
    await expect(service.fetchAndStore("2606.12345")).resolves.toMatchObject({
      kind: "invalid_bibtex",
    });
  });

  it("exports many BibTeX entries and rewrites duplicate citation keys", async () => {
    const { files, storage } = makeStorage();
    const store = new PaperIndexStore(
      storage,
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-13T00:00:00.000Z"),
    );
    await store.upsertManyFromDailyPapers([
      {
        arxivId: "2606.12345",
        title: "A paper",
        authors: "A",
        date: "2026-06-13",
        arxivCategory: "astro-ph",
        primaryTopic: "photo-z",
        detail: false,
      },
      {
        arxivId: "2606.54321",
        title: "Another paper",
        authors: "B",
        date: "2026-06-13",
        arxivCategory: "astro-ph",
        primaryTopic: "photo-z",
        detail: false,
      },
    ]);
    const service = new BibtexService({
      fetcher: {
        fetchBibtex: vi.fn(async (id: string) => `@article{SameKey,\n title={${id}}\n}\n`),
      },
      paperIndex: store,
      storage,
      output: DEFAULT_SETTINGS.output,
      logger: new Logger("error"),
    });

    const result = await service.exportManyToFile(
      [{ arxivId: "2606.12345" }, { arxivId: "2606.54321" }],
      { path: "arxiv-daily/exports/test.bib" },
    );

    expect(result).toMatchObject({
      kind: "done",
      path: "arxiv-daily/exports/test.bib",
      requested: 2,
      exported: 2,
      keysRenamed: 1,
    });
    expect(files["arxiv-daily/exports/test.bib"]).toContain(
      "@article{SameKey,\n title={2606.12345}",
    );
    expect(files["arxiv-daily/exports/test.bib"]).toContain(
      "@article{SameKey_260654321,\n title={2606.54321}",
    );
    const saved = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(saved.papers["2606.12345"].citationKey).toBe("SameKey");
    expect(saved.papers["2606.54321"].citationKey).toBe("SameKey_260654321");
  });

  it("keeps successful entries when some batch BibTeX fetches fail", async () => {
    const { files, storage } = makeStorage();
    const service = new BibtexService({
      fetcher: {
        fetchBibtex: vi.fn(async (id: string) =>
          id === "2606.12345" ? "@misc{GoodKey,\n}\n" : "not bibtex",
        ),
      },
      storage,
      output: DEFAULT_SETTINGS.output,
      logger: new Logger("error"),
    });

    const result = await service.exportManyToFile(
      [{ arxivId: "2606.12345" }, { arxivId: "2606.54321" }],
      { path: "arxiv-daily/exports/partial.bib" },
    );

    expect(result).toMatchObject({
      kind: "done",
      requested: 2,
      exported: 1,
      failures: [{ arxivId: "2606.54321" }],
    });
    expect(files["arxiv-daily/exports/partial.bib"]).toContain(
      "% Failed 2606.54321: could not parse BibTeX citation key",
    );
    expect(files["arxiv-daily/exports/partial.bib"]).toContain("@misc{GoodKey,");
  });
});
