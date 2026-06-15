import { describe, expect, it } from "vitest";
import {
  PaperIndexError,
  PaperIndexStore,
  derivePaperInboxPaths,
} from "../src/services/paper-index";
import type { StorageAdapter } from "../src/core/adapters";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

function makeStorage(initialFiles: Record<string, string> = {}) {
  const files: Record<string, string> = { ...initialFiles };
  const dirs = new Set<string>();
  const storage = {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/");
    },
    async readText(path: string) {
      if (!(path in files)) throw new Error(`missing ${path}`);
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
      if (!(from in files)) throw new Error(`missing ${from}`);
      files[to] = files[from];
      delete files[from];
    },
    async remove(path: string) {
      delete files[path];
      dirs.delete(path);
    },
  } satisfies StorageAdapter;
  return { files, dirs, storage };
}

function makeStore(initialFiles: Record<string, string> = {}) {
  const { files, dirs, storage } = makeStorage(initialFiles);
  const store = new PaperIndexStore(
    storage,
    DEFAULT_SETTINGS.output,
    () => new Date("2026-06-11T01:30:00.000Z"),
  );
  return { files, dirs, store };
}

describe("derivePaperInboxPaths", () => {
  it("uses the shared output root by default", () => {
    expect(derivePaperInboxPaths(DEFAULT_SETTINGS.output)).toEqual({
      rootDir: "arxiv-daily",
      indexDir: "arxiv-daily/.index",
      papersJsonPath: "arxiv-daily/.index/papers.json",
      legacyIndexDir: "arxiv-daily/index",
      legacyPapersJsonPath: "arxiv-daily/index/papers.json",
    });
  });

  it("falls back to the daily parent when output dirs differ", () => {
    expect(
      derivePaperInboxPaths({
        dailyDir: "reports/arxiv-daily",
        papersDir: "paper-notes/arxiv",
      }),
    ).toEqual({
      rootDir: "reports",
      indexDir: "reports/.index",
      papersJsonPath: "reports/.index/papers.json",
      legacyIndexDir: "reports/index",
      legacyPapersJsonPath: "reports/index/papers.json",
    });
  });

  it("uses arxiv-daily when output dirs have no parent", () => {
    expect(
      derivePaperInboxPaths({
        dailyDir: "daily",
        papersDir: "papers",
      }).rootDir,
    ).toBe("arxiv-daily");
  });
});

describe("PaperIndexStore", () => {
  it("loads an empty index when papers.json is missing", async () => {
    const { store } = makeStore();
    await expect(store.load()).resolves.toEqual({
      schemaVersion: 2,
      updatedAt: "2026-06-11T01:30:00.000Z",
      papers: {},
    });
  });

  it("creates papers.json on first upsert", async () => {
    const { files, dirs, store } = makeStore();
    const { entry, wasNew } = await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A. Author et al.",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
    });
    expect(wasNew).toBe(true);
    expect(entry.status).toBe("inbox");
    expect(entry.priority).toBe("normal");
    expect(entry.paperPath).toBeNull();
    expect(entry.seenDates).toEqual(["2026-06-11"]);
    expect(dirs.has("arxiv-daily")).toBe(true);
    expect(dirs.has("arxiv-daily/.index")).toBe(true);
    const saved = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(saved.schemaVersion).toBe(2);
    expect(saved.papers["2606.12345"].title).toBe("A paper");
  });

  it("stores arXiv publish dates separately from local seen dates", async () => {
    const { store } = makeStore();
    const { entry } = await store.upsertFromDailyPaper({
      arxivId: "2602.01548",
      title: "An older paper",
      authors: "A. Author",
      date: "2026-06-13",
      published: "2026-02-02T02:28:06Z",
      updated: "2026-06-15T02:34:08Z",
      arxivCategory: "astro-ph.GA",
      primaryTopic: "photo-z",
      detail: true,
    });

    expect(entry.published).toBe("2026-02-02");
    expect(entry.updated).toBe("2026-06-15");
    expect(entry.seenDates).toEqual(["2026-06-13"]);
  });

  it("migrates from the old visible index path on next save", async () => {
    const legacy = {
      schemaVersion: 1,
      updatedAt: "2026-06-10T00:00:00.000Z",
      papers: {
        "2606.12345": {
          arxivId: "2606.12345",
          source: "arxiv",
          title: "Legacy paper",
          authors: ["A"],
          published: "2026-06-10",
          updated: "2026-06-10",
          category: "astro-ph",
          topics: ["photo-z"],
          primaryTopic: "photo-z",
          detail: false,
          status: "to_read",
          priority: "high",
          seenDates: ["2026-06-10"],
          dailyReports: ["arxiv-daily/daily/2026-06-10.md"],
          paperPath: null,
          arxivUrl: "https://arxiv.org/abs/2606.12345",
          pdfUrl: "https://arxiv.org/pdf/2606.12345",
          pdfPath: "",
          zoteroKey: "",
          zoteroUri: "",
          citationKey: "",
          projects: [],
        },
      },
    };
    const { files, store } = makeStore({
      "arxiv-daily/index/papers.json": JSON.stringify(legacy),
    });

    await store.setStatus("2606.12345", "saved");

    expect(files["arxiv-daily/index/papers.json"]).toBeUndefined();
    const migrated = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(migrated.schemaVersion).toBe(2);
    expect(migrated.papers["2606.12345"].status).toBe("saved");
    expect(migrated.papers["2606.12345"].priority).toBe("high");
  });

  it("merges repeated papers without overwriting user fields", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "Original title",
      authors: ["A. Author"],
      date: "2026-06-10",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
      dailyReport: "arxiv-daily/daily/2026-06-10.md",
    });
    await store.setStatus("2606.12345", "saved");
    await store.setPriority("2606.12345", "high");
    const { entry, wasNew } = await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "Updated title",
      authors: ["B. Author"],
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "galaxy-cluster",
      detail: true,
      dailyReport: "arxiv-daily/daily/2026-06-11.md",
    });

    expect(wasNew).toBe(false);
    expect(entry.title).toBe("Updated title");
    expect(entry.published).toBe("2026-06-10");
    expect(entry.updated).toBe("2026-06-10");
    expect(entry.topics).toEqual(["photo-z", "galaxy-cluster"]);
    expect(entry.detail).toBe(true);
    expect(entry.status).toBe("saved");
    expect(entry.priority).toBe("high");
    expect(entry.seenDates).toEqual(["2026-06-10", "2026-06-11"]);
    expect(entry.dailyReports).toEqual([
      "arxiv-daily/daily/2026-06-10.md",
      "arxiv-daily/daily/2026-06-11.md",
    ]);
  });

  it("does not duplicate seen dates or daily reports", async () => {
    const { store } = makeStore();
    const input = {
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A. Author",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
      dailyReport: "arxiv-daily/daily/2026-06-11.md",
    };
    await store.upsertFromDailyPaper(input);
    const { entry } = await store.upsertFromDailyPaper(input);
    expect(entry.seenDates).toEqual(["2026-06-11"]);
    expect(entry.dailyReports).toEqual(["arxiv-daily/daily/2026-06-11.md"]);
  });

  it("stores and merges source arXiv categories", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A. Author",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      arxivCategories: ["astro-ph", "cs.LG"],
      primaryTopic: "photo-z",
      detail: false,
    });
    const { entry } = await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A. Author",
      date: "2026-06-12",
      arxivCategory: "cs.CL",
      arxivCategories: ["cs.CL", "cs.LG"],
      primaryTopic: "photo-z",
      detail: false,
    });

    expect(entry.category).toBe("cs.CL");
    expect(entry.categories).toEqual(["astro-ph", "cs.LG", "cs.CL"]);
  });

  it("stores structured paper summaries", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A. Author",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
    });

    const changed = await store.setSummaries({
      "2606.12345": {
        coreProblem: "Problem",
        keyMethod: "Method",
      },
      "2606.99999": {
        coreProblem: "Missing",
      },
    });

    expect(changed).toBe(1);
    const entry = await store.get("2606.12345");
    expect(entry?.summary).toEqual({
      coreProblem: "Problem",
      keyMethod: "Method",
    });
  });

  it("clears detail metadata without removing the paper entry", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A. Author",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: true,
      paperPath: "arxiv-daily/papers/2606.12345.md",
    });

    const changed = await store.clearPaperDetails(["2606.12345", "missing"]);

    expect(changed).toBe(1);
    const entry = await store.get("2606.12345");
    expect(entry).toMatchObject({
      detail: false,
      paperPath: null,
    });
  });

  it("removes paper entries explicitly", async () => {
    const { store } = makeStore();
    await store.upsertManyFromDailyPapers([
      {
        arxivId: "2606.00001",
        title: "A paper",
        authors: "A",
        date: "2026-06-11",
        arxivCategory: "astro-ph",
        primaryTopic: "photo-z",
        detail: false,
      },
      {
        arxivId: "2606.00002",
        title: "Another paper",
        authors: "B",
        date: "2026-06-11",
        arxivCategory: "astro-ph",
        primaryTopic: "photo-z",
        detail: false,
      },
    ]);

    const changed = await store.removePapers([
      "2606.00001",
      "missing",
      "2606.00001",
    ]);

    expect(changed).toBe(1);
    expect(await store.get("2606.00001")).toBeNull();
    expect(await store.get("2606.00002")).not.toBeNull();
  });

  it("throws PaperIndexError for malformed JSON", async () => {
    const { store } = makeStore({
      "arxiv-daily/index/papers.json": "{not-json",
    });
    await expect(store.load()).rejects.toBeInstanceOf(PaperIndexError);
  });

  it("lists papers by status in topic, priority, date order", async () => {
    const { store } = makeStore();
    await store.upsertManyFromDailyPapers([
      {
        arxivId: "2606.00001",
        title: "B normal",
        authors: "A",
        date: "2026-06-10",
        arxivCategory: "astro-ph",
        primaryTopic: "photo-z",
        detail: false,
      },
      {
        arxivId: "2606.00002",
        title: "A high",
        authors: "A",
        date: "2026-06-11",
        arxivCategory: "astro-ph",
        primaryTopic: "photo-z",
        detail: false,
      },
      {
        arxivId: "2606.00003",
        title: "Cluster",
        authors: "A",
        date: "2026-06-11",
        arxivCategory: "astro-ph",
        primaryTopic: "galaxy-cluster",
        detail: false,
      },
    ]);
    await store.setPriority("2606.00002", "high");
    const papers = await store.listByStatus("inbox");
    expect(papers.map((p) => p.arxivId)).toEqual([
      "2606.00003",
      "2606.00002",
      "2606.00001",
    ]);
  });
});
