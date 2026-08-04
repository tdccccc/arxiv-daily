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

  it("rejects portable collisions at the final derivation boundary", () => {
    expect(() => derivePaperInboxPaths({
      dailyDir: "Café/Notes",
      papersDir: "CAFE\u0301/notes",
    })).toThrow(/must be different/);
  });

  it("uses the canonical daily parent when parent paths collide portably", () => {
    expect(derivePaperInboxPaths({
      dailyDir: "Café/daily",
      papersDir: "CAFE\u0301/papers",
    }).rootDir).toBe("Café");
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
      schemaVersion: 5,
      updatedAt: "2026-06-11T01:30:00.000Z",
      papers: {},
    });
  });

  it.each([1, 2, 3, 4, 5])("reads paper index schema %i", async (schemaVersion) => {
    const bareOrKey =
      schemaVersion >= 4
        ? {
            "arxiv:2606.12345": {
              paperKey: "arxiv:2606.12345",
              source: "arxiv",
              externalId: "2606.12345",
              arxivId: "2606.12345",
              title: `Schema ${schemaVersion}`,
              abstract: "  Persisted abstract.  ",
            },
          }
        : {
            "2606.12345": {
              arxivId: "2606.12345",
              title: `Schema ${schemaVersion}`,
              abstract: schemaVersion === 3 ? "  Persisted abstract.  " : undefined,
            },
          };
    const { store } = makeStore({
      "arxiv-daily/.index/papers.json": JSON.stringify({
        schemaVersion,
        updatedAt: "2026-06-11T00:00:00.000Z",
        papers: bareOrKey,
      }),
    });

    const inbox = await store.load();

    expect(inbox.schemaVersion).toBe(5);
    expect(inbox.papers["arxiv:2606.12345"]).toMatchObject({
      paperKey: "arxiv:2606.12345",
      source: "arxiv",
      externalId: "2606.12345",
      arxivId: "2606.12345",
      title: `Schema ${schemaVersion}`,
      ...(schemaVersion >= 3 ? { abstract: "Persisted abstract." } : {}),
    });
  });

  it("rejects a persisted key/entry arXiv ID mismatch", async () => {
    const { store } = makeStore({
      "arxiv-daily/.index/papers.json": JSON.stringify({
        schemaVersion: 2,
        updatedAt: "2026-06-11T00:00:00.000Z",
        papers: {
          "2606.12345": { arxivId: "2606.54321", title: "mismatch" },
        },
      }),
    });
    await expect(store.load()).rejects.toThrow(/key\/entry mismatch/);
  });

  it("ignores persisted arXiv URLs and derives canonical links", async () => {
    const { store } = makeStore({
      "arxiv-daily/.index/papers.json": JSON.stringify({
        schemaVersion: 2,
        updatedAt: "2026-06-11T00:00:00.000Z",
        papers: {
          "2606.12345": {
            arxivId: "2606.12345v2",
            title: "safe",
            arxivUrl: "javascript:alert(1)",
            pdfUrl: "https://evil.test/file.pdf",
          },
        },
      }),
    });
    const entry = (await store.load()).papers["arxiv:2606.12345"];
    expect(entry.arxivUrl).toBe("https://arxiv.org/abs/2606.12345");
    expect(entry.pdfUrl).toBe("https://arxiv.org/pdf/2606.12345");
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
    expect(saved.schemaVersion).toBe(5);
    expect(saved.papers["arxiv:2606.12345"].title).toBe("A paper");
    expect(saved.papers["arxiv:2606.12345"]).toMatchObject({
      paperKey: "arxiv:2606.12345",
      source: "arxiv",
      externalId: "2606.12345",
      arxivId: "2606.12345",
    });
    expect(saved.papers["2606.12345"]).toBeUndefined();
  });

  it("persists occurrence provenance per repeated committed report and preserves user fields", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345", title: "A paper", authors: "A", date: "2026-06-11",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: false,
      dailyReport: "arxiv-daily/daily/2026-06-11.md",
    });
    await store.setStatus("2606.12345", "saved");
    const first = {
      manualTopicTags: ["photo-z"],
      directions: [],
    };
    const second = {
      manualTopicTags: [],
      directions: [{
        id: "direction-1", name: "Library direction", representatives: [{
          paperKey: "arxiv:2501.00001", title: "Prior", evidenceDepth: "metadata-and-abstract" as const,
        }],
      }],
    };
    await store.reconcileDailyReportOccurrenceProvenance(
      "arxiv-daily/daily/2026-06-11.md",
      [{ arxivId: "2606.12345", provenance: first }],
    );
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345", title: "A paper", authors: "A", date: "2026-06-12",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: false,
      dailyReport: "arxiv-daily/daily/2026-06-12.md",
    });
    await store.reconcileDailyReportOccurrenceProvenance(
      "arxiv-daily/daily/2026-06-12.md",
      [{ arxivId: "2606.12345", provenance: second }],
    );
    const entry = (await store.load()).papers["arxiv:2606.12345"]!;
    expect(entry.status).toBe("saved");
    expect(entry.dailyReports).toHaveLength(2);
    expect(entry.discoveryProvenanceByReport).toEqual({
      "arxiv-daily/daily/2026-06-11.md": first,
      "arxiv-daily/daily/2026-06-12.md": second,
    });
    await store.reconcileDailyReportOccurrenceProvenance(
      "arxiv-daily/daily/2026-06-11.md", [],
    );
    expect((await store.load()).papers["arxiv:2606.12345"]!.discoveryProvenanceByReport)
      .toEqual({ "arxiv-daily/daily/2026-06-12.md": second });
  });

  it("normalizes and preserves abstracts during upsert", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A. Author",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      abstract: "  Enriched abstract.  ",
      primaryTopic: "photo-z",
      detail: false,
    });

    const { entry } = await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A. Author",
      date: "2026-06-12",
      arxivCategory: "astro-ph",
      abstract: "   ",
      primaryTopic: "photo-z",
      detail: false,
    });

    expect(entry.abstract).toBe("Enriched abstract.");
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
    expect(migrated.schemaVersion).toBe(5);
    expect(migrated.papers["arxiv:2606.12345"].status).toBe("saved");
    expect(migrated.papers["arxiv:2606.12345"].priority).toBe("high");
    expect(migrated.papers["arxiv:2606.12345"]).toMatchObject({
      paperKey: "arxiv:2606.12345",
      externalId: "2606.12345",
      arxivId: "2606.12345",
    });
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

    await expect(store.setSummaries({
      "2606.12345": { coreProblem: "Problem" },
    })).resolves.toBe(0);
    await expect(store.setSummaries({
      "2606.12345": { mainResult: "Result" },
    })).resolves.toBe(1);
    expect((await store.get("2606.12345"))?.summary).toEqual({
      coreProblem: "Problem",
      keyMethod: "Method",
      mainResult: "Result",
    });
  });

  it("serializes concurrent mutations from separate stores targeting the same path", async () => {
    const { files, storage } = makeStorage();
    const seedStore = new PaperIndexStore(
      storage,
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-11T01:30:00.000Z"),
    );
    await seedStore.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A. Author",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
    });

    const delayedStorage = {
      ...storage,
      async readText(path: string) {
        const content = await storage.readText(path);
        await new Promise((resolve) => setTimeout(resolve, 0));
        return content;
      },
    } satisfies StorageAdapter;
    const storeA = new PaperIndexStore(delayedStorage, DEFAULT_SETTINGS.output);
    const storeB = new PaperIndexStore(delayedStorage, DEFAULT_SETTINGS.output);

    await Promise.all([
      storeA.setPriority("2606.12345", "high"),
      storeB.setStatus("2606.12345", "saved"),
    ]);

    const saved = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(saved.papers["arxiv:2606.12345"]).toMatchObject({
      priority: "high",
      status: "saved",
      paperKey: "arxiv:2606.12345",
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

  it("refuses detail removal when the current indexed path mismatches", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: true,
      paperPath: "other/place/2606.12345.md",
    });
    let prepared = false;

    const result = await store.removePaperDetailsAtPath(
      "2606.12345",
      "arxiv-daily/papers/2606.12345.md",
      async () => { prepared = true; },
    );

    expect(result).toEqual({
      kind: "path_mismatch",
      actualPath: "other/place/2606.12345.md",
    });
    expect(prepared).toBe(false);
    expect(await store.get("2606.12345")).not.toBeNull();
  });

  it("uses current daily-report state to clear rather than remove", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: true,
      paperPath: "arxiv-daily/papers/2606.12345.md",
    });
    await store.addDailyReports(
      ["2606.12345"],
      "arxiv-daily/daily/2026-06-12.md",
    );

    const result = await store.removePaperDetailsAtPath(
      "2606.12345",
      "arxiv-daily/papers/2606.12345.md",
    );

    expect(result.kind).toBe("cleared");
    expect(await store.get("2606.12345")).toMatchObject({
      detail: false,
      paperPath: null,
      dailyReports: ["arxiv-daily/daily/2026-06-12.md"],
    });
  });

  it("atomically creates a recovered manual detail directly as saved", async () => {
    const { store } = makeStore();
    const result = await store.reconcileManualDetail({
      arxivId: "2606.12345",
      title: "Recovered",
      authors: "A. Author",
      date: "2026-06-11",
      published: "2026-06-10",
      updated: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: true,
    }, "arxiv-daily/papers/2606.12345.md", "saved");

    expect(result).toMatchObject({
      wasNew: true,
      entry: {
        status: "saved",
        detail: true,
        paperPath: "arxiv-daily/papers/2606.12345.md",
      },
    });
  });

  it("preserves existing user status while repairing verified manual detail state", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345", title: "Existing", authors: "A", date: "2026-06-11",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: false,
    });
    await store.setStatus("2606.12345", "reading");

    const result = await store.reconcileManualDetail({
      arxivId: "2606.12345", title: "Verified", authors: "A", date: "2026-06-11",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: true,
    }, "arxiv-daily/papers/2606.12345.md", "saved");

    expect(result).toMatchObject({
      wasNew: false,
      entry: { status: "reading", detail: true, paperPath: "arxiv-daily/papers/2606.12345.md" },
    });
  });

  it("leaves no default-inbox partial reconstruction after any atomic save failure", async () => {
    const base = makeStorage();
    let failure: "write" | "install" | null = "write";
    const storage = {
      ...base.storage,
      async writeText(path: string, content: string) {
        if (failure === "write" && path.endsWith("papers.json.tmp")) throw new Error("write failed");
        await base.storage.writeText(path, content);
      },
      async rename(from: string, to: string) {
        if (failure === "install" && from.endsWith("papers.json.tmp")) throw new Error("install failed");
        await base.storage.rename(from, to);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);
    const reconcile = () => store.reconcileManualDetail({
      arxivId: "2606.12345", title: "Recovered", authors: "A", date: "2026-06-11",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: true,
    }, "arxiv-daily/papers/2606.12345.md", "saved");

    await expect(reconcile()).rejects.toThrow("write failed");
    expect(await store.get("2606.12345")).toBeNull();
    failure = "install";
    await expect(reconcile()).rejects.toThrow();
    expect(await store.get("2606.12345")).toBeNull();
    failure = null;
    await expect(reconcile()).resolves.toMatchObject({ entry: { status: "saved" } });
  });

  it("reports an index failure after the pre-mutation action succeeds", async () => {
    const base = makeStorage();
    const seed = new PaperIndexStore(base.storage, DEFAULT_SETTINGS.output);
    await seed.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A paper",
      authors: "A",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: true,
      paperPath: "arxiv-daily/papers/2606.12345.md",
    });
    let failWrites = true;
    const failingStorage = {
      ...base.storage,
      async writeText(path: string, content: string) {
        if (failWrites && path.endsWith("papers.json.tmp")) {
          throw new Error("index write failed");
        }
        await base.storage.writeText(path, content);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(failingStorage, DEFAULT_SETTINGS.output);
    let trashed = false;

    const result = await store.removePaperDetailsAtPath(
      "2606.12345",
      "arxiv-daily/papers/2606.12345.md",
      async () => { trashed = true; },
    );

    expect(trashed).toBe(true);
    expect(result).toMatchObject({ kind: "index_failed", action: "removed" });
    failWrites = false;
    expect(await store.get("2606.12345")).not.toBeNull();
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

  it("looks up by bare arXiv id or paperKey after schema-4 normalize", async () => {
    const { store } = makeStore({
      "arxiv-daily/.index/papers.json": JSON.stringify({
        schemaVersion: 3,
        updatedAt: "2026-06-11T00:00:00.000Z",
        papers: {
          "2606.12345": {
            arxivId: "2606.12345",
            title: "Legacy",
            paperPath: "arxiv-daily/papers/2606.12345.md",
          },
        },
      }),
    });

    await expect(store.get("2606.12345")).resolves.toMatchObject({
      paperKey: "arxiv:2606.12345",
      externalId: "2606.12345",
      arxivId: "2606.12345",
      title: "Legacy",
      paperPath: "arxiv-daily/papers/2606.12345.md",
    });
    await expect(store.get("arxiv:2606.12345")).resolves.toMatchObject({
      title: "Legacy",
    });

    await store.setStatus("arxiv:2606.12345", "saved");
    const reloaded = await store.load();
    expect(Object.keys(reloaded.papers)).toEqual(["arxiv:2606.12345"]);
    expect(reloaded.papers["arxiv:2606.12345"]?.status).toBe("saved");
    expect(reloaded.papers["arxiv:2606.12345"]?.paperPath).toBe(
      "arxiv-daily/papers/2606.12345.md",
    );
    expect(reloaded.papers["arxiv:2606.12345"]?.paperPath).not.toContain(
      "arxiv:",
    );
  });
});
