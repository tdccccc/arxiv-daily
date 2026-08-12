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

function paperIndexJson(
  id: string,
  title: string,
  status: "inbox" | "saved" = "inbox",
): string {
  return JSON.stringify({
    schemaVersion: 4,
    updatedAt: "2026-06-10T00:00:00.000Z",
    papers: {
      [`arxiv:${id}`]: {
        paperKey: `arxiv:${id}`,
        source: "arxiv",
        externalId: id,
        arxivId: id,
        title,
        status,
      },
    },
  });
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

  it("starts from a valid backup when the primary is missing", async () => {
    const { store } = makeStore({
      "arxiv-daily/.index/papers.json.bak": paperIndexJson(
        "2606.10001",
        "Backup only",
        "saved",
      ),
    });

    await expect(store.load()).resolves.toMatchObject({
      papers: {
        "arxiv:2606.10001": { title: "Backup only", status: "saved" },
      },
    });
  });

  it("recovers from a valid backup when the primary is corrupt", async () => {
    const { store } = makeStore({
      "arxiv-daily/.index/papers.json": "{corrupt",
      "arxiv-daily/.index/papers.json.bak": paperIndexJson(
        "2606.10002",
        "Recovered backup",
      ),
    });

    await expect(store.load()).resolves.toMatchObject({
      papers: {
        "arxiv:2606.10002": { title: "Recovered backup" },
      },
    });
  });

  it("uses a valid primary before backup and legacy files", async () => {
    const { store } = makeStore({
      "arxiv-daily/.index/papers.json": paperIndexJson("2606.10003", "Primary"),
      "arxiv-daily/.index/papers.json.bak": paperIndexJson("2606.10004", "Backup"),
      "arxiv-daily/index/papers.json": paperIndexJson("2606.10005", "Legacy"),
    });

    const loaded = await store.load();

    expect(Object.keys(loaded.papers)).toEqual(["arxiv:2606.10003"]);
    expect(loaded.papers["arxiv:2606.10003"]?.title).toBe("Primary");
  });

  it("falls back to legacy after invalid primary and backup documents", async () => {
    const { store } = makeStore({
      "arxiv-daily/.index/papers.json": "{corrupt",
      "arxiv-daily/.index/papers.json.bak": JSON.stringify({
        schemaVersion: 999,
        papers: {},
      }),
      "arxiv-daily/index/papers.json": paperIndexJson("2606.10006", "Legacy"),
    });

    await expect(store.load()).resolves.toMatchObject({
      papers: { "arxiv:2606.10006": { title: "Legacy" } },
    });
  });

  it("fails closed when the primary exists but cannot be read", async () => {
    const base = makeStorage({
      "arxiv-daily/.index/papers.json": paperIndexJson("2606.10007", "Unreadable"),
      "arxiv-daily/.index/papers.json.bak": paperIndexJson("2606.10008", "Backup"),
    });
    const storage = {
      ...base.storage,
      async readText(path: string) {
        if (path === "arxiv-daily/.index/papers.json") {
          throw new Error("permission denied");
        }
        return base.storage.readText(path);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.load()).rejects.toMatchObject({
      name: "PaperIndexError",
      cause: expect.objectContaining({ message: "permission denied" }),
    });
  });

  it("throws instead of returning an empty inbox when all existing copies are invalid", async () => {
    const { store } = makeStore({
      "arxiv-daily/.index/papers.json": "{corrupt",
      "arxiv-daily/.index/papers.json.bak": JSON.stringify({
        schemaVersion: 999,
        papers: {},
      }),
    });

    await expect(store.load()).rejects.toBeInstanceOf(PaperIndexError);
  });

  it.each([1, 2, 3, 4])("reads paper index schema %i", async (schemaVersion) => {
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

  it("persists occurrence novelty per repeated committed report and preserves user fields", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345", title: "A paper", authors: "A", date: "2026-06-11",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: false,
      dailyReport: "arxiv-daily/daily/2026-06-11.md",
    });
    await store.setStatus("2606.12345", "saved");
    const first = {
      differenceType: "new-method",
      comparisonBasis: ["arxiv:2501.00001"],
      evidenceDepth: "metadata-and-abstract" as const,
      explanation: "First explanation.",
    };
    const second = {
      differenceType: "new-task",
      comparisonBasis: ["arxiv:2501.00001", "arxiv:2501.00002"],
      evidenceDepth: "metadata-and-abstract" as const,
      explanation: "Second explanation.",
    };
    await store.reconcileDailyReportOccurrenceNovelty(
      "arxiv-daily/daily/2026-06-11.md",
      [{ arxivId: "2606.12345", novelty: first }],
    );
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345", title: "A paper", authors: "A", date: "2026-06-12",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: false,
      dailyReport: "arxiv-daily/daily/2026-06-12.md",
    });
    await store.reconcileDailyReportOccurrenceNovelty(
      "arxiv-daily/daily/2026-06-12.md",
      [{ arxivId: "2606.12345", novelty: second }],
    );
    const entry = (await store.load()).papers["arxiv:2606.12345"]!;
    expect(entry.status).toBe("saved");
    expect(entry.dailyReports).toHaveLength(2);
    expect(entry.noveltyByReport).toEqual({
      "arxiv-daily/daily/2026-06-11.md": first,
      "arxiv-daily/daily/2026-06-12.md": second,
    });
    await store.reconcileDailyReportOccurrenceNovelty(
      "arxiv-daily/daily/2026-06-11.md", [],
    );
    expect((await store.load()).papers["arxiv:2606.12345"]!.noveltyByReport)
      .toEqual({ "arxiv-daily/daily/2026-06-12.md": second });
  });

  it("rejects malformed novelty occurrences without mutating the index", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345", title: "A paper", authors: "A", date: "2026-06-11",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: false,
    });
    await expect(store.reconcileDailyReportOccurrenceNovelty(
      "arxiv-daily/daily/2026-06-11.md",
      [{ arxivId: "2606.12345", novelty: {
        differenceType: "breakthrough",
        comparisonBasis: ["arxiv:2501.00001"],
        evidenceDepth: "metadata-and-abstract",
        explanation: "Invalid type.",
      } }],
    )).rejects.toThrow(PaperIndexError);
    expect((await store.load()).papers["arxiv:2606.12345"]!.noveltyByReport).toEqual({});
  });

  it("normalizes novelty by report strictly on load and round-trips through schema 5", async () => {
    const valid = {
      differenceType: "new-method",
      comparisonBasis: ["arxiv:2501.00001"],
      evidenceDepth: "metadata-and-abstract",
      explanation: "Valid explanation.",
    };
    const { files, store } = makeStore({
      "arxiv-daily/.index/papers.json": JSON.stringify({
        schemaVersion: 5,
        updatedAt: "2026-06-11T00:00:00.000Z",
        papers: {
          "arxiv:2606.12345": {
            paperKey: "arxiv:2606.12345",
            source: "arxiv",
            externalId: "2606.12345",
            arxivId: "2606.12345",
            title: "Novelty paper",
            noveltyByReport: {
              "arxiv-daily/daily/2026-06-11.md": valid,
              "arxiv-daily/daily/bad-type.md": {
                ...valid,
                differenceType: "breakthrough",
              },
              "arxiv-daily/daily/bad-basis.md": {
                ...valid,
                comparisonBasis: ["arxiv:2501.00001", "arxiv:2501.00001"],
              },
              "arxiv-daily/daily/bad-depth.md": {
                ...valid,
                evidenceDepth: "full-text",
              },
              "arxiv-daily/daily/padded.md": {
                ...valid,
                explanation: "  padded  ",
              },
              "": valid,
            },
          },
        },
      }),
    });

    const inbox = await store.load();
    expect(inbox.papers["arxiv:2606.12345"]!.noveltyByReport).toEqual({
      "arxiv-daily/daily/2026-06-11.md": valid,
    });
    await store.setStatus("2606.12345", "saved");
    const saved = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(saved.schemaVersion).toBe(5);
    expect(saved.papers["arxiv:2606.12345"].noveltyByReport).toEqual({
      "arxiv-daily/daily/2026-06-11.md": valid,
    });
  });

  it("preserves novelty by report across repeated upserts", async () => {
    const { store } = makeStore();
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345", title: "A paper", authors: "A", date: "2026-06-11",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: false,
      dailyReport: "arxiv-daily/daily/2026-06-11.md",
    });
    const novelty = {
      differenceType: "new-method",
      comparisonBasis: ["arxiv:2501.00001"],
      evidenceDepth: "metadata-and-abstract" as const,
      explanation: "Persisted explanation.",
    };
    await store.reconcileDailyReportOccurrenceNovelty(
      "arxiv-daily/daily/2026-06-11.md",
      [{ arxivId: "2606.12345", novelty }],
    );
    const { entry, wasNew } = await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "Updated title",
      authors: "B",
      date: "2026-06-12",
      arxivCategory: "astro-ph",
      primaryTopic: "galaxy-cluster",
      detail: true,
    });
    expect(wasNew).toBe(false);
    expect(entry.noveltyByReport).toEqual({
      "arxiv-daily/daily/2026-06-11.md": novelty,
    });
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

  it("does not expose first-save content before primary promotion", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const backupPath = `${primaryPath}.bak`;
    const base = makeStorage();
    let reachPromotion!: () => void;
    const promotionReached = new Promise<void>((resolve) => {
      reachPromotion = resolve;
    });
    let releasePromotion!: () => void;
    const promotionRelease = new Promise<void>((resolve) => {
      releasePromotion = resolve;
    });
    const storage = {
      ...base.storage,
      async rename(from: string, to: string) {
        if (from === `${primaryPath}.tmp` && to === primaryPath) {
          reachPromotion();
          await promotionRelease;
          throw new Error("primary promotion blocked");
        }
        await base.storage.rename(from, to);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);
    const observer = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

    const save = store.upsertFromDailyPaper({
      arxivId: "2606.10024",
      title: "Uncommitted first save",
      authors: "A. Author",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
    });
    await promotionReached;
    const backupAtCommitBarrier = base.files[backupPath];
    const observedAtCommitBarrier = await observer.load();
    releasePromotion();

    await expect(save).rejects.toMatchObject({
      name: "PaperIndexError",
      cause: expect.objectContaining({ message: "primary promotion blocked" }),
    });
    expect(backupAtCommitBarrier).toBeUndefined();
    expect(observedAtCommitBarrier.papers["arxiv:2606.10024"]).toBeUndefined();
    expect(base.files[backupPath]).toBeUndefined();
    await expect(observer.load()).resolves.toMatchObject({ papers: {} });
  });

  it("keeps first-save promotion failure primary when rollback cleanup fails", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const backupPath = `${primaryPath}.bak`;
    const base = makeStorage();
    let promotionFailed = false;
    const failedCleanupPaths: string[] = [];
    const storage = {
      ...base.storage,
      async rename(from: string, to: string) {
        if (from === `${primaryPath}.tmp` && to === primaryPath) {
          promotionFailed = true;
          throw new Error("original primary promotion failed");
        }
        await base.storage.rename(from, to);
      },
      async remove(path: string) {
        if (promotionFailed) {
          failedCleanupPaths.push(path);
          throw new Error("rollback cleanup failed");
        }
        await base.storage.remove(path);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.upsertFromDailyPaper({
      arxivId: "2606.10025",
      title: "Failed first save",
      authors: "A. Author",
      date: "2026-06-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
    })).rejects.toMatchObject({
      name: "PaperIndexError",
      cause: expect.objectContaining({ message: "original primary promotion failed" }),
    });
    expect(failedCleanupPaths).toContain(`${primaryPath}.tmp`);
    expect(base.files[backupPath]).toBeUndefined();
    await expect(store.load()).resolves.toMatchObject({ papers: {} });
  });

  it("mutates backup-derived state without losing the valid recovery copy", async () => {
    const backup = paperIndexJson("2606.10009", "Backup state");
    const { files, store } = makeStore({
      "arxiv-daily/.index/papers.json": "{corrupt",
      "arxiv-daily/.index/papers.json.bak": backup,
    });

    await store.setStatus("2606.10009", "saved");

    expect(JSON.parse(files["arxiv-daily/.index/papers.json"]).papers[
      "arxiv:2606.10009"
    ].status).toBe("saved");
    expect(JSON.parse(files["arxiv-daily/.index/papers.json.bak"]).papers[
      "arxiv:2606.10009"
    ].status).toBe("inbox");
  });

  it("restores valid recovery content when primary promotion fails", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const backupPath = `${primaryPath}.bak`;
    const base = makeStorage({
      [primaryPath]: paperIndexJson("2606.10010", "Before promotion"),
    });
    let failPromotion = true;
    const storage = {
      ...base.storage,
      async rename(from: string, to: string) {
        if (failPromotion && from === `${primaryPath}.tmp` && to === primaryPath) {
          failPromotion = false;
          throw new Error("promotion failed");
        }
        await base.storage.rename(from, to);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.setStatus("2606.10010", "saved")).rejects.toThrow(
      /failed to save paper index/,
    );

    expect(JSON.parse(base.files[primaryPath]).papers["arxiv:2606.10010"].status)
      .toBe("inbox");
    expect(JSON.parse(base.files[backupPath]).papers["arxiv:2606.10010"].status)
      .toBe("inbox");
  });

  it("keeps the valid primary when publishing its replacement backup fails", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const backupPath = `${primaryPath}.bak`;
    const base = makeStorage({
      [primaryPath]: paperIndexJson("2606.10011", "Current primary"),
      [backupPath]: paperIndexJson("2606.10012", "Older backup"),
    });
    const storage = {
      ...base.storage,
      async rename(from: string, to: string) {
        if (from === `${backupPath}.tmp` && to === backupPath) {
          throw new Error("backup promotion failed");
        }
        await base.storage.rename(from, to);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.setStatus("2606.10011", "saved")).rejects.toThrow(
      /failed to save paper index/,
    );

    expect(JSON.parse(base.files[primaryPath]).papers["arxiv:2606.10011"].status)
      .toBe("inbox");
    expect(JSON.parse(base.files[backupPath]).papers["arxiv:2606.10012"].title)
      .toBe("Older backup");
  });

  it("treats temp exists cleanup failure after primary promotion as best effort", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const primaryTmp = `${primaryPath}.tmp`;
    const base = makeStorage({
      [primaryPath]: paperIndexJson("2606.10020", "Committed cleanup state"),
    });
    const operations: string[] = [];
    let promotionComplete = false;
    const storage = {
      ...base.storage,
      async exists(path: string) {
        operations.push(`exists:${path}`);
        if (promotionComplete && path === primaryTmp) {
          throw new Error("post-commit temp exists failed");
        }
        return base.storage.exists(path);
      },
      async rename(from: string, to: string) {
        operations.push(`rename:${from}->${to}`);
        await base.storage.rename(from, to);
        if (from === primaryTmp && to === primaryPath) promotionComplete = true;
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.setStatus("2606.10020", "saved")).resolves.toMatchObject({
      status: "saved",
    });
    await expect(store.load()).resolves.toMatchObject({
      papers: { "arxiv:2606.10020": { status: "saved" } },
    });
    expect(operations).toContain(`rename:${primaryTmp}->${primaryPath}`);
  });

  it("treats temp remove cleanup failure after primary promotion as best effort", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const primaryTmp = `${primaryPath}.tmp`;
    const base = makeStorage({
      [primaryPath]: paperIndexJson("2606.10021", "Committed temp removal"),
    });
    const operations: string[] = [];
    let promotedContent = "";
    let promotionComplete = false;
    const storage = {
      ...base.storage,
      async rename(from: string, to: string) {
        operations.push(`rename:${from}->${to}`);
        if (from === primaryTmp && to === primaryPath) {
          promotedContent = base.files[from];
        }
        await base.storage.rename(from, to);
        if (from === primaryTmp && to === primaryPath) {
          promotionComplete = true;
          base.files[primaryTmp] = promotedContent;
        }
      },
      async remove(path: string) {
        operations.push(`remove:${path}`);
        if (promotionComplete && path === primaryTmp) {
          throw new Error("post-commit temp remove failed");
        }
        await base.storage.remove(path);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.setStatus("2606.10021", "saved")).resolves.toMatchObject({
      status: "saved",
    });
    await expect(store.load()).resolves.toMatchObject({
      papers: { "arxiv:2606.10021": { status: "saved" } },
    });
    expect(operations).toContain(`remove:${primaryTmp}`);
  });

  it.each(["exists", "remove"] as const)(
    "treats legacy %s cleanup failure after commit as best effort",
    async (failure) => {
      const primaryPath = "arxiv-daily/.index/papers.json";
      const legacyPath = "arxiv-daily/index/papers.json";
      const base = makeStorage({
        [primaryPath]: paperIndexJson("2606.10022", "Committed legacy cleanup"),
        [legacyPath]: paperIndexJson("2606.19999", "Stale legacy"),
      });
      const operations: string[] = [];
      let promotionComplete = false;
      const storage = {
        ...base.storage,
        async exists(path: string) {
          operations.push(`exists:${path}`);
          if (failure === "exists" && promotionComplete && path === legacyPath) {
            throw new Error("legacy exists failed");
          }
          return base.storage.exists(path);
        },
        async remove(path: string) {
          operations.push(`remove:${path}`);
          if (failure === "remove" && promotionComplete && path === legacyPath) {
            throw new Error("legacy remove failed");
          }
          await base.storage.remove(path);
        },
        async rename(from: string, to: string) {
          operations.push(`rename:${from}->${to}`);
          await base.storage.rename(from, to);
          if (from === `${primaryPath}.tmp` && to === primaryPath) {
            promotionComplete = true;
          }
        },
      } satisfies StorageAdapter;
      const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

      await expect(store.setStatus("2606.10022", "saved")).resolves.toMatchObject({
        status: "saved",
      });
      await expect(store.load()).resolves.toMatchObject({
        papers: { "arxiv:2606.10022": { status: "saved" } },
      });
      expect(operations).toContain(`rename:${primaryPath}.tmp->${primaryPath}`);
    },
  );

  it("does not let final cleanup failure replace the original promotion error", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const primaryTmp = `${primaryPath}.tmp`;
    const base = makeStorage({
      [primaryPath]: paperIndexJson("2606.10023", "Original promotion error"),
    });
    const operations: string[] = [];
    let primaryTmpExistsCalls = 0;
    let failPromotion = true;
    const storage = {
      ...base.storage,
      async exists(path: string) {
        operations.push(`exists:${path}`);
        if (path === primaryTmp) {
          primaryTmpExistsCalls += 1;
          if (primaryTmpExistsCalls === 3) {
            throw new Error("final cleanup failed");
          }
        }
        return base.storage.exists(path);
      },
      async rename(from: string, to: string) {
        operations.push(`rename:${from}->${to}`);
        if (failPromotion && from === primaryTmp && to === primaryPath) {
          failPromotion = false;
          throw new Error("original promotion failed");
        }
        await base.storage.rename(from, to);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.setStatus("2606.10023", "saved")).rejects.toMatchObject({
      name: "PaperIndexError",
      cause: expect.objectContaining({ message: "original promotion failed" }),
    });
    await expect(store.load()).resolves.toMatchObject({
      papers: { "arxiv:2606.10023": { status: "inbox" } },
    });
    expect(operations.filter((item) => item === `exists:${primaryTmp}`)).toHaveLength(3);
  });

  it("continues queued mutations after a failed job", async () => {
    const primaryPath = "arxiv-daily/.index/papers.json";
    const base = makeStorage({
      [primaryPath]: paperIndexJson("2606.10013", "Queue recovery"),
    });
    let failWrite = true;
    const storage = {
      ...base.storage,
      async writeText(path: string, content: string) {
        if (failWrite && path === `${primaryPath}.tmp`) {
          failWrite = false;
          throw new Error("one-shot write failure");
        }
        await base.storage.writeText(path, content);
      },
    } satisfies StorageAdapter;
    const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);

    await expect(store.setStatus("2606.10013", "saved")).rejects.toThrow(
      "one-shot write failure",
    );
    await expect(store.setPriority("2606.10013", "high")).resolves.toMatchObject({
      priority: "high",
    });

    await expect(store.get("2606.10013")).resolves.toMatchObject({
      status: "inbox",
      priority: "high",
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
