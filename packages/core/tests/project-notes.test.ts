import { describe, expect, it } from "vitest";
import { ProjectNotesService } from "../src/services/project-notes";
import { PaperIndexStore, type PaperIndexEntry } from "../src/services/paper-index";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { Logger } from "../src/services/logger";
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
  return { files, dirs, storage };
}

function entry(overrides: Partial<PaperIndexEntry> = {}): PaperIndexEntry {
  return {
    paperKey: "arxiv:2606.12345",
    source: "arxiv",
    externalId: "2606.12345",
    arxivId: "2606.12345",
    title: "A project paper",
    authors: ["A"],
    published: "2026-06-13",
    updated: "2026-06-13",
    category: "astro-ph",
    categories: ["astro-ph"],
    topics: ["photo-z"],
    primaryTopic: "photo-z",
    detail: false,
    status: "saved",
    priority: "normal",
    seenDates: ["2026-06-13"],
    dailyReports: [],
    paperPath: "arxiv-daily/papers/2606.12345.md",
    arxivUrl: "https://arxiv.org/abs/2606.12345",
    pdfUrl: "https://arxiv.org/pdf/2606.12345",
    pdfPath: "",
    zoteroKey: "",
    zoteroUri: "",
    citationKey: "",
    projects: [],
    ...overrides,
  };
}

describe("ProjectNotesService", () => {
  it("appends paper links to project notes and records projects once", async () => {
    const { files, storage } = makeStorage();
    const store = new PaperIndexStore(
      storage,
      DEFAULT_SETTINGS.output,
      () => new Date("2026-06-13T00:00:00.000Z"),
    );
    await store.upsertFromDailyPaper({
      arxivId: "2606.12345",
      title: "A project paper",
      authors: "A",
      date: "2026-06-13",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
      paperPath: "arxiv-daily/papers/2606.12345.md",
    });
    const service = new ProjectNotesService({
      storage,
      paperIndex: store,
      output: DEFAULT_SETTINGS.output,
      logger: new Logger("error"),
    });

    const first = await service.addPaperToProject(entry(), "Projects/photo-z");
    const second = await service.addPaperToProject(entry(), "Projects/photo-z.md");

    expect(first).toMatchObject({
      kind: "done",
      projectPath: "Projects/photo-z.md",
      appended: true,
      entryUpdated: true,
    });
    expect(second).toMatchObject({
      kind: "done",
      appended: false,
      entryUpdated: true,
    });
    expect(files["Projects/photo-z.md"]).toContain("# photo z");
    expect(files["Projects/photo-z.md"]).toContain(
      "- [[arxiv-daily/papers/2606.12345|2606.12345]] — A project paper <!-- arxiv-daily-project:2606.12345 -->",
    );
    expect(files["Projects/photo-z.md"].match(/arxiv-daily-project/g)).toHaveLength(1);
    const saved = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(saved.papers["arxiv:2606.12345"].projects).toEqual(["Projects/photo-z.md"]);
  });

  it("uses relative markdown links when configured", async () => {
    const { files, storage } = makeStorage();
    const service = new ProjectNotesService({
      storage,
      output: { ...DEFAULT_SETTINGS.output, linkStyle: "relative" },
      logger: new Logger("error"),
    });

    await service.addPaperToProject(entry(), "Projects/photo-z.md");

    expect(files["Projects/photo-z.md"]).toContain(
      "[2606.12345](../arxiv-daily/papers/2606.12345.md)",
    );
  });
});
