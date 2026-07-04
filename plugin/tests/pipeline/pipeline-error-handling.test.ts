import { describe, it, expect, vi } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import type { PipelineResult } from "../../src/pipeline/pipeline";
import { ArxivPipeline } from "../../src/pipeline/pipeline";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";

vi.mock("obsidian", () => ({
  Notice: class {
    constructor() {}
  },
  normalizePath: (p: string) => p,
  requestUrl: vi.fn(),
}));

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(
  resolve(here, "../fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

const testArxiv = {
  ...DEFAULT_SETTINGS.arxiv,
  topics: [
    { id: "t1", name: "Test Topic", tag: "test", description: "test topic", detail: true },
  ],
};

/** HTML that parses to a date bucket with 0 papers */
const emptyPapersHtml = `<html><body>
<dl id="articles">
  <h3>Mon, 22 Jun 2026</h3>
</dl>
</body></html>`;

describe("PipelineResult types", () => {
  it("should support pending result kind", () => {
    const result: PipelineResult = { kind: "pending", reason: "no papers from arXiv" };
    expect(result.kind).toBe("pending");
    expect(result.reason).toBe("no papers from arXiv");
  });
});

describe("Pipeline arXiv 0 papers handling", () => {
  it("should return pending when arXiv returns 0 papers", async () => {
    const fetcher = {
      fetchRecent: vi.fn().mockResolvedValue(emptyPapersHtml),
      fetchMetadataByIds: vi.fn(),
      fetchAbstractsByIds: vi.fn(),
      fetchBySubmittedDate: vi.fn(),
      fetchPaperHtml: vi.fn(),
      fetchPaperAbsPage: vi.fn(),
    };
    const writer = {
      writeDaily: vi.fn(),
      writePaperDetail: vi.fn(),
      writeEmptyDaily: vi.fn(),
      dailyPath: vi.fn(),
      paperDetailPath: vi.fn(),
      paperDetailLink: vi.fn(),
      dailyExists: vi.fn(async () => false),
      paperDetailExists: vi.fn(async () => false),
    };
    const llm = { call: vi.fn() };
    const paperFetcher = { fetch: vi.fn() };
    const logger = new Logger("error");

    const pipeline = new ArxivPipeline({
      fetcher: fetcher as any,
      paperFetcher: paperFetcher as any,
      writer: writer as any,
      llm: llm as any,
      logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate("2026-06-22");
    expect(result).toEqual({ kind: "pending", reason: "no papers from arXiv" });
    expect(writer.writeEmptyDaily).not.toHaveBeenCalled();
  });
});

describe("Pipeline LLM 0 papers handling", () => {
  it("should return completed with 0 papers when LLM filtering results in 0 papers", async () => {
    const fetcher = {
      fetchRecent: vi.fn().mockResolvedValue(recentHtml),
      fetchMetadataByIds: vi.fn(async (ids: string[]) =>
        new Map(ids.map((id) => [{
          id,
          title: `Title ${id}`,
          authors: "Author et al.",
          abstract: "abstract",
          published: "2026-05-11T02:28:06Z",
          updated: "2026-05-11T02:34:08Z",
          primaryCategory: "astro-ph.GA",
          categories: ["astro-ph.GA"],
        }]).map(([id, meta]) => [id, meta])),
      ),
      fetchAbstractsByIds: vi.fn(),
      fetchBySubmittedDate: vi.fn(),
      fetchPaperHtml: vi.fn(),
      fetchPaperAbsPage: vi.fn(),
    };
    const writer = {
      writeDaily: vi.fn(),
      writePaperDetail: vi.fn(),
      writeEmptyDaily: vi.fn(),
      dailyPath: vi.fn(),
      paperDetailPath: vi.fn(),
      paperDetailLink: vi.fn(),
      dailyExists: vi.fn(async () => false),
      paperDetailExists: vi.fn(async () => false),
    };
    // LLM returns empty papers list to simulate all papers filtered out
    const llm = {
      call: vi.fn().mockResolvedValue(JSON.stringify({ papers: [] })),
    };
    const paperFetcher = { fetch: vi.fn() };
    const logger = new Logger("error");

    const pipeline = new ArxivPipeline({
      fetcher: fetcher as any,
      paperFetcher: paperFetcher as any,
      writer: writer as any,
      llm: llm as any,
      logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate("2026-05-11");
    // Should return completed with 0 papers
    expect(result.kind).toBe("completed");
    if (result.kind === "completed") {
      expect(result.papersWritten).toBe(0);
    }
    // Should NOT write empty file
    expect(writer.writeEmptyDaily).not.toHaveBeenCalled();
  });
});

describe("Pipeline index 0 papers handling", () => {
  it("should return completed with 0 papers when all papers are ignored in index", async () => {
    const fetcher = {
      fetchRecent: vi.fn().mockResolvedValue(recentHtml),
      fetchMetadataByIds: vi.fn(async (ids: string[]) =>
        new Map(ids.map((id) => [{
          id,
          title: `Title ${id}`,
          authors: "Author et al.",
          abstract: "abstract",
          published: "2026-05-11T02:28:06Z",
          updated: "2026-05-11T02:34:08Z",
          primaryCategory: "astro-ph.GA",
          categories: ["astro-ph.GA"],
        }]).map(([id, meta]) => [id, meta])),
      ),
      fetchAbstractsByIds: vi.fn(),
      fetchBySubmittedDate: vi.fn(),
      fetchPaperHtml: vi.fn(),
      fetchPaperAbsPage: vi.fn(),
    };
    const writer = {
      writeDaily: vi.fn(),
      writePaperDetail: vi.fn(),
      writeEmptyDaily: vi.fn(),
      dailyPath: vi.fn(),
      paperDetailPath: vi.fn(),
      paperDetailLink: vi.fn(),
      dailyExists: vi.fn(async () => false),
      paperDetailExists: vi.fn(async () => false),
    };
    // LLM returns 1 paper so it passes filtering, but we'll set it to ignored in the index
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({
          papers: [
            { id: "2605.00001", category: "photo-z", detail: false },
          ],
        }),
      ),
    };
    const paperFetcher = { fetch: vi.fn() };
    const logger = new Logger("error");

    // Create a paper index that returns all papers as ignored
    const paperIndex = {
      upsertManyFromDailyPapers: vi.fn().mockResolvedValue([
        {
          entry: { status: "ignored" },
          wasNew: false,
        },
      ]),
      addDailyReports: vi.fn(),
      setSummaries: vi.fn(),
      setPaperPath: vi.fn(),
    };

    const pipeline = new ArxivPipeline({
      fetcher: fetcher as any,
      paperFetcher: paperFetcher as any,
      writer: writer as any,
      paperIndex: paperIndex as any,
      llm: llm as any,
      logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate("2026-05-11");
    // Should return completed with 0 papers
    expect(result.kind).toBe("completed");
    if (result.kind === "completed") {
      expect(result.papersWritten).toBe(0);
    }
    // Should NOT write empty file
    expect(writer.writeEmptyDaily).not.toHaveBeenCalled();
  });
});

describe("Pipeline partial failure consistency", () => {
  function makeOnePaperPipeline(overrides: {
    ids?: string[];
    writer?: Record<string, unknown>;
    paperIndex?: Record<string, unknown>;
    paperFetcher?: Record<string, unknown>;
  } = {}) {
    const ids = overrides.ids ?? ["2605.08080"];
    const fetcher = {
      fetchRecent: vi.fn().mockResolvedValue(recentHtml),
      fetchMetadataByIds: vi.fn(async (ids: string[]) =>
        new Map(ids.map((id) => [id, {
          id,
          title: `Title ${id}`,
          authors: "Author et al.",
          abstract: "abstract",
          published: "2026-05-11T02:28:06Z",
          updated: "2026-05-11T02:34:08Z",
          primaryCategory: "astro-ph.GA",
          categories: ["astro-ph.GA"],
        }])),
      ),
      fetchAbstractsByIds: vi.fn(),
      fetchBySubmittedDate: vi.fn(),
      fetchPaperHtml: vi.fn(),
      fetchPaperAbsPage: vi.fn(),
    };
    const writer = {
      writeDaily: vi.fn(async () => "arxiv-daily/daily/2026-05-11.md"),
      writePaperDetail: vi.fn(),
      writeEmptyDaily: vi.fn(),
      dailyPath: vi.fn(() => "arxiv-daily/daily/2026-05-11.md"),
      paperDetailPath: vi.fn((id: string) => `arxiv-daily/papers/${id}.md`),
      paperDetailLink: vi.fn((id: string) => `[[${id}]]`),
      dailyExists: vi.fn(async () => false),
      paperDetailExists: vi.fn(async () => false),
      ...overrides.writer,
    };
    const paperIndex = {
      upsertManyFromDailyPapers: vi.fn(async (papers: any[]) =>
        papers.map((paper) => ({
          entry: {
            arxivId: paper.arxivId,
            status: "inbox",
            paperPath: null,
          },
          wasNew: true,
        })),
      ),
      addDailyReports: vi.fn(),
      setSummaries: vi.fn(),
      setPaperPath: vi.fn(),
      ...overrides.paperIndex,
    };
    const llm = {
      call: vi
        .fn()
        .mockResolvedValueOnce(JSON.stringify({
          papers: ids.map((id) => ({ id, category: "test", detail: false })),
        }))
        .mockResolvedValue(
          [
            "## Test Topic",
            ...ids.map((id) => `### [${id}]\n- **研究问题**: test`),
          ].join("\n\n"),
        ),
    };
    const paperFetcher = {
      fetch: vi.fn(async () => ({
        abstractConclusion: "abstract and conclusion",
        fullSections: null,
      })),
      ...overrides.paperFetcher,
    };
    const pipeline = new ArxivPipeline({
      fetcher: fetcher as any,
      paperFetcher: paperFetcher as any,
      writer: writer as any,
      paperIndex: paperIndex as any,
      llm: llm as any,
      logger: new Logger("error"),
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    return { pipeline, writer, paperIndex, paperFetcher };
  }

  it("does not add daily report references when daily write fails", async () => {
    const { pipeline, writer, paperIndex } = makeOnePaperPipeline({
      writer: {
        writeDaily: vi.fn(async () => {
          throw new Error("disk full");
        }),
      },
    });

    await expect(pipeline.runForDate("2026-05-11")).rejects.toThrow("disk full");

    expect(writer.writeDaily).toHaveBeenCalled();
    expect(paperIndex.addDailyReports).not.toHaveBeenCalled();
    expect(paperIndex.setSummaries).not.toHaveBeenCalled();
  });

  it("does not update index when cancellation is requested after daily write", async () => {
    const controller = new AbortController();
    const { pipeline, paperIndex } = makeOnePaperPipeline({
      paperIndex: {
        addDailyReports: vi.fn(),
      },
      writer: {
        writeDaily: vi.fn(async () => {
          controller.abort("cancelled after daily write");
          return "arxiv-daily/daily/2026-05-11.md";
        }),
      },
    });

    const result = await pipeline.runForDate("2026-05-11", controller.signal);

    expect(result).toEqual({
      kind: "failed_transient",
      reason: "cancelled after daily write",
    });
    expect(paperIndex.addDailyReports).not.toHaveBeenCalled();
    expect(paperIndex.setSummaries).not.toHaveBeenCalled();
  });

  it("fetches paper content with bounded concurrency", async () => {
    const ids = [
      "2605.08080",
      "2605.08068",
      "2605.08051",
      "2605.07998",
      "2605.07995",
      "2605.07976",
      "2605.07965",
      "2605.07928",
    ];
    let active = 0;
    let maxActive = 0;
    const { pipeline, paperFetcher } = makeOnePaperPipeline({
      ids,
      paperFetcher: {
        fetch: vi.fn(async () => {
          active += 1;
          maxActive = Math.max(maxActive, active);
          await new Promise((resolve) => setTimeout(resolve, 5));
          active -= 1;
          return {
            abstractConclusion: "abstract and conclusion",
            fullSections: null,
          };
        }),
      },
    });

    const result = await pipeline.runForDate("2026-05-11");

    expect(result).toEqual({ kind: "completed", papersWritten: ids.length });
    expect(paperFetcher.fetch).toHaveBeenCalledTimes(ids.length);
    expect(maxActive).toBeGreaterThan(1);
    expect(maxActive).toBeLessThanOrEqual(6);
  });
});
