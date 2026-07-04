import { describe, it, expect, vi } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { ArxivPipeline } from "../src/pipeline/pipeline";
import { parseRecent } from "../src/pipeline/arxiv-parser";
import { RunCancelledError } from "../src/services/cancellation";
import { Logger } from "../src/services/logger";
import { PaperIndexStore } from "../src/services/paper-index";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { StorageAdapter } from "../src/core/adapters";

vi.mock("obsidian", () => ({
  Notice: class {
    constructor() {}
  },
  normalizePath: (p: string) => p,
  requestUrl: vi.fn(),
}));

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(
  resolve(here, "fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

const testArxiv = {
  ...DEFAULT_SETTINGS.arxiv,
  topics: [
    { id: "t1", name: "Photo-z", tag: "photo-z", description: "photo-z methods", detail: true },
    { id: "t2", name: "Galaxy Cluster", tag: "galaxy-cluster", description: "cluster surveys", detail: true },
  ],
};

function atomMeta(
  id: string,
  overrides: Partial<ReturnType<typeof baseAtomMeta>> = {},
) {
  return { ...baseAtomMeta(id), ...overrides };
}

function baseAtomMeta(id: string) {
  return {
    id,
    title: `Atom title ${id}`,
    authors: "Atom Author et al.",
    abstract: "atom abstract",
    published: "2026-02-02T02:28:06Z",
    updated: "2026-06-15T02:34:08Z",
    primaryCategory: "astro-ph.GA",
    categories: ["astro-ph.GA"],
  };
}

function makeDeps() {
  const writes: Record<string, string> = {};
  const fetcher = {
    fetchRecent: vi.fn().mockResolvedValue(recentHtml),
    fetchAbstractsByIds: vi.fn().mockResolvedValue(new Map<string, string>()),
    fetchMetadataByIds: vi.fn(async (ids: string[]) =>
      new Map(ids.map((id) => [id, atomMeta(id)])),
    ),
    fetchBySubmittedDate: vi.fn().mockResolvedValue([]),
    fetchPaperHtml: vi.fn().mockResolvedValue({ ok: false, status: 404 }),
    fetchPaperAbsPage: vi
      .fn()
      .mockResolvedValue(
        `<html><body><blockquote class="abstract">Abstract: stub abstract</blockquote></body></html>`,
      ),
  };
  const paperFetcher = {
    fetch: vi
      .fn()
      .mockResolvedValue({ abstractConclusion: "## Abstract\nstub", fullSections: null }),
  };
  const writer = {
    writeDaily: vi.fn(async (date: string, content: string, _options?: any) => {
      writes[`daily/${date}.md`] = content;
      return `daily/${date}.md`;
    }),
    writePaperDetail: vi.fn(async (p: any, date: string, content: string) => {
      writes[`papers/${p.id}.md`] = content;
      return `papers/${p.id}.md`;
    }),
    writeEmptyDaily: vi.fn(async (date: string, _options?: any) => {
      writes[`daily/${date}.md`] = "empty";
      return `daily/${date}.md`;
    }),
    dailyPath: vi.fn((date: string) => `daily/${date}.md`),
    paperDetailPath: vi.fn((id: string) => `papers/${id}.md`),
    paperDetailLink: vi.fn((id: string) => `[[${id}]]`),
    dailyExists: vi.fn(async () => false),
    paperDetailExists: vi.fn(async () => false),
  };
  const llm = {
    call: vi.fn().mockResolvedValueOnce(JSON.stringify({ papers: [] })),
  };
  const logger = new Logger("error");
  return { writes, fetcher, paperFetcher, writer, llm, logger };
}

function makePaperIndex() {
  const files: Record<string, string> = {};
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
  const store = new PaperIndexStore(
    storage,
    DEFAULT_SETTINGS.output,
    () => new Date("2026-06-11T01:30:00.000Z"),
  );
  return { files, store };
}

function firstDateFromFixture(): string {
  const m = /(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})/.exec(recentHtml)!;
  const months: Record<string, number> = {
    January: 1, February: 2, March: 3, April: 4, May: 5, June: 6,
    July: 7, August: 8, September: 9, October: 10, November: 11, December: 12,
  };
  return `${m[3]}-${String(months[m[2]]).padStart(2, "0")}-${String(Number(m[1])).padStart(2, "0")}`;
}

function firstBucketPapersFromFixture() {
  const date = firstDateFromFixture();
  const bucket = parseRecent(recentHtml).find((b) => b.announceDate === date);
  if (!bucket) throw new Error(`fixture bucket not found: ${date}`);
  return bucket.papers;
}

describe("ArxivPipeline", () => {
  it("returns cancelled without fetching when the signal is already cancelled", async () => {
    const d = makeDeps();
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const controller = new AbortController();
    controller.abort("cancelled by test");
    const result = await pipeline.runForDate(firstDateFromFixture(), controller.signal);
    expect(result.kind).toBe("cancelled");
    expect((result as any).reason).toBe("cancelled by test");
    expect(d.fetcher.fetchRecent).not.toHaveBeenCalled();
    expect(d.writer.writeDaily).not.toHaveBeenCalled();
  });

  it("passes the abort signal to recent and metadata fetches", async () => {
    const d = makeDeps();
    const controller = new AbortController();
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    await pipeline.runForDate(firstDateFromFixture(), controller.signal);

    expect(d.fetcher.fetchRecent).toHaveBeenCalledWith("astro-ph", controller.signal);
    expect(d.fetcher.fetchMetadataByIds).toHaveBeenCalledWith(
      expect.any(Array),
      controller.signal,
    );
  });

  it("returns failed_transient when /recent misses the date", async () => {
    const d = makeDeps();
    d.fetcher.fetchBySubmittedDate = vi.fn().mockResolvedValue([]);
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const result = await pipeline.runForDate("1999-01-01");
    expect(result.kind).toBe("failed_transient");
    expect((result as any).reason).toContain("not in astro-ph /recent");
    expect(d.fetcher.fetchBySubmittedDate).not.toHaveBeenCalled();
  });

  it("keeps newer-than-recent announce dates retryable without submittedDate fallback", async () => {
    const d = makeDeps();
    d.fetcher.fetchRecent = vi
      .fn()
      .mockResolvedValue(
        `<html><body><dl id="articles"><h3>Wed, 10 Jun 2026</h3></dl></body></html>`,
      );
    d.fetcher.fetchBySubmittedDate = vi.fn().mockResolvedValue([]);
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate("2026-06-13");

    expect(result.kind).toBe("failed_transient");
    expect((result as any).reason).toContain("newer than newest");
    expect(d.fetcher.fetchBySubmittedDate).not.toHaveBeenCalled();
    expect(d.writer.writeDaily).not.toHaveBeenCalled();
  });

  it("keeps missing dates inside the recent window retryable without submittedDate fallback", async () => {
    const d = makeDeps();
    d.fetcher.fetchRecent = vi.fn().mockResolvedValue(
      [
        `<dl id="articles"><h3>Fri, 12 Jun 2026</h3></dl>`,
        `<dl id="articles"><h3>Wed, 10 Jun 2026</h3></dl>`,
      ].join("\n"),
    );
    d.fetcher.fetchBySubmittedDate = vi.fn().mockResolvedValue([]);
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate("2026-06-11");

    expect(result.kind).toBe("failed_transient");
    expect((result as any).reason).toContain("not in astro-ph /recent");
    expect(d.fetcher.fetchBySubmittedDate).not.toHaveBeenCalled();
    expect(d.writer.writeDaily).not.toHaveBeenCalled();
  });

  it("returns completed with 0 papers when LLM returns no relevant papers", async () => {
    const d = makeDeps();
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const date = firstDateFromFixture();
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect((result as any).papersWritten).toBe(0);
    // Should NOT write empty file - calendar shows "0" instead
    expect(d.writer.writeEmptyDaily).not.toHaveBeenCalled();
  });

  it("returns failed_transient when the filter LLM call fails", async () => {
    const d = makeDeps();
    d.llm.call = vi.fn().mockRejectedValue(new Error("api unavailable"));
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate(firstDateFromFixture());

    expect(result).toEqual({
      kind: "failed_transient",
      reason: "paper filter LLM failed: api unavailable",
    });
    expect(d.writer.writeDaily).not.toHaveBeenCalled();
  });

  it("returns failed_permanent when the filter LLM call fails with a non-429 4xx", async () => {
    const d = makeDeps();
    d.llm.call = vi.fn().mockRejectedValue(
      Object.assign(new Error("Unauthorized"), { status: 401 }),
    );
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate(firstDateFromFixture());

    expect(result).toEqual({
      kind: "failed_permanent",
      reason: "paper filter LLM failed: Unauthorized",
    });
    expect(d.writer.writeDaily).not.toHaveBeenCalled();
  });

  it("continues with papers from successful categories when another category fetch fails", async () => {
    const d = makeDeps();
    d.fetcher.fetchRecent = vi.fn(async (category: string) => {
      if (category === "astro-ph") throw new Error("network down");
      return recentHtml;
    });
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: {
        ...testArxiv,
        category: "astro-ph",
        categories: ["astro-ph", "cs.CL"],
      },
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate(firstDateFromFixture());

    expect(result.kind).toBe("completed");
    expect(d.fetcher.fetchRecent).toHaveBeenCalledWith("astro-ph");
    expect(d.fetcher.fetchRecent).toHaveBeenCalledWith("cs.CL");
  });

  it("enriches abstracts and runs filter+summarize for a kept paper", async () => {
    const d = makeDeps();
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];
    // Override LLM call sequence: filter returns 1 paper, then daily summary returns markdown
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: false }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## Photo-z 相关\n### Stub title\n- summary\n";
      }
      return "";
    });
    d.fetcher.fetchMetadataByIds = vi
      .fn()
      .mockResolvedValue(new Map([[arxivId, atomMeta(arxivId)]]));

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const date = firstDateFromFixture();
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect((result as any).papersWritten).toBe(1);
    expect(d.fetcher.fetchMetadataByIds).toHaveBeenCalled();
    expect(d.paperFetcher.fetch).toHaveBeenCalledWith(
      arxivId,
      expect.objectContaining({ isDetail: true }),
    );
    expect(d.writer.paperDetailLink).toHaveBeenCalledWith(
      arxivId,
      date,
      undefined,
    );
    expect(d.writer.writeDaily).toHaveBeenCalled();
  });

  it("persists non-detail kept papers to the paper index without writing a paper note", async () => {
    const d = makeDeps();
    const { files, store } = makePaperIndex();
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: false }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return [
          "## Photo-z",
          "### Stub",
          "> 信息来源：Abstract",
          `- **arXiv**: [${arxivId}](https://arxiv.org/abs/${arxivId})`,
          "- **核心问题**: Problem.",
          "- **关键方法**: Method.",
          "- **主要结果**: Result.",
          "- **为什么值得看**: Relevant.",
          "- **局限或边界**: Limits.",
        ].join("\n");
      }
      return "";
    });

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: store,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const date = firstDateFromFixture();
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
    const json = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    const entry = json.papers[arxivId];
    expect(entry.status).toBe("inbox");
    expect(entry.priority).toBe("normal");
    expect(entry.paperPath).toBeNull();
    expect(entry.summary).toEqual({
      sourceSections: "Abstract",
      coreProblem: "Problem.",
      keyMethod: "Method.",
      mainResult: "Result.",
      whyRelevant: "Relevant.",
      limitations: "Limits.",
    });
    expect(entry.seenDates).toEqual([date]);
    expect(entry.published).toBe(date);
    expect(entry.updated).toBe("2026-06-15");
    expect(entry.dailyReports).toEqual([`daily/${date}.md`]);
  });

  it("fetches multiple categories and deduplicates papers before filtering", async () => {
    const d = makeDeps();
    const { files, store } = makePaperIndex();
    const papers = firstBucketPapersFromFixture();
    const arxivId = papers[0].id;
    d.fetcher.fetchMetadataByIds = vi.fn(async (ids: string[]) =>
      new Map(
        ids.map((id) => [
          id,
          atomMeta(id, { primaryCategory: "", categories: [] }),
        ]),
      ),
    );
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: false }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## Photo-z\n### Stub\n";
      }
      return "";
    });

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: store,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: {
        ...testArxiv,
        category: "astro-ph",
        categories: ["astro-ph", "cs.LG"],
      },
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate(firstDateFromFixture());

    expect(result.kind).toBe("completed");
    expect(d.fetcher.fetchRecent).toHaveBeenCalledWith("astro-ph");
    expect(d.fetcher.fetchRecent).toHaveBeenCalledWith("cs.LG");
    expect(d.fetcher.fetchMetadataByIds.mock.calls[0][0]).toHaveLength(
      papers.length,
    );
    const filterUserPrompt = d.llm.call.mock.calls[0][0][1].content as string;
    expect(
      filterUserPrompt.match(new RegExp(arxivId.replace(".", "\\."), "g")),
    ).toHaveLength(1);
    const json = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(json.papers[arxivId].category).toBe("astro-ph");
    expect(json.papers[arxivId].categories).toEqual(["astro-ph", "cs.LG"]);
  });

  it("does not use submittedDate export API when date is outside recent", async () => {
    const d = makeDeps();
    d.fetcher.fetchRecent = vi
      .fn()
      .mockResolvedValue(
        `<html><body><dl id="articles"><h3>Wed, 10 Jun 2026</h3></dl></body></html>`,
      );
    d.fetcher.fetchBySubmittedDate = vi.fn().mockResolvedValue([]);

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate("2026-06-09");

    expect(result.kind).toBe("failed_transient");
    expect((result as any).reason).toContain("not in astro-ph /recent");
    expect(d.fetcher.fetchBySubmittedDate).not.toHaveBeenCalled();
    expect(d.writer.writeDaily).not.toHaveBeenCalled();
  });

  it("updates ignored papers in the paper index without including them in the daily body", async () => {
    const d = makeDeps();
    const { files, store } = makePaperIndex();
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];
    const date = firstDateFromFixture();
    await store.upsertFromDailyPaper({
      arxivId,
      title: "Old",
      authors: "A",
      date: "2026-05-01",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
    });
    await store.setStatus(arxivId, "ignored");
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: false }],
        });
      }
      throw new Error("daily summarizer should not be called");
    });

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: store,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect(d.paperFetcher.fetch).not.toHaveBeenCalled();
    // Should NOT write empty file - calendar shows "0" instead
    expect(d.writer.writeEmptyDaily).not.toHaveBeenCalled();
    const json = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(json.papers[arxivId].status).toBe("ignored");
    expect(json.papers[arxivId].seenDates).toContain(date);
  });

  it("short-circuits with completed when daily file already exists", async () => {
    const d = makeDeps();
    (d.writer as any).dailyExists = vi.fn().mockResolvedValue(true);
    (d.writer as any).paperDetailExists = vi.fn().mockResolvedValue(false);

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const result = await pipeline.runForDate("2026-05-11");
    expect(result.kind).toBe("completed");
    expect((result as any).papersWritten).toBe(0);
    expect(d.fetcher.fetchRecent).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
  });

  it("skips paper detail when paper file already exists", async () => {
    const d = makeDeps();

    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];

    (d.writer as any).paperDetailExists = vi.fn(async (id: string) =>
      id === arxivId,
    );
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: true }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## stub daily summary\n";
      }
      return "## detail summary\n";
    });
    d.fetcher.fetchMetadataByIds = vi
      .fn()
      .mockResolvedValue(new Map([[arxivId, atomMeta(arxivId)]]));
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub",
      fullSections: "## Section\nbody",
    });

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const date = firstDateFromFixture();
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect(d.writer.writeDaily).toHaveBeenCalled();
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
  });

  it("does not leave a daily file that makes retry skip unfinished details", async () => {
    const d = makeDeps();
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];
    let dailyExists = false;
    let failFirstDetail = true;
    d.writer.dailyExists = vi.fn(async () => dailyExists);
    d.writer.writeDaily = vi.fn(async (date: string, content: string) => {
      dailyExists = true;
      d.writes[`daily/${date}.md`] = content;
      return `daily/${date}.md`;
    });
    d.writer.writePaperDetail = vi.fn(async (p: any, date: string, content: string) => {
      if (failFirstDetail) {
        failFirstDetail = false;
        throw new RunCancelledError("cancelled during detail");
      }
      d.writes[`papers/${p.id}.md`] = content;
      return `papers/${p.id}.md`;
    });
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: true }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## stub daily summary\n";
      }
      return "## detail summary\n";
    });
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub",
      fullSections: "## Section\nbody",
    });
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const date = firstDateFromFixture();

    const first = await pipeline.runForDate(date);
    const second = await pipeline.runForDate(date);

    expect(first).toEqual({
      kind: "cancelled",
      reason: "cancelled during detail",
    });
    expect(second.kind).toBe("completed");
    expect(d.writer.writePaperDetail).toHaveBeenCalledTimes(2);
    expect(d.writer.writeDaily).toHaveBeenCalledTimes(1);
  });

  it("writes detail reports and stores paperPath in the paper index", async () => {
    const d = makeDeps();
    const { files, store } = makePaperIndex();
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: true }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## stub daily summary\n";
      }
      return "## detail summary\n";
    });
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub",
      fullSections: "## Section\nbody",
    });

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: store,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    await pipeline.runForDate(firstDateFromFixture());
    expect(d.writer.writePaperDetail).toHaveBeenCalledTimes(1);
    const json = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(json.papers[arxivId].paperPath).toBe(`papers/${arxivId}.md`);
  });

  it("emits progress stages in order", async () => {
    const d = makeDeps();
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: false }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## stub\n";
      }
      return "";
    });
    d.fetcher.fetchMetadataByIds = vi
      .fn()
      .mockResolvedValue(new Map([[arxivId, atomMeta(arxivId)]]));

    const calls: Array<[string, number?, number?]> = [];
    const progress = {
      setBatch: vi.fn(),
      setStage: vi.fn((stage: string, current?: number, total?: number) =>
        calls.push([stage, current, total]),
      ),
      setIdle: vi.fn(),
      setDisabled: vi.fn(),
    };

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      progress: progress as any,
    });
    await pipeline.runForDate(firstDateFromFixture());

    const stages = calls.map((c) => c[0]);
    expect(stages).toContain("fetch-recent");
    expect(stages).toContain("enrich-abstract");
    expect(stages).toContain("filter");
    expect(stages).toContain("fetch-content");
    expect(stages).toContain("summarize-daily");
  });
});
