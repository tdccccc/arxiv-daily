import { markupParser } from "./markup-parser";
import { describe, it, expect, vi } from "vitest";
import { ArxivPipeline } from "../src/pipeline/pipeline";
import { assembleDailySummary } from "../src/pipeline/daily-summary-assembler";
import { parseRecent } from "../src/pipeline/arxiv-parser";
import { RunCancelledError } from "../src/services/cancellation";
import { Logger } from "../src/services/logger";
import { PaperIndexStore } from "../src/services/paper-index";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { StorageAdapter } from "../src/core/adapters";


const recentHtml = `
  <dl id="articles">
    <h3>Mon, 11 May 2026 (showing 2 of 2 entries)</h3>
    <dt><a title="Abstract" href="/abs/2605.08080">arXiv:2605.08080</a></dt>
    <dd>
      <div class="list-title">Title: First pipeline paper</div>
      <div class="list-authors"><a>First Author</a><a>Second Author</a></div>
    </dd>
    <dt><a title="Abstract" href="/abs/2605.08068">arXiv:2605.08068</a></dt>
    <dd>
      <div class="list-title">Title: Second pipeline paper</div>
      <div class="list-authors"><a>Another Author</a></div>
    </dd>
  </dl>
`;

const testDetailSelection = {
  normalThreshold: 70,
  exceptionalThreshold: 90,
  softLimit: 2,
};

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

function structuredDailyResponse(messages: any[]): string | null {
  const system = messages[0]?.content ?? "";
  if (
    !system.includes("严格 JSON 对象") &&
    !system.includes("strict JSON object")
  ) {
    return null;
  }
  const user = messages[1]?.content ?? "";
  const id = /ID: (\d{4}\.\d{4,5})/.exec(user)?.[1];
  if (!id) throw new Error("daily summary test input is missing an ID");
  return JSON.stringify({
    id,
    coreProblem: `${id} problem`,
    keyMethod: `${id} method`,
    mainResult: `${id} result`,
    whyRelevant: `${id} value`,
    limitations: `${id} limits`,
  });
}

function verifiedDetailMarkdown(id: string): string {
  return [
    "---",
    `arxiv_id: "${id}"`,
    "---",
    "# Existing detail",
    "## Research question",
    "A".repeat(150),
    "## Method",
    "B".repeat(150),
    "## Evidence",
    "C".repeat(150),
    "## Limitations",
    "D".repeat(150),
  ].join("\n");
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
    readDaily: vi.fn(async (date: string) => writes[`daily/${date}.md`] ?? ""),
    paperDetailExists: vi.fn(async () => false),
    readPaperDetail: vi.fn(async (id: string) =>
      writes[`papers/${id}.md`] ?? verifiedDetailMarkdown(id)),
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
  const bucket = parseRecent(recentHtml, markupParser).find((b) => b.announceDate === date);
  if (!bucket) throw new Error(`fixture bucket not found: ${date}`);
  return bucket.papers;
}

describe("ArxivPipeline", () => {
  it("returns cancelled without fetching when the signal is already cancelled", async () => {
    const d = makeDeps();
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    const result = await pipeline.runForDate(firstDateFromFixture());

    expect(result).toEqual({
      kind: "failed_permanent",
      reason: "paper filter LLM failed: Unauthorized",
    });
    expect(d.writer.writeDaily).not.toHaveBeenCalled();
  });

  it("rejects all papers when another configured category fetch fails", async () => {
    const d = makeDeps();
    d.fetcher.fetchRecent = vi.fn(async (category: string) => {
      if (category === "astro-ph") throw new Error("network down");
      return recentHtml;
    });
    const pipeline = new ArxivPipeline({
      markupParser,
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
      detailSelection: testDetailSelection,
    });

    const result = await pipeline.runForDate(firstDateFromFixture());

    expect(result.kind).toBe("failed_transient");
    expect(d.fetcher.fetchRecent).toHaveBeenCalledWith("astro-ph");
    expect(d.fetcher.fetchRecent).toHaveBeenCalledWith("cs.CL");
    expect(d.fetcher.fetchMetadataByIds).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
    expect(d.writer.writeDaily).not.toHaveBeenCalled();
  });

  it("enriches abstracts and runs filter+summarize for a kept paper", async () => {
    const d = makeDeps();
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];
    // Override LLM call sequence: filter returns 1 paper, then daily summary returns markdown
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      const daily = structuredDailyResponse(msgs);
      if (daily) return daily;
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z" }],
        });
      }
      return "";
    });
    d.fetcher.fetchMetadataByIds = vi
      .fn()
      .mockResolvedValue(new Map([[arxivId, atomMeta(arxivId)]]));

    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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

  it("persists canonicalized summaries through daily Markdown and PaperIndex.setSummaries", async () => {
    const d = makeDeps();
    const { files, store } = makePaperIndex();
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];
    const canonicalProblem = String.raw`Constraint $z<0.1$ with $\alpha_i^2$.`;
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      const daily = structuredDailyResponse(msgs);
      if (daily) {
        const response = JSON.parse(daily);
        response.coreProblem = String.raw`Constraint \(z<0.1\) with \(\alpha_i^2\).`;
        return JSON.stringify(response);
      }
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z" }],
        });
      }
      return "";
    });

    const pipeline = new ArxivPipeline({
      markupParser,
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
      detailSelection: testDetailSelection,
    });
    const date = firstDateFromFixture();
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
    const json = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    const entry = json.papers[`arxiv:${arxivId}`];
    expect(entry.status).toBe("inbox");
    expect(entry.priority).toBe("normal");
    expect(entry.paperPath).toBeNull();
    expect(d.writes[`daily/${date}.md`]).toContain(
      `- **研究问题**: ${canonicalProblem}`,
    );
    expect(entry.summary).toEqual({
      sourceSections: "Abstract",
      coreProblem: canonicalProblem,
      keyMethod: `${arxivId} method`,
      mainResult: `${arxivId} result`,
      whyRelevant: `${arxivId} value`,
      limitations: `${arxivId} limits`,
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
      const daily = structuredDailyResponse(msgs);
      if (daily) return daily;
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z" }],
        });
      }
      return "";
    });

    const pipeline = new ArxivPipeline({
      markupParser,
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
      detailSelection: testDetailSelection,
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
    expect(json.papers[`arxiv:${arxivId}`].category).toBe("astro-ph");
    expect(json.papers[`arxiv:${arxivId}`].categories).toEqual(["astro-ph", "cs.LG"]);
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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      const daily = structuredDailyResponse(msgs);
      if (daily) return daily;
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z" }],
        });
      }
      throw new Error("daily summarizer should not be called");
    });

    const pipeline = new ArxivPipeline({
      markupParser,
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
      detailSelection: testDetailSelection,
    });
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect(d.paperFetcher.fetch).not.toHaveBeenCalled();
    // Should NOT write empty file - calendar shows "0" instead
    expect(d.writer.writeEmptyDaily).not.toHaveBeenCalled();
    const json = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(json.papers[`arxiv:${arxivId}`].status).toBe("ignored");
    expect(json.papers[`arxiv:${arxivId}`].seenDates).toContain(date);
  });

  it("repairs daily-report links and summaries when a daily file already exists", async () => {
    const d = makeDeps();
    const id = "2607.00001";
    const markdown = [
      "## Topic",
      "### Existing paper",
      `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
      "- **核心问题**: Repaired problem.",
    ].join("\n");
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue(markdown);
    const paperIndex = {
      reconcilePaperDetails: vi.fn(async () => 0),
      addDailyReports: vi.fn(async () => undefined),
      setSummaries: vi.fn(async () => 1),
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    const result = await pipeline.runForDate("2026-05-11");

    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(paperIndex.reconcilePaperDetails).toHaveBeenCalledWith({ [id]: null });
    expect(paperIndex.addDailyReports).toHaveBeenCalledWith(
      [id],
      "daily/2026-05-11.md",
    );
    expect(paperIndex.setSummaries).toHaveBeenCalledWith({
      [id]: { coreProblem: "Repaired problem." },
    });
    expect(d.fetcher.fetchRecent).not.toHaveBeenCalled();
  });

  it("repairs IDs from standalone legacy controls but ignores inline fake marker prose", async () => {
    const d = makeDeps();
    const watchId = "2607.01001";
    const highlightId = "2607.01002";
    const fakeId = "2607.01999";
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue([
      "### Legacy controls",
      `- [x] Watch <!--  arxiv-daily:${watchId}:watch  -->`,
      `* [X] Highlight <!--\tarxiv-daily:${highlightId}:selection:highlight\t-->`,
      `- **Research problem**: inline fake <!-- arxiv-daily:${fakeId}:watch -->`,
    ].join("\r\n"));
    const paperIndex = {
      reconcilePaperDetails: vi.fn(async () => 0),
      addDailyReports: vi.fn(async () => undefined),
      setSummaries: vi.fn(async () => 0),
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    expect(await pipeline.runForDate("2026-05-11")).toMatchObject({ kind: "completed", papersWritten: 2,
    });
    expect(paperIndex.addDailyReports).toHaveBeenCalledWith(
      [watchId, highlightId],
      "daily/2026-05-11.md",
    );
    expect(paperIndex.reconcilePaperDetails).toHaveBeenCalledWith({
      [watchId]: null,
      [highlightId]: null,
    });
  });

  it("repairs every emergency daily ID without projecting fallback content as a summary", async () => {
    const d = makeDeps();
    const structuredId = "2607.00001";
    const fallbackId = "2607.00002";
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue([
      "<!-- arxiv-daily-emergency-report:v1 -->",
      "## Topic",
      "### Structured paper",
      `- **arXiv**: [${structuredId}](https://arxiv.org/abs/${structuredId})`,
      "- **研究问题**: Repaired structured problem.",
      "### Fallback paper",
      `<!-- arxiv-daily-fallback:${fallbackId} -->`,
      `- **arXiv**: [${fallbackId}](https://arxiv.org/abs/${fallbackId})`,
      "- **原始摘要**: Must not become a generated summary.",
    ].join("\n"));
    const paperIndex = {
      reconcilePaperDetails: vi.fn(async () => 0),
      addDailyReports: vi.fn(async () => undefined),
      setSummaries: vi.fn(async () => 1),
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    expect(await pipeline.runForDate("2026-05-11")).toMatchObject({ kind: "completed", papersWritten: 2,
    });
    expect(paperIndex.addDailyReports).toHaveBeenCalledWith(
      [structuredId, fallbackId],
      "daily/2026-05-11.md",
    );
    expect(paperIndex.setSummaries).toHaveBeenCalledWith({
      [structuredId]: { coreProblem: "Repaired structured problem." },
    });
  });

  it("repairs mixed scientific Markdown without hostile prose changing block identity", async () => {
    const d = makeDeps();
    const structuredId = "2607.10001";
    const fallbackId = "2607.10002";
    const fakeId = "2607.19999";
    const structuredMath = String.raw`$\mathrm{NMAD}$ and $\eta$ at z<0.1 and z>3.5`;
    const fallbackMath = String.raw`\(r_{\rm cut}/R_{\rm vir}\) with M_\odot and \left|x\right|`;
    const markdown = assembleDailySummary({
      dateStr: "2026-05-11",
      arxivSettings: {
        ...testArxiv,
        topics: [testArxiv.topics[0]!],
      },
      summaryLanguage: "en",
      slots: [
        {
          paper: {
            id: structuredId,
            title: "Structured science",
            authors: "A & B",
            category: "photo-z",
            sourceSections: "Abstract",
            isDetail: false,
          },
          result: {
            kind: "structured",
            summary: {
              id: structuredId,
              coreProblem: structuredMath,
              keyMethod: `line one\n### Fake block\n- **arXiv**: [${fakeId}](https://arxiv.org/abs/${fakeId})`,
              mainResult: `inline <!-- arxiv-daily-fallback:${structuredId} --> marker`,
              whyRelevant: "PS1+WISE A & B",
              limitations: "<script>raw tag</script>",
            },
          },
        },
        {
          paper: {
            id: fallbackId,
            title: "Fallback science",
            authors: "C",
            category: "photo-z",
            sourceSections: "Abstract",
            isDetail: false,
          },
          result: {
            kind: "fallback",
            reasonCode: "validation-exhausted",
            attempts: 3,
            originalAbstract: `${fallbackMath}\n### Fake block\n<!-- arxiv-daily-fallback:${fakeId} -->`,
          },
        },
      ],
    });
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue(markdown);
    const paperIndex = {
      reconcilePaperDetails: vi.fn(async () => 0),
      addDailyReports: vi.fn(async () => undefined),
      setSummaries: vi.fn(async () => 1),
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    expect(await pipeline.runForDate("2026-05-11")).toMatchObject({ kind: "completed", papersWritten: 2,
    });
    expect(markdown).toContain(structuredMath);
    expect(markdown).toContain(fallbackMath);
    expect(markdown.match(/^### /gm)).toHaveLength(2);
    expect(markdown.match(new RegExp(`^<!-- arxiv-daily-fallback:${fallbackId} -->$`, "gm"))).toHaveLength(1);
    expect(markdown).not.toMatch(/^### Fake block$/m);
    expect(markdown).not.toContain(`<!-- arxiv-daily-fallback:${fakeId} -->`);
    expect(markdown).not.toContain("<script>");
    expect(paperIndex.addDailyReports).toHaveBeenCalledWith(
      [structuredId, fallbackId],
      "daily/2026-05-11.md",
    );
    expect(paperIndex.setSummaries).toHaveBeenCalledWith({
      [structuredId]: expect.objectContaining({
        coreProblem: structuredMath,
        whyRelevant: "PS1+WISE A & B",
      }),
    });
    expect(paperIndex.setSummaries.mock.calls[0]?.[0]).not.toHaveProperty(fallbackId);
    expect(paperIndex.setSummaries.mock.calls[0]?.[0]).not.toHaveProperty(fakeId);
  });

  it("repairs a missing indexed paperPath from a real canonical detail file", async () => {
    const d = makeDeps();
    const { store } = makePaperIndex();
    const id = "2607.00001";
    await store.upsertFromDailyPaper({
      arxivId: id,
      title: "Existing detail",
      authors: "A",
      date: "2026-05-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: false,
    });
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue(
      `### Paper\n- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
    );
    d.writer.paperDetailExists.mockImplementation(async (candidate: string) => candidate === id);
    const pipeline = new ArxivPipeline({
      markupParser,
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
      detailSelection: testDetailSelection,
    });

    expect(await pipeline.runForDate("2026-05-11")).toMatchObject({ kind: "completed", papersWritten: 1,
    });
    expect(await store.get(id)).toMatchObject({
      detail: true,
      paperPath: `papers/${id}.md`,
    });
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
  });

  it("does not repair detail state from an unverified canonical-path note", async () => {
    const d = makeDeps();
    const { store } = makePaperIndex();
    const id = "2607.00001";
    await store.upsertFromDailyPaper({
      arxivId: id, title: "Handwritten note", authors: "A", date: "2026-05-11",
      arxivCategory: "astro-ph", primaryTopic: "photo-z", detail: false,
    });
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue(
      `### Paper\n- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
    );
    d.writer.paperDetailExists.mockResolvedValue(true);
    d.writer.readPaperDetail.mockResolvedValue(
      `---\narxiv_id: \"${id}\"\n---\n# My notes\nDo not classify as generated detail.`,
    );
    const pipeline = new ArxivPipeline({
      markupParser, fetcher: d.fetcher as any, paperFetcher: d.paperFetcher as any,
      writer: d.writer as any, paperIndex: store, llm: d.llm as any,
      logger: d.logger, arxiv: testArxiv, advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output, llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    expect((await pipeline.runForDate("2026-05-11")).kind).toBe("completed");
    expect(await store.get(id)).toMatchObject({ detail: false, paperPath: null });
  });

  it("does not mutate detail state when a canonical note cannot be read", async () => {
    const d = makeDeps();
    const id = "2607.00001";
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue(
      `### Paper\n- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
    );
    d.writer.paperDetailExists.mockResolvedValue(true);
    d.writer.readPaperDetail.mockRejectedValue(new Error("permission denied"));
    const paperIndex = {
      reconcilePaperDetails: vi.fn(), addDailyReports: vi.fn(), setSummaries: vi.fn(),
    };
    const pipeline = new ArxivPipeline({
      markupParser, fetcher: d.fetcher as any, paperFetcher: d.paperFetcher as any,
      writer: d.writer as any, paperIndex: paperIndex as any, llm: d.llm as any,
      logger: d.logger, arxiv: testArxiv, advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output, llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    await expect(pipeline.runForDate("2026-05-11")).resolves.toEqual({
      kind: "failed_transient", reason: "paper index repair failed: permission denied",
    });
    expect(paperIndex.reconcilePaperDetails).not.toHaveBeenCalled();
  });

  it("clears a stale indexed detail path when the canonical file is absent", async () => {
    const d = makeDeps();
    const { store } = makePaperIndex();
    const id = "2607.00001";
    await store.upsertFromDailyPaper({
      arxivId: id,
      title: "Missing detail",
      authors: "A",
      date: "2026-05-11",
      arxivCategory: "astro-ph",
      primaryTopic: "photo-z",
      detail: true,
      paperPath: `papers/${id}.md`,
    });
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue(
      `### Paper\n- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
    );
    d.writer.paperDetailExists.mockResolvedValue(false);
    const pipeline = new ArxivPipeline({
      markupParser,
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
      detailSelection: testDetailSelection,
    });

    expect((await pipeline.runForDate("2026-05-11")).kind).toBe("completed");
    expect(await store.get(id)).toMatchObject({ detail: false, paperPath: null });
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
  });

  it("retries canonical detail reconciliation after an index write failure", async () => {
    const d = makeDeps();
    const id = "2607.00001";
    const canonicalPath = `papers/${id}.md`;
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue(
      `### Paper\n- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
    );
    d.writer.paperDetailExists.mockResolvedValue(true);
    let failWrite = true;
    const paperIndex = {
      reconcilePaperDetails: vi.fn(async () => {
        if (failWrite) {
          failWrite = false;
          throw new Error("index write failed");
        }
        return 1;
      }),
      addDailyReports: vi.fn(async () => undefined),
      setSummaries: vi.fn(async () => 0),
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    expect(await pipeline.runForDate("2026-05-11")).toEqual({
      kind: "failed_transient",
      reason: "paper index repair failed: index write failed",
    });
    expect(await pipeline.runForDate("2026-05-11")).toMatchObject({ kind: "completed", papersWritten: 1,
    });
    expect(paperIndex.reconcilePaperDetails).toHaveBeenCalledTimes(2);
    expect(paperIndex.reconcilePaperDetails).toHaveBeenNthCalledWith(1, {
      [id]: canonicalPath,
    });
    expect(paperIndex.reconcilePaperDetails).toHaveBeenNthCalledWith(2, {
      [id]: canonicalPath,
    });
    expect(paperIndex.addDailyReports).toHaveBeenCalledTimes(1);
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
  });

  it.each(["addDailyReports", "setSummaries"] as const)(
    "retries a failed %s repair until the full daily index is synchronized",
    async (failedMethod) => {
      const d = makeDeps();
      const id = "2607.00001";
      d.writer.dailyExists.mockResolvedValue(true);
      d.writer.readDaily.mockResolvedValue(
        `### Paper\n- **arXiv**: [${id}](https://arxiv.org/abs/${id})\n- **核心问题**: Problem.`,
      );
      let fail = true;
      const paperIndex = {
        reconcilePaperDetails: vi.fn(async () => 0),
        addDailyReports: vi.fn(async () => {
          if (failedMethod === "addDailyReports" && fail) {
            fail = false;
            throw new Error("daily link write failed");
          }
        }),
        setSummaries: vi.fn(async () => {
          if (failedMethod === "setSummaries" && fail) {
            fail = false;
            throw new Error("summary write failed");
          }
          return 1;
        }),
      };
      const pipeline = new ArxivPipeline({
        markupParser,
        fetcher: d.fetcher as any,
        paperFetcher: d.paperFetcher as any,
        writer: d.writer as any,
        paperIndex: paperIndex as any,
        llm: d.llm as any,
        logger: d.logger,
        arxiv: testArxiv,
        advanced: DEFAULT_SETTINGS.advanced,
        output: DEFAULT_SETTINGS.output,
        llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
      });

      expect((await pipeline.runForDate("2026-05-11")).kind).toBe("failed_transient");
      expect(await pipeline.runForDate("2026-05-11")).toMatchObject({ kind: "completed", papersWritten: 1,
      });
      expect(paperIndex.addDailyReports).toHaveBeenCalledTimes(2);
      expect(paperIndex.setSummaries).toHaveBeenCalledTimes(
        failedMethod === "addDailyReports" ? 1 : 2,
      );
    },
  );

  it("short-circuits with completed when daily file already exists without an index", async () => {
    const d = makeDeps();
    (d.writer as any).dailyExists = vi.fn().mockResolvedValue(true);
    (d.writer as any).paperDetailExists = vi.fn().mockResolvedValue(false);

    const checkpointStore = {
      lookupReusable: vi.fn(),
      upsert: vi.fn(),
      removeAll: vi.fn(async () => undefined),
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      checkpointStore,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });
    const result = await pipeline.runForDate("2026-05-11");
    expect(result.kind).toBe("completed");
    expect((result as any).papersWritten).toBe(0);
    expect(checkpointStore.removeAll).toHaveBeenCalledWith("2026-05-11");
    expect(d.fetcher.fetchRecent).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
  });

  it.each([
    { failure: "write", expected: "rejected" },
    { failure: "cancel", expected: "cancelled" },
  ] as const)("does not clean checkpoints before a daily commit on $failure", async ({ failure, expected }) => {
    const d = makeDeps();
    const date = firstDateFromFixture();
    const id = firstBucketPapersFromFixture()[0]!.id;
    const controller = new AbortController();
    d.llm.call = vi.fn(async () =>
      JSON.stringify({ papers: [{ id, category: "photo-z" }] }),
    );
    const checkpointStore = {
      lookupReusable: vi.fn(),
      upsert: vi.fn(),
      removeAll: vi.fn(async () => undefined),
    };
    const summarize = vi.fn(async () => {
      if (failure === "cancel") controller.abort("cancelled before daily write");
      return { markdown: "complete report", slots: [] };
    });
    if (failure === "write") {
      d.writer.writeDaily.mockRejectedValue(new Error("daily write failed"));
    }
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      checkpointStore,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
      summarizeDaily: summarize as any,
    });

    if (expected === "rejected") {
      await expect(pipeline.runForDate(date, controller.signal)).rejects.toThrow("daily write failed");
    } else {
      await expect(pipeline.runForDate(date, controller.signal)).resolves.toEqual({
        kind: "cancelled",
        reason: "cancelled before daily write",
      });
      expect(d.writer.writeDaily).not.toHaveBeenCalled();
    }
    expect(checkpointStore.removeAll).not.toHaveBeenCalled();
  });

  it.each([
    { index: false, expectedKind: "completed" },
    { index: true, expectedKind: "failed_transient" },
  ] as const)(
    "keeps the original $expectedKind result when existing-daily cleanup fails (index=$index)",
    async ({ index, expectedKind }) => {
      const d = makeDeps();
      const date = "2026-05-11";
      const id = "2607.00001";
      d.writer.dailyExists.mockResolvedValue(true);
      d.writer.readDaily.mockResolvedValue(
        `### Paper\n- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
      );
      const checkpointStore = {
        lookupReusable: vi.fn(),
        upsert: vi.fn(),
        removeAll: vi.fn(async () => { throw new Error("cleanup denied"); }),
      };
      const paperIndex = index ? {
        reconcilePaperDetails: vi.fn(async () => { throw new Error("repair denied"); }),
        addDailyReports: vi.fn(),
        setSummaries: vi.fn(),
      } : undefined;
      const warn = vi.spyOn(d.logger, "warn");
      const pipeline = new ArxivPipeline({
        markupParser,
        fetcher: d.fetcher as any,
        paperFetcher: d.paperFetcher as any,
        writer: d.writer as any,
        paperIndex: paperIndex as any,
        checkpointStore,
        llm: d.llm as any,
        logger: d.logger,
        arxiv: testArxiv,
        advanced: DEFAULT_SETTINGS.advanced,
        output: DEFAULT_SETTINGS.output,
        llmSettings: DEFAULT_SETTINGS.llm,
        detailSelection: testDetailSelection,
      });

      const result = await pipeline.runForDate(date);

      expect(result.kind).toBe(expectedKind);
      expect(checkpointStore.removeAll).toHaveBeenCalledWith(date);
      expect(warn).toHaveBeenCalledWith(
        expect.stringContaining("checkpoint cleanup failed"),
        expect.any(Error),
      );
      if (paperIndex) {
        expect(checkpointStore.removeAll.mock.invocationCallOrder[0]).toBeLessThan(
          paperIndex.reconcilePaperDetails.mock.invocationCallOrder[0]!,
        );
      }
    },
  );

  it("attempts both date-scoped checkpoint cleanups independently for an authoritative daily", async () => {
    const d = makeDeps();
    const date = "2026-05-11";
    d.writer.dailyExists.mockResolvedValue(true);
    const filter = {
      lookupReusable: vi.fn(),
      save: vi.fn(),
      removeAll: vi.fn(async () => { throw new Error("filter cleanup denied"); }),
    };
    const summary = {
      lookupReusable: vi.fn(),
      upsert: vi.fn(),
      removeAll: vi.fn(async () => { throw new Error("summary cleanup denied"); }),
    };
    const warn = vi.spyOn(d.logger, "warn");
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      checkpointStores: { filter, summary },
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    await expect(pipeline.runForDate(date)).resolves.toEqual({
      kind: "completed",
      papersWritten: 0,
    });
    expect(filter.removeAll).toHaveBeenCalledWith(date);
    expect(summary.removeAll).toHaveBeenCalledWith(date);
    expect(warn).toHaveBeenCalledWith(
      `pipeline: committed daily filter checkpoint cleanup failed for ${date}`,
      expect.any(Error),
    );
    expect(warn).toHaveBeenCalledWith(
      `pipeline: committed daily summary checkpoint cleanup failed for ${date}`,
      expect.any(Error),
    );
  });

  it("stops existing-daily repair between derived index mutations when cancelled", async () => {
    const d = makeDeps();
    const controller = new AbortController();
    const id = "2607.00001";
    d.writer.dailyExists.mockResolvedValue(true);
    d.writer.readDaily.mockResolvedValue(
      `### Paper\n- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
    );
    const checkpointStore = {
      lookupReusable: vi.fn(),
      upsert: vi.fn(),
      removeAll: vi.fn(async () => undefined),
    };
    const paperIndex = {
      reconcilePaperDetails: vi.fn(async () => { controller.abort("cancelled after detail repair"); }),
      addDailyReports: vi.fn(),
      setSummaries: vi.fn(),
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      checkpointStore,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    await expect(pipeline.runForDate("2026-05-11", controller.signal)).resolves.toEqual({
      kind: "cancelled",
      reason: "cancelled after detail repair",
    });
    expect(checkpointStore.removeAll).toHaveBeenCalledTimes(1);
    expect(paperIndex.addDailyReports).not.toHaveBeenCalled();
    expect(paperIndex.setSummaries).not.toHaveBeenCalled();
  });

  it("skips paper detail when paper file already exists", async () => {
    const d = makeDeps();

    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];

    const existingPath = `papers/${arxivId}.md`;
    d.writer.paperDetailExists = vi.fn(async (id: string) => id === arxivId);
    d.writer.readPaperDetail.mockResolvedValue(verifiedDetailMarkdown(arxivId));
    const paperIndex = {
      upsertManyFromDailyPapers: vi.fn(async () => [{
        entry: { status: "inbox", detail: true, paperPath: existingPath },
        wasNew: false,
      }]),
      addDailyReports: vi.fn(),
      setSummaries: vi.fn(),
      setPaperPath: vi.fn(),
    };
    let selectorCalls = 0;
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      const daily = structuredDailyResponse(msgs);
      if (daily) return daily;
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z" }],
        });
      }
      if (sys.includes("strict research-paper evaluator")) {
        selectorCalls += 1;
        return JSON.stringify({ papers: [] });
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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });
    const date = firstDateFromFixture();
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect(d.writer.paperDetailLink).toHaveBeenCalledWith(arxivId, date, existingPath);
    expect(selectorCalls).toBe(0);
    expect(d.writer.writeDaily).toHaveBeenCalled();
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
  });

  it("detects canonical details missing from the index before selector scoring and repairs their paths", async () => {
    const d = makeDeps();
    const [existingId, candidateId] = firstBucketPapersFromFixture()
      .slice(0, 2)
      .map((paper) => paper.id);
    const canonicalPath = `papers/${existingId}.md`;
    const paperIndex = {
      upsertManyFromDailyPapers: vi.fn(async (papers: any[]) =>
        papers.map((paper) => ({
          entry: { status: "inbox", detail: false, paperPath: null },
          wasNew: false,
        }))),
      addDailyReports: vi.fn(),
      setSummaries: vi.fn(),
      setPaperPath: vi.fn(),
    };
    d.writer.paperDetailExists = vi.fn(async (id: string) => id === existingId);
    d.writer.readPaperDetail.mockResolvedValue(verifiedDetailMarkdown(existingId));
    d.llm.call = vi.fn(async (messages: any[]) => {
      const system = messages[0]?.content ?? "";
      const daily = structuredDailyResponse(messages);
      if (daily) return daily;
      if (system.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [
            { id: existingId, category: "photo-z" },
            { id: candidateId, category: "photo-z" },
          ],
        });
      }
      if (system.includes("strict research-paper evaluator")) {
        const selectorInput = messages[1]?.content ?? "";
        expect(selectorInput).not.toContain(existingId);
        expect(selectorInput).toContain(candidateId);
        return JSON.stringify({
          papers: [{ id: candidateId, score: 40, reason: "not selected" }],
        });
      }
      return "";
    });
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub",
      fullSections: "## Section\nbody",
    });
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: { ...testDetailSelection, softLimit: 1 },
    });

    expect((await pipeline.runForDate(firstDateFromFixture())).kind).toBe("completed");
    expect(d.writer.paperDetailExists).toHaveBeenCalledTimes(2);
    expect(paperIndex.setPaperPath).toHaveBeenCalledWith(existingId, canonicalPath);
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
    const daily = d.writer.writeDaily.mock.calls[0]?.[1] as string;
    expect(daily).toContain("共 2 篇相关论文，其中 1 篇详细收录。");
    expect(daily).toContain(`→ [[${existingId}]]`);
    expect(daily).not.toContain(`→ [[${candidateId}]]`);
  });

  it("stops fresh post-commit index projection between mutations when cancelled", async () => {
    const d = makeDeps();
    const date = firstDateFromFixture();
    const id = firstBucketPapersFromFixture()[0]!.id;
    const controller = new AbortController();
    d.llm.call = vi.fn(async () =>
      JSON.stringify({ papers: [{ id, category: "photo-z" }] }),
    );
    const paperIndex = {
      upsertManyFromDailyPapers: vi.fn(async (inputs: any[]) =>
        inputs.map((input) => ({
          wasNew: true,
          entry: { status: "inbox", paperPath: null, ...input },
        })),
      ),
      addDailyReports: vi.fn(async () => { controller.abort("cancelled after daily links"); }),
      setSummaries: vi.fn(),
    };
    const checkpointStore = {
      lookupReusable: vi.fn(),
      upsert: vi.fn(),
      removeAll: vi.fn(async () => undefined),
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      checkpointStore,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
      summarizeDaily: vi.fn(async () => ({ markdown: "complete report", slots: [] })) as any,
    });

    await expect(pipeline.runForDate(date, controller.signal)).resolves.toEqual({
      kind: "cancelled",
      reason: "cancelled after daily links",
    });
    expect(checkpointStore.removeAll).toHaveBeenCalledTimes(1);
    expect(checkpointStore.removeAll.mock.invocationCallOrder[0]).toBeLessThan(
      paperIndex.addDailyReports.mock.invocationCallOrder[0]!,
    );
    expect(paperIndex.setSummaries).not.toHaveBeenCalled();
  });

  it("keeps a fresh committed report authoritative when index update fails, then repairs on rerun", async () => {
    const d = makeDeps();
    const date = firstDateFromFixture();
    const id = firstBucketPapersFromFixture()[0]!.id;
    let dailyExists = false;
    d.writer.dailyExists = vi.fn(async () => dailyExists);
    d.writer.writeDaily = vi.fn(async (writtenDate: string, markdown: string) => {
      dailyExists = true;
      d.writes[`daily/${writtenDate}.md`] = markdown;
      return `daily/${writtenDate}.md`;
    });
    d.llm.call = vi.fn(async () =>
      JSON.stringify({ papers: [{ id, category: "photo-z" }] }),
    );
    let failDailyLink = true;
    const paperIndex = {
      upsertManyFromDailyPapers: vi.fn(async (inputs: any[]) =>
        inputs.map((input) => ({
          wasNew: true,
          entry: { status: "inbox", paperPath: null, ...input },
        })),
      ),
      reconcilePaperDetails: vi.fn(async () => 0),
      addDailyReports: vi.fn(async () => {
        if (failDailyLink) {
          failDailyLink = false;
          throw new Error("index disk full");
        }
      }),
      setSummaries: vi.fn(async () => 1),
    };
    const checkpointStore = {
      lookupReusable: vi.fn(),
      upsert: vi.fn(),
      removeAll: vi.fn(async () => undefined),
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      checkpointStore,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
      summarizeDaily: vi.fn(async () => ({
        markdown: `### Paper\n- **arXiv**: [${id}](https://arxiv.org/abs/${id})\n- **核心问题**: Problem.`,
        slots: [],
      })) as any,
    });

    await expect(pipeline.runForDate(date)).resolves.toEqual({
      kind: "failed_transient",
      reason: "paper index daily report update failed: index disk full",
    });
    expect(checkpointStore.removeAll).toHaveBeenCalledTimes(1);
    expect(checkpointStore.removeAll.mock.invocationCallOrder[0]).toBeLessThan(
      paperIndex.addDailyReports.mock.invocationCallOrder[0]!,
    );

    await expect(pipeline.runForDate(date)).resolves.toMatchObject({
      kind: "completed",
      papersWritten: 1,
    });
    expect(checkpointStore.removeAll).toHaveBeenCalledTimes(2);
    expect(d.fetcher.fetchRecent).toHaveBeenCalledTimes(1);
    expect(paperIndex.addDailyReports).toHaveBeenCalledTimes(2);
    expect(paperIndex.setSummaries).toHaveBeenCalledTimes(1);
  });

  it("repairs the full index on retry after cancellation immediately after writeDaily", async () => {
    const d = makeDeps();
    const id = firstBucketPapersFromFixture()[0]!.id;
    const controller = new AbortController();
    let dailyExists = false;
    const summary = [
      "### Paper",
      `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
      "- **核心问题**: Problem.",
    ].join("\n");
    d.writer.dailyExists = vi.fn(async () => dailyExists);
    d.writer.readDaily = vi.fn(async () => summary);
    d.writer.writeDaily = vi.fn(async () => {
      dailyExists = true;
      controller.abort("cancelled after daily write");
      return "daily/report.md";
    });
    d.llm.call = vi.fn().mockImplementation(async (messages: any[]) => {
      const system = messages[0]?.content ?? "";
      const daily = structuredDailyResponse(messages);
      if (daily) return daily;
      if (system.includes("选择最匹配的主题")) {
        return JSON.stringify({ papers: [{ id, category: "photo-z" }] });
      }
      return summary;
    });
    const paperIndex = {
      upsertManyFromDailyPapers: vi.fn(async (inputs: any[]) =>
        inputs.map((input) => ({
          wasNew: true,
          entry: { status: "inbox", paperPath: null, ...input },
        })),
      ),
      reconcilePaperDetails: vi.fn(async () => 0),
      addDailyReports: vi.fn(async () => undefined),
      setSummaries: vi.fn(async () => 1),
    };
    const checkpointStore = {
      lookupReusable: vi.fn(async () => null),
      upsert: vi.fn(async () => undefined),
      removeAll: vi.fn(async () => { throw new Error("cleanup failed after commit"); }),
    };
    const cleanupWarning = vi.spyOn(d.logger, "warn");
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      paperIndex: paperIndex as any,
      checkpointStore,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    expect(await pipeline.runForDate(firstDateFromFixture(), controller.signal)).toEqual({
      kind: "cancelled",
      reason: "cancelled after daily write",
    });
    expect(checkpointStore.removeAll).toHaveBeenCalledTimes(1);
    expect(cleanupWarning).toHaveBeenCalledWith(
      expect.stringContaining("checkpoint cleanup failed"),
      expect.any(Error),
    );
    expect(checkpointStore.removeAll.mock.invocationCallOrder[0]).toBeLessThan(
      paperIndex.addDailyReports.mock.invocationCallOrder[0] ?? Number.POSITIVE_INFINITY,
    );
    expect(paperIndex.addDailyReports).not.toHaveBeenCalled();

    expect(await pipeline.runForDate(firstDateFromFixture())).toMatchObject({ kind: "completed", papersWritten: 1,
    });
    expect(checkpointStore.removeAll).toHaveBeenCalledTimes(2);
    expect(paperIndex.addDailyReports).toHaveBeenCalledWith(
      [id],
      `daily/${firstDateFromFixture()}.md`,
    );
    expect(paperIndex.setSummaries).toHaveBeenCalledTimes(1);
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
      const daily = structuredDailyResponse(msgs);
      if (daily) return daily;
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z" }],
        });
      }
      if (sys.includes("strict research-paper evaluator")) {
        return JSON.stringify({
          papers: [{ id: arxivId, score: 85, reason: "strong direct contribution" }],
        });
      }
      return "## detail summary\n";
    });
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub",
      fullSections: "## Section\nbody",
    });
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      const daily = structuredDailyResponse(msgs);
      if (daily) return daily;
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z" }],
        });
      }
      if (sys.includes("strict research-paper evaluator")) {
        return JSON.stringify({
          papers: [{ id: arxivId, score: 85, reason: "strong direct contribution" }],
        });
      }
      return "## detail summary\n";
    });
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub",
      fullSections: "## Section\nbody",
    });

    const pipeline = new ArxivPipeline({
      markupParser,
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
      detailSelection: testDetailSelection,
    });
    await pipeline.runForDate(firstDateFromFixture());
    expect(d.writer.writePaperDetail).toHaveBeenCalledTimes(1);
    const json = JSON.parse(files["arxiv-daily/.index/papers.json"]);
    expect(json.schemaVersion).toBe(4);
    expect(json.papers[`arxiv:${arxivId}`]).toMatchObject({
      abstract: "atom abstract",
      paperPath: `papers/${arxivId}.md`,
    });
  });

  it("selects a mixed candidate set once and only deep-dives selected IDs", async () => {
    const d = makeDeps();
    const [selectedId, unselectedId] = firstBucketPapersFromFixture().slice(0, 2).map((p) => p.id);
    let selectorCalls = 0;
    d.llm.call = vi.fn(async (messages: any[]) => {
      const system = messages[0]?.content ?? "";
      const daily = structuredDailyResponse(messages);
      if (daily) return daily;
      if (system.includes("选择最匹配的主题")) return JSON.stringify({ papers: [
        { id: selectedId, category: "photo-z" },
        { id: unselectedId, category: "photo-z" },
      ] });
      if (system.includes("strict research-paper evaluator")) {
        selectorCalls += 1;
        return JSON.stringify({ papers: [
          { id: selectedId, score: 85, reason: "direct contribution" },
          { id: unselectedId, score: 40, reason: "limited contribution" },
        ] });
      }
      return "## detail summary\n";
    });
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub",
      fullSections: "## Section\nbody",
    });
    const pipeline = new ArxivPipeline({
      markupParser, fetcher: d.fetcher as any, paperFetcher: d.paperFetcher as any,
      writer: d.writer as any, llm: d.llm as any, logger: d.logger, arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced, output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm, detailSelection: testDetailSelection,
    });

    expect((await pipeline.runForDate(firstDateFromFixture())).kind).toBe("completed");
    expect(selectorCalls).toBe(1);
    expect(d.writer.writePaperDetail).toHaveBeenCalledTimes(1);
    expect(d.writer.writePaperDetail.mock.calls[0]?.[0].id).toBe(selectedId);
  });

  it("continues daily generation with no deep dives when selector transport fails", async () => {
    const d = makeDeps();
    const id = firstBucketPapersFromFixture()[0]!.id;
    d.llm.call = vi.fn(async (messages: any[]) => {
      const system = messages[0]?.content ?? "";
      const daily = structuredDailyResponse(messages);
      if (daily) return daily;
      if (system.includes("选择最匹配的主题")) {
        return JSON.stringify({ papers: [{ id, category: "photo-z" }] });
      }
      if (system.includes("strict research-paper evaluator")) throw new Error("selector unavailable");
      return "## daily still generated\n";
    });
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub", fullSections: "## Section\nbody",
    });
    const pipeline = new ArxivPipeline({
      markupParser, fetcher: d.fetcher as any, paperFetcher: d.paperFetcher as any,
      writer: d.writer as any, llm: d.llm as any, logger: d.logger, arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced, output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm, detailSelection: testDetailSelection,
    });

    expect(await pipeline.runForDate(firstDateFromFixture())).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(d.writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
  });

  it("does not persist fake detail state when selected detail generation fails", async () => {
    const d = makeDeps();
    const { files, store } = makePaperIndex();
    const id = firstBucketPapersFromFixture()[0]!.id;
    d.writer.writePaperDetail = vi.fn(async () => { throw new Error("detail write failed"); });
    d.llm.call = vi.fn(async (messages: any[]) => {
      const system = messages[0]?.content ?? "";
      const daily = structuredDailyResponse(messages);
      if (daily) return daily;
      if (system.includes("选择最匹配的主题")) {
        return JSON.stringify({ papers: [{ id, category: "photo-z" }] });
      }
      if (system.includes("strict research-paper evaluator")) {
        return JSON.stringify({ papers: [{ id, score: 85, reason: "direct contribution" }] });
      }
      return "## detail summary\n";
    });
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub", fullSections: "## Section\nbody",
    });
    const pipeline = new ArxivPipeline({
      markupParser, fetcher: d.fetcher as any, paperFetcher: d.paperFetcher as any,
      writer: d.writer as any, paperIndex: store, llm: d.llm as any, logger: d.logger,
      arxiv: testArxiv, advanced: DEFAULT_SETTINGS.advanced, output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm, detailSelection: testDetailSelection,
    });

    expect((await pipeline.runForDate(firstDateFromFixture())).kind).toBe("completed");
    const entry = JSON.parse(files["arxiv-daily/.index/papers.json"]).papers[`arxiv:${id}`];
    expect(entry).toMatchObject({ abstract: "atom abstract", detail: false, paperPath: null });
  });

  it("emits summarize-daily progress from 1/N through N/N", async () => {
    const d = makeDeps();
    const ids = firstBucketPapersFromFixture().slice(0, 2).map((paper) => paper.id);
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      const daily = structuredDailyResponse(msgs);
      if (daily) return daily;
      if (sys.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: ids.map((id) => ({ id, category: "photo-z" })),
        });
      }
      if (sys.includes("strict research-paper evaluator")) {
        return JSON.stringify({ papers: [] });
      }
      return "";
    });
    d.fetcher.fetchMetadataByIds = vi.fn().mockResolvedValue(
      new Map(ids.map((id) => [id, atomMeta(id)])),
    );

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
      markupParser,
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
      progress: progress as any,
    });
    await pipeline.runForDate(firstDateFromFixture());

    const stages = calls.map((c) => c[0]);
    expect(stages).toContain("fetch-recent");
    expect(stages).toContain("enrich-abstract");
    expect(stages).toContain("filter");
    expect(stages).toContain("fetch-content");
    expect(stages).toContain("summarize-daily");
    expect(calls.filter(([stage]) => stage === "summarize-daily")).toEqual([
      ["summarize-daily", undefined, undefined],
      ["summarize-daily", 1, 2],
      ["summarize-daily", 2, 2],
    ]);
  });
});
