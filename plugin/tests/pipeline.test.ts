import { describe, it, expect, vi } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { ArxivPipeline } from "../src/pipeline/pipeline";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

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

function makeDeps() {
  const writes: Record<string, string> = {};
  const fetcher = {
    fetchRecent: vi.fn().mockResolvedValue(recentHtml),
    fetchAbstractsByIds: vi.fn().mockResolvedValue(new Map<string, string>()),
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
    writeDaily: vi.fn(async (date: string, content: string) => {
      writes[`daily/${date}.md`] = content;
      return `daily/${date}.md`;
    }),
    writePaperDetail: vi.fn(async (p: any, date: string, content: string) => {
      writes[`papers/${p.id}.md`] = content;
      return `papers/${p.id}.md`;
    }),
    writeEmptyDaily: vi.fn(async (date: string) => {
      writes[`daily/${date}.md`] = "empty";
      return `daily/${date}.md`;
    }),
    dailyExists: vi.fn(async () => false),
    paperDetailExists: vi.fn(async () => false),
  };
  const llm = {
    call: vi.fn().mockResolvedValueOnce(JSON.stringify({ papers: [] })),
  };
  const logger = new Logger("error");
  return { writes, fetcher, paperFetcher, writer, llm, logger };
}

function firstDateFromFixture(): string {
  const m = /(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})/.exec(recentHtml)!;
  const months: Record<string, number> = {
    January: 1, February: 2, March: 3, April: 4, May: 5, June: 6,
    July: 7, August: 8, September: 9, October: 10, November: 11, December: 12,
  };
  return `${m[3]}-${String(months[m[2]]).padStart(2, "0")}-${String(Number(m[1])).padStart(2, "0")}`;
}

describe("ArxivPipeline", () => {
  it("returns failed_transient when date not in /recent", async () => {
    const d = makeDeps();
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const result = await pipeline.runForDate("1999-01-01");
    expect(result.kind).toBe("failed_transient");
  });

  it("writes empty daily when LLM returns no relevant papers", async () => {
    const d = makeDeps();
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const date = firstDateFromFixture();
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect((result as any).papersWritten).toBe(0);
    expect(d.writer.writeEmptyDaily).toHaveBeenCalled();
  });

  it("enriches abstracts and runs filter+summarize for a kept paper", async () => {
    const d = makeDeps();
    // Override LLM call sequence: filter returns 1 paper, then daily summary returns markdown
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("筛选出相关论文")) {
        // Pick the first arxiv id present in the fixture
        const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
        return JSON.stringify({
          papers: [{ id: m[1], category: "photo-z", detail: false }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## Photo-z 相关\n### Stub title\n- summary\n";
      }
      return "";
    });
    d.fetcher.fetchAbstractsByIds = vi
      .fn()
      .mockResolvedValue(new Map([["stub", "abstract"]]));

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const date = firstDateFromFixture();
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect((result as any).papersWritten).toBe(1);
    expect(d.fetcher.fetchAbstractsByIds).toHaveBeenCalled();
    expect(d.writer.writeDaily).toHaveBeenCalled();
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
      arxiv: DEFAULT_SETTINGS.arxiv,
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
      if (sys.includes("筛选出相关论文")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: true }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## stub daily summary\n";
      }
      return "## detail summary\n";
    });
    d.fetcher.fetchAbstractsByIds = vi
      .fn()
      .mockResolvedValue(new Map([[arxivId, "abstract"]]));
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
      arxiv: DEFAULT_SETTINGS.arxiv,
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
});
