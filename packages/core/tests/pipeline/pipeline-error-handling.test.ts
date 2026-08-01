import { markupParser } from "../markup-parser";
import { describe, it, expect, vi } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { LlmTransientExhaustedError } from "../../src/llm/client";
import type { PipelineResult } from "../../src/pipeline/pipeline";
import { ArxivPipeline } from "../../src/pipeline/pipeline";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";


const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(
  resolve(here, "../fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

const testDetailSelection = {
  normalThreshold: 70,
  exceptionalThreshold: 90,
  softLimit: 2,
};

const testArxiv = {
  ...DEFAULT_SETTINGS.arxiv,
  topics: [
    { id: "t1", name: "Test Topic", tag: "test", description: "test topic", detail: true },
  ],
};

function structuredDailyResponse(messages: any[]): string | null {
  const system = messages[0]?.content ?? "";
  if (
    !system.includes("严格 JSON 对象") &&
    !system.includes("strict JSON object")
  ) {
    return null;
  }
  const id = /ID: (\d{4}\.\d{4,5})/.exec(messages[1]?.content ?? "")?.[1];
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
      markupParser,
      fetcher: fetcher as any,
      paperFetcher: paperFetcher as any,
      writer: writer as any,
      llm: llm as any,
      logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
      markupParser,
      fetcher: fetcher as any,
      paperFetcher: paperFetcher as any,
      writer: writer as any,
      llm: llm as any,
      logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
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
            { id: "2605.00001", category: "photo-z" },
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
      markupParser,
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
      detailSelection: testDetailSelection,
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
    detail?: boolean;
    writer?: Record<string, unknown>;
    paperIndex?: Record<string, unknown>;
    paperFetcher?: Record<string, unknown>;
    checkpointStore?: Record<string, unknown>;
    checkpointStores?: Record<string, unknown>;
    summarizeDaily?: (...args: any[]) => Promise<{ markdown: string; slots: any[] }>;
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
      readPaperDetail: vi.fn(async (id: string) => [
        "---",
        `arxiv_id: \"${id}\"`,
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
      ].join("\n")),
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
      call: vi.fn(async (messages: any[]) => {
        const system = messages[0]?.content ?? "";
        const daily = structuredDailyResponse(messages);
        if (daily) return daily;
        if (system.includes("选择最匹配的主题")) {
          return JSON.stringify({
            papers: ids.map((id) => ({ id, category: "test" })),
          });
        }
        if (system.includes("strict research-paper evaluator")) {
          return JSON.stringify({
            papers: ids.map((id) => ({
              id,
              score: overrides.detail ? 80 : 0,
              reason: overrides.detail ? "strong contribution" : "not selected",
            })),
          });
        }
        return [
          "## Test Topic",
          ...ids.map((id) => `### [${id}]\n- **研究问题**: test`),
        ].join("\n\n");
      }),
    };
    const paperFetcher = {
      fetch: vi.fn(async () => ({
        abstractConclusion: "abstract and conclusion",
        fullSections: null,
      })),
      ...overrides.paperFetcher,
    };
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: fetcher as any,
      paperFetcher: paperFetcher as any,
      writer: writer as any,
      paperIndex: paperIndex as any,
      checkpointStore: overrides.checkpointStore as any,
      checkpointStores: overrides.checkpointStores as any,
      llm: llm as any,
      logger: new Logger("error"),
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
      summarizeDaily: overrides.summarizeDaily,
    });
    return { pipeline, writer, paperIndex, paperFetcher, llm };
  }

  it("writes a daily report with fallback when the second structured summary fails", async () => {
    const ids = ["2605.08080", "2605.08068"];
    const { pipeline, writer, paperIndex, llm } = makeOnePaperPipeline({ ids });
    llm.call = vi.fn(async (messages: any[]) => {
      const system = messages[0]?.content ?? "";
      if (system.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: ids.map((id) => ({ id, category: "test" })),
        });
      }
      if (system.includes("strict research-paper evaluator")) {
        return JSON.stringify({ papers: [] });
      }
      const id = /ID: (\d{4}\.\d{4,5})/.exec(messages[1]?.content ?? "")?.[1];
      if (id === ids[1]) {
        throw new LlmTransientExhaustedError(
          new Error("second paper unavailable"),
        );
      }
      return structuredDailyResponse(messages) ?? "";
    });

    expect(await pipeline.runForDate("2026-05-11")).toMatchObject({ kind: "completed", papersWritten: 2,
    });
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    const dailyMarkdown = writer.writeDaily.mock.calls[0]?.[1] as string;
    expect(dailyMarkdown).toContain("<!-- arxiv-daily-fallback:2605.08068 -->");
    expect(dailyMarkdown).toContain("其中 1 篇使用回退内容。");
    expect(paperIndex.setSummaries).toHaveBeenCalledWith(
      expect.not.objectContaining({ "2605.08068": expect.anything() }),
    );
  });

  it("commits mixed emergency output and indexes all IDs but only structured summaries", async () => {
    const ids = ["2605.08080", "2605.08068"];
    const emergency = [
      "<!-- arxiv-daily-emergency-report:v1 -->",
      "> **降级应急报告。**",
      "## Test Topic",
      `### Structured\n> 信息来源： Abstract\n- **作者**: A\n- **arXiv**: [${ids[0]}](https://arxiv.org/abs/${ids[0]})\n- **研究问题**: Trusted problem`,
      `### Fallback\n> **自动摘要不可用。**\n<!-- arxiv-daily-fallback:${ids[1]} -->\n> 信息来源： Abstract\n- **作者**: B\n- **arXiv**: [${ids[1]}](https://arxiv.org/abs/${ids[1]})\n- **原始摘要**: Original abstract`,
    ].join("\n");
    const summarizeDaily = vi.fn(async (papers: any[]) => {
      expect(papers.map((paper) => paper.id)).toEqual(ids);
      expect(papers.map((paper) => paper.abstract)).toEqual(["abstract", "abstract"]);
      return { markdown: emergency, slots: [] };
    });
    const { pipeline, writer, paperIndex } = makeOnePaperPipeline({
      ids,
      summarizeDaily,
    });

    const completed = await pipeline.runForDate("2026-05-11");
    expect(completed.kind).toBe("completed");
    if (completed.kind === "completed") {
      expect(completed.papersWritten).toBe(2);
      expect(completed.digest).toBeDefined();
    }
    expect(writer.writeDaily).toHaveBeenCalledWith(
      "2026-05-11",
      emergency,
      expect.any(Object),
    );
    expect(paperIndex.addDailyReports).toHaveBeenCalledWith(
      ids,
      "arxiv-daily/daily/2026-05-11.md",
    );
    expect(paperIndex.setSummaries).toHaveBeenCalledWith({
      [ids[0]]: { sourceSections: "Abstract", coreProblem: "Trusted problem" },
    });
  });

  it("stops before every downstream mutation when filter checkpoint persistence fails", async () => {
    const filterStore = {
      lookupReusable: vi.fn(async () => null),
      save: vi.fn(async () => { throw new Error("filter checkpoint disk full"); }),
      removeAll: vi.fn(),
    };
    const summarizeDaily = vi.fn();
    const { pipeline, writer, paperIndex, paperFetcher } = makeOnePaperPipeline({
      checkpointStores: { filter: filterStore },
      summarizeDaily,
    });

    expect(await pipeline.runForDate("2026-05-11")).toEqual({
      kind: "failed_transient",
      reason: "paper filter checkpoint failed: save failed for 2026-05-11: filter checkpoint disk full",
    });
    expect(paperIndex.upsertManyFromDailyPapers).not.toHaveBeenCalled();
    expect(paperFetcher.fetch).not.toHaveBeenCalled();
    expect(summarizeDaily).not.toHaveBeenCalled();
    expect(writer.writeDaily).not.toHaveBeenCalled();
  });

  it("retains a strict zero-result filter checkpoint when no daily report is committed", async () => {
    const filterStore = {
      lookupReusable: vi.fn(async () => null),
      save: vi.fn(async () => undefined),
      removeAll: vi.fn(async () => undefined),
    };
    const { pipeline, llm, writer } = makeOnePaperPipeline({
      checkpointStores: { filter: filterStore },
    });
    llm.call = vi.fn(async () => JSON.stringify({ papers: [] }));

    expect(await pipeline.runForDate("2026-05-11")).toMatchObject({
      kind: "completed",
      papersWritten: 0,
    });
    expect(filterStore.save).toHaveBeenCalledWith(
      "2026-05-11",
      expect.any(Object),
      [],
    );
    expect(filterStore.removeAll).not.toHaveBeenCalled();
    expect(writer.writeDaily).not.toHaveBeenCalled();
  });

  it("passes filter and summary checkpoint scopes to their orchestration seams", async () => {
    const filterStore = {
      lookupReusable: vi.fn(async () => [{ id: "2605.08080", category: "test" }]),
      save: vi.fn(),
      removeAll: vi.fn(async () => undefined),
    };
    const summaryStore = {
      lookupReusable: vi.fn(),
      upsert: vi.fn(),
      removeAll: vi.fn(async () => undefined),
    };
    const summarizeDaily = vi.fn(async (_papers: any[], date: string, deps: any) => {
      expect(date).toBe("2026-05-11");
      expect(deps.checkpointStore).toBe(summaryStore);
      return { markdown: "# injected daily", slots: [] };
    });
    const { pipeline, llm } = makeOnePaperPipeline({
      checkpointStores: { filter: filterStore, summary: summaryStore },
      summarizeDaily,
    });

    expect(await pipeline.runForDate("2026-05-11")).toMatchObject({ kind: "completed" });
    expect(filterStore.lookupReusable).toHaveBeenCalledWith(
      "2026-05-11",
      expect.objectContaining({ llm: DEFAULT_SETTINGS.llm }),
    );
    expect(llm.call).not.toHaveBeenCalledWith(
      expect.arrayContaining([expect.objectContaining({ content: expect.stringContaining("选择最匹配的主题") })]),
      expect.anything(),
    );
    expect(summarizeDaily).toHaveBeenCalledTimes(1);
  });

  it("passes report checkpoint scope to an injected summarizeDaily mock", async () => {
    const checkpointStore = {
      lookupReusable: vi.fn(),
      upsert: vi.fn(),
    };
    const summarizeDaily = vi.fn(async (_papers: any[], date: string, deps: any) => {
      expect(date).toBe("2026-05-11");
      expect(deps.llmSettings).toBe(DEFAULT_SETTINGS.llm);
      expect(deps.checkpointStore).toBe(checkpointStore);
      return { markdown: "# injected daily", slots: [] };
    });
    const { pipeline } = makeOnePaperPipeline({
      checkpointStore,
      summarizeDaily,
    });

    expect(await pipeline.runForDate("2026-05-11")).toMatchObject({
      kind: "completed",
      papersWritten: 1,
    });
    expect(summarizeDaily).toHaveBeenCalledTimes(1);
  });

  it("writes a fallback daily report after three strict structured-validation failures", async () => {
    const ids = ["2605.08080"];
    const { pipeline, writer, paperIndex, llm } = makeOnePaperPipeline({ ids });
    llm.call = vi.fn(async (messages: any[]) => {
      const system = messages[0]?.content ?? "";
      if (system.includes("选择最匹配的主题")) {
        return JSON.stringify({ papers: [{ id: ids[0], category: "test" }] });
      }
      if (system.includes("strict research-paper evaluator")) {
        return JSON.stringify({ papers: [] });
      }
      return "not json";
    });

    const completed = await pipeline.runForDate("2026-05-11");
    expect(completed.kind).toBe("completed");
    if (completed.kind === "completed") {
      expect(completed.papersWritten).toBe(1);
      expect(completed.digest).toBeDefined();
    }
    // filter + 3 daily validation attempts (detail selection is skipped)
    expect(llm.call).toHaveBeenCalledTimes(4);
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    const dailyMarkdown = writer.writeDaily.mock.calls[0]?.[1] as string;
    expect(dailyMarkdown).toContain("<!-- arxiv-daily-fallback:2605.08080 -->");
    expect(paperIndex.setSummaries).toHaveBeenCalledWith({});
  });

  it("labels unrelated daily assembly failures without claiming an LLM failure", async () => {
    const { pipeline, writer } = makeOnePaperPipeline({
      summarizeDaily: vi.fn(async () => {
        throw new TypeError("assembler invariant broke");
      }),
    });

    expect(await pipeline.runForDate("2026-05-11")).toEqual({
      kind: "failed_transient",
      reason: "daily summary failed: assembler invariant broke",
    });
    expect(writer.writeDaily).not.toHaveBeenCalled();
  });

  it.each([401, 403])(
    "keeps provider %i daily failures permanent and does not write",
    async (status) => {
      const ids = ["2605.08080"];
      const { pipeline, writer, llm } = makeOnePaperPipeline({ ids });
      llm.call = vi.fn(async (messages: any[]) => {
        const system = messages[0]?.content ?? "";
        if (system.includes("选择最匹配的主题")) {
          return JSON.stringify({ papers: [{ id: ids[0], category: "test" }] });
        }
        if (system.includes("strict research-paper evaluator")) {
          return JSON.stringify({ papers: [] });
        }
        throw Object.assign(new Error(`daily request failed with ${status}`), {
          status,
        });
      });

      expect(await pipeline.runForDate("2026-05-11")).toEqual({
        kind: "failed_permanent",
        reason: `daily summary LLM failed: daily request failed with ${status}`,
      });
      expect(writer.writeDaily).not.toHaveBeenCalled();
    },
  );

  it("does not write a partial daily report when cancelled between papers", async () => {
    const ids = ["2605.08080", "2605.08068"];
    const controller = new AbortController();
    const { pipeline, writer, llm } = makeOnePaperPipeline({ ids });
    llm.call = vi.fn(async (messages: any[]) => {
      const system = messages[0]?.content ?? "";
      if (system.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: ids.map((id) => ({ id, category: "test" })),
        });
      }
      if (system.includes("strict research-paper evaluator")) {
        return JSON.stringify({ papers: [] });
      }
      const response = structuredDailyResponse(messages);
      if (response) controller.abort("stop after first daily paper");
      return response ?? "";
    });

    expect(await pipeline.runForDate("2026-05-11", controller.signal)).toEqual({
      kind: "cancelled",
      reason: "stop after first daily paper",
    });
    expect(llm.call).toHaveBeenCalledTimes(2);
    expect(writer.writeDaily).not.toHaveBeenCalled();
  });

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
      kind: "cancelled",
      reason: "cancelled after daily write",
    });
    expect(paperIndex.addDailyReports).not.toHaveBeenCalled();
    expect(paperIndex.setSummaries).not.toHaveBeenCalled();
  });

  it("continues to write the daily report when setPaperPath fails for an existing detail", async () => {
    const logger = new Logger("error");
    const logError = vi.spyOn(logger, "error");
    const { pipeline, writer, paperIndex } = makeOnePaperPipeline({
      detail: true,
      writer: {
        paperDetailExists: vi.fn(async () => true),
      },
      paperIndex: {
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
        setPaperPath: vi.fn(async () => {
          throw new Error("index write failed");
        }),
      },
      paperFetcher: {
        fetch: vi.fn(async () => ({
          abstractConclusion: "abstract and conclusion",
          fullSections: "detail content",
        })),
      },
    });
    (pipeline as any).deps.logger = logger;

    const result = await pipeline.runForDate("2026-05-11");

    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(writer.writeDaily).toHaveBeenCalled();
    expect(paperIndex.setPaperPath).toHaveBeenCalled();
    expect(logError).toHaveBeenCalledWith(
      expect.stringContaining("pipeline: detail index repair failed for 2605.08080"),
      expect.any(Error),
    );
  });

  it("rejects partial category discovery before pipeline mutations", async () => {
    const logger = new Logger("debug");
    const logError = vi.spyOn(logger, "error");
    const logWarn = vi.spyOn(logger, "warn");
    const fetcher = {
      fetchRecent: vi
        .fn()
        .mockResolvedValueOnce(Symbol("schema drift") as any)
        .mockResolvedValueOnce(recentHtml),
      fetchMetadataByIds: vi.fn(async () => new Map()),
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
    const pipeline = new ArxivPipeline({
      markupParser,
      fetcher: fetcher as any,
      paperFetcher: { fetch: vi.fn() } as any,
      writer: writer as any,
      llm: { call: vi.fn().mockResolvedValue(JSON.stringify({ papers: [] })) } as any,
      logger,
      arxiv: { ...testArxiv, categories: ["bad.cat", "astro-ph"] },
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      detailSelection: testDetailSelection,
    });

    await expect(pipeline.runForDate("2026-05-11")).resolves.toMatchObject({
      kind: "failed_permanent",
      reason: expect.stringContaining("parse failed for bad.cat"),
    });

    expect(logError).toHaveBeenCalledWith(
      expect.stringContaining("arxiv-source: parse failed for bad.cat"),
    );
    expect(logWarn).not.toHaveBeenCalledWith(
      expect.stringContaining("parse failed for bad.cat"),
    );
    expect(logWarn).toHaveBeenCalledWith(
      "arxiv-source: rejecting partial discovery; 1/2 categories succeeded, 1 failed",
    );
    expect(fetcher.fetchMetadataByIds).not.toHaveBeenCalled();
    expect(writer.writeDaily).not.toHaveBeenCalled();
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

    expect(result).toMatchObject({ kind: "completed", papersWritten: ids.length });
    expect(paperFetcher.fetch).toHaveBeenCalledTimes(ids.length);
    expect(maxActive).toBeGreaterThan(1);
    expect(maxActive).toBeLessThanOrEqual(6);
  });
});
