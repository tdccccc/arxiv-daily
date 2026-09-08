import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { ArxivPipeline, type PipelineDeps } from "../src/pipeline/pipeline";
import type { FilterRecord } from "../src/pipeline/paper-filter";
import { DailyFilterCheckpointStore } from "../src/services/daily-filter-checkpoint-store";
import { PaperIndexStore } from "../src/services/paper-index";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeTopic } from "../src/settings/topics";
import type { SourceAdapter } from "../src/sources";
import { markupParser } from "./markup-parser";

const date = "2026-09-05";
const id = (index: number): string => `2609.${String(index).padStart(5, "0")}`;
const decision = (index: number, relevanceScore: number, category = "topic-a"): FilterRecord => ({
  id: id(index), category, relevanceScore, directions: [`${category}#1`],
});

function harness(records: FilterRecord[]) {
  const files: Record<string, string> = {};
  const directories = new Set<string>();
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/"),
    exists: async (path) => path in files || directories.has(path),
    readText: async (path) => {
      if (!(path in files)) throw new Error("missing " + path);
      return files[path]!;
    },
    writeText: async (path, value) => { files[path] = value; },
    writeTextAtomic: async (path, value) => { files[path] = value; },
    mkdir: async (path) => { directories.add(path); },
    remove: async (path) => { delete files[path]; directories.delete(path); },
    rename: async (from, to) => { files[to] = files[from]!; delete files[from]; },
  };
  const paperIndex = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);
  const filterStore = new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output);
  const source: SourceAdapter = {
    sourceId: "arxiv",
    listForDate: vi.fn(async () => ({
      kind: "ok" as const, channels: ["astro-ph.GA"],
      papers: records.map(({ id }) => ({
        paperKey: "arxiv:" + id, source: "arxiv", externalId: id,
        title: "Research paper " + id, authors: "A. Researcher", abstract: "Abstract " + id,
        categories: ["astro-ph.GA"], canonicalUrl: "https://arxiv.org/abs/" + id,
      })),
    })),
    fetchContent: vi.fn(async (id) => ({
      abstract: "Abstract " + id,
      sections: [{ heading: "Methods", text: "Full paper evidence " + id }],
      quality: "full" as const, canonicalUrl: "https://arxiv.org/abs/" + id,
    })),
  };
  const llm = {
    call: vi.fn(async (messages: Array<{ content: string }>) => {
      if (messages[0]!.content.includes("relevanceScore")) return JSON.stringify({ papers: records });
      // Detail selection is real; an empty evaluation chooses no extra note.
      return JSON.stringify({ papers: [] });
    }),
  };
  const summarize = vi.fn<NonNullable<PipelineDeps["summarizeDaily"]>>(async (papers) => ({
    markdown: "# Daily report\n\n" + papers.map((paper) => `### ${paper.id}\n${paper.title}\n`).join("\n"),
    slots: [],
  }));
  const writer = {
    dailyExists: vi.fn(async () => false),
    dailyPath: (day: string) => `arxiv-daily/daily/${day}.md`,
    writeDaily: vi.fn(async (day: string, content: string) => { files[`arxiv-daily/daily/${day}.md`] = content; }),
    paperDetailPath: (paperId: string) => `arxiv-daily/papers/${paperId}.md`,
    paperDetailExists: vi.fn(async () => false),
    paperDetailLink: (paperId: string) => `[[${paperId}]]`,
  };
  const logger = new Logger("error");
  const info = vi.spyOn(logger, "info");
  const makePipeline = (maximum?: number, withIndex = false): ArxivPipeline => new ArxivPipeline({
    markupParser, fetcher: {} as PipelineDeps["fetcher"], paperFetcher: {} as PipelineDeps["paperFetcher"],
    sourceAdapter: source, writer: writer as unknown as PipelineDeps["writer"],
    llm: llm as unknown as PipelineDeps["llm"], logger,
    arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: ["topic-a", "topic-b"].map((tag) => normalizeTopic({
      name: tag, tag, detail: true,
      directions: [{ id: tag + "-direction", text: "A continuing research direction", origin: "manual" }],
    })) },
    advanced: DEFAULT_SETTINGS.advanced, output: { ...DEFAULT_SETTINGS.output, ...(maximum === undefined ? {} : { maxDailyPapers: maximum }) },
    llmSettings: DEFAULT_SETTINGS.llm,
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
    summarizeDaily: summarize,
    checkpointStores: { filter: filterStore },
    ...(withIndex ? { paperIndex } : {}),
  });
  return { files, source, llm, summarize, writer, paperIndex, info, makePipeline };
}

describe("daily paper cap", () => {
  it("selects globally across topics before fetching content, scoring details, and summarizing", async () => {
    const h = harness([decision(1, 10), decision(2, 80), decision(3, 95, "topic-b")]);
    const result = await h.makePipeline(2).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 2 });
    expect(vi.mocked(h.source.fetchContent).mock.calls.map(([paperId]) => paperId)).toEqual([id(3), id(2)]);
    expect(h.summarize.mock.calls[0]![0].map(({ id }) => id)).toEqual([id(3), id(2)]);
    expect(h.summarize.mock.calls[0]![2].omittedByTopic).toEqual({ "topic-a": 1 });
    const detailRequest = h.llm.call.mock.calls.find(([messages]) => !messages[0]!.content.includes("relevanceScore"));
    expect(detailRequest).toBeDefined();
    expect(detailRequest![0][1]!.content).toContain(id(3));
    expect(detailRequest![0][1]!.content).toContain(id(2));
    expect(detailRequest![0][1]!.content).not.toContain(id(1));
    expect(h.writer.writeDaily.mock.calls[0]![1]).not.toContain(id(1));
    expect(h.info).toHaveBeenCalledWith("pipeline: daily paper limit=2 kept=2/3 omitted=1");
  });

  it("defaults to twenty papers and never fills the quota with lower-ranked papers", async () => {
    const h = harness(Array.from({ length: 25 }, (_, index) => decision(index + 1, index + 1)));
    const result = await h.makePipeline().runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 20 });
    expect(h.summarize.mock.calls[0]![0].map(({ id }) => id))
      .toEqual(Array.from({ length: 20 }, (_, index) => id(25 - index)));
    expect(h.source.fetchContent).toHaveBeenCalledTimes(20);
  });

  it("breaks equal-score ties by canonical ID independently of model response order", async () => {
    for (const order of [[3, 2, 1], [2, 1, 3]]) {
      const h = harness(order.map((index) => decision(index, 80)));
      await h.makePipeline(2).runForDate(date);
      expect(h.summarize.mock.calls[0]![0].map(({ id }) => id)).toEqual([id(1), id(2)]);
    }
  });

  it("excludes ignored papers before taking the quota and links only selected papers to the report", async () => {
    const h = harness([decision(1, 100), decision(2, 90), decision(3, 80)]);
    await h.paperIndex.upsertFromDailyPaper({
      arxivId: id(1), title: "Ignored paper", authors: "A", date: "2026-09-04",
      arxivCategory: "astro-ph.GA", primaryTopic: "topic-a", detail: false,
    });
    await h.paperIndex.setStatus(id(1), "ignored");
    const result = await h.makePipeline(1, true).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(h.summarize.mock.calls[0]![0].map(({ id }) => id)).toEqual([id(2)]);
    const index = JSON.parse(h.files["arxiv-daily/.index/papers.json"]!);
    expect(index.papers["arxiv:" + id(1)].status).toBe("ignored");
    expect(index.papers["arxiv:" + id(1)].dailyReports).toEqual([]);
    expect(index.papers["arxiv:" + id(3)].dailyReports).toEqual([]);
    expect(index.papers["arxiv:" + id(2)].dailyReports).toEqual([`arxiv-daily/daily/${date}.md`]);
    expect(h.source.fetchContent).toHaveBeenCalledTimes(1);
    expect(h.summarize.mock.calls[0]![2].omittedByTopic).toEqual({ "topic-a": 1 });
  });

  it("retains the number of matches for a topic entirely omitted by the daily cap", async () => {
    const h = harness([decision(1, 95), decision(2, 90, "topic-b"), decision(3, 80, "topic-b")]);
    await h.makePipeline(1).runForDate(date);
    expect(h.summarize.mock.calls[0]![0].map(({ id }) => id)).toEqual([id(1)]);
    expect(h.summarize.mock.calls[0]![2].omittedByTopic).toEqual({ "topic-b": 2 });
  });

  it("reuses complete cached scores when a retry changes only the daily cap", async () => {
    const h = harness([decision(1, 70), decision(2, 90), decision(3, 80)]);
    h.summarize.mockRejectedValueOnce(new Error("interrupted before the report was committed"));
    await expect(h.makePipeline(1).runForDate(date)).resolves.toMatchObject({ kind: "failed_transient" });
    const result = await h.makePipeline(2).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 2 });
    expect(h.summarize.mock.calls[0]![0].map(({ id }) => id)).toEqual([id(2)]);
    expect(h.summarize.mock.calls[1]![0].map(({ id }) => id)).toEqual([id(2), id(3)]);
    expect(h.llm.call.mock.calls.filter(([messages]) => messages[0]!.content.includes("relevanceScore"))).toHaveLength(1);
  });
});
