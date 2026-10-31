import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { ArxivPipeline, type PipelineDeps } from "../src/pipeline/pipeline";
import type { FilterRecord } from "../src/pipeline/paper-filter";
import { Logger } from "../src/services/logger";
import { NoopProgressReporter } from "../src/services/progress";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeTopic, type Topic } from "../src/settings/topics";
import type { SourceAdapter } from "../src/sources";
import { markupParser } from "./markup-parser";

// Announcement weekday; weekends stop before discovery.
const date = "2026-09-07";
const ids = ["2609.00001", "2609.00002"];
const decisions: FilterRecord[] = ids.map((id) => ({
  id, category: "topic-a", relevanceScore: 90, directions: ["topic-a#1"],
}));

function topic(tag: string): Topic {
  return normalizeTopic({
    name: tag, tag, detail: false,
    directions: [{ id: `${tag}-direction`, text: "A continuing research direction", origin: "manual" }],
  });
}

function harness(options: { topics?: Topic[]; fetchDelayMs?: number } = {}) {
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
  let fetchesSettled = 0;
  const source: SourceAdapter = {
    sourceId: "arxiv",
    listForDate: vi.fn(async () => ({
      kind: "ok" as const, channels: ["astro-ph.GA"],
      papers: ids.map((id) => ({
        paperKey: "arxiv:" + id, source: "arxiv", externalId: id,
        title: "Research paper " + id, authors: "A. Researcher", abstract: "Abstract " + id,
        categories: ["astro-ph.GA"], canonicalUrl: "https://arxiv.org/abs/" + id,
      })),
    })),
    fetchContent: vi.fn(async (id: string) => {
      await new Promise((resolve) => setTimeout(resolve, options.fetchDelayMs ?? 0));
      fetchesSettled += 1;
      return {
        abstract: "Abstract " + id,
        sections: [{ heading: "Methods", text: "Full paper evidence " + id }],
        quality: "full" as const, canonicalUrl: "https://arxiv.org/abs/" + id,
      };
    }),
  };
  const llm = {
    call: vi.fn(async (messages: Array<{ content: string }>) =>
      messages[0]!.content.includes("relevanceScore")
        ? JSON.stringify({ papers: decisions })
        : JSON.stringify({ papers: [] })),
  };
  const writer = {
    dailyExists: vi.fn(async () => false),
    dailyPath: (day: string) => `arxiv-daily/daily/${day}.md`,
    writeDaily: vi.fn(async (day: string, content: string) => { files[`arxiv-daily/daily/${day}.md`] = content; }),
    paperDetailPath: (paperId: string) => `arxiv-daily/papers/${paperId}.md`,
    paperDetailExists: vi.fn(async () => false),
    paperDetailLink: (paperId: string) => `[[${paperId}]]`,
  };
  // Each fetch-content report, with whether every fetch so far had settled when it was made.
  const fetchProgress: Array<{ completed: number | undefined; settled: number }> = [];
  const progress = new NoopProgressReporter();
  vi.spyOn(progress, "setStage").mockImplementation((stage, completed) => {
    if (stage === "fetch-content") fetchProgress.push({ completed, settled: fetchesSettled });
  });
  const pipeline = new ArxivPipeline({
    markupParser, fetcher: {} as PipelineDeps["fetcher"], paperFetcher: {} as PipelineDeps["paperFetcher"],
    sourceAdapter: source, writer: writer as unknown as PipelineDeps["writer"],
    llm: llm as unknown as PipelineDeps["llm"], logger: new Logger("error"), progress,
    arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: options.topics ?? [topic("topic-a")] },
    advanced: DEFAULT_SETTINGS.advanced, output: DEFAULT_SETTINGS.output, llmSettings: DEFAULT_SETTINGS.llm,
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
    summarizeDaily: vi.fn(async (papers: ReadonlyArray<{ id: string }>) => ({
      markdown: "# Daily report\n\n" + papers.map((paper) => `### ${paper.id}\n`).join("\n"),
      slots: [],
    })),
  });
  return { pipeline, source, llm, fetchProgress };
}

describe("daily run preflight", () => {
  it("rejects an unusable topic list before anything is fetched from arXiv", async () => {
    const h = harness({ topics: [topic("topic-a"), topic("topic-a")] });
    const result = await h.pipeline.runForDate(date);
    expect(result).toMatchObject({ kind: "failed_permanent", reason: expect.stringContaining("Duplicate topic tag") });
    expect(h.source.listForDate).not.toHaveBeenCalled();
    expect(h.llm.call).not.toHaveBeenCalled();
  });
});

describe("content-fetch progress", () => {
  it("counts a paper only once its fetch has settled", async () => {
    const h = harness({ fetchDelayMs: 5 });
    await h.pipeline.runForDate(date);
    expect(h.fetchProgress[0]).toEqual({ completed: 0, settled: 0 });
    for (const report of h.fetchProgress.slice(1)) {
      expect(report.completed).toBeLessThanOrEqual(report.settled);
    }
    expect(h.fetchProgress.at(-1)).toEqual({ completed: ids.length, settled: ids.length });
  });
});
