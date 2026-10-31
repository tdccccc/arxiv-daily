import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { ArxivPipeline, type PipelineDeps } from "../src/pipeline/pipeline";
import type { FilterRecord } from "../src/pipeline/paper-filter";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeTopic } from "../src/settings/topics";
import type { SourceAdapter } from "../src/sources";
import { markupParser } from "./markup-parser";

// Announcement weekday; weekends stop before discovery.
const date = "2026-09-07";
const id = "2609.00001";
const decision: FilterRecord = { id, category: "topic-a", relevanceScore: 90, directions: ["topic-a#1"] };

function harness(selectorAnswers: string[]) {
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
  const source: SourceAdapter = {
    sourceId: "arxiv",
    listForDate: vi.fn(async () => ({
      kind: "ok" as const, channels: ["astro-ph.GA"],
      papers: [{
        paperKey: "arxiv:" + id, source: "arxiv", externalId: id,
        title: "Research paper " + id, authors: "A. Researcher", abstract: "Abstract " + id,
        categories: ["astro-ph.GA"], canonicalUrl: "https://arxiv.org/abs/" + id,
      }],
    })),
    fetchContent: vi.fn(async () => ({
      abstract: "Abstract " + id,
      sections: [{ heading: "Methods", text: "Full paper evidence " + id }],
      quality: "full" as const, canonicalUrl: "https://arxiv.org/abs/" + id,
    })),
  };
  const selectorCalls: Array<Array<{ role: string; content: string }>> = [];
  const llm = {
    call: vi.fn(async (messages: Array<{ role: string; content: string }>) => {
      if (messages[0]!.content.includes("relevanceScore")) return JSON.stringify({ papers: [decision] });
      selectorCalls.push(messages);
      return selectorAnswers[Math.min(selectorCalls.length, selectorAnswers.length) - 1]!;
    }),
  };
  const writer = {
    dailyExists: vi.fn(async () => false),
    dailyPath: (day: string) => `arxiv-daily/daily/${day}.md`,
    writeDaily: vi.fn(async (day: string, content: string) => { files[`arxiv-daily/daily/${day}.md`] = content; }),
    paperDetailPath: (paperId: string) => `arxiv-daily/papers/${paperId}.md`,
    paperDetailExists: vi.fn(async () => false),
    paperDetailLink: (paperId: string) => `[[${paperId}]]`,
  };
  const pipeline = new ArxivPipeline({
    markupParser, fetcher: {} as PipelineDeps["fetcher"], paperFetcher: {} as PipelineDeps["paperFetcher"],
    sourceAdapter: source, writer: writer as unknown as PipelineDeps["writer"],
    llm: llm as unknown as PipelineDeps["llm"], logger: new Logger("error"),
    arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: [normalizeTopic({
      name: "topic-a", tag: "topic-a", detail: true,
      directions: [{ id: "topic-a-direction", text: "A continuing research direction", origin: "manual" }],
    })] },
    advanced: DEFAULT_SETTINGS.advanced, output: DEFAULT_SETTINGS.output, llmSettings: DEFAULT_SETTINGS.llm,
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
    summarizeDaily: vi.fn(async (papers: ReadonlyArray<{ id: string }>) => ({
      markdown: "# Daily report\n\n" + papers.map((paper) => `### ${paper.id}\n`).join("\n"),
      slots: [],
    })),
  });
  return { pipeline, selectorCalls, writer };
}

describe("daily run when paper notes cannot be chosen", () => {
  it("says so on the completed result instead of silently writing no notes", async () => {
    const h = harness(["not json"]);
    const result = await h.pipeline.runForDate(date);
    expect(result).toMatchObject({
      kind: "completed",
      papersWritten: 1,
      detailSelectionSkipped: expect.stringContaining("failed validation 2 times"),
    });
    expect(h.selectorCalls).toHaveLength(2);
    expect(h.writer.writeDaily).toHaveBeenCalledTimes(1);
  });

  it("carries no such note when the selector answered properly", async () => {
    const h = harness([JSON.stringify({ papers: [{ id, score: 50, reason: "peripheral" }] })]);
    const result = await h.pipeline.runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(result).not.toHaveProperty("detailSelectionSkipped");
    expect(h.selectorCalls).toHaveLength(1);
  });
});
