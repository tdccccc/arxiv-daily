import { describe, expect, it, vi } from "vitest";
import {
  FILTER_BATCH_CHAR_BUDGET,
  filterPapers,
  splitPapersForFilter,
} from "../src/pipeline/paper-filter";
import { DailyFilterCheckpointStore } from "../src/services/daily-filter-checkpoint-store";
import type { PaperMeta } from "../src/pipeline/arxiv-parser";
import type { StorageAdapter } from "../src/core/adapters";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeTopic } from "../src/settings/topics";
import type { ArxivSettings } from "../src/settings/types";

const arxivSettings: ArxivSettings = {
  category: "astro-ph",
  categories: ["astro-ph"],
  timezone: "UTC",
  topics: [
    normalizeTopic({
      id: "t1", name: "Photo-z", tag: "photo-z", detail: true,
      directions: [{ id: "d1", text: "photo-z methods", origin: "manual" }],
    }),
  ],
};

function paper(n: number, abstractChars = 10): PaperMeta {
  return {
    id: `2608.${String(n).padStart(5, "0")}`,
    title: `Paper ${n}`,
    authors: "A",
    abstract: "x".repeat(abstractChars),
  };
}

function memoryStorage(withList = true) {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const storage: StorageAdapter = {
    normalizePath: (path) => path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, ""),
    exists: async (path) => path in files || dirs.has(path),
    readText: async (path) => {
      if (!(path in files)) throw new Error(`missing ${path}`);
      return files[path]!;
    },
    writeText: async (path, content) => { files[path] = content; },
    mkdir: async (path) => { dirs.add(path); },
    remove: async (path) => { delete files[path]; dirs.delete(path); },
    rename: async (from, to) => { files[to] = files[from]!; delete files[from]; },
    ...(withList
      ? {
          list: async (dir: string) =>
            Object.keys(files)
              .filter((path) => path.startsWith(`${dir}/`) && !path.slice(dir.length + 1).includes("/"))
              .map((path) => ({ path, type: "file" as const })),
        }
      : {}),
  };
  return { files, storage };
}

function answerFor(messages: any[]): string {
  const ids = [...String(messages[1].content).matchAll(/^ID: (\S+)$/gm)].map((m) => m[1]!);
  return JSON.stringify({
    papers: ids.map((id, index) => (index % 2 === 0
      ? { id, category: "photo-z", directions: ["photo-z#1"], relevanceScore: 80 }
      : { id, category: "skip", directions: [], relevanceScore: 0 })),
  });
}

const scope = { reportDate: "2026-08-01", llmSettings: DEFAULT_SETTINGS.llm };

describe("splitPapersForFilter", () => {
  it("keeps an ordinary day in one batch and preserves order", () => {
    const papers = Array.from({ length: 250 }, (_, i) => paper(i + 1, 1500));
    const batches = splitPapersForFilter(papers);
    expect(batches).toHaveLength(1);
    expect(batches[0]).toEqual(papers);
  });

  it("splits an oversized day into ordered batches within the budget", () => {
    const papers = Array.from({ length: 8 }, (_, i) => paper(i + 1, 100_000));
    const batches = splitPapersForFilter(papers);
    expect(batches.length).toBeGreaterThan(1);
    expect(batches.flat()).toEqual(papers);
    for (const batch of batches) {
      const size = batch.reduce((sum, p) => sum + p.id.length + p.title.length + p.abstract.length + 32, 0);
      expect(size <= FILTER_BATCH_CHAR_BUDGET || batch.length === 1).toBe(true);
    }
  });

  it("gives a single over-budget paper its own batch", () => {
    const batches = splitPapersForFilter([paper(1, 5), paper(2, FILTER_BATCH_CHAR_BUDGET * 2), paper(3, 5)]);
    expect(batches.map((b) => b.length)).toEqual([1, 1, 1]);
  });
});

describe("filterPapers batching", () => {
  it("sends an ordinary day as one request with the unchanged checkpoint call shape", async () => {
    const { storage } = memoryStorage();
    const base = new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output);
    const store = { lookupReusable: vi.fn(base.lookupReusable.bind(base)), save: vi.fn(base.save.bind(base)) };
    const llm = { call: vi.fn(async (messages: any[]) => answerFor(messages)) };
    const papers = [paper(1), paper(2), paper(3)];

    const kept = await filterPapers(papers, { llm: llm as any, logger: new Logger("error"), arxivSettings, checkpointStore: store, ...scope });

    expect(llm.call).toHaveBeenCalledTimes(1);
    expect(kept.map((p) => p.id)).toEqual(["2608.00001", "2608.00003"]);
    expect(store.lookupReusable.mock.calls[0]).toHaveLength(2);
    expect(store.save.mock.calls[0]).toHaveLength(3);
  });

  it("classifies a very large day in several requests and merges results in paper order", async () => {
    const { storage } = memoryStorage();
    const store = new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output);
    const llm = { call: vi.fn(async (messages: any[]) => answerFor(messages)) };
    const papers = Array.from({ length: 8 }, (_, i) => paper(i + 1, 100_000));

    const kept = await filterPapers(papers, { llm: llm as any, logger: new Logger("error"), arxivSettings, checkpointStore: store, ...scope });

    expect(llm.call.mock.calls.length).toBeGreaterThan(1);
    const sentIds = llm.call.mock.calls.flatMap(([m]) => [...String(m[1].content).matchAll(/^ID: (\S+)$/gm)].map((x) => x[1]));
    expect(sentIds).toEqual(papers.map((p) => p.id));
    expect(kept.every((p) => p.topicDirections?.[0]?.id === "d1")).toBe(true);
    expect(kept.map((p) => p.id)).toEqual([...kept.map((p) => p.id)].sort());
  });

  it("resumes a multi-batch day from per-batch checkpoints after a failure", async () => {
    const { files, storage } = memoryStorage();
    const store = new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output);
    const papers = Array.from({ length: 8 }, (_, i) => paper(i + 1, 100_000));
    let calls = 0;
    const flaky = {
      call: vi.fn(async (messages: any[]) => {
        calls += 1;
        if (calls === 3) throw new Error("provider hiccup");
        return answerFor(messages);
      }),
    };
    const deps = { logger: new Logger("error"), arxivSettings, checkpointStore: store, ...scope };

    await expect(filterPapers(papers, { ...deps, llm: flaky as any })).rejects.toThrow("provider hiccup");
    expect(Object.keys(files).filter((p) => p.includes("filter-checkpoints") && p.endsWith(".json"))).toHaveLength(2);

    const healthy = { call: vi.fn(async (messages: any[]) => answerFor(messages)) };
    const kept = await filterPapers(papers, { ...deps, llm: healthy as any });

    expect(healthy.call.mock.calls.length).toBe(flaky.call.mock.calls.length - 2);
    expect(kept.length).toBeGreaterThan(0);
  });

  it.each([true, false])("removeAll deletes every batch checkpoint (list support: %s)", async (withList) => {
    const { files, storage } = memoryStorage(withList);
    const store = new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output);
    const llm = { call: vi.fn(async (messages: any[]) => answerFor(messages)) };
    const papers = Array.from({ length: 8 }, (_, i) => paper(i + 1, 100_000));
    await filterPapers(papers, { llm: llm as any, logger: new Logger("error"), arxivSettings, checkpointStore: store, ...scope });
    expect(Object.keys(files).some((p) => p.includes(".batch-"))).toBe(true);

    await store.removeAll("2026-08-01");

    expect(Object.keys(files).filter((p) => p.includes("filter-checkpoints"))).toEqual([]);
  });
});
