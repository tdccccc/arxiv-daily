import { describe, expect, it, vi } from "vitest";
import { LlmContentBlockedError } from "../src/llm/client";
import { filterPapers } from "../src/pipeline/paper-filter";
import { DailyFilterCheckpointStore } from "../src/services/daily-filter-checkpoint-store";
import type { PaperMeta } from "../src/pipeline/arxiv-parser";
import type { StorageAdapter } from "../src/core/adapters";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeTopic } from "../src/settings/topics";
import type { ArxivSettings } from "../src/settings/types";

const FLAGGED = "FLAGGED-CONTENT";

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
const scope = { reportDate: "2026-08-01", llmSettings: DEFAULT_SETTINGS.llm };

function paper(n: number, flagged = false, abstractChars = 10): PaperMeta {
  return {
    id: `2608.${String(n).padStart(5, "0")}`,
    title: `Paper ${n}`,
    authors: "A",
    abstract: flagged ? `${FLAGGED} ${"x".repeat(abstractChars)}` : "x".repeat(abstractChars),
  };
}

function idsOf(messages: any[]): string[] {
  return [...String(messages[1].content).matchAll(/^ID: (\S+)$/gm)].map((match) => match[1]!);
}

/** Keeps every paper it is shown, under the topic's first direction. */
function keepAll(messages: any[]): string {
  return JSON.stringify({
    papers: idsOf(messages).map((id) => ({
      id, category: "photo-z", directions: ["photo-z#1"], relevanceScore: 80,
    })),
  });
}

/** A provider whose content filter refuses any request that carries flagged text. */
function refusingProvider() {
  return vi.fn(async (messages: any[]) => {
    if (String(messages[1].content).includes(FLAGGED)) {
      throw new LlmContentBlockedError("Content Exists Risk", 400);
    }
    return keepAll(messages);
  });
}

function memoryStorage() {
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
    list: async (dir: string) =>
      Object.keys(files)
        .filter((path) => path.startsWith(`${dir}/`) && !path.slice(dir.length + 1).includes("/"))
        .map((path) => ({ path, type: "file" as const })),
  };
  return { files, storage };
}

describe("filterPapers when the provider refuses a request", () => {
  it("finds the refused paper by halving the batch and classifies the others", async () => {
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const llm = { call: refusingProvider() };
    const papers = Array.from({ length: 40 }, (_, i) => paper(i + 1, i + 1 === 17));

    const kept = await filterPapers(papers, { llm: llm as any, logger, arxivSettings, ...scope });

    expect(kept.map(({ id }) => id)).toEqual(
      papers.filter((_, i) => i + 1 !== 17).map(({ id }) => id),
    );
    // Whole day, then one refused half per level: about two requests per halving.
    expect(llm.call.mock.calls.length).toBeLessThanOrEqual(2 * Math.ceil(Math.log2(papers.length)) + 1);
    const sizes = llm.call.mock.calls.map(([messages]) => idsOf(messages).length);
    expect(sizes[0]).toBe(40);
    expect(sizes[1]).toBe(20);
    // Every other paper was classified exactly once by a request that was accepted.
    const accepted = llm.call.mock.calls.filter(([messages]) => !String(messages[1].content).includes(FLAGGED));
    expect(accepted.flatMap(([messages]) => idsOf(messages)).sort())
      .toEqual(papers.filter((_, i) => i + 1 !== 17).map(({ id }) => id));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("2608.00017"));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("set aside"));
  });

  it("sets the refused paper aside in the classification the caller sees", async () => {
    const papers = [paper(1), paper(2, true), paper(3)];
    const classified = vi.fn();

    await filterPapers(papers, {
      llm: { call: refusingProvider() } as any,
      logger: new Logger("error"),
      arxivSettings,
      onClassified: classified,
      ...scope,
    });

    const { records } = classified.mock.calls[0]![0];
    expect(records.map((record: { id: string }) => record.id)).toEqual(["2608.00001", "2608.00003"]);
  });

  it("keeps the narrowed-down result as the batch checkpoint, so a resume asks nothing", async () => {
    const { storage } = memoryStorage();
    const store = new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output);
    const checkpoint = { lookupReusable: vi.fn(store.lookupReusable.bind(store)), save: vi.fn(store.save.bind(store)) };
    const papers = Array.from({ length: 6 }, (_, i) => paper(i + 1, i + 1 === 4));
    const deps = { logger: new Logger("error"), arxivSettings, checkpointStore: checkpoint, ...scope };

    const first = { call: refusingProvider() };
    const kept = await filterPapers(papers, { ...deps, llm: first as any });
    expect(kept.map(({ id }) => id)).not.toContain("2608.00004");
    expect(checkpoint.save).toHaveBeenCalledTimes(1);
    expect(checkpoint.save.mock.calls[0]![2].map((record: { id: string }) => record.id))
      .toEqual(["2608.00001", "2608.00002", "2608.00003", "2608.00005", "2608.00006"]);

    const second = { call: vi.fn() };
    const again = await filterPapers(papers, { ...deps, llm: second as any });
    expect(second.call).not.toHaveBeenCalled();
    expect(again.map(({ id }) => id)).toEqual(kept.map(({ id }) => id));
  });

  it("narrows down several refused papers", async () => {
    const llm = { call: refusingProvider() };
    const flagged = new Set([3, 11, 12]);
    const papers = Array.from({ length: 16 }, (_, i) => paper(i + 1, flagged.has(i + 1)));

    const kept = await filterPapers(papers, { llm: llm as any, logger: new Logger("error"), arxivSettings, ...scope });

    expect(kept.map(({ id }) => id)).toEqual(
      papers.filter((_, i) => !flagged.has(i + 1)).map(({ id }) => id),
    );
  });

  it("narrows down each batch of a very large day on its own", async () => {
    const llm = { call: refusingProvider() };
    const papers = Array.from({ length: 8 }, (_, i) => paper(i + 1, i + 1 === 2 || i + 1 === 7, 100_000));

    const kept = await filterPapers(papers, { llm: llm as any, logger: new Logger("error"), arxivSettings, ...scope });

    expect(kept.map(({ id }) => id)).toEqual(
      papers.filter((_, i) => i + 1 !== 2 && i + 1 !== 7).map(({ id }) => id),
    );
  });

  it("gives up with the original refusal when more papers are refused than a few strays", async () => {
    const logger = new Logger("error");
    const llm = { call: vi.fn(async () => { throw new LlmContentBlockedError("Content Exists Risk", 400); }) };
    const papers = Array.from({ length: 200 }, (_, i) => paper(i + 1));

    const error = await filterPapers(papers, { llm: llm as any, logger, arxivSettings, ...scope })
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(LlmContentBlockedError);
    expect((error as Error).message).toBe("Content Exists Risk");
    // A bounded search, not one request per paper.
    expect(llm.call.mock.calls.length).toBeLessThan(40);
  });

  it("sets aside up to eight refused papers in a batch and no more", async () => {
    const run = (refused: number) => filterPapers(
      Array.from({ length: 40 }, (_, i) => paper(i + 1, i % 4 === 0 && i / 4 < refused)),
      { llm: { call: refusingProvider() } as any, logger: new Logger("error"), arxivSettings, ...scope },
    );

    expect(await run(8)).toHaveLength(32);
    await expect(run(9)).rejects.toBeInstanceOf(LlmContentBlockedError);
  });

  it("fails with the original refusal when every paper is refused, never reporting an empty day", async () => {
    const llm = { call: vi.fn(async () => { throw new LlmContentBlockedError("Content Exists Risk", 400); }) };

    await expect(filterPapers([paper(1), paper(2), paper(3)], {
      llm: llm as any, logger: new Logger("error"), arxivSettings, ...scope,
    })).rejects.toBeInstanceOf(LlmContentBlockedError);
    await expect(filterPapers([paper(1)], {
      llm: llm as any, logger: new Logger("error"), arxivSettings, ...scope,
    })).rejects.toBeInstanceOf(LlmContentBlockedError);
  });

  it("does not narrow down other failures", async () => {
    const llm = { call: vi.fn(async () => { throw Object.assign(new Error("account suspended"), { status: 403 }); }) };

    await expect(filterPapers([paper(1), paper(2), paper(3)], {
      llm: llm as any, logger: new Logger("error"), arxivSettings, ...scope,
    })).rejects.toThrow("account suspended");
    expect(llm.call).toHaveBeenCalledTimes(1);
  });
});
