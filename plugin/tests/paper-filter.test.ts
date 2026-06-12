import { describe, it, expect, vi } from "vitest";
import { filterPapers } from "../src/pipeline/paper-filter";
import { Logger } from "../src/services/logger";
import type { ArxivSettings, Topic } from "../src/settings/types";
import type { PaperMeta } from "../src/pipeline/arxiv-parser";

vi.mock("obsidian", () => ({
  Notice: class { constructor() {} },
  normalizePath: (p: string) => p,
  requestUrl: vi.fn(),
}));

function makeTopics(): Topic[] {
  return [
    { id: "t1", name: "Photo-z",     tag: "photo-z",        description: "photo-z methods", detail: true },
    { id: "t2", name: "Galaxy",      tag: "galaxy-cluster", description: "cluster surveys", detail: true },
    { id: "t3", name: "ML in Astro", tag: "ml-astro",       description: "ML/DL in astro", detail: false },
  ];
}

function makeArxiv(topics: Topic[]): ArxivSettings {
  return { category: "astro-ph", categories: ["astro-ph"], topics, timezone: "UTC" };
}

const samplePaper: PaperMeta = {
  id: "2601.12345",
  title: "A new photo-z method",
  authors: "X. Author et al.",
  abstract: "We propose ...",
};

describe("filterPapers", () => {
  it("returns [] without calling LLM when topics is empty", async () => {
    const llm = { call: vi.fn() };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv([]),
    });
    expect(out).toEqual([]);
    expect(llm.call).not.toHaveBeenCalled();
  });

  it("includes topic list with [DETAIL] markers in the system prompt", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "photo-z", detail: true }] }),
      ),
    };
    await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    const sys = llm.call.mock.calls[0][0][0].content as string;
    expect(sys).toContain("- photo-z [DETAIL]: photo-z methods");
    expect(sys).toContain("- galaxy-cluster [DETAIL]: cluster surveys");
    expect(sys).toContain("- ml-astro: ML/DL in astro");
    expect(sys).toContain("photo-z|galaxy-cluster|ml-astro|skip");
  });

  it("keeps papers with valid tag and respects detail flag", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "photo-z", detail: true }] }),
      ),
    };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    expect(out).toHaveLength(1);
    expect(out[0].category).toBe("photo-z");
    expect(out[0].isDetail).toBe(true);
  });

  it("demotes detail=true when the chosen topic has detail=false", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "ml-astro", detail: true }] }),
      ),
    };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    expect(out).toHaveLength(1);
    expect(out[0].category).toBe("ml-astro");
    expect(out[0].isDetail).toBe(false);
  });

  it("drops papers with category 'skip'", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "skip", detail: false }] }),
      ),
    };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    expect(out).toEqual([]);
  });

  it("drops papers with an unknown tag", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "nope", detail: false }] }),
      ),
    };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    expect(out).toEqual([]);
  });
});
