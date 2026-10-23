import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { PaperContentFetcher } from "../../src/pipeline/paper-content";
import { ArxivPipeline } from "../../src/pipeline/pipeline";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import { normalizeTopic } from "../../src/settings/topics";
import { markupParser } from "../markup-parser";

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(resolve(here, "../fixtures/arxiv-recent-astroph.html"), "utf8");

const ID = "2605.08080";
const LISTING_ABSTRACT = "We measure the photometric redshift bias of 4,000 galaxies and find a 3% offset.";
const arxiv = {
  ...DEFAULT_SETTINGS.arxiv,
  topics: [
    normalizeTopic({
      id: "t1", name: "Test Topic", tag: "test", detail: false,
      directions: [{ id: "d1", text: "test topic", origin: "manual" }],
    }),
  ],
};

/** arxiv.org answers 503 for the rendered HTML, the source package and the abs page. */
function downFetcher() {
  const down = () => { throw new Error("HTTP 503 Service Unavailable"); };
  return {
    fetchPaperHtml: vi.fn(async () => down()),
    fetchSource: vi.fn(async () => down()),
    fetchPaperAbsPage: vi.fn(async () => down()),
  };
}

function setup(listingAbstract: string) {
  const logger = new Logger("error");
  const cache = {
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
  };
  const paperFetcher = new PaperContentFetcher(downFetcher() as any, cache as any, logger, markupParser);
  const requests: Array<{ system: string; user: string }> = [];
  const llm = {
    call: vi.fn(async (messages: Array<{ content: string }>) => {
      const system = messages[0]!.content;
      const user = messages[1]!.content;
      requests.push({ system, user });
      if (system.includes("判断它命中了哪些方向")) {
        return JSON.stringify({ papers: [{ id: ID, category: "test", directions: ["test#1"], relevanceScore: 80 }] });
      }
      return JSON.stringify({
        id: ID, coreProblem: "grounded problem", keyMethod: "method", mainResult: "result",
        whyRelevant: "value", limitations: "limits",
      });
    }),
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
  };
  const fetcher = {
    fetchRecent: vi.fn().mockResolvedValue(recentHtml),
    fetchMetadataByIds: vi.fn(async (ids: string[]) =>
      new Map(ids.map((id) => [id, {
        id, title: `Title ${id}`, authors: "Author et al.", abstract: listingAbstract,
        published: "2026-05-11T02:28:06Z", updated: "2026-05-11T02:34:08Z",
        primaryCategory: "astro-ph.GA", categories: ["astro-ph.GA"],
      }])),
    ),
    fetchAbstractsByIds: vi.fn(),
    fetchBySubmittedDate: vi.fn(),
    fetchPaperHtml: vi.fn(),
    fetchPaperAbsPage: vi.fn(),
  };
  const checkpoint = {
    lookupReusable: vi.fn(async () => null),
    upsert: vi.fn(async () => undefined),
    removeAll: vi.fn(async () => undefined),
  };
  const pipeline = new ArxivPipeline({
    markupParser,
    fetcher: fetcher as any,
    paperFetcher,
    writer: writer as any,
    checkpointStores: { summary: checkpoint },
    llm: llm as any,
    logger,
    arxiv,
    advanced: DEFAULT_SETTINGS.advanced,
    output: DEFAULT_SETTINGS.output,
    llmSettings: DEFAULT_SETTINGS.llm,
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
  });
  const summaryRequests = () => requests.filter(({ system }) => !system.includes("判断它命中了哪些方向"));
  return { pipeline, writer, checkpoint, summaryRequests };
}

describe("a day on which arxiv.org content cannot be fetched", () => {
  it("summarizes from the listing abstract, not from the failure placeholder", async () => {
    const { pipeline, writer, summaryRequests, checkpoint } = setup(LISTING_ABSTRACT);

    const result = await pipeline.runForDate("2026-05-11");

    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    const requests = summaryRequests();
    expect(requests).toHaveLength(1);
    expect(requests[0]!.user).toContain(LISTING_ABSTRACT);
    expect(requests[0]!.user).toContain("Source sections: Abstract");
    expect(requests[0]!.user).not.toContain("获取失败");
    const markdown = (writer.writeDaily.mock.calls[0] as unknown as [string, string])[1];
    expect(markdown).toContain("- **研究问题**: grounded problem");
    expect(markdown).toContain("信息来源： Abstract");
    expect(markdown).not.toContain("获取失败");
    // What is kept for resuming is the summary of the abstract, under the fingerprint of that input.
    const saved = checkpoint.upsert.mock.calls[0] as unknown as [string, { paper: { abstract: string } }, { kind: string }];
    expect(saved[2].kind).toBe("structured");
    expect(saved[1].paper.abstract).toBe(LISTING_ABSTRACT);
  });

  it("shows a fallback block, and asks the model nothing, when there is no abstract either", async () => {
    const { pipeline, writer, summaryRequests, checkpoint } = setup("");

    const result = await pipeline.runForDate("2026-05-11");

    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(summaryRequests()).toHaveLength(0);
    const markdown = (writer.writeDaily.mock.calls[0] as unknown as [string, string])[1];
    expect(markdown).toContain(`<!-- arxiv-daily-fallback:${ID} -->`);
    expect(markdown).toContain("其中 1 篇使用回退内容。");
    expect(markdown).toContain("无法获取这篇论文的内容");
    expect(markdown).toContain("信息来源： 获取失败");
    expect(markdown).not.toContain("grounded problem");
    const saved = checkpoint.upsert.mock.calls[0] as unknown as [string, unknown, { kind: string; reasonCode?: string }];
    expect(saved[2]).toMatchObject({ kind: "fallback", reasonCode: "content-unavailable" });
  });
});
