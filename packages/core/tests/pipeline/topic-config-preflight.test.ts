import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { ArxivPipeline } from "../../src/pipeline/pipeline";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import { normalizeTopic } from "../../src/settings/topics";
import { markupParser } from "../markup-parser";

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(resolve(here, "../fixtures/arxiv-recent-astroph.html"), "utf8");
const ID = "2605.08080";

const topic = (name: string, tag: string, direction = "test topic") => normalizeTopic({
  id: tag, name, tag, detail: false, directions: [{ id: "d1", text: direction, origin: "manual" }],
});

function run(topics: ReturnType<typeof normalizeTopic>[]) {
  const logger = new Logger("error");
  const llm = {
    call: vi.fn(async (messages: Array<{ content: string }>) => {
      const system = messages[0]!.content;
      if (system.includes("判断它命中了哪些方向")) {
        return JSON.stringify({ papers: [{ id: ID, category: topics[0]!.tag, directions: [`${topics[0]!.tag}#1`], relevanceScore: 80 }] });
      }
      return JSON.stringify({
        id: ID, coreProblem: "problem", keyMethod: "method", mainResult: "result",
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
        id, title: `Title ${id}`, authors: "Author et al.", abstract: "abstract",
        published: "2026-05-11T02:28:06Z", updated: "2026-05-11T02:34:08Z",
        primaryCategory: "astro-ph.GA", categories: ["astro-ph.GA"],
      }])),
    ),
    fetchAbstractsByIds: vi.fn(),
    fetchBySubmittedDate: vi.fn(),
    fetchPaperHtml: vi.fn(),
    fetchPaperAbsPage: vi.fn(),
  };
  const pipeline = new ArxivPipeline({
    markupParser,
    fetcher: fetcher as any,
    paperFetcher: { fetch: vi.fn(async () => ({ abstractConclusion: "## Abstract\nabstract", fullSections: null })) } as any,
    writer: writer as any,
    llm: llm as any,
    logger,
    arxiv: { ...DEFAULT_SETTINGS.arxiv, topics },
    advanced: DEFAULT_SETTINGS.advanced,
    output: DEFAULT_SETTINGS.output,
    llmSettings: DEFAULT_SETTINGS.llm,
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
  });
  return { result: pipeline.runForDate("2026-05-11"), llm, writer };
}

describe("a topic list the report cannot be built from", () => {
  it.each([
    ["two topics share a display name", [topic("Photo-z", "photo-z"), topic("Photo-z", "photo-z-2")], 'Topics 1 and 2 are both named "Photo-z"'],
    ["a direction is longer than 1000 characters", [topic("Photo-z", "photo-z", "x".repeat(1001))], 'Topic 1 "Photo-z" direction 1 is 1001 characters long'],
    ["a tag is longer than 120 characters", [topic("Photo-z", "t".repeat(121))], 'Topic 1 "Photo-z" tag is 121 characters long'],
  ])("fails permanently before any model call when %s", async (_name, topics, reason) => {
    const { result, llm, writer } = run(topics);

    await expect(result).resolves.toEqual({
      kind: "failed_permanent",
      reason: expect.stringContaining(reason),
    });
    expect(await result).toMatchObject({ reason: expect.stringMatching(/^invalid topic settings: /) });
    expect(llm.call).not.toHaveBeenCalled();
    expect(writer.writeDaily).not.toHaveBeenCalled();
  });

  it("lists every problem so the researcher can fix them in one go", async () => {
    const { result } = run([topic("Dup", "dup", "x".repeat(1200)), topic("Dup", "dup-2")]);

    const outcome = await result;
    expect(outcome.kind).toBe("failed_permanent");
    expect((outcome as { reason: string }).reason).toContain('both named "Dup"');
    expect((outcome as { reason: string }).reason).toContain("direction 1 is 1200 characters long");
  });

  it("still writes the report with the longest direction and tag the report can carry", async () => {
    const { result, llm, writer } = run([topic("Photo-z", "t".repeat(120), "x".repeat(1000))]);

    await expect(result).resolves.toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(llm.call).toHaveBeenCalled();
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
  });

  it("does not fail a topic that merely has no directions: the filter has always skipped those", async () => {
    const empty = normalizeTopic({ id: "empty", name: "Empty", tag: "empty", detail: false, directions: [] });
    const { result, writer } = run([topic("Photo-z", "photo-z"), empty]);

    await expect(result).resolves.toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
  });
});
