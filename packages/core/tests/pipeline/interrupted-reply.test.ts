import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LlmClient } from "../../src/llm/client";
import { ArxivPipeline, type PipelineResult } from "../../src/pipeline/pipeline";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import { normalizeTopic } from "../../src/settings/topics";
import type { HttpClient, HttpRequest, HttpResponse } from "../../src/core/adapters";
import { markupParser } from "../markup-parser";

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(resolve(here, "../fixtures/arxiv-recent-astroph.html"), "utf8");

const ID = "2605.08080";
const arxiv = {
  ...DEFAULT_SETTINGS.arxiv,
  topics: [
    normalizeTopic({
      id: "t1", name: "Test Topic", tag: "test", detail: true,
      directions: [{ id: "d1", text: "test topic", origin: "manual" }],
    }),
  ],
};

const PARTIAL_NOTE = `# Title ${ID}\n\n- **arXiv**: https://arxiv.org/abs/${ID}\n\n## 研究问题\n${"部分内容".repeat(120)}\n\n## 方法设计\n${"被截断".repeat(60)}`;

const data = (event: unknown) => `data: ${JSON.stringify(event)}\n\n`;
const reply = (content: string, finish?: string) =>
  data({ choices: [{ delta: { content }, ...(finish ? { finish_reason: finish } : {}) }] }) + "data: [DONE]\n\n";
const ok = (bodyText: string): HttpResponse => ({ status: 200, headers: {}, bodyText });

type Kind = "filter" | "detail-selection" | "detail" | "summary";

function setup(wire: Partial<Record<Kind, string>>) {
  const requests: Kind[] = [];
  const http: HttpClient = {
    request: async (req: HttpRequest) => {
      const system = (JSON.parse(String(req.body)) as { messages: Array<{ content: string }> }).messages[0]!.content;
      const kind: Kind = system.includes("判断它命中了哪些方向") ? "filter"
        : system.includes("strict research-paper evaluator") ? "detail-selection"
        : system.includes("详细的中文论文总结") ? "detail" : "summary";
      requests.push(kind);
      const custom = wire[kind];
      if (custom !== undefined) return ok(custom);
      if (kind === "filter") {
        return ok(reply(JSON.stringify({ papers: [{ id: ID, category: "test", directions: ["test#1"], relevanceScore: 80 }] }), "stop"));
      }
      if (kind === "detail-selection") {
        return ok(reply(JSON.stringify({ papers: [{ id: ID, score: 95, reason: "strong contribution" }] }), "stop"));
      }
      if (kind === "summary") {
        return ok(reply(JSON.stringify({
          id: ID, coreProblem: "problem", keyMethod: "method", mainResult: "result",
          whyRelevant: "value", limitations: "limits",
        }), "stop"));
      }
      return ok(reply(PARTIAL_NOTE, "stop"));
    },
  };
  const logger = new Logger("error");
  const writer = {
    writeDaily: vi.fn(async () => "arxiv-daily/daily/2026-05-11.md"),
    writePaperDetail: vi.fn(async (paper: { id: string }) => `arxiv-daily/papers/${paper.id}.md`),
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
  const paperFetcher = {
    fetch: vi.fn(async () => ({ abstractConclusion: "## Abstract\nabstract", fullSections: "## Results\nfull text" })),
  };
  const pipeline = new ArxivPipeline({
    markupParser,
    fetcher: fetcher as any,
    paperFetcher: paperFetcher as any,
    writer: writer as any,
    llm: new LlmClient({ ...DEFAULT_SETTINGS.llm, apiKey: "sk-test-key", model: "test" }, logger, http),
    logger,
    arxiv,
    advanced: DEFAULT_SETTINGS.advanced,
    output: DEFAULT_SETTINGS.output,
    llmSettings: DEFAULT_SETTINGS.llm,
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
  });
  const run = async (options?: Parameters<ArxivPipeline["runForDate"]>[2]): Promise<PipelineResult> => {
    const result = pipeline.runForDate("2026-05-11", undefined, options);
    await vi.runAllTimersAsync();
    return result;
  };
  return { writer, requests, run };
}

describe("a reply the provider did not finish never reaches the report", () => {
  afterEach(() => vi.useRealTimers());

  it("writes a complete detail note and links it (control)", async () => {
    vi.useFakeTimers();
    const { writer, run } = setup({});

    await expect(run()).resolves.toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(writer.writePaperDetail).toHaveBeenCalledTimes(1);
    expect((writer.writeDaily.mock.calls[0] as unknown as [string, string])[1]).toContain(`[[${ID}]]`);
  });

  it.each([
    ["the provider's content filter stops it", reply(PARTIAL_NOTE, "content_filter"), 1],
    ["the provider reports an interrupted generation", reply(PARTIAL_NOTE, "insufficient_system_resource"), 2],
    ["the connection closes cleanly in the middle of it", data({ choices: [{ delta: { content: PARTIAL_NOTE } }] }), 2],
    ["a gateway reports an upstream failure after the text", data({ choices: [{ delta: { content: PARTIAL_NOTE } }] }) + data({ error: { message: "upstream disconnected" } }) + "data: [DONE]\n\n", 3],
  ])("does not persist a detail note when %s", async (_name, wireNote, attempts) => {
    vi.useFakeTimers();
    const { writer, requests, run } = setup({ detail: wireNote });

    await expect(run()).resolves.toMatchObject({ kind: "completed", papersWritten: 1 });

    expect(writer.writePaperDetail).not.toHaveBeenCalled();
    expect(requests.filter((kind) => kind === "detail")).toHaveLength(attempts);
    const markdown = (writer.writeDaily.mock.calls[0] as unknown as [string, string])[1];
    expect(markdown).not.toContain(`[[${ID}]]`);
    // The day itself is still reported, with the paper's own summary.
    expect(markdown).toContain("- **研究问题**: problem");
  });

  it("retries the day later, keeping finished work, when a paper summary is cut off", async () => {
    vi.useFakeTimers();
    const cut = data({ choices: [{ delta: { content: '{"id": "2605.08080", "coreProblem": "prob' } }] });
    const { writer, requests, run } = setup({ summary: cut });

    const result = await run({ retryTransportFallbacks: true });

    expect(result).toMatchObject({ kind: "failed_transient" });
    expect((result as { reason: string }).reason).toContain(ID);
    expect(writer.writeDaily).not.toHaveBeenCalled();
    // One re-send of the cut-off request, not three validation rounds on a fragment.
    expect(requests.filter((kind) => kind === "summary")).toHaveLength(2);
  });

  it("falls back to the abstract on the final attempt instead of parsing a fragment", async () => {
    vi.useFakeTimers();
    const cut = data({ choices: [{ delta: { content: '{"id": "2605.08080", "coreProblem": "prob' } }] });
    const { writer, run } = setup({ summary: cut });

    await expect(run({ retryTransportFallbacks: false })).resolves.toMatchObject({ kind: "completed" });

    const markdown = (writer.writeDaily.mock.calls[0] as unknown as [string, string])[1];
    expect(markdown).toContain(`<!-- arxiv-daily-fallback:${ID} -->`);
  });
});
