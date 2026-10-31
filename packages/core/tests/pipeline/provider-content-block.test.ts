import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { LlmClient } from "../../src/llm/client";
import { ArxivPipeline } from "../../src/pipeline/pipeline";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import { normalizeTopic } from "../../src/settings/topics";
import type { HttpClient, HttpRequest, HttpResponse } from "../../src/core/adapters";
import { markupParser } from "../markup-parser";

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(resolve(here, "../fixtures/arxiv-recent-astroph.html"), "utf8");

const FLAGGED = "FLAGGED-CONTENT";
const PAPERS = ["2605.08080", "2605.08068"];
const arxiv = {
  ...DEFAULT_SETTINGS.arxiv,
  topics: [
    normalizeTopic({
      id: "t1", name: "Test Topic", tag: "test", detail: false,
      directions: [{ id: "d1", text: "test topic", origin: "manual" }],
    }),
  ],
};

const DEEPSEEK_REFUSAL = {
  error: { message: "Content Exists Risk", type: "invalid_request_error", param: null, code: "invalid_request_error" },
};

function stream(content: string): HttpResponse {
  const event = { choices: [{ delta: { content }, finish_reason: "stop" }] };
  return { status: 200, headers: {}, bodyText: `data: ${JSON.stringify(event)}\n\ndata: [DONE]\n\n` };
}

function refusal(status: number, body: unknown): HttpResponse {
  return { status, headers: {}, bodyText: JSON.stringify(body) };
}

type Kind = "filter" | "detail-selection" | "summary";

/**
 * A real LlmClient over a fake provider. `flaggedPaper` is refused (400
 * `refusal`) whenever a request carries its text; `reject` instead answers
 * every request of that kind with the given response.
 */
function setup(options: {
  flaggedPaper?: string;
  /** Stages whose requests the provider refuses when they carry the flagged text. */
  refuseIn?: Kind[];
  refusal?: HttpResponse;
  reject?: Partial<Record<Kind, HttpResponse>>;
} = {}) {
  const requests: Array<{ kind: Kind; ids: string[] }> = [];
  const refused = options.refusal ?? refusal(400, DEEPSEEK_REFUSAL);
  const http: HttpClient = {
    request: async (req: HttpRequest) => {
      const body = JSON.parse(String(req.body)) as { messages: Array<{ content: string }> };
      const system = body.messages[0]!.content;
      const user = body.messages[1]!.content;
      const ids = [...user.matchAll(/^(?:ID: )(\d{4}\.\d{4,5})$/gm)].map((match) => match[1]!);
      const kind: Kind = system.includes("判断它命中了哪些方向")
        ? "filter"
        : system.includes("strict research-paper evaluator") ? "detail-selection" : "summary";
      requests.push({ kind, ids });
      const rejected = options.reject?.[kind];
      if (rejected) return rejected;
      if (options.flaggedPaper && user.includes(FLAGGED) && (options.refuseIn ?? [kind]).includes(kind)) {
        return refused;
      }
      if (kind === "filter") {
        return stream(JSON.stringify({
          papers: PAPERS.filter((id) => ids.includes(id))
            .map((id) => ({ id, category: "test", directions: ["test#1"], relevanceScore: 80 })),
        }));
      }
      if (kind === "detail-selection") return stream(JSON.stringify({ papers: [] }));
      const id = ids[0]!;
      return stream(JSON.stringify({
        id,
        coreProblem: `${id} problem`,
        keyMethod: `${id} method`,
        mainResult: `${id} result`,
        whyRelevant: `${id} value`,
        limitations: `${id} limits`,
      }));
    },
  };
  const logger = new Logger("error");
  const warn = vi.spyOn(logger, "warn");
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
        id,
        title: `Title ${id}`,
        authors: "Author et al.",
        abstract: id === options.flaggedPaper ? `${FLAGGED} abstract` : "abstract",
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
  const paperFetcher = {
    fetch: vi.fn(async (id: string) => ({
      abstractConclusion: id === options.flaggedPaper ? `## Abstract\n${FLAGGED} body` : "## Abstract\nabstract and conclusion",
      fullSections: null,
    })),
  };
  const llm = new LlmClient(
    { ...DEFAULT_SETTINGS.llm, apiKey: "sk-test-key", model: "test" },
    logger,
    http,
  );
  const pipeline = new ArxivPipeline({
    markupParser,
    fetcher: fetcher as any,
    paperFetcher: paperFetcher as any,
    writer: writer as any,
    llm,
    logger,
    arxiv,
    // One summary request at a time, so a permanent failure can be shown to
    // stop the day before any further request is made.
    advanced: { ...DEFAULT_SETTINGS.advanced, generationConcurrency: 1 },
    output: DEFAULT_SETTINGS.output,
    llmSettings: DEFAULT_SETTINGS.llm,
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
  });
  return { pipeline, writer, requests, warn };
}

const daily = (writer: { writeDaily: { mock: { calls: unknown[][] } } }) =>
  writer.writeDaily.mock.calls[0]?.[1] as string;

describe("provider content-safety refusal in the daily pipeline", () => {
  it("sets one refused paper summary aside instead of failing the day", async () => {
    const { pipeline, writer, requests } = setup({ flaggedPaper: PAPERS[1], refuseIn: ["summary"] });

    const result = await pipeline.runForDate("2026-05-11");

    expect(result).toMatchObject({ kind: "completed", papersWritten: 2 });
    const markdown = daily(writer);
    expect(markdown).toContain("<!-- arxiv-daily-fallback:2605.08068 -->");
    expect(markdown).toContain("其中 1 篇使用回退内容。");
    expect(markdown).toContain("模型服务商拒绝为这篇论文生成摘要");
    // The other paper keeps its generated summary.
    expect(markdown).toContain("2605.08080 problem");
    expect(markdown).not.toContain("<!-- arxiv-daily-fallback:2605.08080 -->");
    // One request per stage and paper: the refusal is not retried.
    expect(requests.map(({ kind }) => kind)).toEqual(["filter", "summary", "summary"]);
  });

  it("isolates the refused paper when the whole filter request is refused", async () => {
    const { pipeline, writer, requests, warn } = setup({ flaggedPaper: PAPERS[1], refuseIn: ["filter"] });

    const result = await pipeline.runForDate("2026-05-11");

    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    const markdown = daily(writer);
    expect(markdown).toContain("2605.08080 problem");
    expect(markdown).not.toContain("2605.08068");
    const filterRequests = requests.filter(({ kind }) => kind === "filter");
    expect(filterRequests.length).toBeGreaterThan(1);
    expect(filterRequests[0]!.ids).toContain(PAPERS[1]);
    expect(filterRequests.at(-1)!.ids).not.toContain(PAPERS[1]);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining(PAPERS[1]!));
  });

  it("still fails the filter permanently when the provider refuses every paper", async () => {
    const { pipeline, writer, requests } = setup({ reject: { filter: refusal(400, DEEPSEEK_REFUSAL) } });

    const result = await pipeline.runForDate("2026-05-11");

    expect(result).toMatchObject({
      kind: "failed_permanent",
      reason: "paper filter LLM failed: Content Exists Risk",
    });
    expect(writer.writeDaily).not.toHaveBeenCalled();
    // Bounded: the search for a culprit gives up instead of asking once per paper.
    expect(requests.length).toBeLessThanOrEqual(80);
  });

  it.each([
    [401, { error: { message: "Incorrect API key provided", code: "invalid_api_key" } }],
    [403, { error: { message: "Forbidden" } }],
    [404, { error: { message: "The model `test` does not exist", code: "model_not_found" } }],
  ])("keeps failing the whole day on HTTP %i from the summary request", async (status, body) => {
    const { pipeline, writer, requests } = setup({ reject: { summary: refusal(status, body) } });

    const result = await pipeline.runForDate("2026-05-11");

    expect(result).toMatchObject({
      kind: "failed_permanent",
      reason: `daily summary LLM failed: ${body.error.message}`,
    });
    expect(writer.writeDaily).not.toHaveBeenCalled();
    expect(requests.filter(({ kind }) => kind === "summary")).toHaveLength(1);
  });

  it.each([401, 403, 404])("keeps failing the whole day on HTTP %i from the filter request", async (status) => {
    const { pipeline, writer, requests } = setup({
      reject: { filter: refusal(status, { error: { message: "account problem" } }) },
    });

    await expect(pipeline.runForDate("2026-05-11")).resolves.toMatchObject({
      kind: "failed_permanent",
      reason: "paper filter LLM failed: account problem",
    });
    expect(writer.writeDaily).not.toHaveBeenCalled();
    expect(requests).toHaveLength(1);
  });

  it("does not take a 400 about the request itself for a refusal", async () => {
    const { pipeline, requests } = setup({
      reject: {
        filter: refusal(400, { error: { message: "messages: text content blocks must be non-empty", type: "invalid_request_error" } }),
      },
    });

    await expect(pipeline.runForDate("2026-05-11")).resolves.toMatchObject({
      kind: "failed_permanent",
      reason: "paper filter LLM failed: messages: text content blocks must be non-empty",
    });
    expect(requests).toHaveLength(1);
  });
});
