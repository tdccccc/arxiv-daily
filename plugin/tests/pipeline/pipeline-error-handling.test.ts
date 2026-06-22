import { describe, it, expect, vi } from "vitest";
import type { PipelineResult } from "../../src/pipeline/pipeline";
import { ArxivPipeline } from "../../src/pipeline/pipeline";
import { Logger } from "../../src/services/logger";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";

vi.mock("obsidian", () => ({
  Notice: class {
    constructor() {}
  },
  normalizePath: (p: string) => p,
  requestUrl: vi.fn(),
}));

const testArxiv = {
  ...DEFAULT_SETTINGS.arxiv,
  topics: [
    { id: "t1", name: "Test Topic", tag: "test", description: "test topic", detail: true },
  ],
};

/** HTML that parses to a date bucket with 0 papers */
const emptyPapersHtml = `<html><body>
<dl id="articles">
  <h3>Mon, 22 Jun 2026</h3>
</dl>
</body></html>`;

describe("PipelineResult types", () => {
  it("should support pending result kind", () => {
    const result: PipelineResult = { kind: "pending", reason: "no papers from arXiv" };
    expect(result.kind).toBe("pending");
    expect(result.reason).toBe("no papers from arXiv");
  });
});

describe("Pipeline arXiv 0 papers handling", () => {
  it("should return pending when arXiv returns 0 papers", async () => {
    const fetcher = {
      fetchRecent: vi.fn().mockResolvedValue(emptyPapersHtml),
      fetchMetadataByIds: vi.fn(),
      fetchAbstractsByIds: vi.fn(),
      fetchBySubmittedDate: vi.fn(),
      fetchPaperHtml: vi.fn(),
      fetchPaperAbsPage: vi.fn(),
    };
    const writer = {
      writeDaily: vi.fn(),
      writePaperDetail: vi.fn(),
      writeEmptyDaily: vi.fn(),
      dailyPath: vi.fn(),
      paperDetailPath: vi.fn(),
      paperDetailLink: vi.fn(),
      dailyExists: vi.fn(async () => false),
      paperDetailExists: vi.fn(async () => false),
    };
    const llm = { call: vi.fn() };
    const paperFetcher = { fetch: vi.fn() };
    const logger = new Logger("error");

    const pipeline = new ArxivPipeline({
      fetcher,
      paperFetcher,
      writer,
      llm,
      logger,
      arxiv: testArxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });

    const result = await pipeline.runForDate("2026-06-22");
    expect(result).toEqual({ kind: "pending", reason: "no papers from arXiv" });
    expect(writer.writeEmptyDaily).not.toHaveBeenCalled();
  });
});
