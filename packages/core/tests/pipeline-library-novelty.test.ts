import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { ArxivPipeline, type PipelineDeps } from "../src/pipeline/pipeline";
import type { FilterRecord } from "../src/pipeline/paper-filter";
import {
  LibraryPriorWorkUnavailableError,
  type LibraryPriorWorkPort,
  type NoveltyRepresentativePaper,
} from "../src/pipeline/personalized-novelty";
import { assembleDailySummary } from "../src/pipeline/daily-summary-assembler";
import { DailyFilterCheckpointStore } from "../src/services/daily-filter-checkpoint-store";
import { throwIfCancelled } from "../src/services/cancellation";
import { NoopProgressReporter } from "../src/services/progress";
import { PaperIndexStore } from "../src/services/paper-index";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeTopic } from "../src/settings/topics";
import type { SourceAdapter } from "../src/sources";
import { markupParser } from "./markup-parser";

// Exercise the stage on an announcement weekday; weekends stop before discovery.
const date = "2026-09-07";
const id = (index: number): string => `2609.${String(index).padStart(5, "0")}`;
const decision = (index: number, relevanceScore: number): FilterRecord => ({
  id: id(index), category: "topic-a", relevanceScore, directions: ["topic-a#1"],
});
const libraryPaper = (index: number): NoveltyRepresentativePaper => ({
  paperKey: `arxiv:2501.${String(index).padStart(5, "0")}`,
  title: `Library paper ${index}`,
  authors: ["L. Reader"],
  abstract: `Library abstract ${index}`,
  published: "2025-01-02",
  categories: ["astro-ph.GA"],
});
const NOVELTY_EXPLANATION = "Adds a measurement the library papers do not report.";
const isNoveltyCall = (messages: Array<{ content: string }>): boolean =>
  messages[1]?.content.startsWith("Compare this new paper") ?? false;

function noveltyPayload(messages: Array<{ content: string }>): {
  paper: { paperKey: string };
  basis: Array<{ paperKey: string }>;
} {
  const body = messages[1]!.content.split("<paper_data>\n")[1]!.split("\n</paper_data>")[0]!;
  return JSON.parse(body);
}

type Port = LibraryPriorWorkPort["findSimilar"];

function harness(records: FilterRecord[]) {
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
  const paperIndex = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);
  const filterStore = new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output);
  const source: SourceAdapter = {
    sourceId: "arxiv",
    listForDate: vi.fn(async () => ({
      kind: "ok" as const, channels: ["astro-ph.GA"],
      papers: records.map(({ id }) => ({
        paperKey: "arxiv:" + id, source: "arxiv", externalId: id,
        title: "Research paper " + id, authors: "A. Researcher", abstract: "Abstract " + id,
        categories: ["astro-ph.GA"], canonicalUrl: "https://arxiv.org/abs/" + id,
      })),
    })),
    fetchContent: vi.fn(async (id) => ({
      abstract: "Abstract " + id,
      sections: [{ heading: "Methods", text: "Full paper evidence " + id }],
      quality: "full" as const, canonicalUrl: "https://arxiv.org/abs/" + id,
    })),
  };
  const noveltyAnswer = vi.fn((messages: Array<{ content: string }>): string => {
    const [first] = noveltyPayload(messages).basis;
    return JSON.stringify({
      differenceType: "new-method", comparisonBasis: [first!.paperKey],
      evidenceDepth: "metadata-and-abstract", explanation: NOVELTY_EXPLANATION,
    });
  });
  const llm = {
    call: vi.fn(async (messages: Array<{ content: string }>) => {
      if (isNoveltyCall(messages)) return noveltyAnswer(messages);
      if (messages[0]!.content.includes("relevanceScore")) return JSON.stringify({ papers: records });
      // Detail selection is real; an empty evaluation chooses no extra note.
      return JSON.stringify({ papers: [] });
    }),
  };
  // Real assembly so the report carries exactly the markers the pipeline attached.
  const summarize = vi.fn<NonNullable<PipelineDeps["summarizeDaily"]>>(async (papers, dateStr, deps) => {
    const slots = papers.map((paper) => ({
      paper: {
        id: paper.id, title: paper.title, authors: paper.authors, category: paper.category,
        sourceSections: "Abstract", isDetail: false, personalNovelty: paper.personalNovelty,
      },
      result: {
        kind: "fallback" as const, reasonCode: "validation-exhausted" as const,
        attempts: 3, originalAbstract: paper.abstract,
      },
    }));
    return {
      markdown: assembleDailySummary({
        slots, dateStr, arxivSettings: deps.arxivSettings, summaryLanguage: "en",
      }),
      slots,
    };
  });
  const writer = {
    dailyExists: vi.fn(async () => false),
    dailyPath: (day: string) => `arxiv-daily/daily/${day}.md`,
    writeDaily: vi.fn(async (day: string, content: string) => { files[`arxiv-daily/daily/${day}.md`] = content; }),
    paperDetailPath: (paperId: string) => `arxiv-daily/papers/${paperId}.md`,
    paperDetailExists: vi.fn(async () => false),
    paperDetailLink: (paperId: string) => `[[${paperId}]]`,
  };
  const logger = new Logger("error");
  const warn = vi.spyOn(logger, "warn");
  const info = vi.spyOn(logger, "info");
  const progress = new NoopProgressReporter();
  const setStage = vi.spyOn(progress, "setStage");
  const makePipeline = (
    options: {
      libraryPriorWork?: LibraryPriorWorkPort; maximum?: number; sourceAdapter?: SourceAdapter;
      summaryLanguage?: "zh" | "en";
    } = {},
  ): ArxivPipeline => new ArxivPipeline({
    markupParser, fetcher: {} as PipelineDeps["fetcher"], paperFetcher: {} as PipelineDeps["paperFetcher"],
    sourceAdapter: options.sourceAdapter ?? source, writer: writer as unknown as PipelineDeps["writer"],
    llm: llm as unknown as PipelineDeps["llm"], logger, progress,
    arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: [normalizeTopic({
      name: "topic-a", tag: "topic-a", detail: true,
      directions: [{ id: "topic-a-direction", text: "A continuing research direction", origin: "manual" }],
    })] },
    advanced: DEFAULT_SETTINGS.advanced,
    output: {
      ...DEFAULT_SETTINGS.output,
      ...(options.maximum === undefined ? {} : { maxDailyPapers: options.maximum }),
      ...(options.summaryLanguage === undefined ? {} : { summaryLanguage: options.summaryLanguage }),
    },
    llmSettings: DEFAULT_SETTINGS.llm,
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
    summarizeDaily: summarize,
    checkpointStores: { filter: filterStore },
    paperIndex,
    ...(options.libraryPriorWork ? { libraryPriorWork: options.libraryPriorWork } : {}),
  });
  const noveltyCalls = () => llm.call.mock.calls.filter(([messages]) => isNoveltyCall(messages));
  const report = () => files[`arxiv-daily/daily/${date}.md`];
  // Per-paper outcome lines of the novelty stage, in log order.
  const outcomes = () => info.mock.calls
    .filter(([message]) => message.startsWith("pipeline: personal novelty 2609."))
    .map(([message]) => message.replace("pipeline: personal novelty ", ""));
  return { files, source, llm, noveltyAnswer, summarize, writer, paperIndex, warn, info, outcomes, setStage, makePipeline, noveltyCalls, report };
}

const portOf = (findSimilar: Port): LibraryPriorWorkPort => ({ findSimilar: vi.fn(findSimilar) });
const twoPapers: Port = async () => [libraryPaper(1), libraryPaper(2)];
const markerCount = (markdown: string | undefined): number =>
  (markdown?.match(/^<!-- arxiv-daily-personal-novelty:v[12]:/gm) ?? []).length;

describe("library novelty stage", () => {
  it("writes a marker and visible line from retrieved library papers and indexes the novelty", async () => {
    const h = harness([decision(1, 90)]);
    const port = portOf(twoPapers);
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(markerCount(h.report())).toBe(1);
    expect(h.report()).toContain("> Personal novelty:");
    expect(h.report()).toContain("Library paper 1 (arxiv:2501\\.00001)");
    expect(h.noveltyCalls()).toHaveLength(1);
    expect(noveltyPayload(h.noveltyCalls()[0]![0]).basis.map(({ paperKey }) => paperKey))
      .toEqual(["arxiv:2501.00001", "arxiv:2501.00002"]);
    const index = JSON.parse(h.files["arxiv-daily/.index/papers.json"]!);
    expect(index.papers["arxiv:" + id(1)].noveltyByReport[`arxiv-daily/daily/${date}.md`])
      .toMatchObject({
        differenceType: "new-method",
        comparisonBasis: ["arxiv:2501.00001"],
        // Titles ride in the marker, so the index (a projection of the report) has them.
        comparisonBasisTitles: { "arxiv:2501.00001": "Library paper 1" },
      });
  });

  it("compares against a file: library document: title-only visible line, marker and index keep the key", async () => {
    const fileKey = `file:sha256:${"ab".repeat(32)}`;
    const fileDocument: NoveltyRepresentativePaper = {
      paperKey: fileKey, title: "My library PDF", abstract: "Extracted from a PDF.",
    };
    const h = harness([decision(1, 90)]);
    const port = portOf(async () => [fileDocument]);
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    // The model sees exactly title and abstract for the file document.
    const [basis] = noveltyPayload(h.noveltyCalls()[0]![0]).basis;
    expect(basis).toEqual(fileDocument);
    expect(markerCount(h.report())).toBe(1);
    const visible = h.report()!.split("\n").find((line) => line.startsWith("> Personal novelty:"))!;
    expect(visible).toContain("My library PDF");
    expect(visible).not.toContain("file:sha256");
    expect(visible).not.toContain("ab".repeat(32));
    const index = JSON.parse(h.files["arxiv-daily/.index/papers.json"]!);
    expect(index.papers["arxiv:" + id(1)].noveltyByReport[`arxiv-daily/daily/${date}.md`])
      .toMatchObject({ comparisonBasis: [fileKey], comparisonBasisTitles: { [fileKey]: "My library PDF" } });
  });

  it("asks the port for the digest paper's title and abstract with a limit of five and the run signal", async () => {
    const h = harness([decision(1, 90)]);
    const port = portOf(twoPapers);
    const controller = new AbortController();
    await h.makePipeline({ libraryPriorWork: port }).runForDate(date, controller.signal);
    expect(port.findSimilar).toHaveBeenCalledWith(
      { paperKey: "arxiv:" + id(1), title: "Research paper " + id(1), abstract: "Abstract " + id(1) },
      5,
      controller.signal,
    );
  });

  it("skips the stage entirely when no port is supplied: no calls, no progress, unchanged report", async () => {
    const without = harness([decision(1, 90), decision(2, 80)]);
    await without.makePipeline().runForDate(date);
    expect(without.noveltyCalls()).toHaveLength(0);
    expect(without.setStage.mock.calls.map(([stage]) => stage)).not.toContain("personal-novelty");
    expect(markerCount(without.report())).toBe(0);

    // A port that finds nothing must leave the report byte-identical.
    const empty = harness([decision(1, 90), decision(2, 80)]);
    await empty.makePipeline({ libraryPriorWork: portOf(async () => []) }).runForDate(date);
    expect(empty.noveltyCalls()).toHaveLength(0);
    expect(empty.report()).toBe(without.report());
  });

  it("does not run for non-arXiv sources", async () => {
    const h = harness([decision(1, 90)]);
    const port = portOf(twoPapers);
    const other: SourceAdapter = { ...h.source, sourceId: "other" };
    const result = await h.makePipeline({ libraryPriorWork: port, sourceAdapter: other }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed" });
    expect(port.findSimilar).not.toHaveBeenCalled();
    expect(h.noveltyCalls()).toHaveLength(0);
  });

  it("leaves a paper without a marker when its retrieval fails, and still completes the others", async () => {
    const h = harness([decision(1, 90), decision(2, 80)]);
    const port = portOf(async (query) => {
      if (query.paperKey === "arxiv:" + id(1)) throw new Error("index read failed");
      return [libraryPaper(1)];
    });
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 2 });
    expect(markerCount(h.report())).toBe(1);
    expect(h.noveltyCalls()).toHaveLength(1);
    expect(noveltyPayload(h.noveltyCalls()[0]![0]).paper.paperKey).toBe("arxiv:" + id(2));
    expect(h.warn).toHaveBeenCalled();
  });

  it("stops retrieving for the remaining papers once the port reports it is unavailable", async () => {
    const h = harness([decision(1, 90), decision(2, 80), decision(3, 70)]);
    const port = portOf(async () => { throw new LibraryPriorWorkUnavailableError("authorization revoked"); });
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 3 });
    expect(port.findSimilar).toHaveBeenCalledTimes(1);
    expect(h.noveltyCalls()).toHaveLength(0);
    expect(markerCount(h.report())).toBe(0);
    expect(h.warn.mock.calls.filter(([message]) => message.includes("library prior work"))).toHaveLength(1);
  });

  it("warns once and stops looking up the remaining papers when every lookup fails the same way", async () => {
    const h = harness(Array.from({ length: 20 }, (_, index) => decision(index + 1, index + 1)));
    const port = portOf(async () => { throw new TypeError("searchGenerationBm25: query has more than 64 unique terms"); });
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 20 });
    expect(port.findSimilar).toHaveBeenCalledTimes(3);
    expect(h.noveltyCalls()).toHaveLength(0);
    expect(markerCount(h.report())).toBe(0);
    const warnings = h.warn.mock.calls.filter(([message]) => message.includes("library lookup"));
    expect(warnings).toHaveLength(1);
    expect(warnings[0]![0]).toContain("TypeError");
    expect(warnings[0]![0]).toContain("more than 64 unique terms");
    // No per-paper stack traces: the only pipeline warning is the single systematic one.
    expect(h.warn.mock.calls.filter(([message]) => message.startsWith("pipeline:"))).toHaveLength(1);
  });

  it("keeps looking up papers after isolated or differing failures, warning for each", async () => {
    const h = harness(Array.from({ length: 6 }, (_, index) => decision(index + 1, 100 - index)));
    let calls = 0;
    const port = portOf(async () => {
      calls += 1;
      // Two same-class failures, a different class, then a success, then a failure again.
      if (calls === 1 || calls === 2) throw new TypeError("bad query");
      if (calls === 3) throw new RangeError("odd index");
      if (calls === 4) return [libraryPaper(1)];
      throw new TypeError("bad query");
    });
    await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(port.findSimilar).toHaveBeenCalledTimes(6);
    expect(markerCount(h.report())).toBe(1);
    expect(h.warn.mock.calls.filter(([message]) => message.includes("library lookup failed for"))).toHaveLength(5);
  });

  it("logs the real reason for a paper without an abstract and does not search for it", async () => {
    const h = harness([decision(1, 90), decision(2, 80)]);
    const withoutAbstract: SourceAdapter = {
      ...h.source,
      listForDate: vi.fn(async () => ({
        kind: "ok" as const, channels: ["astro-ph.GA"],
        papers: [1, 2].map((index) => ({
          paperKey: "arxiv:" + id(index), source: "arxiv", externalId: id(index),
          title: "Research paper " + id(index), authors: "A. Researcher",
          abstract: index === 1 ? "  " : "Abstract " + id(index),
          categories: ["astro-ph.GA"], canonicalUrl: "https://arxiv.org/abs/" + id(index),
        })),
      })),
    };
    const port = portOf(twoPapers);
    await h.makePipeline({ libraryPriorWork: port, sourceAdapter: withoutAbstract }).runForDate(date);
    const searched = vi.mocked(port.findSimilar).mock.calls.map(([query]) => query.paperKey);
    expect(searched).toEqual(["arxiv:" + id(2)]);
    const skipped = h.warn.mock.calls.map(([message]) => message).filter((message) => message.includes("skipped"));
    expect(skipped).toHaveLength(1);
    expect(skipped[0]).toContain(id(1));
    expect(skipped[0]).toContain("missing abstract");
    expect(skipped[0]).not.toContain("not comparable");
  });

  it("still compares papers retrieved before the port became unavailable", async () => {
    const h = harness([decision(1, 90), decision(2, 80)]);
    let calls = 0;
    const port = portOf(async () => {
      calls += 1;
      if (calls === 2) throw new LibraryPriorWorkUnavailableError("authorization revoked");
      return [libraryPaper(1)];
    });
    await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(port.findSimilar).toHaveBeenCalledTimes(2);
    expect(markerCount(h.report())).toBe(1);
  });

  it("makes no model call and writes no marker for a paper whose basis is empty", async () => {
    const h = harness([decision(1, 90), decision(2, 80)]);
    const port = portOf(async (query) => query.paperKey === "arxiv:" + id(1) ? [] : [libraryPaper(1)]);
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 2 });
    expect(h.noveltyCalls()).toHaveLength(1);
    expect(markerCount(h.report())).toBe(1);
  });

  it("drops a malformed basis for one paper without losing the others", async () => {
    const h = harness([decision(1, 90), decision(2, 80)]);
    const port = portOf(async (query) => query.paperKey === "arxiv:" + id(1)
      ? [{ ...libraryPaper(1), title: "" }]
      : [libraryPaper(1)]);
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 2 });
    expect(markerCount(h.report())).toBe(1);
  });

  it("writes no marker when the model says no retrieved paper is comparable, and does not ask again on rerun", async () => {
    const h = harness([decision(1, 90)]);
    h.noveltyAnswer.mockReturnValue(JSON.stringify({ noComparablePrior: true }));
    h.summarize.mockRejectedValueOnce(new Error("interrupted before the report was committed"));
    const port = portOf(twoPapers);
    await expect(h.makePipeline({ libraryPriorWork: port }).runForDate(date))
      .resolves.toMatchObject({ kind: "failed_transient" });
    expect(h.noveltyCalls()).toHaveLength(1);
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(h.noveltyCalls()).toHaveLength(1);
    expect(markerCount(h.report())).toBe(0);
  });

  it("reuses a saved novelty on rerun when the library basis is unchanged", async () => {
    const h = harness([decision(1, 90)]);
    h.summarize.mockRejectedValueOnce(new Error("interrupted before the report was committed"));
    const port = portOf(twoPapers);
    await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(h.noveltyCalls()).toHaveLength(1);
    expect(markerCount(h.report())).toBe(1);
  });

  it("degrades a model transport failure to no marker and retries it on rerun", async () => {
    const h = harness([decision(1, 90)]);
    h.noveltyAnswer.mockImplementationOnce(() => { throw new Error("socket hang up"); });
    h.summarize.mockRejectedValueOnce(new Error("interrupted before the report was committed"));
    const port = portOf(twoPapers);
    await expect(h.makePipeline({ libraryPriorWork: port }).runForDate(date))
      .resolves.toMatchObject({ kind: "failed_transient" });
    expect(h.noveltyCalls()).toHaveLength(1);
    expect(h.warn).toHaveBeenCalled();
    // The failed attempt was not saved, so the rerun asks the model again.
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(h.noveltyCalls()).toHaveLength(2);
    expect(markerCount(h.report())).toBe(1);
  });

  it("completes the run without a marker when the model fails during novelty", async () => {
    const h = harness([decision(1, 90)]);
    h.noveltyAnswer.mockImplementation(() => { throw new Error("socket hang up"); });
    const result = await h.makePipeline({ libraryPriorWork: portOf(twoPapers) }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(markerCount(h.report())).toBe(0);
  });

  it("compares at most the daily cap of papers, not every paper the filter kept", async () => {
    const h = harness(Array.from({ length: 25 }, (_, index) => decision(index + 1, index + 1)));
    const port = portOf(twoPapers);
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(result).toMatchObject({ kind: "completed", papersWritten: 20 });
    expect(port.findSimilar).toHaveBeenCalledTimes(20);
    expect(h.noveltyCalls()).toHaveLength(20);
    const queried = vi.mocked(port.findSimilar).mock.calls.map(([query]) => query.paperKey);
    expect(queried).not.toContain("arxiv:" + id(1));
    expect(markerCount(h.report())).toBe(20);
  });

  it("keeps novelty text out of the email digest", async () => {
    const h = harness([decision(1, 90)]);
    const result = await h.makePipeline({ libraryPriorWork: portOf(twoPapers) }).runForDate(date);
    expect(markerCount(h.report())).toBe(1);
    expect(result.kind).toBe("completed");
    const digest = JSON.stringify((result as { digest?: unknown }).digest);
    expect(digest).toBeDefined();
    expect(digest).not.toContain(NOVELTY_EXPLANATION);
    expect(digest).not.toContain("Library paper");
    expect(digest).not.toMatch(/novelty/i);
  });

  it("runs between the filter and content fetching and reports progress per paper", async () => {
    const h = harness([decision(1, 90), decision(2, 80)]);
    const port = portOf(twoPapers);
    await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    const stages = h.setStage.mock.calls.map(([stage]) => stage);
    const first = stages.indexOf("personal-novelty");
    expect(first).toBeGreaterThan(stages.indexOf("filter"));
    expect(first).toBeLessThan(stages.indexOf("fetch-content"));
    expect(h.setStage.mock.calls.filter(([stage]) => stage === "personal-novelty").map(([, current, total]) => [current, total]))
      .toEqual([[1, 2], [2, 2]]);
    const lastRetrieval = Math.max(...vi.mocked(port.findSimilar).mock.invocationCallOrder);
    expect(lastRetrieval).toBeLessThan(Math.min(...vi.mocked(h.source.fetchContent).mock.invocationCallOrder));
  });

  it("returns cancelled when the run is cancelled during retrieval", async () => {
    const h = harness([decision(1, 90)]);
    const controller = new AbortController();
    const port = portOf(async (_query, _limit, signal) => {
      controller.abort();
      throwIfCancelled(signal);
      return [libraryPaper(1)];
    });
    const result = await h.makePipeline({ libraryPriorWork: port }).runForDate(date, controller.signal);
    expect(result.kind).toBe("cancelled");
    expect(h.writer.writeDaily).not.toHaveBeenCalled();
  });
});

describe("library novelty stage language and sampling", () => {
  it("asks for the explanation in the daily summary language", async () => {
    for (const [summaryLanguage, phrase] of [["zh", "Simplified Chinese"], ["en", "English"]] as const) {
      const h = harness([decision(1, 90)]);
      await h.makePipeline({ libraryPriorWork: portOf(twoPapers), summaryLanguage }).runForDate(date);
      expect(h.noveltyCalls()).toHaveLength(1);
      expect(h.noveltyCalls()[0]![0][0]!.content).toContain(`written in ${phrase}`);
    }
  });

  it("does not reuse a saved result after the summary language changes", async () => {
    const h = harness([decision(1, 90)]);
    h.summarize.mockRejectedValueOnce(new Error("interrupted before the report was committed"));
    const port = portOf(twoPapers);
    await h.makePipeline({ libraryPriorWork: port, summaryLanguage: "zh" }).runForDate(date);
    await h.makePipeline({ libraryPriorWork: port, summaryLanguage: "en" }).runForDate(date);
    expect(h.noveltyCalls()).toHaveLength(2);
  });

  it("calls the model deterministically like the other structured-JSON stages", async () => {
    const h = harness([decision(1, 90)]);
    await h.makePipeline({ libraryPriorWork: portOf(twoPapers) }).runForDate(date);
    expect(h.noveltyCalls()[0]![1]).toMatchObject({ temperature: 0 });
  });
});

describe("library novelty stage outcome log", () => {
  it("logs one line per paper: written and no-comparable-prior, without warnings or stack traces", async () => {
    const h = harness([decision(1, 90), decision(2, 80)]);
    h.noveltyAnswer.mockImplementation((messages) => noveltyPayload(messages).paper.paperKey === "arxiv:" + id(2)
      ? JSON.stringify({ noComparablePrior: true })
      : JSON.stringify({
        differenceType: "new-method", comparisonBasis: ["arxiv:2501.00001"],
        evidenceDepth: "metadata-and-abstract", explanation: NOVELTY_EXPLANATION,
      }));
    await h.makePipeline({ libraryPriorWork: portOf(twoPapers) }).runForDate(date);
    expect(h.outcomes()).toEqual([`${id(1)}: written`, `${id(2)}: no-comparable-prior`]);
    // Outcome lines carry no error object and no warning is raised for expected outcomes.
    for (const call of h.info.mock.calls.filter(([message]) => message.startsWith("pipeline: personal novelty 2609."))) {
      expect(call).toHaveLength(1);
    }
    expect(h.warn.mock.calls.filter(([message]) => /novelty|library/.test(message))).toHaveLength(0);
  });

  it("logs the skip reason for a paper without an abstract or a library match", async () => {
    const h = harness([decision(1, 90), decision(2, 80)]);
    const withoutAbstract: SourceAdapter = {
      ...h.source,
      listForDate: vi.fn(async () => ({
        kind: "ok" as const, channels: ["astro-ph.GA"],
        papers: [1, 2].map((index) => ({
          paperKey: "arxiv:" + id(index), source: "arxiv", externalId: id(index),
          title: "Research paper " + id(index), authors: "A. Researcher",
          abstract: index === 1 ? "  " : "Abstract " + id(index),
          categories: ["astro-ph.GA"], canonicalUrl: "https://arxiv.org/abs/" + id(index),
        })),
      })),
    };
    await h.makePipeline({ libraryPriorWork: portOf(async () => []), sourceAdapter: withoutAbstract }).runForDate(date);
    expect(h.outcomes()).toEqual([`${id(1)}: skipped:missing abstract`, `${id(2)}: skipped:no similar library paper`]);
  });

  it("logs a failed lookup by error class and the papers never looked up as skipped", async () => {
    const h = harness(Array.from({ length: 5 }, (_, index) => decision(index + 1, 100 - index)));
    const port = portOf(async () => { throw new TypeError("bad query"); });
    await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(h.outcomes()).toEqual([
      `${id(1)}: failed:TypeError`, `${id(2)}: failed:TypeError`, `${id(3)}: failed:TypeError`,
      `${id(4)}: skipped:lookup stopped`, `${id(5)}: skipped:lookup stopped`,
    ]);
  });

  it("logs a model failure as failed with its reason", async () => {
    const h = harness([decision(1, 90)]);
    h.noveltyAnswer.mockImplementation(() => { throw new Error("socket hang up"); });
    await h.makePipeline({ libraryPriorWork: portOf(twoPapers) }).runForDate(date);
    expect(h.outcomes()).toEqual([`${id(1)}: failed:transport`]);
  });

  it("logs unusable model output as failed:validation-exhausted", async () => {
    const h = harness([decision(1, 90)]);
    h.noveltyAnswer.mockReturnValue("not json");
    await h.makePipeline({ libraryPriorWork: portOf(twoPapers) }).runForDate(date);
    expect(h.outcomes()).toEqual([`${id(1)}: failed:validation-exhausted`]);
  });

  it("logs the outcome again when a rerun reuses the saved result", async () => {
    const h = harness([decision(1, 90)]);
    h.summarize.mockRejectedValueOnce(new Error("interrupted before the report was committed"));
    const port = portOf(twoPapers);
    await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    await h.makePipeline({ libraryPriorWork: port }).runForDate(date);
    expect(h.outcomes()).toEqual([`${id(1)}: written`, `${id(1)}: written`]);
  });
});

describe("pipeline module boundary", () => {
  it("never imports library modules: the library arrives only through the injected port", async () => {
    const directory = join(__dirname, "../src/pipeline");
    const names = (await readdir(directory)).filter((name) => name.endsWith(".ts"));
    expect(names).toContain("library-novelty-stage.ts");
    for (const name of names) {
      const source = await readFile(join(directory, name), "utf8");
      expect(source, name).not.toMatch(/(?:from\s*|import\s*\(\s*|import\s+)["']\.\.\/library/);
    }
  });
});
