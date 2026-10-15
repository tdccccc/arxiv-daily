import { describe, expect, it, vi } from "vitest";
import type { StorageAdapter } from "../src/core/adapters";
import { ArxivPipeline, type PipelineDeps } from "../src/pipeline/pipeline";
import type { FilterRecord } from "../src/pipeline/paper-filter";
import { DailyFilterCheckpointStore } from "../src/services/daily-filter-checkpoint-store";
import {
  FilterDecisionRecordStore,
  type FilterDecisionRecordPort,
} from "../src/services/filter-decision-record-store";
import { PaperIndexStore } from "../src/services/paper-index";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeTopic } from "../src/settings/topics";
import type { SourceAdapter } from "../src/sources";
import { markupParser } from "./markup-parser";

const date = "2026-09-07";
const recordPath = `arxiv-daily/.index/filter-decisions/${date}.json`;
const checkpointPath = `arxiv-daily/.index/filter-checkpoints/${date}.json`;
const id = (index: number): string => `2609.${String(index).padStart(5, "0")}`;
const decision = (index: number, relevanceScore: number, category = "topic-a"): FilterRecord => ({
  id: id(index), category, relevanceScore, directions: [`${category}#1`],
});
const skipped = (index: number): FilterRecord => ({
  id: id(index), category: "skip", relevanceScore: 0, directions: [],
});

function harness(initial: FilterRecord[]) {
  let records = initial;
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
    mkdir: async (path) => { directories.add(path); },
    remove: async (path) => { delete files[path]; directories.delete(path); },
    rename: async (from, to) => { files[to] = files[from]!; delete files[from]; },
    list: async (dir) => Object.keys(files)
      .filter((path) => path.startsWith(`${dir}/`))
      .map((path) => ({ path, type: "file" as const })),
  };
  const paperIndex = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);
  const filterStore = new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output);
  const decisionStore = new FilterDecisionRecordStore(storage, DEFAULT_SETTINGS.output, {
    now: () => new Date("2026-09-07T08:00:00.000Z"),
  });
  const source: SourceAdapter = {
    sourceId: "arxiv",
    listForDate: vi.fn(async () => ({
      kind: "ok" as const, channels: ["astro-ph.GA"],
      papers: records.map(({ id }) => ({
        paperKey: "arxiv:" + id, source: "arxiv", externalId: id,
        title: "Research paper " + id, authors: "A. Researcher", abstract: "SECRET-ABSTRACT " + id,
        categories: ["astro-ph.GA"], canonicalUrl: "https://arxiv.org/abs/" + id,
      })),
    })),
    fetchContent: vi.fn(async (id) => ({
      abstract: "Abstract " + id,
      sections: [{ heading: "Methods", text: "Full paper evidence " + id }],
      quality: "full" as const, canonicalUrl: "https://arxiv.org/abs/" + id,
    })),
  };
  const llm = {
    call: vi.fn(async (messages: Array<{ content: string }>) => {
      if (messages[0]!.content.includes("relevanceScore")) return JSON.stringify({ papers: records });
      return JSON.stringify({ papers: [] });
    }),
  };
  const summarize = vi.fn<NonNullable<PipelineDeps["summarizeDaily"]>>(async (papers) => ({
    markdown: "# Daily report\n\n" + papers.map((paper) => `### ${paper.id}\n${paper.title}\n`).join("\n"),
    slots: [],
  }));
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
  const makePipeline = (options: {
    maximum?: number;
    withIndex?: boolean;
    checkpoint?: boolean;
    port?: FilterDecisionRecordPort | null;
  } = {}): ArxivPipeline => new ArxivPipeline({
    markupParser, fetcher: {} as PipelineDeps["fetcher"], paperFetcher: {} as PipelineDeps["paperFetcher"],
    sourceAdapter: source, writer: writer as unknown as PipelineDeps["writer"],
    llm: llm as unknown as PipelineDeps["llm"], logger,
    arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: ["topic-a", "topic-b"].map((tag) => normalizeTopic({
      name: tag, tag, detail: true,
      directions: [{ id: tag + "-direction", text: `Direction of ${tag}`, origin: "manual" }],
    })) },
    advanced: DEFAULT_SETTINGS.advanced,
    output: { ...DEFAULT_SETTINGS.output, ...(options.maximum === undefined ? {} : { maxDailyPapers: options.maximum }) },
    llmSettings: {
      ...DEFAULT_SETTINGS.llm,
      provider: "custom",
      baseUrl: "https://user:hunter2@llm.example.test/private/v1?token=hunter2",
      apiKey: "sk-never-persist",
      model: "model-a",
    },
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
    summarizeDaily: summarize,
    ...(options.checkpoint ? { checkpointStores: { filter: filterStore } } : {}),
    ...(options.port === null ? {} : { filterDecisionStore: options.port ?? decisionStore }),
    ...(options.withIndex ? { paperIndex } : {}),
  });
  return {
    files, source, llm, summarize, writer, paperIndex, warn, filterStore, decisionStore, makePipeline,
    respondWith: (next: FilterRecord[]) => { records = next; },
  };
}

function readRecord(files: Record<string, string>) {
  return JSON.parse(files[recordPath]!);
}

describe("filter decision record on commit", () => {
  it("keeps every candidate with its outcome, including papers the model skipped", async () => {
    const h = harness([
      decision(1, 100), decision(2, 90), decision(3, 80), skipped(4), decision(5, 60, "topic-b"),
    ]);
    await h.paperIndex.upsertFromDailyPaper({
      arxivId: id(1), title: "Ignored paper", authors: "A", date: "2026-09-04",
      arxivCategory: "astro-ph.GA", primaryTopic: "topic-a", detail: false,
    });
    await h.paperIndex.setStatus(id(1), "ignored");

    const result = await h.makePipeline({ maximum: 2, withIndex: true }).runForDate(date);

    expect(result).toMatchObject({ kind: "completed", papersWritten: 2 });
    const record = readRecord(h.files);
    expect(record).toMatchObject({
      schemaVersion: 1,
      reportDate: date,
      writtenAt: "2026-09-07T08:00:00.000Z",
      promptContractVersion: 4,
      resultContractVersion: 3,
      generation: { provider: "custom", model: "model-a" },
    });
    expect(record.directions).toEqual([
      { ref: "topic-a#1", tag: "topic-a", id: "topic-a-direction", text: "Direction of topic-a" },
      { ref: "topic-b#1", tag: "topic-b", id: "topic-b-direction", text: "Direction of topic-b" },
    ]);
    const byId = Object.fromEntries(record.candidates.map((c: { id: string }) => [c.id, c]));
    expect(Object.keys(byId).sort()).toEqual([1, 2, 3, 4, 5].map(id));
    expect(byId[id(1)]).toMatchObject({ outcome: "excluded_ignored", relevanceScore: 100 });
    expect(byId[id(2)]).toEqual({
      id: id(2),
      title: `Research paper ${id(2)}`,
      category: "topic-a",
      directions: [{ id: "topic-a-direction", text: "Direction of topic-a", tag: "topic-a" }],
      relevanceScore: 90,
      outcome: "in_report",
    });
    expect(byId[id(3)]).toMatchObject({ outcome: "in_report" });
    expect(byId[id(5)]).toMatchObject({
      outcome: "cut_by_limit",
      category: "topic-b",
      directions: [{ id: "topic-b-direction", text: "Direction of topic-b", tag: "topic-b" }],
    });
    expect(byId[id(4)]).toEqual({
      id: id(4), title: `Research paper ${id(4)}`, category: "skip",
      directions: [], relevanceScore: 0, outcome: "skipped_by_model",
    });
  });

  it("stores no abstract, request text, key or raw endpoint", async () => {
    const h = harness([decision(1, 90), skipped(2)]);
    await h.makePipeline().runForDate(date);

    const raw = h.files[recordPath]!;
    for (const forbidden of ["SECRET-ABSTRACT", "sk-never-persist", "hunter2", "llm.example.test", "paper_data"]) {
      expect(raw).not.toContain(forbidden);
    }
    expect(readRecord(h.files).generation.endpointDigest).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it("records without a filter checkpoint store and after the checkpoint is cleaned", async () => {
    const without = harness([decision(1, 90), skipped(2)]);
    await without.makePipeline().runForDate(date);
    expect(readRecord(without.files).candidates).toHaveLength(2);

    const withCheckpoint = harness([decision(1, 90), skipped(2)]);
    await withCheckpoint.makePipeline({ checkpoint: true }).runForDate(date);
    expect(withCheckpoint.files[checkpointPath]).toBeUndefined();
    expect(readRecord(withCheckpoint.files).candidates).toHaveLength(2);
  });

  it("writes after the daily report and before the transient checkpoints are removed", async () => {
    const h = harness([decision(1, 90), skipped(2)]);
    const order: string[] = [];
    h.writer.writeDaily.mockImplementation(async () => { order.push("daily"); });
    const removeAll = h.filterStore.removeAll.bind(h.filterStore);
    vi.spyOn(h.filterStore, "removeAll").mockImplementation(async (day) => {
      order.push("cleanup");
      await removeAll(day);
    });
    const port: FilterDecisionRecordPort = {
      save: async (input) => { order.push("record"); await h.decisionStore.save(input); },
    };

    await h.makePipeline({ checkpoint: true, port }).runForDate(date);

    expect(order).toEqual(["daily", "record", "cleanup"]);
  });

  it("includes skipped papers when the classification comes from a reused filter checkpoint", async () => {
    const h = harness([decision(1, 90), skipped(2)]);
    h.summarize.mockRejectedValueOnce(new Error("interrupted before the report was committed"));
    await expect(h.makePipeline({ checkpoint: true }).runForDate(date)).resolves
      .toMatchObject({ kind: "failed_transient" });
    expect(h.files[recordPath]).toBeUndefined();

    await h.makePipeline({ checkpoint: true }).runForDate(date);

    expect(h.llm.call.mock.calls.filter(([m]) => m[0]!.content.includes("relevanceScore"))).toHaveLength(1);
    expect(readRecord(h.files).candidates.map((c: { outcome: string }) => c.outcome).sort())
      .toEqual(["in_report", "skipped_by_model"]);
  });

  it("only warns when the record cannot be written, leaving the report committed", async () => {
    const h = harness([decision(1, 90), skipped(2)]);
    const port: FilterDecisionRecordPort = {
      save: vi.fn(async () => { throw new Error("disk full"); }),
    };

    const result = await h.makePipeline({ checkpoint: true, port }).runForDate(date);

    expect(result).toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(port.save).toHaveBeenCalledTimes(1);
    expect(h.writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(h.warn).toHaveBeenCalledWith(
      expect.stringContaining("filter decision record"),
      expect.any(Error),
    );
    // Checkpoint cleanup still runs after a failed record write.
    expect(h.files[checkpointPath]).toBeUndefined();
  });

  it("replaces the record when the same date is generated again", async () => {
    const h = harness([decision(1, 90), skipped(2)]);
    await h.makePipeline().runForDate(date);
    h.respondWith([decision(1, 40), decision(2, 85)]);

    await h.makePipeline().runForDate(date);

    const record = readRecord(h.files);
    expect(record.candidates.map((c: { id: string; relevanceScore: number }) => [c.id, c.relevanceScore]))
      .toEqual([[id(1), 40], [id(2), 85]]);
    expect(Object.keys(h.files).filter((path) => path.includes("filter-decisions"))).toEqual([recordPath]);
  });

  it("writes nothing when no filter call happens", async () => {
    const existing = harness([decision(1, 90)]);
    existing.writer.dailyExists.mockResolvedValue(true);
    await existing.makePipeline().runForDate(date);
    expect(existing.files[recordPath]).toBeUndefined();

    const empty = harness([]);
    await empty.makePipeline().runForDate(date);
    expect(empty.llm.call).not.toHaveBeenCalled();
    expect(empty.files[recordPath]).toBeUndefined();
  });

  it("writes nothing when the filter or the daily summary fails", async () => {
    const badFilter = harness([decision(1, 90)]);
    badFilter.llm.call.mockResolvedValueOnce("not json");
    await expect(badFilter.makePipeline().runForDate(date)).resolves
      .toMatchObject({ kind: "failed_transient" });
    expect(badFilter.files[recordPath]).toBeUndefined();

    const badSummary = harness([decision(1, 90)]);
    badSummary.summarize.mockRejectedValueOnce(new Error("summary failed"));
    await expect(badSummary.makePipeline().runForDate(date)).resolves
      .toMatchObject({ kind: "failed_transient" });
    expect(badSummary.files[recordPath]).toBeUndefined();
  });

  it("records a day on which the model skipped every candidate", async () => {
    const h = harness([skipped(1), skipped(2)]);

    const result = await h.makePipeline({ checkpoint: true }).runForDate(date);

    expect(result).toMatchObject({ kind: "completed", papersWritten: 0, outcome: "no_matches" });
    expect(h.writer.writeDaily).not.toHaveBeenCalled();
    expect(readRecord(h.files).candidates.map((c: { id: string; outcome: string }) => [c.id, c.outcome]))
      .toEqual([[id(1), "skipped_by_model"], [id(2), "skipped_by_model"]]);
    // Checkpoint handling on this path is unchanged: nothing cleans it up.
    expect(h.files[checkpointPath]).toBeDefined();
  });

  it("records a day on which every kept paper was ignored", async () => {
    const h = harness([decision(1, 90), decision(2, 70, "topic-b"), skipped(3)]);
    for (const [index, tag] of [[1, "topic-a"], [2, "topic-b"]] as const) {
      await h.paperIndex.upsertFromDailyPaper({
        arxivId: id(index), title: "Ignored paper", authors: "A", date: "2026-09-04",
        arxivCategory: "astro-ph.GA", primaryTopic: tag, detail: false,
      });
      await h.paperIndex.setStatus(id(index), "ignored");
    }

    const result = await h.makePipeline({ withIndex: true, checkpoint: true }).runForDate(date);

    expect(result).toMatchObject({ kind: "completed", papersWritten: 0, outcome: "no_matches" });
    expect(h.writer.writeDaily).not.toHaveBeenCalled();
    expect(readRecord(h.files).candidates.map((c: { id: string; outcome: string }) => [c.id, c.outcome]))
      .toEqual([[id(1), "excluded_ignored"], [id(2), "excluded_ignored"], [id(3), "skipped_by_model"]]);
    expect(h.files[checkpointPath]).toBeDefined();
  });

  it("only warns when a no-match day cannot be recorded", async () => {
    const h = harness([skipped(1)]);
    const port: FilterDecisionRecordPort = {
      save: vi.fn(async () => { throw new Error("disk full"); }),
    };

    await expect(h.makePipeline({ port }).runForDate(date)).resolves
      .toMatchObject({ kind: "completed", outcome: "no_matches" });
    expect(h.warn).toHaveBeenCalledWith(expect.stringContaining("filter decision record"), expect.any(Error));
  });

  it("runs unchanged when no record store is wired", async () => {
    const h = harness([decision(1, 90), skipped(2)]);
    await expect(h.makePipeline({ port: null }).runForDate(date)).resolves
      .toMatchObject({ kind: "completed", papersWritten: 1 });
    expect(h.files[recordPath]).toBeUndefined();
  });
});
