import { markupParser } from "../markup-parser";
import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { ArxivPipeline } from "../../src/pipeline/pipeline";
import { parseRecent } from "../../src/pipeline/arxiv-parser";
import { summarizeDaily as realSummarizeDaily } from "../../src/pipeline/summarizer";
import {
  parseDailyReportDiscoveryProvenance,
  parseDailyReportPersonalNovelty,
} from "../../src/pipeline/daily-summary-parser";
import type { ChatMessage } from "../../src/llm/client";
import {
  buildDailySummaryCheckpointFingerprintInput,
  createDailySummaryCompatibilityFingerprint,
} from "../../src/services/daily-summary-checkpoint-store";
import { Logger } from "../../src/services/logger";
import { RunCancelledError } from "../../src/services/cancellation";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type { PersonalizedDiscoveryInput } from "../../src/pipeline/personalized-paper-filter";
import * as personalizedNovelty from "../../src/pipeline/personalized-novelty";
import {
  PERSONAL_NOVELTY_MAX_OUTPUT_CODE_UNITS,
  type NoveltyCheckpointRecord,
  type PersonalNoveltyMatchInput,
  type PersonalizedNoveltyInput,
} from "../../src/pipeline/personalized-novelty";

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(
  resolve(here, "../fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

const testDetailSelection = {
  normalThreshold: 70,
  exceptionalThreshold: 90,
  softLimit: 2,
};

const testArxiv = {
  ...DEFAULT_SETTINGS.arxiv,
  topics: [
    { id: "t1", name: "Photo-z", tag: "photo-z", description: "photo-z methods", detail: true },
  ],
};

const DATE = "2026-05-11";

function firstBucketIds(): string[] {
  const bucket = parseRecent(recentHtml, markupParser)
    .find((entry) => entry.announceDate === DATE);
  if (!bucket) throw new Error(`fixture bucket not found: ${DATE}`);
  return bucket.papers.map((paper) => paper.id);
}

function atomMeta(id: string) {
  return {
    id,
    title: `Atom title ${id}`,
    authors: "Atom Author et al.",
    abstract: "atom abstract",
    published: "2026-02-02T02:28:06Z",
    updated: "2026-06-15T02:34:08Z",
    primaryCategory: "astro-ph.GA",
    categories: ["astro-ph.GA"],
  };
}

function structuredDailyResponse(messages: ChatMessage[]): string | null {
  const system = messages[0]?.content ?? "";
  if (
    !system.includes("严格 JSON 对象") &&
    !system.includes("strict JSON object")
  ) {
    return null;
  }
  const id = /ID: (\d{4}\.\d{4,5})/.exec(messages[1]?.content ?? "")?.[1];
  if (!id) throw new Error("daily summary test input is missing an ID");
  return JSON.stringify({
    id,
    coreProblem: `${id} problem`,
    keyMethod: `${id} method`,
    mainResult: `${id} result`,
    whyRelevant: `${id} value`,
    limitations: `${id} limits`,
  });
}

function noveltyResponse(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    differenceType: "new-method",
    comparisonBasis: ["arxiv:2501.00001"],
    evidenceDepth: "metadata-and-abstract",
    explanation: "Introduces a method absent from the representative abstracts.",
    ...overrides,
  });
}

const discoveryOne: PersonalizedDiscoveryInput = {
  directions: [{
    id: "direction.001",
    name: "Direction one",
    description: "Strict personalized direction",
    discoveryCues: ["strict discovery"],
    representatives: [{
      paperKey: "arxiv:2501.00001",
      title: "Representative one",
      evidenceDepth: "metadata-and-abstract",
    }],
  }],
};

function noveltyInput(paperKeys: string[]): PersonalizedNoveltyInput {
  return {
    papers: paperKeys.map((paperKey) => ({
      paperKey,
      title: `New paper ${paperKey}`,
      abstract: `Abstract ${paperKey}`,
    })),
    representatives: [{
      paperKey: "arxiv:2501.00001",
      title: "Prior paper one",
      authors: ["Author One"],
      abstract: "Prior abstract one",
      published: "2026-08-01T00:00:00.000Z",
      categories: ["cs.AI"],
    }],
  };
}

function noveltyMatches(): PersonalNoveltyMatchInput {
  return {
    paperMatches: [],
    directionRepresentatives: [{
      directionId: "direction.001",
      representativePaperKeys: ["arxiv:2501.00001"],
    }],
  };
}

interface NoveltyHarnessOptions {
  keptIds?: string[];
  matchedIds?: string[];
  discovery?: PersonalizedDiscoveryInput;
  input?: PersonalizedNoveltyInput;
  matches?: PersonalNoveltyMatchInput;
  onNovelty?: (messages: ChatMessage[], callNumber: number) => string;
  filterStores?: Record<string, unknown>;
  summaryStore?: Record<string, unknown>;
  summarizeDaily?: (...args: any[]) => Promise<{ markdown: string; slots: any[] }>;
  llmSettings?: typeof DEFAULT_SETTINGS.llm;
}

function makePipeline(options: NoveltyHarnessOptions = {}) {
  const ids = firstBucketIds();
  const keptIds = options.keptIds ?? [ids[0]!, ids[1]!];
  const matchedIds = options.matchedIds ?? [ids[0]!];
  const writes: Record<string, string> = {};
  let noveltyCalls = 0;
  const fetcher = {
    fetchRecent: vi.fn().mockResolvedValue(recentHtml),
    fetchAbstractsByIds: vi.fn().mockResolvedValue(new Map<string, string>()),
    fetchMetadataByIds: vi.fn(async (requested: string[]) =>
      new Map(requested.map((id) => [id, atomMeta(id)]))),
    fetchBySubmittedDate: vi.fn().mockResolvedValue([]),
    fetchPaperHtml: vi.fn().mockResolvedValue({ ok: false, status: 404 }),
    fetchPaperAbsPage: vi
      .fn()
      .mockResolvedValue(
        `<html><body><blockquote class="abstract">Abstract: stub abstract</blockquote></body></html>`,
      ),
  };
  const paperFetcher = {
    fetch: vi
      .fn()
      .mockResolvedValue({ abstractConclusion: "## Abstract\nstub", fullSections: null }),
  };
  const writer = {
    writeDaily: vi.fn(async (date: string, content: string, _options?: any) => {
      writes[`daily/${date}.md`] = content;
      return `daily/${date}.md`;
    }),
    writePaperDetail: vi.fn(async (paper: any, date: string, content: string) => {
      writes[`papers/${paper.id}.md`] = content;
      return `papers/${paper.id}.md`;
    }),
    writeEmptyDaily: vi.fn(),
    dailyPath: vi.fn((date: string) => `daily/${date}.md`),
    paperDetailPath: vi.fn((id: string) => `papers/${id}.md`),
    paperDetailLink: vi.fn((id: string) => `[[${id}]]`),
    dailyExists: vi.fn(async () => false),
    readDaily: vi.fn(async (date: string) => writes[`daily/${date}.md`] ?? ""),
    paperDetailExists: vi.fn(async () => false),
    readPaperDetail: vi.fn(async (id: string) => `# ${id}`),
  };
  const llm = {
    call: vi.fn(async (messages: ChatMessage[]) => {
      const system = messages[0]?.content ?? "";
      if (system.includes("选择最匹配的主题")) {
        return JSON.stringify({
          papers: keptIds.map((id) => ({ id, category: "photo-z" })),
        });
      }
      if (system.includes("You classify new arXiv papers against researcher-confirmed directions")) {
        const fence = /<paper_data>\n([\s\S]*)\n<\/paper_data>/.exec(messages[1]?.content ?? "");
        const payload = JSON.parse(fence?.[1] ?? "{}") as { papers: Array<{ paperKey: string }> };
        return JSON.stringify({
          papers: payload.papers.map((paper) => ({
            paperKey: paper.paperKey,
            directionIds: matchedIds.includes(paper.paperKey.slice("arxiv:".length))
              ? (options.discovery ?? discoveryOne).directions.map(({ id: directionId }) => directionId)
              : [],
          })),
        });
      }
      if (system.includes("compare one new arXiv paper against its representative prior papers")) {
        noveltyCalls += 1;
        return options.onNovelty?.(messages, noveltyCalls) ?? noveltyResponse();
      }
      if (system.includes("strict research-paper evaluator")) {
        return JSON.stringify({ papers: [] });
      }
      const structured = structuredDailyResponse(messages);
      if (structured) return structured;
      throw new Error(`unexpected LLM prompt: ${system.slice(0, 100)}`);
    }),
  };
  const logger = new Logger("error");
  const capturedPapers: any[] = [];
  const filterStores = options.filterStores ?? {};
  const summaryStore = options.summaryStore ?? {};
  const summarizeDaily = options.summarizeDaily ?? (async (papers: any[]) => {
    capturedPapers.push(...papers);
    return { markdown: "# injected daily", slots: [] };
  });
  const pipeline = new ArxivPipeline({
    markupParser,
    fetcher: fetcher as any,
    paperFetcher: paperFetcher as any,
    writer: writer as any,
    checkpointStores: {
      filter: {
        lookupReusable: vi.fn(async () => null),
        save: vi.fn(),
        lookupPersonalizedReusable: vi.fn(async () => undefined),
        savePersonalized: vi.fn(),
        lookupNoveltyReusable: vi.fn(async () => null),
        saveNovelty: vi.fn(),
        removeAll: vi.fn(async () => undefined),
        ...filterStores,
      },
      summary: {
        lookupReusable: vi.fn(async () => null),
        upsert: vi.fn(),
        ...summaryStore,
      },
    } as any,
    llm: llm as any,
    logger,
    arxiv: testArxiv,
    advanced: DEFAULT_SETTINGS.advanced,
    output: DEFAULT_SETTINGS.output,
    llmSettings: options.llmSettings ?? DEFAULT_SETTINGS.llm,
    detailSelection: testDetailSelection,
    personalizedDiscovery: options.discovery ?? discoveryOne,
    personalizedNoveltyInput: options.input
      ?? noveltyInput(keptIds.map((id) => `arxiv:${id}`).sort()),
    personalizedNoveltyMatches: options.matches ?? noveltyMatches(),
    summarizeDaily: summarizeDaily as any,
  });
  return { pipeline, writer, llm, logger, capturedPapers, noveltyCalls: () => noveltyCalls, writes };
}

function record(paperKey: string, overrides: Record<string, unknown> = {}): NoveltyCheckpointRecord {
  return {
    paperKey,
    status: "novelty",
    novelty: {
      differenceType: "new-task",
      comparisonBasis: ["arxiv:2501.00001"],
      evidenceDepth: "metadata-and-abstract",
      explanation: "Reused validated novelty.",
      ...overrides,
    },
  };
}

describe("pipeline personal novelty stage", () => {
  it("attaches novelty only to library-derived papers and leaves digest/email semantics unchanged", async () => {
    const { pipeline, writer, capturedPapers, noveltyCalls } = makePipeline();
    const result = await pipeline.runForDate(DATE);
    expect(result.kind).toBe("completed");
    if (result.kind === "completed") expect(result.digest).toBeDefined();
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(noveltyCalls()).toBe(1);
    const byId = Object.fromEntries(capturedPapers.map((paper) => [paper.id, paper]));
    expect(byId["2605.08080"].personalNovelty).toEqual({
      differenceType: "new-method",
      comparisonBasis: ["arxiv:2501.00001"],
      evidenceDepth: "metadata-and-abstract",
      explanation: "Introduces a method absent from the representative abstracts.",
      comparisonBasisTitles: { "arxiv:2501.00001": "Prior paper one" },
    });
    expect(byId["2605.08068"].personalNovelty).toBeUndefined();
    expect(JSON.stringify(result)).not.toContain("personalNovelty");
  });

  it("persists generated outcomes through the novelty checkpoint port", async () => {
    const saveNovelty = vi.fn(async () => undefined);
    const { pipeline, noveltyCalls } = makePipeline({
      filterStores: { saveNovelty },
    });
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(noveltyCalls()).toBe(1);
    expect(saveNovelty).toHaveBeenCalledTimes(1);
    const [date, prepared, outcomes] = saveNovelty.mock.calls[0]!;
    expect(date).toBe(DATE);
    expect(prepared.fingerprint).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(outcomes).toEqual([expect.objectContaining({
      paperKey: "arxiv:2605.08080",
      status: "novelty",
    })]);
  });

  it("reuses an exact checkpoint hit without any novelty calls", async () => {
    const saveNovelty = vi.fn();
    const lookupNoveltyReusable = vi.fn(async () => [
      record("arxiv:2605.08080"),
    ]);
    const { pipeline, capturedPapers, noveltyCalls } = makePipeline({
      filterStores: { lookupNoveltyReusable, saveNovelty },
    });
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(noveltyCalls()).toBe(0);
    expect(saveNovelty).not.toHaveBeenCalled();
    const byId = Object.fromEntries(capturedPapers.map((paper) => [paper.id, paper]));
    expect(byId["2605.08080"].personalNovelty).toMatchObject({
      differenceType: "new-task",
      explanation: "Reused validated novelty.",
    });
    expect(byId["2605.08068"].personalNovelty).toBeUndefined();
  });

  it("regenerates every planned paper when a checkpoint hit has partial coverage", async () => {
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const saveNovelty = vi.fn();
    const lookupNoveltyReusable = vi.fn(async () => [
      // Only one of the two planned papers is persisted: partial coverage is
      // never reused and the whole plan is regenerated.
      record("arxiv:2605.08068"),
    ]);
    const { pipeline, capturedPapers, noveltyCalls } = makePipeline({
      matchedIds: firstBucketIds().slice(0, 2),
      filterStores: { lookupNoveltyReusable, saveNovelty },
    });
    (pipeline as any).deps.logger = logger;
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(noveltyCalls()).toBe(2);
    expect(saveNovelty).toHaveBeenCalledTimes(1);
    const outcomes = saveNovelty.mock.calls[0]![2] as any[];
    expect(outcomes.map((outcome) => outcome.paperKey)).toEqual([
      "arxiv:2605.08068", "arxiv:2605.08080",
    ]);
    expect(outcomes.every((outcome) => outcome.status === "novelty")).toBe(true);
    const byId = Object.fromEntries(capturedPapers.map((paper) => [paper.id, paper]));
    expect(byId["2605.08068"].personalNovelty).toBeDefined();
    expect(byId["2605.08080"].personalNovelty).toBeDefined();
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("personal novelty checkpoint result invalid"),
      undefined,
    );
  });

  it("skips the checkpoint save entirely when any paper degrades on transport", async () => {
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const transport = Object.assign(new Error("provider 500"), { status: 500 });
    const saveNovelty = vi.fn();
    const { pipeline, writer, capturedPapers, noveltyCalls } = makePipeline({
      matchedIds: firstBucketIds().slice(0, 2),
      filterStores: { saveNovelty },
      onNovelty: (_messages, callNumber) => {
        if (callNumber === 2) throw transport;
        return noveltyResponse();
      },
    });
    (pipeline as any).deps.logger = logger;
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(noveltyCalls()).toBe(2);
    // Degraded papers must never be durably marked no-novelty: no save at all.
    expect(saveNovelty).not.toHaveBeenCalled();
    const byId = Object.fromEntries(capturedPapers.map((paper) => [paper.id, paper]));
    expect(byId["2605.08068"].personalNovelty).toBeDefined();
    expect(byId["2605.08080"].personalNovelty).toBeUndefined();
    expect(warn).toHaveBeenCalledWith(
      "pipeline: personal novelty degraded: personal novelty call degraded for arxiv:2605.08080 (transport)",
      transport,
    );
  });

  it("degrades a checkpoint lookup failure to no-novelty without failing the run", async () => {
    const logger = new Logger("error");
    const warn = vi.spyOn(logger, "warn");
    const lookupNoveltyReusable = vi.fn(async () => {
      throw Object.assign(new Error("checkpoint EIO"), { code: "EIO" });
    });
    const { pipeline, writer, capturedPapers, noveltyCalls } = makePipeline({
      filterStores: { lookupNoveltyReusable },
    });
    (pipeline as any).deps.logger = logger;
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(noveltyCalls()).toBe(0);
    expect(capturedPapers.every((paper) => paper.personalNovelty === undefined)).toBe(true);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("personal novelty degraded"),
      expect.any(Error),
    );
  });

  it("degrades only the affected paper on a transport error and keeps other novelty intact", async () => {
    const transport = Object.assign(new Error("provider 500"), { status: 500 });
    const { pipeline, writer, capturedPapers, noveltyCalls } = makePipeline({
      matchedIds: firstBucketIds().slice(0, 2),
      onNovelty: (_messages, callNumber) => {
        if (callNumber === 2) throw transport;
        return noveltyResponse();
      },
    });
    const result = await pipeline.runForDate(DATE);
    expect(result.kind).toBe("completed");
    if (result.kind === "completed") expect(result.papersWritten).toBe(2);
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(noveltyCalls()).toBe(2);
    const byId = Object.fromEntries(capturedPapers.map((paper) => [paper.id, paper]));
    expect(byId["2605.08068"].personalNovelty).toBeDefined();
    expect(byId["2605.08080"].personalNovelty).toBeUndefined();
  });

  it("degrades a single paper on an output-limit error and still completes", async () => {
    const { pipeline, writer, capturedPapers, noveltyCalls } = makePipeline({
      matchedIds: firstBucketIds().slice(0, 2),
      onNovelty: (_messages, callNumber) => callNumber === 2
        ? "x".repeat(PERSONAL_NOVELTY_MAX_OUTPUT_CODE_UNITS + 1)
        : noveltyResponse(),
    });
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(noveltyCalls()).toBe(2);
    const byId = Object.fromEntries(capturedPapers.map((paper) => [paper.id, paper]));
    expect(byId["2605.08080"].personalNovelty).toBeUndefined();
    expect(byId["2605.08068"].personalNovelty).toBeDefined();
  });

  it("degrades to no-novelty after validation exhaustion without retrying the whole run", async () => {
    const { pipeline, writer, capturedPapers, noveltyCalls } = makePipeline({
      onNovelty: () => "not json",
    });
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(noveltyCalls()).toBe(3);
    expect(capturedPapers.every((paper) => paper.personalNovelty === undefined)).toBe(true);
  });

  it("returns a cancelled result when cancellation arrives during the novelty stage", async () => {
    const controller = new AbortController();
    const { pipeline, writer, noveltyCalls } = makePipeline({
      matchedIds: firstBucketIds().slice(0, 2),
      onNovelty: (_messages, callNumber) => {
        if (callNumber === 2) controller.abort("stop during novelty");
        return noveltyResponse();
      },
    });
    const result = await pipeline.runForDate(DATE, controller.signal);
    expect(result).toEqual({ kind: "cancelled", reason: "stop during novelty" });
    expect(noveltyCalls()).toBe(2);
    expect(writer.writeDaily).not.toHaveBeenCalled();
  });

  it("makes no novelty calls and attaches nothing on a complete basis that exceeds per-call bounds", async () => {
    const directions = Array.from({ length: 9 }, (_, index) => {
      const offset = index * 5;
      return {
        directionId: `direction.${String(index + 1).padStart(3, "0")}`,
        representativePaperKeys: Array.from({ length: 5 }, (_, offsetIndex) =>
          `arxiv:2501.${String(offset + offsetIndex + 1).padStart(5, "0")}`),
      };
    });
    const discovery: PersonalizedDiscoveryInput = {
      directions: directions.map((entry) => ({
        id: entry.directionId,
        name: `Direction ${entry.directionId}`,
        description: "Plan-too-large direction",
        discoveryCues: ["strict"],
        representatives: entry.representativePaperKeys.map((paperKey) => ({
          paperKey,
          title: `Rep ${paperKey}`,
          evidenceDepth: "metadata-and-abstract" as const,
        })),
      })),
    };
    const representatives = directions.flatMap((entry) =>
      entry.representativePaperKeys.map((paperKey, index) => ({
        paperKey,
        title: `Prior ${paperKey}`,
        authors: [`Author ${index}`],
        abstract: `Prior abstract ${paperKey}`,
        published: "2026-08-01T00:00:00.000Z",
        categories: ["cs.AI"],
      })),
    );
    const ids = firstBucketIds();
    const input: PersonalizedNoveltyInput = {
      papers: [{ paperKey: `arxiv:${ids[0]!}`, title: "New paper", abstract: "Abstract" }],
      representatives,
    };
    const matches: PersonalNoveltyMatchInput = {
      paperMatches: [],
      directionRepresentatives: directions,
    };
    const saveNovelty = vi.fn();
    const { pipeline, writer, capturedPapers, noveltyCalls } = makePipeline({
      keptIds: [ids[0]!],
      matchedIds: [ids[0]!],
      discovery,
      input,
      matches,
      filterStores: { saveNovelty },
    });
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(noveltyCalls()).toBe(0);
    expect(saveNovelty).toHaveBeenCalledTimes(1);
    expect(capturedPapers.every((paper) => paper.personalNovelty === undefined)).toBe(true);
  });

  it("never includes personal novelty in the summary checkpoint compatibility input", async () => {
    const captured: any[] = [];
    const summaryStore = {
      lookupReusable: vi.fn(async (_date: string, input: unknown) => {
        captured.push(input);
        return null;
      }),
      upsert: vi.fn(async () => undefined),
    };
    const { pipeline, writes } = makePipeline({
      summaryStore,
      summarizeDaily: realSummarizeDaily as any,
    });
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(captured.length).toBe(2);
    const input = captured.find((entry) => entry.paper.id === "2605.08080");
    expect(input.paper.personalNovelty).toBeDefined();
    const sansNovelty = { ...input, paper: { ...input.paper, personalNovelty: undefined } };
    expect(createDailySummaryCompatibilityFingerprint(input))
      .toBe(createDailySummaryCompatibilityFingerprint(sansNovelty));
    const raw = JSON.stringify(buildDailySummaryCheckpointFingerprintInput(input));
    expect(raw).not.toContain("novelty");
    expect(raw).not.toContain("Introduces a method");
    const markdown = Object.values(writes)[0] ?? "";
    // The committed report legitimately carries the rendered novelty (marker +
    // visible line), but never the internal camelCase field name.
    expect(markdown).not.toContain("personalNovelty");
    expect(markdown).toContain("<!-- arxiv-daily-personal-novelty:v1:");
    expect(markdown).toContain("Prior paper one (arxiv:2501\\.00001)");
    expect(markdown).toContain("Introduces a method");
  });

  it("carries validated novelty through deterministic daily rendering into the committed report", async () => {
    const { pipeline, writes } = makePipeline({
      summarizeDaily: realSummarizeDaily as any,
    });
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    const markdown = Object.values(writes)[0]!;
    expect(markdown).toContain("Prior paper one (arxiv:2501\\.00001)");
    expect(markdown).toContain("证据深度：元数据与摘要");
    expect(markdown.match(/^<!-- arxiv-daily-personal-novelty:v1:/gm)).toHaveLength(1);
    expect(parseDailyReportPersonalNovelty(markdown, DATE)).toEqual({
      kind: "valid",
      occurrences: [{
        arxivId: "2605.08080",
        novelty: {
          differenceType: "new-method",
          comparisonBasis: ["arxiv:2501.00001"],
          evidenceDepth: "metadata-and-abstract",
          explanation: "Introduces a method absent from the representative abstracts.",
        },
      }],
    });
    // Marker families stay independent in the committed report.
    expect(parseDailyReportDiscoveryProvenance(markdown, DATE).kind).toBe("valid");
  });

  it("does no novelty stage work when no papers survive filtering", async () => {
    const lookupNoveltyReusable = vi.fn(async () => null);
    const saveNovelty = vi.fn();
    const ids = firstBucketIds();
    const { pipeline, writer, noveltyCalls } = makePipeline({
      keptIds: [],
      matchedIds: [],
      filterStores: { lookupNoveltyReusable, saveNovelty },
    });
    const result = await pipeline.runForDate(DATE);
    expect(result.kind).toBe("completed");
    if (result.kind === "completed") expect(result.papersWritten).toBe(0);
    expect(writer.writeDaily).not.toHaveBeenCalled();
    expect(noveltyCalls()).toBe(0);
    expect(lookupNoveltyReusable).not.toHaveBeenCalled();
    expect(saveNovelty).not.toHaveBeenCalled();
    expect(ids.length).toBeGreaterThan(0);
  });

  it("stays byte-compatible when novelty deps are absent (manual-only run)", async () => {
    const ids = firstBucketIds();
    const lookupNoveltyReusable = vi.fn(async () => null);
    const { pipeline, writer, capturedPapers, noveltyCalls } = makePipeline({
      keptIds: [ids[0]!],
    });
    (pipeline as any).deps.personalizedNoveltyInput = undefined;
    (pipeline as any).deps.personalizedNoveltyMatches = undefined;
    (pipeline as any).deps.checkpointStores.filter.lookupNoveltyReusable = lookupNoveltyReusable;
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(writer.writeDaily).toHaveBeenCalledTimes(1);
    expect(noveltyCalls()).toBe(0);
    expect(lookupNoveltyReusable).not.toHaveBeenCalled();
    expect(capturedPapers.every((paper) => paper.personalNovelty === undefined)).toBe(true);
  });

  it("does not attach novelty when the discovery snapshot carries no directions", async () => {
    const lookupNoveltyReusable = vi.fn(async () => null);
    const saveNovelty = vi.fn();
    const { pipeline, capturedPapers, noveltyCalls } = makePipeline({
      discovery: { directions: [] },
      filterStores: { lookupNoveltyReusable, saveNovelty },
    });
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(noveltyCalls()).toBe(0);
    expect(lookupNoveltyReusable).not.toHaveBeenCalled();
    expect(saveNovelty).not.toHaveBeenCalled();
    expect(capturedPapers.every((paper) => paper.personalNovelty === undefined)).toBe(true);
  });

  it("degrades only the affected paper when basis enrichment fails and the run still completes", async () => {
    const ids = firstBucketIds();
    const warnSpy = vi.spyOn(Logger.prototype, "warn");
    const attachSpy = vi.spyOn(personalizedNovelty, "attachPersonalNoveltyBasis");
    attachSpy.mockImplementationOnce(() => {
      throw new Error("basis enrichment failed");
    });
    const { pipeline, capturedPapers, noveltyCalls } = makePipeline({
      matchedIds: [ids[0]!, ids[1]!],
    });
    expect(await pipeline.runForDate(DATE)).toMatchObject({ kind: "completed" });
    expect(noveltyCalls()).toBe(2);
    expect(attachSpy).toHaveBeenCalledTimes(2);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("personal novelty basis enrichment failed"),
      expect.any(Error),
    );
    // Exactly one paper lost its novelty to the injected attach failure; the
    // other keeps its validated novelty with trusted basis titles.
    const kept = capturedPapers.filter((paper) => paper.personalNovelty !== undefined);
    const degraded = capturedPapers.filter((paper) => paper.personalNovelty === undefined);
    expect(kept).toHaveLength(1);
    expect(degraded).toHaveLength(1);
    expect(kept[0]!.personalNovelty).toEqual({
      differenceType: "new-method",
      comparisonBasis: ["arxiv:2501.00001"],
      evidenceDepth: "metadata-and-abstract",
      explanation: "Introduces a method absent from the representative abstracts.",
      comparisonBasisTitles: { "arxiv:2501.00001": "Prior paper one" },
    });
  });

  it("rethrows cancellation raised during basis enrichment", async () => {
    const ids = firstBucketIds();
    const attachSpy = vi.spyOn(personalizedNovelty, "attachPersonalNoveltyBasis");
    attachSpy.mockImplementation(() => {
      throw new RunCancelledError("cancelled during attach");
    });
    const { pipeline, writer, noveltyCalls } = makePipeline({
      matchedIds: [ids[0]!],
    });
    expect(await pipeline.runForDate(DATE)).toEqual({
      kind: "cancelled",
      reason: "cancelled during attach",
    });
    expect(noveltyCalls()).toBe(1);
    expect(writer.writeDaily).not.toHaveBeenCalled();
  });
});
