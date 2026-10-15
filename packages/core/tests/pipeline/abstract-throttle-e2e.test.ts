import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HttpClient, HttpRequest, HttpResponse, StorageAdapter } from "../../src/core/adapters";
import {
  ArxivFetcher,
  ArxivRequestCoordinator,
  resetArxivRequestCoordinatorForTests,
} from "../../src/pipeline/arxiv-fetcher";
import { HtmlCache } from "../../src/pipeline/html-cache";
import { PaperContentFetcher } from "../../src/pipeline/paper-content";
import { ArxivPipeline, type PipelineDeps } from "../../src/pipeline/pipeline";
import { summarizeDaily } from "../../src/pipeline/summarizer";
import { FilterDecisionRecordStore } from "../../src/services/filter-decision-record-store";
import { Logger } from "../../src/services/logger";
import { RunHistoryStore } from "../../src/services/run-history";
import { RunLock } from "../../src/services/run-lock";
import { SchedulerService } from "../../src/services/scheduler";
import { createStorageStateStore } from "../../src/services/state-store";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import { normalizeTopic } from "../../src/settings/topics";
import { markupParser } from "../markup-parser";

/**
 * Phase verification for the abstract fallback: the REAL ArxivFetcher,
 * ArxivSourceAdapter (built by the pipeline), PaperContentFetcher and HtmlCache
 * run over a fake HttpClient. Only HTTP, the model, the daily writer and the
 * storage are fake. Pacing runs on an injected fake-clock coordinator; the
 * retry util's sleeps are shortened (and recorded) at the global timer.
 */

const date = "2026-09-07"; // a Monday
const API_HOST = "export.arxiv.org";
/** Unique so the request watchdog's timer is never mistaken for a retry sleep. */
const WATCHDOG_MS = 600_000;
const id = (index: number): string => `2609.${String(index).padStart(5, "0")}`;
const abstractFor = (paperId: string): string => `ABS-PAGE-ABSTRACT for ${paperId}`;
const titleOnlyLine = (count: number, total: number): string =>
  `筛选时 ${total} 篇候选中有 ${count} 篇没有摘要，只依据标题判断。`;

function listing(ids: string[]): string {
  return `<dl id="articles"><h3>Mon, 7 Sep 2026 (showing ${ids.length} of ${ids.length} entries )</h3>`
    + ids.map((paperId, index) => `<dt><a name="item${index}">[${index}]</a> <a href="/abs/${paperId}" title="Abstract">arXiv:${paperId}</a></dt>`
      + `<dd><div class="meta"><div class="list-title mathjax"><span class="descriptor">Title:</span> Research paper ${paperId}</div>`
      + `<div class="list-authors"><a href="#">A. Researcher</a></div></div></dd>`).join("")
    + "</dl>";
}

function absPage(paperId: string): string {
  return `<html><body><h1 class="title">Research paper ${paperId}</h1>`
    + `<blockquote class="abstract mathjax"><span class="descriptor">Abstract:</span> ${abstractFor(paperId)}</blockquote>`
    + `</body></html>`;
}

function memoryStorage(): { storage: StorageAdapter; files: Record<string, string> } {
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
  return { storage, files };
}

interface RoutedRequest { url: string; kind: "list" | "api" | "abs" | "other"; at: number }

function kindOf(url: string): RoutedRequest["kind"] {
  const parsed = new URL(url);
  if (parsed.host === API_HOST && parsed.pathname === "/api/query") return "api";
  if (parsed.host === "arxiv.org" && /^\/list\/[^/]+\/recent$/.test(parsed.pathname)) return "list";
  if (parsed.host === "arxiv.org" && parsed.pathname.startsWith("/abs/")) return "abs";
  return "other";
}

interface ScenarioOptions {
  ids: string[];
  /** Response of every export.arxiv.org/api/query call. */
  api: HttpResponse;
  /** Response for an abs page (called for each network attempt). */
  abs: (paperId: string, attempt: number) => HttpResponse;
}

function buildScenario(options: ScenarioOptions) {
  // Fake clock for the coordinator: sleeping advances it instantly.
  const clock = { monotonic: 1_000 };
  const sleeps: number[] = [];
  const coordinator = new ArxivRequestCoordinator({
    monotonicNow: () => clock.monotonic,
    wallNow: () => Date.parse("2026-09-07T10:00:00.000Z") + clock.monotonic,
    sleep: async (ms) => { sleeps.push(ms); clock.monotonic += ms; },
  });

  const requests: RoutedRequest[] = [];
  const absAttempts = new Map<string, number>();
  const http: HttpClient = {
    request: vi.fn(async (req: HttpRequest): Promise<HttpResponse> => {
      const kind = kindOf(req.url);
      requests.push({ url: req.url, kind, at: clock.monotonic });
      if (kind === "list") return { status: 200, headers: {}, bodyText: listing(options.ids) };
      if (kind === "api") return options.api;
      if (kind === "abs") {
        const paperId = new URL(req.url).pathname.replace("/abs/", "");
        const attempt = (absAttempts.get(paperId) ?? 0) + 1;
        absAttempts.set(paperId, attempt);
        return options.abs(paperId, attempt);
      }
      return { status: 404, headers: {}, bodyText: "" };
    }),
  };

  const logger = new Logger("error");
  const warn = vi.spyOn(logger, "warn");
  const fetcher = new ArxivFetcher({
    categories: ["astro-ph"], markupParser, http, logger,
    requestDelayMs: 0, requestCoordinator: coordinator,
    textTimeoutMs: WATCHDOG_MS, binaryTimeoutMs: WATCHDOG_MS,
  });
  const { storage: cacheStorage, files: cacheFiles } = memoryStorage();
  const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage: cacheStorage });
  const paperFetcher = new PaperContentFetcher(fetcher, cache, logger, markupParser);

  const { storage, files } = memoryStorage();
  const decisionStore = new FilterDecisionRecordStore(storage, DEFAULT_SETTINGS.output, {
    now: () => new Date("2026-09-07T08:00:00.000Z"),
  });
  const writer = {
    dailyExists: vi.fn(async () => false),
    dailyPath: (day: string) => `arxiv-daily/daily/${day}.md`,
    writeDaily: vi.fn(async (day: string, content: string) => { files[`arxiv-daily/daily/${day}.md`] = content; }),
    paperDetailPath: (paperId: string) => `arxiv-daily/papers/${paperId}.md`,
    paperDetailExists: vi.fn(async () => false),
    paperDetailLink: (paperId: string) => `[[${paperId}]]`,
  };
  /** Requests already made when the filter model is called = the discovery + abstract-fallback phase. */
  let beforeFilter: RoutedRequest[] | undefined;
  const llm = {
    call: vi.fn(async (messages: Array<{ content: string }>) => {
      if (messages[0]!.content.includes("relevanceScore")) {
        beforeFilter ??= [...requests];
        return JSON.stringify({
          papers: options.ids.map((paperId) => ({
            id: paperId, category: "topic-a", relevanceScore: 90, directions: ["topic-a#1"],
          })),
        });
      }
      const paperId = /\d{4}\.\d{5}/.exec(messages.map((message) => message.content).join("\n"))![0];
      return JSON.stringify({
        id: paperId, coreProblem: "problem", keyMethod: "method",
        mainResult: "result", whyRelevant: "value", limitations: "limits",
      });
    }),
  };
  // No `sourceAdapter`: the pipeline builds the real ArxivSourceAdapter itself.
  const pipeline = new ArxivPipeline({
    markupParser, fetcher, paperFetcher,
    writer: writer as unknown as PipelineDeps["writer"],
    llm: llm as unknown as PipelineDeps["llm"], logger,
    arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: [normalizeTopic({
      name: "topic-a", tag: "topic-a", detail: false,
      directions: [{ id: "topic-a-direction", text: "Direction of topic-a", origin: "manual" }],
    })] },
    advanced: DEFAULT_SETTINGS.advanced,
    output: DEFAULT_SETTINGS.output,
    llmSettings: { ...DEFAULT_SETTINGS.llm, provider: "custom", baseUrl: "https://llm.example.test/v1", apiKey: "k", model: "m" },
    detailSelection: { normalThreshold: 70, exceptionalThreshold: 90, softLimit: 2 },
    summarizeDaily,
    filterDecisionStore: decisionStore,
  });

  const requestsOf = (kind: RoutedRequest["kind"]) => requests.filter((request) => request.kind === kind);
  /** Requests of one kind made before the filter ran (discovery + fallback only, not later content fetches). */
  const beforeFilterOf = (kind: RoutedRequest["kind"]) =>
    (beforeFilter ?? []).filter((request) => request.kind === kind);
  const filterPrompt = (): string => llm.call.mock.calls
    .map(([messages]) => messages.map((message) => message.content).join("\n"))
    .find((text) => text.includes("relevanceScore")) ?? "";
  const decisionRecord = () => JSON.parse(files[`arxiv-daily/.index/filter-decisions/${date}.json`]!) as {
    candidates: Array<{ id: string; hadAbstract: boolean }>;
  };
  const daily = (): string => files[`arxiv-daily/daily/${date}.md`] ?? "";
  return {
    clock, sleeps, requests, requestsOf, beforeFilterOf, http, pipeline, paperFetcher, cacheFiles, filterPrompt,
    decisionRecord, daily, warn, llm,
  };
}

describe("abstract fallback end to end (fake HTTP only)", () => {
  /** Waits requested through the global timer by the retry util, shortened to 0 ms. */
  let retryWaits: number[];

  beforeEach(async () => {
    await resetArxivRequestCoordinatorForTests();
    retryWaits = [];
    const realSetTimeout = globalThis.setTimeout;
    vi.spyOn(globalThis, "setTimeout").mockImplementation(((handler: () => void, ms?: number, ...args: unknown[]) => {
      if (typeof ms === "number" && ms >= 1_000 && ms !== WATCHDOG_MS) {
        retryWaits.push(ms);
        return realSetTimeout(handler, 0, ...args);
      }
      return realSetTimeout(handler, ms, ...args);
    }) as typeof setTimeout);
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await resetArxivRequestCoordinatorForTests();
  });

  const ids = [id(1), id(2), id(3), id(4)];
  const okAbs = (paperId: string): HttpResponse => ({ status: 200, headers: {}, bodyText: absPage(paperId) });

  async function expectFullyRecovered(s: ReturnType<typeof buildScenario>) {
    const result = await s.pipeline.runForDate(date);

    expect(result).toMatchObject({
      kind: "completed", outcome: "papers_written", papersWritten: ids.length,
      titleOnlyCandidates: { count: 0, total: ids.length },
    });
    const prompt = s.filterPrompt();
    for (const paperId of ids) expect(prompt).toContain(abstractFor(paperId));
    expect(s.decisionRecord().candidates.map((c) => [c.id, c.hadAbstract]))
      .toEqual(ids.map((paperId) => [paperId, true]));
    expect(s.daily()).not.toContain("只依据标题判断");
    expect(s.daily()).toContain(`共 ${ids.length} 篇相关论文`);

    // The abs pages were written to the abs cache: reading them again is free.
    const absRequestsBefore = s.requestsOf("abs").length;
    expect(Object.keys(s.cacheFiles).filter((path) => path.includes("abs"))).toHaveLength(ids.length);
    for (const paperId of ids) {
      await expect(s.paperFetcher.fetchListingAbstract(paperId)).resolves.toEqual({
        abstract: abstractFor(paperId), fromNetwork: false,
      });
    }
    expect(s.requestsOf("abs")).toHaveLength(absRequestsBefore);
    return result;
  }

  it("(a) API 429 without Retry-After: one API request, abs pages feed the filter, no title-only line", async () => {
    const s = buildScenario({
      ids, api: { status: 429, headers: {}, bodyText: "slow down" }, abs: okAbs,
    });

    await expectFullyRecovered(s);

    expect(s.requestsOf("list")).toHaveLength(1);
    expect(s.requestsOf("api")).toHaveLength(1); // no retries, no second probe
    expect(s.requestsOf("abs").map((request) => new URL(request.url).pathname))
      .toEqual(ids.map((paperId) => `/abs/${paperId}`));
    // Fake-clock pacing only (3 s between requests), never a long cooldown wait.
    expect(Math.max(0, ...s.sleeps)).toBeLessThanOrEqual(3_000);
    // And the retry util never slept for the API (header-less 429 is not retried).
    expect(retryWaits).toEqual([]);
  });

  it("(b) API 429 with Retry-After: not retried, abs pages start at once", async () => {
    const s = buildScenario({
      ids, api: { status: 429, headers: { "Retry-After": "120" }, bodyText: "slow down" }, abs: okAbs,
    });

    await expectFullyRecovered(s);

    const api = s.requestsOf("api");
    const abs = s.beforeFilterOf("abs");
    expect(abs).toHaveLength(ids.length);
    // Any query API 429 is final: one API request, no inline 120 s waits.
    expect(api).toHaveLength(1);
    expect(Math.max(0, ...s.sleeps)).toBeLessThanOrEqual(3_000);
    expect(retryWaits).toEqual([]);
    // The cooldown belongs to export.arxiv.org: from the API request to the
    // first abs request only the shared 3 s pacing elapses, never the 120 s.
    expect(abs[0]!.at - api[0]!.at).toBeLessThanOrEqual(3_000);
    // Consecutive abs requests are paced, not cooled down.
    for (let index = 1; index < abs.length; index += 1) {
      expect(abs[index]!.at - abs[index - 1]!.at).toBeLessThanOrEqual(3_000);
    }
  });

  describe("(c) API 429 and abs pages failing", () => {
    const paperIds = [id(1), id(2), id(3), id(4), id(5), id(6)];
    const cases: Array<[string, HttpResponse, number | null]> = [
      ["HTTP 500", { status: 500, headers: {}, bodyText: "boom" }, null],
      ["header-less 429", { status: 429, headers: {}, bodyText: "slow down" }, 30_000],
    ];

    it.each(cases)("stops after 3 consecutive %s failures, completes, and surfaces the count", async (_label, failure, minRetryWait) => {
      const s = buildScenario({
        ids: paperIds, api: { status: 429, headers: {}, bodyText: "slow down" },
        abs: () => failure,
      });

      const result = await s.pipeline.runForDate(date);

      expect(result).toMatchObject({
        kind: "completed", outcome: "papers_written", papersWritten: paperIds.length,
        titleOnlyCandidates: { count: 6, total: 6 },
      });
      // 3 papers tried (3 HTTP attempts each, the fetcher's own retry budget), then the loop stops.
      const abs = s.beforeFilterOf("abs");
      expect(abs).toHaveLength(9);
      expect(new Set(abs.map((request) => request.url))).toEqual(new Set(paperIds.slice(0, 3).map((paperId) => `https://arxiv.org/abs/${paperId}`)));
      if (minRetryWait != null) {
        expect(retryWaits.length).toBeGreaterThan(0);
        expect(Math.min(...retryWaits)).toBeGreaterThanOrEqual(minRetryWait);
      }
      expect(s.decisionRecord().candidates.map((c) => c.hadAbstract)).toEqual(paperIds.map(() => false));
      expect(s.daily()).toContain(titleOnlyLine(6, 6));
      expect(s.daily().match(/只依据标题判断/g)).toHaveLength(1);
      expect(s.warn.mock.calls.map(([line]) => String(line)).join("\n"))
        .toContain("6/6 candidates filtered on title only");
    });

    it("counts only the papers still without an abstract after a partial recovery", async () => {
      // First two abs pages load; the next three fail, which trips the stop before the sixth is tried.
      const s = buildScenario({
        ids: paperIds, api: { status: 429, headers: {}, bodyText: "slow down" },
        abs: (paperId) => (paperId === id(1) || paperId === id(2)
          ? okAbs(paperId)
          : { status: 404, headers: {}, bodyText: "gone" }),
      });

      const result = await s.pipeline.runForDate(date);

      expect(result).toMatchObject({ kind: "completed", titleOnlyCandidates: { count: 4, total: 6 } });
      expect(s.beforeFilterOf("abs")).toHaveLength(5); // 404 is not retried; the sixth paper is never tried
      expect(s.decisionRecord().candidates.map((c) => c.hadAbstract))
        .toEqual([true, true, false, false, false, false]);
      expect(s.daily()).toContain(titleOnlyLine(4, 6));
    });

    it("carries the count into run history through the scheduler, leaving the run completed", async () => {
      const s = buildScenario({
        ids: paperIds, api: { status: 429, headers: {}, bodyText: "slow down" },
        abs: () => ({ status: 404, headers: {}, bodyText: "gone" }),
      });
      const { storage: stateStorage } = memoryStorage();
      const history = RunHistoryStore.fromStorage(stateStorage, DEFAULT_SETTINGS.output);
      const noop = vi.fn();
      const scheduler = new SchedulerService({
        store: createStorageStateStore(stateStorage, DEFAULT_SETTINGS.output),
        lock: new RunLock(),
        getSettings: () => DEFAULT_SETTINGS,
        runForDate: (day, signal) => s.pipeline.runForDate(day, signal),
        runHistory: history,
        onDailyCompleted: vi.fn(async () => {}),
        logger: { info: noop, warn: noop, error: noop, debug: noop, notice: noop } as never,
        now: () => new Date("2026-09-07T12:00:00Z"),
      });

      await scheduler.runForDateNow(date);

      const completed = (await history.readLatest()).find((record) => record.event === "completed");
      expect(completed).toMatchObject({
        status: "completed", resultKind: "completed", papersWritten: 6,
        titleOnlyCandidates: { count: 6, total: 6 },
      });
    });
  });
});
