import { beforeEach, describe, expect, it, vi } from "vitest";
import { Notice } from "obsidian";
import {
  DEFAULT_SETTINGS,
  Logger,
  RunLock,
  SchedulerService,
  StateStore,
  type PipelineResult,
  type RunState,
} from "@arxiv-daily/core";
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  registerDashboardView,
} from "../src/dashboard/view";

const NOW = new Date("2026-10-21T12:00:00Z");
const DATE = "2026-10-20";

async function setup(initial: RunState, reportDates: string[] = []) {
  const durable = { runState: structuredClone(initial) };
  const stateStore = new StateStore(
    async () => ({ runState: structuredClone(durable.runState) }),
    async ({ runState }) => { durable.runState = structuredClone(runState); },
  );
  await stateStore.load();
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.arxiv.timezone = "UTC";
  settings.llm.apiKey = "configured";
  settings.arxiv.topics.push({
    id: "topic-1",
    name: "Language models",
    tag: "language-models",
    description: "Language model research",
    directions: [{ id: "d1", text: "Language model research", origin: "manual" }],
    detail: false,
  });
  const runForDate = vi.fn(async (): Promise<PipelineResult> => ({ kind: "completed", papersWritten: 4 }));
  const scheduler = new SchedulerService({
    getSettings: () => settings,
    store: stateStore,
    lock: new RunLock(),
    logger: new Logger("error"),
    runForDate,
    now: () => NOW,
    dailyExists: async (date) => reportDates.includes(date),
  });
  let create: ((leaf: unknown) => unknown) | undefined;
  const plugin = {
    app: { vault: {}, workspace: {} },
    settings,
    registerView: (type: string, creator: (leaf: unknown) => unknown) => {
      if (type === ARXIV_DAILY_DASHBOARD_VIEW) create = creator;
    },
    logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    recentDates: { refresh: vi.fn(async () => undefined) },
    operations: { snapshot: () => [] },
    stateStore,
    scheduler,
  };
  registerDashboardView(plugin as never);
  const view = create!({}) as unknown as {
    calendarDailyReports: Map<string, { date: string; path: string; papers: number; starred: number }>;
    reloadIndex: () => Promise<void>;
    runDateFromCalendar(date: string): Promise<void>;
  };
  view.reloadIndex = vi.fn(async () => {});
  for (const date of reportDates) {
    view.calendarDailyReports.set(date, { date, path: `arxiv-daily/daily/${date}.md`, papers: 4, starred: 0 });
  }
  const normalRun = vi.spyOn(scheduler, "runForDateNow");
  const notices = () => Notice.calls.map((call) => call.message);
  return { durable, stateStore, runForDate, normalRun, view, notices };
}

const completed = (papersWritten = 10): RunState => ({
  [DATE]: { status: "completed", lastAttempt: 1, attempts: 1, papersWritten, outcome: "papers_written" },
});

beforeEach(() => {
  Notice.calls = [];
});

describe("running a date from the calendar", () => {
  it("regenerates a completed date whose daily report was deleted", async () => {
    const { runForDate, normalRun, view, notices, stateStore } = await setup(completed());

    await view.runDateFromCalendar(DATE);

    expect(runForDate).toHaveBeenCalledTimes(1);
    expect((runForDate.mock.calls as unknown[][])[0]![0]).toBe(DATE);
    expect(normalRun).toHaveBeenCalledWith(DATE, { trigger: "calendar" });
    expect(stateStore.get(DATE)).toMatchObject({ status: "completed", papersWritten: 4 });
    expect(notices().join("\n")).not.toContain("already done");
    expect(view.reloadIndex).toHaveBeenCalled();
  });

  it("keeps the normal run for a date that was never completed", async () => {
    for (const state of [
      {},
      { [DATE]: { status: "pending", lastAttempt: 1, attempts: 1, outcome: "awaiting_announcement" } },
      { [DATE]: { status: "failed_transient", lastAttempt: 1, attempts: 1, failureAttempts: 1, error: "offline" } },
    ] as RunState[]) {
      const { runForDate, normalRun, view } = await setup(state);

      await view.runDateFromCalendar(DATE);

      expect(runForDate).toHaveBeenCalledTimes(1);
      expect(normalRun).toHaveBeenCalledWith(DATE, { trigger: "calendar" });
    }
  });

  it("does not clear a completed date that still has its daily report", async () => {
    const { runForDate, normalRun, view, notices, durable } = await setup(completed(), [DATE]);

    await view.runDateFromCalendar(DATE);

    expect(normalRun).toHaveBeenCalledWith(DATE, { trigger: "calendar" });
    expect(runForDate).not.toHaveBeenCalled();
    expect(notices().join("\n")).toContain("already done");
    expect(durable.runState[DATE]).toMatchObject({ status: "completed", papersWritten: 10 });
  });

  it("leaves a date that is running alone", async () => {
    const { runForDate, view, notices } = await setup({
      [DATE]: { status: "running", lastAttempt: NOW.getTime(), attempts: 1 },
    });

    await view.runDateFromCalendar(DATE);

    expect(runForDate).not.toHaveBeenCalled();
    expect(notices().join("\n")).toContain("already running");
  });
});
