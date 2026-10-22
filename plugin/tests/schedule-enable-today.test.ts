import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS, StateStore, type RunState } from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";
import { chooseModal } from "../src/services/modal";
import { getSetupStatus } from "../src/onboarding";
import { SettingsChangeService } from "../src/settings/change-service";

vi.mock("../src/services/modal", () => ({ chooseModal: vi.fn() }));

const TODAY = "2026-10-21";

beforeEach(() => {
  vi.mocked(chooseModal).mockReset();
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date(`${TODAY}T12:00:00Z`));
});
afterEach(() => { vi.useRealTimers(); });

async function enablePlugin(initial: RunState) {
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
  const scheduler = { start: vi.fn(), stop: vi.fn(), tickToday: vi.fn(async () => undefined) };
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  Object.assign(plugin, {
    settings,
    stateStore,
    scheduler,
    app: {},
    progress: { setDisabled: vi.fn() },
    logger: { notice: vi.fn(), error: vi.fn() },
  });
  plugin.settingsChanges = new SettingsChangeService({
    settings,
    persistSettings: vi.fn().mockResolvedValue(undefined),
    setScheduleEnabled: (enabled) => (plugin as any).applyScheduleEnabledRuntime(enabled),
  });
  return { plugin, stateStore, settings, scheduler, durable };
}

const completedToday: RunState = {
  [TODAY]: { status: "completed", lastAttempt: 1, attempts: 1, papersWritten: 7, outcome: "papers_written" },
};

describe("enabling the scheduler after today's report was generated", () => {
  it("does not ask what to do with today and leaves the completed run alone", async () => {
    const { plugin, stateStore, settings, scheduler, durable } = await enablePlugin(completedToday);

    await expect(plugin.setScheduleEnabled(true)).resolves.toBe(true);

    expect(chooseModal).not.toHaveBeenCalled();
    expect(settings.schedule.enabled).toBe(true);
    expect(scheduler.start).toHaveBeenCalledTimes(1);
    expect(scheduler.tickToday).not.toHaveBeenCalled();
    expect(durable.runState[TODAY]).toMatchObject({ status: "completed", papersWritten: 7 });
    expect(getSetupStatus(settings, stateStore.snapshot()).firstReportComplete).toBe(true);
  });

  it.each(["failed_permanent", "skipped", "running"] as const)(
    "does not ask or touch a day that is already %s",
    async (status) => {
      const { plugin, scheduler, durable } = await enablePlugin({
        [TODAY]: { status, lastAttempt: 1, attempts: 1, error: "earlier" },
      });

      await expect(plugin.setScheduleEnabled(true)).resolves.toBe(true);

      expect(chooseModal).not.toHaveBeenCalled();
      expect(scheduler.start).toHaveBeenCalledTimes(1);
      expect(durable.runState[TODAY]).toMatchObject({ status, error: "earlier" });
    },
  );

  it("keeps the completed run when today finished while the question was open and the answer was Skip", async () => {
    const { plugin, stateStore, settings, durable } = await enablePlugin({});
    vi.mocked(chooseModal).mockImplementation(async () => {
      // A scheduled run elsewhere (CLI cron, another window) completes today meanwhile.
      durable.runState[TODAY] = { status: "completed", lastAttempt: 2, attempts: 1, papersWritten: 7 };
      return "skip";
    });

    await expect(plugin.setScheduleEnabled(true)).resolves.toBe(true);

    expect(chooseModal).toHaveBeenCalledTimes(1);
    expect(durable.runState[TODAY]).toMatchObject({ status: "completed", papersWritten: 7 });
    await stateStore.loadAuthoritative();
    expect(getSetupStatus(settings, stateStore.snapshot()).firstReportComplete).toBe(true);
    expect((plugin.logger.notice as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[0]).toContain("left as it is");
  });
});

describe("enabling the scheduler while today is still open", () => {
  it.each([
    ["has no record", {}],
    ["is pending", { [TODAY]: { status: "pending", lastAttempt: 1, attempts: 1, error: "no papers yet" } }],
    ["failed transiently", { [TODAY]: { status: "failed_transient", lastAttempt: 1, attempts: 1, failureAttempts: 1, error: "offline" } }],
  ] as const)("still records Skip today when today %s", async (_label, initial) => {
    const { plugin, durable } = await enablePlugin(initial as unknown as RunState);
    vi.mocked(chooseModal).mockResolvedValue("skip");

    await expect(plugin.setScheduleEnabled(true)).resolves.toBe(true);

    expect(chooseModal).toHaveBeenCalledTimes(1);
    expect(durable.runState[TODAY]).toMatchObject({ status: "skipped", error: "user opted out at enable time" });
  });

  it("still runs today when asked to", async () => {
    const { plugin, scheduler } = await enablePlugin({});
    vi.mocked(chooseModal).mockResolvedValue("run");

    await expect(plugin.setScheduleEnabled(true)).resolves.toBe(true);

    await vi.waitFor(() => expect(scheduler.tickToday).toHaveBeenCalledTimes(1));
  });
});
