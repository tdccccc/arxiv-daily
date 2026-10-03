import { afterEach, expect, it, vi } from "vitest";
import { WorkbenchSchedule } from "../src/workbench/schedule";
afterEach(() => vi.useRealTimers());
it("checks at the configured minute interval, skips busy work, updates cadence and stops on close", async () => {
  vi.useFakeTimers();
  const run = vi.fn(); let busy = false;
  const timer = new WorkbenchSchedule(() => busy, run);
  timer.update({ enabled: true, runAtLocal: "09:30", runUntilLocal: "18:00", tickIntervalMin: 20 });
  await vi.advanceTimersByTimeAsync(19 * 60_000); expect(run).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(60_000); expect(run).toHaveBeenCalledTimes(1);
  busy = true; await vi.advanceTimersByTimeAsync(20 * 60_000); expect(run).toHaveBeenCalledTimes(1);
  busy = false; timer.update({ enabled: true, runAtLocal: "09:30", runUntilLocal: "18:00", tickIntervalMin: 5 });
  await vi.advanceTimersByTimeAsync(5 * 60_000); expect(run).toHaveBeenCalledTimes(2);
  timer.update({ enabled: false, runAtLocal: "09:30", runUntilLocal: "18:00", tickIntervalMin: 5 });
  await vi.advanceTimersByTimeAsync(60 * 60_000); expect(run).toHaveBeenCalledTimes(2);
  timer.update({ enabled: true, runAtLocal: "09:30", runUntilLocal: "18:00", tickIntervalMin: 5 }); timer.close();
  await vi.advanceTimersByTimeAsync(60 * 60_000); expect(run).toHaveBeenCalledTimes(2);
});

it("connects the workbench timer to scheduled CLI checks and stops its timer on shutdown", async () => {
  const { startWorkbench } = await import('../src/workbench/server');
  const { DEFAULT_SETTINGS } = await import('@arxiv-daily/core');
  const { DEFAULT_CLI_SCHEDULE } = await import('../src/config');
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
  const run = vi.fn(async () => 0);
  const app = await startWorkbench({ config: { settings: structuredClone(DEFAULT_SETTINGS), vaultRoot:'/unused-test-vault',cacheDir:'/unused-test-cache',configPath:'/unused-test-config',linkStyle:'relative',scheduleIntent:{...DEFAULT_CLI_SCHEDULE},workbenchSchedule:{enabled:true,runAtLocal:'09:00',runUntilLocal:'18:00',tickIntervalMin:7}},run });
  try {
    await vi.advanceTimersByTimeAsync(7 * 60_000);
    expect(run).toHaveBeenCalledWith(['run','--scheduled'],expect.anything(),expect.any(AbortSignal));
  } finally { await app.close(); }
  await vi.advanceTimersByTimeAsync(7 * 60_000);
  expect(run).toHaveBeenCalledOnce();
});
