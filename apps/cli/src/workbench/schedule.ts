import { clearRepeatingTimer, setRepeatingTimer, type PluginSettings } from "@arxiv-daily/core";
/** UI-process cadence only; the shared scheduler owns windows, weekdays and completion. */
export class WorkbenchSchedule {
  constructor(private busy: () => boolean, private run: () => void) {}
  private timer?: ReturnType<typeof setInterval>;
  update(schedule: PluginSettings["schedule"] | undefined): void {
    this.close();
    if (!schedule?.enabled) return;
    this.timer = setRepeatingTimer(() => { if (!this.busy()) this.run(); }, Math.max(1, schedule.tickIntervalMin) * 60_000);
    this.timer.unref?.();
  }
  close(): void { clearRepeatingTimer(this.timer); this.timer = undefined; }
}
