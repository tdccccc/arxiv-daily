import type { RunState, RunStateEntry, RunStatus } from "../settings/types";

const MAX_TRANSIENT_ATTEMPTS = 10;

export type StateLoadFn = () => Promise<{ runState: RunState }>;
export type StateSaveFn = (data: { runState: RunState }) => Promise<void>;

export class StateStore {
  private state: RunState = {};

  constructor(
    private readonly loadFn: StateLoadFn,
    private readonly saveFn: StateSaveFn,
  ) {}

  async load(): Promise<void> {
    const data = await this.loadFn();
    this.state = data?.runState ?? {};
  }

  get(date: string): RunStateEntry {
    return (
      this.state[date] ?? {
        status: "pending" as RunStatus,
        lastAttempt: 0,
        attempts: 0,
      }
    );
  }

  isDone(date: string): boolean {
    const s = this.get(date).status;
    return s === "completed" || s === "failed_permanent";
  }

  async setRunning(date: string): Promise<void> {
    const prev = this.get(date);
    this.state[date] = {
      ...prev,
      status: "running",
      lastAttempt: Date.now(),
      attempts: prev.attempts + 1,
    };
    await this.saveFn({ runState: this.state });
  }

  async setCompleted(date: string, papersWritten: number): Promise<void> {
    const prev = this.get(date);
    this.state[date] = {
      ...prev,
      status: "completed",
      lastAttempt: Date.now(),
      papersWritten,
      error: undefined,
    };
    await this.saveFn({ runState: this.state });
  }

  async setFailed(
    date: string,
    kind: "transient" | "permanent",
    message: string,
  ): Promise<void> {
    const prev = this.get(date);
    let status: RunStatus = kind === "permanent" ? "failed_permanent" : "failed_transient";
    if (status === "failed_transient" && prev.attempts >= MAX_TRANSIENT_ATTEMPTS) {
      status = "failed_permanent";
    }
    this.state[date] = {
      ...prev,
      status,
      lastAttempt: Date.now(),
      error: message,
    };
    await this.saveFn({ runState: this.state });
  }

  snapshot(): RunState {
    const out: RunState = {};
    for (const [k, v] of Object.entries(this.state)) {
      out[k] = { ...v };
    }
    return out;
  }
}
