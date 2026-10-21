import type { StorageLock } from "../core/adapters";

// All dates share one Vault-wide lock, including startup temporary-file cleanup.
export const DAILY_RUN_LOCK_KEY = "daily-run";

/** What the holder of a run lock may conclude about other processes. */
export interface RunLockHold {
  /**
   * True when the Vault-wide lock shared with other programs on this machine
   * is held. Runs mark their date `running` only while holding it, so no live
   * run exists anywhere else and any `running` record is a leftover of a dead
   * one. False when the host has no such lock and only this one is excluded.
   */
  crossProcess: boolean;
}

export class RunLock {
  private held = new Set<string>();

  constructor(
    private readonly acquireShared?: (key: string) => Promise<StorageLock | null> | undefined,
  ) {}

  tryAcquire(key: string): boolean {
    if (this.held.has(key)) return false;
    this.held.add(key);
    return true;
  }

  release(key: string): void {
    this.held.delete(key);
  }

  isHeld(key: string): boolean {
    return this.held.has(key);
  }

  async withLock<T>(key: string, fn: (hold: RunLockHold) => Promise<T>): Promise<T | undefined> {
    if (!this.tryAcquire(key)) return undefined;
    let shared: StorageLock | null | undefined;
    try {
      shared = await this.acquireShared?.(key);
      if (shared === null) return undefined;
      return await fn({ crossProcess: shared !== undefined });
    } finally {
      try {
        await shared?.release();
      } finally {
        this.release(key);
      }
    }
  }
}
