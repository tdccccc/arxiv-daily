import type { StorageLock } from "../core/adapters";

// All dates share one Vault-wide lock, including startup temporary-file cleanup.
export const DAILY_RUN_LOCK_KEY = "daily-run";

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

  async withLock<T>(key: string, fn: () => Promise<T>): Promise<T | undefined> {
    if (!this.tryAcquire(key)) return undefined;
    let shared: StorageLock | null | undefined;
    try {
      shared = await this.acquireShared?.(key);
      if (shared === null) return undefined;
      return await fn();
    } finally {
      try {
        await shared?.release();
      } finally {
        this.release(key);
      }
    }
  }
}
