export class RunLock {
  private held = new Set<string>();

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
    if (!this.tryAcquire(key)) {
      if (typeof console !== "undefined") {
        console.log(`[arxiv-daily] lock: ${key} already held, skipping`);
      }
      return undefined;
    }
    if (typeof console !== "undefined") {
      console.log(`[arxiv-daily] lock: ${key} acquired`);
    }
    try {
      return await fn();
    } finally {
      this.release(key);
      if (typeof console !== "undefined") {
        console.log(`[arxiv-daily] lock: ${key} released`);
      }
    }
  }
}
