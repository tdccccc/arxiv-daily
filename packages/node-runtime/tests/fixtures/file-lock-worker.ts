import * as fs from "node:fs/promises";
import * as path from "node:path";
import { NodeFileLock } from "../../src/file-lock";
import { NodeStorageAdapter } from "../../src/storage-adapter";
import { DAILY_RUN_LOCK_KEY, DEFAULT_SETTINGS, PaperIndexStore, RunLock } from "@arxiv-daily/core";

async function main() {
  const [vault, lockRoot, mode, count = "10"] = process.argv.slice(2) as [string, string, string, string?];
  const locks = new NodeFileLock(vault, { lockRoot });
  if (mode === "run") {
    const storage = new NodeStorageAdapter(vault, { lockRoot });
    const lock = new RunLock(() => storage.acquireLock(DAILY_RUN_LOCK_KEY));
    await lock.withLock("2026-09-28", async () => {
      process.send?.("acquired");
      await new Promise<void>((resolve) => process.once("message", () => resolve()));
    });
  } else if (mode === "index") {
    class PausedStorage extends NodeStorageAdapter {
      private paused = false;
      override async readText(file: string): Promise<string> {
        const value = await super.readText(file);
        if (!this.paused && file.endsWith("papers.json")) {
          this.paused = true;
          process.send?.("read");
          await new Promise<void>((resolve) => process.once("message", () => resolve()));
        }
        return value;
      }
    }
    const storage = new PausedStorage(vault, { lockRoot });
    process.send?.("started");
    await new PaperIndexStore(storage, DEFAULT_SETTINGS.output).upsertFromDailyPaper({
      arxivId: `2609.0000${count}`, title: `Paper ${count}`, authors: "Test", date: "2026-09-28",
      arxivCategory: "cs.AI", primaryTopic: "test", detail: false,
    });
  } else if (mode === "hold") {
    const lease = await locks.acquire("counter", { wait: true });
    if (!lease) throw new Error("lock unavailable");
    process.send?.("acquired");
    await new Promise<void>((resolve) => process.once("message", () => resolve()));
    await lease.release();
  } else {
    for (let i = 0; i < Number(count); i += 1) {
      const lease = await locks.acquire("counter", { wait: true });
      if (!lease) throw new Error("lock unavailable");
      try {
        const marker = await fs.open(path.join(vault, "inside"), "wx");
        await marker.close();
        const value = Number(await fs.readFile(path.join(vault, "counter"), "utf8"));
        await new Promise((resolve) => setTimeout(resolve, 4));
        await fs.writeFile(path.join(vault, "counter"), String(value + 1));
        await fs.unlink(path.join(vault, "inside"));
      } finally {
        await lease.release();
      }
    }
  }
  process.disconnect?.();
}
void main().catch((error) => { console.error(error); process.exit(1); });
