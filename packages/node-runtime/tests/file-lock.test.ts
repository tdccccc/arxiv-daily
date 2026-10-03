import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { fork, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { NodeFileLock } from "../src/file-lock";
import { NodeStorageAdapter } from "../src/storage-adapter";
import { DAILY_RUN_LOCK_KEY, DEFAULT_SETTINGS, Logger, MarkdownWriter, PaperIndexStore, RunLock } from "@arxiv-daily/core";

let root: string;
let worker: string;
const children = new Set<ChildProcess>();
// These cases serialize dozens of real fsync calls across processes. Keep
// correctness assertions intact while allowing slow CI disks beyond 5 seconds.
const contentionTestTimeoutMs = 30_000;

beforeAll(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-lock-tests-"));
  worker = path.join(root, "worker.cjs");
  await build({ entryPoints: [fileURLToPath(new URL("./fixtures/file-lock-worker.ts", import.meta.url))], outfile: worker, bundle: true, platform: "node", format: "cjs", loader: { ".md": "text" }, logLevel: "silent" });
});
afterEach(async () => {
  await Promise.all(Array.from(children, (child) => new Promise<void>((resolve) => {
    if (child.exitCode !== null || child.signalCode !== null) return resolve();
    // 'close' (not 'exit') guarantees the child's stdio has fully drained.
    child.once("close", () => resolve());
    child.kill("SIGKILL");
  })));
  children.clear();
});
afterAll(async () => { if (root) await fs.rm(root, { recursive: true, force: true }); });

async function fixture() {
  const vault = await fs.mkdtemp(path.join(root, "vault-"));
  const lockRoot = path.join(vault, "host-locks");
  await fs.writeFile(path.join(vault, "counter"), "0");
  return { vault, lockRoot, locks: new NodeFileLock(vault, { lockRoot }) };
}

function child(vault: string, lockRoot: string, mode: string, count = 10) {
  const process = fork(worker, [vault, lockRoot, mode, String(count)], { stdio: ["ignore", "pipe", "pipe", "ipc"] });
  children.add(process);
  let errors = "";
  process.stderr?.on("data", (chunk) => { errors += String(chunk); });
  const done = new Promise<void>((resolve, reject) => {
    process.once("error", reject);
    // 'close' (not 'exit') guarantees `errors` has the worker's complete
    // stderr output before the rejection message is built from it.
    process.once("close", (code, signal) => code === 0 || signal === "SIGKILL" ? resolve() : reject(new Error(errors || `worker exit ${code}`)));
  });
  const acquired = new Promise<void>((resolve) => process.on("message", (message) => {
    if (message === "acquired" || message === "read") resolve();
  }));
  const started = new Promise<void>((resolve) => process.on("message", (message) => {
    if (message === "started") resolve();
  }));
  return { process, done, acquired, started };
}

describe("machine-local file lock", () => {
  it("excludes another daily-run host across dates and recovers after its process exits", async () => {
    const { vault, lockRoot } = await fixture();
    const holder = child(vault, lockRoot, "run");
    await holder.acquired;
    const storage = new NodeStorageAdapter(vault, { lockRoot });
    const lock = new RunLock(() => storage.acquireLock(DAILY_RUN_LOCK_KEY));
    expect(await lock.withLock("2026-09-29", async () => "entered")).toBeUndefined();
    holder.process.kill("SIGKILL");
    await holder.done;
    expect(await lock.withLock("2026-09-29", async () => "entered")).toBe("entered");
  });

  it("does not clean the temporary Markdown of a live daily-run subprocess", async () => {
    const { vault, lockRoot } = await fixture();
    const holder = child(vault, lockRoot, "run");
    await holder.acquired;
    const storage = new NodeStorageAdapter(vault, { lockRoot });
    const writer = new MarkdownWriter({ storage, logger: new Logger("error"), arxiv: DEFAULT_SETTINGS.arxiv, output: DEFAULT_SETTINGS.output });
    const temp = "arxiv-daily/daily/2026-09-28.md.tmp";
    await storage.mkdir("arxiv-daily/daily");
    await storage.writeText(temp, "in progress");
    expect(await writer.cleanupTemporaryFiles()).toEqual([]);
    expect(await storage.readText(temp)).toBe("in progress");
    holder.process.kill("SIGKILL");
    await holder.done;
    expect(await writer.cleanupTemporaryFiles()).toEqual([temp]);
  });

  it("holds the shared index lock across the entire read-modify-save transaction", async () => {
    const { vault, lockRoot } = await fixture();
    const store = new PaperIndexStore(new NodeStorageAdapter(vault, { lockRoot }), DEFAULT_SETTINGS.output);
    await store.upsertFromDailyPaper({ arxivId: "2609.00000", title: "Seed", authors: "Test", date: "2026-09-28", arxivCategory: "cs.AI", primaryTopic: "test", detail: false });
    const first = child(vault, lockRoot, "index", 1);
    await first.acquired;
    const second = child(vault, lockRoot, "index", 2);
    let secondRead = false;
    void second.acquired.then(() => { secondRead = true; });
    await second.started;
    try {
      await new Promise((resolve) => setTimeout(resolve, 300));
      expect(secondRead).toBe(false);
    } finally {
      first.process.send("continue");
      await first.done;
      await second.acquired;
      second.process.send("continue");
      await second.done;
    }
    expect(Object.keys((await store.load()).papers).sort()).toEqual(["arxiv:2609.00000", "arxiv:2609.00001", "arxiv:2609.00002"]);
  });

  it("does not admit a stalled acquirer whose claimed generation was pruned meanwhile", async () => {
    const { vault, lockRoot } = await fixture();
    const other = new NodeFileLock(vault, { lockRoot });
    const seed = await other.acquire("counter");
    await seed!.release();
    let stalled = false;
    // The stalled acquirer has read g0 as released and is about to claim g1
    // while three full cycles finish elsewhere; their releases prune g0 and g1.
    const slow = new NodeFileLock(vault, {
      lockRoot,
      beforeOwnerPublish: async () => {
        if (stalled) return;
        stalled = true;
        for (let cycle = 0; cycle < 3; cycle++) {
          const lease = await other.acquire("counter");
          await lease!.release();
        }
      },
    });
    const lease = await slow.acquire("counter", { wait: true });
    expect(lease).not.toBeNull();
    try {
      expect(await other.acquire("counter")).toBeNull();
    } finally {
      await lease!.release();
    }
    const next = await other.acquire("counter");
    expect(next).not.toBeNull();
    await next!.release();
  });

  it("excludes another instance and allows reacquisition after release", async () => {
    const { vault, lockRoot, locks } = await fixture();
    const other = new NodeFileLock(vault, { lockRoot });
    const lease = await locks.acquire("counter");
    expect(lease).not.toBeNull();
    expect(await other.acquire("counter")).toBeNull();
    await lease!.release();
    const next = await other.acquire("counter");
    expect(next).not.toBeNull();
    await next!.release();
  });

  it("serializes actual child processes around a read-modify-write", async () => {
    const { vault, lockRoot } = await fixture();
    await Promise.all([child(vault, lockRoot, "increment").done, child(vault, lockRoot, "increment").done, child(vault, lockRoot, "increment").done]);
    expect(await fs.readFile(path.join(vault, "counter"), "utf8")).toBe("30");
  }, contentionTestTimeoutMs);

  it("recovers a crashed owner once despite competing recovery processes", async () => {
    const { vault, lockRoot } = await fixture();
    const holder = child(vault, lockRoot, "hold");
    await holder.acquired;
    holder.process.kill("SIGKILL");
    await holder.done;
    await Promise.all([child(vault, lockRoot, "increment").done, child(vault, lockRoot, "increment").done]);
    expect(await fs.readFile(path.join(vault, "counter"), "utf8")).toBe("20");
  }, contentionTestTimeoutMs);

  it("does not steal a live holder on timeout or cancellation", async () => {
    const { vault, lockRoot, locks } = await fixture();
    const holder = child(vault, lockRoot, "hold");
    await holder.acquired;
    await expect(locks.acquire("counter", { wait: true, timeoutMs: 30 })).rejects.toThrow("timed out");
    const controller = new AbortController();
    controller.abort();
    await expect(locks.acquire("counter", { wait: true, signal: controller.signal })).rejects.toThrow();
    expect(await locks.acquire("counter")).toBeNull();
    holder.process.send("release");
    await holder.done;
    const next = await locks.acquire("counter");
    expect(next).not.toBeNull();
    await next!.release();
  });
});
