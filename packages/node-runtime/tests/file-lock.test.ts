import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { fork, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { NodeFileLock } from "../src/file-lock";

let root: string;
let worker: string;
const children = new Set<ChildProcess>();

beforeAll(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-lock-tests-"));
  worker = path.join(root, "worker.cjs");
  await build({ entryPoints: [fileURLToPath(new URL("./fixtures/file-lock-worker.ts", import.meta.url))], outfile: worker, bundle: true, platform: "node", format: "cjs", loader: { ".md": "text" }, logLevel: "silent" });
});
afterEach(async () => {
  await Promise.all(Array.from(children, (child) => new Promise<void>((resolve) => {
    if (child.exitCode !== null || child.signalCode !== null) return resolve();
    child.once("exit", () => resolve());
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
    process.once("exit", (code, signal) => code === 0 || signal === "SIGKILL" ? resolve() : reject(new Error(errors || `worker exit ${code}`)));
  });
  const acquired = new Promise<void>((resolve) => process.once("message", () => resolve()));
  return { process, done, acquired };
}

describe("machine-local file lock", () => {
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
  });

  it("recovers a crashed owner once despite competing recovery processes", async () => {
    const { vault, lockRoot } = await fixture();
    const holder = child(vault, lockRoot, "hold");
    await holder.acquired;
    holder.process.kill("SIGKILL");
    await holder.done;
    await Promise.all([child(vault, lockRoot, "increment").done, child(vault, lockRoot, "increment").done]);
    expect(await fs.readFile(path.join(vault, "counter"), "utf8")).toBe("20");
  });

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
