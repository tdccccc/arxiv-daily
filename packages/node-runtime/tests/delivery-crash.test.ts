import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { fork } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { build } from "esbuild";
import { DEFAULT_SETTINGS, deliverDailyEmailIfEnabled, deliveryStatePath, sampleDailyDigest } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "../src/storage-adapter";
import type { NativeStorageBinding } from "../src/native-private-storage";

const bindingPath = fileURLToPath(new URL("../native/build/Release/private_storage.node", import.meta.url));
const nativeBinding = createRequire(import.meta.url)(bindingPath) as NativeStorageBinding;
const claimsDir = `${deliveryStatePath(DEFAULT_SETTINGS.output)}.claims`;
const email = { enabled: true, mode: "self" as const, to: "crash@example.test", fromEmail: "from@example.test", apiKey: "re_test" };

let root: string;
let worker: string;
beforeAll(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-delivery-crash-"));
  worker = path.join(root, "worker.cjs");
  await build({ entryPoints: [fileURLToPath(new URL("./fixtures/delivery-crash-worker.ts", import.meta.url))], outfile: worker, bundle: true, platform: "node", format: "cjs", loader: { ".md": "text" }, logLevel: "silent" });
});
afterAll(async () => { if (root) await fs.rm(root, { recursive: true, force: true }); });

/** Runs a delivery in another process that dies while creating the chosen sidecar. */
async function crashWhileCreating(vault: string, backend: string, date: string, killOn: string): Promise<void> {
  const child = fork(worker, [vault, backend === "native" ? bindingPath : "-", date, killOn], { stdio: ["ignore", "pipe", "pipe", "ipc"] });
  let errors = "";
  child.stderr?.on("data", (chunk) => { errors += String(chunk); });
  const exit = await new Promise<{ code: number | null; signal: string | null }>((resolve, reject) => {
    child.once("error", reject);
    child.once("close", (code, signal) => resolve({ code, signal }));
  });
  // Exit code 0 or 3 means the worker finished normally: it was never killed mid-create.
  expect(exit.code === 0 || exit.code === 3, `worker was not killed (${JSON.stringify(exit)}): ${errors}`).toBe(false);
}

async function claimFiles(vault: string): Promise<Array<{ name: string; size: number }>> {
  const dir = path.join(vault, claimsDir);
  const names = (await fs.readdir(dir)).sort();
  return Promise.all(names.map(async (name) => ({ name, size: (await fs.stat(path.join(dir, name))).size })));
}

const backends = process.platform === "linux" ? ["native", "descriptor-anchored"] : ["native"];

describe.each(backends)("a delivery process killed while creating a sidecar (%s storage)", (backend) => {
  async function scenario(killOn: "claim" | "decision" | "result") {
    const vault = await fs.mkdtemp(path.join(root, "vault-"));
    await crashWhileCreating(vault, backend, "2026-08-10", killOn);
    const files = await claimFiles(vault);
    const empty = files.filter((file) => file.size === 0);
    expect(empty.map((file) => file.name.replace(/^[0-9a-f]{64}\./, ""))).toEqual([`g0.${killOn}.json`]);
    const storage = new NodeStorageAdapter(vault, backend === "native" ? { nativeBinding } : {});
    const request = vi.fn(async () => ({ status: 200, headers: {}, bodyText: JSON.stringify({ id: "parent-message" }) }));
    const deliver = (date: string) => deliverDailyEmailIfEnabled(sampleDailyDigest({ date }), {
      storage, http: { request }, output: DEFAULT_SETTINGS.output, email, sleep: async () => {},
    });
    return { vault, empty, deliver, request };
  }

  it("does not stop other dates when the claim file was left empty, and sends the crashed date once", async () => {
    const { deliver, request } = await scenario("claim");

    expect(await deliver("2026-08-11")).toEqual({ kind: "delivered", attempts: 1 });
    expect(await deliver("2026-08-10")).toEqual({ kind: "delivered", attempts: 1 });
    expect(await deliver("2026-08-10")).toEqual({ kind: "skipped", reason: "already_delivered" });
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("does not stop other dates when the decision file was left empty, and keeps the crashed date from sending", async () => {
    const { deliver, request } = await scenario("decision");

    expect(await deliver("2026-08-11")).toEqual({ kind: "delivered", attempts: 1 });
    expect(await deliver("2026-08-10")).toEqual({ kind: "failed", reason: "delivery_claim_unreadable", attempts: 0 });
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("does not stop other dates when the result file was left empty, and never resends the crashed date", async () => {
    const { deliver, request } = await scenario("result");

    expect(await deliver("2026-08-11")).toEqual({ kind: "delivered", attempts: 1 });
    expect(await deliver("2026-08-10")).toEqual({ kind: "skipped", reason: "provider_attempt_started" });
    expect(request).toHaveBeenCalledTimes(1);
  });
});
