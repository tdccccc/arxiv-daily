import { afterAll, beforeAll, expect, it } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runCliChild } from "../src/workbench/launch";

// Stands in for `arxiv-daily run`: like the real command it cancels gracefully on
// the first SIGINT or workbench cancel, and exits at once on a second signal.
const FAKE_RUN = `
let signals = 0;
let cancels = 0;
let timer;
const finish = () => process.stdout.write(JSON.stringify({ signals, cancels }) + "\\n", () => process.exit(0));
const settle = () => { clearTimeout(timer); timer = setTimeout(finish, 150); };
process.on("SIGINT", () => { signals += 1; if (signals > 1) process.exit(130); settle(); });
process.on("message", message => {
  if (message && message.type === "arxiv-daily/cancel") { cancels += 1; settle(); }
});
process.stdout.write("pid " + process.pid + "\\n");
setInterval(() => {}, 1000);
`;

let dir: string;
let script: string;
beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "arxiv-workbench-child-"));
  script = join(dir, "fake-run.cjs");
  await writeFile(script, FAKE_RUN);
});
afterAll(async () => { await rm(dir, { recursive: true, force: true }); });

function start(signal: AbortSignal) {
  const out: string[] = [];
  const promise = runCliChild(script, ["run", "--date", "2026-10-20"], { ...process.env }, { stdout: { write: chunk => out.push(String(chunk)) }, stderr: { write: () => {} } }, signal);
  const text = () => out.join("");
  return {
    promise,
    pid: async () => { await waitFor(() => /pid \d+/.test(text())); return Number(/pid (\d+)/.exec(text())![1]); },
    summary: () => JSON.parse(text().split("\n").find(line => line.startsWith("{"))!) as { signals: number; cancels: number },
  };
}

async function waitFor(done: () => boolean) {
  for (let i = 0; i < 400 && !done(); i += 1) await new Promise(resolve => setTimeout(resolve, 25));
  expect(done()).toBe(true);
}

it("cancels a run with one message to the child and no signal", async () => {
  const controller = new AbortController();
  const run = start(controller.signal);
  await run.pid();

  controller.abort();

  expect(await run.promise).toBe(0);
  expect(run.summary()).toEqual({ signals: 0, cancels: 1 });
});

it("delivers the cancel once when the task was cancelled before the child started", async () => {
  const controller = new AbortController();
  controller.abort();
  const run = start(controller.signal);

  expect(await run.promise).toBe(0);
  expect(run.summary()).toEqual({ signals: 0, cancels: 1 });
});

it.skipIf(process.platform === "win32")("lets the child finish cancelling when a terminal Ctrl+C has signalled it as well", async () => {
  const controller = new AbortController();
  const run = start(controller.signal);
  const pid = await run.pid();

  // A terminal sends SIGINT to the whole foreground group: the child gets its own copy
  // while the workbench starts stopping and cancels the task.
  process.kill(pid, "SIGINT");
  controller.abort();

  expect(await run.promise).toBe(0);
  expect(run.summary()).toEqual({ signals: 1, cancels: 1 });
});
