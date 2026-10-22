import { afterEach, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";
import { runWorkbench, type RunWorkbenchOptions } from "../src/workbench/launch";

afterEach(() => { vi.unstubAllGlobals(); });

/** Starts the real workbench in this process; `stop` is the handler it registered for SIGTERM. */
async function launch(options: Partial<RunWorkbenchOptions> = {}) {
  vi.stubGlobal("__ARXIV_DAILY_WORKBENCH_ASSETS__", { "index.html": { type: "text/html", body: "workbench" } });
  const known = new Set(process.listeners("SIGTERM"));
  const text: string[] = [];
  const io = { stdout: { write: (value: string) => text.push(value) }, stderr: { write: (value: string) => text.push(value) } };
  const config: CliRuntimeConfig = { settings: structuredClone(DEFAULT_SETTINGS), vaultRoot: "/vault", cacheDir: "/cache", configPath: "/config/config.toml", linkStyle: "wikilink", scheduleIntent: { ...DEFAULT_CLI_SCHEDULE } };
  const done = runWorkbench(config, io, { port: 0, open: false, env: {}, ...options });
  await vi.waitFor(() => expect(text.join("")).toMatch(/Workbench: http/));
  const url = /Workbench: (\S+)/.exec(text.join(""))![1]!;
  const stop = () => (process.listeners("SIGTERM").find(listener => !known.has(listener)) as () => void)();
  return { done, url, text, stop };
}

it("stops, as it does on SIGTERM, when the pipe on stdin closes because the host that started it died", async () => {
  const watching = process.stdin.listenerCount("end");
  const { done, url, text } = await launch({ exitOnStdinClose: true });
  expect(process.stdin.listenerCount("end")).toBe(watching + 1);
  expect((await fetch(url)).status).toBe(200);
  process.stdin.emit("end");
  expect(await done).toBe(0);
  expect(text.join("")).toContain("Stopping workbench");
  await expect(fetch(url)).rejects.toThrow();
  expect(process.stdin.listenerCount("end")).toBe(watching);
});

it("does not watch stdin unless asked to, so a terminal's Ctrl+D or a closed stdin never stops it", async () => {
  const watching = process.stdin.listenerCount("end");
  const { done, url, stop } = await launch();
  expect(process.stdin.listenerCount("end")).toBe(watching);
  process.stdin.emit("end");
  expect((await fetch(url)).status).toBe(200);
  stop();
  expect(await done).toBe(0);
});
