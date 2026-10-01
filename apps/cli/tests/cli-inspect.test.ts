import { afterEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_SETTINGS, PaperIndexStore, createStorageStateStore } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { runCli } from "../src/main";
import { DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });

async function setup() {
  const root = await mkdtemp(join(tmpdir(), "arxiv-inspect-"));
  roots.push(root);
  const vaultRoot = join(root, "vault");
  await mkdir(vaultRoot);
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.llm.apiKey = "secret-that-must-never-appear";
  settings.email.apiKey = "mail-secret-that-must-never-appear";
  settings.embedding.apiKey = "embedding-secret-that-must-never-appear";
  settings.llm.baseUrl = "https://user:password@example.test/v1?token=private-endpoint-token";
  settings.arxiv.topics = [{ id: "t", name: "Inference", tag: "inference", description: "Efficient inference", detail: true }];
  const config: CliRuntimeConfig = { settings, vaultRoot, cacheDir: join(root, "cache"), configPath: join(root, "config.toml"), linkStyle: "wikilink", scheduleIntent: { ...DEFAULT_CLI_SCHEDULE } };
  const storage = new NodeStorageAdapter(vaultRoot);
  const index = new PaperIndexStore(storage, settings.output);
  return { config, storage, index, vaultRoot };
}

async function invoke(config: CliRuntimeConfig, argv: string[]) {
  const out: string[] = [], err: string[] = [];
  const runtime = vi.fn(() => { throw new Error("Read-only inspection must not build a generation runtime"); });
  const code = await runCli({ argv, loadConfig: async () => config, buildRuntime: runtime,
    io: { stdout: { write: text => out.push(text) }, stderr: { write: text => err.push(text) } },
  });
  expect(runtime).not.toHaveBeenCalled();
  return { code, stdout: out.join(""), stderr: err.join("") };
}

it("reports existing product configuration without exposing credentials or requiring a library", async () => {
  const { config, vaultRoot } = await setup();
  const result = await invoke(config, ["status"]);
  expect(result.code).toBe(0);
  const data = JSON.parse(result.stdout);
  expect(data.vaultRoot).toBe(vaultRoot);
  expect(data.topics[0].tag).toBe("inference");
  expect(data.paperCount).toBe(0);
  expect(data.recentRuns).toEqual([]);
  expect(data.llm.keyConfigured).toBe(true);
  expect(result.stdout).not.toMatch(/secret-that|user:password|private-endpoint-token|apiKey|api_key/);
  expect(await readdir(vaultRoot)).toEqual([]);
});

it("reports stored run results and searches the real Paper Index", async () => {
  const { config, storage, index } = await setup();
  for (const [arxivId, title] of [["2605.08080", "Efficient inference"], ["2605.08081", "Galaxy measurements"]]) {
    await index.upsertFromDailyPaper({ arxivId, title, authors: "An Author", date: "2026-05-11", arxivCategory: "cs.AI", primaryTopic: "inference", detail: false });
  }
  const state = createStorageStateStore(storage, config.settings.output);
  await state.load();
  await state.setCompleted("2026-05-11", 2);
  await state.setFailed("2026-05-10", "transient", `network error ${config.settings.embedding.apiKey}`);
  const status = await invoke(config, ["status"]);
  expect(status.code).toBe(0);
  expect(status.stdout).not.toContain(config.settings.embedding.apiKey);
  expect(JSON.parse(status.stdout).paperCount).toBe(2);
  expect(JSON.parse(status.stdout).recentRuns[0]).toMatchObject({ date: "2026-05-11", status: "completed", papersWritten: 2 });
  const papers = await invoke(config, ["papers", "--query", "Efficient", "--limit", "1"]);
  expect(papers.code).toBe(0);
  const result = JSON.parse(papers.stdout);
  expect(result.total).toBe(1);
  expect(result.papers[0].arxivId).toBe("2605.08080");
  expect(result.nextOffset).toBeNull();
  const page = await invoke(config, ["papers", "--limit", "1"]);
  expect(JSON.parse(page.stdout).nextOffset).toBe(1);
});

it("can inspect incomplete model setup and rejects malformed pagination", async () => {
  const { config } = await setup();
  config.settings.llm.apiKey = "";
  expect((await invoke(config, ["status"])).code).toBe(0);
  const result = await invoke(config, ["status"]);
  expect(JSON.parse(result.stdout).llm.ready).toBe(false);
  for (const argv of [["papers", "--limit", "0"], ["papers", "--offset", "-1"], ["papers", "--limit", "abc"]]) {
    expect((await invoke(config, argv)).code).toBe(2);
  }
});

it("exposes optional library status through the ordinary product CLI", async () => {
  const { config } = await setup();
  const result = await invoke(config, ["library", "status"]);
  expect(result.code).toBe(0);
  expect(JSON.parse(result.stdout).status.kind).toBe("disconnected");
});
