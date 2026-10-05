import { afterEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DEFAULT_SETTINGS, PaperIndexStore, appendGenerationMetrics } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";
import { startWorkbench } from "../src/workbench/server";
import { WorkbenchError } from "../src/workbench/documents";

const cleanup: Array<() => Promise<unknown>> = [];
afterEach(async () => { for (const fn of cleanup.splice(0).reverse()) await fn(); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "paper-workspace-")); cleanup.push(() => rm(root, { recursive: true, force: true }));
  const settings = structuredClone(DEFAULT_SETTINGS); settings.output.dailyDir = "research/daily"; settings.output.papersDir = "research/papers";
  const config: CliRuntimeConfig = { settings, vaultRoot: join(root, "vault"), cacheDir: join(root, "cache"), configPath: join(root, "config.toml"), linkStyle: "relative", scheduleIntent: { ...DEFAULT_CLI_SCHEDULE } };
  await mkdir(join(config.vaultRoot, "research/daily"), { recursive: true }); await mkdir(join(config.vaultRoot, "research/papers"), { recursive: true });
  const report = "research/daily/2026-10-01.md", note = "research/papers/2609.10001.md";
  await writeFile(join(config.vaultRoot, report), "# A saved report\n"); await writeFile(join(config.vaultRoot, note), "# Existing detail\n");
  const index = new PaperIndexStore(new NodeStorageAdapter(config.vaultRoot), settings.output);
  for (const [id, title] of [["2609.10001", "Efficient inference"], ["2609.10002", "Calibration methods"], ["2609.10003", "Manual-only paper"]]) {
    await index.upsertFromDailyPaper({ arxivId: id!, title: title!, authors: "Ada", date: "2026-10-01", arxivCategory: "cs.AI", primaryTopic: "inference", detail: id === "2609.10001", ...(id !== "2609.10003" ? { dailyReport: report } : {}), ...(id === "2609.10001" ? { paperPath: note } : {}) });
  }
  await index.setSummaries({ "2609.10002": { whyRelevant: "Saved calibration reason", coreProblem: "Calibration" } });
  const run = vi.fn(async () => 0), beforeWrite = vi.fn(async () => {});
  const app = await startWorkbench({ config, run, beforeWrite }); cleanup.push(app.close);
  const get = (route: string, init?: RequestInit) => fetch(new URL(route, app.url), init);
  const mark = (body: unknown) => get("api/paper/mark", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return { root, config, index, app, get, mark, run, beforeWrite, report, note };
}

it("lists index-only papers, real details and exact daily occurrences without writes or generation", async () => {
  const { get, config, index, run, report, note } = await fixture();
  const before = await readFile(join(config.vaultRoot, index.paths.papersJsonPath), "utf8");
  const response = await get("api/papers"); expect(response.status).toBe(200);
  const all = await response.json(); expect(all.total).toBe(3);
  expect(all.papers.find((p: { arxivId: string }) => p.arxivId === "2609.10002")).toMatchObject({ detailPath: null, summary: { whyRelevant: "Saved calibration reason" } });
  expect(all.papers.find((p: { arxivId: string }) => p.arxivId === "2609.10001").detailPath).toBe(note);
  const dated = await (await get("api/papers?date=2026-10-01")).json(); expect(dated.total).toBe(2); expect(dated.day.reportPath).toBe(report);
  expect((await (await get("api/papers?q=Calibration")).json()).total).toBe(1);
  const page = await (await get("api/papers?limit=1&sort=title&direction=asc")).json(); expect(page.papers[0].title).toBe("Calibration methods"); expect(page.nextOffset).toBe(1);
  expect(await readFile(join(config.vaultRoot, index.paths.papersJsonPath), "utf8")).toBe(before); expect(run).not.toHaveBeenCalled();
});

it("persists independent reading and star marks through the existing index, preserving notes and metadata", async () => {
  const { mark, get, index, config, note } = await fixture();
  const key = "arxiv:2609.10002", original = await index.get(key);
  expect((await mark({ key, action: "status", value: "to_read", expected: "inbox" })).status).toBe(200);
  expect((await mark({ key, action: "star", value: true, expected: "normal" })).status).toBe(200);
  const fresh = new PaperIndexStore(new NodeStorageAdapter(config.vaultRoot), config.settings.output);
  expect(await fresh.get(key)).toEqual({ ...original, status: "to_read", priority: "high" });
  expect((await (await get("api/papers?scope=to_read")).json()).total).toBe(1);
  expect((await (await get("api/papers?scope=starred")).json()).total).toBe(1);
  expect((await mark({ key, action: "status", value: "read", expected: "to_read" })).status).toBe(200);
  expect((await fresh.get(key))?.priority).toBe("high"); expect((await (await get("api/papers?scope=read")).json()).total).toBe(1);
  expect(await readFile(join(config.vaultRoot, note), "utf8")).toBe("# Existing detail\n");
});

it("rejects stale conflicts and invalid writes, supports no-op retries and preserves legacy saved/low state", async () => {
  const { mark, get, index, config, beforeWrite } = await fixture(); const key = "arxiv:2609.10002";
  await index.setStatus(key, "saved"); await index.setPriority(key, "low");
  expect((await mark({ key, action: "status", value: "read", expected: "inbox" })).status).toBe(409);
  expect((await mark({ key, action: "star", value: true, expected: "low" })).status).toBe(200);
  expect((await index.get(key))?.status).toBe("saved");
  const before = await readFile(join(config.vaultRoot, index.paths.papersJsonPath), "utf8");
  expect((await mark({ key, action: "star", value: true, expected: "low" })).status).toBe(200);
  expect(await readFile(join(config.vaultRoot, index.paths.papersJsonPath), "utf8")).toBe(before);
  expect((await mark({ key, action: "status", value: "destroy", expected: "saved" })).status).toBe(400);
  expect((await mark({ key, action: "status", value: ["read"], expected: "saved" })).status).toBe(400);
  expect((await mark({ key: "arxiv:0000.00000", action: "star", value: true, expected: "normal" })).status).toBe(404);
  beforeWrite.mockRejectedValueOnce(new WorkbenchError(409, "配置已改变"));
  expect((await mark({ key, action: "status", value: "read", expected: "saved" })).status).toBe(409);
  expect((await get("api/papers?limit=0")).status).toBe(400); expect((await get("api/papers?date=2026-02-30")).status).toBe(400);
  expect((await get("api/paper/mark", { method: "POST", headers: { Origin: "https://example.com", "Content-Type": "application/json" }, body: "{}" })).status).toBe(403);
});

it("never exposes an index path outside allowed documents and reports missing papers", async () => {
  const { get, index } = await fixture();
  await index.setPaperPath("2609.10002", "private.md");
  const response = await get("api/paper?key=arxiv:2609.10002"); expect(response.status).toBe(200);
  expect((await response.json()).paper.detailPath).toBeNull();
  expect((await get("api/paper?key=missing")).status).toBe(404);
});

it("keeps layout preferences outside Vault data and restores them in another server", async () => {
  const { get, config } = await fixture();
  const before = await get("api/preferences"); expect(before.status).toBe(200); expect(await before.json()).toEqual({ sidebarWidth: null, sidebarCollapsed: false, appearance: { theme: "light", language: "zh" } });
  const post = (body: unknown) => get("api/preferences", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  expect((await post({ sidebarWidth: 570, sidebarCollapsed: true })).status).toBe(200);
  const another = await startWorkbench({ config }); cleanup.push(another.close);
  expect(await (await fetch(new URL("api/preferences", another.url))).json()).toEqual({ sidebarWidth: 570, sidebarCollapsed: true, appearance: { theme: "light", language: "zh" } });
  expect((await post({ sidebarWidth: 9000, sidebarCollapsed: false })).status).toBe(400);
});

it("serializes competing marks and preserves independent fields across requests", async () => {
  const { mark, index } = await fixture(); const key = "arxiv:2609.10002";
  const responses = await Promise.all([
    mark({ key, action: "status", value: "to_read", expected: "inbox" }),
    mark({ key, action: "status", value: "read", expected: "inbox" }),
    mark({ key, action: "star", value: true, expected: "normal" }),
  ]);
  expect(responses.slice(0, 2).map(response => response.status).sort()).toEqual([200, 409]);
  expect(responses[2]!.status).toBe(200);
  expect(await index.get(key)).toMatchObject({ priority: "high" });
});

it("counts search and topic matches before scope while preserving absent-index report access", async () => {
  const { get, index, config, report } = await fixture();
  await index.setStatus("2609.10002", "read");
  await index.setStatus("2609.10003", "ignored");
  const data = await (await get("api/papers?q=calibration&topic=inference&scope=inbox")).json();
  expect(data).toMatchObject({ total: 0, libraryCount: 2, counts: { all: 1, inbox: 0, read: 1 } });
  await rm(join(config.vaultRoot, index.paths.papersJsonPath));
  await rm(join(config.vaultRoot, `${index.paths.papersJsonPath}.bak`), { force: true });
  const missing = await (await get("api/papers?date=2026-10-01")).json();
  expect(missing).toMatchObject({ total: 0, day: { reportPath: report, papers: null } });
  expect((await get(`api/document?path=${encodeURIComponent(report)}`)).status).toBe(200);
  await expect(readFile(join(config.vaultRoot, index.paths.papersJsonPath))).rejects.toMatchObject({ code: "ENOENT" });
});


it("loads generation metrics only for one paper and identifies the whole daily report scope", async () => {
  const { get, config, report, note, index } = await fixture();
  const metrics = { logicalCalls: 2, attempts: 2, elapsedMs: 800, usageComplete: true, inputTokens: 120, outputTokens: 30, totalTokens: 150, pipelineElapsedMs: 1600, generatedAt: "2026-10-01T11:22:33.000Z" };
  await writeFile(join(config.vaultRoot, report), appendGenerationMetrics("# Daily", metrics));
  const newerMissing = "research/daily/2026-10-02.md";
  await index.mutate(inbox => { inbox.papers["arxiv:2609.10002"]!.dailyReports.push(newerMissing); return { result: undefined, changed: true }; });
  const all = await (await get("api/papers")).json();
  expect(all.papers.every((paper: Record<string, unknown>) => !("generation" in paper))).toBe(true);
  const { paper } = await (await get("api/paper?key=arxiv:2609.10002")).json();
  expect(paper.generation).toEqual({ scope: "daily", sourcePath: report, metrics });
  await writeFile(join(config.vaultRoot, note), appendGenerationMetrics("# Detail", { ...metrics, inputTokens: 10 }));
  const detail = await (await get("api/paper?key=arxiv:2609.10001")).json();
  expect(detail.paper.generation).toMatchObject({ scope: "paper", sourcePath: note, metrics: { inputTokens: 10 } });
  const legacy = await (await get("api/paper?key=arxiv:2609.10003")).json();
  expect(legacy.paper.generation).toBeNull();
});
