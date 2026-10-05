import { afterEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { get as httpGet } from "node:http";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import { loadCliConfig, DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";
import { startWorkbench, type WorkbenchOptions } from "../src/workbench/server";

const cleanup: Array<() => Promise<unknown>> = [];
afterEach(async () => { for (const close of cleanup.splice(0).reverse()) await close(); });

async function setup(run?: WorkbenchOptions["run"]) {
  const root = await mkdtemp(join(tmpdir(), "arxiv-reader-"));
  cleanup.push(() => rm(root, { recursive: true, force: true }));
  const vaultRoot = join(root, "vault");
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.output.dailyDir = "research/daily";
  settings.output.papersDir = "research/papers";
  settings.llm.apiKey = "secret-model-key";
  settings.embedding.apiKey = "secret-embedding-key";
  const config: CliRuntimeConfig = { settings, vaultRoot, configPath: join(root, "config.toml"), cacheDir: join(root, "cache"), linkStyle: "wikilink", scheduleIntent: { ...DEFAULT_CLI_SCHEDULE } };
  await mkdir(join(vaultRoot, "research/daily"), { recursive: true });
  await mkdir(join(vaultRoot, "research/papers/figures"), { recursive: true });
  const daily = "---\ndate: 2026-10-01\ntags: [arxiv, daily]\n---\n# 今日论文\n\n[[1706.03762|Attention]]\n\n[相对链接](../papers/1706.03762.md#方法)\n";
  const paper = '---\ntitle: "Attention Is All You Need"\nauthors: "A. Researcher"\narxiv_id: "1706.03762"\npublished: "[[research/daily/2026-10-01|2026-10-01]]"\n---\n# Attention\n\n## 方法\n\n正文 $x^2$。\n\n![Figure](figures/chart.png)\n';
  await writeFile(join(vaultRoot, "research/daily/2026-10-01.md"), daily);
  await writeFile(join(vaultRoot, "research/papers/1706.03762.md"), paper);
  await writeFile(join(vaultRoot, "research/papers/figures/chart.png"), Buffer.from([137, 80, 78, 71]));
  await writeFile(join(vaultRoot, "private.md"), "private note must not be served");
  await writeFile(config.configPath, "secret configuration");
  const app = await startWorkbench({ config, run, assets: { "index.html": { type: "text/html", body: "<!doctype html><title>arXiv Daily</title>" }, "app.js": { type: "text/javascript", body: "console.log('reader');" } } });
  cleanup.push(app.close);
  return { ...app, root, config, daily, paper, vaultRoot, get: (path: string, init?: RequestInit) => fetch(new URL(path, app.url), init) };
}

it("lists configured Markdown outputs and reads them without writing user files", async () => {
  const { get, vaultRoot, daily, paper } = await setup();
  const response = await get("api/documents?kind=daily");
  expect(response.status).toBe(200);
  const list = await response.json();
  expect(list.counts).toEqual({ daily: 1, papers: 1 });
  expect(list.documents).toMatchObject([{ path: "research/daily/2026-10-01.md", kind: "daily", title: "今日论文" }]);
  const search = await (await get("api/documents?kind=papers&q=attention")).json();
  expect(search.total).toBe(1);
  expect(search.documents[0].title).toBe("Attention Is All You Need");
  const doc = await (await get("api/document?path=research/daily/2026-10-01.md")).json();
  expect(doc.html).toContain("?document=research%2Fpapers%2F1706.03762.md");
  expect(doc.html).toContain("#方法");
  const detail = await (await get("api/document?path=research/papers/1706.03762.md")).json();
  expect(detail.html).toContain("katex");
  expect(detail.html).toContain("api/asset?path=research%2Fpapers%2Ffigures%2Fchart.png");
  expect(detail.related).toContainEqual({ title: "2026-10-01", path: "research/daily/2026-10-01.md" });
  const raw = await get("api/raw?path=research/papers/1706.03762.md");
  expect(raw.headers.get("content-type")).toContain("text/plain");
  expect(await raw.text()).toBe(paper);
  expect(await readFile(join(vaultRoot, "research/daily/2026-10-01.md"), "utf8")).toBe(daily);
  expect(await readdir(join(vaultRoot, "research"))).toEqual(["daily", "papers"]);
});

it("serves local image assets but does not expose unrelated files or follow escaping symlinks", async () => {
  const { get, vaultRoot, config } = await setup();
  await symlink(join(vaultRoot, "private.md"), join(vaultRoot, "research/papers/escape.md"));
  await symlink(config.configPath, join(vaultRoot, "research/papers/figures/escape.png"));
  await symlink(vaultRoot, join(vaultRoot, "research/papers/other"));
  expect((await get("api/asset?path=research/papers/figures/chart.png")).status).toBe(200);
  for (const path of ["private.md", "../config.toml", "/etc/passwd", "research/papers/escape.md", "research/papers/other/private.md", "research/papers/1706.03762.md%00"]) {
    expect((await get(`api/document?path=${encodeURIComponent(path)}`)).status, path).toBe(404);
  }
  for (const path of ["../config.toml", "research/papers/figures/escape.png", "research/papers/1706.03762.md"]) {
    expect((await get(`api/asset?path=${encodeURIComponent(path)}`)).status, path).toBe(404);
  }
  const list = await (await get("api/documents?kind=papers")).json();
  expect(list.total).toBe(1);
});

it("requires the launch capability, correct Host and same-origin browser requests", async () => {
  const { get, url } = await setup();
  expect(new URL(url).hostname).toBe("127.0.0.1");
  expect(new URL(url).pathname).toMatch(/^\/[a-f0-9]{48}\/$/);
  const root = await fetch(new URL("/api/status", url));
  expect(root.status).toBe(404);
  expect((await get("api/status", { headers: { Origin: "https://example.com" } })).status).toBe(403);
  // Fetch rewrites Host in current Node; use the actual HTTP wire boundary here.
  const foreignHost = await new Promise<number | undefined>((resolve, reject) => {
    httpGet(new URL("api/status", url), { headers: { Host: "evil.example" } }, res => { res.resume(); resolve(res.statusCode); }).on("error", reject);
  });
  expect(foreignHost).toBe(403);
  const html = await get("");
  expect(html.status).toBe(200);
  expect(html.headers.get("content-security-policy")).toContain("frame-ancestors 'none'");
  expect(html.headers.get("referrer-policy")).toBe("no-referrer");
  expect(html.headers.get("access-control-allow-origin")).toBeNull();
  expect((await get("api/status")).headers.get("cache-control")).toBe("no-store");
  // Users may click their private launch URL from a chat website.
  expect((await get("", { headers: { "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document" } })).status).toBe(200);
  expect((await get("api/status", { headers: { "Sec-Fetch-Site": "cross-site" } })).status).toBe(403);
});

it("shows safe configuration and empty / missing data without calling a model", async () => {
  const run = vi.fn<NonNullable<WorkbenchOptions["run"]>>();
  const { get, vaultRoot } = await setup(run);
  const status = await get("api/status");
  expect(status.status).toBe(200);
  expect(await status.text()).not.toContain("secret-");
  await rm(join(vaultRoot, "research"), { recursive: true });
  expect((await (await get("api/documents")).json()).documents).toEqual([]);
  expect((await get("api/document?path=research/daily/missing.md")).status).toBe(404);
  expect(run).not.toHaveBeenCalled();
});

it("validates pagination and request bodies at the HTTP boundary", async () => {
  const { get } = await setup();
  for (const query of ["limit=0", "offset=-1", "kind=private", "limit=foo"]) {
    expect((await get(`api/documents?${query}`)).status).toBe(400);
  }
  for (const body of ["{", JSON.stringify({ kind: "shell", command: "echo bad" }), JSON.stringify({ kind: "daily", date: "2026-02-31" }), JSON.stringify({ kind: "paper", id: "x; echo bad" })]) {
    expect((await get("api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body })).status).toBe(400);
  }
  expect((await get("api/runs", { method: "POST", body: "{}" })).status).toBe(415);
});

it("dispatches explicit runs once, reports progress, redacts secrets and serializes jobs", async () => {
  let finish!: (code: number) => void;
  const run = vi.fn<NonNullable<WorkbenchOptions["run"]>>((_args, io) => {
    io.stdout.write("Running filter\n");
    io.stderr.write("provider secret-model-key\n");
    return new Promise(resolve => { finish = resolve; });
  });
  const { get } = await setup(run);
  const post = () => get("api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind: "daily", date: "2026-10-01" }) });
  expect((await post()).status).toBe(202);
  expect(run.mock.calls[0]![0]).toEqual(["run", "--date", "2026-10-01"]);
  const running = await (await get("api/runs/current")).json();
  expect(running.run.status).toBe("running");
  expect(running.run.output).toContain("Running filter");
  expect(running.run.output).not.toContain("secret-model-key");
  expect((await post()).status).toBe(409);
  expect(run).toHaveBeenCalledOnce();
  finish(0);
  await vi.waitFor(async () => expect((await (await get("api/runs/current")).json()).run.status).toBe("completed"));
});

it("cancels a run using its AbortSignal and exposes execution failures", async () => {
  const run = vi.fn<NonNullable<WorkbenchOptions["run"]>>((_args, _io, signal) => new Promise(resolve => signal.addEventListener("abort", () => resolve(1), { once: true })));
  const { get } = await setup(run);
  const started = await (await get("api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind: "paper", id: "1706.03762" }) })).json();
  expect(run.mock.calls[0]![0]).toEqual(["run", "--id", "1706.03762"]);
  expect((await get("api/runs/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: started.run.id }) })).status).toBe(202);
  await vi.waitFor(async () => expect((await (await get("api/runs/current")).json()).run.status).toBe("cancelled"));
  run.mockRejectedValueOnce(new Error("provider secret-model-key failed"));
  await get("api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind: "daily", date: "2026-10-01" }) });
  await vi.waitFor(async () => {
    const { run: result } = await (await get("api/runs/current")).json();
    expect(result.status).toBe("failed");
    expect(result.output).toContain("provider");
    expect(result.output).not.toContain("secret-model-key");
  });
});

it("allows an explicitly configured local frame without admitting that parent's API writes", async () => {
  const { config, get } = await setup();
  expect((await get("")).headers.get("content-security-policy")).toContain("frame-ancestors 'none'");
  const framed = await startWorkbench({ config, frameOrigin: "http://127.0.0.1:3080" }); cleanup.push(framed.close);
  const response = await fetch(new URL("api/status", framed.url));
  expect(response.headers.get("content-security-policy")).toContain("frame-ancestors http://127.0.0.1:3080");
  expect((await fetch(new URL("api/preferences", framed.url), { method: "POST", headers: { Origin: "http://127.0.0.1:3080", "Content-Type": "application/json" }, body: '{}' })).status).toBe(403);
  await expect(startWorkbench({ config, frameOrigin: "https://example.com" })).rejects.toThrow();
});

it("permits the DSH desktop frame but still refuses API writes from its parent", async () => {
  const { config } = await setup();
  const app = await startWorkbench({ config, frameOrigin: "dsh-app://app" }); cleanup.push(app.close);
  expect((await fetch(new URL("api/status", app.url))).headers.get("content-security-policy")).toContain("frame-ancestors dsh-app://app");
  expect((await fetch(new URL("api/preferences", app.url), { method: "POST", headers: { Origin: "dsh-app://app", "Content-Type": "application/json" }, body: '{}' })).status).toBe(403);
});

it.each([
  ["pending", "awaiting_announcement"],
  ["completed", "no_updates"],
  ["completed", "no_matches"],
] as const)("keeps structured %s/%s in the job response", async (kind, outcome) => {
  const { get } = await setup(async (_args, io) => {
    io.onRunResult?.({ date: "2026-10-02", kind, outcome });
    return 0;
  });
  await get("api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind: "daily", date: "2026-10-02" }) });
  await vi.waitFor(async () => expect((await (await get("api/runs/current")).json()).run).toMatchObject({ status: kind, outcome, exitCode: 0 }));
});

it("exposes durable generation metadata without changing the Markdown or inferring missing time",async()=>{
 const {get,vaultRoot}=await setup();
 const source='# Report\n\nContent\n\n<!-- arxiv-daily:generation-metrics -->\n> [!info]- Generation metrics\n> - LLM calls: 1 logical, 1 HTTP attempt\n> - LLM duration: 500 ms\n> - Provider token usage: 20 input / 10 output / 30 total\n';
 const path='research/daily/2026-10-01.md';
 await writeFile(join(vaultRoot,path),source);
 const doc=await(await get(`api/document?path=${encodeURIComponent(path)}`)).json();
 expect(doc.generationMetrics).toMatchObject({totalTokens:30,elapsedMs:500});
 expect(doc.generationMetrics.generatedAt).toBeUndefined();
 expect(doc.html).not.toContain('Generation metrics');
 expect(await (await get(`api/raw?path=${encodeURIComponent(path)}`)).text()).toBe(source);
});

it("exposes the disconnected personal-library view without running processing", async () => {
  const run = vi.fn<NonNullable<WorkbenchOptions["run"]>>();
  const fixture = await setup(run);
  await writeFile(fixture.config.configPath, `schema_version = 1\nvault_root = ${JSON.stringify(fixture.vaultRoot)}\n`);
  const app = await startWorkbench({ config: await loadCliConfig({ configPath: fixture.config.configPath }), run });
  cleanup.push(app.close);
  const get = (path: string, init?: RequestInit) => fetch(new URL(path, app.url), init);
  const response = await get("api/library");
  expect(response.status).toBe(200);
  expect(await response.json()).toMatchObject({ connected: false, papers: [], total: 0 });
  expect((await get("api/library/pdf?key=../../config.toml")).status).not.toBe(200);
  expect((await get("api/library/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: "x", mode: "shell" }) })).status).toBe(400);
  expect(run).not.toHaveBeenCalled();
});
