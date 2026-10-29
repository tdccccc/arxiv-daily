import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test, { after, before } from "node:test";
import { loadQualityEval } from "../quality-eval/load.mjs";

const directories = [];
let qe;

before(async () => { qe = await loadQualityEval(); });
after(async () => { await Promise.all(directories.map(directory => rm(directory, { recursive: true, force: true }))); });

const temp = async prefix => {
  const directory = await mkdtemp(path.join(tmpdir(), `quality-eval-${prefix}-`));
  directories.push(directory);
  return directory;
};
const repoFile = relative => path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..", relative);

const NOW = new Date("2026-10-28T12:00:00Z");
const instantCoordinator = () => {
  let now = 0;
  return new qe.ArxivRequestCoordinator({
    monotonicNow: () => now, wallNow: () => 1_800_000_000_000 + now, sleep: async ms => { now += ms; },
  });
};

const HTML_PAGE = (version = "v2") => `<html><head><base href="/html/2610.00001${version}/"></head><body>
<div class="ltx_abstract">We measure the cluster mass ratio to be 0.85 with a careful calibration of the weak lensing signal.</div>
<h2>Introduction</h2><p>The introduction motivates the problem with enough natural language to be treated as paper text.</p>
<h2>Methods</h2><p>We stack shear profiles and calibrate the selection function against simulations with a baseline.</p>
<h2>Conclusions</h2><p>The final remarks summarize the validated result and the conditions where it applies.</p>
<h2>References</h2><p>[1] paper</p>
</body></html>`;
const TEX = String.raw`
\documentclass{article}
\begin{document}
\begin{abstract}
This paper has source text available even though rendered HTML is unavailable for the downstream summary.
\end{abstract}
\section{Method}
We describe the data, model assumptions, calibration strategy, and validation experiment in detail.
\section{Results}
The results include a measured improvement of 12 percent, uncertainty estimates, and a baseline comparison.
\end{document}
`;
const ABS_PAGE = `<html><body><blockquote class="abstract mathjax"><span class="descriptor">Abstract:</span>Abstract text only, with $x_1$ math and a number 3.5.</blockquote></body></html>`;
const NOT_FOUND = { status: 404, headers: {}, bodyText: "not found" };
const ok = bodyText => ({ status: 200, headers: {}, bodyText });

/** routes: id -> { html?, tex?, abs? } (a response, or a string for 200); anything missing is a 404. */
function fakeHttp(routes) {
  const calls = [];
  return {
    calls,
    request: async request => {
      calls.push(request.url);
      const match = /^https:\/\/arxiv\.org\/(html|e-print|abs)\/(\d{4}\.\d{4,5})(v\d+)?$/.exec(request.url);
      if (!match) return NOT_FOUND;
      const route = routes[match[2]]?.[match[1] === "e-print" ? "tex" : match[1]];
      if (route === undefined) return NOT_FOUND;
      if (match[1] === "e-print" && typeof route === "string") {
        const buffer = Buffer.from(route, "utf8");
        return { status: 200, headers: {}, bodyText: "", bodyBuffer: buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) };
      }
      return typeof route === "string" ? ok(route) : route;
    },
  };
}

const run = async (ids, routes, options = {}, deps = {}) => {
  const sourcesDir = options.sourcesDir ?? path.join(await temp("sources"), "src");
  const http = deps.http ?? fakeHttp(routes);
  const result = await qe.fetchSources(ids, { sourcesDir, yes: true, ...options }, {
    http, requestCoordinator: instantCoordinator(), now: () => NOW, logger: new qe.Logger("error"), ...deps,
  });
  return { result, http, sourcesDir };
};
const readRecord = async (sourcesDir, id) => JSON.parse(await readFile(path.join(sourcesDir, `${id}.json`), "utf8"));
const hashOf = (abstractConclusion, fullSections) => createHash("sha256").update(`${abstractConclusion}\u0000${fullSections ?? ""}`).digest("hex");

test("rendered HTML is cached with the content the daily pipeline would feed the model", async () => {
  const { result, sourcesDir } = await run(["2610.00001"], { "2610.00001": { html: HTML_PAGE() } });
  assert.equal(result.fetched, 1);
  const record = await readRecord(sourcesDir, "2610.00001");

  assert.equal(record.schemaVersion, 1);
  assert.equal(record.id, "2610.00001");
  assert.equal(record.fetchedAt, NOW.toISOString());
  assert.equal(record.sourcePath, "html");
  assert.equal(record.arxivVersion, "v2");
  assert.deepEqual(record.budgets, { sectionCharLimit: qe.DEFAULT_SETTINGS.advanced.sectionCharLimit, paperCharLimit: qe.DEFAULT_SETTINGS.advanced.paperCharLimit });
  assert.match(record.content.abstractConclusion, /^## Abstract\nWe measure the cluster mass ratio/);
  assert.match(record.content.abstract, /^We measure the cluster mass ratio/);
  assert.match(record.content.fullSections, /## Methods\nWe stack shear profiles/);
  assert.ok(!record.content.fullSections.includes("References"));
  assert.equal(record.content.fullTextFailure, null);
  assert.equal(record.contentHash, hashOf(record.content.abstractConclusion, record.content.fullSections));
});

test("the LaTeX source is used when rendered HTML does not exist", async () => {
  const { sourcesDir, http } = await run(["2610.00001"], { "2610.00001": { tex: TEX } });
  const record = await readRecord(sourcesDir, "2610.00001");
  assert.equal(record.sourcePath, "latex");
  assert.match(record.content.fullSections, /## Method\nWe describe the data/);
  assert.match(record.content.abstractConclusion, /^## Abstract/);
  assert.ok(!http.calls.some(url => url.includes("/abs/")), "no abs request once the source worked");
});

test("a paper with only an abs page is cached without full sections", async () => {
  const { result, sourcesDir } = await run(["2610.00001"], { "2610.00001": { abs: ABS_PAGE } });
  assert.equal(result.fetched, 1);
  const record = await readRecord(sourcesDir, "2610.00001");
  assert.equal(record.sourcePath, "abs");
  assert.equal(record.content.fullSections, null);
  assert.equal(record.content.abstractConclusion, "## Abstract\nAbstract text only, with $x_1$ math and a number 3.5.");
  assert.match(record.content.fullTextFailure, /HTTP 404/);
  assert.equal(record.contentHash, hashOf(record.content.abstractConclusion, null));
});

test("an id that is already cached is skipped unless --refresh is given", async () => {
  const routes = { "2610.00001": { html: HTML_PAGE() }, "2610.00002": { html: HTML_PAGE() } };
  const first = await run(["2610.00001"], routes);
  const second = await run(["2610.00001", "2610.00002"], routes, { sourcesDir: first.sourcesDir });
  assert.equal(second.result.fetched, 1);
  assert.equal(second.result.skipped, 1);
  assert.ok(second.http.calls.every(url => !url.includes("2610.00001")), second.http.calls.join(","));

  const refreshed = await run(["2610.00001"], routes, { sourcesDir: first.sourcesDir, refresh: true });
  assert.equal(refreshed.result.fetched, 1);
  assert.ok(refreshed.http.calls.some(url => url.includes("2610.00001")));
});

test("a paper that cannot be fetched is listed in failures.json and does not stop the others", async () => {
  const { result, sourcesDir } = await run(["2610.00001", "2610.00002", "2610.00003"], { "2610.00001": { html: HTML_PAGE() }, "2610.00003": { html: HTML_PAGE() } });
  assert.equal(result.fetched, 2);
  assert.equal(result.failed, 1);
  assert.equal(existsSync(path.join(sourcesDir, "2610.00002.json")), false);
  const failures = JSON.parse(await readFile(path.join(sourcesDir, "failures.json"), "utf8"));
  assert.equal(failures.failures.length, 1);
  assert.equal(failures.failures[0].id, "2610.00002");
  assert.match(failures.failures[0].reason, /HTTP 404/);

  // A later run that succeeds removes it from the list.
  const retry = await run(["2610.00002"], { "2610.00002": { html: HTML_PAGE() } }, { sourcesDir });
  assert.equal(retry.result.fetched, 1);
  const after = JSON.parse(await readFile(path.join(sourcesDir, "failures.json"), "utf8"));
  assert.deepEqual(after.failures, []);
});

test("an invalid id is a failure that costs no request", async () => {
  const { result, http } = await run(["hep-th/9901001"], {});
  assert.equal(result.failed, 1);
  assert.equal(http.calls.length, 0);
});

test("classifyContent tells a usable result from a failure worth retrying", () => {
  const good = { abstractConclusion: "## Abstract\nx", fullSections: "## A\nbody", fullTextSource: "arxiv-html" };
  assert.deepEqual(qe.classifyContent(good), { kind: "ok", sourcePath: "html" });
  assert.deepEqual(qe.classifyContent({ ...good, fullTextSource: "arxiv-source" }), { kind: "ok", sourcePath: "latex" });
  const absOnly = { abstractConclusion: "## Abstract\nx", fullSections: null, fullTextFailure: "rendered HTML unavailable for 2610.00001 (HTTP 404); no arXiv source for 2610.00001 (HTTP 404)" };
  assert.deepEqual(qe.classifyContent(absOnly), { kind: "ok", sourcePath: "abs" });
  const placeholder = { abstractConclusion: "[获取失败] arXiv ID: 2610.00001", fullSections: null, fullTextFailure: "abs/cache failed for 2610.00001: HTTP 404" };
  assert.equal(qe.classifyContent(placeholder).kind, "failed");
  const transient = { ...absOnly, fullTextFailure: "rendered HTML/cache failed for 2610.00001: network down; no arXiv source for 2610.00001 (HTTP 404)" };
  const classified = qe.classifyContent(transient);
  assert.equal(classified.kind, "failed");
  assert.match(classified.reason, /network down/);
  assert.equal(qe.classifyContent({ ...placeholder, fullTextFailure: "arXiv requested a later retry; retry at 2026-10-28T12:30:00.000Z (30 min remaining)" }).rateLimited, true);
});

test("a rate-limit cooldown stops the run and says how many ids remain", async () => {
  const seen = [];
  const fetchContent = async id => {
    seen.push(id);
    if (id === "2610.00002") throw new (class extends Error { constructor() { super("arXiv requested a later retry; retry at x"); this.name = "ArxivRetryDeferredError"; this.retryAt = new Date(NOW.getTime() + 60_000); this.remainingMs = 60_000; } })();
    return { abstractConclusion: "## Abstract\nx", fullSections: "## A\nbody", fullTextSource: "arxiv-html" };
  };
  const { result, sourcesDir } = await run(["2610.00001", "2610.00002", "2610.00003", "2610.00004"], {}, {}, { fetchContent });
  assert.deepEqual(seen, ["2610.00001", "2610.00002"]);
  assert.equal(result.fetched, 1);
  assert.equal(result.stopped, "rate-limit");
  assert.equal(result.remaining, 2);
  assert.ok(existsSync(path.join(sourcesDir, "2610.00001.json")));
  const failures = JSON.parse(await readFile(path.join(sourcesDir, "failures.json"), "utf8"));
  assert.deepEqual(failures.failures.map(failure => failure.id), ["2610.00002"]);
});

test("without --yes nothing is requested or written, and the plan counts cached and pending ids", async () => {
  const routes = { "2610.00001": { html: HTML_PAGE() } };
  const warm = await run(["2610.00001"], routes);
  const sourcesDir = warm.sourcesDir;
  const before = await readdir(sourcesDir);

  const dry = await run(["2610.00001", "2610.00002", "2610.00003"], routes, { sourcesDir, yes: false });
  assert.equal(dry.result.dryRun, true);
  assert.deepEqual({ total: dry.result.plan.total, cached: dry.result.plan.cached, toFetch: dry.result.plan.toFetch }, { total: 3, cached: 1, toFetch: 2 });
  assert.equal(dry.http.calls.length, 0);
  assert.deepEqual(await readdir(sourcesDir), before);

  const fresh = path.join(await temp("fresh"), "never-created");
  const none = await run(["2610.00001"], routes, { sourcesDir: fresh, yes: false });
  assert.equal(none.http.calls.length, 0);
  assert.equal(existsSync(fresh), false);
});

test("--limit caps how many ids are fetched in one run", async () => {
  const routes = Object.fromEntries(["2610.00001", "2610.00002", "2610.00003"].map(id => [id, { html: HTML_PAGE() }]));
  const { result, sourcesDir } = await run(["2610.00001", "2610.00002", "2610.00003"], routes, { limit: 2 });
  assert.equal(result.fetched, 2);
  assert.equal(result.remaining, 1);
  assert.deepEqual((await readdir(sourcesDir)).filter(name => /^\d/.test(name)), ["2610.00001.json", "2610.00002.json"]);
});

test("the budgets of the settings in effect are used and recorded", async () => {
  const { sourcesDir } = await run(["2610.00001"], { "2610.00001": { html: HTML_PAGE() } }, { budgets: { sectionCharLimit: 40, paperCharLimit: 90 } });
  const record = await readRecord(sourcesDir, "2610.00001");
  assert.deepEqual(record.budgets, { sectionCharLimit: 40, paperCharLimit: 90 });
  assert.ok(record.content.fullSections.length <= 90 + 40);
});

test("rebuilding the daily input gives exactly what the summarizer sends the model", async () => {
  const { sourcesDir } = await run(["2610.00001"], { "2610.00001": { html: HTML_PAGE() } });
  const record = await readRecord(sourcesDir, "2610.00001");
  const rebuilt = await qe.rebuildDailyInput(record, { title: "A title", authors: "A. Author et al." });

  const sent = [];
  const llm = { call: async messages => { sent.push(messages); throw new Error("stop"); } };
  await assert.rejects(qe.summarizeDailyPaper({
    id: "2610.00001", title: "A title", authors: "A. Author et al.",
    abstractConclusion: record.content.abstractConclusion, fullSections: record.content.fullSections,
  }, { llm }), /stop/);
  assert.deepEqual(rebuilt.messages, sent[0]);
  assert.equal(rebuilt.options.temperature, 0);
  assert.equal(rebuilt.messages[0].role, "system");
  assert.ok(rebuilt.messages[1].content.includes("A title"));
  assert.ok(rebuilt.messages[1].content.includes(record.content.fullSections));
  assert.equal(await qe.rebuildDailyInput({ ...record, content: { ...record.content, abstractConclusion: "[获取失败] arXiv ID: 2610.00001", fullSections: null } }, { title: "t", authors: "a" }), null);
});

test("rebuilding the detail input gives exactly what the detail summarizer sends the model", async () => {
  const { sourcesDir } = await run(["2610.00001"], { "2610.00001": { html: HTML_PAGE() } });
  const record = await readRecord(sourcesDir, "2610.00001");
  const topics = [{ tag: "galaxy-cluster", name: "Galaxy Cluster" }];
  const rebuilt = await qe.rebuildDetailInput(record, { title: "A title", authors: "A. Author et al.", topic: "galaxy-cluster", topics });

  const sent = [];
  const llm = { call: async messages => { sent.push(messages); return "# A title\n"; } };
  await qe.summarizePaperDetail({
    id: "2610.00001", title: "A title", authors: "A. Author et al.", category: "galaxy-cluster", abstractConclusion: "", fullSections: record.content.fullSections,
  }, { llm, arxivSettings: { ...qe.DEFAULT_SETTINGS.arxiv, topics }, logger: new qe.Logger("error"), advanced: qe.DEFAULT_SETTINGS.advanced });
  assert.deepEqual(rebuilt.messages, sent[0]);
  assert.equal(rebuilt.options.temperature, undefined);
  assert.ok(rebuilt.messages[0].content.includes("Galaxy Cluster"));
  assert.ok(rebuilt.messages[1].content.includes("以下是论文各章节内容"));
  // Abs-only content has no full text, and a detail note cannot be written from it.
  assert.equal(await qe.rebuildDetailInput({ ...record, content: { ...record.content, fullSections: null } }, { title: "t", authors: "a", topic: "x" }), null);
});

async function vaultWith(ids, noteIds, fallbackIds = []) {
  const vault = await temp("vault");
  const daily = path.join(vault, "arxiv-daily", "daily");
  const papers = path.join(vault, "arxiv-daily", "papers");
  await mkdir(daily, { recursive: true });
  await mkdir(papers, { recursive: true });
  const fields = ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"].map(label => `- **${label}**: ${label}的内容。`);
  const block = id => [`### Paper ${id}`, "- **作者**: A. Author et al.", `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`, ...fields, ""];
  const fallback = id => [`### Fallback ${id}`, "> **自动摘要不可用。**", `<!-- arxiv-daily-fallback:${id} -->`, "- **作者**: C. Author", `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`, "- **原始摘要**: abstract.", ""];
  await writeFile(path.join(daily, "2026-10-20.md"), ["---", "date: 2026-10-20", "---", "", "# t", "", "## Photo-z", "", ...ids.flatMap(block), ...fallbackIds.flatMap(fallback)].join("\n"));
  await writeFile(path.join(daily, "2026-10-21.md"), ["---", "date: 2026-10-21", "---", "", "# t", "", "## Photo-z", "", ...block(ids[0])].join("\n"));
  for (const id of noteIds) {
    await writeFile(path.join(papers, `${id}.md`), ["---", `title: "N ${id}"`, `arxiv_id: "${id}"`, "primary_topic: photo-z", "published: 2026-10-20", "---", "", `# N ${id}`, "", "## 研究问题", "", "内容。", ""].join("\n"));
  }
  return vault;
}

test("the command collects the distinct ids of the daily entries and the notes, and fallbacks only on request", async () => {
  const vault = await vaultWith(["2610.00001", "2610.00002"], ["2610.00002", "2607.00009"], ["2609.00005"]);
  const routes = Object.fromEntries(["2610.00001", "2610.00002", "2607.00009", "2609.00005"].map(id => [id, { html: HTML_PAGE() }]));
  const sourcesDir = path.join(await temp("out"), "src");
  const plan = await qe.runSummarySources({ vaultRoot: vault, sourcesDir }, { http: fakeHttp(routes), requestCoordinator: instantCoordinator() });
  assert.equal(plan.dryRun, true);
  assert.deepEqual({ total: plan.plan.total, cached: plan.plan.cached, toFetch: plan.plan.toFetch, invalid: plan.plan.invalid, limit: plan.plan.limit }, { total: 3, cached: 0, toFetch: 3, invalid: 0, limit: null });
  assert.deepEqual(plan.plan.budgets, { sectionCharLimit: qe.DEFAULT_SETTINGS.advanced.sectionCharLimit, paperCharLimit: qe.DEFAULT_SETTINGS.advanced.paperCharLimit });

  const withFallbacks = await qe.runSummarySources({ vaultRoot: vault, sourcesDir, fallbacks: true }, { http: fakeHttp(routes), requestCoordinator: instantCoordinator() });
  assert.equal(withFallbacks.plan.total, 4);

  const done = await qe.runSummarySources({ vaultRoot: vault, sourcesDir, yes: true }, { http: fakeHttp(routes), requestCoordinator: instantCoordinator(), now: () => NOW, logger: new qe.Logger("error") });
  assert.equal(done.fetched, 3);
  assert.deepEqual((await readdir(sourcesDir)).filter(name => /^\d/.test(name)).sort(), ["2607.00009.json", "2610.00001.json", "2610.00002.json"]);
});

test("the date range limits which ids are collected", async () => {
  const vault = await vaultWith(["2610.00001", "2610.00002"], []);
  const sourcesDir = path.join(await temp("out"), "src");
  const dry = await qe.runSummarySources({ vaultRoot: vault, sourcesDir, from: "2026-10-21", to: "2026-10-21" }, { http: fakeHttp({}), requestCoordinator: instantCoordinator() });
  assert.equal(dry.plan.total, 1);
});

test("command-line arguments are validated", () => {
  assert.deepEqual(
    qe.parseSummarySourcesArgs(["--vault", "/v", "--daily-dir", "d", "--papers-dir", "p", "--from", "2026-10-01", "--to", "2026-10-31", "--sources", "/s", "--refresh", "--limit", "5", "--yes", "--fallbacks"]),
    { vaultRoot: "/v", dailyDir: "d", papersDir: "p", from: "2026-10-01", to: "2026-10-31", sourcesDir: "/s", refresh: true, limit: 5, yes: true, fallbacks: true },
  );
  assert.deepEqual(qe.parseSummarySourcesArgs(["--sources", "/s", "--config", "/c.toml"]), { sourcesDir: "/s", configPath: "/c.toml" });
  assert.equal(qe.parseSummarySourcesArgs(["--help"]).help, true);
  assert.throws(() => qe.parseSummarySourcesArgs(["--vault", "/v"]), /--sources/);
  assert.throws(() => qe.parseSummarySourcesArgs(["--sources", "/s", "--limit", "0"]), /--limit/);
  assert.throws(() => qe.parseSummarySourcesArgs(["--sources", "/s", "--limit", "x"]), /--limit/);
  assert.throws(() => qe.parseSummarySourcesArgs(["--sources", "/s", "--bogus"]), /unknown option.*--bogus/i);
  assert.throws(() => qe.parseSummarySourcesArgs(["--sources", "/s", "--from", "2026-13-01"]), /--from must be a date/);
  assert.throws(() => qe.parseSummarySourcesArgs(["--sources", "/s", "--from", "2026-10-02", "--to", "2026-10-01"]), /after/);
});

test("the script prints usage on bad arguments and, without --yes, only a plan", async () => {
  const { spawnSync } = await import("node:child_process");
  const script = repoFile("scripts/quality-eval/summary-sources.mjs");
  const bad = spawnSync(process.execPath, [script, "--vault", "/nope"], { encoding: "utf8" });
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /--sources/);
  assert.match(bad.stderr, /Usage: node scripts\/quality-eval\/summary-sources\.mjs/);
  assert.equal(spawnSync(process.execPath, [script, "--help"], { encoding: "utf8" }).status, 0);

  const vault = await vaultWith(["2610.00001"], []);
  const sourcesDir = path.join(await temp("cli"), "src");
  const plan = spawnSync(process.execPath, [script, "--vault", vault, "--sources", sourcesDir], { encoding: "utf8" });
  assert.equal(plan.status, 0, plan.stderr);
  assert.match(plan.stdout, /1 papers? would be fetched from arXiv/);
  assert.match(plan.stdout, /--yes/);
  assert.equal(existsSync(sourcesDir), false);
});
