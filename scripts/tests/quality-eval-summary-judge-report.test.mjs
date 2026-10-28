import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
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
const NOW = new Date("2026-10-29T09:00:00Z");

const verdicts = (accuracy = "ok", completeness = "ok", clarity = "ok") => ({ accuracy, completeness, clarity });
const issue = (overrides = {}) => ({
  dimension: "accuracy", severity: "major", location: "mainResult", summaryQuote: "样本为120个星团",
  sourceQuote: "using 130 clusters", explanation: "样本数与原文不符。", ...overrides,
});
const record = (kind, id, date, overrides = {}) => ({
  schemaVersion: 1, status: "judged", invalid: false,
  item: { kind, id, title: `Title of ${id}`, date, topic: kind === "daily" ? "Photo-z" : "galaxy-cluster", ...(kind === "note" ? { version: "current" } : {}), source: { sourcePath: "html", arxivVersion: "v1", fetchedAt: "2026-10-28T10:00:00.000Z", contentHash: "h" } },
  judge: { provider: "openai", model: "judge-model", thinkingMode: false, reasoningEffort: "medium" },
  judgePromptHash: "p".repeat(64),
  attempts: [{ raw: "{}", reasons: null, usage: null, elapsedMs: 3 }],
  judgment: { verdicts: verdicts(), issues: [] },
  startedAt: NOW.toISOString(), finishedAt: NOW.toISOString(), elapsedMs: 3, ...overrides,
});
const withIssues = (issues, extra = {}) => {
  const rank = { ok: 0, minor: 1, major: 2 };
  const v = verdicts();
  for (const item of issues) if (rank[item.severity] > rank[v[item.dimension]]) v[item.dimension] = item.severity;
  return { judgment: { verdicts: v, issues }, ...extra };
};

const RUN = {
  schemaVersion: 1, startedAt: "2026-10-29T08:00:00.000Z", finishedAt: "2026-10-29T08:30:00.000Z",
  args: { daily: 4, notes: 1, ids: null, from: null, to: null, includeOlderNotes: false, rejudge: false, sourcesDir: "/sources" },
  seed: "seed-1", requested: { daily: 4, notes: 1 }, allocation: { "2026-08": 2, "2026-09": 2 },
  judge: { provider: "openai", model: "judge-model", thinkingMode: false, reasoningEffort: "medium" }, generatorModel: "generator-model",
  sample: { daily: ["2608.00001", "2608.00002", "2609.00001", "2609.00002", "2609.00003"], notes: ["2607.00001"] },
  skipped: [{ kind: "daily", id: "2609.00003", reason: "no-source", detail: "no cached source record" }],
  counts: { judged: 3, invalid: 1, errors: 1, reused: 0, skipped: 1, remaining: 0 }, stopped: null,
};

/** daily: 2608.00001 ok first try; 2608.00002 issues after a retry; 2609.00001 invalid; 2609.00002 error. Note 2607.00001 with issues. */
async function fixtureOut(extraRecords = [], run = RUN) {
  const out = await temp("out");
  await mkdir(path.join(out, "judgments"), { recursive: true });
  await writeFile(path.join(out, "run.json"), JSON.stringify(run));
  const records = [
    record("daily", "2608.00001", "2026-08-10"),
    record("daily", "2608.00002", "2026-08-12", {
      ...withIssues([
        issue(),
        issue({ dimension: "completeness", severity: "minor", location: "limitations", summaryQuote: "仅覆盖光学选择的星团", sourceQuote: "X-ray selected samples are not tested", explanation: "漏了 X 射线样本。" }),
        issue({ dimension: "clarity", severity: "minor", location: "keyMethod", summaryQuote: "与模拟标定", sourceQuote: null, explanation: "没说明哪类模拟。" }),
      ]),
      attempts: [{ raw: "坏", reasons: ["issues[0].location 的值 \"abstract\" 不是允许的位置，只能是 \"coreProblem\"、\"keyMethod\"", "verdicts.accuracy 是 \"ok\"，但该维度最严重的问题是 \"major\"，判定必须等于最严重的问题。"], usage: null, elapsedMs: 1 }, { raw: "好", reasons: null, usage: null, elapsedMs: 1 }],
    }),
    record("daily", "2609.00001", "2026-09-03", {
      status: "invalid", invalid: true, judgment: null,
      attempts: [
        { raw: "x", reasons: ["issues[2].location 的值 \"abstract\" 不是允许的位置，只能是 \"coreProblem\""], usage: null, elapsedMs: 1 },
        { raw: "y", reasons: ["issues[0].summaryQuote 不是“mainResult”里的原句：\"foo\""], usage: null, elapsedMs: 1 },
      ],
    }),
    record("daily", "2609.00002", "2026-09-04", { status: "error", judgment: null, error: "HTTP 500", attempts: [] }),
    record("note", "2607.00001", "2026-07-10", withIssues([
      issue({ location: "核心结果", summaryQuote: "质量比为 0.85", severity: "minor", sourceQuote: null, explanation: "原文未提及质量比 0.85。" }),
      issue({ dimension: "completeness", severity: "major", location: "适用边界", summaryQuote: "原文未讨论", sourceQuote: "Only optically selected clusters are included", explanation: "原文其实讨论了边界。" }),
    ])),
    ...extraRecords,
  ];
  for (const item of records) await writeFile(path.join(out, "judgments", `${item.item.kind}-${item.item.id}.json`), JSON.stringify(item));
  return out;
}
const readReport = async out => ({ md: await readFile(path.join(out, "report.md"), "utf8"), json: JSON.parse(await readFile(path.join(out, "judgments.json"), "utf8")) });
const section = (md, heading) => { const start = md.indexOf(`\n## ${heading}`); assert.ok(start >= 0, heading); const end = md.indexOf("\n## ", start + 1); return md.slice(start, end < 0 ? undefined : end); };

test("the report describes what was judged, by whom and how", async () => {
  const out = await fixtureOut();
  const result = await qe.writeJudgeReport(out, { now: () => NOW });
  assert.deepEqual(result.files.map(file => path.basename(file)), ["report.md", "judgments.json"]);
  const { md, json } = await readReport(out);
  const subject = section(md, "评审对象");
  for (const text of ["judge-model", "openai", "generator-model", "seed-1", "2026-10-29T08:00:00.000Z", "2026-08", "no-source", "2609.00003", "medium"]) assert.ok(subject.includes(text), text);
  assert.match(subject, /重新取回/);
  assert.match(subject, /当前版本的提示词|当前的生成提示词|当前提示词/);
  assert.equal(json.schemaVersion, 1);
  assert.equal(json.generatedAt, NOW.toISOString());
  assert.equal(json.records.length, 5);
  assert.deepEqual(json.run.sample.notes, ["2607.00001"]);
  assert.ok(!JSON.stringify(json).includes("apiKey"));
});

test("verdict distributions count judged items per dimension, for daily entries, notes and months", async () => {
  const out = await fixtureOut();
  await qe.writeJudgeReport(out, { now: () => NOW });
  const { md, json } = await readReport(out);
  const daily = json.aggregate.verdicts.daily;
  assert.equal(daily.judged, 2);
  assert.deepEqual(daily.accuracy, { ok: 1, minor: 0, major: 1 });
  assert.deepEqual(daily.completeness, { ok: 1, minor: 1, major: 0 });
  assert.deepEqual(daily.clarity, { ok: 1, minor: 1, major: 0 });
  const note = json.aggregate.verdicts.note;
  assert.equal(note.judged, 1);
  assert.deepEqual(note.accuracy, { ok: 0, minor: 1, major: 0 });
  assert.deepEqual(note.completeness, { ok: 0, minor: 0, major: 1 });
  assert.deepEqual(json.aggregate.verdicts.dailyByMonth["2026-08"].accuracy, { ok: 1, minor: 0, major: 1 });
  assert.equal(json.aggregate.verdicts.dailyByMonth["2026-09"].judged, 0);

  const text = section(md, "判定分布");
  assert.match(text, /\| 准确 \| 1（50\.0%） \| 0（0\.0%） \| 1（50\.0%） \|/);
  assert.match(text, /\| 完整 \| 1（50\.0%） \| 1（50\.0%） \| 0（0\.0%） \|/);
  assert.ok(text.includes("2026-08"));
  assert.ok(text.includes("日报条目") && text.includes("详细笔记"));
});

test("issues are counted by dimension and location, major apart from minor", async () => {
  const out = await fixtureOut();
  await qe.writeJudgeReport(out, { now: () => NOW });
  const { md, json } = await readReport(out);
  assert.deepEqual(json.aggregate.issues.daily.mainResult.accuracy, { major: 1, minor: 0 });
  assert.deepEqual(json.aggregate.issues.daily.limitations.completeness, { major: 0, minor: 1 });
  assert.deepEqual(json.aggregate.issues.note["核心结果"].accuracy, { major: 0, minor: 1 });
  assert.deepEqual(json.aggregate.issues.note["适用边界"].completeness, { major: 1, minor: 0 });
  const text = section(md, "问题分布");
  for (const location of ["mainResult", "limitations", "keyMethod", "核心结果", "适用边界"]) assert.ok(text.includes(location), location);
});

test("the issue list puts major first, shows the evidence, and is capped per dimension", async () => {
  const many = Array.from({ length: 34 }, (_, index) => record("daily", `2608.${String(index + 10).padStart(5, "0")}`, "2026-08-20", withIssues([
    issue({ severity: "minor", summaryQuote: `第 ${index} 句`, sourceQuote: null, explanation: `原文未提及第 ${index} 句。` }),
  ])));
  const out = await fixtureOut(many);
  await qe.writeJudgeReport(out, { now: () => NOW });
  const { md, json } = await readReport(out);
  const list = section(md, "问题清单");
  assert.match(list, /共 35 条，仅列出前 30 条，完整清单见 judgments\.json/);
  const accuracyTable = list.slice(list.indexOf("### 日报条目：准确"), list.indexOf("### 日报条目：完整"));
  assert.equal(accuracyTable.split("\n").filter(line => line.startsWith("| 2026")).length, 30);
  // Major first: the one major accuracy issue is the first row of its table.
  const firstRow = accuracyTable.split("\n").find(line => line.startsWith("| 2026"));
  assert.ok(firstRow.includes("样本数与原文不符") && firstRow.includes("major"), firstRow);
  assert.ok(list.includes("原文未提及"), "a null source quote is shown as such");
  assert.ok(list.includes("using 130 clusters"));
  assert.ok(list.includes("Title of 2608.00002"));
  assert.equal(json.records.length, 5 + 34);
});

test("table cells are escaped", async () => {
  const out = await fixtureOut([record("daily", "2608.00099", "2026-08-21", withIssues([issue({ summaryQuote: "a|b\n\\alpha", sourceQuote: "x|y", explanation: "含 | 竖线" })]))]);
  await qe.writeJudgeReport(out, { now: () => NOW });
  const { md } = await readReport(out);
  const lines = md.split("\n");
  const row = lines.find(line => line.includes("2608.00099") && line.startsWith("|"));
  assert.ok(row.includes("a\\|b"), row);
  const header = lines.slice(0, lines.indexOf(row)).reverse().find(line => line.startsWith("| 日期"));
  const cells = line => line.replace(/\\\|/g, "").split("|").length;
  assert.equal(cells(row), cells(header), row);
});

test("judge quality counts first-try, retried, invalid and errored items and the common reasons", async () => {
  const out = await fixtureOut();
  await qe.writeJudgeReport(out, { now: () => NOW });
  const { md, json } = await readReport(out);
  assert.deepEqual(json.aggregate.quality.counts, { validFirstTry: 2, validAfterRetry: 1, invalid: 1, errors: 1, total: 5 });
  const reasons = json.aggregate.quality.reasons;
  const location = reasons.find(entry => entry.reason.startsWith("issues[].location"));
  assert.equal(location.count, 2);
  assert.ok(reasons.every((entry, index) => index === 0 || reasons[index - 1].count >= entry.count));
  const text = section(md, "评审质量");
  assert.match(text, /第一次就合格[^\n]*2/);
  assert.match(text, /重试后合格[^\n]*1/);
  assert.ok(text.includes("issues[].location"));
});

test("the offline cross-check lists numbers both sides flagged, only the offline check, and only the judge", async () => {
  const out = await fixtureOut();
  const offline = path.join(await temp("offline"), "findings.json");
  await writeFile(offline, JSON.stringify({ grounding: { run: true, daily: { results: [
    { id: "2608.00002", date: "2026-08-12", ungrounded: [
      { where: "mainResult", token: "120", context: "样本为120个星团" },
      { where: "mainResult", token: "0.99", context: "误差 0.99" },
    ] },
    { id: "2608.00001", date: "2026-08-10", ungrounded: [{ where: "coreProblem", token: "7.5", context: "约 7.5" }] },
  ] }, notes: { results: [{ id: "2607.00001", date: "2026-07-10", ungrounded: [{ where: "核心结果", token: "0.85", context: "质量比为 0.85" }] }] } } }));
  await qe.writeJudgeReport(out, { now: () => NOW, offlinePath: offline });
  const { md, json } = await readReport(out);
  const cross = json.crossCheck;
  assert.deepEqual(cross.counts, { both: 2, onlyOffline: 2, onlyJudge: 0, judgeWithoutNumber: 0 });
  assert.deepEqual(cross.both.map(entry => [entry.id, entry.token]).sort(), [["2607.00001", "0.85"], ["2608.00002", "120"]]);
  assert.deepEqual(cross.onlyOffline.map(entry => [entry.id, entry.token]).sort(), [["2608.00001", "7.5"], ["2608.00002", "0.99"]]);
  const text = section(md, "与离线检查对照");
  assert.ok(text.includes("120") && text.includes("0.99") && text.includes("7.5"));
  assert.match(text, /两边都指出[^\n]*2/);
  assert.match(text, /只有离线检查[^\n]*2/);
});

test("an accuracy issue about a number the offline check found grounded is listed as only the judge", async () => {
  const out = await fixtureOut();
  const offline = path.join(await temp("offline"), "findings.json");
  await writeFile(offline, JSON.stringify({ grounding: { run: true, daily: { results: [{ id: "2608.00002", date: "2026-08-12", ungrounded: [] }] }, notes: { results: [] } } }));
  await qe.writeJudgeReport(out, { now: () => NOW, offlinePath: offline });
  const { json } = await readReport(out);
  assert.equal(json.crossCheck.counts.onlyJudge, 1);
  assert.equal(json.crossCheck.onlyJudge[0].id, "2608.00002");
  assert.equal(json.crossCheck.onlyJudge[0].location, "mainResult");
});

test("without --offline there is no cross-check section; an offline file without number data is refused", async () => {
  const out = await fixtureOut();
  await qe.writeJudgeReport(out, { now: () => NOW });
  const { md, json } = await readReport(out);
  assert.ok(!md.includes("与离线检查对照"));
  assert.equal(json.crossCheck, null);
  const bad = path.join(await temp("offline"), "findings.json");
  await writeFile(bad, JSON.stringify({ grounding: { run: false } }));
  await assert.rejects(qe.writeJudgeReport(out, { now: () => NOW, offlinePath: bad }), /grounding|--sources/);
});

test("--report-only rebuilds the report without a judge config, a model or any request", async () => {
  const out = await fixtureOut();
  const result = await qe.runJudgeReportOnly({ outDir: out }, { now: () => NOW });
  assert.equal(result.files.length, 2);
  assert.ok(existsSync(path.join(out, "report.md")));
  await assert.rejects(qe.runJudgeReportOnly({ outDir: await temp("empty") }, {}), /no judgments/i);
  const parsed = qe.parseSummaryJudgeArgs(["--report-only", "--out", "/o", "--offline", "/f.json"]);
  assert.deepEqual(parsed, { reportOnly: true, outDir: "/o", offlinePath: "/f.json" });
  assert.throws(() => qe.parseSummaryJudgeArgs(["--report-only"]), /--out/);
  assert.throws(() => qe.parseSummaryJudgeArgs(["--out", "/o", "--sources", "/s", "--judge-config", "/j", "--offline", "/f.json", "--bogus"]), /unknown option/i);
});

test("a judge run writes the report itself; a preview does not", async () => {
  const vault = await temp("vault");
  const sources = await temp("sources");
  await mkdir(path.join(vault, "arxiv-daily", "daily"), { recursive: true });
  await mkdir(path.join(vault, "arxiv-daily", "papers"), { recursive: true });
  const labels = ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"];
  await writeFile(path.join(vault, "arxiv-daily", "daily", "2026-10-15.md"), ["---", "date: 2026-10-15", "---", "", "# t", "", "## Photo-z", "", "### Paper", "- **作者**: A. Author et al.", "- **arXiv**: [2610.00001](https://arxiv.org/abs/2610.00001)", ...labels.map(label => `- **${label}**: 内容。`), ""].join("\n"));
  await writeFile(path.join(sources, "2610.00001.json"), JSON.stringify({ schemaVersion: 1, id: "2610.00001", fetchedAt: "2026-10-28T10:00:00.000Z", sourcePath: "html", arxivVersion: "v1", budgets: { sectionCharLimit: 16000, paperCharLimit: 100000 }, content: { abstract: "a", abstractConclusion: "## Abstract\na", fullSections: "## Methods\nWe use 120 clusters.", fullTextFailure: null }, contentHash: "h" }));
  const judgeConfig = path.join(await temp("jc"), "judge.toml");
  await writeFile(judgeConfig, '[llm]\napi_key = "k-1"\nbase_url = "https://judge.example.invalid/v1"\nmodel = "judge-model"\nprovider = "openai"\n');
  const outDir = path.join(await temp("out"), "judge");
  const options = { judgeConfigPath: judgeConfig, vaultRoot: vault, sourcesDir: sources, outDir, daily: 1, notes: 0 };
  const llm = { calls: 0, call: async () => { llm.calls += 1; return JSON.stringify({ verdicts: verdicts(), issues: [] }); } };
  const deps = { llm, now: () => NOW, log: () => undefined, loadGeneratorConfig: async () => ({ model: "generator-model", topics: [] }) };

  await qe.runSummaryJudge(options, deps);
  assert.equal(existsSync(outDir), false);
  await qe.runSummaryJudge({ ...options, yes: true }, deps);
  assert.equal(llm.calls, 1);
  const { md } = await readReport(outDir);
  assert.ok(md.includes("judge-model"));
  assert.ok(!md.includes("k-1") && !md.includes("judge.example.invalid"));
  const { json } = await readReport(outDir);
  assert.equal(json.aggregate.verdicts.daily.judged, 1);
});
