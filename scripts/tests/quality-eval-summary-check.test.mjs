import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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

const frontmatter = date => `---\ndate: ${date}\ntags: [arxiv, daily]\n---\n\n# arXiv astro-ph.CO 每日追踪 ${date}\n`;
const header = (id, title) => [`### ${title}`, "- **作者**: A. Author et al.", `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`];
const fields = (overrides = {}) => {
  const base = {
    研究问题: "星团质量标定能否摆脱对静力学平衡假设的依赖。",
    方法设计: "以弱透镜剪切叠加测得质量，并与 $M_{500}$ 的X射线估计比较。",
    核心结果: "字".repeat(100),
    研究价值: "该比值直接决定宇宙学约束中使用的质量偏差先验取值范围。",
    适用边界: "仅覆盖光学选择的星团。",
  };
  return Object.entries({ ...base, ...overrides }).map(([label, value]) => `- **${label}**: ${value}`);
};

const DAILY_09 = [
  frontmatter("2026-09-15"), "", "## Galaxy Cluster", "",
  "### Fallback paper", "> **自动摘要不可用。** 请直接阅读 [arXiv 原文](https://arxiv.org/abs/2609.00001)。",
  "<!-- arxiv-daily-fallback:2609.00001 -->", "- **作者**: C. Author et al.", "- **arXiv**: [2609.00001](https://arxiv.org/abs/2609.00001)",
  "- **原始摘要**: The original abstract.", "",
  ...header("2609.00002", "Clean paper"), ...fields(), "",
  ...header("2609.00003", "Phrase paper"), ...fields({ 研究价值: "该工作具有重要意义。", 适用边界: "原文未说明" }), "",
].join("\n");
const DAILY_10 = [
  frontmatter("2026-10-14"), "", "## Photo-z", "",
  ...header("2610.00001", "Math paper"), ...fields({ 方法设计: "裸 \\alpha 与 a|b 竖线，路径 C:\\data\\x。", 核心结果: "字".repeat(250) }), "",
  ...header("2610.00002", "Ratio paper"), ...fields({ 方法设计: "质量比 $6:1$，冲量参数可由 $500$ kpc 类比。", 适用边界: "区间 $g \\in (0, 0.15]$ 内恢复。" }), "",
].join("\n");
const LEGACY = [frontmatter("2026-05-05"), "", "## Galaxy Cluster 相关", ...header("2605.00001", "Legacy"), "- **一句话总结**: 一句话。", "- **主要结果**: 结果。", ""].join("\n");

const NOTE_HEADINGS = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "贡献与创新点", "适用边界", "学术价值判断"];
const note = (id, bodies = {}, headings = NOTE_HEADINGS, published = "2026-07-10") => [
  "---", `title: "Note ${id}"`, `arxiv_id: "${id}"`, "primary_topic: galaxy-cluster", `published: ${published}`, "tags: [arxiv, paper]", "---", "",
  `# Note ${id}`, "", `- **arXiv**: https://arxiv.org/abs/${id}`, "",
  ...headings.flatMap(heading => [`## ${heading}`, "", bodies[heading] ?? "正常内容 $z<0.5$。", ""]),
].join("\n");

async function fixtureVault(dailyDir = "arxiv-daily/daily", papersDir = "arxiv-daily/papers") {
  const vault = await temp("vault");
  await mkdir(path.join(vault, dailyDir), { recursive: true });
  await mkdir(path.join(vault, papersDir), { recursive: true });
  await writeFile(path.join(vault, dailyDir, "2026-05-05.md"), LEGACY);
  await writeFile(path.join(vault, dailyDir, "2026-09-15.md"), DAILY_09);
  await writeFile(path.join(vault, dailyDir, "2026-10-14.md"), DAILY_10);
  await writeFile(path.join(vault, papersDir, "2607.00001.md"), note("2607.00001", { 适用边界: "原文未讨论。", 核心结果: "质量比 $6:1$，另有 \\(x\\) 的写法。" }));
  await writeFile(path.join(vault, papersDir, "2607.00002.md"), note("2607.00002", {}, ["研究问题", "方法设计", "关键证据", "主要结论", "适用边界", "阅读价值"]));
  await writeFile(path.join(vault, papersDir, "2603.00001.md"), "---\ntitle: \"Old\"\narxiv: \"2603.00001\"\ndate: 2026-03-05\n---\n\n# Old\n\n## 背景与动机\n旧。\n");
  return vault;
}

const run = async (options = {}) => {
  const vaultRoot = options.vaultRoot ?? await fixtureVault();
  const outDir = options.outDir ?? path.join(await temp("out"), "report");
  const result = await qe.runSummaryCheck({ vaultRoot, outDir, ...options.rest }, { now: () => new Date("2026-10-28T12:00:00Z") });
  return { result, outDir, vaultRoot };
};
const readReport = async outDir => ({
  md: await readFile(path.join(outDir, "report.md"), "utf8"),
  json: JSON.parse(await readFile(path.join(outDir, "findings.json"), "utf8")),
});

test("the command writes a Chinese report and findings.json with the key numbers", async () => {
  const { outDir, result } = await run();
  assert.deepEqual(result.files.map(file => path.basename(file)), ["report.md", "findings.json"]);
  const { md, json } = await readReport(outDir);

  for (const heading of ["## 评估范围", "## 日报条目", "## 详细笔记", "## 问题清单"]) assert.ok(md.includes(heading), heading);
  assert.ok(md.includes("2026-10-28T12:00:00.000Z"));
  assert.match(md, /1 份是旧格式（没有五字段），只计数、不评估/);
  assert.match(md, /五字段条目 4 条.*回退条目 1 条/);
  assert.match(md, /旧格式 1 篇只计数、不评估/);

  assert.equal(json.schemaVersion, 1);
  assert.equal(json.generatedAt, "2026-10-28T12:00:00.000Z");
  assert.deepEqual(json.corpus.counts, {
    reports: 3, evaluableReports: 2, legacyReports: 1, entries: 4, incompleteEntries: 0, fallbacks: 1, notes: 2, legacyNotes: 1,
  });
  assert.equal(json.daily.results.length, 4);
  assert.equal(json.daily.aggregate.fallback.overall.count, 1);
  assert.equal(json.daily.aggregate.fallback.overall.of, 5);
  assert.equal(json.notes.results.length, 2);
  assert.equal(json.notes.aggregate.versions.older, 1);
  assert.ok(json.monthlyMainResult.some(row => row.month === "2026-10"));
});

test("the report names the prompt files by the SHA-256 of what was bundled and says what that means", async () => {
  const { outDir } = await run();
  const { md, json } = await readReport(outDir);
  const hash = async file => createHash("sha256").update(await readFile(repoFile(file), "utf8")).digest("hex");
  assert.equal(json.prompts.dailySummary.sha256, await hash("packages/core/src/prompts/daily-paper-summary.system.md"));
  assert.equal(json.prompts.paperDetail.sha256, await hash("packages/core/src/prompts/paper-detail.system.md"));
  assert.ok(md.includes(json.prompts.dailySummary.sha256.slice(0, 12)));
  assert.ok(md.includes(json.prompts.paperDetail.sha256.slice(0, 12)));
  assert.match(md, /没有记录[^\n]*模型/);
});

test("the report lists defects, shows core false positives apart and keeps them out of the defect counts", async () => {
  const { outDir } = await run();
  const { md, json } = await readReport(outDir);

  // Defects: the bare TeX, the generic phrase, the note's \(x\).
  assert.deepEqual(json.daily.aggregate.codes.math, { "bare-tex": 1 });
  assert.equal(json.daily.aggregate.entriesWith.math.count, 1);
  assert.equal(json.daily.aggregate.coreFalsePositives.entries, 1);
  assert.deepEqual(json.notes.aggregate.codes.math, { "paren-delimiter": 1 });
  assert.equal(json.notes.aggregate.coreFalsePositives.notes, 1);

  const [defects, falsePositives] = md.split("### core 规则误报（渲染正常）");
  assert.ok(falsePositives, "a separate false-positive section");
  assert.ok(defects.includes("2610.00001") && defects.includes("bare-tex"));
  assert.ok(!defects.slice(defects.indexOf("## 问题清单")).includes("2610.00002"));
  assert.ok(falsePositives.includes("2610.00002"));
  assert.ok(falsePositives.includes("2607.00001"));
  assert.ok(md.includes("具有重要意义"));
});

test("snippets are escaped for Markdown tables", async () => {
  const { outDir } = await run();
  const { md } = await readReport(outDir);
  const row = md.split("\n").find(line => line.includes("2610.00001") && line.includes("bare-tex"));
  assert.ok(row, "the bare-tex row");
  // The raw field reads `裸 \alpha 与 a|b 竖线`: the backslash is doubled and the pipe escaped, so the row keeps its 5 cells.
  assert.ok(row.includes("\\\\alpha"), row);
  assert.equal(row.replace(/\\\|/g, "").split("|").length, 7, row);
});

test("the monthly table lists the mainResult median, p90 and share inside 80-180", async () => {
  const { outDir } = await run();
  const { md, json } = await readReport(outDir);
  const sep = json.monthlyMainResult.find(row => row.month === "2026-09");
  assert.deepEqual(sep.inRange, { count: 2, of: 2, rate: 1 });
  const oct = json.monthlyMainResult.find(row => row.month === "2026-10");
  assert.equal(oct.inRange.count, 1);
  assert.match(md, /\| 2026-09 \| 2 \| 100 \| 100 \| 100\.0%/);
  assert.match(md, /\| 2026-10 \| 2 \| 175 \| 250 \| 50\.0%/);
});

test("long lists are capped at 30 per kind and point to findings.json", async () => {
  const vault = await fixtureVault();
  const many = [frontmatter("2026-10-20"), "", "## Photo-z", ""];
  for (let index = 0; index < 35; index += 1) many.push(...header(`2610.${String(index + 10).padStart(5, "0")}`, `Phrase ${index}`), ...fields({ 研究价值: "该工作具有重要意义。" }), "");
  await writeFile(path.join(vault, "arxiv-daily", "daily", "2026-10-20.md"), many.join("\n"));
  const { outDir } = await run({ vaultRoot: vault });
  const { md, json } = await readReport(outDir);
  assert.equal(json.daily.results.filter(result => result.findings.some(finding => finding.kind === "generic-phrase")).length, 36);
  assert.equal(md.split("\n").filter(line => line.startsWith("|") && line.includes("具有重要意义")).length, 30);
  assert.match(md, /共 36 条，仅列出前 30 条，完整清单见 findings\.json/);
});

test("length is reference data: summarised and tabulated, never listed as a problem", async () => {
  const vault = await fixtureVault();
  const long = [frontmatter("2026-10-20"), "", "## Photo-z", "", ...header("2610.00099", "Very long"), ...fields({ 核心结果: "字".repeat(400) }), ""];
  await writeFile(path.join(vault, "arxiv-daily", "daily", "2026-10-20.md"), long.join("\n"));
  const { outDir } = await run({ vaultRoot: vault });
  const { md, json } = await readReport(outDir);

  const lengthSection = md.slice(md.indexOf("### 字段长度"), md.indexOf("## 详细笔记"));
  assert.match(lengthSection, /长度只是参考数据，不算问题/);
  assert.ok(lengthSection.includes("| 2026-10 |"));
  const list = md.slice(md.indexOf("## 问题清单"));
  assert.ok(!list.includes("核心结果长度超出参考范围"));
  assert.ok(!list.includes("main-result-long"));
  assert.ok(!list.includes("2610.00099"));
  // The per-entry data stays in findings.json.
  assert.ok(json.daily.results.some(result => result.id === "2610.00099" && result.findings.some(finding => finding.code === "main-result-long")));
});

test("the report says what it checks and that accuracy and completeness are left to the model judge", async () => {
  const { outDir } = await run();
  const { md } = await readReport(outDir);
  const opening = md.split("\n## ")[0];
  assert.match(opening, /格式、公式、空话/);
  assert.match(opening, /长度[^\n]*参考/);
  assert.match(opening, /不判断[^\n]*准确[^\n]*完整/);
  assert.match(opening, /模型评审/);
});

test("an existing report is not overwritten without --force", async () => {
  const { outDir, vaultRoot } = await run();
  await writeFile(path.join(outDir, "report.md"), "my notes\n");
  await assert.rejects(qe.runSummaryCheck({ vaultRoot, outDir }), /already exists.*--force/s);
  assert.equal(await readFile(path.join(outDir, "report.md"), "utf8"), "my notes\n");

  const only = await temp("only");
  await writeFile(path.join(only, "findings.json"), "{}");
  await assert.rejects(qe.runSummaryCheck({ vaultRoot, outDir: only }), /findings\.json already exists/);

  await qe.runSummaryCheck({ vaultRoot, outDir, force: true });
  assert.match(await readFile(path.join(outDir, "report.md"), "utf8"), /^# /);
});

test("the date range limits what is checked and is written into the report", async () => {
  const { outDir } = await run({ rest: { from: "2026-10-01", to: "2026-10-31" } });
  const { md, json } = await readReport(outDir);
  assert.deepEqual(json.scope.range, { from: "2026-10-01", to: "2026-10-31" });
  assert.equal(json.corpus.counts.reports, 1);
  assert.ok(md.includes("2026-10-01") && md.includes("2026-10-31"));
});

test("a vault with no reports or notes is an error, and a bad out directory is reported early", async () => {
  await assert.rejects(qe.runSummaryCheck({ vaultRoot: await temp("blank"), outDir: path.join(await temp("o"), "r") }), /no daily reports or detail notes/i);
  const file = path.join(await temp("f"), "plain");
  await writeFile(file, "x");
  await assert.rejects(qe.runSummaryCheck({ vaultRoot: await fixtureVault(), outDir: path.join(file, "sub") }), /not a directory/);
});

test("a config supplies the vault root and the output layout; explicit options win", async () => {
  const vault = await fixtureVault("research/daily", "research/papers");
  const configPath = path.join(await temp("config"), "config.toml");
  await writeFile(configPath, [
    "schema_version = 1", `vault_root = ${JSON.stringify(vault)}`, "",
    "[llm]", 'api_key = "never-printed"', 'base_url = "https://api.example.com/v1"', 'model = "m"', "",
    "[arxiv]", 'categories = ["astro-ph"]', 'timezone = "UTC"', "",
    "[output]", 'daily_dir = "research/daily"', 'papers_dir = "research/papers"', "",
  ].join("\n"));
  const outDir = path.join(await temp("out"), "r");
  await qe.runSummaryCheck({ configPath, outDir });
  const { md, json } = await readReport(outDir);
  assert.equal(json.corpus.counts.entries, 4);
  assert.ok(!md.includes("never-printed") && !JSON.stringify(json).includes("never-printed"));

  // An explicit directory wins over the config's.
  const out2 = path.join(await temp("out"), "r");
  await assert.rejects(qe.runSummaryCheck({ configPath, outDir: out2, dailyDir: "nowhere", papersDir: "nowhere" }), /no daily reports or detail notes/i);
});

test("command-line arguments are validated", () => {
  assert.deepEqual(
    qe.parseSummaryCheckArgs(["--vault", "/v", "--daily-dir", "d", "--papers-dir", "p", "--from", "2026-10-01", "--to", "2026-10-31", "--out", "/o", "--force"]),
    { vaultRoot: "/v", dailyDir: "d", papersDir: "p", from: "2026-10-01", to: "2026-10-31", outDir: "/o", force: true },
  );
  assert.deepEqual(qe.parseSummaryCheckArgs(["--out", "/o", "--config", "/c.toml"]), { outDir: "/o", configPath: "/c.toml" });
  assert.equal(qe.parseSummaryCheckArgs(["--help"]).help, true);
  assert.throws(() => qe.parseSummaryCheckArgs(["--vault", "/v"]), /--out/);
  assert.throws(() => qe.parseSummaryCheckArgs(["--out", "/o", "--bogus"]), /unknown option.*--bogus/i);
  assert.throws(() => qe.parseSummaryCheckArgs(["--out"]), /--out needs a value/);
  assert.throws(() => qe.parseSummaryCheckArgs(["--out", "/o", "--from", "2026-13-01"]), /--from must be a date/);
  assert.throws(() => qe.parseSummaryCheckArgs(["--out", "/o", "--from", "2026-10-02", "--to", "2026-10-01"]), /after/);
});

test("the script prints usage on bad arguments and writes nothing", async () => {
  const { spawnSync } = await import("node:child_process");
  const script = repoFile("scripts/quality-eval/summary-check.mjs");
  const bad = spawnSync(process.execPath, [script, "--vault", "/nope"], { encoding: "utf8" });
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /--out/);
  assert.match(bad.stderr, /Usage: node scripts\/quality-eval\/summary-check\.mjs/);
  const help = spawnSync(process.execPath, [script, "--help"], { encoding: "utf8" });
  assert.equal(help.status, 0);
  assert.match(help.stdout, /--daily-dir/);

  const vaultRoot = await fixtureVault();
  const outDir = path.join(await temp("cli"), "r");
  const good = spawnSync(process.execPath, [script, "--vault", vaultRoot, "--out", outDir], { encoding: "utf8" });
  assert.equal(good.status, 0, good.stderr);
  assert.match(good.stdout, /report\.md/);
});

async function groundingVault() {
  const vault = await temp("grounding-vault");
  const daily = path.join(vault, "arxiv-daily", "daily");
  const papers = path.join(vault, "arxiv-daily", "papers");
  await mkdir(daily, { recursive: true });
  await mkdir(papers, { recursive: true });
  await writeFile(path.join(daily, "2026-10-20.md"), [
    frontmatter("2026-10-20"), "", "## Photo-z", "",
    ...header("2610.00101", "With source"), ...fields({ 核心结果: "质量比为 $0.85\\pm0.04$，样本 120 个，误差 0.99。" }), "",
    ...header("2610.00102", "Without source"), ...fields({ 核心结果: "样本 450 个。" }), "",
  ].join("\n"));
  const quiet = Object.fromEntries(NOTE_HEADINGS.map(heading => [heading, "没有数字的内容。"]));
  await writeFile(path.join(papers, "2607.00101.md"), note("2607.00101", { ...quiet, 核心结果: "测得 $0.85$，另有 777 个。" }));
  return vault;
}
const sourceRecord = (id, text, extra = {}) => ({
  schemaVersion: 1, id, fetchedAt: "2026-10-28T10:00:00.000Z", sourcePath: "html", arxivVersion: "v2",
  budgets: { sectionCharLimit: 16000, paperCharLimit: 100000 },
  content: { abstract: "ratio 0.85", abstractConclusion: "## Abstract\nratio 0.85", fullSections: text, fullTextFailure: null },
  contentHash: "x", ...extra,
});
async function writeSources(records) {
  const dir = await temp("sources");
  for (const record of records) await writeFile(path.join(dir, `${record.id}.json`), JSON.stringify(record));
  return dir;
}

test("without --sources the report says the number check was not run", async () => {
  const { outDir } = await run({ vaultRoot: await groundingVault() });
  const { md, json } = await readReport(outDir);
  assert.equal(json.grounding.run, false);
  assert.match(md, /没有指定 --sources，未做数字溯源检查/);
  assert.ok(!md.includes("疑似无出处数字（需人工核对"));
});

test("with --sources numbers are looked up in the cached source and the rest are listed as suspect", async () => {
  const sourcesDir = await writeSources([
    sourceRecord("2610.00101", "## Methods\nWe use 120 clusters and measure 0.85 with an error of 0.04."),
    sourceRecord("2607.00101", "## Results\nWe obtain 0.85 for the sample."),
    sourceRecord("2606.99999", "## Unrelated\n999"),
  ]);
  const { outDir } = await run({ vaultRoot: await groundingVault(), rest: { sourcesDir } });
  const { md, json } = await readReport(outDir);

  assert.equal(json.grounding.run, true);
  assert.deepEqual(json.grounding.sources.daily, { withSource: 1, withoutSource: 1 });
  assert.deepEqual(json.grounding.sources.notes, { withSource: 1, withoutSource: 0 });
  const daily = json.grounding.daily.aggregate;
  assert.deepEqual(daily.numbers, { count: 3, of: 4, rate: 0.75 });
  assert.deepEqual(daily.byPlace.mainResult, { count: 3, of: 4, rate: 0.75 });
  assert.deepEqual(json.grounding.notes.aggregate.numbers, { count: 1, of: 2, rate: 0.5 });
  assert.deepEqual(json.grounding.daily.results.map(result => result.id), ["2610.00101"]);

  assert.ok(md.includes("疑似无出处数字（需人工核对，可能是规范化遗漏或原文版本不同）"));
  const list = md.slice(md.indexOf("### 疑似无出处数字"));
  assert.ok(list.includes("0.99") && list.includes("2610.00101"));
  assert.ok(list.includes("777") && list.includes("2607.00101"));
  assert.ok(!list.includes("2610.00102"), "an entry without a source takes no part");
  assert.match(md, /无原文[^\n]*1/);
  assert.match(md, /重新取回|重新抓取/);
  assert.ok(md.includes("2026-10-28T10:00:00.000Z"));
  assert.ok(md.includes("v2"));
});

test("--sources must name an existing directory", async () => {
  const missing = path.join(await temp("nowhere"), "absent");
  await assert.rejects(qe.runSummaryCheck({ vaultRoot: await groundingVault(), outDir: path.join(await temp("o"), "r"), sourcesDir: missing }), /--sources.*does not exist|not a directory/i);
  assert.equal(qe.parseSummaryCheckArgs(["--out", "/o", "--sources", "/s"]).sourcesDir, "/s");
});
