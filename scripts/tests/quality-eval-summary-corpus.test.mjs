import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
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

const METRICS = [
  "<!-- arxiv-daily:generation-metrics -->",
  "> [!info]- Generation metrics",
  '> <!-- arxiv-daily:generation-metrics:v1 {"logicalCalls":2,"attempts":2,"elapsedMs":1000,"usageComplete":true,"inputTokens":10,"outputTokens":5,"totalTokens":15,"generatedAt":"2026-09-15T01:00:00.000Z"} -->',
  "> - Generated at: 2026-09-15T01:00:00.000Z",
  "> - LLM calls: 2 logical, 2 HTTP attempts",
  "> - LLM duration: 1.0 s",
  "> - Provider token usage: 10 input / 5 output / 15 total",
].join("\n");

const frontmatter = date => `---\ndate: ${date}\nweekday: Monday\ntags: [arxiv, daily]\n---\n\n# arXiv astro-ph.CO 每日追踪 ${date}\n`;

const header = (id, title, extra = "") => [
  `### ${title}${extra}`,
  `- **作者**: A. Author et al.`,
  `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
];

const flatFields = (tag, labels = ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"]) =>
  labels.map(label => `- **${label}**: ${label}的内容 ${tag}，含公式 $z_{\\rm phot}$。`);

/** The five-field flat layout the current renderer writes. */
const FLAT_2026_10_28 = [
  frontmatter("2026-10-28"),
  "共 3 篇相关论文，其中 0 篇附独立论文总结。",
  "",
  "## Galaxy Cluster",
  "",
  ...header("2610.00001", "Flat cluster paper one"), ...flatFields("A1"), "",
  ...header("2610.00002", "Flat cluster paper two", " → [[2610.00002]]"), ...flatFields("A2"), "",
  "## Photo-z",
  "",
  ...header("2610.00003", "Flat photo-z paper"), ...flatFields("A3"), "",
  "## 其他关注主题",
  "- **Void** — 今日无相关论文更新。",
  "",
].join("\n");

/** The folded callout layout of 0.6.x: the result bullet stays outside, the rest sits in a folded callout. */
const CALLOUT_2026_10_14 = [
  frontmatter("2026-10-14"),
  "共 1 篇相关论文，其中 0 篇附独立论文总结。",
  "",
  "## Photo-z",
  "",
  "### Callout layout paper",
  "> [!info]- 命中方向与信息来源",
  "> <!-- arxiv-daily-topic-directions:v1:eyJ2IjoxfQ -->",
  "> **命中方向**",
  "> - Photometric redshift methods\\.",
  ">",
  "> 信息来源： Abstract, I Introduction",
  "",
  "- **作者**: B. Author et al.",
  "- **arXiv**: [2610.00010](https://arxiv.org/abs/2610.00010)",
  "",
  "- **核心结果**: 核心结果的内容 C1。",
  "",
  "> [!abstract]- 研究背景、方法与边界",
  "> - **研究问题**: 研究问题的内容 C1。",
  ">",
  "> - **方法设计**: 方法设计的内容 C1。",
  ">",
  "> - **研究价值**: 研究价值的内容 C1。",
  ">",
  "> - **适用边界**: 适用边界的内容 C1。",
  "",
].join("\n");

/** Early field labels of 2026-06-12 .. 2026-06-17. */
const EARLY_2026_06_12 = [
  frontmatter("2026-06-12"),
  "共 1 篇相关论文，其中 0 篇附独立论文总结。",
  "",
  "## Galaxy Cluster",
  "",
  ...header("2606.00001", "Early label paper"),
  ...flatFields("E1", ["核心问题", "关键方法", "主要结果", "为什么值得看", "局限或边界"]), "",
].join("\n");

/** The pre-June report schema: no five fields. */
const LEGACY_2026_05_05 = [
  frontmatter("2026-05-05"),
  "共 1 篇相关论文，其中 1 篇详细收录。",
  "",
  "## Galaxy Cluster 相关",
  ...header("2605.00001", "Legacy report paper"),
  "- **一句话总结**: 一句话。",
  "- **数据**: 数据。",
  "- **方法**: 方法。",
  "- **主要结果**: 结果。",
  "- **意义**: 意义。",
  "",
].join("\n");

/** A fallback block plus a structured paper, with the metrics footer. */
const MIXED_2026_09_15 = [
  frontmatter("2026-09-15"),
  "共 2 篇相关论文，其中 0 篇附独立论文总结。其中 1 篇使用回退内容。",
  "",
  "## Galaxy Cluster",
  "",
  "### Fallback paper",
  "> **自动摘要不可用。** 请直接阅读 [arXiv 原文](https://arxiv.org/abs/2609.00001)。",
  "<!-- arxiv-daily-fallback:2609.00001 -->",
  "- **作者**: C. Author et al.",
  "- **arXiv**: [2609.00001](https://arxiv.org/abs/2609.00001)",
  "- **原始摘要**: The original abstract.",
  "",
  ...header("2609.00002", "Structured beside fallback"), ...flatFields("M1"), "",
  METRICS, "",
].join("\n");

const NOTE_SECTIONS = ["研究问题", "方法设计", "核心结果", "主要结论"];
const noteMarkdown = (id, extra = {}) => [
  "---",
  `title: "Detail note ${id}"`,
  'authors: "D. Author et al."',
  `arxiv_id: "${id}"`,
  `primary_topic: ${extra.topic ?? "galaxy-cluster"}`,
  `published: ${extra.published ?? "2026-07-10"}`,
  "tags: [arxiv, paper]",
  "---",
  "",
  `# Detail note ${id}`,
  "",
  `- **arXiv**: https://arxiv.org/abs/${id}`,
  "",
  ...NOTE_SECTIONS.flatMap(heading => [`## ${heading}`, "", `${heading}的正文，含 $\\Omega_m=0.3$。`, "", "第二段。", ""]),
  ...(extra.footer ? [METRICS, ""] : []),
].join("\n");

const LEGACY_NOTE = [
  "---",
  'title: "Old note"',
  'arxiv: "2603.00001"',
  "date: 2026-03-05",
  "tags: [galaxy-cluster]",
  "---",
  "",
  "# Old note",
  "",
  "## 背景与动机",
  "旧格式内容。",
  "",
].join("\n");

async function fixtureVault() {
  const vault = await temp("vault");
  const daily = path.join(vault, "arxiv-daily", "daily");
  const papers = path.join(vault, "arxiv-daily", "papers");
  await mkdir(daily, { recursive: true });
  await mkdir(papers, { recursive: true });
  const reports = {
    "2026-05-05": LEGACY_2026_05_05, "2026-06-12": EARLY_2026_06_12, "2026-09-15": MIXED_2026_09_15,
    "2026-10-14": CALLOUT_2026_10_14, "2026-10-28": FLAT_2026_10_28,
  };
  for (const [date, markdown] of Object.entries(reports)) await writeFile(path.join(daily, `${date}.md`), markdown);
  await writeFile(path.join(daily, "scratch.md"), "# not a report\n");
  await writeFile(path.join(papers, "2607.00001.md"), noteMarkdown("2607.00001", { footer: true }));
  await writeFile(path.join(papers, "2607.00002.md"), noteMarkdown("2607.00002", { topic: "photo-z", published: '"[[arxiv-daily/daily/2026-06-17|2026-06-17]]"' }));
  await writeFile(path.join(papers, "2603.00001.md"), LEGACY_NOTE);
  return vault;
}

test("loadSummaryCorpus reads five-field entries from flat, callout and early-label reports", async () => {
  const corpus = await qe.loadSummaryCorpus({ vaultRoot: await fixtureVault() });

  const key = entry => `${entry.date} ${entry.id}`;
  assert.deepEqual(corpus.entries.map(key), [
    "2026-06-12 2606.00001", "2026-09-15 2609.00002", "2026-10-14 2610.00010",
    "2026-10-28 2610.00001", "2026-10-28 2610.00002", "2026-10-28 2610.00003",
  ]);

  const flat = corpus.entries.find(entry => entry.id === "2610.00002");
  assert.equal(flat.topic, "Galaxy Cluster");
  assert.equal(flat.title, "Flat cluster paper two");
  assert.deepEqual(Object.keys(flat.fields), ["coreProblem", "keyMethod", "mainResult", "whyRelevant", "limitations"]);
  assert.equal(flat.fields.coreProblem, "研究问题的内容 A2，含公式 $z_{\\rm phot}$。");
  assert.equal(flat.fields.limitations, "适用边界的内容 A2，含公式 $z_{\\rm phot}$。");

  const callout = corpus.entries.find(entry => entry.id === "2610.00010");
  assert.equal(callout.topic, "Photo-z");
  assert.equal(callout.fields.mainResult, "核心结果的内容 C1。");
  assert.equal(callout.fields.coreProblem, "研究问题的内容 C1。");
  assert.equal(callout.fields.limitations, "适用边界的内容 C1。");

  const early = corpus.entries.find(entry => entry.id === "2606.00001");
  assert.equal(early.fields.coreProblem, "核心问题的内容 E1，含公式 $z_{\\rm phot}$。");
  assert.equal(early.fields.keyMethod, "关键方法的内容 E1，含公式 $z_{\\rm phot}$。");
  assert.equal(early.fields.mainResult, "主要结果的内容 E1，含公式 $z_{\\rm phot}$。");
  assert.equal(early.fields.whyRelevant, "为什么值得看的内容 E1，含公式 $z_{\\rm phot}$。");
  assert.equal(early.fields.limitations, "局限或边界的内容 E1，含公式 $z_{\\rm phot}$。");
  assert.equal(corpus.counts.incompleteEntries, 0);
});

test("loadSummaryCorpus counts fallback entries apart and the metrics footer never leaks in", async () => {
  const corpus = await qe.loadSummaryCorpus({ vaultRoot: await fixtureVault() });

  assert.deepEqual(corpus.fallbacks, [
    { date: "2026-09-15", topic: "Galaxy Cluster", id: "2609.00001", title: "Fallback paper" },
  ]);
  assert.ok(!corpus.entries.some(entry => entry.id === "2609.00001"));
  const beside = corpus.entries.find(entry => entry.id === "2609.00002");
  assert.ok(!JSON.stringify(beside).includes("Generation metrics"));
  assert.ok(!beside.fields.limitations.includes("LLM calls"));
});

test("loadSummaryCorpus counts reports and tells legacy reports from evaluable ones", async () => {
  const corpus = await qe.loadSummaryCorpus({ vaultRoot: await fixtureVault() });

  assert.equal(corpus.counts.reports, 5);
  assert.equal(corpus.counts.evaluableReports, 4);
  assert.equal(corpus.counts.legacyReports, 1);
  assert.equal(corpus.counts.entries, 6);
  assert.equal(corpus.counts.fallbacks, 1);
  assert.deepEqual(corpus.legacyReportDates, ["2026-05-05"]);
});

test("loadSummaryCorpus keeps reports inside the inclusive date range", async () => {
  const vault = await fixtureVault();

  const middle = await qe.loadSummaryCorpus({ vaultRoot: vault, from: "2026-06-12", to: "2026-10-14" });
  assert.deepEqual([...new Set(middle.entries.map(entry => entry.date))], ["2026-06-12", "2026-09-15", "2026-10-14"]);
  assert.equal(middle.counts.reports, 3);
  assert.equal(middle.counts.legacyReports, 0);
  assert.equal(middle.counts.fallbacks, 1);

  const onlyLegacy = await qe.loadSummaryCorpus({ vaultRoot: vault, to: "2026-05-31" });
  assert.equal(onlyLegacy.counts.reports, 1);
  assert.equal(onlyLegacy.counts.legacyReports, 1);
  assert.equal(onlyLegacy.entries.length, 0);

  const single = await qe.loadSummaryCorpus({ vaultRoot: vault, from: "2026-10-28", to: "2026-10-28" });
  assert.equal(single.entries.length, 3);
});

test("loadSummaryCorpus reads current-format detail notes and counts legacy ones", async () => {
  const corpus = await qe.loadSummaryCorpus({ vaultRoot: await fixtureVault() });

  assert.deepEqual(corpus.notes.map(note => note.id), ["2607.00001", "2607.00002"]);
  assert.equal(corpus.counts.notes, 2);
  assert.equal(corpus.counts.legacyNotes, 1);

  const note = corpus.notes[0];
  assert.equal(note.topic, "galaxy-cluster");
  assert.equal(note.title, "Detail note 2607.00001");
  assert.equal(note.path, path.join("arxiv-daily", "papers", "2607.00001.md"));
  assert.equal(note.date, "2026-07-10");
  assert.deepEqual(note.sections.map(section => section.heading), NOTE_SECTIONS);
  assert.equal(note.sections[0].body, "研究问题的正文，含 $\\Omega_m=0.3$。\n\n第二段。");
  assert.ok(!JSON.stringify(note).includes("Generation metrics"));
  assert.ok(!note.sections.at(-1).body.includes("generation-metrics"));

  // A published value that links the daily report dates the note by that report.
  assert.equal(corpus.notes[1].date, "2026-06-17");
  assert.equal(corpus.notes[1].topic, "photo-z");
});

test("loadSummaryCorpus dates notes by their report, filters them, and keeps undated ones", async () => {
  const vault = await fixtureVault();
  await writeFile(path.join(vault, "arxiv-daily", "papers", "2607.00003.md"), noteMarkdown("2607.00003").replace(/^published: .*\n/m, ""));

  const all = await qe.loadSummaryCorpus({ vaultRoot: vault });
  assert.equal(all.notes.find(note => note.id === "2607.00003").date, null);

  const june = await qe.loadSummaryCorpus({ vaultRoot: vault, from: "2026-06-01", to: "2026-06-30" });
  assert.deepEqual(june.notes.map(note => note.id), ["2607.00002", "2607.00003"]);
  assert.equal(june.counts.legacyNotes, 0);

  const march = await qe.loadSummaryCorpus({ vaultRoot: vault, from: "2026-03-01", to: "2026-03-31" });
  assert.equal(march.counts.legacyNotes, 1);
});

test("loadSummaryCorpus recognises the English heading and rejects notes missing identity", async () => {
  const vault = await fixtureVault();
  const papers = path.join(vault, "arxiv-daily", "papers");
  const english = noteMarkdown("2607.00004").replace("## 研究问题", "## Research Problem");
  await writeFile(path.join(papers, "2607.00004.md"), english);
  const noTopic = noteMarkdown("2607.00005").replace(/^primary_topic: .*\n/m, "");
  await writeFile(path.join(papers, "2607.00005.md"), noTopic);

  const corpus = await qe.loadSummaryCorpus({ vaultRoot: vault });
  assert.ok(corpus.notes.some(note => note.id === "2607.00004"));
  assert.ok(!corpus.notes.some(note => note.id === "2607.00005"));
  assert.equal(corpus.counts.legacyNotes, 2);
});

test("loadSummaryCorpus honours custom directories and tolerates their absence", async () => {
  const vault = await temp("custom");
  await mkdir(path.join(vault, "d"), { recursive: true });
  await writeFile(path.join(vault, "d", "2026-10-28.md"), FLAT_2026_10_28);

  const corpus = await qe.loadSummaryCorpus({ vaultRoot: vault, dailyDir: "d", papersDir: "missing" });
  assert.equal(corpus.entries.length, 3);
  assert.equal(corpus.notes.length, 0);

  const empty = await qe.loadSummaryCorpus({ vaultRoot: await temp("empty") });
  assert.equal(empty.counts.reports, 0);
  assert.equal(empty.counts.notes, 0);
});

test("loadSummaryCorpus rejects malformed or inverted date ranges", async () => {
  const vault = await fixtureVault();
  await assert.rejects(qe.loadSummaryCorpus({ vaultRoot: vault, from: "2026/10/01" }), /YYYY-MM-DD/);
  await assert.rejects(qe.loadSummaryCorpus({ vaultRoot: vault, to: "2026-13-40" }), /YYYY-MM-DD/);
  await assert.rejects(qe.loadSummaryCorpus({ vaultRoot: vault, from: "2026-10-02", to: "2026-10-01" }), /before/);
});

test("loadSummaryCorpus keeps a five-field report's half-parsed paper as an incomplete entry", async () => {
  const vault = await temp("incomplete");
  await mkdir(path.join(vault, "arxiv-daily", "daily"), { recursive: true });
  const report = [
    frontmatter("2026-10-20"), "", "## Photo-z", "",
    ...header("2610.00020", "Whole paper"), ...flatFields("W1"), "",
    ...header("2610.00021", "Half paper"), "- **研究问题**: 只有这一项。", "",
    ...header("2610.00022", "Bare paper"), "",
  ].join("\n");
  await writeFile(path.join(vault, "arxiv-daily", "daily", "2026-10-20.md"), report);

  const corpus = await qe.loadSummaryCorpus({ vaultRoot: vault });
  assert.deepEqual(corpus.entries.map(entry => entry.id), ["2610.00020", "2610.00021", "2610.00022"]);
  assert.equal(corpus.counts.incompleteEntries, 2);
  assert.deepEqual(corpus.entries[1].fields, {
    coreProblem: "只有这一项。", keyMethod: "", mainResult: "", whyRelevant: "", limitations: "",
  });
});
