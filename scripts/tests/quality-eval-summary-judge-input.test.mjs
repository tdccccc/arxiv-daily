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

const frontmatter = date => `---\ndate: ${date}\ntags: [arxiv, daily]\n---\n\n# arXiv astro-ph.CO 每日追踪 ${date}\n`;
const FIELD_LABELS = ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"];
const block = (id, title, overrides = {}) => [
  `### ${title}`, `- **作者**: ${id} Author et al.`, `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
  ...FIELD_LABELS.map(label => `- **${label}**: ${overrides[label] ?? `${id} 的${label}。质量比为 $0.85\\pm0.04$。`}`), "",
];
const fallbackBlock = id => [
  `### Fallback ${id}`, "> **自动摘要不可用。**", `<!-- arxiv-daily-fallback:${id} -->`, "- **作者**: C. Author", `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`, "- **原始摘要**: abstract.", "",
];

const ZH_HEADINGS = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "贡献与创新点", "适用边界", "学术价值判断"];
const OLDER_HEADINGS = ["研究问题", "方法设计", "关键证据", "主要结论", "适用边界", "阅读价值"];
const noteMarkdown = (id, headings) => [
  "---", `title: "Note ${id}"`, 'authors: "N. Author et al."', `arxiv_id: "${id}"`, "primary_topic: galaxy-cluster", "published: 2026-07-10", "tags: [arxiv, paper]", "---", "",
  `# Note ${id}`, "", `- **arXiv**: https://arxiv.org/abs/${id}`, "",
  ...headings.flatMap(heading => [`## ${heading}`, "", `${id} 的${heading}正文。`, "", "第二段。", ""]),
].join("\n");

const record = (id, extra = {}) => ({
  schemaVersion: 1, id, fetchedAt: "2026-10-28T10:00:00.000Z", sourcePath: "html", arxivVersion: "v1",
  budgets: { sectionCharLimit: 16000, paperCharLimit: 100000 },
  content: {
    abstract: `Abstract of ${id}.`, abstractConclusion: `## Abstract\nAbstract of ${id}.`,
    fullSections: `## Methods\nThe mass ratio of ${id} is 0.85 \\pm 0.04 in <b>this</b> sample.\n\n## Results\nWe use 120 clusters.`,
    fullTextFailure: null,
  },
  contentHash: "x", ...extra,
});

const MONTHS = { "2026-08": 10, "2026-09": 6, "2026-10": 4 };
const idOf = (month, index) => `26${month.slice(5)}.${String(index + 1).padStart(5, "0")}`;
const DAILY_IDS = Object.entries(MONTHS).flatMap(([month, count]) => Array.from({ length: count }, (_, index) => idOf(month, index)));
const NO_SOURCE = "2610.00002";
const NO_INPUT = "2609.00003";
const INCOMPLETE = "2608.00004";
const CURRENT_NOTES = ["2607.00001", "2607.00002", "2607.00003", "2607.00004"];
const OLDER_NOTES = ["2606.00001", "2606.00002"];

async function fixture() {
  const vault = await temp("vault");
  const daily = path.join(vault, "arxiv-daily", "daily");
  const papers = path.join(vault, "arxiv-daily", "papers");
  const sources = await temp("sources");
  await mkdir(daily, { recursive: true });
  await mkdir(papers, { recursive: true });
  for (const month of Object.keys(MONTHS)) {
    const ids = DAILY_IDS.filter(id => id.startsWith(`26${month.slice(5)}.`));
    const lines = [frontmatter(`${month}-15`), "", "## Photo-z", ""];
    for (const id of ids) lines.push(...block(id, `Paper ${id}`, id === INCOMPLETE ? { 核心结果: "" } : {}));
    if (month === "2026-09") lines.push(...fallbackBlock("2609.09999"));
    await writeFile(path.join(daily, `${month}-15.md`), lines.join("\n"));
  }
  for (const id of CURRENT_NOTES) await writeFile(path.join(papers, `${id}.md`), noteMarkdown(id, ZH_HEADINGS));
  for (const id of OLDER_NOTES) await writeFile(path.join(papers, `${id}.md`), noteMarkdown(id, OLDER_HEADINGS));
  for (const id of [...DAILY_IDS, ...CURRENT_NOTES, ...OLDER_NOTES]) {
    if (id === NO_SOURCE) continue;
    const base = record(id);
    const value = id === NO_INPUT
      ? { ...base, content: { ...base.content, abstractConclusion: `[获取失败] arXiv ID: ${id}`, fullSections: null } }
      : base;
    await writeFile(path.join(sources, `${id}.json`), JSON.stringify(value));
  }
  return { vault, sources };
}

const sample = async (options = {}) => {
  const { vault } = await fixture();
  const corpus = await qe.loadSummaryCorpus({ vaultRoot: vault });
  return { corpus, ...qe.sampleJudgeTargets(corpus, options) };
};
const idsOf = targets => targets.daily.map(entry => entry.id);

test("the corpus carries authors for the model input", async () => {
  const { vault } = await fixture();
  const corpus = await qe.loadSummaryCorpus({ vaultRoot: vault });
  assert.equal(corpus.entries[0].authors, `${corpus.entries[0].id} Author et al.`);
  assert.equal(corpus.notes[0].authors, "N. Author et al.");
});

test("the same seed gives the same sample and a different seed a different one", async () => {
  const a = await sample({ seed: "s1", daily: 7, notes: 2 });
  const b = await sample({ seed: "s1", daily: 7, notes: 2 });
  const c = await sample({ seed: "s2", daily: 7, notes: 2 });
  assert.deepEqual(idsOf(a), idsOf(b));
  assert.deepEqual(a.notes.map(note => note.id), b.notes.map(note => note.id));
  assert.notDeepEqual(idsOf(a), idsOf(c));
  assert.equal(idsOf(a).length, 7);
  assert.equal(new Set(idsOf(a)).size, 7);
  // A default seed exists and is stable.
  assert.deepEqual(idsOf(await sample({ daily: 5 })), idsOf(await sample({ daily: 5 })));
});

test("daily entries are stratified across months in proportion to their counts", async () => {
  const exact = await sample({ daily: 10 });
  assert.deepEqual(exact.allocation, { "2026-08": 5, "2026-09": 3, "2026-10": 2 });
  const months = targets => targets.daily.reduce((all, entry) => ({ ...all, [entry.date.slice(0, 7)]: (all[entry.date.slice(0, 7)] ?? 0) + 1 }), {});
  assert.deepEqual(months(exact), exact.allocation);
  // The remainder goes to the largest fractions, ties to the earlier month.
  assert.deepEqual((await sample({ daily: 7 })).allocation, { "2026-08": 4, "2026-09": 2, "2026-10": 1 });
  assert.deepEqual((await sample({ daily: 1 })).allocation, { "2026-08": 1, "2026-09": 0, "2026-10": 0 });
  // More than exist: everything, once.
  const all = await sample({ daily: 100 });
  assert.equal(idsOf(all).length, DAILY_IDS.length - 0);
  assert.deepEqual(all.allocation, MONTHS);
});

test("fallback entries are never part of the sample", async () => {
  const all = await sample({ daily: 100 });
  assert.ok(!idsOf(all).includes("2609.09999"));
});

test("--ids overrides sampling and takes the daily entry and the note with that id", async () => {
  const picked = await sample({ ids: ["2608.00002", "2607.00002", "2610.00001"], daily: 3, notes: 3, seed: "x" });
  assert.deepEqual(idsOf(picked).sort(), ["2608.00002", "2610.00001"]);
  assert.deepEqual(picked.notes.map(note => note.id), ["2607.00002"]);
  assert.deepEqual(picked.missing, []);

  const withUnknown = await sample({ ids: ["2608.00002", "2699.99999", "2609.09999"] });
  assert.deepEqual(withUnknown.missing.map(entry => [entry.id, entry.reason]), [["2699.99999", "not-found"], ["2609.09999", "fallback"]]);

  // An explicit id may name an older-format note.
  assert.deepEqual((await sample({ ids: ["2606.00001"] })).notes.map(note => note.id), ["2606.00001"]);
});

test("only current-version notes are sampled unless older ones are asked for", async () => {
  const current = await sample({ notes: 100 });
  assert.deepEqual(current.notes.map(note => note.id).sort(), CURRENT_NOTES);
  const everything = await sample({ notes: 100, includeOlderNotes: true });
  assert.deepEqual(everything.notes.map(note => note.id).sort(), [...OLDER_NOTES, ...CURRENT_NOTES].sort());
});

test("the date range limits what is sampled", async () => {
  const { vault } = await fixture();
  const corpus = await qe.loadSummaryCorpus({ vaultRoot: vault, from: "2026-10-01" });
  const targets = qe.sampleJudgeTargets(corpus, { daily: 100 });
  assert.ok(targets.daily.every(entry => entry.date >= "2026-10-01"));
  assert.equal(targets.daily.length, 4);
});

const buildAll = async (options = {}) => {
  const { vault, sources } = await fixture();
  return qe.prepareJudgeInput({ vaultRoot: vault, sourcesDir: sources, daily: 100, notes: 100, ...options });
};

test("items without a source or with an unusable input are skipped with a reason, not replaced", async () => {
  const prepared = await buildAll();
  const skipped = prepared.skipped.map(entry => [entry.kind, entry.id, entry.reason]).sort();
  assert.deepEqual(skipped, [["daily", INCOMPLETE, "incomplete"], ["daily", NO_INPUT, "no-input"], ["daily", NO_SOURCE, "no-source"]].sort());
  assert.equal(prepared.items.filter(item => item.kind === "daily").length, DAILY_IDS.length - 3);
  assert.deepEqual(prepared.shortfall, { daily: 3, notes: 0 });
  assert.ok(prepared.items.every(item => ![NO_SOURCE, NO_INPUT, INCOMPLETE].includes(item.id)));
  assert.ok(prepared.skipped.every(entry => typeof entry.detail === "string" && entry.detail.length > 0));
});

test("a daily item shows the judge the exact paper data and every field", async () => {
  const prepared = await buildAll();
  const item = prepared.items.find(candidate => candidate.kind === "daily" && candidate.id === "2608.00001");
  const entry = prepared.corpus.entries.find(candidate => candidate.id === item.id);
  const rebuilt = await qe.rebuildDailyInput(JSON.parse(await (await import("node:fs/promises")).readFile(path.join(prepared.sourcesDir, `${item.id}.json`), "utf8")), { title: entry.title, authors: entry.authors });

  assert.equal(item.paperData, rebuilt.messages[1].content);
  assert.ok(item.paperData.startsWith("<paper_data>"));
  assert.deepEqual(item.messages.map(message => message.role), ["system", "user"]);
  assert.equal(item.messages[0].content, qe.buildJudgeSystemPrompt({ kind: "daily" }));
  assert.ok(item.messages[1].content.startsWith(item.paperData));
  assert.deepEqual(item.locations.map(location => location.key), qe.SUMMARY_FIELDS);
  for (const [key, text] of Object.entries(entry.fields)) {
    assert.ok(item.messages[1].content.includes(`[${key}] ${text}`), key);
  }
  assert.match(item.messages[1].content, /<summary_under_review>[\s\S]*<\/summary_under_review>/);
  assert.equal(item.date, entry.date);
  assert.equal(item.topic, "Photo-z");
  assert.equal(item.source.sourcePath, "html");
});

test("a note item shows the judge the note body with its own headings", async () => {
  const prepared = await buildAll();
  const item = prepared.items.find(candidate => candidate.kind === "note" && candidate.id === "2607.00001");
  const note = prepared.corpus.notes.find(candidate => candidate.id === item.id);
  assert.deepEqual(item.locations.map(location => location.key), ZH_HEADINGS);
  assert.equal(item.messages[0].content, qe.buildJudgeSystemPrompt({ kind: "note", headings: ZH_HEADINGS, topic: "galaxy-cluster" }));
  for (const section of note.sections) {
    assert.ok(item.messages[1].content.includes(`## ${section.heading}\n${section.body}`), section.heading);
  }
  assert.ok(item.messages[1].content.startsWith(item.paperData));
  assert.ok(item.paperData.includes("以下是论文各章节内容"));
  assert.equal(item.version, "current");
  assert.equal(item.topic, "galaxy-cluster");
  const older = (await buildAll({ includeOlderNotes: true })).items.find(candidate => candidate.id === "2606.00001");
  assert.equal(older.version, "older");
});

test("the topic name from a config is used for the note prompt", async () => {
  const prepared = await buildAll({ topics: [{ tag: "galaxy-cluster", name: "Galaxy Cluster" }] });
  const item = prepared.items.find(candidate => candidate.kind === "note");
  assert.ok(item.messages[0].content.includes("「Galaxy Cluster」"));
});

test("the location texts used for quote checks are exactly what the judge sees", async () => {
  const prepared = await buildAll();
  for (const item of prepared.items) {
    for (const location of item.locations) assert.ok(item.judgedText.includes(location.text), `${item.id} ${location.key}`);
    assert.equal(item.messages[1].content, `${item.paperData}\n\n<summary_under_review>\n${item.judgedText}\n</summary_under_review>`);
    assert.deepEqual(Object.keys(item.context.locations), item.locations.map(location => location.key));
    assert.equal(item.context.sourceText, item.paperData);
    assert.equal(item.context.kind, item.kind);
  }
});

test("the validation context accepts a valid judgment built from the item", async () => {
  const prepared = await buildAll();
  for (const kind of ["daily", "note"]) {
    const item = prepared.items.find(candidate => candidate.kind === kind);
    const location = item.locations[2];
    const judgment = {
      verdicts: { accuracy: "minor", completeness: "ok", clarity: "ok" },
      issues: [{
        dimension: "accuracy", severity: "minor", location: location.key, summaryQuote: location.text.slice(0, 6),
        sourceQuote: "We use 120 clusters.", explanation: "示例。",
      }],
    };
    const result = qe.validateJudgment(judgment, item.context);
    assert.equal(result.ok, true, JSON.stringify(result));
  }
});

test("the estimate counts the characters of every message per kind", async () => {
  const prepared = await buildAll({ daily: 5, notes: 2 });
  const estimate = qe.estimateJudgeInput(prepared.items);
  const chars = items => items.map(item => item.messages.reduce((sum, message) => sum + message.content.length, 0));
  const daily = chars(prepared.items.filter(item => item.kind === "daily"));
  assert.deepEqual(estimate.daily, { items: daily.length, totalChars: daily.reduce((a, b) => a + b, 0), averageChars: Math.round(daily.reduce((a, b) => a + b, 0) / daily.length), maxChars: Math.max(...daily) });
  assert.equal(estimate.note.items, prepared.items.filter(item => item.kind === "note").length);
  assert.equal(estimate.totalChars, estimate.daily.totalChars + estimate.note.totalChars);
  assert.deepEqual(qe.estimateJudgeInput([]).daily, { items: 0, totalChars: 0, averageChars: 0, maxChars: 0 });
});

test("preparing the input reports the allocation and the shortfall, and needs the sources directory", async () => {
  const prepared = await buildAll({ daily: 10, notes: 2 });
  assert.deepEqual(prepared.allocation, { "2026-08": 5, "2026-09": 3, "2026-10": 2 });
  assert.equal(prepared.requested.daily, 10);
  assert.equal(prepared.requested.notes, 2);
  assert.equal(prepared.items.length + prepared.skipped.length, 12);
  const { vault } = await fixture();
  await assert.rejects(qe.prepareJudgeInput({ vaultRoot: vault, sourcesDir: path.join(await temp("none"), "absent"), daily: 1 }), /--sources.*does not exist/);
});
