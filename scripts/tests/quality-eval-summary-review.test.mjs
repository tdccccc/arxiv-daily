import assert from "node:assert/strict";
import { existsSync } from "node:fs";
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
const NOW = new Date("2026-10-29T15:00:00Z");

const SOURCE_TEXT = `## Methods\n${"填充文字 ".repeat(5)}We stack the weak lensing shear profiles of 120 clusters and calibrate against simulations. ${"padding words ".repeat(60)}\n## Results\nThe mass ratio is $0.85 \\pm 0.04$ for the full sample.`;

const issue = (overrides = {}) => ({
  dimension: "accuracy", severity: "major", location: "mainResult", summaryQuote: "样本为130个星团", sourceQuote: "stack the weak lensing shear profiles of 120 clusters",
  explanation: "摘要把样本数写成 130，原文是 120。", ...overrides,
});
const record = (kind, id, issues, extra = {}) => ({
  schemaVersion: 1, status: "judged", invalid: false,
  item: { kind, id, title: `Title of ${id}`, date: "2026-08-10", topic: "Photo-z", source: { sourcePath: "html", arxivVersion: "v1", fetchedAt: "2026-10-28T10:00:00.000Z", contentHash: `hash-${id}` } },
  judge: { provider: "openai", model: "judge-model" }, judgePromptHash: "p".repeat(64), attempts: [],
  judgment: { verdicts: { accuracy: "ok", completeness: "ok", clarity: "ok" }, issues },
  summary: { locations: kind === "daily"
    ? qe.SUMMARY_FIELDS.map(key => ({ key, text: key === "mainResult" ? "弱透镜质量比为 0.85，样本为130个星团，误差较小。" : `${id} 的 ${key} 内容。` }))
    : [{ key: "核心结果", text: "第一段。\n\n样本为130个星团。\n\n第三段。" }] },
  ...extra,
});
const writeJudgeDir = async (records, run = {}) => {
  const dir = await temp("judge");
  await mkdir(path.join(dir, "judgments"), { recursive: true });
  await writeFile(path.join(dir, "run.json"), JSON.stringify({ schemaVersion: 1, startedAt: "x", seed: "s", judge: records[0]?.judge ?? { provider: "openai", model: "judge-model" }, skipped: [], sample: {
    daily: records.filter(item => item.item.kind === "daily").map(item => item.item.id),
    notes: records.filter(item => item.item.kind === "note").map(item => item.item.id),
  }, ...run }));
  for (const item of records) await writeFile(path.join(dir, "judgments", `${item.item.kind}-${item.item.id}.json`), JSON.stringify(item));
  return dir;
};
const sourcesWith = async ids => {
  const dir = await temp("sources");
  for (const id of ids) {
    await writeFile(path.join(dir, `${id}.json`), JSON.stringify({ schemaVersion: 1, id, fetchedAt: "x", sourcePath: "html", arxivVersion: "v1", budgets: {}, content: { abstract: "a", abstractConclusion: "## Abstract\nAbstract.", fullSections: SOURCE_TEXT, fullTextFailure: null }, contentHash: `hash-${id}` }));
  }
  return dir;
};

/** 6 items; majors: 4, minors: 5; dimensions mixed. */
const mainRecords = () => [
  record("daily", "2608.00001", [issue(), issue({ dimension: "completeness", severity: "minor", location: "limitations", summaryQuote: "2608.00001 的 limitations 内容", sourceQuote: "padding words padding words" })]),
  record("daily", "2608.00002", [issue({ dimension: "clarity", severity: "major", sourceQuote: null, location: "keyMethod", summaryQuote: "2608.00002 的 keyMethod 内容", explanation: "缩写未解释。" })]),
  record("daily", "2608.00003", [issue({ severity: "minor", sourceQuote: null, explanation: "原文未提及这个说法。" }), issue({ dimension: "completeness", severity: "major", location: "whyRelevant", summaryQuote: "2608.00003 的 whyRelevant 内容", sourceQuote: "The mass ratio is $0.85 \\pm 0.04$" })]),
  record("daily", "2608.00004", [issue({ severity: "minor" }), issue({ dimension: "clarity", severity: "minor", location: "coreProblem", summaryQuote: "2608.00004 的 coreProblem 内容", sourceQuote: null })]),
  record("note", "2607.00001", [issue({ location: "核心结果", summaryQuote: "样本为130个星团", severity: "major" })]),
  record("daily", "2608.00005", []),
];
const extraRecords = () => [
  record("daily", "2609.00001", [issue()], { origin: { kind: "regenerated", generator: { provider: "openai", model: "gen-model" }, generationPromptHash: "g".repeat(64), generatedAt: "t", summariesDir: "/regen" } }),
  record("daily", "2609.00002", [issue({ severity: "minor" })], { origin: { kind: "regenerated", generator: { provider: "openai", model: "gen-model" }, generationPromptHash: "g".repeat(64), generatedAt: "t", summariesDir: "/regen" } }),
];
const select = async (options = {}) => {
  const main = await writeJudgeDir(mainRecords());
  const extra = options.extra === false ? undefined : await writeJudgeDir(extraRecords());
  const sources = await sourcesWith(["2608.00001", "2608.00002", "2608.00003", "2608.00004", "2607.00001", "2609.00001", "2609.00002"]);
  return { main, extra, sources, ...(await qe.buildReviewCards({ judgmentsDir: main, extraDir: extra, sourcesDir: sources, count: 6, seed: "s1", ...options })) };
};

test("cards are chosen deterministically, majors first, spread over papers and dimensions", async () => {
  const a = await select();
  const b = await select();
  assert.deepEqual(a.cards.map(card => card.cardId), b.cards.map(card => card.cardId));
  assert.equal(a.cards.length, 6);
  assert.ok(a.cards.slice(0, 4).every(card => card.severity === "major"), a.cards.map(card => card.severity).join());
  // Four majors from four different papers/dimension mixes, before any minor.
  assert.equal(new Set(a.cards.slice(0, 4).map(card => card.id)).size, 4);
  assert.notDeepEqual(a.cards.map(card => card.cardId), (await select({ seed: "other" })).cards.map(card => card.cardId));
  const first = a.cards[0];
  assert.deepEqual(Object.keys(first).sort(), ["cardId", "dimension", "explanation", "id", "kind", "location", "origin", "pool", "severity", "sourceContext", "sourceContextNote", "sourceQuote", "summaryQuote", "summaryText", "title"].sort());
});

test("cards come from --judgments first and are filled from --extra when it has too few", async () => {
  const full = await select({ count: 3 });
  assert.ok(full.cards.every(card => card.pool === "judgments"));
  const filled = await select({ count: 12 });
  assert.equal(filled.cards.length, 5 + 4 + 2 - 0 > 0 ? filled.cards.length : 0);
  const pools = filled.cards.map(card => card.pool);
  assert.equal(pools.filter(pool => pool === "judgments").length, 8, "all eight issues of the main directory come first");
  assert.equal(pools.filter(pool => pool === "extra").length, 2);
  assert.equal(filled.cards.length, 10, "no more cards than issues exist");
  assert.ok(filled.cards.filter(card => card.pool === "extra").every(card => card.origin.kind === "regenerated"));
  assert.ok(filled.cards.filter(card => card.pool === "judgments").every(card => card.origin.kind === "historical"));
  assert.equal(filled.shortfall, 2);
  assert.equal((await select({ count: 12, extra: false })).cards.length, 8);
});

test("each card carries the full summary text, the quote's source context and the origin", async () => {
  const { cards } = await select({ count: 10 });
  const card = cards.find(candidate => candidate.id === "2608.00001" && candidate.dimension === "accuracy");
  assert.equal(card.summaryText, "弱透镜质量比为 0.85，样本为130个星团，误差较小。");
  assert.equal(card.origin.kind, "historical");
  assert.ok(card.origin.judgeDir.length > 0);
  assert.ok(card.sourceContext.includes("stack the weak lensing shear profiles of 120 clusters"));
  const quoteAt = card.sourceContext.indexOf(card.sourceQuote);
  assert.ok(quoteAt >= 30 && quoteAt <= 300, `${quoteAt}: the text before the quote is kept, up to 300 characters`);
  assert.ok(card.sourceContext.length <= card.sourceQuote.length + 2 * 300 + 4);
  assert.ok(card.sourceContext.length > card.sourceQuote.length + 50);

  const none = cards.find(candidate => candidate.sourceQuote === null);
  assert.equal(none.sourceContext, null);
  const missing = (await qe.buildReviewCards({ judgmentsDir: await writeJudgeDir([record("daily", "2608.00001", [issue({ sourceQuote: "not in the cached text" })])]), sourcesDir: await sourcesWith(["2608.00001"]), count: 1, seed: "s" })).cards[0];
  assert.equal(missing.sourceContext, null);
  assert.equal(missing.sourceContextNote, "cached-source-lacks-quote");
  const noSources = (await qe.buildReviewCards({ judgmentsDir: await writeJudgeDir([record("daily", "2608.00001", [issue()])]), count: 1, seed: "s" })).cards[0];
  assert.equal(noSources.sourceContext, null);
  assert.equal(noSources.sourceContextNote, "no-sources");
});

test("the summary text comes from a regenerated record or the vault when the judgment did not keep it", async () => {
  const regen = await temp("regen");
  await mkdir(path.join(regen, "summaries"), { recursive: true });
  await writeFile(path.join(regen, "summaries", "daily-2609.00001.json"), JSON.stringify({ schemaVersion: 1, status: "generated", item: { kind: "daily", id: "2609.00001" }, output: { kind: "structured", summary: { id: "2609.00001", coreProblem: "p", keyMethod: "m", mainResult: "再生成的核心结果：样本为130个星团。", whyRelevant: "w", limitations: "l" } } }));
  await writeFile(path.join(regen, "summaries", "note-2607.00001.json"), JSON.stringify({ schemaVersion: 1, status: "generated", item: { kind: "note", id: "2607.00001" }, output: { kind: "markdown", markdown: "# T\n\n## 核心结果\n\n再生成的笔记：样本为130个星团。\n" } }));
  const withOrigin = (kind, id, extraIssue) => { const base = record(kind, id, [extraIssue]); delete base.summary; return { ...base, origin: { kind: "regenerated", generator: { provider: "p", model: "g" }, generationPromptHash: "h".repeat(64), generatedAt: "t", summariesDir: regen } }; };
  const dir = await writeJudgeDir([withOrigin("daily", "2609.00001", issue()), withOrigin("note", "2607.00001", issue({ location: "核心结果" }))]);
  const cards = (await qe.buildReviewCards({ judgmentsDir: dir, count: 5, seed: "s" })).cards;
  assert.equal(cards.find(card => card.kind === "daily").summaryText, "再生成的核心结果：样本为130个星团。");
  assert.equal(cards.find(card => card.kind === "note").summaryText, "再生成的笔记：样本为130个星团。");

  const vault = await temp("vault");
  await mkdir(path.join(vault, "arxiv-daily", "daily"), { recursive: true });
  await mkdir(path.join(vault, "arxiv-daily", "papers"), { recursive: true });
  const labels = ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"];
  await writeFile(path.join(vault, "arxiv-daily", "daily", "2026-08-10.md"), ["---", "date: 2026-08-10", "---", "", "# t", "", "## Photo-z", "", "### P", "- **作者**: A", "- **arXiv**: [2608.00001](https://arxiv.org/abs/2608.00001)", ...labels.map(label => `- **${label}**: ${label === "核心结果" ? "来自 Vault 的核心结果：样本为130个星团。" : "x。"}`), ""].join("\n"));
  const historical = record("daily", "2608.00001", [issue()]);
  delete historical.summary;
  const vaultCard = (await qe.buildReviewCards({ judgmentsDir: await writeJudgeDir([historical]), vaultRoot: vault, count: 1, seed: "s" })).cards[0];
  assert.equal(vaultCard.summaryText, "来自 Vault 的核心结果：样本为130个星团。");
  const lost = (await qe.buildReviewCards({ judgmentsDir: await writeJudgeDir([historical]), count: 1, seed: "s" })).cards[0];
  assert.equal(lost.summaryText, null);
});

const pageOf = async (records, extraOptions = {}) => {
  const dir = await writeJudgeDir(records);
  const built = await qe.buildReviewCards({ judgmentsDir: dir, sourcesDir: await sourcesWith(records.map(item => item.item.id)), count: 10, seed: "s" });
  return { ...built, html: qe.renderReviewPage(built.cards, { pageId: built.pageId, createdAt: NOW.toISOString(), ...extraOptions }) };
};

test("the page is a self-contained Chinese document with every card, the context and the buttons", async () => {
  const { html, cards, pageId } = await pageOf(mainRecords());
  assert.match(html, /^<!doctype html>/i);
  assert.match(html, /<html lang="zh/);
  assert.match(html, /<meta name="viewport"/);
  assert.ok(!/(?:src|href)=["']https?:/i.test(html), "no external resources");
  assert.ok(!/@import|<link /i.test(html));
  assert.equal((html.match(/<script/g) ?? []).length, 1, "only the page's own script");
  for (const card of cards) {
    assert.ok(html.includes(`data-card-id="${qe.escapeHtml(card.cardId)}"`), card.cardId);
    assert.ok(html.includes(qe.escapeHtml(card.title)));
  }
  for (const text of ["同意", "不同意", "拿不准", "备注", "导出结果", "复制结果", "大问题", "小问题", "准确", "完整", "清楚", "原文未提及", "摘要原句", "原文依据", "评审说明", "历史摘要"]) assert.ok(html.includes(text), text);
  assert.ok(html.includes(pageId));
  assert.match(html, /<mark[^>]*>样本为130个星团<\/mark>/, "the quote is highlighted within the summary text");
  assert.match(html, /localStorage/);
  assert.match(html, /try\s*\{/);
});

test("quotes, titles and explanations are escaped and never become markup", async () => {
  const evil = "<script>alert(1)</script><img src=x onerror=alert(2)>&amp;";
  const base = record("daily", "2608.00001", [issue({ summaryQuote: "样本为130个星团", sourceQuote: "stack the weak lensing shear profiles of 120 clusters", explanation: evil })]);
  base.item.title = evil;
  base.summary.locations[2].text = `弱透镜 ${evil} 样本为130个星团`;
  const { html } = await pageOf([base]);
  assert.equal((html.match(/<script/g) ?? []).length, 1);
  assert.ok(!html.includes("<img src=x"));
  assert.ok(html.includes("&lt;script&gt;alert(1)&lt;/script&gt;&lt;img src=x onerror=alert(2)&gt;&amp;amp;"));
  const cards = [{ ...(await pageOf([base])).cards[0], summaryQuote: evil, sourceQuote: evil, sourceContext: `ctx ${evil}` }];
  const page = qe.renderReviewPage(cards, { pageId: "p", createdAt: NOW.toISOString() });
  assert.equal((page.match(/<script/g) ?? []).length, 1);
  assert.ok(!page.includes("onerror=alert(2)>"));
  // The card ids reach the script only as JSON with markup characters escaped.
  assert.ok(!/<\/script>[\s\S]*<\/script>/.test(page));
});

test("the page shows source context around the source quote and marks the quote", async () => {
  const { html } = await pageOf([record("daily", "2608.00001", [issue()])]);
  assert.match(html, /<mark[^>]*>stack the weak lensing shear profiles of 120 clusters<\/mark>/);
  assert.ok(html.includes("填充文字"));
});

const results = (pageId, decisions) => ({ schemaVersion: 1, pageId, results: decisions.map(([cardId, decision, note = ""]) => ({ cardId, decision, note })) });
async function importFixture() {
  const main = await writeJudgeDir(mainRecords());
  const extra = await writeJudgeDir(extraRecords());
  const out = await temp("review-out");
  const result = await qe.runSummaryReview({ judgmentsDir: main, extraDir: extra, count: 10, seed: "s", outDir: out }, { now: () => NOW });
  return { out, result, manifest: JSON.parse(await readFile(path.join(out, "review.json"), "utf8")) };
}

test("generating the page writes review.html and the manifest of its cards", async () => {
  const { out, result, manifest } = await importFixture();
  assert.deepEqual(result.files.map(file => path.basename(file)), ["review.html", "review.json"]);
  assert.equal(manifest.cards.length, 10);
  assert.equal(manifest.pageId, result.pageId);
  assert.match(await readFile(path.join(out, "review.html"), "utf8"), /^<!doctype html>/i);
  await assert.rejects(qe.runSummaryReview({ judgmentsDir: await writeJudgeDir([]), count: 3, seed: "s", outDir: await temp("o") }, { now: () => NOW }), /no (issues|judgments)/i);
});

test("importing results computes the agreement overall, by dimension, severity and origin", async () => {
  const { out, manifest } = await importFixture();
  const cards = manifest.cards;
  const pick = predicate => cards.filter(predicate).map(card => card.cardId);
  const answers = [];
  const decide = (card, decision, note) => answers.push([card.cardId, decision, note]);
  const accuracy = cards.filter(card => card.dimension === "accuracy");
  accuracy.forEach((card, index) => decide(card, index === 0 ? "disagree" : "agree", index === 0 ? "评审看错了" : ""));
  const others = cards.filter(card => card.dimension !== "accuracy");
  others.forEach((card, index) => decide(card, index === 0 ? "unsure" : index === 1 ? "disagree" : "agree"));
  const dropped = answers.pop();   // one card left unanswered
  const resultFile = path.join(await temp("results"), "r.json");
  await writeFile(resultFile, JSON.stringify(results(manifest.pageId, answers)));
  const { calibration } = await qe.runReviewImport({ resultsPath: resultFile, outDir: out }, { now: () => NOW });

  const count = (decision, predicate = () => true) => answers.filter(([id, d]) => d === decision && predicate(cards.find(card => card.cardId === id))).length;
  const agree = count("agree"); const disagree = count("disagree"); const unsure = count("unsure");
  assert.equal(calibration.total, 10);
  assert.equal(calibration.answered, answers.length);
  assert.equal(calibration.unanswered, 1);
  assert.equal(calibration.overall.agree, agree);
  assert.equal(calibration.overall.disagree, disagree);
  assert.equal(calibration.overall.unsure, unsure);
  assert.equal(calibration.overall.rate, agree / (agree + disagree));
  for (const dimension of ["accuracy", "completeness", "clarity"]) {
    const slice = calibration.byDimension[dimension];
    assert.equal(slice.agree, count("agree", card => card.dimension === dimension), dimension);
    assert.equal(slice.rate, slice.agree + slice.disagree === 0 ? null : slice.agree / (slice.agree + slice.disagree));
  }
  assert.equal(calibration.bySeverity.major.agree, count("agree", card => card.severity === "major"));
  assert.equal(calibration.byOrigin.regenerated.agree, count("agree", card => card.origin.kind === "regenerated"));
  assert.equal(calibration.byOrigin.historical.disagree, count("disagree", card => card.origin.kind === "historical"));
  assert.ok(calibration.disagreements.some(entry => entry.note === "评审看错了"));
  assert.equal(calibration.disagreements.length, disagree);
  void pick; void dropped;

  const md = await readFile(path.join(out, "calibration.md"), "utf8");
  assert.match(md, /^# /);
  assert.ok(md.includes("评审看错了"));
  assert.ok(md.includes(`${(calibration.overall.rate * 100).toFixed(1)}%`));
  for (const text of ["一致率", "拿不准", "按维度", "按严重程度", "按来源", "不同意"]) assert.ok(md.includes(text), text);
  assert.ok(existsSync(path.join(out, "calibration.json")));
});

test("an agreement rate with no agree or disagree answers is not a number", async () => {
  const { out, manifest } = await importFixture();
  const file = path.join(await temp("results"), "r.json");
  await writeFile(file, JSON.stringify(results(manifest.pageId, [[manifest.cards[0].cardId, "unsure"]])));
  const { calibration } = await qe.runReviewImport({ resultsPath: file, outDir: out }, { now: () => NOW });
  assert.equal(calibration.overall.rate, null);
  assert.match(await readFile(path.join(out, "calibration.md"), "utf8"), /n\/a/);
});

test("results that do not belong to the page are rejected", async () => {
  const { out, manifest } = await importFixture();
  const write = async value => { const file = path.join(await temp("results"), "r.json"); await writeFile(file, typeof value === "string" ? value : JSON.stringify(value)); return file; };
  const first = manifest.cards[0].cardId;
  await assert.rejects(qe.runReviewImport({ resultsPath: await write(results(manifest.pageId, [["nope#9@j", "agree"]])), outDir: out }), /nope#9@j/);
  await assert.rejects(qe.runReviewImport({ resultsPath: await write(results("another-page", [[first, "agree"]])), outDir: out }), /another-page|pageId|页面/);
  await assert.rejects(qe.runReviewImport({ resultsPath: await write(results(manifest.pageId, [[first, "maybe"]])), outDir: out }), /maybe/);
  await assert.rejects(qe.runReviewImport({ resultsPath: await write(results(manifest.pageId, [[first, "agree"], [first, "disagree"]])), outDir: out }), /twice|duplicate|重复/i);
  await assert.rejects(qe.runReviewImport({ resultsPath: await write("{not json"), outDir: out }), /JSON/);
  await assert.rejects(qe.runReviewImport({ resultsPath: await write({ results: "x" }), outDir: out }), /results/);
  await assert.rejects(qe.runReviewImport({ resultsPath: await write(results(manifest.pageId, [])), outDir: await temp("no-page") }), /review\.json/);
  assert.equal(existsSync(path.join(out, "calibration.md")), false);
});

test("command-line arguments are validated", () => {
  assert.deepEqual(
    qe.parseSummaryReviewArgs(["--judgments", "/j", "--extra", "/x", "--sources", "/s", "--vault", "/v", "--daily-dir", "d", "--papers-dir", "p", "--count", "12", "--seed", "z", "--out", "/o"]),
    { judgmentsDir: "/j", extraDir: "/x", sourcesDir: "/s", vaultRoot: "/v", dailyDir: "d", papersDir: "p", count: 12, seed: "z", outDir: "/o" },
  );
  assert.deepEqual(qe.parseSummaryReviewArgs(["--import", "/r.json", "--out", "/o"]), { importPath: "/r.json", outDir: "/o" });
  assert.equal(qe.parseSummaryReviewArgs(["--help"]).help, true);
  assert.throws(() => qe.parseSummaryReviewArgs(["--out", "/o"]), /--judgments/);
  assert.throws(() => qe.parseSummaryReviewArgs(["--judgments", "/j"]), /--out/);
  assert.throws(() => qe.parseSummaryReviewArgs(["--judgments", "/j", "--out", "/o", "--count", "0"]), /--count/);
  assert.throws(() => qe.parseSummaryReviewArgs(["--judgments", "/j", "--out", "/o", "--bogus"]), /unknown option.*--bogus/i);
  assert.throws(() => qe.parseSummaryReviewArgs(["--import", "/r.json", "--judgments", "/j", "--out", "/o"]), /--import/);
});

test("the script prints usage on bad arguments", async () => {
  const { spawnSync } = await import("node:child_process");
  const script = repoFile("scripts/quality-eval/summary-review.mjs");
  const bad = spawnSync(process.execPath, [script, "--out", "/o"], { encoding: "utf8" });
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /Usage: node scripts\/quality-eval\/summary-review\.mjs/);
  assert.equal(spawnSync(process.execPath, [script, "--help"], { encoding: "utf8" }).status, 0);
});

test("daily field keys are shown by their Chinese labels, note headings as they are", async () => {
  assert.equal(qe.locationLabel("daily", "coreProblem"), "研究问题");
  assert.deepEqual(qe.SUMMARY_FIELDS.map(key => qe.locationLabel("daily", key)), ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"]);
  assert.equal(qe.locationLabel("note", "核心结果"), "核心结果");
  assert.equal(qe.locationLabel("note", "mainResult"), "mainResult");
  const { html, cards } = await pageOf(mainRecords());
  const daily = cards.find(card => card.location === "mainResult");
  assert.ok(html.includes("位置：核心结果"));
  assert.ok(!html.includes("位置：mainResult") && !html.includes("位置：limitations"));
  assert.equal(daily.location, "mainResult", "the data keeps the raw key");
  const note = cards.find(card => card.kind === "note");
  assert.ok(html.includes(`位置：${note.location}`));
});

test("the calibration report lists disagreements by Chinese field label", async () => {
  const { out, manifest } = await importFixture();
  const card = manifest.cards.find(candidate => candidate.kind === "daily" && candidate.location === "mainResult");
  const file = path.join(await temp("results"), "r.json");
  await writeFile(file, JSON.stringify(results(manifest.pageId, [[card.cardId, "disagree", "备注"]])));
  await qe.runReviewImport({ resultsPath: file, outDir: out }, { now: () => NOW });
  const md = await readFile(path.join(out, "calibration.md"), "utf8");
  assert.ok(md.includes("核心结果") && !md.includes("mainResult"));
});
