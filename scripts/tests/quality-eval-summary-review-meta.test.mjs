import assert from "node:assert/strict";
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
const NOW = new Date("2026-10-30T10:00:00Z");
const SOURCE_TEXT = `## Methods\n${"填充文字 ".repeat(5)}We stack the weak lensing shear profiles of 120 clusters and calibrate against simulations. ${"padding words ".repeat(60)}\n## Results\nThe mass ratio is $0.85 \\pm 0.04$ for the full sample.`;

const issue = (overrides = {}) => ({
  dimension: "accuracy", severity: "major", location: "mainResult", summaryQuote: "样本为130个星团", sourceQuote: "stack the weak lensing shear profiles of 120 clusters",
  explanation: "摘要把样本数写成 130，原文是 120。", ...overrides,
});
const judgeRecord = (kind, id, issues) => ({
  schemaVersion: 1, status: "judged", invalid: false,
  item: { kind, id, title: `Title of ${id}`, date: "2026-08-10", topic: "Photo-z", source: { sourcePath: "html", arxivVersion: "v1", fetchedAt: "2026-10-28T10:00:00.000Z", contentHash: `hash-${id}` } },
  judge: { provider: "openai", model: "judge-model" }, judgePromptHash: "p".repeat(64), attempts: [],
  judgment: { verdicts: { accuracy: "ok", completeness: "ok", clarity: "ok" }, issues },
  summary: { locations: kind === "daily"
    ? qe.SUMMARY_FIELDS.map(key => ({ key, text: key === "mainResult" ? "弱透镜质量比为 0.85，样本为130个星团，误差较小。" : `${id} 的 ${key} 内容。` }))
    : [{ key: "核心结果", text: "第一段。\n\n样本为130个星团。\n\n第三段。" }] },
});
const JUDGE_RECORDS = () => [
  judgeRecord("daily", "2608.00001", [issue(), issue({ dimension: "completeness", severity: "minor", location: "limitations", summaryQuote: "2608.00001 的 limitations 内容", sourceQuote: "padding words padding words" })]),
  judgeRecord("daily", "2608.00002", [issue({ dimension: "clarity", severity: "major", sourceQuote: null, location: "keyMethod", summaryQuote: "2608.00002 的 keyMethod 内容", explanation: "缩写未解释。" })]),
  judgeRecord("daily", "2608.00003", [issue({ severity: "minor", sourceQuote: null, explanation: "原文未提及这个说法。" }), issue({ dimension: "completeness", severity: "major", location: "whyRelevant", summaryQuote: "2608.00003 的 whyRelevant 内容", sourceQuote: "The mass ratio is $0.85 \\pm 0.04$" })]),
  judgeRecord("daily", "2608.00004", [issue({ severity: "minor" }), issue({ dimension: "clarity", severity: "minor", location: "coreProblem", summaryQuote: "2608.00004 的 coreProblem 内容", sourceQuote: null })]),
  judgeRecord("note", "2607.00001", [issue({ location: "核心结果", summaryQuote: "样本为130个星团", severity: "major" })]),
];
// Reviewer verdicts by "id#index": four disputes, four agreed.
const VERDICTS = {
  "2608.00001#0": ["disagree", "too-high"], "2608.00001#1": ["agree", "ok"], "2608.00002#0": ["unsure", "ok"],
  "2608.00003#0": ["agree", "ok"], "2608.00003#1": ["disagree", "ok"], "2608.00004#0": ["agree", "too-low"], "2608.00004#1": ["agree", "ok"], "2607.00001#0": ["unsure", "ok"],
};

async function fixture() {
  const judged = await temp("judged");
  await mkdir(path.join(judged, "judgments"), { recursive: true });
  await writeFile(path.join(judged, "run.json"), JSON.stringify({ schemaVersion: 1, startedAt: "x", seed: "s", sample: { daily: [], notes: [] }, generatorModel: "gen-model" }));
  const records = JUDGE_RECORDS();
  for (const record of records) await writeFile(path.join(judged, "judgments", `${record.item.kind}-${record.item.id}.json`), JSON.stringify(record));
  const sources = await temp("sources");
  for (const record of records) {
    await writeFile(path.join(sources, `${record.item.id}.json`), JSON.stringify({ schemaVersion: 1, id: record.item.id, fetchedAt: "x", sourcePath: "html", arxivVersion: "v1", budgets: {}, content: { abstract: "a", abstractConclusion: "## Abstract\nAbstract.", fullSections: SOURCE_TEXT, fullTextFailure: null }, contentHash: `hash-${record.item.id}` }));
  }
  const meta = await temp("meta");
  await mkdir(path.join(meta, "meta"), { recursive: true });
  const metaRecords = records.map(record => ({
    schemaVersion: 1, status: "reviewed",
    item: { ...record.item, origin: { kind: "historical", vault: "/vault", date: record.item.date } },
    judge: record.judge, issues: record.judgment.issues,
    reviewer: { provider: "openai", model: "reviewer-model", thinkingMode: true, reasoningEffort: "xhigh" }, reviewerPromptHash: "r".repeat(64), judgmentHash: "j".repeat(64), attempts: [],
    reviews: record.judgment.issues.map((_, index) => {
      const [verdict, severity] = VERDICTS[`${record.item.id}#${index}`];
      return { index, verdict, severity, reason: verdict === "agree" ? "问题确实存在。" : `复核认为${verdict === "disagree" ? "摘要没有问题" : "原文信息不足"}：<b>标记</b>。`, evidenceQuote: verdict === "disagree" ? "stack the weak lensing shear profiles of 120 clusters" : null };
    }),
    startedAt: "x", finishedAt: "x", elapsedMs: 1,
  }));
  for (const record of metaRecords) await writeFile(path.join(meta, "meta", `${record.item.kind}-${record.item.id}.json`), JSON.stringify(record));
  const run = { schemaVersion: 1, startedAt: "x", judged: { dir: judged, judgeModels: ["judge-model"], generatorModel: "gen-model" }, reviewer: metaRecords[0].reviewer };
  await writeFile(path.join(meta, "run.json"), JSON.stringify(run));
  const report = qe.buildMetaReport(run, metaRecords, NOW);
  await writeFile(path.join(meta, "meta-review.json"), JSON.stringify(report));
  return { judged, sources, meta, metaRecords };
}
const build = async (state, options = {}) => qe.buildReviewCards({ metaDir: state.meta, sourcesDir: state.sources, seed: "s1", ...options });
const ids = cards => cards.map(card => card.cardId);

test("a dispute page holds every issue the reviewer disagreed with or was unsure about, then a small agreed sample", async () => {
  const state = await fixture();
  const { cards } = await build(state);
  const disputes = cards.filter(card => card.meta.group === "dispute");
  assert.deepEqual(disputes.map(card => `${card.id}#${card.cardId.split("#")[1].split("@")[0]}`).sort(), ["2607.00001#0", "2608.00001#0", "2608.00002#0", "2608.00003#1"]);
  assert.deepEqual(disputes.map(card => card.meta.verdict).sort(), ["disagree", "disagree", "unsure", "unsure"]);
  const sample = cards.filter(card => card.meta.group === "agreed-sample");
  assert.equal(sample.length, 3, "the default agreed sample is 3");
  assert.ok(sample.every(card => card.meta.verdict === "agree"));
  assert.deepEqual(cards.map(card => card.meta.group), [...disputes, ...sample].map(card => card.meta.group), "disputes come first");
  assert.ok(cards.every(card => card.pool === "meta"));
});

test("the agreed sample is seeded, can be sized or switched off, and --count still caps the page", async () => {
  const state = await fixture();
  const a = (await build(state)).cards.filter(card => card.meta.group === "agreed-sample");
  const b = (await build(state)).cards.filter(card => card.meta.group === "agreed-sample");
  assert.deepEqual(ids(a), ids(b));
  const others = new Set();
  for (const seed of ["a", "b", "c", "d", "e", "f"]) others.add(ids((await build(state, { seed })).cards.filter(card => card.meta.group === "agreed-sample")).join());
  assert.ok(others.size > 1, "the seed changes the sample");
  assert.equal((await build(state, { agreedSample: 0 })).cards.length, 4);
  assert.equal((await build(state, { agreedSample: 1 })).cards.length, 5);
  assert.equal((await build(state, { agreedSample: 99 })).cards.length, 8, "no more than the agreed issues there are");
  const capped = await build(state, { count: 3 });
  assert.equal(capped.cards.length, 3);
  assert.ok(capped.cards.every(card => card.meta.group === "dispute"));
});

test("a card carries the reviewer's verdict, severity opinion, reason and evidence with source context", async () => {
  const state = await fixture();
  const { cards } = await build(state, { agreedSample: 0 });
  const card = cards.find(candidate => candidate.id === "2608.00001");
  assert.equal(card.meta.verdict, "disagree");
  assert.equal(card.meta.severityOpinion, "too-high");
  assert.ok(card.meta.reason.includes("摘要没有问题"));
  assert.equal(card.meta.evidenceQuote, "stack the weak lensing shear profiles of 120 clusters");
  assert.ok(card.meta.evidenceContext.includes("填充文字"));
  assert.equal(card.explanation, "摘要把样本数写成 130，原文是 120。");
  assert.equal(card.summaryText, "弱透镜质量比为 0.85，样本为130个星团，误差较小。", "the summary text comes from the judgment");
  const unsure = cards.find(candidate => candidate.id === "2608.00002");
  assert.equal(unsure.meta.evidenceQuote, null);
  assert.equal(unsure.meta.evidenceContext, null);
});

const pageFor = async (state, options = {}) => {
  const built = await build(state, options);
  return { ...built, html: qe.renderReviewPage(built.cards, { pageId: built.pageId, createdAt: NOW.toISOString() }) };
};

test("the page labels dispute buttons by whom the researcher sides with, and shows both reasons", async () => {
  const state = await fixture();
  const { html, cards } = await pageFor(state);
  for (const text of ["评审对", "复核对", "拿不准", "复核分歧", "复核同意的抽查", "评审说明", "复核理由", "复核的裁定", "严重程度意见", "偏重"]) assert.ok(html.includes(text), text);
  const dispute = html.slice(html.indexOf(`data-card-id="${qe.escapeHtml(cards.find(card => card.meta.group === "dispute").cardId)}"`));
  assert.match(dispute.slice(0, dispute.indexOf("</article>")), /data-decision="judge"[^>]*>评审对<[\s\S]*data-decision="reviewer"[^>]*>复核对<[\s\S]*data-decision="unsure"[^>]*>拿不准</);
  const agreed = html.slice(html.indexOf(`data-card-id="${qe.escapeHtml(cards.find(card => card.meta.group === "agreed-sample").cardId)}"`));
  assert.match(agreed.slice(0, agreed.indexOf("</article>")), /data-decision="agree"[^>]*>同意<[\s\S]*data-decision="disagree"[^>]*>不同意<[\s\S]*data-decision="unsure"[^>]*>拿不准</);
  assert.ok(html.includes("斟酌后"), "the intro explains the dispute question");
});

test("the dispute page still escapes the reviewer's text", async () => {
  const state = await fixture();
  const { html } = await pageFor(state);
  assert.ok(html.includes("&lt;b&gt;标记&lt;/b&gt;"));
  assert.ok(!html.includes("<b>标记</b>"));
  assert.equal((html.match(/<script/g) ?? []).length, 1);
  const cards = (await build(state)).cards.map(card => ({ ...card, meta: { ...card.meta, reason: "<script>alert(1)</script>", evidenceQuote: "<img src=x onerror=alert(2)>", evidenceContext: "<img src=x onerror=alert(3)>" } }));
  const page = qe.renderReviewPage(cards, { pageId: "p", createdAt: NOW.toISOString() });
  assert.equal((page.match(/<script/g) ?? []).length, 1);
  assert.ok(!page.includes("<img src=x"));
});

const answers = (pageId, decisions) => ({ schemaVersion: 1, pageId, results: decisions.map(([cardId, decision, note = ""]) => ({ cardId, decision, note })) });
async function generated(options = {}) {
  const state = await fixture();
  const out = await temp("review-out");
  await qe.runSummaryReview({ metaDir: state.meta, sourcesDir: state.sources, seed: "s1", outDir: out, ...options }, { now: () => NOW });
  return { ...state, out, manifest: JSON.parse(await readFile(path.join(out, "review.json"), "utf8")) };
}
const importResults = async (state, decisions) => {
  const file = path.join(await temp("results"), "r.json");
  await writeFile(file, JSON.stringify(answers(state.manifest.pageId, decisions)));
  return qe.runReviewImport({ resultsPath: file, outDir: state.out }, { now: () => NOW });
};

test("a dispute card accepts judge / reviewer / unsure, an agreed-sample card agree / disagree / unsure, and not each other's", async () => {
  const state = await generated();
  const dispute = state.manifest.cards.find(card => card.meta.group === "dispute");
  const agreed = state.manifest.cards.find(card => card.meta.group === "agreed-sample");
  await assert.rejects(importResults(state, [[dispute.cardId, "agree"]]), /agree/);
  await assert.rejects(importResults(state, [[agreed.cardId, "judge"]]), /judge/);
  const ok = await importResults(state, [[dispute.cardId, "reviewer"], [agreed.cardId, "agree"]]);
  assert.equal(ok.calibration.answered, 2);
});

test("the merged calibration combines reviewer agreement, the disputes and the agreed sample into a precision estimate", async () => {
  const state = await generated();
  const cards = state.manifest.cards;
  const dispute = id => cards.find(card => card.id === id && card.meta.group === "dispute");
  const sample = cards.filter(card => card.meta.group === "agreed-sample");
  const decisions = [
    [dispute("2608.00001").cardId, "judge", "评审是对的"],      // accuracy major
    [dispute("2608.00002").cardId, "reviewer", "复核更准"],     // clarity major
    [dispute("2608.00003").cardId, "unsure"],                    // completeness major
    // the note dispute is left unanswered
    [sample[0].cardId, "agree"], [sample[1].cardId, "agree"], [sample[2].cardId, "disagree", "其实不对"],
  ];
  const { calibration, files } = await importResults(state, decisions);
  const m = calibration.meta;
  assert.deepEqual(m.reviewer.overall, { agree: 4, disagree: 2, unsure: 2, rate: 4 / 6 });
  assert.equal(m.issues, 8);
  assert.deepEqual(m.disputes.overall, { judge: 1, reviewer: 1, unsure: 1, unanswered: 1, total: 4 });
  assert.deepEqual(m.disputes.byDimension.accuracy, { judge: 1, reviewer: 0, unsure: 0, unanswered: 1, total: 2 });
  assert.deepEqual(m.disputes.byDimension.clarity, { judge: 0, reviewer: 1, unsure: 0, unanswered: 0, total: 1 });
  assert.deepEqual(m.disputes.byDimension.completeness, { judge: 0, reviewer: 0, unsure: 1, unanswered: 0, total: 1 });
  assert.deepEqual(m.disputes.bySeverity.major, { judge: 1, reviewer: 1, unsure: 1, unanswered: 1, total: 4 });
  assert.deepEqual(m.disputes.bySeverity.minor, { judge: 0, reviewer: 0, unsure: 0, unanswered: 0, total: 0 });
  assert.deepEqual(m.agreedSample, { agree: 2, disagree: 1, unsure: 0, unanswered: 0, rate: 2 / 3, total: 3 });
  // (4 reviewer-agreed x 2/3 + 1 judge-sided dispute) / 8 issues
  const point = (4 * (2 / 3) + 1) / 8;
  assert.ok(Math.abs(m.precisionEstimate.point - point) < 1e-9, JSON.stringify(m.precisionEstimate));
  assert.deepEqual({ agreed: m.precisionEstimate.components.reviewerAgreed, rate: m.precisionEstimate.components.agreedSampleRate, judgeSided: m.precisionEstimate.components.judgeSidedDisputes, issues: m.precisionEstimate.components.issues }, { agreed: 4, rate: 2 / 3, judgeSided: 1, issues: 8 });
  assert.ok(Math.abs(m.precisionEstimate.upper - (4 * (2 / 3) + 3) / 8) < 1e-9, "unsure and unanswered disputes counted for the judge");
  assert.equal(m.disagreedWithJudge.length, 2);
  assert.deepEqual(m.disagreedWithJudge.map(entry => entry.note).sort(), ["其实不对", "复核更准"]);

  const md = await readFile(files[0], "utf8");
  for (const text of ["复核一致率", "评审对", "复核对", "拿不准", "未回答", "评审精确率估计", "估计", `${(point * 100).toFixed(1)}%`, "复核同意的抽查", "按维度", "按严重程度", "复核更准", "其实不对"]) assert.ok(md.includes(text), text);
  assert.ok(!md.includes("mainResult"));
});

test("the estimate is not a number when the agreed sample was not answered", async () => {
  const state = await generated();
  const dispute = state.manifest.cards.find(card => card.meta.group === "dispute");
  const { calibration, files } = await importResults(state, [[dispute.cardId, "judge"]]);
  assert.equal(calibration.meta.agreedSample.rate, null);
  assert.equal(calibration.meta.precisionEstimate.point, null);
  assert.match(await readFile(files[0], "utf8"), /n\/a/);
});

test("without an agreed sample the estimate says it has no spot-check behind it", async () => {
  const state = await generated({ agreedSample: 0 });
  const dispute = state.manifest.cards.find(card => card.meta.group === "dispute");
  const { calibration } = await importResults(state, [[dispute.cardId, "judge"]]);
  assert.equal(calibration.meta.precisionEstimate.point, null);
  assert.equal(calibration.meta.agreedSample.total, 0);
});

test("a plain review page (no --meta) still imports as before", async () => {
  const state = await fixture();
  const out = await temp("plain");
  await qe.runSummaryReview({ judgmentsDir: state.judged, sourcesDir: state.sources, count: 4, seed: "s", outDir: out }, { now: () => NOW });
  const manifest = JSON.parse(await readFile(path.join(out, "review.json"), "utf8"));
  const file = path.join(await temp("results"), "r.json");
  await writeFile(file, JSON.stringify(answers(manifest.pageId, [[manifest.cards[0].cardId, "agree"]])));
  const { calibration } = await qe.runReviewImport({ resultsPath: file, outDir: out }, { now: () => NOW });
  assert.equal(calibration.meta, undefined);
  assert.equal(calibration.overall.agree, 1);
});

test("command-line arguments accept --meta and --agreed-sample", () => {
  assert.deepEqual(qe.parseSummaryReviewArgs(["--meta", "/m", "--sources", "/s", "--agreed-sample", "5", "--out", "/o"]), { metaDir: "/m", sourcesDir: "/s", agreedSample: 5, outDir: "/o" });
  assert.deepEqual(qe.parseSummaryReviewArgs(["--meta", "/m", "--agreed-sample", "0", "--out", "/o"]), { metaDir: "/m", agreedSample: 0, outDir: "/o" });
  assert.throws(() => qe.parseSummaryReviewArgs(["--meta", "/m", "--agreed-sample", "x", "--out", "/o"]), /--agreed-sample/);
  assert.throws(() => qe.parseSummaryReviewArgs(["--meta", "/m", "--judgments", "/j", "--extra", "/x", "--out", "/o"]), /--meta/);
  assert.throws(() => qe.parseSummaryReviewArgs(["--agreed-sample", "2", "--judgments", "/j", "--out", "/o"]), /--agreed-sample/);
});
