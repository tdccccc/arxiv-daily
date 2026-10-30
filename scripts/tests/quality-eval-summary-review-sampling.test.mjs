import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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
const NOW = new Date("2026-10-30T12:00:00Z");
const LOCATION = { accuracy: "mainResult", completeness: "limitations", clarity: "keyMethod" };

/**
 * 20 disputes: A1-A3 accuracy (minor); M1-M2 completeness (major); C1-C5 completeness minor, two issues each;
 * L1-L5 clarity minor. Plus 6 agreed issues on G1-G3. Reviewer verdict: disagree, except M1-M2 unsure.
 */
const SPEC = [
  ...["A1", "A2", "A3"].map(paper => ({ paper, issues: [["accuracy", "minor", "disagree"]] })),
  ...["M1", "M2"].map(paper => ({ paper, issues: [["completeness", "major", "unsure"]] })),
  ...["C1", "C2", "C3", "C4", "C5"].map(paper => ({ paper, issues: [["completeness", "minor", "disagree"], ["completeness", "minor", "disagree"]] })),
  ...["L1", "L2", "L3", "L4", "L5"].map(paper => ({ paper, issues: [["clarity", "minor", "disagree"]] })),
  ...["G1", "G2", "G3"].map(paper => ({ paper, issues: [["completeness", "minor", "agree"], ["clarity", "minor", "agree"]] })),
];
const idOf = paper => `2611.${String(SPEC.findIndex(entry => entry.paper === paper) + 1).padStart(5, "0")}`;
const paperOf = id => SPEC[Number(id.slice(5)) - 1].paper;
const hash = text => createHash("sha256").update(text).digest("hex");

async function fixture() {
  const judged = await temp("judged");
  await mkdir(path.join(judged, "judgments"), { recursive: true });
  const sample = { daily: SPEC.map(entry => idOf(entry.paper)), notes: [] };
  const judge = { provider: "openai", model: "judge-model" };
  const judgeManifest = [];
  const meta = await temp("meta");
  await mkdir(path.join(meta, "meta"), { recursive: true });
  const metaRecords = [];
  for (const entry of SPEC) {
    const id = idOf(entry.paper);
    const issues = entry.issues.map(([dimension, severity]) => ({ dimension, severity, location: LOCATION[dimension], summaryQuote: "某句话", sourceQuote: null, explanation: `${entry.paper} 的说明，原文未提及。` }));
    const item = { kind: "daily", id, title: `Title ${entry.paper}`, date: "2026-08-10", topic: "Photo-z", source: { sourcePath: "html", arxivVersion: "v1", fetchedAt: "x", contentHash: `h-${id}` } };
    const judgeRecord = { schemaVersion: 1, status: "judged", invalid: false, item, judge, judgePromptHash: "p".repeat(64), attempts: [], judgment: { verdicts: { accuracy: "ok", completeness: "ok", clarity: "ok" }, issues }, summary: { locations: Object.values(LOCATION).map(key => ({ key, text: `某句话 在 ${entry.paper} 里。` })) } };
    judgeRecord.summaryHash = hash(judgeRecord.summary.locations.map(location => `[${location.key}] ${location.text}`).join("\n"));
    const serializedJudge = JSON.stringify(judgeRecord);
    await writeFile(path.join(judged, "judgments", `daily-${id}.json`), serializedJudge);
    judgeManifest.push({ kind: "daily", id, recordHash: hash(serializedJudge) });
    metaRecords.push({
      schemaVersion: 1, status: "reviewed", item: { ...item, origin: { kind: "historical", vault: "/v", date: item.date } }, judge: judgeRecord.judge, issues,
      reviewer: { provider: "openai", model: "reviewer-model", thinkingMode: true, reasoningEffort: "xhigh" }, reviewerPromptHash: "r".repeat(64), judgmentHash: hash(JSON.stringify({ summaryHash: judgeRecord.summaryHash, source: item.source.contentHash, issues })), attempts: [],
      reviews: entry.issues.map(([, , verdict], index) => ({ index, verdict, severity: "ok", reason: "复核的理由。", evidenceQuote: null })), startedAt: "x", finishedAt: "x", elapsedMs: 1,
    });
  }
  await writeFile(path.join(judged, "run.json"), JSON.stringify({ schemaVersion: 1, startedAt: "x", seed: "s", sample, judge, skipped: [], resultManifest: judgeManifest, generatorModel: "gen" }));
  const resultManifest = [];
  for (const record of metaRecords) {
    const serialized = JSON.stringify(record);
    await writeFile(path.join(meta, "meta", `daily-${record.item.id}.json`), serialized);
    resultManifest.push({ kind: "daily", id: record.item.id, recordHash: hash(serialized) });
  }
  const run = { schemaVersion: 1, startedAt: "x", sample, skipped: [], resultManifest, judged: { dir: judged, judgeModels: ["judge-model"], generatorModel: "gen" }, reviewer: metaRecords[0].reviewer };
  await writeFile(path.join(meta, "run.json"), JSON.stringify(run));
  await writeFile(path.join(meta, "meta-review.json"), JSON.stringify(qe.buildMetaReport(run, metaRecords, NOW)));
  return { meta };
}
const cardsOf = async (state, options = {}) => (await qe.buildReviewCards({ metaDir: state.meta, seed: "s1", agreedSample: 0, ...options })).cards;
const disputesOf = cards => cards.filter(card => card.meta.group === "dispute");
const key = card => `${paperOf(card.id)}/${card.dimension}/${card.severity}`;

test("with a dispute sample, every accuracy dispute and every dispute on a judge-major issue is included", async () => {
  const state = await fixture();
  const cards = await cardsOf(state, { disputeSample: 11 });
  const papers = disputesOf(cards).map(card => paperOf(card.id));
  for (const must of ["A1", "A2", "A3", "M1", "M2"]) assert.ok(papers.includes(must), must);
  assert.equal(cards.length, 11);
  // Even a sample smaller than the must-include set keeps all of it.
  const small = disputesOf(await cardsOf(state, { disputeSample: 3 }));
  assert.deepEqual(small.map(card => paperOf(card.id)).sort(), ["A1", "A2", "A3", "M1", "M2"]);
});

test("the rest is filled in proportion to each dimension's disputes, by largest remainder", async () => {
  const state = await fixture();
  const split = async n => {
    const cards = disputesOf(await cardsOf(state, { disputeSample: n })).filter(card => card.meta.stratum !== "must");
    return { completeness: cards.filter(card => card.dimension === "completeness").length, clarity: cards.filter(card => card.dimension === "clarity").length };
  };
  assert.deepEqual(await split(11), { completeness: 4, clarity: 2 });
  assert.deepEqual(await split(8), { completeness: 2, clarity: 1 });
  assert.deepEqual(await split(9), { completeness: 3, clarity: 1 }, "2.67 and 1.33: the larger remainder gets the extra one");
  assert.deepEqual(await split(14), { completeness: 6, clarity: 3 });
});

test("within a dimension papers not yet used are preferred", async () => {
  const state = await fixture();
  const picked = disputesOf(await cardsOf(state, { disputeSample: 11 })).filter(card => card.dimension === "completeness" && card.severity === "minor");
  assert.equal(picked.length, 4);
  assert.equal(new Set(picked.map(card => paperOf(card.id))).size, 4, "four completeness picks come from four papers, not twice from one");
});

test("the sample is deterministic by seed", async () => {
  const state = await fixture();
  const ids = async seed => (await cardsOf(state, { disputeSample: 11, seed })).map(card => card.cardId);
  assert.deepEqual(await ids("a"), await ids("a"));
  const variants = new Set();
  for (const seed of ["a", "b", "c", "d", "e"]) variants.add((await ids(seed)).join());
  assert.ok(variants.size > 1);
});

test("a sample at least as large as the disputes, or none, takes every dispute and records no plan", async () => {
  const state = await fixture();
  assert.equal(disputesOf(await cardsOf(state, { disputeSample: 20 })).length, 20);
  assert.equal(disputesOf(await cardsOf(state, { disputeSample: 99 })).length, 20);
  assert.equal(disputesOf(await cardsOf(state)).length, 20);
  const built = await qe.buildReviewCards({ metaDir: state.meta, seed: "s", agreedSample: 0, disputeSample: 99 });
  assert.equal(built.disputeSampling, null);
  assert.ok(built.cards.every(card => card.meta.stratum === undefined));
});

test("the sampling plan lists each stratum's population and sampled count", async () => {
  const state = await fixture();
  const built = await qe.buildReviewCards({ metaDir: state.meta, seed: "s", agreedSample: 0, disputeSample: 11 });
  const plan = built.disputeSampling;
  assert.equal(plan.requested, 11);
  assert.equal(plan.population, 20);
  assert.deepEqual(plan.strata.map(stratum => [stratum.key, stratum.population, stratum.sampled]), [["must", 5, 5], ["completeness", 10, 4], ["clarity", 5, 2]]);
  assert.ok(plan.strata.every(stratum => typeof stratum.label === "string" && stratum.label.length > 0));
  for (const stratum of plan.strata) assert.equal(built.cards.filter(card => card.meta.stratum === stratum.key).length, stratum.sampled);
  const out = await temp("out");
  await qe.runSummaryReview({ metaDir: state.meta, seed: "s", agreedSample: 0, disputeSample: 11, outDir: out }, { now: () => NOW });
  const manifest = JSON.parse(await readFile(path.join(out, "review.json"), "utf8"));
  assert.deepEqual(manifest.disputeSampling, plan);
});

test("--count and --dispute-sample cannot be combined, and --dispute-sample needs --meta", () => {
  assert.deepEqual(qe.parseSummaryReviewArgs(["--meta", "/m", "--dispute-sample", "12", "--agreed-sample", "3", "--seed", "z", "--out", "/o"]), { metaDir: "/m", disputeSample: 12, agreedSample: 3, seed: "z", outDir: "/o" });
  assert.throws(() => qe.parseSummaryReviewArgs(["--meta", "/m", "--dispute-sample", "0", "--out", "/o"]), /--dispute-sample/);
  assert.throws(() => qe.parseSummaryReviewArgs(["--meta", "/m", "--dispute-sample", "x", "--out", "/o"]), /--dispute-sample/);
  assert.throws(() => qe.parseSummaryReviewArgs(["--meta", "/m", "--dispute-sample", "5", "--count", "5", "--out", "/o"]), /--count/);
  assert.throws(() => qe.parseSummaryReviewArgs(["--judgments", "/j", "--dispute-sample", "5", "--out", "/o"]), /--dispute-sample/);
});

async function sampledPage(agreedSample = 3) {
  const state = await fixture();
  const out = await temp("out");
  await qe.runSummaryReview({ metaDir: state.meta, seed: "s", agreedSample, disputeSample: 11, outDir: out }, { now: () => NOW });
  return { out, manifest: JSON.parse(await readFile(path.join(out, "review.json"), "utf8")) };
}
const importAnswers = async (page, decide) => {
  const results = page.manifest.cards.flatMap(card => { const answer = decide(card); return answer ? [{ cardId: card.cardId, decision: answer[0], note: answer[1] ?? "" }] : []; });
  const file = path.join(await temp("results"), "r.json");
  await writeFile(file, JSON.stringify({ schemaVersion: 1, pageId: page.manifest.pageId, results }));
  return qe.runReviewImport({ resultsPath: file, outDir: page.out }, { now: () => NOW });
};

test("the estimate extrapolates the judge-sided share of each stratum to its population", async () => {
  const page = await sampledPage();
  const counters = {};
  // must (5): 3 judge, 1 reviewer, 1 unsure. completeness (4 sampled): 2 judge, 1 reviewer, 1 unanswered. clarity (2): 1 judge, 1 unsure.
  const script = { must: ["judge", "judge", "judge", "reviewer", "unsure"], completeness: ["judge", "judge", "reviewer", null], clarity: ["judge", "unsure"] };
  const sample = page.manifest.cards.filter(card => card.meta.group === "agreed-sample");
  const agreedAnswers = ["agree", "agree", "disagree"];
  const { calibration } = await importAnswers(page, card => {
    if (card.meta.group === "agreed-sample") return [agreedAnswers[sample.indexOf(card)]];
    const stratum = card.meta.stratum;
    counters[stratum] = (counters[stratum] ?? 0) + 1;
    const decision = script[stratum][counters[stratum] - 1];
    return decision ? [decision] : null;
  });
  const m = calibration.meta;
  const byKey = Object.fromEntries(m.sampling.strata.map(stratum => [stratum.key, stratum]));
  assert.deepEqual({ ...byKey.must, label: undefined }, { ...byKey.must, label: undefined, key: "must", population: 5, sampled: 5, judge: 3, reviewer: 1, unsure: 1, unanswered: 0, answered: 5 });
  assert.equal(byKey.must.judgeEstimateLow, 3);
  assert.equal(byKey.must.judgeEstimateHigh, 4);
  assert.deepEqual([byKey.completeness.population, byKey.completeness.sampled, byKey.completeness.judge, byKey.completeness.reviewer, byKey.completeness.unanswered, byKey.completeness.answered], [10, 4, 2, 1, 1, 3]);
  assert.ok(Math.abs(byKey.completeness.judgeEstimateLow - 20 / 3) < 1e-9);
  assert.ok(Math.abs(byKey.completeness.judgeEstimateHigh - 20 / 3) < 1e-9);
  assert.equal(byKey.clarity.judgeEstimateLow, 2.5);
  assert.equal(byKey.clarity.judgeEstimateHigh, 5);
  const low = 3 + 20 / 3 + 2.5;
  const high = 4 + 20 / 3 + 5;
  assert.ok(Math.abs(m.sampling.judgeSidedLow - low) < 1e-9);
  assert.ok(Math.abs(m.sampling.judgeSidedHigh - high) < 1e-9);
  // 6 reviewer-agreed issues x 2/3 agreed-sample rate, over 26 issues.
  assert.equal(m.issues, 26);
  assert.ok(Math.abs(m.precisionEstimate.point - (6 * (2 / 3) + low) / 26) < 1e-9, JSON.stringify(m.precisionEstimate));
  assert.ok(Math.abs(m.precisionEstimate.upper - (6 * (2 / 3) + high) / 26) < 1e-9);
  assert.equal(m.precisionEstimate.components.extrapolated, true);

  // Unsampled disputes are not unanswered ones.
  assert.equal(m.disputes.overall.total, 11);
  assert.equal(m.disputes.overall.unanswered, 1);
  assert.equal(m.disputes.population, 20);
  assert.equal(m.precisionEstimate.components.disputesNotJudged, 0);

  const md = await readFile(path.join(page.out, "calibration.md"), "utf8");
  for (const text of ["抽样", "各层", "总体", "推算", "估计", "必选", "完整", "清楚", "20", "上界", "下界"]) assert.ok(md.includes(text), text);
  assert.match(md, /基于抽样的估计|基于样本的估计/);
  assert.ok(md.includes(`${(((6 * (2 / 3) + low) / 26) * 100).toFixed(1)}%`));
});

test("a stratum nobody answered contributes nothing to the lower bound and everything to the upper", async () => {
  const page = await sampledPage(0);
  const { calibration } = await importAnswers(page, card => (card.meta.stratum === "must" ? ["judge"] : null));
  const strata = Object.fromEntries(calibration.meta.sampling.strata.map(stratum => [stratum.key, stratum]));
  assert.equal(strata.completeness.judgeEstimateLow, 0);
  assert.equal(strata.completeness.judgeEstimateHigh, 10);
  assert.equal(calibration.meta.precisionEstimate.point, null, "no agreed sample, no estimate");
});

test("without sampling the earlier estimate and its wording are unchanged", async () => {
  const state = await fixture();
  const out = await temp("out");
  await qe.runSummaryReview({ metaDir: state.meta, seed: "s", agreedSample: 3, outDir: out }, { now: () => NOW });
  const manifest = JSON.parse(await readFile(path.join(out, "review.json"), "utf8"));
  assert.equal(manifest.disputeSampling ?? null, null);
  const file = path.join(await temp("results"), "r.json");
  await writeFile(file, JSON.stringify({ schemaVersion: 1, pageId: manifest.pageId, results: [] }));
  const { calibration } = await qe.runReviewImport({ resultsPath: file, outDir: out }, { now: () => NOW });
  assert.equal(calibration.meta.sampling, undefined);
  assert.equal(calibration.meta.disputes.overall.total, 20);
  assert.equal(calibration.meta.disputes.overall.unanswered, 20);
});
