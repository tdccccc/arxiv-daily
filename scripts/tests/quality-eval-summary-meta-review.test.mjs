import assert from "node:assert/strict";
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

const SENTINEL_KEY = "SENTINEL-META-KEY-9b1e";
const SENTINEL_HOST = "sentinel-meta-host.example.invalid";
const NOW = new Date("2026-10-30T08:00:00Z");
const FIELD_LABELS = ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"];
const ZH_HEADINGS = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "贡献与创新点", "适用边界", "学术价值判断"];
const DAILY_IDS = ["2610.00001", "2610.00002", "2610.00003"];
const NOTE_ID = "2607.00001";
const GENERATOR = { llm: { provider: "openai", model: "gen-model", baseUrl: "https://gen.example.invalid/v1", apiKey: "gen-key", thinkingMode: false, reasoningEffort: "medium" }, topics: [{ tag: "galaxy-cluster", name: "Galaxy Cluster" }], summaryLanguage: "zh" };

async function fixture() {
  const vault = await temp("vault");
  const sources = await temp("sources");
  await mkdir(path.join(vault, "arxiv-daily", "daily"), { recursive: true });
  await mkdir(path.join(vault, "arxiv-daily", "papers"), { recursive: true });
  const block = id => [`### Paper ${id}`, "- **作者**: A. Author et al.", `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`, ...FIELD_LABELS.map(label => `- **${label}**: ${id} 的${label}。`), ""];
  await writeFile(path.join(vault, "arxiv-daily", "daily", "2026-10-15.md"), ["---", "date: 2026-10-15", "---", "", "# t", "", "## Photo-z", "", ...DAILY_IDS.flatMap(block)].join("\n"));
  await writeFile(path.join(vault, "arxiv-daily", "papers", `${NOTE_ID}.md`), [
    "---", 'title: "Note"', 'authors: "N. Author et al."', `arxiv_id: "${NOTE_ID}"`, "primary_topic: galaxy-cluster", "published: 2026-07-10", "---", "", "# Note", "",
    ...ZH_HEADINGS.flatMap(heading => [`## ${heading}`, "", `${heading}的正文。`, ""]),
  ].join("\n"));
  for (const id of [...DAILY_IDS, NOTE_ID]) {
    await writeFile(path.join(sources, `${id}.json`), JSON.stringify({
      schemaVersion: 1, id, fetchedAt: "2026-10-28T10:00:00.000Z", sourcePath: "html", arxivVersion: "v1", budgets: { sectionCharLimit: 16000, paperCharLimit: 100000 },
      content: { abstract: "a", abstractConclusion: `## Abstract\nAbstract of ${id}.`, fullSections: `## Methods\nWe use 120 clusters in ${id}.`, fullTextFailure: null }, contentHash: `hash-${id}`,
    }));
  }
  return { vault, sources };
}

const goodDaily = id => JSON.stringify({ id, coreProblem: `${id} 新问题。`, keyMethod: "新方法。", mainResult: "新结果 130 个星团。", whyRelevant: "新价值。", limitations: "新边界。" });
const NOTE_MARKDOWN = "# Note\n\n## 研究问题\n\n新的问题正文。\n\n## 核心结果\n\n新的结果：130 个星团。\n";
const idIn = messages => /ID: (\S+)/.exec(messages.find(message => message.role === "user").content)?.[1];
const regenLlm = () => ({ async call(messages, options) {
  const isNote = messages[0].content.includes("详细的中文论文总结");
  options?.onMetrics?.({ logicalCalls: 1, attempts: 1, elapsedMs: 1, inputTokens: 1, outputTokens: 1, totalTokens: 2, usageComplete: true });
  return isNote ? NOTE_MARKDOWN : goodDaily(idIn(messages));
} });

const JUDGE_TOML = '[llm]\napi_key = "judge-key-1"\nbase_url = "https://judge.example.invalid/v1"\nmodel = "judge-model"\nprovider = "openai"\n';
/** The judge raises two issues on 2610.00001, one on the note, none on the others. */
const judgeReply = messages => {
  const text = messages[1].content;
  const issue = (dimension, severity, location, summaryQuote, sourceQuote, explanation) => ({ dimension, severity, location, summaryQuote, sourceQuote, explanation });
  if (text.includes("2610.00001 新问题")) {
    return JSON.stringify({ verdicts: { accuracy: "major", completeness: "ok", clarity: "minor" }, issues: [
      issue("accuracy", "major", "mainResult", "新结果 130 个星团", "We use 120 clusters in 2610.00001", "摘要写 130，原文是 120。"),
      issue("clarity", "minor", "keyMethod", "新方法", null, "方法没有说明用了什么数据。"),
    ] });
  }
  if (text.includes("新的结果：130 个星团")) {
    return JSON.stringify({ verdicts: { accuracy: "minor", completeness: "ok", clarity: "ok" }, issues: [issue("accuracy", "minor", "核心结果", "130 个星团", "We use 120 clusters in 2607.00001", "星团数不一致。")] });
  }
  return JSON.stringify({ verdicts: { accuracy: "ok", completeness: "ok", clarity: "ok" }, issues: [] });
};

/** A real regenerate run and a real judge run, both with fake models. */
async function pipeline() {
  const { vault, sources } = await fixture();
  const regen = path.join(await temp("regen"), "r");
  await qe.runSummaryRegenerate({ vaultRoot: vault, sourcesDir: sources, outDir: regen, daily: 3, notes: 1, seed: "s", yes: true }, { llm: regenLlm(), now: () => NOW, log: () => undefined, loadGeneratorConfig: async () => GENERATOR });
  const judgeConfigPath = path.join(await temp("jc"), "judge.toml");
  await writeFile(judgeConfigPath, JUDGE_TOML);
  const judged = path.join(await temp("judged"), "j");
  const judgeLlm = { async call(messages, options) { options?.onMetrics?.({ logicalCalls: 1, attempts: 1, elapsedMs: 1, usageComplete: true }); return judgeReply(messages); } };
  await qe.runSummaryJudge({ judgeConfigPath, sourcesDir: sources, summariesDir: regen, outDir: judged, yes: true }, { llm: judgeLlm, now: () => NOW, log: () => undefined });
  return { vault, sources, regen, judged };
}

const REVIEWER_BASE = { provider: "openai", model: "cli-model", baseUrl: `https://${SENTINEL_HOST}/v1`, apiKey: SENTINEL_KEY, thinkingMode: false, reasoningEffort: "medium" };
const reviewsFor = (count, overrides = {}) => JSON.stringify({ reviews: Array.from({ length: count }, (_, index) => ({ index, verdict: "agree", severity: "ok", reason: "问题确实存在，原文与摘要不一致。", evidenceQuote: null, ...(overrides[index] ?? {}) })) });
const issueCount = messages => (messages[1].content.match(/^\[(\d+)\] /gm) ?? []).length;

function reviewerLlm(script) {
  const calls = [];
  return {
    calls,
    async call(messages, options) {
      calls.push({ messages, options });
      const scripted = script?.(calls.length, messages);
      options?.onMetrics?.({ logicalCalls: 1, attempts: 1, elapsedMs: 3, inputTokens: 200, outputTokens: 20, totalTokens: 220, usageComplete: true });
      if (scripted instanceof Error) throw scripted;
      return scripted ?? reviewsFor(issueCount(messages));
    },
  };
}
const execute = async (state, { llm = reviewerLlm(), reviewerBase = REVIEWER_BASE } = {}, more = {}) => {
  const log = [];
  const options = { judgedDir: state.judged, sourcesDir: state.sources, configPath: "/fixture/config.toml", reviewerModel: "reviewer-model", reviewerEffort: "xhigh", outDir: state.out, ...more };
  const result = await qe.runSummaryMetaReview(options, { llm, now: () => NOW, log: line => log.push(line), loadReviewerBase: async () => reviewerBase });
  return { result, llm, log };
};
async function setup() {
  const state = await pipeline();
  state.out = path.join(await temp("meta"), "m");
  return state;
}
const readItem = async (out, kind, id) => JSON.parse(await readFile(path.join(out, "meta", `${kind}-${id}.json`), "utf8"));

test("without --yes the plan is printed and nothing is created or called", async () => {
  const state = await setup();
  const { result, llm, log } = await execute(state);
  assert.equal(result.dryRun, true);
  assert.equal(llm.calls.length, 0);
  assert.equal(existsSync(state.out), false);
  const text = log.join("\n");
  assert.match(text, /reviewer-model/);
  assert.match(text, /xhigh/);
  assert.match(text, /2 items/);
  assert.match(text, /3 issues/);
  assert.match(text, /at most 4 requests/);
  assert.ok(!text.includes(SENTINEL_KEY) && !text.includes(SENTINEL_HOST));
  assert.deepEqual({ items: result.plan.items, issues: result.plan.issues, maxRequests: result.plan.maxRequests }, { items: 2, issues: 3, maxRequests: 4 });
  assert.ok(result.plan.totalChars > 0);
});

test("the reviewer settings are the CLI config's with the model and effort overridden and thinking on", async () => {
  const state = await setup();
  const settings = await qe.reviewerSettings(REVIEWER_BASE, { model: "reviewer-model", effort: "xhigh" });
  assert.deepEqual(settings, { ...REVIEWER_BASE, model: "reviewer-model", reasoningEffort: "xhigh", thinkingMode: true });
  void state;
});

test("the reviewer must differ from the judge model, and may equal the generator", async () => {
  const state = await setup();
  await assert.rejects(execute(state, {}, { reviewerModel: " Judge-Model " }), /different model/i);
  const sameAsGenerator = await execute(state, {}, { reviewerModel: "GEN-MODEL" });
  assert.match(sameAsGenerator.log.join("\n"), /same model as the generator/i);
});

test("a request carries the judge's own paper data and summary text, and every numbered issue", async () => {
  const state = await setup();
  const { llm } = await execute(state, {}, { yes: true });
  assert.equal(llm.calls.length, 2);
  const judgeInput = (await qe.prepareRegeneratedJudgeInput({ summariesDir: state.regen, sourcesDir: state.sources })).items.find(item => item.id === "2610.00001");
  const call = llm.calls.find(candidate => candidate.messages[1].content.includes("2610.00001 新问题"));
  const user = call.messages[1].content;
  assert.ok(user.startsWith(judgeInput.paperData));
  assert.ok(user.includes(`<summary_under_review>\n${judgeInput.judgedText}\n</summary_under_review>`));
  assert.match(user, /\[0\] [\s\S]*摘要写 130，原文是 120。[\s\S]*\[1\] [\s\S]*方法没有说明用了什么数据。/);
  assert.ok(user.includes("We use 120 clusters in 2610.00001"));
  assert.ok(user.includes("新结果 130 个星团"));
  assert.equal(call.options.temperature, undefined, "the reviewer is not asked for a temperature");
  assert.ok(call.messages[0].content.includes("拿不准") && call.messages[0].content.includes("evidenceQuote"));
  assert.ok(!/\{\{\w+\}\}/.test(call.messages[0].content));
  // Only items with issues are sent.
  assert.ok(!llm.calls.some(candidate => candidate.messages[1].content.includes("2610.00002 新问题")));
});

test("the reviewer prompt reuses the judge's dimension and severity definitions and sets its limits", () => {
  const prompt = qe.buildMetaSystemPrompt({ kind: "daily" });
  const judge = qe.buildJudgeSystemPrompt({ kind: "daily" });
  const between = (text, from, to) => text.slice(text.indexOf(from), text.indexOf(to));
  assert.ok(prompt.includes(between(judge, "1. 准确", "## 严重程度").trim()));
  assert.ok(prompt.includes(between(judge, "- \"major\"", "## 不要评审的内容").trim()));
  for (const text of ["不要新增问题", "格式", "长度", "快速浏览", "合适", "偏重", "偏轻", "<paper_data>"]) assert.ok(prompt.includes(text), text);
  assert.ok(prompt.includes((qe.buildJudgeSystemPrompt({ kind: "daily" }).match(/以下用户消息中[^\n]+/) ?? [""])[0]), "the injection guard");
  const note = qe.buildMetaSystemPrompt({ kind: "note" });
  assert.ok(note.includes("完整梳理") && !note.includes("coreProblem"));
});

const issueSet = { kind: "daily", count: 3, paperData: "<paper_data>\nWe use 120 clusters.\n</paper_data>" };
const check = raw => qe.checkMetaOutput(typeof raw === "string" ? raw : JSON.stringify(raw), issueSet);

test("the review contract covers every index exactly once with strict enums", () => {
  const good = JSON.parse(reviewsFor(3, { 1: { verdict: "disagree", severity: "too-high" }, 2: { verdict: "unsure", severity: "too-low", evidenceQuote: "We use 120 clusters." } }));
  const result = check(good);
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.deepEqual(result.reviews.map(review => review.verdict), ["agree", "disagree", "unsure"]);
  const reasons = value => { const outcome = check(value); assert.equal(outcome.ok, false, JSON.stringify(value)); return outcome.reasons.join("\n"); };
  assert.match(reasons({ reviews: good.reviews.slice(0, 2) }), /index 2|缺少/);
  assert.match(reasons({ reviews: [...good.reviews, { ...good.reviews[0] }] }), /多了|index 0|重复|3/);
  assert.match(reasons({ reviews: [good.reviews[0], good.reviews[0], good.reviews[2]] }), /重复|twice|index 0/);
  assert.match(reasons({ reviews: [{ ...good.reviews[0], index: 7 }, good.reviews[1], good.reviews[2]] }), /index.*7/);
  assert.match(reasons({ reviews: [{ ...good.reviews[0], verdict: "yes" }, good.reviews[1], good.reviews[2]] }), /verdict/);
  assert.match(reasons({ reviews: [{ ...good.reviews[0], severity: "high" }, good.reviews[1], good.reviews[2]] }), /severity/);
  assert.match(reasons({ reviews: [{ ...good.reviews[0], extra: 1 }, good.reviews[1], good.reviews[2]] }), /extra/);
  assert.match(reasons({ reviews: [{ index: 0, verdict: "agree", severity: "ok", evidenceQuote: null }, good.reviews[1], good.reviews[2]] }), /reason/);
  assert.match(reasons({ items: [] }), /reviews/);
  assert.match(reasons({ reviews: good.reviews, extra: 1 }), /extra/);
  assert.match(reasons("not json"), /JSON/);
  assert.equal(qe.checkMetaOutput(`\`\`\`json\n${JSON.stringify(good)}\n\`\`\``, issueSet).ok, true);
});

test("a reason must be Chinese and an evidence quote must be in the paper data", () => {
  const english = JSON.parse(reviewsFor(3));
  english.reviews[1].reason = "The issue is real because the source says 120 clusters.";
  const outcome = check(english);
  assert.equal(outcome.ok, false);
  assert.ok(outcome.reasons.some(reason => reason.includes("reviews[1].reason") && reason.includes("中文")));
  const chinese = JSON.parse(reviewsFor(3));
  chinese.reviews[1].reason = "原文写的是 “We use 120 clusters”，所以评审说得对。";
  assert.equal(check(chinese).ok, true);

  const quote = JSON.parse(reviewsFor(3));
  quote.reviews[2].evidenceQuote = "We use  130 clusters.";
  assert.ok(check(quote).reasons.some(reason => reason.includes("reviews[2].evidenceQuote")));
  quote.reviews[2].evidenceQuote = "We use\n120 clusters.";
  assert.equal(check(quote).ok, true, "whitespace differences do not matter");
  quote.reviews[2].evidenceQuote = "";
  assert.equal(check(quote).ok, false);
});

test("a reply that fails validation gets one correction, and a second failure makes the item invalid", async () => {
  const state = await setup();
  const first = reviewerLlm((call, messages) => (call === 1 && issueCount(messages) === 2 ? reviewsFor(1) : undefined));
  const { result } = await execute(state, { llm: first }, { yes: true });
  const retried = first.calls.filter(call => call.messages.length === 4);
  assert.equal(retried.length, 1);
  assert.equal(retried[0].messages[2].role, "assistant");
  assert.match(retried[0].messages[3].content, /index 1|缺少/);
  assert.match(retried[0].messages[3].content, /中文/);
  assert.equal(result.counts.reviewed, 2);

  const bad = path.join(await temp("meta"), "m2");
  const never = reviewerLlm(() => "完全不是 JSON");
  const second = await execute({ ...state, out: bad }, { llm: never }, { yes: true });
  assert.equal(never.calls.length, 4);
  assert.equal(second.result.counts.invalid, 2);
  const record = await readItem(bad, "daily", "2610.00001");
  assert.equal(record.status, "invalid");
  assert.equal(record.reviews, null);
  assert.equal(record.attempts.length, 2);
});

test("records keep the reviewer identity, the judge's issues and the reviews, without secrets", async () => {
  const state = await setup();
  await execute(state, { llm: reviewerLlm((call, messages) => (issueCount(messages) === 2 ? reviewsFor(2, { 0: { verdict: "disagree", severity: "too-high", reason: "原文其实写的是 130。", evidenceQuote: "We use 120 clusters in 2610.00001" } }) : undefined)) }, { yes: true });
  const record = await readItem(state.out, "daily", "2610.00001");
  assert.equal(record.status, "reviewed");
  assert.deepEqual(record.reviewer, { provider: "openai", model: "reviewer-model", thinkingMode: true, reasoningEffort: "xhigh" });
  assert.match(record.reviewerPromptHash, /^[0-9a-f]{64}$/);
  assert.match(record.judgmentHash, /^[0-9a-f]{64}$/);
  assert.equal(record.judge.model, "judge-model");
  assert.equal(record.issues.length, 2);
  assert.equal(record.issues[0].location, "mainResult");
  assert.equal(record.reviews[0].verdict, "disagree");
  assert.equal(record.reviews[0].severity, "too-high");
  assert.equal(record.attempts[0].usage.inputTokens, 200);
  assert.equal(record.item.origin.kind, "regenerated");
  const files = [];
  const walk = async dir => { for (const entry of await readdir(dir, { withFileTypes: true })) { const full = path.join(dir, entry.name); if (entry.isDirectory()) await walk(full); else files.push(full); } };
  await walk(state.out);
  for (const file of files) { const text = await readFile(file, "utf8"); assert.ok(!text.includes(SENTINEL_KEY) && !text.includes(SENTINEL_HOST), file); }
  const run = JSON.parse(await readFile(path.join(state.out, "run.json"), "utf8"));
  assert.equal(run.reviewer.model, "reviewer-model");
  assert.equal(run.counts.reviewed, 2);
  assert.ok(!JSON.stringify(run).includes(SENTINEL_KEY));
});

test("a rerun skips reviewed items; a changed reviewer, effort, prompt or judgment is reviewed again; --rereview redoes all", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  assert.equal((await execute(state, {}, { yes: true })).llm.calls.length, 0);
  assert.equal((await execute(state, {}, { yes: true, reviewerEffort: "high" })).llm.calls.length, 2);
  assert.equal((await execute(state, {}, { yes: true, reviewerModel: "reviewer-model-2" })).llm.calls.length, 2);
  await execute(state, {}, { yes: true });
  const file = path.join(state.out, "meta", "daily-2610.00001.json");
  const record = JSON.parse(await readFile(file, "utf8"));
  await writeFile(file, JSON.stringify({ ...record, reviewerPromptHash: "0".repeat(64) }));
  assert.equal((await execute(state, {}, { yes: true })).llm.calls.length, 1);
  await writeFile(file, JSON.stringify({ ...JSON.parse(await readFile(file, "utf8")), judgmentHash: "0".repeat(64) }));
  assert.equal((await execute(state, {}, { yes: true })).llm.calls.length, 1);
  assert.equal((await execute(state, {}, { yes: true, rereview: true })).llm.calls.length, 2);
});

test("three errors in a row stop the run, and an errored item is reviewed on the next run", async () => {
  const state = await setup();
  const failing = reviewerLlm(() => new Error(`HTTP 500 from https://${SENTINEL_HOST}/v1 key ${SENTINEL_KEY}`));
  const first = await execute(state, { llm: failing }, { yes: true });
  assert.equal(first.result.counts.errors, 2);
  const record = await readItem(state.out, "daily", "2610.00001");
  assert.equal(record.status, "error");
  assert.ok(!JSON.stringify(record).includes(SENTINEL_KEY));
  assert.ok(!first.log.join("\n").includes(SENTINEL_HOST));
  assert.equal((await execute(state, {}, { yes: true })).result.counts.reviewed, 2);
});

test("the report gives the agreement rate overall and by dimension, severity and kind, with the severity opinions", async () => {
  const state = await setup();
  // 2610.00001: issue 0 (accuracy, major) disagree/too-high; issue 1 (clarity, minor) agree/ok. Note issue (accuracy, minor): unsure/too-low.
  const llm = reviewerLlm((call, messages) => (issueCount(messages) === 2
    ? reviewsFor(2, { 0: { verdict: "disagree", severity: "too-high", reason: "原文其实写的是 130，评审看错了。", evidenceQuote: "We use 120 clusters in 2610.00001" } })
    : reviewsFor(1, { 0: { verdict: "unsure", severity: "too-low", reason: "无法确定星团数。" } })));
  await execute(state, { llm }, { yes: true });
  const md = await readFile(path.join(state.out, "meta-review.md"), "utf8");
  const json = JSON.parse(await readFile(path.join(state.out, "meta-review.json"), "utf8"));
  assert.deepEqual(json.aggregate.overall, { agree: 1, disagree: 1, unsure: 1, rate: 0.5 });
  assert.deepEqual(json.aggregate.byDimension.accuracy, { agree: 0, disagree: 1, unsure: 1, rate: 0 });
  assert.deepEqual(json.aggregate.byDimension.clarity, { agree: 1, disagree: 0, unsure: 0, rate: 1 });
  assert.deepEqual(json.aggregate.byDimension.completeness, { agree: 0, disagree: 0, unsure: 0, rate: null });
  assert.deepEqual(json.aggregate.bySeverity.major, { agree: 0, disagree: 1, unsure: 0, rate: 0 });
  assert.deepEqual(json.aggregate.bySeverity.minor, { agree: 1, disagree: 0, unsure: 1, rate: 1 });
  assert.deepEqual(json.aggregate.byKind.daily, { agree: 1, disagree: 1, unsure: 0, rate: 0.5 });
  assert.deepEqual(json.aggregate.byKind.note, { agree: 0, disagree: 0, unsure: 1, rate: null });
  assert.deepEqual(json.aggregate.severityOpinion.overall, { ok: 1, "too-high": 1, "too-low": 1 });
  assert.deepEqual(json.aggregate.severityOpinion.byJudgeSeverity.major, { ok: 0, "too-high": 1, "too-low": 0 });
  assert.deepEqual(json.counts, { items: 2, reviewed: 2, invalid: 0, errors: 0, issues: 3 });

  for (const text of ["复核一致率", "50.0%", "按维度", "按严重程度", "按类型", "严重程度意见", "偏重", "偏轻", "不同意和拿不准的问题", "原文其实写的是 130，评审看错了。", "无法确定星团数。", "核心结果", "2610.00001", "gpt", "reviewer-model", "xhigh"].filter(text => text !== "gpt")) assert.ok(md.includes(text), text);
  assert.ok(!md.includes("mainResult"), "daily fields are shown by their Chinese labels");
  assert.match(md, /无效 0[^\n]*出错 0|无效[^\n]*0/);
});

test("--report-only rebuilds the report from the records without any model", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  await rm(path.join(state.out, "meta-review.md"));
  const { files } = await qe.runMetaReportOnly({ outDir: state.out }, { now: () => NOW });
  assert.equal(files.length, 2);
  assert.ok(existsSync(path.join(state.out, "meta-review.md")));
  await assert.rejects(qe.runMetaReportOnly({ outDir: await temp("empty") }, {}), /no (reviews|records)/i);
});

test("historical judgments are rebuilt from the vault", async () => {
  const { vault, sources } = await fixture();
  const judgeConfigPath = path.join(await temp("jc"), "judge.toml");
  await writeFile(judgeConfigPath, JUDGE_TOML);
  const judged = path.join(await temp("judged"), "j");
  const historicalReply = messages => (messages[1].content.includes("2610.00001 的历史核心结果") || messages[1].content.includes("2610.00001 的核心结果")
    ? JSON.stringify({ verdicts: { accuracy: "minor", completeness: "ok", clarity: "ok" }, issues: [{ dimension: "accuracy", severity: "minor", location: "mainResult", summaryQuote: "2610.00001 的核心结果", sourceQuote: "We use 120 clusters in 2610.00001", explanation: "与原文不符的示例。" }] })
    : JSON.stringify({ verdicts: { accuracy: "ok", completeness: "ok", clarity: "ok" }, issues: [] }));
  await qe.runSummaryJudge({ judgeConfigPath, sourcesDir: sources, vaultRoot: vault, outDir: judged, daily: 3, notes: 0, seed: "s", yes: true },
    { llm: { async call(messages) { return historicalReply(messages); } }, now: () => NOW, log: () => undefined, loadGeneratorConfig: async () => ({ model: "gen-model", topics: [] }) });
  const out = path.join(await temp("meta"), "m");
  const llm = reviewerLlm();
  const log = [];
  const result = await qe.runSummaryMetaReview({ judgedDir: judged, sourcesDir: sources, configPath: "/fixture/config.toml", vaultRoot: vault, reviewerModel: "reviewer-model", reviewerEffort: "xhigh", outDir: out, yes: true },
    { llm, now: () => NOW, log: line => log.push(line), loadReviewerBase: async () => REVIEWER_BASE });
  assert.equal(result.counts.reviewed, 1, log.join("\n"));
  assert.ok(llm.calls[0].messages[1].content.includes("We use 120 clusters in 2610.00001"));
  // Without a vault the historical summary cannot be rebuilt.
  const lost = await qe.runSummaryMetaReview({ judgedDir: judged, sourcesDir: sources, configPath: "/fixture/config.toml", reviewerModel: "reviewer-model", reviewerEffort: "xhigh", outDir: path.join(await temp("meta"), "m") },
    { llm: reviewerLlm(), now: () => NOW, log: () => undefined, loadReviewerBase: async () => REVIEWER_BASE });
  assert.equal(lost.plan.items, 0);
  assert.equal(lost.plan.skipped[0].reason, "no-vault");
});

test("a judgment whose summary or source has changed since is skipped, not reviewed against different text", async () => {
  const state = await setup();
  const original = regenLlm();
  const changedGenerator = { async call(messages, options) {
    const reply = await original.call(messages, options);
    if (idIn(messages) !== "2610.00001") return reply;
    return JSON.stringify({ ...JSON.parse(reply), mainResult: "重新生成后不同的结果。" });
  } };
  await qe.runSummaryRegenerate({ vaultRoot: state.vault, sourcesDir: state.sources, outDir: state.regen, daily: 3, notes: 1, seed: "s", yes: true, regenerate: true },
    { llm: changedGenerator, now: () => NOW, log: () => undefined, loadGeneratorConfig: async () => GENERATOR });
  const { result } = await execute(state);
  assert.deepEqual(result.plan.skipped.map(entry => [entry.id, entry.reason]), [["2610.00001", "summary-changed"]]);
  assert.equal(result.plan.items, 1);
});

test("command-line arguments are validated", () => {
  assert.deepEqual(
    qe.parseSummaryMetaReviewArgs(["--judged", "/j", "--sources", "/s", "--config", "/c.toml", "--reviewer-model", "m", "--reviewer-effort", "xhigh", "--vault", "/v", "--daily-dir", "d", "--papers-dir", "p", "--out", "/o", "--yes", "--rereview"]),
    { judgedDir: "/j", sourcesDir: "/s", configPath: "/c.toml", reviewerModel: "m", reviewerEffort: "xhigh", vaultRoot: "/v", dailyDir: "d", papersDir: "p", outDir: "/o", yes: true, rereview: true },
  );
  assert.deepEqual(qe.parseSummaryMetaReviewArgs(["--report-only", "--out", "/o"]), { reportOnly: true, outDir: "/o" });
  assert.equal(qe.parseSummaryMetaReviewArgs(["--help"]).help, true);
  const base = ["--judged", "/j", "--sources", "/s", "--config", "/c", "--reviewer-model", "m", "--reviewer-effort", "high", "--out", "/o"];
  for (const missing of ["--judged", "--sources", "--config", "--reviewer-model", "--reviewer-effort", "--out"]) {
    const at = base.indexOf(missing);
    assert.throws(() => qe.parseSummaryMetaReviewArgs(base.filter((_, position) => position !== at && position !== at + 1)), new RegExp(missing));
  }
  assert.throws(() => qe.parseSummaryMetaReviewArgs([...base, "--bogus"]), /unknown option.*--bogus/i);
  assert.throws(() => qe.parseSummaryMetaReviewArgs(["--judged", "/j", "--sources", "/s", "--config", "/c", "--reviewer-model", "m", "--reviewer-effort", "", "--out", "/o"]), /--reviewer-effort/);
});

test("the script prints usage on bad arguments", async () => {
  const { spawnSync } = await import("node:child_process");
  const script = repoFile("scripts/quality-eval/summary-meta-review.mjs");
  const bad = spawnSync(process.execPath, [script, "--out", "/o"], { encoding: "utf8" });
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /Usage: node scripts\/quality-eval\/summary-meta-review\.mjs/);
  assert.equal(spawnSync(process.execPath, [script, "--help"], { encoding: "utf8" }).status, 0);
});


test("legacy judgment identity limitations survive the meta-review plan and report", async () => {
  const state = await setup();
  const runPath = path.join(state.judged, "run.json");
  const run = JSON.parse(await readFile(runPath, "utf8"));
  delete run.resultManifest;
  await writeFile(runPath, JSON.stringify(run));
  const { log } = await execute(state, {}, { yes: true });
  assert.match(log.join("\n"), /旧格式运行没有结果清单/);
  const report = JSON.parse(await readFile(path.join(state.out, "meta-review.json"), "utf8"));
  assert.equal(report.run.integrityWarnings.length, 1);
  assert.match(await readFile(path.join(state.out, "meta-review.md"), "utf8"), /旧格式运行没有结果清单/);
});


test("a narrowed judge sample excludes old meta reviews from the report and dispute page", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const runPath = path.join(state.judged, "run.json");
  const run = JSON.parse(await readFile(runPath, "utf8"));
  run.sample = { daily: ["2610.00001"], notes: [] };
  if (run.resultManifest) run.resultManifest = run.resultManifest.filter(item => item.kind === "daily" && item.id === "2610.00001");
  await writeFile(runPath, JSON.stringify(run));
  const { result } = await execute(state, {}, { yes: true, reviewerModel: "new-reviewer" });
  assert.equal(result.counts.items, 1);
  assert.equal(result.counts.reviewed, 1);
  await qe.runMetaReportOnly({ outDir: state.out }, { now: () => NOW });
  const report = JSON.parse(await readFile(path.join(state.out, "meta-review.json"), "utf8"));
  assert.deepEqual(report.records.map(item => [item.item.kind, item.item.id, item.reviewer.model]), [["daily", "2610.00001", "new-reviewer"]]);
  const built = await qe.buildReviewCards({ metaDir: state.out, seed: "s", agreedSample: 20 });
  assert.ok(built.cards.length > 0);
  assert.ok(built.cards.every(item => item.kind === "daily" && item.id === "2610.00001"));
});

test("an empty meta result manifest does not resurrect previous successes after an interrupted rereview", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const runPath = path.join(state.out, "run.json");
  const run = JSON.parse(await readFile(runPath, "utf8"));
  run.resultManifest = [];
  run.finishedAt = null;
  run.counts = null;
  await writeFile(runPath, JSON.stringify(run));
  await qe.runMetaReportOnly({ outDir: state.out }, { now: () => NOW });
  const report = JSON.parse(await readFile(path.join(state.out, "meta-review.json"), "utf8"));
  assert.equal(report.counts.items, 0);
  assert.equal(report.counts.reviewed, 0);
  await assert.rejects(qe.buildReviewCards({ metaDir: state.out, seed: "s" }), /no reviews|no issues/);
});

test("meta report and dispute page reject a result that no longer matches the run manifest", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const file = path.join(state.out, "meta", "daily-2610.00001.json");
  const record = JSON.parse(await readFile(file, "utf8"));
  record.reviewer.model = "unrelated-reviewer";
  await writeFile(file, JSON.stringify(record));
  await assert.rejects(qe.runMetaReportOnly({ outDir: state.out }), /manifest/i);
  await assert.rejects(qe.buildReviewCards({ metaDir: state.out, seed: "s" }), /manifest/i);
});

test("a reviewer endpoint change invalidates the cache, while key rotation preserves it", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const reviewerBase = { ...REVIEWER_BASE, baseUrl: "https://new-review-endpoint.example.invalid/v1" };
  const changed = await execute(state, { reviewerBase }, { yes: true });
  assert.equal(changed.llm.calls.length, 2);
  const report = await readFile(path.join(state.out, "meta-review.json"), "utf8");
  assert.ok(!report.includes("new-review-endpoint.example.invalid"));
  const rotated = await execute(state, { reviewerBase: { ...reviewerBase, apiKey: "ROTATED-TEST-SECRET" } }, { yes: true });
  assert.equal(rotated.llm.calls.length, 0);
  assert.equal(rotated.result.counts.reused, 2);
});
