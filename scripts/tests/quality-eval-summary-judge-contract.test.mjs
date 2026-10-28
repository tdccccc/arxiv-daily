import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test, { before } from "node:test";
import { loadQualityEval } from "../quality-eval/load.mjs";

let qe;
before(async () => { qe = await loadQualityEval(); });
const repoFile = relative => path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..", relative);

const SOURCE = [
  "## Abstract", "We measure the cluster mass ratio to be $0.85 \\pm 0.04$ using 120 clusters.",
  "## Methods", "We stack the weak lensing shear profiles of the clusters and calibrate against\nsimulations.",
  "## Limitations", "Only optically selected clusters are included; X-ray selected samples are not tested.",
].join("\n");

const DAILY = {
  kind: "daily",
  locations: {
    coreProblem: "星团质量标定能否摆脱对静力学平衡假设的依赖。",
    keyMethod: "叠加弱透镜剪切并与模拟标定。",
    mainResult: "弱透镜质量与X射线质量之比为 $0.85\\pm0.04$，样本为120个星团。",
    whyRelevant: "该比值决定宇宙学约束中的质量偏差先验。",
    limitations: "仅覆盖光学选择的星团。",
  },
  sourceText: SOURCE,
};
const NOTE = {
  kind: "note",
  locations: { 研究问题: "论文要解决星团质量标定问题。", 核心结果: "质量比为 $0.85\\pm0.04$。\n\n样本为 120 个星团。", 适用边界: "原文未讨论。" },
  sourceText: SOURCE,
};

const verdicts = (accuracy = "ok", completeness = "ok", clarity = "ok") => ({ accuracy, completeness, clarity });
const issue = (overrides = {}) => ({
  dimension: "accuracy", severity: "major", location: "mainResult", summaryQuote: "样本为120个星团",
  sourceQuote: "using 120 clusters", explanation: "样本数与原文一致，仅作示例。", ...overrides,
});
const judgment = (issues = [], v) => {
  const rank = { ok: 0, minor: 1, major: 2 };
  const computed = verdicts();
  for (const item of issues) if (rank[item.severity] > rank[computed[item.dimension]]) computed[item.dimension] = item.severity;
  return { verdicts: v ?? computed, issues };
};
const check = (value, context = DAILY) => qe.validateJudgment(value, context);
const reasonsOf = (value, context = DAILY) => { const result = check(value, context); assert.equal(result.ok, false, JSON.stringify(result)); return result.reasons; };

test("an all-ok judgment with no issues is valid", () => {
  const result = check(judgment());
  assert.equal(result.ok, true);
  assert.deepEqual(result.judgment, { verdicts: verdicts(), issues: [] });
});

test("a valid daily judgment with every kind of issue passes", () => {
  const value = judgment([
    issue(),
    issue({ dimension: "completeness", severity: "minor", location: "limitations", summaryQuote: "仅覆盖光学选择的星团", sourceQuote: "X-ray selected samples are not tested", explanation: "漏掉了未检验 X 射线选择样本这一限制。" }),
    issue({ dimension: "clarity", severity: "minor", location: "keyMethod", summaryQuote: "与模拟标定", sourceQuote: null, explanation: "没说明与哪类模拟比较。" }),
    issue({ dimension: "accuracy", severity: "minor", location: "whyRelevant", summaryQuote: "质量偏差先验", sourceQuote: null, explanation: "原文未提及这一用途，属于模型自行补充。" }),
  ]);
  const result = check(value);
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.deepEqual(result.judgment.verdicts, verdicts("major", "minor", "minor"));
});

test("a valid note judgment locates issues by the note's own section headings", () => {
  const result = check(judgment([issue({ location: "核心结果", summaryQuote: "样本为 120 个星团" })]), NOTE);
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.ok(reasonsOf(judgment([issue({ location: "mainResult" })]), NOTE).some(reason => reason.includes("mainResult")));
});

test("keys and enums are strict", () => {
  assert.match(reasonsOf({ ...judgment(), extra: 1 }).join("\n"), /extra/);
  assert.match(reasonsOf({ verdicts: verdicts() }).join("\n"), /issues/);
  assert.match(reasonsOf({ verdicts: { accuracy: "ok", completeness: "ok" }, issues: [] }).join("\n"), /clarity/);
  assert.match(reasonsOf({ verdicts: { ...verdicts(), tone: "ok" }, issues: [] }).join("\n"), /tone/);
  assert.match(reasonsOf({ verdicts: verdicts("fine"), issues: [] }).join("\n"), /accuracy.*fine|fine/);
  assert.match(reasonsOf(judgment([issue({ severity: "critical" })], verdicts("major"))).join("\n"), /severity/);
  assert.match(reasonsOf(judgment([issue({ dimension: "style" })], verdicts())).join("\n"), /dimension/);
  assert.match(reasonsOf(judgment([{ ...issue(), extra: true }])).join("\n"), /extra/);
  const { explanation, ...missing } = issue();
  assert.match(reasonsOf(judgment([missing], verdicts("major"))).join("\n"), /explanation/);
  assert.match(reasonsOf("not an object").join("\n"), /对象/);
  assert.match(reasonsOf([]).join("\n"), /对象/);
  assert.match(reasonsOf({ verdicts: verdicts(), issues: {} }).join("\n"), /issues.*数组|数组/);
});

test("an issue must be at an allowed location", () => {
  const reasons = reasonsOf(judgment([issue({ location: "abstract" })]));
  assert.ok(reasons.some(reason => reason.includes("issues[0].location") && reason.includes("abstract")));
  assert.ok(reasons.some(reason => reason.includes("coreProblem")), "the allowed values are named");
});

test("the summary quote must be an exact piece of the text at that location", () => {
  const notThere = reasonsOf(judgment([issue({ summaryQuote: "样本为130个星团" })]));
  assert.ok(notThere.some(reason => reason.includes("issues[0].summaryQuote")));
  // A real quote, but from another field.
  assert.ok(reasonsOf(judgment([issue({ summaryQuote: "仅覆盖光学选择的星团" })])).some(reason => reason.includes("summaryQuote")));
  assert.ok(reasonsOf(judgment([issue({ summaryQuote: "" })])).some(reason => reason.includes("summaryQuote")));
  assert.ok(reasonsOf(judgment([issue({ summaryQuote: 7 })])).some(reason => reason.includes("summaryQuote")));
  // Whitespace is normalised on both sides: a note paragraph break is a space.
  const ok = check(judgment([issue({ location: "核心结果", summaryQuote: "$0.85\\pm0.04$。 样本为 120" })]), NOTE);
  assert.equal(ok.ok, true, JSON.stringify(ok));
});

test("the source quote must be an exact piece of the source", () => {
  assert.ok(reasonsOf(judgment([issue({ sourceQuote: "using 130 clusters" })])).some(reason => reason.includes("issues[0].sourceQuote")));
  assert.equal(check(judgment([issue({ sourceQuote: "calibrate against simulations", location: "keyMethod", summaryQuote: "叠加弱透镜剪切" })])).ok, true);
  assert.equal(check(judgment([issue({ sourceQuote: "$0.85 \\pm 0.04$" })])).ok, true);
  assert.ok(reasonsOf(judgment([issue({ sourceQuote: 5 })])).some(reason => reason.includes("sourceQuote")));
  assert.ok(reasonsOf(judgment([issue({ sourceQuote: "" })])).some(reason => reason.includes("sourceQuote")));
});

test("a completeness issue must quote what was left out", () => {
  const value = judgment([issue({ dimension: "completeness", severity: "major", sourceQuote: null, explanation: "漏掉了核心结果。" })]);
  assert.ok(reasonsOf(value).some(reason => reason.includes("completeness") && reason.includes("sourceQuote")));
  assert.equal(check(judgment([issue({ dimension: "completeness", severity: "major", explanation: "漏掉了样本量。" })])).ok, true);
});

test("an accuracy issue may leave the source quote null only by saying the claim is absent from the source", () => {
  const silent = judgment([issue({ sourceQuote: null, explanation: "这个数值不对。" })]);
  assert.ok(reasonsOf(silent).some(reason => reason.includes("原文未提及")));
  assert.equal(check(judgment([issue({ sourceQuote: null, explanation: "原文未提及样本量为 130 个。" })])).ok, true);
});

test("a clarity issue needs no source quote", () => {
  assert.equal(check(judgment([issue({ dimension: "clarity", severity: "minor", sourceQuote: null, explanation: "缩写未定义。" })])).ok, true);
});

test("the verdicts must agree with the issues", () => {
  const issues = [issue({ severity: "minor" }), issue({ dimension: "clarity", severity: "major", location: "keyMethod", summaryQuote: "叠加弱透镜剪切", sourceQuote: null, explanation: "不清楚。" })];
  assert.ok(reasonsOf({ verdicts: verdicts("ok", "ok", "major"), issues }).some(reason => reason.includes("accuracy") && reason.includes("minor")));
  assert.ok(reasonsOf({ verdicts: verdicts("minor", "ok", "minor"), issues }).some(reason => reason.includes("clarity") && reason.includes("major")));
  assert.ok(reasonsOf({ verdicts: verdicts("minor", "major", "major"), issues }).some(reason => reason.includes("completeness")));
  assert.equal(check({ verdicts: verdicts("minor", "ok", "major"), issues }).ok, true);
});

test("the output parser accepts one code fence and repairs lone backslashes in LaTeX", () => {
  const body = JSON.stringify(judgment());
  assert.equal(qe.parseJudgeOutput(body).ok, true);
  assert.equal(qe.parseJudgeOutput(`\`\`\`json\n${body}\n\`\`\``).ok, true);
  assert.equal(qe.parseJudgeOutput(`  \`\`\`\n${body}\n\`\`\`  \n`).ok, true);
  assert.equal(qe.parseJudgeOutput(`好的，结果如下：\n${body}`).ok, false);
  assert.equal(qe.parseJudgeOutput(`\`\`\`json\n${body}\n\`\`\`\n\`\`\`json\n${body}\n\`\`\``).ok, false);
  assert.match(qe.parseJudgeOutput("{not json").reason, /JSON/);
  const latex = '{"verdicts":{"accuracy":"ok","completeness":"ok","clarity":"ok"},"issues":[],"x":"$0.85\\pm0.04$"}';
  assert.equal(qe.parseJudgeOutput(latex).value.x, "$0.85\\pm0.04$");
});

test("checkJudgeOutput parses and validates in one step with a reason for each failure", () => {
  const good = qe.checkJudgeOutput(`\`\`\`json\n${JSON.stringify(judgment([issue()]))}\n\`\`\``, DAILY);
  assert.equal(good.ok, true);
  const unparsable = qe.checkJudgeOutput("sorry", DAILY);
  assert.deepEqual([unparsable.ok, unparsable.reasons.length], [false, 1]);
  const invalid = qe.checkJudgeOutput(JSON.stringify(judgment([issue({ location: "x" })])), DAILY);
  assert.equal(invalid.ok, false);
  assert.ok(invalid.reasons[0].includes("issues[0].location"));
});

test("the correction message names every failed reason and repeats the contract", () => {
  const message = qe.buildJudgeCorrection(["issues[0].location 不是允许的位置", "verdicts.accuracy 与问题不一致"]);
  assert.ok(message.includes("issues[0].location 不是允许的位置"));
  assert.ok(message.includes("verdicts.accuracy 与问题不一致"));
  assert.match(message, /只输出.*JSON/);
  assert.match(message, /逐字/);
});

test("the daily system prompt renders fully and carries the generation prompt and the guard", async () => {
  const prompt = qe.buildJudgeSystemPrompt({ kind: "daily" });
  assert.ok(!/\{\{\w+\}\}/.test(prompt), "no placeholder is left");
  const generation = await readFile(repoFile("packages/core/src/prompts/daily-paper-summary.system.md"), "utf8");
  assert.ok(prompt.includes("`mainResult`：先直接给出最重要的结论"));
  assert.ok(prompt.includes(generation.split("{{injectionGuard}}")[0].trim().split("\n").slice(0, 3).join("\n")));
  const guard = (await readFile(repoFile("packages/core/src/prompts/injection-guard.md"), "utf8")).trim();
  assert.ok(prompt.includes(guard));
  for (const word of ["准确", "完整", "清楚", "major", "minor", "<paper_data>"]) assert.ok(prompt.includes(word), word);
  for (const key of qe.SUMMARY_FIELDS) assert.ok(prompt.includes(key), key);
  assert.match(prompt, /格式[^\n]*长度[^\n]*不要评/s);
  assert.match(prompt, /日报[^\n]*快速/);
  assert.ok(!prompt.includes("学术价值判断"), "the note variant is not mixed in");
});

test("the note system prompt renders fully, names the headings and carries the detail generation prompt", async () => {
  const prompt = qe.buildJudgeSystemPrompt({ kind: "note", headings: ["研究问题", "核心结果", "适用边界"], topic: "星团" });
  assert.ok(!/\{\{\w+\}\}/.test(prompt));
  assert.ok(prompt.includes("学术价值判断"), "the generation prompt's own sections are quoted as reference");
  assert.ok(prompt.includes("「星团」"), "the topic the generator was told is rendered");
  assert.ok(prompt.includes("原文未讨论"));
  for (const heading of ["研究问题", "核心结果", "适用边界"]) assert.ok(prompt.includes(`"${heading}"`), heading);
  assert.match(prompt, /详细笔记|完整/);
  assert.ok(!prompt.includes("coreProblem"), "the daily variant is not mixed in");
  assert.throws(() => qe.buildJudgeSystemPrompt({ kind: "note", headings: [], topic: "x" }), /headings/);
});

test("the prompt template and the contract agree on the output shape", () => {
  const prompt = qe.buildJudgeSystemPrompt({ kind: "daily" });
  for (const key of ["verdicts", "issues", "dimension", "severity", "location", "summaryQuote", "sourceQuote", "explanation", "accuracy", "completeness", "clarity"]) {
    assert.ok(prompt.includes(`"${key}"`), key);
  }
  assert.ok(prompt.includes("原文未提及"));
});
