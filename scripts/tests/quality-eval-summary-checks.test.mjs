import assert from "node:assert/strict";
import test, { before } from "node:test";
import { loadQualityEval } from "../quality-eval/load.mjs";

let qe;
before(async () => { qe = await loadQualityEval(); });

const CLEAN = {
  coreProblem: "星团质量标定能否摆脱对X射线静力学平衡假设的依赖。",
  keyMethod: "以弱透镜剪切叠加测得质量，并与 $M_{500}$ 的X射线估计比较。",
  mainResult: "弱透镜质量与X射线质量之比为 $0.85\\pm0.04$，在 $z<0.5$ 的子样本中更低，说明静力学平衡假设在低红移星团中偏差更明显，且结果对质心选择不敏感，在不同半径内保持一致，样本量为120个，误差主要来自剪切校准。",
  whyRelevant: "该比值直接决定宇宙学约束中使用的质量偏差先验取值范围。",
  limitations: "仅覆盖光学选择的星团，未检验X射线选择样本的偏差。",
};
const entry = (fields = {}, overrides = {}) => ({
  date: "2026-10-28", topic: "Galaxy Cluster", id: "2610.00001", title: "Title",
  fields: { ...CLEAN, ...fields }, ...overrides,
});
const findings = (fields, kind) => qe.checkEntry(entry(fields)).findings.filter(finding => !kind || finding.kind === kind);
const codes = (fields, kind) => findings(fields, kind).map(finding => `${finding.field}:${finding.code}`);

test("a clean entry has no findings and measures each field", () => {
  const result = qe.checkEntry(entry());
  assert.deepEqual(result.findings, []);
  assert.deepEqual(Object.keys(result.lengths), qe.SUMMARY_FIELDS);
  assert.equal(result.claimCount, 0);
  assert.equal(result.id, "2610.00001");
});

test("length counts characters outside math and whitespace", () => {
  assert.equal(qe.measureFieldLength("一二 三 $x_{1}$ 四，"), 5);
  assert.equal(qe.measureFieldLength("$a$ $b$"), 0);
  assert.equal(qe.measureFieldLength("abc 𠮷"), 4);
});

test("mainResult length is flagged outside 80-180 as a soft finding only", () => {
  const short = findings({ mainResult: "结果较好，误差很小。" }, "length");
  assert.deepEqual(short.map(finding => [finding.code, finding.soft]), [["main-result-short", true]]);
  const long = findings({ mainResult: "结果".repeat(100) }, "length");
  assert.deepEqual(long.map(finding => [finding.code, finding.soft]), [["main-result-long", true]]);
  assert.equal(qe.checkEntry(entry({ mainResult: "字".repeat(80) })).findings.filter(finding => finding.kind === "length").length, 0);
  assert.equal(qe.checkEntry(entry({ mainResult: "字".repeat(180) })).findings.filter(finding => finding.kind === "length").length, 0);
  // Other fields have no numeric range.
  assert.deepEqual(findings({ coreProblem: "短" }, "length"), []);
});

test("math: well-formed inline math raises nothing", () => {
  assert.deepEqual(findings({ keyMethod: "测量 $M_{*}=10^{10}\\,M_{\\odot}$，并要求 $z<0.5$ 与 $a$ 和 $b$ 无关。" }, "math"), []);
  assert.deepEqual(findings({ keyMethod: "比较 $a$、$b$ 与 $c$，或 $x$ 以及 $y$。" }, "math"), []);
});

test("math: core issue codes are recorded per field", () => {
  assert.deepEqual(codes({ keyMethod: "使用 \\alpha 作为参数" }, "math"), ["keyMethod:bare-tex"]);
  assert.deepEqual(codes({ limitations: "误差为 $\\sigma_{8" }, "math"), ["limitations:unclosed-delimiter"]);
});

test("math: adjacent split fragments are found, independent formulas are not", () => {
  assert.deepEqual(codes({ keyMethod: "质量 $M_{*}$ $=10^{10}$ 太阳质量" }, "math"), ["keyMethod:split-fragment"]);
  // Touching spans also trip core's nested-delimiter check; the split is still named.
  assert.ok(codes({ keyMethod: "质量 $M_{*}$$=10^{10}$ 太阳质量" }, "math").includes("keyMethod:split-fragment"));
  assert.deepEqual(codes({ keyMethod: "比值 $a$ = $b$" }, "math"), ["keyMethod:split-fragment"]);
  assert.deepEqual(codes({ keyMethod: "差 $a$ + $b$ 与 $c$ ± $d$" }, "math"), ["keyMethod:split-fragment", "keyMethod:split-fragment"]);
  assert.deepEqual(codes({ keyMethod: "$a$ 和 $b$ 以及 $c$，$d$ 与 $e$-$f$ 关系" }, "math"), []);
  // A slash lists two quantities (or two values of one); it does not join them into one formula.
  assert.deepEqual(codes({ keyMethod: "对 $\\sigma_8$/$S_8$ 与 $+37\\%$ / $-37\\%$ 的影响" }, "math"), []);
});

test("math: \\(..\\) and \\[..\\] delimiters are found even though core would accept and rewrite them", () => {
  assert.equal(qe.canonicalizeScientificMarkdownMath("质量 \\(M_{*}\\) 较大").ok, true);
  assert.deepEqual(codes({ keyMethod: "质量 \\(M_{*}\\) 较大" }, "math").filter(code => code.endsWith("paren-delimiter")), ["keyMethod:paren-delimiter"]);
  assert.deepEqual(codes({ keyMethod: "质量 \\[M_{*}\\] 较大" }, "math").filter(code => code.endsWith("paren-delimiter")), ["keyMethod:paren-delimiter"]);
});

test("math: $$ display math is found", () => {
  assert.ok(codes({ keyMethod: "公式 $$a=b$$ 成立" }, "math").includes("keyMethod:display-math"));
});

test("math: renderer escaping of [[ and ![ does not become a math issue", () => {
  const rendered = "与 \\[\\[2601.00001]] 对照，并见 !\\[\\[fig]] 与 $x_{1}$";
  assert.equal(qe.canonicalizeScientificMarkdownMath(rendered).ok, false);
  assert.equal(qe.unescapeRenderedField(rendered), "与 [[2601.00001]] 对照，并见 ![[fig]] 与 $x_{1}$");
  assert.deepEqual(findings({ whyRelevant: rendered }, "math"), []);
  // Inside math the renderer leaves text alone, so nothing is unescaped there.
  assert.equal(qe.unescapeRenderedField("$\\[\\[$ 与 \\[\\[x]]"), "$\\[\\[$ 与 [[x]]");
});

test("not-stated: exact and partial 原文未说明 are told apart", () => {
  assert.deepEqual(codes({ limitations: "原文未说明" }, "not-stated"), ["limitations:exact"]);
  assert.deepEqual(codes({ limitations: "原文未说明。" }, "not-stated"), ["limitations:exact"]);
  const partial = findings({ limitations: "样本量原文未说明，其余适用于低红移。" }, "not-stated");
  assert.deepEqual(partial.map(finding => finding.code), ["partial"]);
  assert.ok(partial[0].snippet.includes("原文未说明"));
  assert.deepEqual(findings({ limitations: "原文说明了全部限制。" }, "not-stated"), []);
});

test("作者声称 is counted as information, not as a finding", () => {
  const result = qe.checkEntry(entry({ mainResult: CLEAN.mainResult + "作者声称更优。", limitations: "作者声称普适，但作者声称的证据不足。" }));
  assert.equal(result.claimCount, 3);
  assert.equal(result.findings.filter(finding => finding.kind === "not-stated").length, 0);
});

test("generic phrases from the prompts are found, with a shared exported list", () => {
  assert.ok(qe.BANNED_GENERIC_PHRASES.includes("具有重要意义"));
  assert.ok(qe.BANNED_GENERIC_PHRASES.includes("提高了理解"));
  assert.deepEqual(codes({ whyRelevant: "该工作具有重要意义。" }, "generic-phrase"), ["whyRelevant:具有重要意义"]);
  assert.deepEqual(codes({ whyRelevant: "这提高了理解，且具有重要意义。" }, "generic-phrase").sort(), ["whyRelevant:具有重要意义", "whyRelevant:提高了理解"]);
  assert.deepEqual(findings({ whyRelevant: "它改变了质量偏差先验的取值。" }, "generic-phrase"), []);
});

test("English text does not crash and gets no Chinese length flag", () => {
  const english = {
    coreProblem: "Can weak lensing calibrate cluster masses without hydrostatic equilibrium?",
    keyMethod: "Stack shear profiles and compare with $M_{500}$ from X-ray data.",
    mainResult: "The mass ratio is $0.85\\pm0.04$.",
    whyRelevant: "It sets the mass bias prior used in cosmological fits.",
    limitations: "Only optically selected clusters; the authors claim robustness.",
  };
  const result = qe.checkEntry(entry(english));
  assert.deepEqual(result.findings, []);
  assert.equal(result.claimCount, 0);
  assert.deepEqual(qe.checkEntry(entry({ ...english, whyRelevant: "This has important significance." })).findings.map(finding => finding.code), ["important significance"]);
});

test("checkSummaries aggregates rates, distributions and the fallback rate", () => {
  const entries = [
    entry({}, { id: "2609.00001", date: "2026-09-10" }),
    entry({ keyMethod: "质量 $M$ $=1$" }, { id: "2610.00001", date: "2026-10-01" }),
    entry({ limitations: "原文未说明。", whyRelevant: "具有重要意义。" }, { id: "2610.00002", date: "2026-10-02" }),
    entry({ mainResult: "短", coreProblem: "作者声称。" }, { id: "2610.00003", date: "2026-10-03" }),
  ];
  const fallbacks = [{ date: "2026-10-01", topic: null, id: "2610.09999", title: "F" }];
  const { results, aggregate } = qe.checkSummaries({ entries, fallbacks });

  assert.equal(results.length, 4);
  assert.deepEqual(aggregate.fallback.overall, { count: 1, of: 5, rate: 0.2 });
  assert.deepEqual(aggregate.fallback.byMonth, {
    "2026-09": { count: 0, of: 1, rate: 0 }, "2026-10": { count: 1, of: 4, rate: 0.25 },
  });
  assert.equal(aggregate.entries, 4);
  assert.deepEqual(aggregate.entriesWith.math, { count: 1, of: 4, rate: 0.25 });
  assert.deepEqual(aggregate.entriesWith["not-stated"], { count: 1, of: 4, rate: 0.25 });
  assert.deepEqual(aggregate.entriesWith["generic-phrase"], { count: 1, of: 4, rate: 0.25 });
  assert.deepEqual(aggregate.codes.math, { "split-fragment": 1 });
  assert.deepEqual(aggregate.notStated.limitations, { exact: 1, partial: 0 });
  assert.deepEqual(aggregate.mainResultOutsideRange, { count: 1, of: 4, rate: 0.25 });
  assert.equal(aggregate.claims.occurrences, 1);
  assert.equal(aggregate.claims.byField.coreProblem, 1);
  assert.deepEqual(aggregate.claims.entriesWith, { count: 1, of: 4, rate: 0.25 });

  const main = aggregate.lengths.mainResult;
  assert.equal(main.min, 1);
  assert.equal(main.max, results[0].lengths.mainResult);
  assert.equal(main.p90, main.max);
  assert.ok(main.median >= main.min && main.median <= main.max);
});

test("checkSummaries copes with an empty corpus", () => {
  const { results, aggregate } = qe.checkSummaries({ entries: [], fallbacks: [] });
  assert.deepEqual(results, []);
  assert.deepEqual(aggregate.fallback.overall, { count: 0, of: 0, rate: 0 });
  assert.deepEqual(aggregate.lengths.mainResult, { min: 0, median: 0, p90: 0, max: 0 });
});

test("checkSummaries leaves a missing field out of the length distribution", () => {
  const { aggregate } = qe.checkSummaries({ entries: [entry(), entry({ mainResult: "" })], fallbacks: [] });
  assert.equal(aggregate.lengths.mainResult.min, aggregate.lengths.mainResult.max);
  assert.ok(aggregate.lengths.mainResult.min > 0);
});
