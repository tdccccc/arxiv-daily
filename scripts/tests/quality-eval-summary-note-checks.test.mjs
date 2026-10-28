import assert from "node:assert/strict";
import test, { before } from "node:test";
import { loadQualityEval } from "../quality-eval/load.mjs";

let qe;
before(async () => { qe = await loadQualityEval(); });

// Written out here on purpose: the test proves the headings really are derived from the prompt files.
const ZH_HEADINGS = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "贡献与创新点", "适用边界", "学术价值判断"];
const EN_HEADINGS = [
  "Research Problem", "Method Design", "Core Results", "Main Conclusions",
  "Key Figures and Tables", "Contributions and Novelty", "Scope and Limits", "Academic Value Assessment",
];

const BODY = "该论文测得 $\\Omega_m=0.30\\pm0.02$，并与 $\\Lambda$CDM 预测一致。";
const sectionsFor = (headings, bodies = {}) => headings.map(heading => ({ heading, body: bodies[heading] ?? BODY }));
const note = (overrides = {}) => ({
  id: "2607.00001", topic: "galaxy-cluster", path: "arxiv-daily/papers/2607.00001.md",
  title: "A title", date: "2026-07-10", sections: sectionsFor(ZH_HEADINGS), ...overrides,
});
const withBody = (bodies, headings = ZH_HEADINGS) => note({ sections: sectionsFor(headings, bodies) });
const kinds = (result, kind) => result.findings.filter(finding => finding.kind === kind);
const codes = (result, kind) => kinds(result, kind).map(finding => `${finding.heading}:${finding.code}`);

test("current headings are derived from the prompt files", () => {
  assert.deepEqual(qe.CURRENT_NOTE_HEADINGS.zh, ZH_HEADINGS);
  assert.deepEqual(qe.CURRENT_NOTE_HEADINGS.en, EN_HEADINGS);
});

test("a note with exactly the current headings is current and clean", () => {
  const result = qe.checkNote(note());
  assert.equal(result.version, "current");
  assert.deepEqual(result.findings, []);
  assert.deepEqual(result.headingSequence, ZH_HEADINGS);
  assert.deepEqual(result.missingHeadings, []);
});

test("an English note with the English headings is current", () => {
  const result = qe.checkNote(note({ sections: sectionsFor(EN_HEADINGS, { "Research Problem": "Plain English prose." }) }));
  assert.equal(result.version, "current");
  assert.deepEqual(result.findings, []);
});

test("an older-prompt note is a version, not a violation", () => {
  const older = ["研究问题", "方法设计", "关键证据", "主要结论", "适用边界", "一句话价值判断"];
  const result = qe.checkNote(note({ sections: sectionsFor(older) }));
  assert.equal(result.version, "older");
  assert.deepEqual(kinds(result, "structure"), []);
  assert.deepEqual(result.missingHeadings, ["核心结果", "关键图表", "贡献与创新点", "学术价值判断"]);
  assert.deepEqual(result.headingSequence, older);
});

test("a current note with an extra heading is flagged", () => {
  const headings = [...ZH_HEADINGS.slice(0, 5), "补充说明", ...ZH_HEADINGS.slice(5)];
  const result = qe.checkNote(note({ sections: sectionsFor(headings) }));
  assert.equal(result.version, "current");
  assert.deepEqual(codes(result, "structure"), ["补充说明:extra-heading"]);
});

test("a current note with headings out of order is flagged once", () => {
  const swapped = [...ZH_HEADINGS];
  [swapped[2], swapped[3]] = [swapped[3], swapped[2]];
  const result = qe.checkNote(note({ sections: sectionsFor(swapped) }));
  assert.equal(result.version, "current");
  assert.deepEqual(codes(result, "structure"), ["null:out-of-order"]);
  assert.ok(result.findings[0].snippet.includes("核心结果"));
});

test("a repeated current heading is flagged as a duplicate", () => {
  const result = qe.checkNote(note({ sections: sectionsFor([...ZH_HEADINGS, "适用边界"]) }));
  assert.deepEqual(codes(result, "structure"), ["适用边界:duplicate-heading"]);
});

test("placeholders are counted per section heading", () => {
  const result = withBody({
    适用边界: "原文未讨论局限；原文未讨论外推。",
    贡献与创新点: "原文未说明",
    关键图表: "输入中未包含图表。",
    学术价值判断: "原文信息不足以判断。",
  });
  const checked = qe.checkNote(result);
  assert.deepEqual(checked.placeholders, {
    适用边界: { 原文未讨论: 2 },
    贡献与创新点: { 原文未说明: 1 },
    关键图表: { 输入中未包含图表: 1 },
    学术价值判断: { 原文信息不足以判断: 1 },
  });
  assert.deepEqual(qe.checkNote(note()).placeholders, {});
  assert.equal(qe.checkNote(withBody({ 适用边界: "Not discussed in the source text." })).placeholders["适用边界"]["Not discussed in the source text"], 1);
});

test("generic phrases are found per section using the shared list", () => {
  const result = qe.checkNote(withBody({ 学术价值判断: "这项工作具有重要意义，提高了理解。" }));
  assert.deepEqual(codes(result, "generic-phrase").sort(), ["学术价值判断:具有重要意义", "学术价值判断:提高了理解"]);
  assert.deepEqual(kinds(qe.checkNote(note()), "generic-phrase"), []);
});

test("math: well-formed inline and $$ display math raises nothing", () => {
  const display = "公式如下：\n\n$$\n\\alpha = \\frac{\\beta}{\\gamma}\n$$\n\n单行 $$E=mc^2$$ 也可以。其中 $a$ 和 $b$ 独立。";
  assert.deepEqual(kinds(qe.checkNote(withBody({ 方法设计: display })), "math"), []);
});

test("math: text inside fenced code is not checked", () => {
  const fenced = "示例：\n\n```python\nx = r'\\alpha \\(y\\) $a$ $b$ $$'\n```\n\n之后恢复检查。";
  assert.deepEqual(kinds(qe.checkNote(withBody({ 方法设计: fenced })), "math"), []);
  const after = qe.checkNote(withBody({ 方法设计: fenced + "\n裸 \\alpha 在公式外" }));
  assert.deepEqual(codes(after, "math"), ["方法设计:bare-tex"]);
});

test("math: bare TeX, paren delimiters and split fragments are found", () => {
  assert.deepEqual(codes(qe.checkNote(withBody({ 核心结果: "红移 \\alpha 偏高" })), "math"), ["核心结果:bare-tex"]);
  assert.deepEqual(codes(qe.checkNote(withBody({ 核心结果: "质量 \\(M_{*}\\) 较大" })), "math"), ["核心结果:paren-delimiter"]);
  assert.deepEqual(codes(qe.checkNote(withBody({ 核心结果: "质量 \\[M\\] 较大" })), "math"), ["核心结果:paren-delimiter"]);
  assert.deepEqual(codes(qe.checkNote(withBody({ 核心结果: "质量 $M_{*}$ $=10^{10}$ 太阳质量" })), "math"), ["核心结果:split-fragment"]);
});

test("math: each line is checked on its own and an unclosed $$ block is named", () => {
  const multiline = qe.checkNote(withBody({ 核心结果: "第一行 $a$ 正常\n第二行 $b 未闭合" }));
  assert.deepEqual(codes(multiline, "math"), ["核心结果:unclosed-delimiter"]);
  const unclosed = qe.checkNote(withBody({ 核心结果: "开始\n$$\n\\alpha = 1\n后面没有结束" }));
  assert.deepEqual(codes(unclosed, "math"), ["核心结果:unclosed-display-math"]);
});

test("math: a multi-line \\[ .. \\] block is one finding, not one per line", () => {
  const block = "测得：\n  \\[\n  S_{144}=19\\pm2,\n  \\]\n  \\[\n  \\alpha=0.9,\n  \\]\n之后正常 $a$。";
  const result = qe.checkNote(withBody({ 核心结果: block }));
  assert.deepEqual(codes(result, "math"), ["核心结果:paren-delimiter", "核心结果:paren-delimiter"]);
});

test("math: list markers, links and bold labels do not raise issues", () => {
  const figures = "- **Figure 3**：展示了红移分布 $z<0.5$，见 [arXiv](https://arxiv.org/abs/2607.00001)。\n- **Table 1**：参数 $H_0=70$。";
  assert.deepEqual(kinds(qe.checkNote(withBody({ 关键图表: figures })), "math"), []);
});

test("english share counts Latin words against Han characters in prose only", () => {
  // Math, link targets, inline code, fenced code and display blocks are not prose; link text is.
  const prose = "测得红移 redshift [link text](https://example.org/path) `code word` $\\Omega_m$\n```\nignored latin words\n```\n$$\nx y z\n$$";
  assert.equal(qe.checkNote(withBody({ 研究问题: prose }, ["研究问题"])).englishShare, 3 / 7);
  const exact = qe.checkNote(note({ sections: [{ heading: "研究问题", body: "测得红移 redshift photometric galaxy cluster" }] }));
  assert.equal(exact.englishShare, 4 / 8);
  assert.equal(qe.checkNote(note({ sections: [{ heading: "研究问题", body: "全中文内容" }] })).englishShare, 0);
  assert.equal(qe.checkNote(note({ sections: [{ heading: "研究问题", body: "$x$ $$y$$" }] })).englishShare, null);
  assert.equal(qe.checkNote(note({ sections: EN_HEADINGS.map(heading => ({ heading, body: "Only English words here." })) })).englishShare, 1);
});

test("checkNotes groups notes by exact heading sequence and aggregates", () => {
  const older = ["研究问题", "方法设计", "关键证据", "主要结论", "适用边界", "一句话价值判断"];
  const notes = [
    note({ id: "2607.00001" }),
    note({ id: "2607.00002", sections: sectionsFor(ZH_HEADINGS, { 适用边界: "原文未讨论。", 学术价值判断: "具有重要意义。" }) }),
    note({ id: "2607.00003", sections: sectionsFor(older) }),
    note({ id: "2607.00004", sections: sectionsFor(older, { 核心结果: "x" }) }),
    note({ id: "2607.00005", sections: sectionsFor([...ZH_HEADINGS.slice(0, 5), "补充说明", ...ZH_HEADINGS.slice(5)]) }),
  ];
  const { results, aggregate } = qe.checkNotes(notes);

  assert.equal(results.length, 5);
  assert.equal(aggregate.notes, 5);
  assert.deepEqual(aggregate.versions, { current: 3, older: 2 });
  assert.deepEqual(aggregate.groups.map(group => [group.version, group.count]), [["current", 2], ["older", 2], ["current", 1]]);
  assert.deepEqual(aggregate.groups[0].sequence, ZH_HEADINGS);
  assert.deepEqual(aggregate.groups[0].exampleIds, ["2607.00001", "2607.00002"]);
  assert.deepEqual(aggregate.structure, { notes: { count: 1, of: 3, rate: 1 / 3 }, codes: { "extra-heading": 1 } });
  assert.deepEqual(aggregate.notesWith["generic-phrase"], { count: 1, of: 5, rate: 0.2 });
  assert.deepEqual(aggregate.notesWith.math, { count: 0, of: 5, rate: 0 });
  assert.deepEqual(aggregate.placeholders, { 适用边界: { 原文未讨论: { notes: 1, occurrences: 1 } } });
  assert.ok(aggregate.englishShare.max >= aggregate.englishShare.min);
});

test("checkNotes copes with no notes", () => {
  const { results, aggregate } = qe.checkNotes([]);
  assert.deepEqual(results, []);
  assert.equal(aggregate.notes, 0);
  assert.deepEqual(aggregate.groups, []);
  assert.deepEqual(aggregate.versions, { current: 0, older: 0 });
  assert.deepEqual(aggregate.englishShare, { min: 0, median: 0, p90: 0, max: 0 });
});

test("math: digit-leading ratios and half-open intervals in a note are known false positives", () => {
  const detail = withBody({ 核心结果: "质量比约 $6:1$，冲量参数可由 $500$ kpc 类比。\n范围 $g \\in (0, 0.15]$ 内恢复。\n同行有 \\(x\\) 的真问题。" });
  const result = qe.checkNote(detail);
  const math = kinds(result, "math");
  assert.ok(math.some(finding => finding.knownFalsePositive === true && finding.code !== "paren-delimiter"));
  const real = math.filter(finding => !finding.knownFalsePositive);
  assert.deepEqual(real.map(finding => finding.code), ["paren-delimiter"]);
  const { aggregate } = qe.checkNotes([detail]);
  assert.deepEqual(aggregate.notesWith.math, { count: 1, of: 1, rate: 1 });
  assert.deepEqual(aggregate.codes.math, { "paren-delimiter": 1 });
  assert.equal(aggregate.coreFalsePositives.notes, 1);
});
