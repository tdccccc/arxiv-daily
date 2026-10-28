import assert from "node:assert/strict";
import test, { before } from "node:test";
import { loadQualityEval } from "../quality-eval/load.mjs";

let qe;
before(async () => { qe = await loadQualityEval(); });

const tokens = text => qe.extractNumbers(text).map(number => number.token);

test("extraction reads plain numbers, uncertainties and ranges, inside and outside math", () => {
  assert.deepEqual(tokens("测得 $\\Omega_m=0.30\\pm0.02$，另有 0.85±0.04"), ["0.30", "0.02", "0.85", "0.04"]);
  assert.deepEqual(tokens("范围 $0.8$–$0.95$ 与 0.4-0.6 以及 $12$–$15$"), ["0.8", "0.95", "0.4", "0.6", "12", "15"]);
  assert.deepEqual(tokens("$H_0=67.4$ km/s/Mpc，$z=0.5$"), ["67.4", "0.5"]);
  assert.deepEqual(tokens("样本 2019 年"), ["2019"]);
  assert.deepEqual(tokens("$M_\\star>10^{10.5}M_\\odot$ 与 $10^{10.3}$"), ["1e10.5", "1e10.3"]);
  assert.deepEqual(tokens("$^{+0.1}_{-0.2}$ 误差"), ["0.1", "-0.2"]);
});

test("extraction unifies minus signs and keeps the sign in the token", () => {
  assert.deepEqual(tokens("斜率 -0.5"), ["-0.5"]);
  assert.deepEqual(tokens("斜率 $-0.5$"), ["-0.5"]);
  assert.deepEqual(tokens("斜率 −0.5"), ["-0.5"]);
  assert.deepEqual(tokens("$z=-0.5$ 与 (−1.25)"), ["-0.5", "-1.25"]);
  // A dash between two numbers is a range, not a minus.
  assert.deepEqual(tokens("0.4–0.6 与 0.4−0.6"), ["0.4", "0.6", "0.4", "0.6"]);
});

test("extraction drops thousands separators in every common spelling", () => {
  for (const text of ["1,234", "1{,}234", "$1\\,234$", "1 234", "$1{,}234$", "1 234"]) assert.deepEqual(tokens(text), ["1234"], text);
  assert.deepEqual(tokens("3,963,602 个源"), ["3963602"]);
  assert.deepEqual(tokens("9,000 deg2"), ["9000"]);
  // A comma and a space is a list, not a thousands separator.
  assert.deepEqual(tokens("共 12, 345 项"), ["12", "345"]);
});

test("a comma between digits is a list when the number is an uncertainty or more comma-digits follow", () => {
  assert.deepEqual(tokens("$\\sigma_v=135\\pm67,150\\pm85,232\\pm88$"), ["135", "67", "150", "85", "232", "88"]);
  assert.deepEqual(tokens("$l_f=256,102.4,64\\,{\\rm kpc}$"), ["256", "102.4", "64"]);
  assert.deepEqual(tokens("共 1,234 与 1,234,567 与 12,345.67"), ["1234", "1234567", "12345.67"]);
});

test("a sentence-ending period after a thousands run does not break the run", () => {
  assert.deepEqual(tokens("the number is 4,454,600. The average"), ["4454600"]);
  assert.deepEqual(tokens("数量为 4,454,600。以及 (4,454,600)."), ["4454600", "4454600"]);
  assert.deepEqual(tokens("$4\\,454\\,600$"), ["4454600"]);
  assert.deepEqual(ground("$4\\,454\\,600$ 个源", "within R200m from BCGs is 4,454,600. The average number"), [["4454600", true, "exact"]]);
  // Still lists, with or without a trailing period.
  assert.deepEqual(tokens("$\\sigma_v=135\\pm67,150\\pm85,232$."), ["135", "67", "150", "85", "232"]);
  assert.deepEqual(tokens("scales 256,102.4,64."), ["256", "102.4", "64"]);
});

test("a decimal written without its leading zero is the same number", () => {
  assert.deepEqual(tokens("$p=.028$"), ["0.028"]);
  assert.deepEqual(tokens("($p=.028$)"), ["0.028"]);
  assert.deepEqual(tokens("值 .5 与 -.25 与 (.75)"), ["0.5", "-0.25", "0.75"]);
  // A dot that ends a sentence or follows a digit is not a decimal point.
  assert.deepEqual(tokens("共 120. 然后 3.5.7 与 a.5"), ["120", "3.5"]);
  assert.deepEqual(ground("$p=0.028$", "differences ($p=.028$)"), [["0.028", true, "exact"]]);
  assert.deepEqual(ground("$p=.028$", "differences p = 0.028"), [["0.028", true, "exact"]]);
  assert.deepEqual(ground("$p=0.029$", "differences ($p=.028$)"), [["0.029", false, null]]);
});

test("author-year citations are not numbers", () => {
  assert.deepEqual(tokens("与 Higuchi+19、Helton+24 比较，Burgett et al. 2004 与 Smith et al. (2019)"), []);
  assert.deepEqual(tokens("样本 2019 年"), ["2019"]);
});

test("extraction keeps scientific notation as one value with its mantissa and exponent", () => {
  const [sci] = qe.extractNumbers("$2\\times10^{44}$ erg/s");
  assert.equal(sci.token, "2e44");
  assert.equal(sci.value, 2e44);
  assert.equal(sci.mantissa, "2");
  assert.equal(sci.exponent, 44);
  for (const text of ["2×10^{44}", "$2 \\times 10^{44}$", "$2\\,\\times\\,10^{44}$", "2\\cdot10^{44}", "2×10⁴⁴", "2e44"]) assert.deepEqual(tokens(text), ["2e44"], text);
  assert.deepEqual(tokens("$1.5\\times10^{-3}$"), ["1.5e-3"]);
  assert.deepEqual(tokens("$10^{10}\\,M_{\\odot}$"), ["1e10"]);
  assert.equal(qe.extractNumbers("$10^{10}$")[0].mantissa, "1");
  assert.deepEqual(tokens("$M_{*}=10^{10}$"), ["1e10"]);
});

test("extraction reads percents and scale words", () => {
  const [percent] = qe.extractNumbers("提高 28%");
  assert.equal(percent.token, "28%");
  assert.equal(percent.percent, true);
  assert.deepEqual(tokens("$28\\%$ 与 28 per cent 与 28 percent"), ["28%", "28%", "28%"]);
  const [million] = qe.extractNumbers("18.7 million objects");
  assert.equal(million.token, "18.7");
  assert.equal(million.value, 18_700_000);
  assert.equal(qe.extractNumbers("1870万个")[0].value, 18_700_000);
  assert.equal(qe.extractNumbers("约 2.5 亿")[0].value, 250_000_000);
  // The unit may sit after the closing dollar of the math the number is in.
  assert.equal(qe.extractNumbers("约 $33$ 万个卫星")[0].value, 330_000);
});

test("extraction skips small standalone integers, references, identifiers and the paper id", () => {
  assert.deepEqual(tokens("两个，3 倍，1–2 个，10 kpc，$3\\sigma$，$5\\sigma$，z=0"), []);
  assert.deepEqual(tokens("$12\\sigma$ 与 11 个"), ["12", "11"]);
  // Not "standing alone": decimals, percents, scale words and notation keep small values.
  assert.deepEqual(tokens("1.0 与 5% 与 3 million"), ["1.0", "5%", "3"]);
  assert.deepEqual(tokens("见 Figure 3 和 Table 12，Fig. 14a、Sec. 3.2、Section 4、Eq. (15)、Appendix B2"), []);
  assert.deepEqual(tokens("见图 5、表 13，式 (25)，第 14 节，附录 A"), []);
  assert.deepEqual(tokens("arXiv:2610.12345v2 与 2610.12345"), []);
  assert.deepEqual(tokens("DR10, PS1DR2, MACSJ0417.8+1234, A2537, W1, Y1, LSDR10, DESI DR1"), []);
  assert.deepEqual(tokens("$M_{500}$, $M_{200c}$, $\\sigma_8$, $H_0$, $z_{\\rm phot}$, $r^2$, cm^{-2}"), []);
  assert.deepEqual(tokens("先前结果 [33] 和 [77, 78]，引用 [12–15]"), []);
  assert.deepEqual(tokens("1. **剪切恢复精度**：内容\n2. 第二项"), []);
  assert.deepEqual(tokens("详见 https://example.org/path/2024/12345"), []);
});

test("source extraction keeps the small integers and the same masks", () => {
  assert.deepEqual(qe.extractNumbers("we use 3 bins at 10 kpc, Figure 3", { keepSmallIntegers: true }).map(number => number.token), ["3", "10"]);
});

test("a number carries the text around it", () => {
  const [number] = qe.extractNumbers("质量比为 $0.85\\pm0.04$，高于基线");
  assert.ok(number.context.includes("0.85"));
  assert.ok(number.context.includes("质量比"));
});

const ground = (summary, source, options) => {
  const index = qe.buildSourceIndex(Array.isArray(source) ? source : [source]);
  return qe.groundNumbers(qe.extractNumbers(summary, options), index).map(number => [number.token, number.grounded, number.via ?? null]);
};

test("a number found in the source is grounded; one that is not is flagged", () => {
  assert.deepEqual(ground("比值 0.85", "the ratio is 0.85 ± 0.03"), [["0.85", true, "exact"]]);
  assert.deepEqual(ground("比值 0.86", "the ratio is 0.85 ± 0.03"), [["0.86", false, null]]);
  assert.deepEqual(ground("样本 451 个", "a sample of 449 clusters"), [["451", false, null]]);
});

test("a different spelling of the same value is grounded", () => {
  assert.deepEqual(ground("$2\\times10^{44}$", "L = 2.0×10^{44} erg/s"), [["2e44", true, "exact"]]);
  assert.deepEqual(ground("3963602 个源", "matched 3,963,602 sources"), [["3963602", true, "exact"]]);
  assert.deepEqual(ground("$\\pm0.04$", "$0.85 \\pm 0.04$"), [["0.04", true, "exact"]]);
  assert.deepEqual(ground("1.2", "value of $1.2$ here"), [["1.2", true, "exact"]]);
  assert.deepEqual(ground("-0.5", "slope −0.5"), [["-0.5", true, "exact"]]);
  assert.deepEqual(ground("0.5", "slope −0.5"), [["0.5", true, "exact"]]);
  assert.deepEqual(ground("$10^{10}$", "mass $10^{10}\\,M_\\odot$"), [["1e10", true, "exact"]]);
});

test("a number rounded to the summary's precision is grounded, but invented precision is not", () => {
  assert.deepEqual(ground("0.25", "measured 0.248"), [["0.25", true, "rounded"]]);
  assert.deepEqual(ground("12.3", "equal to 12.34"), [["12.3", true, "rounded"]]);
  assert.deepEqual(ground("0.248", "measured 0.25"), [["0.248", false, null]]);
  assert.deepEqual(ground("0.26", "measured 0.2551"), [["0.26", true, "rounded"]]);
  assert.deepEqual(ground("0.25", "measured 0.2551"), [["0.25", false, null]]);
  // Whole numbers are not rounded: 12 is not 12.4.
  assert.deepEqual(ground("12", "equal to 12.4 and 13"), [["12", false, null]]);
  assert.deepEqual(ground("1.5\\times10^{14}", "mass 1.53×10^{14}"), [["1.5e14", true, "rounded"]]);
});

test("a round number is read as rounded to its trailing zeros", () => {
  assert.deepEqual(ground("约 41,000 个", "the catalogue contains 41,430 galaxies"), [["41000", true, "rounded"]]);
  assert.deepEqual(ground("约 450 个", "a sample of 449 clusters"), [["450", true, "rounded"]]);
  assert.deepEqual(ground("约 450 个", "a sample of 460 clusters"), [["450", false, null]]);
  assert.deepEqual(ground("约 41,000 个", "contains 42,000 galaxies and 41,600"), [["41000", false, null]]);
  // Below 100 a trailing zero is just a number.
  assert.deepEqual(ground("50 个", "a sample of 49 clusters"), [["50", false, null]]);
  assert.deepEqual(ground("5830 万条", "to 58,291,334 spectra"), [["5830", true, "converted"]]);
});

test("percent and fraction forms match each other", () => {
  assert.deepEqual(ground("25%", "a fraction 0.25"), [["25%", true, "converted"]]);
  assert.deepEqual(ground("0.25", "a share of 25%"), [["0.25", true, "converted"]]);
  assert.deepEqual(ground("28%", "a share of 28 per cent"), [["28%", true, "exact"]]);
  assert.deepEqual(ground("28%", "a fraction 0.31"), [["28%", false, null]]);
  assert.deepEqual(ground("25%", "measured 0.248"), [["25%", true, "converted"]]);
});

test("scale words and Chinese units match their expanded numbers", () => {
  assert.deepEqual(ground("1870万个", "18.7 million objects"), [["1870", true, "converted"]]);
  assert.deepEqual(ground("396万", "matched 3,963,602 sources"), [["396", true, "converted"]]);
  assert.deepEqual(ground("11.4 million", "11.4 million spectra"), [["11.4", true, "exact"]]);
  assert.deepEqual(ground("2.5 亿", "250,000,000 sources"), [["2.5", true, "converted"]]);
});

test("citations, figure numbers and identifiers in the source do not ground anything", () => {
  assert.deepEqual(ground("相差 33", "see previous work [33] and Figure 33 and DR33"), [["33", false, null]]);
  assert.deepEqual(ground("相差 0.33", "see [0.33]"), [["0.33", true, "exact"]]);
  assert.deepEqual(ground("样本 2000 个", "arXiv:2610.02000 reports it"), [["2000", false, null]]);
});

test("small standalone integers in the summary are not checked at all", () => {
  assert.deepEqual(ground("两个样本，3 倍，z=0", "nothing numeric here"), []);
});

test("checkGrounding splits the numbers by where they were written", () => {
  const source = { abstract: "ratio 0.85", abstractConclusion: "## Abstract\nratio 0.85", fullSections: "## Methods\nwe use 120 clusters and 3.5 sigma" };
  const result = qe.checkGrounding([
    { where: "mainResult", text: "比值 0.85，样本 120 个，误差 0.99" },
    { where: "limitations", text: "仅 7 个星团" },
  ], source);
  assert.equal(result.total, 3);
  assert.equal(result.grounded, 2);
  assert.deepEqual(result.ungrounded.map(number => [number.where, number.token]), [["mainResult", "0.99"]]);
  assert.deepEqual(result.byWhere, { mainResult: { grounded: 2, total: 3 }, limitations: { grounded: 0, total: 0 } });
  assert.ok(result.ungrounded[0].context.includes("0.99"));
  assert.equal(qe.checkGrounding([{ where: "a", text: "无数字" }], source).total, 0);
  // Abs-only content has no full text: only the abstract can ground.
  assert.equal(qe.checkGrounding([{ where: "a", text: "样本 120 个" }], { ...source, fullSections: null }).grounded, 0);
});
