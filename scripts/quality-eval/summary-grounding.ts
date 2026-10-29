import { distribution, rate, type LengthDistribution, type Rate } from "./summary-checks";

/**
 * What the extraction leaves out because it is not a claim about the paper. One entry per kind, so the report and
 * the code say the same thing. Integers 0-10 standing alone are counts, ordinals and labels ("两个", "3σ", "1–2 个").
 */
export const SKIPPED_NUMBER_KINDS = [
  "standalone integers 0-10 (no decimals, percent, scale word or scientific notation)",
  "figure, table, section, equation, appendix and panel numbers (Figure 3, 图 5, 式 (25), 第 4 节)",
  "numbered-list markers and numbered headings",
  "citation brackets ([33], [77, 78], [12–15])",
  "the digits of identifiers: letters directly followed by digits (DR10, PS1DR2, W1, A2537), object names with coordinates (MACSJ0417.8+1234)",
  "arXiv ids and URLs",
  "author-year citations (Higuchi+19, Burgett et al. 2004)",
  "subscript labels and integer exponents of non-ten bases (M_{500}, σ_8, r^2, cm^{-2})",
] as const;

export const STANDALONE_INTEGER_SKIP = { min: 0, max: 10 } as const;

const REFERENCE_PATTERNS: readonly RegExp[] = [
  /https?:\/\/\S+/g,
  // A new-style id: YYMM from 2007 on, then four or five digits.
  /(?<![\d.,])(?:0[7-9]|[12]\d)(?:0[1-9]|1[0-2])\.\d{4,5}(?:v\d+)?(?!\d)/g,
  /\b(?:Figs?|Figures?|Tables?|Sections?|Secs?|Eqs?|Equations?|Appendix|Appendices|Panels?)\b\.?[ \t]*\(?[A-Z]?\d+(?:\.\d+)*[a-z]?\)?/gi,
  /图[ \t]*\d+(?:[-–]\d+)?[a-z]?/g,
  /表[ \t]*\d+[a-z]?/g,
  /第[ \t]*\d+[ \t]*(?:节|章|部分|图|表|页|步)/g,
  /(?:式|公式|方程|等式)[ \t]*[（(]?[ \t]*\d+[ \t]*[）)]?/g,
  /附录[ \t]*[A-Z]?\d*/g,
  /\[[ \t]*\d+(?:[ \t]*[,，–-][ \t]*\d+)*[ \t]*\]/g,
  /\bet al\.?[,;]?[ \t]*\(?\d{4}[a-z]?\)?/g,
  /(?<=[A-Za-z]\+)\d{2,4}\b/g,
  /^[ \t]*(?:\d{1,2}[.)、]|[（(]\d{1,2}[）)])[ \t]+/gm,
  /^#{1,6}[ \t]+(?:[A-Z]\.)?\d+(?:\.\d+)*\.?(?=\s)/gm,
];
// Object names with coordinates keep their digits as part of the name; masked whole.
const OBJECT_NAME = /(?<![\\A-Za-z\d.])[A-Za-z][A-Za-z0-9]*\d+(?:\.\d+)?[+\-−]\d+(?:\.\d+)?/g;
// Letters directly followed by digits are an identifier. Not after a backslash (`\times10`, `\pm0.04`) or a digit (`2e44`).
const IDENTIFIER_DIGITS = /(?<![\\A-Za-z\d.])([A-Za-zͰ-Ͽ]+)(\d+(?:\.\d+)?)/g;

const SEP = String.raw`(?:,|\{,\}|\\,|\;|\\!|[    ])`;
// Thousands grouping is only read when the digits are not a list: not after a ±, and not followed by more comma-digits
// (`135\pm67,150\pm85`, `256,102.4,64` are lists of numbers).
const GROUPED = String.raw`\d{1,3}(?:${SEP}\d{3})+`;
const NUM = String.raw`(?:(?<!\\pm[ \t]*|±[ \t]*)(?<![\d.])${GROUPED}(?:\.\d+(?!,\d)|(?!\.\d)(?!\d)(?!,\d))|\d+(?:\.\d+)?)`;
const SP = String.raw`(?:\s|\\[,;!: ]|~|\{\})*`;
const SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹";

const SCI_TIMES = new RegExp(String.raw`(?<![\d.])(${NUM})${SP}(?:\\times|\\cdot|×|·|\*)${SP}10${SP}\^${SP}\{?${SP}([-−–+]?\d+(?:\.\d+)?)${SP}\}?`, "g");
const SCI_TIMES_SUPERSCRIPT = new RegExp(String.raw`(?<![\d.])(${NUM})${SP}(?:\\times|\\cdot|×|·|\*)${SP}10([⁻⁺]?[${SUPERSCRIPT_DIGITS}]+)`, "g");
const SCI_E = /(?<![\w.])(\d+(?:\.\d+)?)[eE]([-+]?\d+)(?![\w.])/g;
const POWER_OF_TEN = new RegExp(String.raw`(?<![\d.])10${SP}\^${SP}\{?${SP}([-−–+]?\d+(?:\.\d+)?)${SP}\}?`, "g");
const POWER_OF_TEN_SUPERSCRIPT = new RegExp(String.raw`(?<![\d.])10([⁻⁺]?[${SUPERSCRIPT_DIGITS}]+)`, "g");
// Labels and squares, not claims: `_{500}`, `_8`, `^2`, `^{-2}`. A signed or decimal exponent (`^{+0.1}_{-0.2}`) is an error bar and stays.
const SUBSCRIPT = /_\{(?![+\-−])(?:[^{}]|\{[^{}]*\})*\}|_\d+(?:\.\d+)?/g;
const INTEGER_EXPONENT = /\^[ \t]*\{?[ \t]*[-−+]?\d+[ \t]*\}?(?![\d.])/g;
// A decimal may be written without its leading zero (`.028`) when it does not follow a digit, letter or dot.
const PLAIN = new RegExp(
  String.raw`(?:(?<![\d.])(${NUM})|(?<![\w.])\.(\d+))(?:[ \t]*(\\?%|％|per[ \t]?cent\b|percent\b)|[ \t]*(million|billion|trillion|thousand)\b|(?:\$[ \t]*|[ \t]*)(千万|百万|万|亿))?`, "gi");

const SCALE_WORDS: Record<string, number> = {
  thousand: 1e3, million: 1e6, billion: 1e9, trillion: 1e12, 万: 1e4, 亿: 1e8, 千万: 1e7, 百万: 1e6,
};
const MINUS = /[-−]/;
// What may stand before a minus sign for it to be a sign and not a dash between two numbers.
const SIGN_CONTEXT = /[\s(\[{$=<>,;:：，；±~∼]/;
const CONTEXT_BEFORE = 24;
const CONTEXT_AFTER = 32;

export interface ExtractedNumber {
  /** Canonical form: sign, digits without separators, `%` for percents, `<mantissa>e<exponent>` for scientific notation. */
  token: string;
  /** The number's value with scale words applied (as a percent it stays the written number). */
  value: number;
  /** Digits after the decimal point as written (of the mantissa for scientific notation). */
  decimals: number;
  percent: boolean;
  /** 1, or 1e4 for 万, 1e6 for "million". */
  multiplier: number;
  mantissa?: string;
  exponent?: number;
  /** The text around it, whitespace collapsed. */
  context: string;
  offset: number;
}

export interface ExtractOptions {
  /** Keep small standalone integers: the source side wants every number it has. */
  keepSmallIntegers?: boolean;
}

function blank(text: string, patterns: readonly RegExp[]): string {
  let masked = text;
  for (const pattern of patterns) masked = masked.replace(pattern, (match) => " ".repeat(match.length));
  masked = masked.replace(OBJECT_NAME, (match) => " ".repeat(match.length));
  return masked.replace(IDENTIFIER_DIGITS, (_match, letters: string, digits: string) => letters + " ".repeat(digits.length));
}

const stripSeparators = (digits: string): string => digits.replace(/\{,\}|\\[,;!]|[,\s   ]/g, "");

function superscriptToInteger(value: string): number {
  const negative = value.startsWith("⁻");
  const digits = [...value].filter((char) => SUPERSCRIPT_DIGITS.includes(char)).map((char) => SUPERSCRIPT_DIGITS.indexOf(char)).join("");
  return (negative ? -1 : 1) * Number(digits);
}

const parseExponent = (value: string): number => Number(value.replace(/[−–]/g, "-").replace(/^\+/, ""));

interface Candidate { start: number; end: number; priority: number; number?: Omit<ExtractedNumber, "context" | "offset"> }

function signBefore(text: string, start: number): string {
  const sign = text[start - 1];
  if (sign === undefined || !MINUS.test(sign)) return "";
  const before = text[start - 2];
  return before === undefined || SIGN_CONTEXT.test(before) ? "-" : "";
}

/** The numbers a text states, in reading order. The skip list is {@link SKIPPED_NUMBER_KINDS}. */
export function extractNumbers(text: string, options: ExtractOptions = {}): ExtractedNumber[] {
  const masked = blank(text, REFERENCE_PATTERNS);
  const candidates: Candidate[] = [];
  const collect = (pattern: RegExp, priority: number, build: (match: RegExpExecArray) => Candidate["number"] | undefined) => {
    for (const match of masked.matchAll(pattern)) {
      const number = build(match);
      candidates.push({ start: match.index, end: match.index + match[0].length, priority, number });
    }
  };
  const scientific = (mantissa: string, exponent: number, start: number): Candidate["number"] => {
    const clean = stripSeparators(mantissa);
    const sign = signBefore(masked, start);
    return {
      token: `${sign}${clean}e${exponent}`,
      value: (sign ? -1 : 1) * (Number.isInteger(exponent) ? Number(`${clean}e${exponent}`) : Number(clean) * 10 ** exponent), decimals: clean.split(".")[1]?.length ?? 0,
      percent: false, multiplier: 1, mantissa: `${sign}${clean}`, exponent,
    };
  };
  collect(SCI_TIMES, 0, (m) => scientific(m[1]!, parseExponent(m[2]!), m.index));
  collect(SCI_TIMES_SUPERSCRIPT, 0, (m) => scientific(m[1]!, superscriptToInteger(m[2]!), m.index));
  collect(SCI_E, 0, (m) => scientific(m[1]!, parseExponent(m[2]!), m.index));
  collect(POWER_OF_TEN, 1, (m) => scientific("1", parseExponent(m[1]!), m.index));
  collect(POWER_OF_TEN_SUPERSCRIPT, 1, (m) => scientific("1", superscriptToInteger(m[1]!), m.index));
  collect(SUBSCRIPT, 2, () => undefined);
  collect(INTEGER_EXPONENT, 2, () => undefined);
  collect(PLAIN, 3, (m) => {
    const clean = m[1] ? stripSeparators(m[1]) : `0.${m[2]}`;
    const percent = Boolean(m[3]);
    const multiplier = m[4] ? SCALE_WORDS[m[4].toLowerCase()]! : m[5] ? SCALE_WORDS[m[5]]! : 1;
    const sign = signBefore(masked, m.index);
    return {
      token: `${sign}${clean}${percent ? "%" : ""}`, value: Number(`${sign}${clean}`) * multiplier,
      decimals: clean.split(".")[1]?.length ?? 0, percent, multiplier,
    };
  });

  candidates.sort((a, b) => a.start - b.start || a.priority - b.priority || b.end - a.end);
  const numbers: ExtractedNumber[] = [];
  let cursor = 0;
  for (const candidate of candidates) {
    if (candidate.start < cursor) continue;
    cursor = candidate.end;
    const number = candidate.number;
    if (!number) continue;
    const standaloneInteger = number.exponent === undefined && !number.percent && number.multiplier === 1 && number.decimals === 0
      && Math.abs(number.value) >= STANDALONE_INTEGER_SKIP.min && Math.abs(number.value) <= STANDALONE_INTEGER_SKIP.max;
    if (standaloneInteger && !options.keepSmallIntegers) continue;
    numbers.push({
      ...number, offset: candidate.start,
      context: text.slice(Math.max(0, candidate.start - CONTEXT_BEFORE), candidate.end + CONTEXT_AFTER).replace(/\s+/g, " ").trim(),
    });
  }
  return numbers;
}

// ---- matching ---------------------------------------------------------------------------------------------

/** How a value was written: as is, as a percent, as a fraction derived from a percent, or with a scale word. */
type Form = "plain" | "pct" | "frac" | `mult:${number}`;
/** `half` is only used on the summary side: the range of values that would have been written this way. */
interface Entry { value: number; form: Form; half: number }

export interface SourceIndex { values: Entry[] }
export interface SourceTexts { abstract: string; abstractConclusion: string; fullSections: string | null }

const EPSILON = 1e-9;

/** Trailing zeros of a whole number of at least 100 ("41000", "450"): the zeros may be rounding, not digits. */
function roundingZeros(number: ExtractedNumber): number {
  if (number.decimals > 0) return 0;
  const digits = number.token.replace(/^-/, "").replace(/%/g, "");
  return Number(digits) >= 100 ? (/0+$/.exec(digits)?.[0].length ?? 0) : 0;
}

/**
 * Half the last written place: the range of values that would have been written this way. A plain whole number
 * has none unless it ends in zeros (read as rounded to them); a percent or a scaled number always has one.
 */
function entriesOf(number: ExtractedNumber): Entry[] {
  const magnitude = Math.abs(number.value);
  const place = 0.5 * 10 ** -number.decimals;
  const zeros = roundingZeros(number);
  const wholeHalf = zeros > 0 ? 0.5 * 10 ** zeros : 0;
  if (number.exponent !== undefined) return [{ value: magnitude, form: "plain", half: place * 10 ** number.exponent }];
  if (number.percent) {
    return [{ value: magnitude, form: "pct", half: place }, { value: magnitude / 100, form: "frac", half: place / 100 }];
  }
  if (number.multiplier !== 1) {
    return [
      { value: magnitude, form: `mult:${number.multiplier}`, half: (number.decimals === 0 ? 0.5 * 10 ** zeros : place) * number.multiplier },
      { value: magnitude / number.multiplier, form: "plain", half: number.decimals === 0 ? wholeHalf : place },
    ];
  }
  return [{ value: magnitude, form: "plain", half: number.decimals === 0 ? wholeHalf : place }];
}

/** Every number the source states, as magnitudes in each form it might be quoted in. */
export function buildSourceIndex(texts: readonly string[]): SourceIndex {
  const values = texts.flatMap((text) => extractNumbers(text, { keepSmallIntegers: true }).flatMap(entriesOf));
  values.sort((a, b) => a.value - b.value);
  return { values };
}

export function indexOfSource(source: SourceTexts): SourceIndex {
  return buildSourceIndex([source.abstract, source.abstractConclusion, source.fullSections ?? ""]);
}

export type GroundedVia = "exact" | "rounded" | "converted";

export interface GroundedNumber extends ExtractedNumber { grounded: boolean; via?: GroundedVia }

const RANK: Record<GroundedVia, number> = { exact: 0, rounded: 1, converted: 2 };

function lowerBound(values: readonly Entry[], target: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (values[middle]!.value < target) low = middle + 1;
    else high = middle;
  }
  return low;
}

/**
 * A number is grounded when the source states the same value: equal, or equal once the source is rounded to the
 * summary's own precision (0.25 for 0.248, never 0.248 for 0.25), including the percent/fraction pair and scale
 * words. Whole numbers must match exactly. Signs are not compared: a dash and a minus are too easily confused.
 */
export function groundNumbers(numbers: readonly ExtractedNumber[], index: SourceIndex): GroundedNumber[] {
  return numbers.map((number) => {
    let best: GroundedVia | undefined;
    for (const candidate of entriesOf(number)) {
      const slack = EPSILON * Math.max(1, candidate.value);
      for (let at = lowerBound(index.values, candidate.value - candidate.half - slack); at < index.values.length; at += 1) {
        const entry = index.values[at]!;
        if (entry.value > candidate.value + candidate.half + slack) break;
        const via: GroundedVia = entry.form !== candidate.form ? "converted" : Math.abs(entry.value - candidate.value) <= slack ? "exact" : "rounded";
        if (best === undefined || RANK[via] < RANK[best]) best = via;
      }
    }
    return best === undefined ? { ...number, grounded: false } : { ...number, grounded: true, via: best };
  });
}

// ---- checking text against a source ---------------------------------------------------------------------

export interface WrittenText { where: string; text: string }

export interface UngroundedNumber { where: string; token: string; context: string }

export interface GroundingResult {
  total: number;
  grounded: number;
  byWhere: Record<string, { grounded: number; total: number }>;
  ungrounded: UngroundedNumber[];
}

/** Drop fenced code blocks: numbers in code are not claims about the paper. */
export function stripFencedCode(text: string): string {
  const kept: string[] = [];
  let fence: { char: string; length: number } | null = null;
  for (const line of text.split(/\r?\n/)) {
    const fenced = /^ {0,3}(`{3,}|~{3,})(.*)$/u.exec(line);
    if (fenced) {
      const run = fenced[1]!;
      if (!fence) fence = { char: run[0]!, length: run.length };
      else if (run[0] === fence.char && run.length >= fence.length && !fenced[2]!.trim()) fence = null;
      continue;
    }
    if (!fence) kept.push(line);
  }
  return kept.join("\n");
}

/** Look every number of the written texts up in a source, given as texts or as an index built once for it. */
export function checkGrounding(written: readonly WrittenText[], source: SourceTexts | SourceIndex): GroundingResult {
  const index = "values" in source ? source : indexOfSource(source);
  const result: GroundingResult = { total: 0, grounded: 0, byWhere: {}, ungrounded: [] };
  for (const { where, text } of written) {
    const slot = (result.byWhere[where] ??= { grounded: 0, total: 0 });
    for (const number of groundNumbers(extractNumbers(text), index)) {
      slot.total += 1;
      result.total += 1;
      if (number.grounded) {
        slot.grounded += 1;
        result.grounded += 1;
      } else {
        result.ungrounded.push({ where, token: number.token, context: number.context });
      }
    }
  }
  return result;
}

// ---- aggregates --------------------------------------------------------------------------------------------

export interface GroundingAggregate {
  /** Entries (or notes) that stated at least one number. */
  withNumbers: number;
  numbers: Rate;
  /** Per field (daily) or per section heading (notes). */
  byPlace: Record<string, Rate>;
  withUngrounded: Rate;
  /** Spread of the grounded share per entry or note, among those with numbers. */
  perItemShare: LengthDistribution;
}

export function aggregateGrounding(items: readonly GroundingResult[]): GroundingAggregate {
  const numbers = items.reduce((sum, item) => ({ grounded: sum.grounded + item.grounded, total: sum.total + item.total }), { grounded: 0, total: 0 });
  const places = new Map<string, { grounded: number; total: number }>();
  for (const item of items) {
    for (const [where, value] of Object.entries(item.byWhere)) {
      const slot = places.get(where) ?? { grounded: 0, total: 0 };
      slot.grounded += value.grounded;
      slot.total += value.total;
      places.set(where, slot);
    }
  }
  const withNumbers = items.filter((item) => item.total > 0);
  return {
    withNumbers: withNumbers.length,
    numbers: rate(numbers.grounded, numbers.total),
    byPlace: Object.fromEntries([...places].map(([where, value]) => [where, rate(value.grounded, value.total)])),
    withUngrounded: rate(withNumbers.filter((item) => item.grounded < item.total).length, withNumbers.length),
    perItemShare: distribution(withNumbers.map((item) => item.grounded / item.total)),
  };
}
