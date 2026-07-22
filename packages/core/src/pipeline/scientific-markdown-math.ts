import { containsRawHtmlConstruct } from "./raw-html";

export const SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES = {
  BARE_TEX: "bare-tex",
  DISPLAY_ENVIRONMENT: "display-environment",
  DISPLAY_LINE_BREAK: "display-line-break",
  DISPLAY_MATH: "display-math",
  EMPTY_MATH: "empty-math",
  MULTILINE: "multiline",
  NESTED_DELIMITER: "nested-delimiter",
  RAW_HTML_IN_MATH: "raw-html-in-math",
  UNBALANCED_BRACES: "unbalanced-braces",
  UNBALANCED_OPTIONAL_ARGUMENT: "unbalanced-optional-argument",
  UNBALANCED_PARENTHESES: "unbalanced-parentheses",
  UNCLOSED_DELIMITER: "unclosed-delimiter",
  UNMATCHED_DELIMITER: "unmatched-delimiter",
  UNMATCHED_LEFT_RIGHT: "unmatched-left-right",
} as const;

export type ScientificMarkdownMathIssueCode =
  (typeof SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES)[keyof typeof SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES];

export interface ScientificMarkdownMathIssue {
  code: ScientificMarkdownMathIssueCode;
  offset: number;
  length: number;
}

export type ScientificMarkdownMathResult =
  | { ok: true; value: string; issues: readonly [] }
  | { ok: false; value: string; issues: readonly ScientificMarkdownMathIssue[] };

interface Delimiter {
  open: "$" | "\\(" | "\\[";
  close: "$" | "\\)" | "\\]";
}

const TEX_COMMAND = /\\[A-Za-z]+/g;
const DISPLAY_ENVIRONMENT = /\\(?:begin|end)\s*\{(?:equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|displaymath|eqnarray\*?|split|cases|matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix)\}/g;
const AUTOLINK_URI = /^[A-Za-z][A-Za-z0-9+.-]{1,31}:[^\x00-\x20<>]*$/u;
const AUTOLINK_EMAIL = /^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$/u;

function isEscaped(value: string, offset: number): boolean {
  let slashes = 0;
  for (let index = offset - 1; index >= 0 && value[index] === "\\"; index -= 1) {
    slashes += 1;
  }
  return slashes % 2 === 1;
}

function markRange(mask: boolean[], start: number, end: number): void {
  for (let index = start; index < end; index += 1) mask[index] = true;
}

interface MarkdownDestination {
  start: number;
  end: number;
  next: number;
}

function parseMarkdownDestination(value: string, start: number): MarkdownDestination | undefined {
  if (value[start] === "<") {
    for (let index = start + 1; index < value.length; index += 1) {
      if (value[index] === ">" && !isEscaped(value, index)) {
        return { start, end: index + 1, next: index + 1 };
      }
      if (value[index] === "\n" || value[index] === "\r" || value[index] === "<") return undefined;
    }
    return undefined;
  }

  let depth = 0;
  let index = start;
  for (; index < value.length; index += 1) {
    if (isEscaped(value, index)) continue;
    const character = value[index];
    if (character === "(") depth += 1;
    else if (character === ")") {
      if (depth === 0) break;
      depth -= 1;
    } else if (character === "\n" || character === "\r" || /\s/u.test(character ?? "")) break;
  }
  if (index === start || depth !== 0) return undefined;
  return { start, end: index, next: index };
}

function skipLinkTitle(value: string, start: number): number | undefined {
  let index = start;
  while (value[index] === " " || value[index] === "\t") index += 1;
  const opener = value[index];
  const closer = opener === "(" ? ")" : opener;
  if (opener !== "\"" && opener !== "'" && opener !== "(") return start;
  if (index === start) return start;
  index += 1;
  for (; index < value.length; index += 1) {
    if (value[index] === "\n" || value[index] === "\r") return undefined;
    if (value[index] === closer && !isEscaped(value, index)) return index + 1;
  }
  return undefined;
}

function hasValidInlineLinkSuffix(value: string, start: number): boolean {
  const afterTitle = skipLinkTitle(value, start);
  if (afterTitle === undefined) return false;
  let index = afterTitle;
  while (value[index] === " " || value[index] === "\t") index += 1;
  return value[index] === ")";
}

function hasValidReferenceDefinitionSuffix(value: string, start: number): boolean {
  const afterTitle = skipLinkTitle(value, start);
  if (afterTitle === undefined) return false;
  return value.slice(afterTitle).trim().length === 0;
}

function hasLinkLabel(value: string, close: number): boolean {
  for (let index = close - 1; index >= 0; index -= 1) {
    if (value[index] === "[" && !isEscaped(value, index)) return true;
    if (value[index] === "]" && !isEscaped(value, index)) return false;
  }
  return false;
}

function protectedMarkdownMask(value: string): boolean[] {
  const mask = Array<boolean>(value.length).fill(false);

  for (let start = 0; start < value.length;) {
    if (value[start] !== "`" || isEscaped(value, start)) {
      start += 1;
      continue;
    }
    let runEnd = start + 1;
    while (value[runEnd] === "`") runEnd += 1;
    const run = value.slice(start, runEnd);
    let close = value.indexOf(run, runEnd);
    while (close >= 0 && (
      isEscaped(value, close) ||
      value[close - 1] === "`" ||
      value[close + run.length] === "`"
    )) {
      close = value.indexOf(run, close + run.length);
    }
    if (close < 0) {
      start = runEnd;
      continue;
    }
    markRange(mask, start, close + run.length);
    start = close + run.length;
  }

  for (let start = 0; start < value.length; start += 1) {
    if (mask[start] || value[start] !== "<") continue;
    const close = value.indexOf(">", start + 1);
    if (close < 0) continue;
    const body = value.slice(start + 1, close);
    if (AUTOLINK_URI.test(body) || AUTOLINK_EMAIL.test(body)) markRange(mask, start, close + 1);
  }

  for (let start = 1; start < value.length - 1; start += 1) {
    if (mask[start] || value[start - 1] !== "]" || value[start] !== "(" || !hasLinkLabel(value, start - 1)) continue;
    let destinationStart = start + 1;
    while (value[destinationStart] === " " || value[destinationStart] === "\t") destinationStart += 1;
    if (value[destinationStart] === ")") continue;
    const destination = parseMarkdownDestination(value, destinationStart);
    if (destination && hasValidInlineLinkSuffix(value, destination.next)) {
      let suffixEnd = skipLinkTitle(value, destination.next) ?? destination.next;
      while (value[suffixEnd] === " " || value[suffixEnd] === "\t") suffixEnd += 1;
      markRange(mask, destination.start, suffixEnd);
    }
  }

  const definition = value.match(/^ {0,3}\[[^\]\n]+\]:[ \t]*/u);
  if (definition) {
    const destination = parseMarkdownDestination(value, definition[0].length);
    if (destination && hasValidReferenceDefinitionSuffix(value, destination.next)) {
      const suffixEnd = skipLinkTitle(value, destination.next) ?? destination.next;
      markRange(mask, destination.start, suffixEnd);
    }
  }

  return mask;
}

function findClosingDollar(value: string, offset: number, protectedMask: boolean[]): number | undefined {
  for (let index = offset + 1; index < value.length; index += 1) {
    if (!protectedMask[index] && value[index] === "$" && !isEscaped(value, index)) return index;
  }
  return undefined;
}

function isConservativePairedDollarMath(
  value: string,
  open: number,
  close: number,
): boolean {
  const body = value.slice(open + 1, close);
  if (!body.trim() || /\s$/u.test(body)) return false;
  if (!/^\d/u.test(body)) return true;
  return /^\d+(?:,\d{3})*(?:\.\d+)?$/u.test(body) ||
    /[\\_^{}+*/=<>-]/u.test(body) ||
    /^\d+(?:\.\d+)?\s+[A-Za-z](?:\b|_)/u.test(body);
}

function isCurrencyDollar(value: string, offset: number): boolean {
  const amount = value.slice(offset + 1).match(/^\d+(?:,\d{3})*(?:\.\d+)?/u)?.[0];
  if (!amount) return false;
  const after = value[offset + 1 + amount.length];
  return after === undefined || /[\s,.;:!?%)\]}A-Za-z]/u.test(after);
}

function isLikelyProsePathCommand(value: string, offset: number): boolean {
  const tokenStart = Math.max(
    value.lastIndexOf(" ", offset - 1),
    value.lastIndexOf("\t", offset - 1),
    value.lastIndexOf("\n", offset - 1),
  ) + 1;
  const prefix = value.slice(tokenStart, offset);
  if (/^[A-Za-z]:(?:\\[A-Za-z0-9_. -]+)*$/u.test(prefix)) return true;
  return prefix.includes("/") && !prefix.includes("://");
}

function issue(
  issues: ScientificMarkdownMathIssue[],
  code: ScientificMarkdownMathIssueCode,
  offset: number,
  length: number,
): void {
  issues.push({ code, offset, length });
}

function isHalfOpenInterval(body: string, openOffset: number): boolean {
  const prefix = body.slice(0, openOffset).trimEnd();
  if (prefix !== "" && !prefix.endsWith("\\left")) return false;
  const candidate = body.slice(openOffset);
  return /^\(\s*[^,()[\]]+\s*,\s*[^,()[\]]+\s*(?:\\right)?\]\s*$/u.test(candidate);
}

function validateCommandOptionalArguments(
  body: string,
  bodyOffset: number,
  issues: ScientificMarkdownMathIssue[],
): void {
  const commandWithOptionalArgument = /\\[A-Za-z]+\s*\[/g;
  for (const match of body.matchAll(commandWithOptionalArgument)) {
    if (isEscaped(body, match.index)) continue;
    const open = match.index + match[0].lastIndexOf("[");
    let braceDepth = 0;
    let squareDepth = 1;
    let closed = false;
    for (let index = open + 1; index < body.length; index += 1) {
      if (isEscaped(body, index)) continue;
      if (body[index] === "{") braceDepth += 1;
      else if (body[index] === "}") braceDepth = Math.max(0, braceDepth - 1);
      else if (body[index] === "[" && braceDepth === 0) squareDepth += 1;
      else if (body[index] === "]" && braceDepth === 0) {
        squareDepth -= 1;
        if (squareDepth === 0) {
          closed = true;
          break;
        }
      }
    }
    if (!closed) {
      issue(
        issues,
        SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.UNBALANCED_OPTIONAL_ARGUMENT,
        bodyOffset + open,
        1,
      );
    }
  }
}

function validateMathBody(
  body: string,
  bodyOffset: number,
  issues: ScientificMarkdownMathIssue[],
): void {
  if (body.trim().length === 0) {
    issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.EMPTY_MATH, bodyOffset, body.length);
    return;
  }

  if (containsRawHtmlConstruct(body)) {
    issue(
      issues,
      SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.RAW_HTML_IN_MATH,
      bodyOffset + Math.max(0, body.indexOf("<")),
      1,
    );
  }

  validateCommandOptionalArguments(body, bodyOffset, issues);

  const braces: number[] = [];
  const parentheses: number[] = [];
  let textDepth: number | undefined;
  for (let index = 0; index < body.length; index += 1) {
    if (isEscaped(body, index)) continue;
    if (body.startsWith("\\text{", index)) textDepth = braces.length + 1;
    if (body[index] === "{") braces.push(index);
    else if (body[index] === "}") {
      if (braces.length === 0) {
        issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.UNBALANCED_BRACES, bodyOffset + index, 1);
      } else braces.pop();
      if (textDepth !== undefined && braces.length < textDepth) textDepth = undefined;
    } else if (textDepth === undefined && body[index] === "(") parentheses.push(index);
    else if (textDepth === undefined && body[index] === ")") {
      if (parentheses.length === 0) {
        issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.UNBALANCED_PARENTHESES, bodyOffset + index, 1);
      } else parentheses.pop();
    }
  }
  for (const offset of braces) {
    issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.UNBALANCED_BRACES, bodyOffset + offset, 1);
  }
  for (const offset of parentheses) {
    if (!isHalfOpenInterval(body, offset)) {
      issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.UNBALANCED_PARENTHESES, bodyOffset + offset, 1);
    }
  }

  const leftRight: Array<{ command: "left" | "right"; offset: number }> = [];
  const commandPattern = /\\(left|right)\b/g;
  for (const match of body.matchAll(commandPattern)) {
    if (!isEscaped(body, match.index)) {
      leftRight.push({ command: match[1] as "left" | "right", offset: match.index });
    }
  }
  const leftOffsets: number[] = [];
  for (const command of leftRight) {
    if (command.command === "left") leftOffsets.push(command.offset);
    else if (leftOffsets.length === 0) {
      issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.UNMATCHED_LEFT_RIGHT, bodyOffset + command.offset, 6);
    } else leftOffsets.pop();
  }
  for (const offset of leftOffsets) {
    issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.UNMATCHED_LEFT_RIGHT, bodyOffset + offset, 5);
  }
}

function isSuspiciousInnerDollar(value: string, offset: number): boolean {
  const next = value[offset + 1];
  const previous = value[offset - 1];
  return previous !== undefined && /\s/u.test(previous)
    && next !== undefined && !/\s/u.test(next);
}

/**
 * Canonicalize inline scientific Markdown math without guessing missing math boundaries.
 * Invalid input is returned byte-for-byte with stable, positional diagnostics.
 */
export function canonicalizeScientificMarkdownMath(value: string): ScientificMarkdownMathResult {
  const protectedMask = protectedMarkdownMask(value);
  const mathMask = Array<boolean>(value.length).fill(false);
  const issues: ScientificMarkdownMathIssue[] = [];
  const output: string[] = [];

  if (value.includes("\n") || value.includes("\r")) {
    const offset = Math.min(
      ...[value.indexOf("\n"), value.indexOf("\r")].filter((candidate) => candidate >= 0),
    );
    issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.MULTILINE, offset, 1);
  }

  const delimiters: readonly Delimiter[] = [
    { open: "\\(", close: "\\)" },
    { open: "\\[", close: "\\]" },
  ];

  for (let offset = 0; offset < value.length;) {
    if (protectedMask[offset]) {
      output.push(value[offset] ?? "");
      offset += 1;
      continue;
    }

    if (value.startsWith("$$", offset) && !isEscaped(value, offset)) {
      issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.DISPLAY_MATH, offset, 2);
      output.push("$$");
      offset += 2;
      continue;
    }

    let delimiter = delimiters.find(({ open }) => value.startsWith(open, offset) && !isEscaped(value, offset));
    if (value[offset] === "$" && !isEscaped(value, offset)) {
      const close = findClosingDollar(value, offset, protectedMask);
      const pairedMath = close !== undefined && isConservativePairedDollarMath(value, offset, close);
      if (pairedMath || !isCurrencyDollar(value, offset)) delimiter = { open: "$", close: "$" };
    }

    if (!delimiter) {
      if ((value.startsWith("\\)", offset) || value.startsWith("\\]", offset)) && !isEscaped(value, offset)) {
        issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.UNMATCHED_DELIMITER, offset, 2);
      }
      output.push(value[offset] ?? "");
      offset += 1;
      continue;
    }

    const bodyStart = offset + delimiter.open.length;
    let close = bodyStart;
    let foundClose = false;
    for (; close < value.length; close += 1) {
      if (protectedMask[close]) continue;
      if (delimiter.close === "$" && value[close] === "$" && !isEscaped(value, close)) {
        if (value[close + 1] === "$" || isSuspiciousInnerDollar(value, close)) {
          issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.NESTED_DELIMITER, close, value[close + 1] === "$" ? 2 : 1);
          if (value[close + 1] === "$") close += 1;
          continue;
        }
        foundClose = true;
        break;
      }
      if (delimiter.close !== "$" && value.startsWith(delimiter.close, close) && !isEscaped(value, close)) {
        foundClose = true;
        break;
      }
      const nested = value[close] === "$" || delimiters.some(({ open }) => value.startsWith(open, close));
      if (nested && !isEscaped(value, close)) {
        issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.NESTED_DELIMITER, close, value[close] === "$" ? 1 : 2);
      }
    }

    const spanEnd = foundClose ? close + delimiter.close.length : value.length;
    markRange(mathMask, offset, spanEnd);
    if (!foundClose) {
      issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.UNCLOSED_DELIMITER, offset, delimiter.open.length);
      output.push(value.slice(offset));
      break;
    }

    const body = value.slice(bodyStart, close);
    if (delimiter.open === "\\[" && /(?:^|[^\\])(?:\\\\){1,}/u.test(body)) {
      issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.DISPLAY_LINE_BREAK, bodyStart, body.length);
    }
    validateMathBody(body, bodyStart, issues);
    output.push(`$${body}$`);
    offset = spanEnd;
  }

  for (const match of value.matchAll(DISPLAY_ENVIRONMENT)) {
    if (!isEscaped(value, match.index) && !protectedMask[match.index]) {
      issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.DISPLAY_ENVIRONMENT, match.index, match[0].length);
    }
  }
  for (const match of value.matchAll(TEX_COMMAND)) {
    if (!isEscaped(value, match.index) && !isLikelyProsePathCommand(value, match.index) && !protectedMask[match.index] && !mathMask[match.index]) {
      issue(issues, SCIENTIFIC_MARKDOWN_MATH_ISSUE_CODES.BARE_TEX, match.index, match[0].length);
    }
  }

  issues.sort((left, right) => left.offset - right.offset || left.code.localeCompare(right.code));
  if (issues.length > 0) return { ok: false, value, issues };
  return { ok: true, value: output.join(""), issues: [] };
}
