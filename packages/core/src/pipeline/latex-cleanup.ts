/** Linear scans over author-controlled TeX; never search the same missing closer repeatedly. */
const MATH_ENVIRONMENTS = new Set([
  "math", "displaymath", "equation", "equation*", "align", "align*", "alignat", "alignat*",
  "eqnarray", "eqnarray*", "multline", "multline*", "gather", "gather*", "split",
]);

function escaped(text: string, offset: number): boolean {
  let start = offset;
  while (start > 0 && text[start - 1] === "\\") start--;
  return (offset - start) % 2 === 1;
}

function closingOffset(text: string, closer: string, start: number): number {
  let found = text.indexOf(closer, start);
  while (found >= 0) {
    if (!escaped(text, found)) return found;
    found = text.indexOf(closer, found + closer.length);
  }
  return -1;
}

/** Protect complete formulas before prose cleanup, including display environments. */
export function protectLatexMath(input: string): { text: string; restore: (value: string) => string } {
  const text = input.replace(/[\uE000\uE001]/g, "");
  const formulas: string[] = [];
  const missing = new Set<string>();
  let out = "", copied = 0;
  for (let i = 0; i < text.length;) {
    let opener = "", closer = "";
    if (text[i] === "$") {
      opener = closer = text[i + 1] === "$" ? "$$" : "$";
    } else if (text[i] === "\\") {
      if (text[i + 1] === "(" || text[i + 1] === "[") {
        opener = text.slice(i, i + 2);
        closer = opener === "\\(" ? "\\)" : "\\]";
      } else {
        const environment = /^\\begin\{([A-Za-z*]{1,16})\}/.exec(text.slice(i, i + 25));
        if (environment && MATH_ENVIRONMENTS.has(environment[1]!)) {
          opener = environment[0];
          closer = `\\end{${environment[1]!}}`;
        } else {
          i += 2; // An escaped dollar is prose, not a math opener.
          continue;
        }
      }
    }
    if (!opener) { i++; continue; }
    const end = missing.has(closer) ? -1 : closingOffset(text, closer, i + opener.length);
    if (end < 0) {
      missing.add(closer);
      i += opener.length;
      continue;
    }
    const after = end + closer.length;
    out += text.slice(copied, i) + `\uE000${formulas.length}\uE001`;
    formulas.push(text.slice(i, after));
    copied = i = after;
  }
  return {
    text: out + text.slice(copied),
    restore: value => value.replace(/\uE000(\d+)\uE001/g, (_, index: string) => formulas[Number(index)] ?? ""),
  };
}

export function latexEnvironmentBody(text: string, name: string): string | null {
  const open = `\\begin{${name}}`, close = `\\end{${name}}`;
  const start = text.indexOf(open);
  if (start < 0) return null;
  const end = text.indexOf(close, start + open.length);
  return end < 0 ? null : text.slice(start + open.length, end);
}

export function removeLatexEnvironments(text: string, environments: readonly string[]): string {
  let out = text;
  for (const name of environments) {
    const folded = out.replace(/[A-Z]/g, c => c.toLowerCase());
    const open = `\\begin{${name}}`, close = `\\end{${name}}`;
    let copied = 0, result = "";
    for (let start = folded.indexOf(open); start >= 0; start = folded.indexOf(open, copied)) {
      const end = folded.indexOf(close, start + open.length);
      if (end < 0) break;
      result += out.slice(copied, start) + "\n";
      copied = end + close.length;
    }
    out = result + out.slice(copied);
  }
  return out;
}

/** Pre-index paired arguments once; malformed optional arguments never rescan the tail. */
function argumentEnds(text: string): Map<number, number> {
  const ends = new Map<number, number>();
  const braces: number[] = [], brackets: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === "\\") { i++; continue; }
    if (c === "{") braces.push(i);
    else if (c === "[") brackets.push(i);
    else if (c === "}" || c === "]") {
      const start = (c === "}" ? braces : brackets).pop();
      if (start !== undefined) ends.set(start, i + 1);
    }
  }
  return ends;
}

const OMIT_ARGUMENT = new Set(["cite", "citep", "citet", "citealp", "ref", "autoref", "eqref", "label", "pageref"]);
const OMIT_LINE = new Set(["newcommand", "renewcommand", "providecommand", "def"]);

export function plainLatex(input: string): string {
  const protectedMath = protectLatexMath(input);
  return protectedMath.restore(plainLatexProse(protectedMath.text));
}

/** Clean prose whose formulas have already been protected by the caller. */
export function plainLatexProse(text: string): string {
  const ends = argumentEnds(text);
  let out = "";
  for (let i = 0; i < text.length;) {
    const c = text[i]!;
    if (c !== "\\") {
      if (c !== "{" && c !== "}") out += c === "~" ? " " : c;
      i++;
      continue;
    }
    const start = ++i;
    while (i < text.length && /[A-Za-z]/.test(text[i]!)) i++;
    if (i === start) {
      const escapedChar = text[i++];
      if (escapedChar && "%$#_{}&".includes(escapedChar)) out += escapedChar;
      else if (escapedChar === "\\") out += " ";
      continue;
    }
    const command = text.slice(start, i);
    if (text[i] === "*") i++;
    if (OMIT_LINE.has(command)) {
      const newline = text.indexOf("\n", i);
      i = newline < 0 ? text.length : newline;
      continue;
    }
    while (i < text.length && /\s/.test(text[i]!)) i++;
    while (text[i] === "[" && ends.has(i)) {
      i = ends.get(i)!;
      while (i < text.length && /\s/.test(text[i]!)) i++;
    }
    if ((OMIT_ARGUMENT.has(command) || command === "href") && text[i] === "{" && ends.has(i)) {
      i = ends.get(i)!;
    }
    // Formatting commands lose their names; their braced prose is visited normally.
  }
  return out.replace(/[ \t]{2,}/g, " ");
}
