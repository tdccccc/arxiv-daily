import { DAILY_SUMMARY_FIELD_LABELS } from "../pipeline/daily-summary-rendering";
import { noCategoryPapersText } from "../settings/summary-language";
import type { SummaryLanguage } from "../settings/types";
import type { DailyDigest, DigestPaper } from "./types";
import { digestLanguage } from "./digest";

export function renderEmailSubject(digest: DailyDigest): string {
  const language = digestLanguage(digest);
  return `arXiv Daily ${digest.date} · ${paperCountLabel(digest.paperCount, language)}`;
}

export function renderEmailHtml(digest: DailyDigest): string {
  const language = digestLanguage(digest);
  const parts: string[] = [];
  parts.push(`<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>`);
  parts.push(
    `<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;line-height:1.5;color:#111;max-width:40rem;">`,
  );
  parts.push(`<h1 style="font-size:1.25rem;margin:0 0 0.5rem;">${escapeHtml(renderEmailSubject(digest))}</h1>`);
  parts.push(
    `<p style="margin:0 0 1rem;color:#444;">${escapeHtml(headerMetaLine(digest, language))}</p>`,
  );

  if (digest.paperCount === 0) {
    parts.push(`<p style="margin:0 0 1rem;"><strong>${escapeHtml(zeroDayLead(language))}</strong></p>`);
  }

  for (const topic of digest.topics) {
    parts.push(`<h2 style="font-size:1.1rem;margin:1.25rem 0 0.5rem;">${escapeHtml(topic.name)}</h2>`);
    if (topic.papers.length === 0) {
      parts.push(`<p style="margin:0 0 0.75rem;color:#555;">${escapeHtml(noCategoryPapersText(language))}</p>`);
      continue;
    }
    for (const paper of topic.papers) {
      parts.push(renderPaperHtml(paper, language));
    }
  }

  parts.push(`<hr style="border:none;border-top:1px solid #ddd;margin:1.5rem 0 0.75rem;" />`);
  const footer = footerLine(digest, language);
  if (footer) {
    parts.push(`<p style="margin:0;font-size:0.9rem;color:#666;">${escapeHtml(footer)}</p>`);
  }
  parts.push(
    `<p style="margin:0.25rem 0 0;font-size:0.85rem;color:#888;">arXiv Daily</p>`,
    `</div></body></html>`,
  );
  return parts.join("");
}

export function renderEmailText(digest: DailyDigest): string {
  const language = digestLanguage(digest);
  const lines: string[] = [];
  lines.push(renderEmailSubject(digest));
  lines.push(headerMetaLine(digest, language));
  lines.push("");

  if (digest.paperCount === 0) {
    lines.push(zeroDayLead(language));
    lines.push("");
  }

  for (const topic of digest.topics) {
    lines.push(`## ${topic.name}`);
    if (topic.papers.length === 0) {
      lines.push(noCategoryPapersText(language));
      lines.push("");
      continue;
    }
    for (const paper of topic.papers) {
      lines.push(...renderPaperText(paper, language));
      lines.push("");
    }
  }

  lines.push("---");
  const footer = footerLine(digest, language);
  if (footer) lines.push(footer);
  lines.push("arXiv Daily");
  return lines.join("\n").trimEnd() + "\n";
}

/** "1 paper" / "2 papers" / "2 篇"; the subject and the header must agree. */
function paperCountLabel(count: number, language: SummaryLanguage): string {
  if (language === "en") return `${count} ${count === 1 ? "paper" : "papers"}`;
  return `${count} 篇`;
}

function headerMetaLine(digest: DailyDigest, language: SummaryLanguage): string {
  const count = paperCountLabel(digest.paperCount, language);
  if (language === "en") {
    return `Date: ${digest.date} · ${count} · Categories: ${digest.categories}`;
  }
  return `日期：${digest.date} · ${count} · 分类：${digest.categories}`;
}

function zeroDayLead(language: SummaryLanguage): string {
  return language === "en" ? "No relevant papers today." : "今日无相关论文";
}

/** A day without relevant papers writes no report file, so there is no path to give. */
function footerLine(
  digest: DailyDigest,
  language: SummaryLanguage,
): string | undefined {
  if (digest.paperCount === 0) return undefined;
  return language === "en"
    ? `Full Markdown daily report: ${digest.dailyPath}`
    : `完整 Markdown 日报路径：${digest.dailyPath}`;
}

function renderPaperHtml(paper: DigestPaper, language: SummaryLanguage): string {
  const chunks: string[] = [];
  chunks.push(`<section style="margin:0 0 1.25rem;">`);
  // Omit source-sections noise (section lists are long and vault-oriented).
  chunks.push(
    `<h3 style="font-size:1rem;margin:0 0 0.35rem;">${escapeHtml(emailProse(paper.title))}</h3>`,
  );
  const authorLabel = language === "en" ? "Authors" : "作者";
  chunks.push(
    `<p style="margin:0 0 0.35rem;">${escapeHtml(authorLabel)}: ${escapeHtml(emailProse(paper.authors))}</p>`,
  );
  chunks.push(
    `<p style="margin:0 0 0.5rem;">` +
      `<a href="${escapeHtml(paper.absUrl)}">arXiv</a>` +
      ` · ` +
      `<a href="${escapeHtml(paper.pdfUrl)}">PDF</a>` +
      `</p>`,
  );

  if (paper.kind === "structured" && paper.fields) {
    chunks.push(`<ul style="margin:0;padding-left:1.2rem;">`);
    for (const [key, label] of DAILY_SUMMARY_FIELD_LABELS[language]) {
      const value = emailProse(paper.fields[key]);
      chunks.push(
        `<li style="margin:0 0 0.25rem;"><strong>${escapeHtml(label)}</strong>: ${escapeHtml(value)}</li>`,
      );
    }
    chunks.push(`</ul>`);
  } else {
    const warning =
      language === "en"
        ? "Summary unavailable. Read the original paper on arXiv."
        : "自动摘要不可用。请直接阅读 arXiv 原文。";
    chunks.push(`<p style="margin:0 0 0.35rem;"><strong>${escapeHtml(warning)}</strong></p>`);
    const abstractLabel = language === "en" ? "Original abstract" : "原始摘要";
    const abstract =
      emailProse(paper.abstract ?? "") ||
      (language === "en" ? "Unavailable." : "不可用。");
    chunks.push(
      `<p style="margin:0;"><strong>${escapeHtml(abstractLabel)}</strong>: ${escapeHtml(abstract)}</p>`,
    );
  }

  chunks.push(`</section>`);
  return chunks.join("");
}

function renderPaperText(paper: DigestPaper, language: SummaryLanguage): string[] {
  const lines: string[] = [];
  lines.push(`### ${emailProse(paper.title)}`);
  const authorLabel = language === "en" ? "Authors" : "作者";
  lines.push(`${authorLabel}: ${emailProse(paper.authors)}`);
  lines.push(`arXiv: ${paper.absUrl}`);
  lines.push(`PDF: ${paper.pdfUrl}`);
  if (paper.kind === "structured" && paper.fields) {
    for (const [key, label] of DAILY_SUMMARY_FIELD_LABELS[language]) {
      lines.push(`- ${label}: ${emailProse(paper.fields[key])}`);
    }
  } else {
    const warning =
      language === "en"
        ? "Summary unavailable. Read the original paper on arXiv."
        : "自动摘要不可用。请直接阅读 arXiv 原文。";
    lines.push(warning);
    const abstractLabel = language === "en" ? "Original abstract" : "原始摘要";
    const abstract =
      emailProse(paper.abstract ?? "") ||
      (language === "en" ? "Unavailable." : "不可用。");
    lines.push(`- ${abstractLabel}: ${abstract}`);
  }
  return lines;
}

/**
 * Make summary prose more readable in plain email clients:
 * strip TeX delimiters and simplify common LaTeX (no MathJax in mail).
 *
 * Prefer this deterministic path over asking the LLM for a second "email math"
 * summary (extra cost, drift from vault, harder to validate).
 */
export function emailProse(value: string): string {
  if (!value) return "";
  let s = value;

  // Display / block math delimiters → simplified body. A lazy [\s\S]*?
  // between two multi-character delimiters can backtrack quadratically when
  // the opening delimiter repeats without ever closing (CodeQL
  // js/polynomial-redos), so scan delimiter positions directly instead.
  s = replaceDelimited(s, "\\[", "\\]", simplifyLatex);
  s = replaceDelimited(s, "$$", "$$", simplifyLatex);
  s = replaceDelimited(s, "\\(", "\\)", simplifyLatex);
  // Inline $...$ (single line). [^$\n]+? cannot overlap with the single-char
  // $ delimiter, so this lazy match is already linear and left as a regex.
  s = s.replace(/\$([^$\n]+?)\$/g, (_, body: string) => simplifyLatex(body));

  // Any remaining TeX-ish fragments outside delimiters
  s = simplifyLatex(s);

  return s
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/([([⟨])\s+/g, "$1")
    .replace(/\s+([)\]⟩])/g, "$1")
    .trim();
}

/**
 * Replace every `open ... close` delimited span with `transform(body)`,
 * scanning literal delimiter positions with indexOf instead of a lazy regex.
 * Matches the lazy-lookahead-regex semantics (leftmost open, nearest
 * following close, non-overlapping, unmatched tails left untouched) in
 * linear time regardless of how often an unterminated `open` repeats.
 */
function replaceDelimited(
  value: string,
  open: string,
  close: string,
  transform: (body: string) => string,
): string {
  let result = "";
  let cursor = 0;
  for (;;) {
    const start = value.indexOf(open, cursor);
    if (start < 0) {
      result += value.slice(cursor);
      return result;
    }
    const end = value.indexOf(close, start + open.length);
    if (end < 0) {
      result += value.slice(cursor);
      return result;
    }
    result += value.slice(cursor, start);
    result += transform(value.slice(start + open.length, end));
    cursor = end + close.length;
  }
}

/** End of a TeX control word: not another letter (underscore may follow: \Omega_m). */
const TEX_END = "(?![A-Za-z])";

function texCmd(name: string): RegExp {
  return new RegExp(`\\\\${name}${TEX_END}`, "g");
}

// Private-use marks around the index of a piece of text that simplifyLatex sets
// aside and puts back unchanged.
const KEEP_OPEN = "";
const KEEP_CLOSE = "";

/** Environments that stay as TeX: flattening a matrix or aligned equation reads worse than the source. */
const TEX_ENVIRONMENTS = new Set([
  "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix",
  "cases", "array", "align", "align*", "aligned", "alignedat", "gather", "gather*",
  "gathered", "split", "equation", "equation*", "eqnarray", "eqnarray*", "multline",
  "multline*",
]);

const BLACKBOARD_LETTERS: Record<string, string> = {
  R: "ℝ",
  N: "ℕ",
  Z: "ℤ",
  Q: "ℚ",
  C: "ℂ",
};

/** Commands that are words (operators), so dropping the backslash reads fine. */
const TEX_FUNCTION_NAMES =
  "sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|coth|log|ln|lg|exp|lim|liminf|limsup|max|min|sup|inf|arg|det|dim|ker|gcd|deg|Pr|hom";

/**
 * Park each known environment (\begin{pmatrix} ... \end{pmatrix}) behind a
 * mark. Scans with indexOf rather than a lazy regex, and gives up on a name
 * once its \end is missing, so repeated unterminated openers stay linear.
 */
function keepEnvironments(input: string, keep: (text: string) => string): string {
  const begin = /\\begin\{([A-Za-z]{1,12}\*?)\}/g;
  const unterminated = new Set<string>();
  let out = "";
  let cursor = 0;
  for (let match = begin.exec(input); match; match = begin.exec(input)) {
    const name = match[1]!;
    if (!TEX_ENVIRONMENTS.has(name) || unterminated.has(name)) continue;
    const closer = `\\end{${name}}`;
    const close = input.indexOf(closer, match.index + match[0].length);
    if (close < 0) {
      unterminated.add(name);
      continue;
    }
    const end = close + closer.length;
    out += input.slice(cursor, match.index) + keep(input.slice(match.index, end));
    cursor = end;
    begin.lastIndex = end;
  }
  return out + input.slice(cursor);
}

function simplifyLatex(raw: string): string {
  // Text that must come through unchanged (environments, commands this cannot
  // render) is set aside behind marks while everything else is rewritten.
  const kept: string[] = [];
  const keep = (text: string): string =>
    `${KEEP_OPEN}${kept.push(text) - 1}${KEEP_CLOSE}`;
  let s = keepEnvironments(raw.replace(/[]/g, ""), keep);

  // Spaces first so content inside \mathrm{km\,s} becomes "km s" before unwrap.
  s = s.replace(/\\[,;]/g, " ");
  s = s.replace(/\\!/g, "");
  s = s.replace(/\\(?:quad|qquad)(?![A-Za-z])/g, " ");

  s = s.replace(
    /\\mathbb\s*\{\s*([RNZQC])\s*\}/g,
    (_, letter: string) => BLACKBOARD_LETTERS[letter]!,
  );

  // Unwrap \mathrm{...} / \text{...} even when the body still has ^ / _ / spaces.
  // (Simple [^{}]* fails on nested braces; balanced scan handles one level deep.)
  s = unwrapLatexGroups(s);

  for (let i = 0; i < 3; i += 1) {
    s = s.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)");
  }
  // \sqrt[n]{x} and \sqrt{x}; innermost first, so nested roots take a few passes.
  for (let i = 0; i < 3; i += 1) {
    s = s.replace(/\\sqrt\s*\[([A-Za-z0-9]{1,8})\]\s*\{([^{}]*)\}/g, "$1√($2)");
    s = s.replace(/\\sqrt\s*\{([^{}]*)\}/g, "√($1)");
  }
  s = s.replace(/\\sqrt(?![A-Za-z])\s*/g, "√");

  // \rm / \it / \bf as font switches (common in N_{\rm side})
  s = s.replace(/\\(?:rm|it|bf|cal|tt)\s+/g, "");
  s = s.replace(/\\(?:rm|it|bf|cal|tt)(?=[A-Za-z])/g, "");

  // Sizing and style commands only affect the layout the reader cannot see here.
  s = s.replace(/\\(?:left|right)\s*\./g, "");
  s = s.replace(
    /\\(?:left|right|bigg?[lr]?|Bigg?[lr]?|displaystyle|textstyle|scriptstyle|scriptscriptstyle|limits|nolimits)(?![A-Za-z])\s*/g,
    "",
  );
  // Accents the braced form of unwrapLatexGroups did not reach (\vec v, \hat x).
  s = s.replace(
    /\\(?:hat|widehat|bar|vec|tilde|widetilde|dot|ddot|overline|underline)(?![A-Za-z])\s*/g,
    "",
  );

  // A degree sign, not a composition circle.
  s = s.replace(/\^\s*(?:\\circ(?![A-Za-z])|\{\s*\\circ\s*\})/g, "°");

  // Common symbols — use (?![A-Za-z]) so \Omega_m / \sum_i match
  const symbols: Array<[RegExp, string]> = [
    [texCmd("leqslant"), "≤"],
    [texCmd("geqq"), "≥"],
    [texCmd("leq"), "≤"],
    [texCmd("geq"), "≥"],
    [texCmd("neq"), "≠"],
    [texCmd("le"), "≤"],
    [texCmd("ge"), "≥"],
    [texCmd("ne"), "≠"],
    [texCmd("ll"), "≪"],
    [texCmd("gg"), "≫"],
    [texCmd("equiv"), "≡"],
    [texCmd("cong"), "≅"],
    [texCmd("approx"), "≈"],
    [texCmd("simeq"), "≃"],
    [texCmd("sim"), "~"],
    [texCmd("lesssim"), "≲"],
    [texCmd("gtrsim"), "≳"],
    [texCmd("times"), "×"],
    [texCmd("otimes"), "⊗"],
    [texCmd("oplus"), "⊕"],
    [texCmd("odot"), "⊙"],
    [texCmd("star"), "⋆"],
    [texCmd("ast"), "∗"],
    [texCmd("dagger"), "†"],
    [texCmd("circ"), "∘"],
    [texCmd("degree"), "°"],
    [texCmd("AA"), "Å"],
    [texCmd("cdot"), "·"],
    [texCmd("pm"), "±"],
    [texCmd("mp"), "∓"],
    [texCmd("infty"), "∞"],
    [texCmd("propto"), "∝"],
    [texCmd("perp"), "⊥"],
    [texCmd("parallel"), "∥"],
    [texCmd("prime"), "′"],
    [texCmd("rightarrow"), "→"],
    [texCmd("leftarrow"), "←"],
    [texCmd("leftrightarrow"), "↔"],
    [texCmd("longrightarrow"), "→"],
    [texCmd("Rightarrow"), "⇒"],
    [texCmd("Leftarrow"), "⇐"],
    [texCmd("Leftrightarrow"), "⇔"],
    [texCmd("iff"), "⇔"],
    [texCmd("mapsto"), "↦"],
    [texCmd("to"), "→"],
    [texCmd("subseteq"), "⊆"],
    [texCmd("subset"), "⊂"],
    [texCmd("supseteq"), "⊇"],
    [texCmd("supset"), "⊃"],
    [texCmd("notin"), "∉"],
    [texCmd("in"), "∈"],
    [texCmd("cup"), "∪"],
    [texCmd("cap"), "∩"],
    [texCmd("emptyset"), "∅"],
    [texCmd("varnothing"), "∅"],
    [texCmd("forall"), "∀"],
    [texCmd("exists"), "∃"],
    [texCmd("neg"), "¬"],
    [texCmd("wedge"), "∧"],
    [texCmd("vee"), "∨"],
    [texCmd("setminus"), "∖"],
    [texCmd("colon"), ":"],
    [texCmd("bullet"), "•"],
    [texCmd("langle"), "⟨"],
    [texCmd("rangle"), "⟩"],
    [texCmd("lfloor"), "⌊"],
    [texCmd("rfloor"), "⌋"],
    [texCmd("lceil"), "⌈"],
    [texCmd("rceil"), "⌉"],
    [texCmd("Vert"), "‖"],
    [texCmd("vert"), "|"],
    [texCmd("mid"), "|"],
    [texCmd("sum"), "∑"],
    [texCmd("prod"), "∏"],
    [texCmd("int"), "∫"],
    [texCmd("alpha"), "α"],
    [texCmd("beta"), "β"],
    [texCmd("gamma"), "γ"],
    [texCmd("delta"), "δ"],
    [texCmd("epsilon"), "ε"],
    [texCmd("varepsilon"), "ε"],
    [texCmd("zeta"), "ζ"],
    [texCmd("eta"), "η"],
    [texCmd("theta"), "θ"],
    [texCmd("vartheta"), "ϑ"],
    [texCmd("iota"), "ι"],
    [texCmd("kappa"), "κ"],
    [texCmd("lambda"), "λ"],
    [texCmd("mu"), "μ"],
    [texCmd("nu"), "ν"],
    [texCmd("xi"), "ξ"],
    [texCmd("pi"), "π"],
    [texCmd("varpi"), "ϖ"],
    [texCmd("rho"), "ρ"],
    [texCmd("varrho"), "ϱ"],
    [texCmd("sigma"), "σ"],
    [texCmd("varsigma"), "ς"],
    [texCmd("tau"), "τ"],
    [texCmd("upsilon"), "υ"],
    [texCmd("phi"), "φ"],
    [texCmd("varphi"), "φ"],
    [texCmd("chi"), "χ"],
    [texCmd("psi"), "ψ"],
    [texCmd("omega"), "ω"],
    [texCmd("Gamma"), "Γ"],
    [texCmd("Delta"), "Δ"],
    [texCmd("Theta"), "Θ"],
    [texCmd("Lambda"), "Λ"],
    [texCmd("Xi"), "Ξ"],
    [texCmd("Pi"), "Π"],
    [texCmd("Sigma"), "Σ"],
    [texCmd("Upsilon"), "Υ"],
    [texCmd("Phi"), "Φ"],
    [texCmd("Psi"), "Ψ"],
    [texCmd("Omega"), "Ω"],
    [texCmd("ell"), "ℓ"],
    [texCmd("hbar"), "ℏ"],
    [texCmd("partial"), "∂"],
    [texCmd("nabla"), "∇"],
    [texCmd("ldots"), "…"],
    [texCmd("cdots"), "⋯"],
    [texCmd("dots"), "…"],
  ];
  for (const [re, rep] of symbols) s = s.replace(re, rep);

  // _{...} / ^{...}
  s = s.replace(/_\{([^{}]+)\}/g, "_$1");
  s = s.replace(/\^\{([^{}]+)\}/g, "^$1");
  // bare _i ^2 (single token)
  s = s.replace(/_([A-Za-z0-9+-]+)/g, "_$1");
  s = s.replace(/\^([A-Za-z0-9+-]+)/g, "^$1");

  // Operators such as \log and \max are words; any other command is kept as
  // written, with its arguments, so it reads as TeX instead of fusing its name
  // onto the argument (\foo{x} -> "foox").
  s = s.replace(new RegExp(`\\\\(${TEX_FUNCTION_NAMES})${TEX_END}`, "g"), "$1");
  s = s.replace(/\\[A-Za-z]+\*?(?:\s*\{(?:[^{}]|\{[^{}]*\})*\})*/g, keep);
  // Row breaks, norm bars, stray braces and escapes (\%, \&, \_).
  s = s.replace(/\\\\/g, " ");
  s = s.replace(/\\\|/g, "‖");
  s = s.replace(/[{}]/g, "");
  s = s.replace(/\\/g, "");

  // "Omega _m" / "sum _i" style gaps from older partial transforms
  s = s.replace(/\s+(_[A-Za-z0-9+-]+)/g, "$1");
  s = s.replace(/\s+(\^[A-Za-z0-9+-]+)/g, "$1");

  let out = s.replace(/\s+/g, " ").trim();
  // A kept command can hold a kept environment, so unpack a few levels.
  const marked = new RegExp(`${KEEP_OPEN}(\\d+)${KEEP_CLOSE}`, "g");
  for (let pass = 0; pass < 3 && out.includes(KEEP_OPEN); pass += 1) {
    out = out.replace(marked, (_, index: string) => kept[Number(index)] ?? "");
  }
  return out;
}

const GROUP_CMD_OPEN =
  /^\\(?:text|mathrm|mathbf|mathit|textrm|textit|textbf|operatorname|mathsf|mathtt|mathcal|mathscr|mathfrak|mathbb|boldsymbol|bm|emph|textsc|textsf|textup|mbox|hat|widehat|bar|vec|tilde|widetilde|dot|ddot|overline|underline)\s*\{/i;

/** Replace \mathrm{...} / \text{...} (and typesetting-only wrappers like \mathcal{...}) with inner text (balanced braces). */
function unwrapLatexGroups(input: string): string {
  let s = input;
  for (let pass = 0; pass < 8; pass += 1) {
    let out = "";
    let i = 0;
    let changed = false;
    while (i < s.length) {
      const rest = s.slice(i);
      const m = GROUP_CMD_OPEN.exec(rest);
      if (!m) {
        out += s[i];
        i += 1;
        continue;
      }
      const openEnd = i + m[0].length;
      let depth = 1;
      let j = openEnd;
      while (j < s.length && depth > 0) {
        if (s[j] === "{") depth += 1;
        else if (s[j] === "}") depth -= 1;
        j += 1;
      }
      if (depth !== 0) {
        out += s[i];
        i += 1;
        continue;
      }
      out += s.slice(openEnd, j - 1);
      i = j;
      changed = true;
    }
    s = out;
    if (!changed) break;
  }
  return s;
}

/** Escape prose for email HTML after emailProse normalization. */
export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
