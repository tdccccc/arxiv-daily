export interface AbstractConclusionOpts {
  sectionCharLimit: number;
}

export interface ExtractSectionsOpts {
  sectionCharLimit: number;
  paperCharLimit: number;
  skipSections: string[];
  prioritySections: string[];
}

export type SectionKind =
  | "abstract"
  | "introduction"
  | "related"
  | "data"
  | "method"
  | "experiment"
  | "result"
  | "discussion"
  | "limitation"
  | "conclusion"
  | "appendix"
  | "reference"
  | "acknowledgement"
  | "other";

function parse(html: string): Document {
  return new DOMParser().parseFromString(html, "text/html");
}

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/\\[a-z]+/g, " ")
    .replace(/^\s*(\d+(\.\d+)*|[ivxlcdm]+|[a-z])\s*[\).:-]?\s+/i, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function has(text: string, pattern: RegExp): boolean {
  pattern.lastIndex = 0;
  return pattern.test(text);
}

function addScore(
  scores: Map<SectionKind, number>,
  kind: SectionKind,
  amount: number,
) {
  scores.set(kind, (scores.get(kind) ?? 0) + amount);
}

function scorePatterns(
  scores: Map<SectionKind, number>,
  kind: SectionKind,
  text: string,
  amount: number,
  patterns: RegExp[],
) {
  for (const pattern of patterns) {
    if (has(text, pattern)) addScore(scores, kind, amount);
  }
}

export function classifySection(
  title: string,
  bodyPreview: string = "",
): SectionKind[] {
  const titleText = normalizeText(title);
  const bodyText = normalizeText(bodyPreview.slice(0, 1200));
  const scores = new Map<SectionKind, number>();

  scorePatterns(scores, "reference", titleText, 6, [
    /\b(references?|bibliography)\b/,
  ]);
  scorePatterns(scores, "appendix", titleText, 6, [/\bappendix\b/]);
  scorePatterns(scores, "acknowledgement", titleText, 6, [
    /\b(acknowledgements?|acknowledgments?|author contributions?|data availability|conflict of interest|orcid)\b/,
  ]);
  scorePatterns(scores, "conclusion", titleText, 5, [
    /\b(conclusions?|summary|concluding remarks?|final remarks?)\b/,
  ]);
  scorePatterns(scores, "abstract", titleText, 5, [/\babstract\b/]);
  scorePatterns(scores, "related", titleText, 4, [
    /\b(related work|previous work|literature review)\b/,
  ]);
  scorePatterns(scores, "introduction", titleText, 4, [
    /\b(introduction|background|motivation|overview)\b/,
  ]);
  scorePatterns(scores, "data", titleText, 4, [
    /\b(data|datasets?|samples?|observations?|survey|surveys|catalogs?|catalogues?|spectra|spectroscopic|photometry|photometric|imaging|images?|light curves?|data release)\b/,
    /\b(the\s+)?\w+\s+(catalog|catalogue|sample|survey)\b/,
  ]);
  scorePatterns(scores, "method", titleText, 4, [
    /\b(methods?|methodology|approach|models?|modelling|modeling|algorithm|framework|pipeline|inference|calibration|estimator|likelihood|selection function|selection effects|forward model|training|architecture|network|reconstruction|synthesis|fitting)\b/,
  ]);
  scorePatterns(scores, "experiment", titleText, 4, [
    /\b(experiments?|evaluation|benchmark|validation|tests?|setup|baselines?|ablation|comparison)\b/,
  ]);
  scorePatterns(scores, "result", titleText, 4, [
    /\b(results?|findings?|measurements?|constraints?|performance|detections?|properties|estimates?)\b/,
  ]);
  scorePatterns(scores, "discussion", titleText, 4, [
    /\b(discussion|implications?|interpretation|analysis)\b/,
  ]);
  scorePatterns(scores, "limitation", titleText, 4, [
    /\b(limitations?|caveats?|uncertaint(y|ies)|systematics?|future work|robustness|biases?)\b/,
  ]);

  scorePatterns(scores, "data", bodyText, 1, [
    /\b(we use|we used|our sample|observed with|observations were|data release|catalogue|catalog|survey|spectra|photometry)\b/,
  ]);
  scorePatterns(scores, "method", bodyText, 1, [
    /\b(we model|we estimate|we infer|we train|we calibrate|algorithm|likelihood|pipeline|selection function|forward model)\b/,
  ]);
  scorePatterns(scores, "experiment", bodyText, 1, [
    /\b(we evaluate|we validate|benchmark|baseline|test set|simulation setup|experimental setup)\b/,
  ]);
  scorePatterns(scores, "result", bodyText, 1, [
    /\b(we find|we found|we measure|we measured|we show|our results|we obtain|we detect|we constrain|we report|improves?|outperforms?)\b/,
  ]);
  scorePatterns(scores, "discussion", bodyText, 1, [
    /\b(we discuss|this suggests|this implies|interpretation|implication)\b/,
  ]);
  scorePatterns(scores, "limitation", bodyText, 1, [
    /\b(uncertainty|uncertainties|limitation|limitations|caveat|caveats|systematic|systematics|bias|future work)\b/,
  ]);

  const kinds = Array.from(scores.entries())
    .filter(([, score]) => score > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([kind]) => kind);
  return kinds.length ? kinds : ["other"];
}

function compactText(el: Element, maxLen: number): string {
  return (el.textContent ?? "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLen);
}

function preserveFigureAndTableText(doc: Document) {
  const nodes = Array.from(
    doc.querySelectorAll("figure, .ltx_figure, table, .ltx_table"),
  );
  for (const el of nodes) {
    if (!el.parentNode) continue;
    const tag = el.tagName.toLowerCase();
    const isTable = tag === "table" || el.classList.contains("ltx_table");
    const caption = el.querySelector("figcaption, caption, .ltx_caption");
    const captionText = caption ? compactText(caption, 1600) : "";
    const fullText = compactText(el, 1600);
    const text = captionText || fullText;
    if (!text) {
      el.parentNode.removeChild(el);
      continue;
    }
    const replacement = doc.createElement("p");
    replacement.textContent = isTable
      ? `Table text: ${text}`
      : `Figure caption: ${text}`;
    el.parentNode.replaceChild(replacement, el);
  }
}

function stripNoise(doc: Document) {
  preserveFigureAndTableText(doc);
  for (const tag of ["script", "style", "nav", "footer"]) {
    for (const el of Array.from(doc.querySelectorAll(tag))) {
      el.parentNode?.removeChild(el);
    }
  }
}

function textBetween(start: Element): string {
  const parts: string[] = [];
  let n: Element | null = start.nextElementSibling;
  while (n && !/^h[2-4]$/i.test(n.tagName)) {
    const t = (n.textContent ?? "").replace(/\s+/g, " ").trim();
    if (t) parts.push(t);
    n = n.nextElementSibling;
  }
  return parts.join("\n");
}

export function extractAbstractConclusion(
  html: string,
  opts: AbstractConclusionOpts,
): string | null {
  const doc = parse(html);
  stripNoise(doc);
  const sections: string[] = [];

  const abstractDiv = doc.querySelector("div.ltx_abstract");
  if (abstractDiv) {
    const txt = (abstractDiv.textContent ?? "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, opts.sectionCharLimit);
    if (txt) sections.push(`## Abstract\n${txt}`);
  }

  const headers = Array.from(doc.querySelectorAll("h2, h3, h4"));
  for (const h of headers) {
    const title = (h.textContent ?? "").trim();
    const lower = title.toLowerCase();
    if (!/conclusion|summary/.test(lower)) continue;
    const body = textBetween(h).slice(0, opts.sectionCharLimit);
    if (body) sections.push(`## ${title}\n${body}`);
  }

  return sections.length ? sections.join("\n\n") : null;
}

export function extractSections(html: string, opts: ExtractSectionsOpts): string | null {
  const doc = parse(html);
  stripNoise(doc);
  const headers = Array.from(doc.querySelectorAll("h2, h3, h4"));
  if (headers.length === 0) return null;

  type S = {
    title: string;
    body: string;
    kinds: SectionKind[];
    priority: boolean;
    rank: number;
    index: number;
  };
  const all: S[] = [];
  for (const [index, h] of headers.entries()) {
    const title = (h.textContent ?? "").trim();
    const lower = title.toLowerCase();
    if (opts.skipSections.some((s) => lower.includes(s.toLowerCase()))) continue;
    const body = textBetween(h).slice(0, opts.sectionCharLimit);
    if (!body) continue;
    const kinds = classifySection(title, body);
    if (kinds.some((k) => ["reference", "appendix", "acknowledgement"].includes(k))) {
      continue;
    }
    const priority =
      opts.prioritySections.some((s) => lower.includes(s.toLowerCase())) ||
      kinds.some((k) => k === "abstract" || k === "conclusion");
    const rank = sectionRank(kinds, priority);
    all.push({ title, body, kinds, priority, rank, index });
  }
  if (all.length === 0) return null;

  const reserved = all.filter((s) => s.priority).reduce((sum, s) => sum + s.body.length, 0);
  const budget = opts.paperCharLimit - reserved;
  const selected: S[] = [];
  let used = 0;
  const candidates = all
    .filter((s) => !s.priority)
    .sort((a, b) => a.rank - b.rank || a.index - b.index);
  for (const s of candidates) {
    if (s.priority) continue;
    if (used + s.body.length > budget) {
      const remaining = budget - used;
      if (remaining > 500) {
        selected.push({ ...s, body: s.body.slice(0, remaining) });
        used += remaining;
      }
      continue;
    }
    selected.push(s);
    used += s.body.length;
  }
  const merged = [...selected, ...all.filter((s) => s.priority)];
  merged.sort((a, b) => a.index - b.index);
  return merged.length
    ? merged.map((s) => `## ${s.title}\n${s.body}`).join("\n\n")
    : null;
}

function sectionRank(kinds: SectionKind[], configuredPriority: boolean): number {
  if (
    configuredPriority ||
    kinds.some((k) => k === "abstract" || k === "conclusion")
  ) {
    return 0;
  }
  if (
    kinds.some((k) =>
      ["result", "experiment", "method", "data", "limitation", "discussion"].includes(k),
    )
  ) {
    return 1;
  }
  if (kinds.includes("other")) return 2;
  if (kinds.some((k) => k === "introduction" || k === "related")) return 3;
  return 2;
}
