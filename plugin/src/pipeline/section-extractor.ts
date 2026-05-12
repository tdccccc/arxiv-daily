export interface AbstractConclusionOpts {
  sectionCharLimit: number;
}

export interface ExtractSectionsOpts {
  sectionCharLimit: number;
  paperCharLimit: number;
  skipSections: string[];
  prioritySections: string[];
}

function parse(html: string): Document {
  return new DOMParser().parseFromString(html, "text/html");
}

function stripNoise(doc: Document) {
  for (const tag of ["script", "style", "nav", "footer", "figure", "table"]) {
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

  type S = { title: string; body: string; priority: boolean };
  const all: S[] = [];
  for (const h of headers) {
    const title = (h.textContent ?? "").trim();
    const lower = title.toLowerCase();
    if (opts.skipSections.some((s) => lower.includes(s.toLowerCase()))) continue;
    const body = textBetween(h).slice(0, opts.sectionCharLimit);
    if (!body) continue;
    const priority = opts.prioritySections.some((s) => lower.includes(s.toLowerCase()));
    all.push({ title, body, priority });
  }
  if (all.length === 0) return null;

  const reserved = all.filter((s) => s.priority).reduce((sum, s) => sum + s.body.length, 0);
  const budget = opts.paperCharLimit - reserved;
  const order = new Map(all.map((s, i) => [s.title, i] as const));
  const selected: S[] = [];
  let used = 0;
  for (const s of all) {
    if (s.priority) continue;
    if (used + s.body.length > budget) {
      const remaining = budget - used;
      if (remaining > 500) selected.push({ ...s, body: s.body.slice(0, remaining) });
      break;
    }
    selected.push(s);
    used += s.body.length;
  }
  const merged = [...selected, ...all.filter((s) => s.priority)];
  merged.sort((a, b) => (order.get(a.title) ?? 999) - (order.get(b.title) ?? 999));
  return merged.length
    ? merged.map((s) => `## ${s.title}\n${s.body}`).join("\n\n")
    : null;
}
