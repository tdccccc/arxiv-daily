import { gunzipSync } from "node:zlib";
import { classifySection } from "./section-extractor";

export interface SourceExtractOpts {
  sectionCharLimit: number;
  paperCharLimit: number;
  skipSections: string[];
  prioritySections: string[];
}

export interface SourceExtractResult {
  abstractConclusion: string | null;
  fullSections: string | null;
  mainFile: string | null;
}

interface SourceFile {
  path: string;
  text: string;
}

interface Section {
  title: string;
  body: string;
  index: number;
}

const TEXT_DECODER = new TextDecoder("utf-8");

export function extractLatexSource(
  input: ArrayBuffer,
  opts: SourceExtractOpts,
): SourceExtractResult {
  const files = unpackSourceFiles(input);
  const texFiles = files.filter((file) => isTexPath(file.path));
  if (texFiles.length === 0) {
    return { abstractConclusion: null, fullSections: null, mainFile: null };
  }

  const fileMap = new Map(
    texFiles.map((file) => [normalizeSourcePath(file.path), file.text]),
  );
  const mainFile = chooseMainTex(texFiles);
  if (!mainFile) {
    return { abstractConclusion: null, fullSections: null, mainFile: null };
  }

  const expanded = expandLatexInputs(
    normalizeSourcePath(mainFile.path),
    mainFile.text,
    fileMap,
  );
  const markdown = latexToMarkdown(expanded);
  const sections = parseMarkdownSections(markdown);
  const abstract = extractAbstract(expanded);
  const conclusion = firstSectionMatching(sections, (section) =>
    classifySection(section.title, section.body).includes("conclusion"),
  );

  const abstractConclusion = buildAbstractConclusion(abstract, conclusion, opts);
  const fullSections = buildFullSections(sections, opts);
  return {
    abstractConclusion,
    fullSections,
    mainFile: normalizeSourcePath(mainFile.path),
  };
}

function unpackSourceFiles(input: ArrayBuffer): SourceFile[] {
  const raw = Buffer.from(input);
  const candidates = [raw, maybeGunzip(raw)].filter(
    (value): value is Buffer => Boolean(value),
  );

  for (const candidate of candidates) {
    const tarFiles = parseTar(candidate);
    if (tarFiles.some((file) => isTexPath(file.path))) return tarFiles;
  }

  for (const candidate of candidates) {
    const text = decodeText(candidate);
    if (looksLikeLatex(text)) {
      return [{ path: "source.tex", text }];
    }
  }

  return [];
}

function maybeGunzip(buffer: Buffer): Buffer | null {
  if (buffer.length < 2 || buffer[0] !== 0x1f || buffer[1] !== 0x8b) {
    return null;
  }
  try {
    return gunzipSync(buffer);
  } catch {
    return null;
  }
}

function parseTar(buffer: Buffer): SourceFile[] {
  const out: SourceFile[] = [];
  let offset = 0;
  while (offset + 512 <= buffer.length) {
    const header = buffer.subarray(offset, offset + 512);
    if (isZeroBlock(header)) break;

    const name = readTarString(header, 0, 100);
    const prefix = readTarString(header, 345, 155);
    const sizeRaw = readTarString(header, 124, 12).trim();
    const size = Number.parseInt(sizeRaw || "0", 8);
    if (!name || !Number.isFinite(size) || size < 0) return [];

    const type = String.fromCharCode(header[156] || 0);
    const path = normalizeSourcePath(prefix ? `${prefix}/${name}` : name);
    const dataStart = offset + 512;
    const dataEnd = dataStart + size;
    if (dataEnd > buffer.length) return [];

    if ((type === "\0" || type === "0" || type === "") && isTextPath(path)) {
      out.push({
        path,
        text: decodeText(buffer.subarray(dataStart, dataEnd)),
      });
    }

    offset = dataStart + Math.ceil(size / 512) * 512;
  }
  return out;
}

function isZeroBlock(block: Buffer): boolean {
  for (const byte of block) {
    if (byte !== 0) return false;
  }
  return true;
}

function readTarString(buffer: Buffer, start: number, length: number): string {
  const slice = buffer.subarray(start, start + length);
  const zero = slice.indexOf(0);
  const end = zero >= 0 ? zero : slice.length;
  return decodeText(slice.subarray(0, end)).trim();
}

function decodeText(data: Uint8Array): string {
  return TEXT_DECODER.decode(data).replace(/\r\n?/g, "\n");
}

function isTextPath(path: string): boolean {
  return /\.(tex|bbl|sty|cls|txt)$/i.test(path);
}

function isTexPath(path: string): boolean {
  return /\.tex$/i.test(path) || path === "source.tex";
}

function looksLikeLatex(text: string): boolean {
  return /\\(?:documentclass|begin\{document\}|section|input|include)\b/.test(
    text,
  );
}

function chooseMainTex(files: SourceFile[]): SourceFile | null {
  let best: { file: SourceFile; score: number } | null = null;
  for (const file of files) {
    const basename = file.path.split("/").pop()?.toLowerCase() ?? "";
    let score = 0;
    if (/\\documentclass\b/.test(file.text)) score += 1000;
    if (/\\begin\{document\}/.test(file.text)) score += 1000;
    if (/\\section\b/.test(file.text)) score += 150;
    if (/^(main|paper|ms|manuscript|article)\.tex$/i.test(basename)) {
      score += 300;
    }
    score += Math.min(file.text.length / 1000, 200);
    if (!best || score > best.score) best = { file, score };
  }
  return best?.file ?? null;
}

function expandLatexInputs(
  currentPath: string,
  text: string,
  files: Map<string, string>,
  seen = new Set<string>(),
  depth = 0,
): string {
  if (depth > 12 || seen.has(currentPath)) return text;
  seen.add(currentPath);
  const dir = parentDir(currentPath);
  return text.replace(/\\(?:input|include)\s*\{([^}]+)\}/g, (_match, raw) => {
    const target = resolveInputPath(dir, String(raw), files);
    if (!target) return "";
    return expandLatexInputs(
      target,
      files.get(target) ?? "",
      files,
      new Set(seen),
      depth + 1,
    );
  });
}

function resolveInputPath(
  dir: string,
  raw: string,
  files: Map<string, string>,
): string | null {
  const cleaned = raw.trim().replace(/^["']|["']$/g, "");
  const base = normalizeSourcePath(dir ? `${dir}/${cleaned}` : cleaned);
  const candidates = [
    base,
    /\.tex$/i.test(base) ? base : `${base}.tex`,
    normalizeSourcePath(cleaned),
    /\.tex$/i.test(cleaned) ? normalizeSourcePath(cleaned) : `${normalizeSourcePath(cleaned)}.tex`,
  ];
  for (const candidate of candidates) {
    if (files.has(candidate)) return candidate;
  }
  const lower = new Map([...files.keys()].map((key) => [key.toLowerCase(), key]));
  for (const candidate of candidates) {
    const found = lower.get(candidate.toLowerCase());
    if (found) return found;
  }
  return null;
}

function latexToMarkdown(input: string): string {
  let text = stripLatexComments(input);
  const body = /\\begin\{document\}([\s\S]*?)\\end\{document\}/.exec(text);
  if (body) text = body[1];

  text = text.replace(/\\begin\{abstract\}/g, "\n## Abstract\n");
  text = text.replace(/\\end\{abstract\}/g, "\n");
  text = removeLatexEnvironments(text, [
    "figure",
    "figure*",
    "table",
    "table*",
    "equation",
    "equation*",
    "align",
    "align*",
    "eqnarray",
    "eqnarray*",
    "multline",
    "multline*",
    "gather",
    "gather*",
    "split",
    "tikzpicture",
    "picture",
    "verbatim",
    "lstlisting",
  ]);
  text = text.replace(/\\appendix\b[\s\S]*$/i, "");
  text = text.replace(/\\bibliography\s*\{[^}]*\}[\s\S]*$/i, "");
  text = text.replace(/\\begin\{thebibliography\}[\s\S]*$/i, "");
  text = text.replace(
    /\\(?:section|subsection|subsubsection)\*?\s*\{([^{}]+)\}/g,
    (_match, title) => `\n## ${plainLatex(String(title))}\n`,
  );
  text = text.replace(/\\paragraph\*?\s*\{([^{}]+)\}/g, (_match, title) => {
    return `\n${plainLatex(String(title))}. `;
  });
  text = plainLatex(text);
  text = text
    .split("\n")
    .map((line) => line.replace(/[ \t]+/g, " ").trim())
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return text;
}

function stripLatexComments(input: string): string {
  const lines = input.split("\n");
  return lines
    .map((line) => {
      let escaped = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === "\\" && !escaped) {
          escaped = true;
          continue;
        }
        if (ch === "%" && !escaped) return line.slice(0, i);
        escaped = false;
      }
      return line;
    })
    .join("\n");
}

function removeLatexEnvironments(text: string, envs: string[]): string {
  let out = text;
  for (const env of envs) {
    const escaped = env.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    out = out.replace(
      new RegExp(`\\\\begin\\{${escaped}\\}[\\s\\S]*?\\\\end\\{${escaped}\\}`, "gi"),
      "\n",
    );
  }
  return out;
}

function plainLatex(input: string): string {
  let text = input;
  text = text.replace(/\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}/g, "$1");
  text = text.replace(/\\url\s*\{([^{}]*)\}/g, "$1");
  text = text.replace(/\\(?:cite|citep|citet|citealp|ref|autoref|eqref)\*?(?:\[[^\]]*\])?\s*\{[^{}]*\}/g, "");
  text = text.replace(/\\(?:label|pageref)\s*\{[^{}]*\}/g, "");
  text = text.replace(/\\(?:newcommand|renewcommand|providecommand|def)\b[^\n]*/g, "");
  text = text.replace(/\\(?:maketitle|tableofcontents)\b/g, "");

  for (let i = 0; i < 6; i++) {
    const next = text.replace(
      /\\[A-Za-z]+\*?(?:\[[^\]]*\])?\s*\{([^{}]*)\}/g,
      "$1",
    );
    if (next === text) break;
    text = next;
  }

  text = text.replace(/\\[A-Za-z]+\*?(?:\[[^\]]*\])?/g, "");
  text = text.replace(/\\[`'"^~=.|&%$#_{}]/g, "");
  text = text.replace(/[{}]/g, "");
  text = text.replace(/~+/g, " ");
  text = text.replace(/[ \t]{2,}/g, " ");
  return text;
}

function extractAbstract(input: string): string | null {
  const match = /\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}/i.exec(
    stripLatexComments(input),
  );
  const text = match ? plainLatex(match[1]).replace(/\s+/g, " ").trim() : "";
  return text.length > 40 ? text : null;
}

function parseMarkdownSections(markdown: string): Section[] {
  const sections: Section[] = [];
  let currentTitle = "";
  let currentBody: string[] = [];
  const flush = () => {
    const body = currentBody.join("\n").trim();
    if (currentTitle && body.length > 80) {
      sections.push({ title: currentTitle, body, index: sections.length });
    }
  };

  for (const line of markdown.split("\n")) {
    const heading = /^##\s+(.+?)\s*$/.exec(line);
    if (heading) {
      flush();
      currentTitle = heading[1].trim();
      currentBody = [];
      continue;
    }
    if (currentTitle) currentBody.push(line);
  }
  flush();
  return sections;
}

function buildAbstractConclusion(
  abstract: string | null,
  conclusion: Section | null,
  opts: SourceExtractOpts,
): string | null {
  const parts: string[] = [];
  if (abstract) {
    parts.push(`## Abstract\n${truncateText(abstract, opts.sectionCharLimit)}`);
  }
  if (conclusion) {
    parts.push(
      `## ${conclusion.title}\n${truncateText(conclusion.body, opts.sectionCharLimit)}`,
    );
  }
  return parts.length ? parts.join("\n\n") : null;
}

function buildFullSections(
  sections: Section[],
  opts: SourceExtractOpts,
): string | null {
  const useful = sections.filter((section) => !shouldSkipSection(section, opts));
  if (useful.length === 0) return null;

  const rendered = useful.map((section) => renderSection(section, opts));
  const total = rendered.reduce((sum, section) => sum + section.text.length, 0);
  if (total <= opts.paperCharLimit) {
    return rendered.map((section) => section.text).join("\n\n");
  }

  const selected: Array<{ index: number; text: string }> = [];
  let size = 0;
  for (const section of [...useful].sort(compareSectionPriority(opts))) {
    const text = renderSection(section, opts).text;
    if (selected.length > 0 && size + text.length > opts.paperCharLimit) {
      continue;
    }
    selected.push({ index: section.index, text });
    size += text.length;
    if (size >= opts.paperCharLimit) break;
  }

  if (selected.length === 0) return null;
  return selected
    .sort((a, b) => a.index - b.index)
    .map((section) => section.text)
    .join("\n\n");
}

function renderSection(
  section: Section,
  opts: SourceExtractOpts,
): { index: number; text: string } {
  return {
    index: section.index,
    text: `## ${section.title}\n${truncateText(section.body, opts.sectionCharLimit)}`,
  };
}

function shouldSkipSection(section: Section, opts: SourceExtractOpts): boolean {
  const title = section.title.toLowerCase();
  const classified = classifySection(section.title, section.body);
  if (
    classified.includes("reference") ||
    classified.includes("appendix") ||
    classified.includes("acknowledgement")
  ) {
    return true;
  }
  const skipTerms = [...opts.skipSections, "references", "bibliography"];
  return skipTerms.some((term) => {
    const lower = term.trim().toLowerCase();
    return lower && title.includes(lower);
  });
}

function compareSectionPriority(opts: SourceExtractOpts) {
  return (a: Section, b: Section): number => {
    const diff = sectionRank(a, opts) - sectionRank(b, opts);
    return diff !== 0 ? diff : a.index - b.index;
  };
}

function sectionRank(section: Section, opts: SourceExtractOpts): number {
  const title = section.title.toLowerCase();
  if (
    opts.prioritySections.some((term) => {
      const lower = term.trim().toLowerCase();
      return lower && title.includes(lower);
    })
  ) {
    return 0;
  }
  const classified = classifySection(section.title, section.body);
  if (classified.includes("abstract")) return 0;
  if (classified.includes("conclusion")) return 1;
  if (classified.includes("result")) return 2;
  if (classified.includes("method")) return 3;
  if (classified.includes("data")) return 4;
  if (/intro|background/i.test(section.title)) return 5;
  return 10;
}

function firstSectionMatching(
  sections: Section[],
  predicate: (section: Section) => boolean,
): Section | null {
  return sections.find(predicate) ?? null;
}

function truncateText(text: string, limit: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= limit) return clean;
  return `${clean.slice(0, Math.max(0, limit - 20)).trim()}...`;
}

function parentDir(path: string): string {
  const parts = normalizeSourcePath(path).split("/").filter(Boolean);
  return parts.length <= 1 ? "" : parts.slice(0, -1).join("/");
}

function normalizeSourcePath(path: string): string {
  return path
    .replace(/\\/g, "/")
    .split("/")
    .filter((part) => part && part !== ".")
    .join("/");
}
