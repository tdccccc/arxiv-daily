import { splitGenerationMetrics, type GenerationMetrics } from "@arxiv-daily/core";
import MarkdownIt, { type MarkdownIt as MarkdownParser, type Token } from "markdown-it";
import katex from "katex";

export interface RenderMarkdownOptions {
  resolveLink?: (target: string, kind: "link" | "image") => string | null;
}

export interface RenderedMarkdown {
  html: string;
  generationMetrics: GenerationMetrics | null;
  title: string;
  headings: Array<{ id: string; title: string; level: number }>;
  metadata: Record<string, string>;
}

/** Listing projection: parse headings and metadata without invoking KaTeX or producing HTML. */
export function describeMarkdown(source: string): Pick<RenderedMarkdown, "title" | "metadata"> {
  const { body, metadata } = frontmatter(source);
  if (metadata.title) return { title: metadata.title, metadata };
  const tokens = readingParser().parse(body, {});
  const headings = tokens.flatMap((token, index) => token.type === "heading_open"
    ? [{ level: Number(token.tag.slice(1)), title: inlineText(tokens[index + 1]?.children ?? []) }]
    : []);
  return { title: headings.find(heading => heading.level === 1)?.title || headings[0]?.title || "", metadata };
}

/** Render a projection only: the source Markdown and its frontmatter remain unchanged. */
export function renderMarkdown(source: string, options: RenderMarkdownOptions = {}): RenderedMarkdown {
  const generation = splitGenerationMetrics(source);
  const { body, metadata } = frontmatter(generation.body);
  const markdown = readingParser();
  const tokens = markdown.parse(body, {});
  const headings: RenderedMarkdown["headings"] = [];
  const usedIds = new Set<string>();
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (token.type === "heading_open") {
      const title = inlineText(tokens[index + 1]?.children ?? []);
      if (["summary sources", "总结依据"].includes(title.trim().replace(/[:：]$/, "").toLowerCase())) token.meta = { readingAppendix: true };
      const stem = title.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}\p{M}_-]+/gu, "-").replace(/^-+|-+$/g, "") || "section";
      let id = stem;
      let suffix = 2;
      while (usedIds.has(id)) id = `${stem}-${suffix++}`;
      usedIds.add(id);
      token.attrSet("id", id);
      headings.push({ id, title, level: Number(token.tag.slice(1)) });
    }
    if (token.children) resolveDestinations(token.children, options);
  }
  return {
    html: markdown.renderer.render(tokens, markdown.options, {}),
    generationMetrics: generation.metrics,
    title: metadata.title || headings.find(heading => heading.level === 1)?.title || headings[0]?.title || "",
    headings,
    metadata,
  };
}

/** Safe inline projection for titles/previews; uses the same syntax and destination policy. */
export function renderInlineMarkdown(source: string, options: RenderMarkdownOptions = {}): string {
  const markdown = readingParser();
  const tokens = markdown.parseInline(source, {});
  for (const token of tokens) if (token.children) resolveDestinations(token.children, options);
  return markdown.renderer.render(tokens, markdown.options, {});
}

function readingParser(): MarkdownParser {
  const markdown = new MarkdownIt({ html: false, linkify: false, typographer: false });
  addReadingSyntax(markdown);
  markdown.renderer.rules.heading_open = (tokens, index) =>
    (tokens[index]!.meta?.readingAppendix ? '<hr class="reading-appendix-divider">\n' : '')
    + markdown.renderer.renderToken(tokens, index, markdown.options);
  return markdown;
}

function frontmatter(source: string): { body: string; metadata: Record<string, string> } {
  const normalized = source.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
  const match = /^---[ \t]*\n([\s\S]*?)\n---[ \t]*(?:\n|$)/.exec(normalized);
  const metadata: Record<string, string> = {};
  if (!match) return { body: normalized, metadata };
  for (const line of match[1]!.split("\n")) {
    const field = /^(title|authors|arxiv_id|date|published|primary_topic):[ \t]*(.*)$/.exec(line);
    if (!field) continue;
    const value = field[2]!.trim();
    if (value.startsWith('"')) {
      try {
        const parsed: unknown = JSON.parse(value);
        if (typeof parsed === "string") metadata[field[1]!] = parsed;
      } catch { /* An invalid scalar is omitted, never interpreted as code or YAML. */ }
    } else if (/^'(?:[^']|'')*'$/.test(value)) {
      metadata[field[1]!] = value.slice(1, -1).replace(/''/g, "'");
    } else if (value && !/^[[\]{}|>&*!]/.test(value)) {
      metadata[field[1]!] = value.replace(/[ \t]+#.*$/, "");
    }
  }
  return { body: normalized.slice(match[0].length), metadata };
}

function inlineText(tokens: Token[]): string {
  return tokens.map(token => {
    // Heading metadata is rendered again in titles/TOC; retain math boundaries.
    if (token.type === "reading_math") return `$${token.content}$`;
    if (token.type === "reading_math_display") return `$$${token.content}$$`;
    return token.children ? inlineText(token.children) : token.nesting === 0 ? token.content : "";
  }).join("").trim();
}

// Equivalent to the C0-control/space/DEL/backslash character class this
// replaced. Scanning char codes (rather than a control-character-ranged
// regex class) avoids tripping no-control-regex while matching the same set
// of disallowed destination characters.
function hasUnsafeDestinationCharacter(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= 0x20 || code === 0x7f || value[index] === "\\") return true;
  }
  return false;
}

function safeDestination(target: string, kind: "link" | "image", allowLocal: boolean): string | null {
  const value = target.trim();
  if (!value || hasUnsafeDestinationCharacter(value) || value.startsWith("//")) return null;
  const scheme = /^([a-z][a-z0-9+.-]*):/i.exec(value)?.[1]?.toLowerCase();
  if (scheme) {
    if (scheme === "http" || scheme === "https") {
      try { return new URL(value).hostname ? value : null; } catch { return null; }
    }
    return scheme === "mailto" && kind === "link" ? value : null;
  }
  if (value.startsWith("#")) return kind === "link" ? value : null;
  return allowLocal ? value : null;
}

function resolveDestinations(tokens: Token[], options: RenderMarkdownOptions): void {
  const linkTags: string[] = [];
  for (const token of tokens) {
    if (token.type === "link_open" || token.type === "image") {
      const kind = token.type === "image" ? "image" : "link";
      const attribute = kind === "image" ? "src" : "href";
      const original = String(token.attrGet(attribute) ?? "");
      const resolved = options.resolveLink ? options.resolveLink(original, kind) : original;
      const destination = resolved === null ? null : safeDestination(resolved, kind, Boolean(options.resolveLink));
      if (destination) {
        token.attrSet(attribute, destination);
        if (kind === "link" && /^(?:https?:|mailto:)/i.test(destination)) {
          token.attrSet("target", "_blank");
          token.attrSet("rel", "noopener noreferrer");
        }
        if (kind === "link") linkTags.push("a");
      } else if (kind === "link") {
        token.tag = "span";
        token.attrs = [["class", "unresolved-link"]];
        linkTags.push("span");
      } else {
        token.type = "text";
        token.tag = "";
        token.content = inlineText(token.children ?? []) || token.content;
        token.children = null;
        token.attrs = null;
      }
    } else if (token.type === "link_close") {
      token.tag = linkTags.pop() ?? "a";
    }
    if (token.children) resolveDestinations(token.children, options);
  }
}

function addReadingSyntax(markdown: MarkdownParser): void {
  markdown.inline.ruler.before("escape", "reading_syntax", (state, silent) => {
    const remaining = state.src.slice(state.pos);
    if (remaining.startsWith("<!--")) {
      const close = state.src.indexOf("-->", state.pos + 4);
      state.pos = close < 0 ? state.posMax : close + 3;
      return true;
    }
    const lineBreak = /^<br\s*\/?\s*>/i.exec(remaining);
    if (lineBreak) {
      if (!silent) state.push("hardbreak", "br", 0);
      state.pos += lineBreak[0].length;
      return true;
    }
    const wiki = /^\[\[([^\]\n|]+)(?:\|([^\]\n]*))?\]\]/.exec(remaining);
    if (wiki) {
      if (!silent) {
        const open = state.push("link_open", "a", 1);
        open.attrSet("href", wiki[1]!.trim());
        state.push("text", "", 0).content = wiki[2] || wiki[1]!;
        state.push("link_close", "a", -1);
      }
      state.pos += wiki[0].length;
      return true;
    }
    const delimiter = remaining.startsWith("$$") ? "$$" : remaining.startsWith("$") ? "$" : remaining.startsWith("\\(") ? "\\(" : remaining.startsWith("\\[") ? "\\[" : null;
    if (!delimiter) return false;
    const closeDelimiter = delimiter === "\\(" ? "\\)" : delimiter === "\\[" ? "\\]" : delimiter;
    const start = state.pos + delimiter.length;
    if (delimiter === "$" && /\s/.test(state.src[start] ?? " ")) return false;
    let close = state.src.indexOf(closeDelimiter, start);
    while (close >= 0 && isEscaped(state.src, close)) close = state.src.indexOf(closeDelimiter, close + closeDelimiter.length);
    if (close < 0 || close === start) return false;
    const content = state.src.slice(start, close);
    if (delimiter === "$" && (/\s$/.test(content) || /\n/.test(content) || /\d/.test(state.src[close + 1] ?? ""))) return false;
    if (!silent) {
      const token = state.push(delimiter === "$$" || delimiter === "\\[" ? "reading_math_display" : "reading_math", "", 0);
      token.content = content;
    }
    state.pos = close + closeDelimiter.length;
    return true;
  });

  // Block parsing must happen before paragraphs split display math at blank lines.
  // Only standalone delimiters interrupt paragraphs; fences/indented code stay literal.
  markdown.block.ruler.before("fence", "reading_math_block", (state, startLine, endLine, silent) => {
    if (state.sCount[startLine]! - state.blkIndent >= 4) return false;
    const opening = state.src.slice(state.bMarks[startLine]! + state.tShift[startLine]!, state.eMarks[startLine]).trim();
    if (opening !== "$$" && opening !== "\\[") return false;
    const closing = opening === "$$" ? "$$" : "\\]";
    let closeLine = startLine + 1;
    for (; closeLine < endLine; closeLine += 1) {
      // Do not consume text belonging to a containing list/blockquote's next block.
      if (state.tShift[closeLine]! < 0 || (!state.isEmpty(closeLine) && state.sCount[closeLine]! < state.blkIndent)) return false;
      const line = state.src.slice(state.bMarks[closeLine]! + state.tShift[closeLine]!, state.eMarks[closeLine]).trim();
      if (line === closing) break;
    }
    if (closeLine >= endLine) return false;
    if (silent) return true;
    const token = state.push("reading_math_display", "", 0);
    token.block = true;
    token.map = [startLine, closeLine + 1];
    token.content = state.getLines(startLine + 1, closeLine, state.blkIndent, false);
    state.line = closeLine + 1;
    return true;
  }, { alt: ["paragraph", "reference", "blockquote", "list"] });

  markdown.block.ruler.before("html_block", "reading_comment", (state, startLine, endLine, silent) => {
    const start = state.bMarks[startLine]! + state.tShift[startLine]!;
    if (state.sCount[startLine]! - state.blkIndent >= 4 || !state.src.startsWith("<!--", start)) return false;
    const close = state.src.indexOf("-->", start + 4);
    const limit = close < 0 ? state.src.length : close + 3;
    let nextLine = startLine;
    while (nextLine < endLine && state.eMarks[nextLine]! < limit) nextLine += 1;
    if (nextLine < endLine && state.src.slice(limit, state.eMarks[nextLine]).trim()) return false;
    if (!silent) state.line = Math.min(nextLine + 1, endLine);
    return true;
  });

  const renderMath = (content: string, displayMode: boolean): string => {
    try {
      return katex.renderToString(content, { displayMode, trust: false, throwOnError: false, strict: "ignore", maxExpand: 100, maxSize: 20, output: "htmlAndMathml" });
    } catch {
      return `<code class="math-error">${markdown.utils.escapeHtml(content)}</code>`;
    }
  };
  markdown.renderer.rules.reading_math = (tokens, index) => renderMath(tokens[index]!.content, false);
  markdown.renderer.rules.reading_math_display = (tokens, index) => renderMath(tokens[index]!.content, true);
}

function isEscaped(source: string, offset: number): boolean {
  let slashes = 0;
  for (let index = offset - 1; index >= 0 && source[index] === "\\"; index -= 1) slashes += 1;
  return slashes % 2 === 1;
}
