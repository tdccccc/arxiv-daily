import katex from "katex";
import type { MarkdownNode } from "../markdown";
import { h, fragment, type DomAttrs } from "./dom";

// Builds real DOM from the token stream `../markdown` produces (`renderMarkdown().nodes`,
// `renderInlineMarkdownNodes()`), instead of parsing an HTML string. `MarkdownNode` already
// carries the same tag/nesting/attrs shape markdown-it's own tokens do, so this walks it the
// same generic way markdown-it's default renderer walks tokens: open/close pairs push and pop
// an element stack, and only the handful of leaf constructs markdown-it itself special-cases
// (`code_inline`, `code_block`, `fence`, `image`, line breaks, our `reading_math*` tokens) need
// bespoke handling below. Every element is created through `h()` (see `./dom`), never through
// `innerHTML` or a direct `document.createElement` call here.

function textOf(nodes: MarkdownNode[]): string {
  let text = "";
  for (const node of nodes) {
    if (node.type === "text" || node.type === "code_inline") text += node.content;
    else if (node.type === "image") text += textOf(node.children ?? []);
    else if (node.type === "softbreak" || node.type === "hardbreak") text += "\n";
  }
  return text;
}

function attrsOf(attrs: MarkdownNode["attrs"]): DomAttrs | null {
  return attrs ? Object.fromEntries(attrs) : null;
}

function mathElement(content: string, displayMode: boolean): HTMLElement {
  const el = h(displayMode ? "div" : "span");
  try {
    katex.render(content, el, { displayMode, trust: false, throwOnError: false, strict: "ignore", maxExpand: 100, maxSize: 20, output: "htmlAndMathml" });
  } catch {
    el.replaceChildren(h("code", { class: "math-error" }, content));
  }
  return el;
}

/** Appends the DOM built from a markdown-it-derived token stream into `container`. */
export function appendMarkdownNodes(nodes: MarkdownNode[], container: ParentNode): void {
  const stack: ParentNode[] = [container];
  const current = (): ParentNode => stack[stack.length - 1]!;
  for (const node of nodes) {
    switch (node.type) {
      case "inline":
        appendMarkdownNodes(node.children ?? [], current());
        break;
      case "text":
        current().append(document.createTextNode(node.content));
        break;
      case "softbreak":
        current().append(document.createTextNode("\n"));
        break;
      case "hardbreak":
        current().append(h("br"));
        break;
      case "code_inline":
        current().append(h("code", attrsOf(node.attrs), node.content));
        break;
      case "code_block":
        current().append(h("pre", attrsOf(node.attrs), h("code", null, node.content)));
        break;
      case "fence": {
        const lang = node.info.trim().split(/\s+/)[0];
        const codeAttrs = attrsOf(node.attrs) ?? {};
        if (lang) codeAttrs.class = `${codeAttrs.class ? `${String(codeAttrs.class)} ` : ""}language-${lang}`;
        current().append(h("pre", null, h("code", codeAttrs, node.content)));
        break;
      }
      case "image":
        current().append(h("img", { ...attrsOf(node.attrs), alt: textOf(node.children ?? []) }));
        break;
      case "reading_math":
        current().append(mathElement(node.content, false));
        break;
      case "reading_math_display":
        current().append(mathElement(node.content, true));
        break;
      default:
        if (node.nesting === 1) {
          if (node.type === "heading_open" && node.meta?.readingAppendix) current().append(h("hr", { class: "reading-appendix-divider" }));
          const el = h(node.tag || "span", attrsOf(node.attrs));
          current().append(el);
          stack.push(el);
        } else if (node.nesting === -1) {
          if (stack.length > 1) stack.pop();
        } else if (node.tag) {
          current().append(h(node.tag, attrsOf(node.attrs)));
        }
    }
  }
}

/** Renders a full document/section body (`renderMarkdown().nodes`) into a `.markdown-body` element. */
export function markdownBody(nodes: MarkdownNode[]): HTMLDivElement {
  const body = h("div", { class: "markdown-body" });
  appendMarkdownNodes(nodes, body);
  return body;
}

/** Renders an inline projection (`renderInlineMarkdownNodes()`) as a fragment, for titles/snippets. */
export function markdownInline(nodes: MarkdownNode[]): DocumentFragment {
  const frag = fragment();
  appendMarkdownNodes(nodes, frag);
  return frag;
}
