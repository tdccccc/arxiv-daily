// Shared DOM-construction helpers for the workbench web UI. Every view used
// to build markup as template-literal HTML strings and assign them through
// `innerHTML`/`outerHTML`/`insertAdjacentHTML`; Obsidian's hosted plugin
// review flags every one of those sinks (it scans this whole repo's
// production TypeScript, not just the plugin). `h()`/`svg()` build real DOM
// nodes instead, so there is no HTML string to parse and no sink to flag.
// Keeping `document.createElement`/`createElementNS` calls inside this one
// module (rather than scattered across every view) keeps that surface small
// and easy to audit.

export type DomChild = Node | string | number | false | null | undefined | DomChild[];
export type DomAttrs = Record<string, string | number | boolean | null | undefined>;

function appendChildren(el: ParentNode, children: DomChild[]): void {
  for (const child of children) {
    if (child === false || child === null || child === undefined) continue;
    if (Array.isArray(child)) { appendChildren(el, child); continue; }
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

function applyAttrs(el: Element, attrs?: DomAttrs | null): void {
  if (!attrs) return;
  for (const [name, value] of Object.entries(attrs)) {
    if (value === false || value === null || value === undefined) continue;
    el.setAttribute(name, value === true ? "" : String(value));
  }
}

/** Builds one HTML element with attributes and children; never touches innerHTML. */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs?: DomAttrs | null, ...children: DomChild[]): HTMLElementTagNameMap[K];
// Dynamic (non-literal) tag names — e.g. a markdown-it token's own `tag` field — fall back to a
// plain `HTMLElement`, so the one generic token-walker in `markdown-dom.ts` can still create
// elements through this shared helper instead of calling `document.createElement` itself.
export function h(tag: string, attrs?: DomAttrs | null, ...children: DomChild[]): HTMLElement;
export function h(tag: string, attrs?: DomAttrs | null, ...children: DomChild[]): HTMLElement {
  const el = document.createElement(tag);
  applyAttrs(el, attrs);
  appendChildren(el, children);
  return el;
}

/** Builds one SVG element (and, recursively, its children) via `createElementNS`. */
export function svg(tag: string, attrs?: DomAttrs | null, ...children: DomChild[]): SVGElement {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  applyAttrs(el, attrs);
  appendChildren(el, children);
  return el;
}

/** Collects a list of nodes under one `DocumentFragment`, for callers that need several siblings. */
export function fragment(...children: DomChild[]): DocumentFragment {
  const frag = document.createDocumentFragment();
  appendChildren(frag, children);
  return frag;
}
