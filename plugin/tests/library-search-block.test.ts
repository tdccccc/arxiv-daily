import { beforeAll, describe, expect, it } from "vitest";
import {
  renderLibrarySearchBlock,
  type LibrarySearchState,
} from "../src/dashboard/library-search-block";

beforeAll(() => {
  type Options = { cls?: string; text?: string; attr?: Record<string, string> };
  const proto = HTMLElement.prototype as any;
  proto.empty ??= function () { this.replaceChildren(); };
  proto.addClass ??= function (...classes: string[]) { this.classList.add(...classes); };
  proto.createEl ??= function (tag: string, options: Options = {}) {
    const element = document.createElement(tag);
    if (options.cls) element.className = options.cls;
    if (options.text !== undefined) element.textContent = options.text;
    for (const [key, value] of Object.entries(options.attr ?? {})) element.setAttribute(key, value);
    this.appendChild(element);
    return element;
  };
  proto.createDiv ??= function (options: Options = {}) {
    return this.createEl!("div", options);
  };
});

function render(state: LibrarySearchState): HTMLElement {
  const container = document.createElement("div");
  renderLibrarySearchBlock(container, state);
  return container;
}

describe("renderLibrarySearchBlock", () => {
  it("renders the loading state without a heading", () => {
    const container = render({ kind: "loading" });
    expect(container.textContent).toBe("Searching your library…");
    expect(container.textContent).not.toContain("Library matches");
  });

  it("renders matches with title, paper key, and similarity", () => {
    const container = render({
      kind: "matches",
      matches: [
        {
          paperKey: "arxiv:2607.01001",
          title: "Library Paper",
          score: 0.812345,
        },
        { paperKey: "arxiv:2607.01002", title: "Second", score: 0.5 },
      ],
    });
    expect(container.textContent).toContain("Library matches");
    expect(container.textContent).toContain("Library Paper");
    expect(container.textContent).toContain("arxiv:2607.01001 · similarity 0.812");
    expect(container.textContent).not.toContain("—");
    expect(container.querySelectorAll("li")).toHaveLength(2);
  });

  it("renders the empty state", () => {
    const container = render({ kind: "empty" });
    expect(container.textContent).toContain("No library matches for this query.");
  });

  it("renders failures inline without escaping", () => {
    const container = render({ kind: "error", message: "no full-text index" });
    expect(container.textContent).toContain("Library search unavailable: no full-text index");
    expect(container.classList.contains("arxiv-daily-dashboard__library-results")).toBe(true);
  });

  it("replaces prior content on each render", () => {
    const container = document.createElement("div");
    renderLibrarySearchBlock(container, { kind: "loading" });
    renderLibrarySearchBlock(container, { kind: "empty" });
    expect(container.textContent).not.toContain("Searching your library…");
    expect(container.textContent).toContain("No library matches");
  });
});
