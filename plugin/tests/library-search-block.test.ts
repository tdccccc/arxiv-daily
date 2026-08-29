import { beforeAll, describe, expect, it, vi } from "vitest";
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
  proto.createSpan ??= function (options: Options = {}) {
    return this.createEl!("span", options);
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
          scoreKind: "cosine",
          rankingScore: 0.0325,
          rankingScoreKind: "rrf",
          hits: [],
        },
        {
          paperKey: "arxiv:2607.01002", title: "Second", score: 1.7, scoreKind: "bm25",
          rankingScore: 0.016, rankingScoreKind: "rrf",
          hits: [],
        },
      ],
    });
    expect(container.textContent).toContain("Library matches");
    expect(container.textContent).toContain("Library Paper");
    expect(container.textContent).toContain("arxiv:2607.01001");
    expect(container.textContent).toContain("arxiv:2607.01002");
    expect(container.textContent).not.toContain("best semantic evidence");
    expect(container.textContent).not.toContain("lexical match");
    expect(container.textContent).not.toContain("0.032");
    expect(container.textContent).not.toContain("—");
    expect(container.querySelectorAll("li")).toHaveLength(2);
  });

  it("opens the whole PDF and hides passage evidence even when hits are present", () => {
    const openLibraryPdf = vi.fn();
    const container = document.createElement("div");
    renderLibrarySearchBlock(container, {
      kind: "matches",
      matches: [{
        paperKey: "arxiv:2607.01001",
        title: "Evidence paper",
        filePath: "papers/evidence paper.pdf",
        score: 0.812345,
        scoreKind: "cosine",
        rankingScore: 0.0325,
        rankingScoreKind: "rrf",
        hits: [{
          source: "dense",
          scoreKind: "cosine",
          score: 0.812345,
          chunkIndex: 3,
          chunkId: "chunk-3",
          headings: ["Methods", "Retrieval"],
          locator: { pageStart: 7 },
          page: 7,
          text: "A matching passage about retrieval evidence.",
        }],
      }],
    }, { openLibraryPdf });

    expect(container.textContent).toContain("Evidence paper");
    expect(container.textContent).toContain("papers/evidence paper.pdf");
    expect(container.textContent).not.toContain("Open PDF");
    expect(container.textContent).not.toContain("Methods / Retrieval");
    expect(container.textContent).not.toContain("A matching passage about retrieval evidence.");
    expect(container.textContent).not.toContain("Page 7");
    expect(container.textContent).not.toContain("Open page 7");
    const button = container.querySelector<HTMLButtonElement>('button[aria-label="Open PDF"]');
    expect(button?.classList.contains("clickable-icon")).toBe(true);
    expect(button?.closest(".arxiv-daily-dashboard__library-header")).not.toBeNull();
    expect(button?.parentElement?.querySelector(".arxiv-daily-dashboard__library-title")?.textContent)
      .toBe("Evidence paper");
    button?.click();
    expect(openLibraryPdf).toHaveBeenCalledWith(
      expect.objectContaining({ paperKey: "arxiv:2607.01001", filePath: "papers/evidence paper.pdf" }),
    );
    expect(openLibraryPdf.mock.calls[0]?.length).toBe(1);
  });

  it("contains evidence action and error-handler failures", () => {
    const container = document.createElement("div");
    renderLibrarySearchBlock(container, {
      kind: "matches",
      matches: [{
        paperKey: "arxiv:2607.01001",
        title: "Evidence paper",
        filePath: "papers/evidence.pdf",
        score: 0.8,
        scoreKind: "cosine",
        rankingScore: 0.03,
        rankingScoreKind: "rrf",
        hits: [{
          source: "dense",
          scoreKind: "cosine",
          score: 0.8,
          chunkIndex: 0,
          chunkId: "chunk-0",
          headings: [],
          locator: { pageStart: 1 },
          page: 1,
          text: "Evidence",
        }],
      }],
    }, {
      openLibraryPdf: () => { throw new Error("open failure"); },
      onActionError: () => { throw new Error("handler failure"); },
    });

    expect(() => container.querySelector<HTMLButtonElement>('button[aria-label="Open PDF"]')?.click())
      .not.toThrow();
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
