import { beforeAll, describe, expect, it, vi } from "vitest";
import { FullTextSearchResultsModal } from "../src/commands";

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
  proto.createDiv ??= function (options: Options = {}) { return this.createEl!("div", options); };
  proto.createSpan ??= function (options: Options = {}) { return this.createEl!("span", options); };
});

describe("FullTextSearchResultsModal", () => {
  it("renders ranked papers and delegates opening the whole PDF", () => {
    const openLibraryPdf = vi.fn();
    const modal = new FullTextSearchResultsModal({} as any, [{
      paperKey: "arxiv:2607.00001",
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
        headings: ["Results"],
        locator: { pageStart: 9 },
        page: 9,
        text: "Evidence from the evaluated method.",
      }],
    }], { openLibraryPdf });

    modal.onOpen();

    expect(modal.contentEl.textContent).toContain("Full-text search results");
    expect(modal.contentEl.textContent).toContain("Evidence paper");
    expect(modal.contentEl.textContent).not.toContain("Results");
    expect(modal.contentEl.textContent).not.toContain("Page 9");
    modal.contentEl.querySelector<HTMLButtonElement>('button[aria-label="Open PDF"]')?.click();
    expect(openLibraryPdf).toHaveBeenCalledWith(
      expect.objectContaining({ paperKey: "arxiv:2607.00001" }),
    );
  });
});
