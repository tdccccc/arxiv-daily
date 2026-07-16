import { beforeAll, describe, expect, it, vi } from "vitest";
import type { PaperIndexEntry, PaperSearchResult } from "@arxiv-daily/core";
import { renderSimilarPapersModal } from "../src/dashboard/similar-papers-modal";

beforeAll(() => {
  const proto = HTMLElement.prototype as HTMLElement & {
    empty?: () => void;
    addClass?: (...classes: string[]) => void;
    createEl?: (tag: string, options?: { cls?: string; text?: string; attr?: Record<string, string> }) => HTMLElement;
  };
  proto.empty ??= function () { this.replaceChildren(); };
  proto.addClass ??= function (...classes: string[]) { this.classList.add(...classes); };
  proto.createEl ??= function (tag, options = {}) {
    const element = document.createElement(tag);
    if (options.cls) element.className = options.cls;
    if (options.text) element.textContent = options.text;
    for (const [key, value] of Object.entries(options.attr ?? {})) element.setAttribute(key, value);
    this.appendChild(element);
    return element;
  };
});

function paper(id: string): PaperIndexEntry {
  return {
    arxivId: id,
    source: "arxiv",
    title: `Paper ${id}`,
    authors: ["A. Author"],
    published: "2026-07-01",
    updated: "2026-07-01",
    category: "cs.LG",
    categories: ["cs.LG"],
    topics: ["retrieval"],
    primaryTopic: "retrieval",
    detail: true,
    status: "inbox",
    priority: "normal",
    seenDates: ["2026-07-01"],
    dailyReports: ["arxiv-daily/daily/2026-07-01.md"],
    paperPath: `arxiv-daily/papers/${id}.md`,
    arxivUrl: `https://arxiv.org/abs/${id}`,
    pdfUrl: `https://arxiv.org/pdf/${id}`,
    pdfPath: "",
    zoteroKey: "",
    zoteroUri: "",
    citationKey: "",
    projects: [],
  };
}

describe("Similar Papers modal", () => {
  it("renders local reasons without percentage scores and exposes accessible callbacks", () => {
    const source = paper("2607.00001");
    const candidate = paper("2607.00002");
    const result: PaperSearchResult = {
      entry: candidate,
      score: 12.345,
      reasons: [{ field: "title", terms: ["retrieval"], text: "Matched title: retrieval" }],
    };
    const callbacks = {
      openDetail: vi.fn(),
      openDaily: vi.fn(),
      openArxiv: vi.fn(),
      openPdf: vi.fn(),
    };
    const content = document.createElement("div");

    renderSimilarPapersModal(content, { source, results: [result], ...callbacks });

    expect(content.textContent).toContain("Matched title: retrieval");
    expect(content.textContent).not.toContain("%");
    const buttons = [...content.querySelectorAll<HTMLButtonElement>("button")];
    expect(buttons.map((button) => button.getAttribute("aria-label"))).toEqual([
      "Open detail", "Open daily report", "Open arXiv", "Open PDF",
    ]);
    buttons.forEach((button) => button.click());
    expect(callbacks.openDetail).toHaveBeenCalledWith(candidate);
    expect(callbacks.openDaily).toHaveBeenCalledWith(candidate);
    expect(callbacks.openArxiv).toHaveBeenCalledWith(candidate);
    expect(callbacks.openPdf).toHaveBeenCalledWith(candidate);
  });

  it("routes rejecting action callbacks to the safe error handler", async () => {
    const source = paper("2607.00001");
    const candidate = paper("2607.00002");
    const error = new Error("open failed");
    const onActionError = vi.fn();
    const content = document.createElement("div");
    renderSimilarPapersModal(content, {
      source,
      results: [{ entry: candidate, score: 1, reasons: [] }],
      openDetail: vi.fn(async () => { throw error; }),
      openDaily: vi.fn(),
      openArxiv: vi.fn(),
      openPdf: vi.fn(),
      onActionError,
    });

    content.querySelector<HTMLButtonElement>('button[aria-label="Open detail"]')?.click();
    await Promise.resolve();
    await Promise.resolve();

    expect(onActionError).toHaveBeenCalledWith(error, "Open detail", candidate);
  });

  it("renders an explicit empty local result", () => {
    const content = document.createElement("div");
    renderSimilarPapersModal(content, {
      source: paper("2607.00001"),
      results: [],
      openDetail: vi.fn(), openDaily: vi.fn(), openArxiv: vi.fn(), openPdf: vi.fn(),
    });
    expect(content.textContent).toContain("local paper index");
  });
});
