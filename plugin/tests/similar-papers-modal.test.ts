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

  it("shows topic, date, and resource availability while disabling missing local actions", () => {
    const source = paper("2607.00001");
    const candidate = paper("2607.00002");
    candidate.detail = false;
    candidate.paperPath = null;
    candidate.dailyReports = [];
    candidate.pdfPath = "arxiv-daily/pdfs/2607.00002.pdf";
    const callbacks = {
      openDetail: vi.fn(), openDaily: vi.fn(), openArxiv: vi.fn(), openPdf: vi.fn(),
    };
    const content = document.createElement("div");

    renderSimilarPapersModal(content, {
      source,
      results: [{ entry: candidate, score: 1, reasons: [] }],
      ...callbacks,
    });

    expect(content.textContent).toContain("retrieval · 2026-07-01");
    expect(content.textContent).toContain("No detail · No daily report · PDF saved");
    const detail = content.querySelector<HTMLButtonElement>('button[aria-label="Open detail unavailable"]');
    const daily = content.querySelector<HTMLButtonElement>('button[aria-label="Open daily report unavailable"]');
    expect(detail?.disabled).toBe(true);
    expect(daily?.disabled).toBe(true);
    detail?.click();
    daily?.click();
    expect(callbacks.openDetail).not.toHaveBeenCalled();
    expect(callbacks.openDaily).not.toHaveBeenCalled();
  });

  it("limits rendering to ten local results", () => {
    const content = document.createElement("div");
    renderSimilarPapersModal(content, {
      source: paper("2607.00001"),
      results: Array.from({ length: 12 }, (_, index) => ({
        entry: paper(`2607.${String(index + 2).padStart(5, "0")}`), score: 12 - index, reasons: [],
      })),
      openDetail: vi.fn(), openDaily: vi.fn(), openArxiv: vi.fn(), openPdf: vi.fn(),
    });
    expect(content.querySelectorAll("li")).toHaveLength(10);
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

  it("contains synchronous callback and error-handler failures", () => {
    const candidate = paper("2607.00002");
    const content = document.createElement("div");
    renderSimilarPapersModal(content, {
      source: paper("2607.00001"),
      results: [{ entry: candidate, score: 1, reasons: [] }],
      openDetail: () => { throw new Error("sync failure"); },
      openDaily: vi.fn(), openArxiv: vi.fn(), openPdf: vi.fn(),
      onActionError: () => { throw new Error("handler failure"); },
    });

    expect(() => content.querySelector<HTMLButtonElement>('button[aria-label="Open detail"]')?.click())
      .not.toThrow();
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
