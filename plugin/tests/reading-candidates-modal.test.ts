import { beforeAll, describe, expect, it, vi } from "vitest";

/**
 * happy-dom elements lack the Obsidian extensions the modal renderer uses;
 * mirror the tiny surface here (same pattern as commands.test.ts).
 */
beforeAll(() => {
  type Options = { cls?: string; text?: string; attr?: Record<string, string> };
  const proto = HTMLElement.prototype as HTMLElement & {
    empty?: () => void;
    addClass?: (...classes: string[]) => void;
    createEl?: (tag: string, options?: Options) => HTMLElement;
    createSpan?: (options?: Options) => HTMLElement;
  };
  proto.empty ??= function () {
    while (this.firstChild) this.removeChild(this.firstChild);
  };
  proto.addClass ??= function (...classes: string[]) {
    this.classList.add(...classes);
  };
  proto.createEl ??= function (tag: string, options: Options = {}) {
    const element = this.ownerDocument.createElement(tag);
    if (options.cls) element.className = options.cls;
    if (options.text !== undefined) element.textContent = options.text;
    for (const [key, value] of Object.entries(options.attr ?? {})) {
      element.setAttribute(key, value);
    }
    this.appendChild(element);
    return element;
  };
  proto.createSpan ??= function (options: Options = {}) {
    return (proto.createEl as (tag: string, options?: Options) => HTMLElement)
      .call(this, "span", options);
  };
  proto.createDiv ??= function (options: Options = {}) {
    return (proto.createEl as (tag: string, options?: Options) => HTMLElement)
      .call(this, "div", options);
  };
});
import {
  groupPendingCandidates,
  renderReadingCandidatesModal,
  type ReadingCandidatesModalOptions,
} from "../src/library/reading-candidates-modal";
import type { ReadingCandidateRecord, ReadingCandidatesDocument } from "@arxiv-daily/core";

function candidate(
  index: number,
  overrides: Partial<ReadingCandidateRecord> = {},
): ReadingCandidateRecord {
  return {
    paperKey: `arxiv:2608.${String(index).padStart(5, "0")}`,
    arxivId: `2608.${String(index).padStart(5, "0")}`,
    title: `Candidate ${index}`,
    authors: "A. Author",
    topic: "astrophysics",
    source: {
      kind: "library",
      manualTopics: [],
      directions: [{ id: "direction-1", name: "Cosmology" }],
      reportPath: "arxiv-daily/daily/2026-08-12.md",
      reportDate: "2026-08-12",
    },
    relatedPriorWorks: [{ paperKey: "arxiv:2305.00001", title: "Prior survey" }],
    provisionalNovelty: {
      differenceType: "new-dataset",
      comparisonBasis: ["arxiv:2305.00001"],
      evidenceDepth: "metadata-and-abstract",
      explanation: "A new dataset.",
    },
    savedAt: `2026-08-1${index}T00:00:00.000Z`,
    updatedAt: `2026-08-1${index}T00:00:00.000Z`,
    ...overrides,
  };
}

function makeDocument(records: ReadingCandidateRecord[]): ReadingCandidatesDocument {
  return {
    schemaVersion: 1,
    revision: 1,
    scopeFingerprint: `sha256:${"a".repeat(64)}`,
    identificationFingerprint: `sha256:${"b".repeat(64)}`,
    updatedAt: "2026-08-13T00:00:00.000Z",
    candidates: Object.fromEntries(records.map((record) => [record.paperKey, record])),
  };
}

describe("groupPendingCandidates", () => {
  it("groups by the first direction and manual topic fallback, newest saved first", () => {
    const manual = candidate(3, {
      source: {
        kind: "manual",
        manualTopics: [{ tag: "cosmology", name: "Cosmology topic" }],
        directions: [],
        reportPath: "arxiv-daily/daily/2026-08-12.md",
        reportDate: "2026-08-12",
      },
    });
    const groups = groupPendingCandidates([candidate(1), candidate(2), manual]);
    expect(groups).toHaveLength(2);
    expect(groups[0]?.label).toBe("Cosmology");
    expect(groups[0]?.candidates.map(({ paperKey }) => paperKey)).toEqual([
      "arxiv:2608.00002",
      "arxiv:2608.00001",
    ]);
    expect(groups[1]?.label).toBe("Cosmology topic");
  });
});

describe("renderReadingCandidatesModal", () => {
  function render(
    records: ReadingCandidateRecord[],
    decide: ReadingCandidatesModalOptions["decide"],
  ): {
    container: HTMLElement;
    options: ReadingCandidatesModalOptions;
  } {
    const container = document.createElement("div");
    const options: ReadingCandidatesModalOptions = {
      getCandidates: () => makeDocument(records),
      decide,
      remove: vi.fn(async () => true),
      onError: vi.fn(),
    };
    renderReadingCandidatesModal(container, options);
    return { container, options };
  }

  it("renders pending candidates grouped with decision buttons and counts", () => {
    const decided = candidate(2, { decision: { kind: "skim", at: "2026-08-12T00:00:00.000Z" } });
    const { container } = render([candidate(1), decided], vi.fn(async () => true));
    expect(container.textContent).toContain("1 pending · 1 decided");
    expect(container.textContent).toContain("Candidate 1");
    expect(container.textContent).not.toContain("Candidate 2");
    expect(container.textContent).toContain("Read closely");
    expect(container.textContent).toContain("Related: Prior survey");
    expect(container.textContent).toContain("A new dataset.");
  });

  it("routes decisions to the decide callback with the right kind", async () => {
    const decide = vi.fn(async () => true);
    const { container } = render([candidate(1)], decide);
    const buttons = container.querySelectorAll<HTMLButtonElement>(
      ".arxiv-daily-reading-candidates-modal__action",
    );
    const skim = [...buttons].find((button) => button.textContent?.includes("Skim"));
    expect(skim).toBeDefined();
    skim?.click();
    await vi.waitFor(() => expect(decide).toHaveBeenCalledWith("arxiv:2608.00001", "skim"));
  });

  it("renders an empty state without candidates", () => {
    const container = document.createElement("div");
    renderReadingCandidatesModal(container, {
      getCandidates: () => makeDocument([]),
      decide: vi.fn(async () => true),
      remove: vi.fn(async () => true),
    });
    expect(container.textContent).toContain("No reading candidates yet");
  });

  it("routes removal to the remove callback", async () => {
    const remove = vi.fn(async () => true);
    const container = document.createElement("div");
    renderReadingCandidatesModal(container, {
      getCandidates: () => makeDocument([candidate(1)]),
      decide: vi.fn(async () => true),
      remove,
    });
    const button = container.querySelector<HTMLButtonElement>(
      ".arxiv-daily-reading-candidates-modal__remove",
    );
    button?.click();
    await vi.waitFor(() => expect(remove).toHaveBeenCalledWith("arxiv:2608.00001"));
  });
});
