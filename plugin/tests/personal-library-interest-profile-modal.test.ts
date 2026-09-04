import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Modal, type App } from "obsidian";
import {
  PersonalLibraryInterestProfileModal,
  bufferPoolHeading,
  describeClusterMembers,
  formatConfidence,
  normalizeLines,
  unclassifiedBufferPoolPapers,
  type InterestProfileReviewController,
  type InterestProfileReviewSnapshot,
} from "../src/library/interest-profile-modal";

beforeAll(() => {
  type Options = { cls?: string; text?: string; type?: string; value?: string; attr?: Record<string, string> };
  const proto = HTMLElement.prototype as any;
  proto.addClass ??= function (...classes: string[]) { this.classList.add(...classes); };
  proto.toggleClass ??= function (name: string, value: boolean) { this.classList.toggle(name, value); };
  proto.empty ??= function () { this.replaceChildren(); };
  proto.createEl ??= function (tag: string, options: Options = {}) {
    const element = document.createElement(tag);
    if (options.cls) element.className = options.cls;
    if (options.text !== undefined) element.textContent = options.text;
    if (options.type) element.setAttribute("type", options.type);
    if (options.value) (element as HTMLInputElement).value = options.value;
    for (const [key, value] of Object.entries(options.attr ?? {})) element.setAttribute(key, value);
    this.appendChild(element);
    return element;
  };
  proto.createDiv ??= function (options: Options = {}) { return this.createEl("div", options); };
  proto.createSpan ??= function (options: Options = {}) { return this.createEl("span", options); };
  proto.appendText ??= function (text: string) { this.appendChild(document.createTextNode(text)); };
  proto.setText ??= function (text: string) { this.textContent = text; };
});

beforeEach(() => { Modal.opened.length = 0; });

const fingerprint = `sha256:${"a".repeat(64)}`;
const evidence = `sha256:${"b".repeat(64)}`;
const candidate = {
  id: "candidate-1", text: "Reliability of long-running research agents", discoveryCues: ["agents"],
  representatives: [{ paperKey: "arxiv:2608.00001", evidenceFingerprint: evidence }],
  representativeSetFingerprint: fingerprint, lineage: { candidateIds: ["candidate-1"] },
};

function snapshot(overrides: Partial<InterestProfileReviewSnapshot> = {}): InterestProfileReviewSnapshot {
  return {
    catalog: {
      schemaVersion: 1, revision: 1, scopeFingerprint: fingerprint, identificationFingerprint: fingerprint,
      scanContractFingerprint: fingerprint, createdAt: "2026-08-03T00:00:00.000Z", updatedAt: "2026-08-03T00:00:00.000Z",
      files: {}, papers: {
        "arxiv:2608.00001": {
          paperKey: "arxiv:2608.00001", source: "arxiv", externalId: "2608.00001", title: "Paper <img src=x>",
          authors: ["A"], abstract: "<script>bad()</script>", published: "2026-08-01T00:00:00.000Z",
          updated: "2026-08-01T00:00:00.000Z", primaryCategory: "cs.AI", categories: ["cs.AI"],
          evidenceDepth: "metadata-and-abstract", filePaths: ["paper.pdf"],
        },
      }, summary: { inventoryCount: 1, eligibleFileCount: 1, readyFileCount: 1, unsupportedFileCount: 0, unidentifiedFileCount: 0, failedFileCount: 0, paperCount: 1 },
    } as any,
    proposal: {
      schemaVersion: 4, revision: 0, proposalId: "proposal-1", scopeFingerprint: fingerprint,
      identificationFingerprint: fingerprint, catalogInputFingerprint: fingerprint,
      catalogInputPapers: candidate.representatives, generationContractFingerprint: fingerprint,
      generatedAt: "2026-08-03T00:00:00.000Z",
      topics: [{ id: "topic-1", suggestedName: "Research agents", directions: [candidate] }],
    },
    suggestions: null,
    authorization: { kind: "authorized", rootLabel: "papers", processingDepth: "metadata-and-abstracts", endpoint: "https://example.test" } as any,
    catalogLoadError: null, proposalLoadError: null, suggestionsLoadError: null,
    ...overrides,
  };
}

function controller(initial = snapshot()) {
  let current = initial;
  const update = vi.fn(async () => current);
  const mock: InterestProfileReviewController = {
    snapshot: () => current,
    reload: vi.fn(async () => current), generate: vi.fn(async () => undefined),
    updateProposal: update, discardProposal: update,
    renameTopic: vi.fn(async () => current),
    acceptTopics: vi.fn(async () => current),
  };
  return { mock, set: (next: InterestProfileReviewSnapshot) => { current = next; } };
}

function open(ctrl: InterestProfileReviewController) {
  const modal = new PersonalLibraryInterestProfileModal({} as App, ctrl);
  modal.open();
  return modal;
}

function button(root: HTMLElement, text: string): HTMLButtonElement {
  const found = Array.from(root.querySelectorAll("button")).find((item) => item.textContent === text);
  if (!found) throw new Error(`missing button ${text}`);
  return found;
}

async function confirmChoice(text: string): Promise<void> {
  await vi.waitFor(() => expect(Modal.opened.length).toBeGreaterThan(1));
  button(Modal.opened.at(-1)!.contentEl, text).click();
  await Promise.resolve();
}

describe("accepting a proposed structure", () => {
  /**
   * ADR 0014 §1: the researcher reviews and accepts a structure. The modal's
   * job here is that nothing reaches settings without being picked, that the
   * suggested name is editable before the tag is derived from it, and that
   * what is uncovered stays visible.
   */
  it("accepts only the topics that were selected", async () => {
    const ctrl = controller();
    const modal = open(ctrl.mock);
    const root = (modal as any).contentEl as HTMLElement;

    const accept = button(root, "Accept 0 topic(s) into settings");
    expect(accept.disabled).toBe(true);

    const checkbox = root.querySelector<HTMLInputElement>('input[aria-label="Accept Research agents"]')!;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event("change"));

    const armed = button((modal as any).contentEl, "Accept 1 topic(s) into settings");
    expect(armed.disabled).toBe(false);
    armed.dispatchEvent(new Event("click"));
    await Promise.resolve();
    await Promise.resolve();
    expect(ctrl.mock.acceptTopics).toHaveBeenCalledWith(["topic-1"]);
  });

  it("routes a renamed topic through the controller before any tag is derived", async () => {
    const ctrl = controller();
    const modal = open(ctrl.mock);
    const root = (modal as any).contentEl as HTMLElement;
    const name = root.querySelector<HTMLInputElement>('input[aria-label="Topic name"]')!;
    expect(name.value).toBe("Research agents");
    name.value = "Agent reliability";
    name.dispatchEvent(new Event("change"));
    await Promise.resolve();
    expect(ctrl.mock.renameTopic).toHaveBeenCalledWith({
      topicId: "topic-1", suggestedName: "Agent reliability",
    });
  });

  it("says what the proposal does not cover instead of leaving it silent", () => {
    const base = snapshot();
    const uncovered = { paperKey: "arxiv:2608.00002", evidenceFingerprint: evidence };
    const ctrl = controller(snapshot({
      proposal: { ...base.proposal!, catalogInputPapers: [...base.proposal!.catalogInputPapers, uncovered] },
    }));
    const modal = open(ctrl.mock);
    const text = ((modal as any).contentEl as HTMLElement).textContent ?? "";
    expect(unclassifiedBufferPoolPapers(ctrl.mock.snapshot().proposal)).toHaveLength(2);
    expect(text).toContain("No proposed direction covers these papers");
    // The incremental flow is dark until it is rebuilt around topics; a blank
    // space would read as a bug.
    expect(text).toContain("Incremental suggestions");
  });
});

describe("review modal stylesheet", () => {
  const css = (): string => readFileSync(resolve(process.cwd(), "styles.css"), "utf-8");

  // The two-column form has a hard minimum of 12rem + 16rem. Whether it fits
  // is a property of the modal, which sizes itself with min(60rem, 92vw) — not
  // of the window. Keying the collapse on the window let the modal be narrower
  // than its own content and clip it.
  // Width has to be asked of the modal box. Obsidian sizes .modal itself, so a
  // wide rule on the content element inside it cannot widen anything — the
  // content simply overflows the box and is clipped, which is what happened.
  it("sizes the modal box, not the content inside it", () => {
    const modal = open(controller().mock);
    expect(modal.modalEl.classList.contains("arxiv-daily-interest-review-modal")).toBe(true);
    const sheet = css();
    expect(sheet).toMatch(/\.arxiv-daily-interest-review-modal\s*\{[^}]*width:/);
    // The content element must not carry a width of its own; having no rule
    // block at all satisfies that just as well as having one without a width.
    const content = /\.arxiv-daily-interest-review\s*\{([^}]*)\}/.exec(sheet);
    if (content) expect(content[1]).not.toMatch(/(^|[^-])width:/);
  });

  // Two columns inside a modal is what cut the name input in half and squeezed
  // the paper list into a slit. One full-width column has nothing to collapse,
  // so no width condition can bring the clipping back.
  it("stacks the edit form in one column at every width", () => {
    const sheet = css();
    const form = /\.arxiv-daily-interest-review__form\s*\{([^}]*)\}/.exec(sheet);
    expect(form).not.toBeNull();
    expect(form![1]).toMatch(/grid-template-columns:\s*1fr/);
    const conditional = sheet.match(/@(?:media|container)[^{]*\{[\s\S]*?\n\}/g) ?? [];
    expect(conditional.some((block) =>
      /\.arxiv-daily-interest-review__form\s*\{[^}]*grid-template-columns/.test(block),
    )).toBe(false);
  });
});
