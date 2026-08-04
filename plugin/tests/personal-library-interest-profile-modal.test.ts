import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { Modal, type App } from "obsidian";
import {
  PersonalLibraryInterestProfileModal,
  normalizeLines,
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
  id: "candidate-1", name: "Agents", description: "Reliable agents", discoveryCues: ["agents"],
  representatives: [{ paperKey: "arxiv:2608.00001", evidenceFingerprint: evidence }],
  representativeSetFingerprint: fingerprint, lineage: { candidateIds: ["candidate-1"] },
};
const direction = {
  id: "direction-1", status: "active" as const, name: "Confirmed agents", description: "Confirmed work",
  discoveryCues: ["confirmed"], representatives: candidate.representatives,
  representativeSetFingerprint: fingerprint,
  lineage: { proposalIds: ["proposal-1"], candidateIds: ["candidate-1"], directionIds: [] },
  createdAt: "2026-08-03T00:00:00.000Z", updatedAt: "2026-08-03T00:00:00.000Z",
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
      schemaVersion: 2, revision: 0, proposalId: "proposal-1", scopeFingerprint: fingerprint,
      identificationFingerprint: fingerprint, catalogInputFingerprint: fingerprint,
      catalogInputPapers: candidate.representatives, generationContractFingerprint: fingerprint,
      generatedAt: "2026-08-03T00:00:00.000Z", candidates: [candidate],
    },
    profile: {
      schemaVersion: 2, revision: 0, scopeFingerprint: fingerprint, identificationFingerprint: fingerprint,
      updatedAt: "2026-08-03T00:00:00.000Z", directions: [direction],
    },
    eligibility: {
      documentDiagnostics: [], eligibleDirections: [direction],
      diagnostics: [{ directionId: direction.id, eligible: true, reasons: [] }],
    },
    authorization: { kind: "authorized", rootLabel: "papers", processingDepth: "metadata-and-abstracts", endpoint: "https://example.test" } as any,
    catalogLoadError: null, proposalLoadError: null, profileLoadError: null,
    ...overrides,
  };
}

function controller(initial = snapshot()) {
  let current = initial;
  const update = vi.fn(async () => current);
  const mock: InterestProfileReviewController = {
    snapshot: () => current,
    reload: vi.fn(async () => current), generate: vi.fn(async () => undefined),
    updateProposal: update, mergeProposals: update, discardProposal: update, confirmProposal: update,
    updateConfirmed: update, mergeConfirmed: update, enable: update, disable: update, remove: update,
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

describe("personal library interest profile modal", () => {
  it("renders accessible tabs, authority disclosure, diagnostics, and hostile text as text only", () => {
    const { mock } = controller(snapshot({
      proposal: { ...snapshot().proposal!, candidates: [{ ...candidate, name: '<img class="injected">' }] },
    }));
    const modal = open(mock);
    const tabs = modal.contentEl.querySelectorAll('[role="tab"]');
    expect(tabs).toHaveLength(2);
    expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");
    expect(modal.contentEl.querySelector('[role="tabpanel"]')?.getAttribute("aria-labelledby")).toBe(tabs[0]?.id);
    expect(modal.contentEl.textContent).toContain("do not affect discovery");
    expect(modal.contentEl.textContent).toContain("metadata and abstracts, not full text");
    expect(modal.contentEl.textContent).toContain('<img class="injected">');
    expect(modal.contentEl.querySelector("img")).toBeNull();
    button(modal.contentEl, "Confirmed").click();
    expect(modal.contentEl.textContent).toContain("Eligible with current catalog");
    expect(modal.contentEl.textContent).toContain("metadata and abstract only");
  });

  it("activates and focuses tabs with arrow, Home, and End keys using roving tabindex", () => {
    const { mock } = controller();
    const modal = open(mock);
    document.body.appendChild(modal.contentEl);
    let proposed = button(modal.contentEl, "Proposed");
    proposed.focus();
    proposed.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    let confirmed = button(modal.contentEl, "Confirmed");
    expect(confirmed.getAttribute("aria-selected")).toBe("true");
    expect(confirmed.tabIndex).toBe(0);
    expect(button(modal.contentEl, "Proposed").tabIndex).toBe(-1);
    expect(document.activeElement).toBe(confirmed);

    confirmed.dispatchEvent(new KeyboardEvent("keydown", { key: "Home", bubbles: true }));
    proposed = button(modal.contentEl, "Proposed");
    expect(proposed.getAttribute("aria-selected")).toBe("true");
    expect(document.activeElement).toBe(proposed);
    proposed.dispatchEvent(new KeyboardEvent("keydown", { key: "End", bubbles: true }));
    confirmed = button(modal.contentEl, "Confirmed");
    expect(confirmed.getAttribute("aria-selected")).toBe("true");
    confirmed.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));
    expect(button(modal.contentEl, "Proposed").getAttribute("aria-selected")).toBe("true");
  });

  it("shows empty and corrupt documents independently and disables generation without current authorization", () => {
    const { mock } = controller(snapshot({
      catalog: null, proposal: null, profile: null,
      authorization: { kind: "authorization-required", rootLabel: "papers" } as any,
      proposalLoadError: { kind: "proposal", code: "invalid", message: "broken proposal" },
      profileLoadError: { kind: "profile", code: "invalid", message: "broken profile" },
      eligibility: { documentDiagnostics: ["profile-invalid", "catalog-invalid"], eligibleDirections: [], diagnostics: [] },
    }));
    const modal = open(mock);
    expect(modal.contentEl.textContent).toContain("broken proposal");
    const generate = button(modal.contentEl, "Generate proposals");
    expect(generate.disabled).toBe(true);
    expect(generate.title).toContain("Authorize");
    button(modal.contentEl, "Confirmed").click();
    expect(modal.contentEl.textContent).toContain("broken profile");
  });

  it("confirms regeneration and normalizes reviewed fields before saving", async () => {
    const { mock } = controller();
    const modal = open(mock);
    button(modal.contentEl, "Regenerate proposals").click();
    await confirmChoice("Regenerate");
    await vi.waitFor(() => expect(mock.generate).toHaveBeenCalledOnce());
    await vi.waitFor(() => expect(button(modal.contentEl, "Refresh").disabled).toBe(false));

    const inputs = modal.contentEl.querySelectorAll<HTMLInputElement>('input[type="text"]');
    inputs[0]!.value = "  Renamed  ";
    const areas = modal.contentEl.querySelectorAll<HTMLTextAreaElement>("textarea");
    areas[0]!.value = "  Description  ";
    areas[1]!.value = " zeta\n alpha \nzeta\n\n";
    button(modal.contentEl, "Save edits").click();
    await vi.waitFor(() => expect(mock.updateProposal).toHaveBeenCalledWith({
      candidateId: "candidate-1",
      patch: { name: "Renamed", description: "Description", discoveryCues: ["alpha", "zeta"] },
      representativePaperKeys: ["arxiv:2608.00001"],
    }));
    expect(normalizeLines(" b\na\nb ")).toEqual(["a", "b"]);
  });

  it("routes proposal confirmation, discard, confirmed edit/toggle/remove, and exposes merge actions", async () => {
    const secondCandidate = { ...candidate, id: "candidate-2", name: "Second", lineage: { candidateIds: ["candidate-2"] } };
    const secondDirection = { ...direction, id: "direction-2", name: "Second confirmed" };
    const state = snapshot({
      proposal: { ...snapshot().proposal!, candidates: [candidate, secondCandidate] },
      profile: { ...snapshot().profile!, directions: [direction, secondDirection] },
      eligibility: { ...snapshot().eligibility, diagnostics: [
        { directionId: "direction-1", eligible: true, reasons: [] },
        { directionId: "direction-2", eligible: false, reasons: [{ reason: "representative-evidence-changed", paperKey: "arxiv:2608.00001" }] },
      ] },
    });
    const { mock } = controller(state);
    const modal = open(mock);
    button(modal.contentEl, "Confirm active").click();
    await confirmChoice("Confirm active");
    await vi.waitFor(() => expect(mock.confirmProposal).toHaveBeenCalledWith(expect.objectContaining({ candidateId: "candidate-1", status: "active" })));
    button(modal.contentEl, "Discard").click();
    await confirmChoice("Discard");
    await vi.waitFor(() => expect(mock.discardProposal).toHaveBeenCalledWith("candidate-1"));

    button(modal.contentEl, "Confirmed").click();
    expect(modal.contentEl.textContent).toContain("representative-evidence-changed");
    button(modal.contentEl, "Save edits").click();
    await vi.waitFor(() => expect(mock.updateConfirmed).toHaveBeenCalled());
    button(modal.contentEl, "Disable").click();
    await confirmChoice("Disable");
    await vi.waitFor(() => expect(mock.disable).toHaveBeenCalledWith("direction-1"));
    button(modal.contentEl, "Remove").click();
    await confirmChoice("Remove");
    await vi.waitFor(() => expect(mock.remove).toHaveBeenCalledWith({ directionId: "direction-1", mode: "restrict" }));
    button(modal.contentEl, "Cascade remove merge family").click();
    await confirmChoice("Cascade remove family");
    await vi.waitFor(() => expect(mock.remove).toHaveBeenCalledWith({ directionId: "direction-1", mode: "cascade" }));
    expect(button(modal.contentEl, "Merge selected confirmed directions").disabled).toBe(true);
  });

  it("merges two proposals and two terminal confirmed directions with reviewed fields and explicit status", async () => {
    const secondCandidate = { ...candidate, id: "candidate-2", name: "Second", lineage: { candidateIds: ["candidate-2"] } };
    const secondDirection = { ...direction, id: "direction-2", name: "Second confirmed" };
    const { mock } = controller(snapshot({
      proposal: { ...snapshot().proposal!, candidates: [candidate, secondCandidate] },
      profile: { ...snapshot().profile!, directions: [direction, secondDirection] },
    }));
    const modal = open(mock);
    let checkbox = modal.contentEl.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event("change"));
    checkbox = Array.from(modal.contentEl.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')).find((item) => !item.checked)!;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event("change"));
    expect(button(modal.contentEl, "Merge selected proposals").disabled).toBe(false);
    button(modal.contentEl, "Merge selected proposals").click();
    await vi.waitFor(() => expect(Modal.opened.at(-1)!.contentEl.textContent).toContain("Merge 2 selected proposed directions (Agents, Second)"));
    button(Modal.opened.at(-1)!.contentEl, "Cancel").click();
    await Promise.resolve();
    expect(mock.mergeProposals).not.toHaveBeenCalled();
    button(modal.contentEl, "Merge selected proposals").click();
    await confirmChoice("Merge proposals");
    await vi.waitFor(() => expect(mock.mergeProposals).toHaveBeenCalledWith(expect.objectContaining({
      sourceCandidateIds: ["candidate-1", "candidate-2"],
    })));

    await vi.waitFor(() => expect(button(modal.contentEl, "Refresh").disabled).toBe(false));
    button(modal.contentEl, "Confirmed").click();
    checkbox = modal.contentEl.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event("change"));
    checkbox = Array.from(modal.contentEl.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')).find((item) => !item.checked)!;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event("change"));
    button(modal.contentEl, "Merge selected confirmed directions").click();
    await confirmChoice("Merge as disabled");
    await vi.waitFor(() => expect(mock.mergeConfirmed).toHaveBeenCalledWith(expect.objectContaining({
      sourceDirectionIds: ["direction-1", "direction-2"], status: "disabled",
    })));
  });

  it("disables actions while pending, exposes operation failures, refreshes, and ignores late completion after close", async () => {
    let release!: () => void;
    const { mock } = controller();
    vi.mocked(mock.reload).mockImplementationOnce(() => new Promise((resolve) => { release = () => resolve(snapshot()); }));
    const modal = open(mock);
    button(modal.contentEl, "Refresh").click();
    expect(button(modal.contentEl, "Refresh").disabled).toBe(true);
    modal.close();
    release();
    await Promise.resolve();
    expect(modal.contentEl.childElementCount).toBe(0);

    const failed = controller();
    const hostile = "/Users/alice/Secret Library/profile.json sha256:deadbeef model-key";
    vi.mocked(failed.mock.enable).mockRejectedValueOnce(new Error(hostile));
    const disabledDirection = { ...direction, status: "disabled" as const };
    failed.set(snapshot({ profile: { ...snapshot().profile!, directions: [disabledDirection] } }));
    const failedModal = open(failed.mock);
    button(failedModal.contentEl, "Confirmed").click();
    button(failedModal.contentEl, "Enable").click();
    await confirmChoice("Enable");
    await vi.waitFor(() => expect(failedModal.contentEl.querySelector('[role="alert"]')?.textContent).toBe("Operation failed. Refresh and try again."));
    expect(failedModal.contentEl.textContent).not.toContain(hostile);
    expect(failedModal.contentEl.textContent).not.toContain("/Users/alice");
    expect(failedModal.contentEl.textContent).not.toContain("sha256:deadbeef");

    vi.mocked(failed.mock.enable).mockRejectedValueOnce(Object.assign(new Error(hostile), { code: "evidence-mismatch" }));
    button(failedModal.contentEl, "Enable").click();
    await confirmChoice("Enable");
    await vi.waitFor(() => expect(failedModal.contentEl.querySelector('[role="alert"]')?.textContent).toContain("evidence is missing or stale"));
    expect(failedModal.contentEl.textContent).not.toContain(hostile);
  });
});
