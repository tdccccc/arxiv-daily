import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { Modal, type App } from "obsidian";
import {
  PersonalLibraryInterestProfileModal,
  bufferPoolHeading,
  describeClusterMembers,
  formatConfidence,
  incrementalSuggestionKey,
  incrementalSuggestionPaperCount,
  normalizeLines,
  timelineEventLabel,
  truncateReason,
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
    suggestions: null,
    eligibility: {
      documentDiagnostics: [], eligibleDirections: [direction],
      diagnostics: [{ directionId: direction.id, eligible: true, reasons: [] }],
    },
    authorization: { kind: "authorized", rootLabel: "papers", processingDepth: "metadata-and-abstracts", endpoint: "https://example.test" } as any,
    catalogLoadError: null, proposalLoadError: null, profileLoadError: null, suggestionsLoadError: null,
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
    applySuggestion: update, dismissSuggestion: update, lock: update, unlock: update,
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

describe("incremental suggestions in the review modal", () => {
  const longReason = "An attached suggestion whose reason exceeds the display budget and must be truncated at the boundary to keep the card compact. ".repeat(4);

  function suggestionsDoc() {
    return {
      schemaVersion: 1, revision: 1, scopeFingerprint: fingerprint, identificationFingerprint: fingerprint,
      updatedAt: "2026-08-03T00:00:00.000Z",
      suggestions: [
        { kind: "attach", directionId: "direction-1", paperKeys: ["arxiv:2608.00001"], reason: "Newly indexed paper matches this confirmed direction." },
        { kind: "new", paperKeys: ["arxiv:2608.00005"], reason: "A new evaluation theme beyond the current directions." },
        { kind: "split", directionId: "direction-1", paperKeys: ["arxiv:2608.00006"], reason: longReason },
        { kind: "merge", directionIds: ["direction-1", "direction-2"], reason: "Merge overlapping confirmed directions." },
      ],
    } as any;
  }

  it("shows an empty state when no incremental suggestions exist", () => {
    const { mock } = controller();
    const modal = open(mock);
    expect(modal.contentEl.textContent).toContain("No incremental suggestions.");
    expect(modal.contentEl.querySelector(".arxiv-daily-interest-review__suggestions")).toBeNull();
  });

  it("renders kind badges, targets, paper counts, and truncated reasons as text", () => {
    const { mock } = controller(snapshot({ suggestions: suggestionsDoc() }));
    const modal = open(mock);
    const section = modal.contentEl.querySelector(".arxiv-daily-interest-review__suggestions");
    expect(section).not.toBeNull();
    expect(section!.querySelector("strong")?.textContent).toBe("Incremental suggestions 4");
    const cards = section!.querySelectorAll(".arxiv-daily-interest-review__suggestion");
    expect(cards).toHaveLength(4);
    expect(Array.from(section!.querySelectorAll(".arxiv-daily-interest-review__suggestion-kind"))
      .map((item) => item.textContent)).toEqual(["attach", "new", "split", "merge"]);
    expect(section!.textContent).toContain("Confirmed agents"); // attach/split target name
    expect(section!.textContent).toContain("1 paper(s)");
    expect(section!.textContent).toContain("New direction");
    expect(section!.textContent).toContain("2 directions");
    expect(section!.textContent).toContain("Newly indexed paper matches this confirmed direction.");
    const splitReason = section!.querySelectorAll(".arxiv-daily-interest-review__suggestion-reason")[2]?.textContent;
    expect(splitReason).toBe(truncateReason(longReason));
    expect(splitReason?.endsWith("…")).toBe(true);
    expect(button(modal.contentEl, "Apply")).toBeDefined();
    expect(button(modal.contentEl, "Convert to proposal")).toBeDefined();
    expect(modal.contentEl.querySelectorAll("button").length).toBeGreaterThanOrEqual(
      Array.from(modal.contentEl.querySelectorAll("button")).filter((item) => item.textContent === "Ignore").length + 2,
    );
    expect(section!.querySelector("img")).toBeNull();
  });

  it("routes Apply and Ignore through controller suggestion keys", async () => {
    const { mock } = controller(snapshot({ suggestions: suggestionsDoc() }));
    const modal = open(mock);
    button(modal.contentEl, "Apply").click();
    await vi.waitFor(() => expect(mock.applySuggestion).toHaveBeenCalledWith("attach:direction-1:arxiv:2608.00001"));
    button(modal.contentEl, "Ignore").click();
    await vi.waitFor(() => expect(mock.dismissSuggestion).toHaveBeenCalledWith("attach:direction-1:arxiv:2608.00001"));
    expect(mock.dismissSuggestion).toHaveBeenCalledOnce();
  });

  it("converts a new suggestion into a review candidate and switches to the proposed tab", async () => {
    const { mock } = controller(snapshot({ suggestions: suggestionsDoc() }));
    const modal = open(mock);
    button(modal.contentEl, "Convert to proposal").click();
    await vi.waitFor(() => expect(mock.applySuggestion).toHaveBeenCalledWith("new::arxiv:2608.00005"));
    await vi.waitFor(() => expect(modal.contentEl.querySelector("#arxiv-daily-interest-proposed-tab")?.getAttribute("aria-selected")).toBe("true"));
  });

  it("formats suggestion keys, paper counts, and truncation deterministically", () => {
    expect(incrementalSuggestionKey({ kind: "attach", directionId: "d-1", paperKeys: ["arxiv:2608.00001"], reason: "r" }))
      .toBe("attach:d-1:arxiv:2608.00001");
    expect(incrementalSuggestionKey({ kind: "new", paperKeys: ["arxiv:2608.00001"], reason: "r" }))
      .toBe("new::arxiv:2608.00001");
    expect(incrementalSuggestionKey({ kind: "split", directionId: "d-1", paperKeys: ["arxiv:2608.00001"], reason: "r" }))
      .toBe("split:d-1:arxiv:2608.00001");
    expect(incrementalSuggestionKey({ kind: "merge", directionIds: ["d-1", "d-2"], reason: "r" }))
      .toBe("merge:d-1:d-2");
    expect(incrementalSuggestionPaperCount({ kind: "attach", directionId: "d-1", paperKeys: ["a", "b"], reason: "r" })).toBe("2 paper(s)");
    expect(incrementalSuggestionPaperCount({ kind: "merge", directionIds: ["d-1", "d-2"], reason: "r" })).toBe("2 directions");
    expect(truncateReason("short")).toBe("short");
    expect(truncateReason("x".repeat(200))).toBe(`${"x".repeat(160)}…`);
  });

  it("shows the lock state and routes lock/unlock on confirmed direction cards", async () => {
    const locked = { ...direction, lockedAt: "2026-08-04T00:00:00.000Z" };
    const { mock, set } = controller(snapshot({
      profile: { ...snapshot().profile!, directions: [locked] },
    }));
    const modal = open(mock);
    button(modal.contentEl, "Confirmed").click();
    expect(modal.contentEl.textContent).toContain("locked");
    button(modal.contentEl, "Unlock").click();
    await vi.waitFor(() => expect(mock.unlock).toHaveBeenCalledWith("direction-1"));

    set(snapshot({ profile: { ...snapshot().profile!, directions: [direction] } }));
    button(modal.contentEl, "Confirmed").click();
    button(modal.contentEl, "Lock").click();
    await vi.waitFor(() => expect(mock.lock).toHaveBeenCalledWith("direction-1"));
  });
});

describe("cluster products in the personal library interest review", () => {
  const input = [
    { paperKey: "arxiv:2608.00001", evidenceFingerprint: evidence },
    { paperKey: "arxiv:2608.00002", evidenceFingerprint: evidence },
    { paperKey: "arxiv:2608.00003", evidenceFingerprint: evidence },
  ];

  function paper(paperKey: string, title: string): any {
    return {
      paperKey, source: "arxiv", externalId: paperKey.slice("arxiv:".length), title,
      authors: ["A"], abstract: "Abstract", published: "2026-08-01T00:00:00.000Z",
      updated: "2026-08-01T00:00:00.000Z", primaryCategory: "cs.AI", categories: ["cs.AI"],
      evidenceDepth: "metadata-and-abstract", filePaths: ["paper.pdf"],
    };
  }

  function catalogWith(extra: Record<string, any> = {}): any {
    const base = snapshot().catalog!;
    return { ...base, papers: { ...base.papers, ...extra } };
  }

  it("shows cluster member count, average confidence, and per-member confidence for proposed candidates", () => {
    const clustered = {
      ...candidate,
      clusterMembers: [
        { paperKey: "arxiv:2608.00001", confidence: 0.9 },
        { paperKey: "arxiv:2608.00002", confidence: 0.8 },
        { paperKey: "arxiv:2608.00003", confidence: 0.5 },
      ],
    };
    const state = snapshot({
      catalog: catalogWith({ "arxiv:2608.00002": paper("arxiv:2608.00002", "Second cluster paper") }),
      proposal: { ...snapshot().proposal!, catalogInputPapers: input, candidates: [clustered] },
    });
    const { mock } = controller(state);
    const modal = open(mock);
    const cluster = modal.contentEl.querySelector(".arxiv-daily-interest-review__cluster");
    expect(cluster).not.toBeNull();
    expect(cluster!.querySelector("summary")?.textContent).toBe("Cluster members 3 · avg. confidence 73%");
    expect(Array.from(cluster!.querySelectorAll("li")).map((item) => item.textContent)).toEqual([
      "Paper <img src=x> — 90%",
      "Second cluster paper — 80%",
      "arxiv:2608.00003 — 50%",
    ]);
    expect(cluster!.querySelector("img")).toBeNull();
  });

  it("derives the buffer pool as catalog input papers minus the union of all cluster members", () => {
    const first = {
      ...candidate,
      clusterMembers: [
        { paperKey: "arxiv:2608.00001", confidence: 0.9 },
        { paperKey: "arxiv:2608.00002", confidence: 0.7 },
      ],
    };
    const second = {
      ...candidate, id: "candidate-2", name: "Second",
      lineage: { candidateIds: ["candidate-2"] },
      clusterMembers: [{ paperKey: "arxiv:2608.00002", confidence: 0.6 }],
    };
    const state = snapshot({
      catalog: catalogWith({
        "arxiv:2608.00003": paper("arxiv:2608.00003", "Buffered third paper"),
        "arxiv:2608.00004": paper("arxiv:2608.00004", "Buffered fourth paper"),
      }),
      proposal: {
        ...snapshot().proposal!,
        catalogInputPapers: [
          { paperKey: "arxiv:2608.00001", evidenceFingerprint: evidence },
          { paperKey: "arxiv:2608.00002", evidenceFingerprint: evidence },
          { paperKey: "arxiv:2608.00003", evidenceFingerprint: evidence },
          { paperKey: "arxiv:2608.00004", evidenceFingerprint: evidence },
        ],
        candidates: [first, second],
      },
    });
    expect(unclassifiedBufferPoolPapers(state.proposal)).toEqual([
      { paperKey: "arxiv:2608.00003", evidenceFingerprint: evidence },
      { paperKey: "arxiv:2608.00004", evidenceFingerprint: evidence },
    ]);
    const { mock } = controller(state);
    const modal = open(mock);
    const buffer = modal.contentEl.querySelector(".arxiv-daily-interest-review__buffer");
    expect(buffer).not.toBeNull();
    expect(buffer!.querySelector("strong")?.textContent).toBe("Unclustered (buffer pool) 2");
    expect(Array.from(buffer!.querySelectorAll("li")).map((item) => item.textContent)).toEqual([
      "Buffered third paper — arxiv:2608.00003",
      "Buffered fourth paper — arxiv:2608.00004",
    ]);
    expect(buffer!.textContent).toContain("这些论文未进入任何方向草案");
  });

  it("omits the buffer pool section when every catalog input paper is covered", () => {
    const covered = { ...candidate, clusterMembers: [{ paperKey: "arxiv:2608.00001", confidence: 1 }] };
    const { mock } = controller(snapshot({
      proposal: { ...snapshot().proposal!, candidates: [covered] },
    }));
    const modal = open(mock);
    expect(modal.contentEl.querySelector(".arxiv-daily-interest-review__buffer")).toBeNull();
    expect(modal.contentEl.textContent).not.toContain("Unclustered (buffer pool)");
  });

  it("treats legacy candidates without cluster members as covering nothing and renders no cluster block", () => {
    const state = snapshot({
      catalog: catalogWith({ "arxiv:2608.00002": paper("arxiv:2608.00002", "Second paper") }),
      proposal: { ...snapshot().proposal!, catalogInputPapers: input, candidates: [{ ...candidate }] },
    });
    const { mock } = controller(state);
    const modal = open(mock);
    expect(modal.contentEl.querySelector(".arxiv-daily-interest-review__cluster")).toBeNull();
    const buffer = modal.contentEl.querySelector(".arxiv-daily-interest-review__buffer");
    expect(buffer?.querySelector("strong")?.textContent).toBe("Unclustered (buffer pool) 3");
    expect(Array.from(buffer!.querySelectorAll("li")).map((item) => item.textContent)).toEqual([
      "Paper <img src=x> — arxiv:2608.00001",
      "Second paper — arxiv:2608.00002",
      "arxiv:2608.00003 — missing from current catalog",
    ]);
  });

  it("shows confirmed cluster member summaries and the most recent five timeline events with mapped labels", () => {
    const confirmed = {
      ...direction,
      clusterMembers: [
        { paperKey: "arxiv:2608.00001", confidence: 0.95 },
        { paperKey: "arxiv:2608.00002", confidence: 0.85 },
      ],
      timeline: [
        { kind: "created", at: "2026-08-01T00:00:00.000Z" },
        { kind: "edited", at: "2026-08-02T00:00:00.000Z" },
        { kind: "members-updated", at: "2026-08-03T00:00:00.000Z" },
        { kind: "merged", at: "2026-08-04T00:00:00.000Z", sourceDirectionIds: ["direction-0", "direction-9"] },
        { kind: "removed", at: "2026-08-05T00:00:00.000Z", mode: "restrict" },
        { kind: "edited", at: "2026-08-06T00:00:00.000Z" },
      ],
    };
    const state = snapshot({
      catalog: catalogWith({ "arxiv:2608.00002": paper("arxiv:2608.00002", "Second confirmed paper") }),
      profile: { ...snapshot().profile!, directions: [confirmed] },
    });
    const { mock } = controller(state);
    const modal = open(mock);
    button(modal.contentEl, "Confirmed").click();
    const cluster = modal.contentEl.querySelector(".arxiv-daily-interest-review__cluster");
    expect(cluster?.querySelector("summary")?.textContent).toBe("Cluster members 2 · avg. confidence 90%");
    expect(Array.from(cluster!.querySelectorAll("li")).map((item) => item.textContent)).toEqual([
      "Paper <img src=x> — 95%",
      "Second confirmed paper — 85%",
    ]);
    expect(Array.from(modal.contentEl.querySelectorAll(".arxiv-daily-interest-review__timeline li"))
      .map((item) => item.textContent)).toEqual([
      "2026-08-06 00:00 — Edited",
      "2026-08-05 00:00 — Removed",
      "2026-08-04 00:00 — Merged",
      "2026-08-03 00:00 — Members updated",
      "2026-08-02 00:00 — Edited",
    ]);
    expect(modal.contentEl.textContent).not.toContain("Created");
  });

  it("omits the cluster block for confirmed directions with no cluster members but keeps the timeline", () => {
    const emptyCluster = {
      ...direction,
      clusterMembers: [],
      timeline: [{ kind: "created", at: "2026-08-03T00:00:00.000Z" }],
    };
    const { mock } = controller(snapshot({
      profile: { ...snapshot().profile!, directions: [emptyCluster] },
    }));
    const modal = open(mock);
    button(modal.contentEl, "Confirmed").click();
    expect(modal.contentEl.querySelector(".arxiv-daily-interest-review__cluster")).toBeNull();
    expect(Array.from(modal.contentEl.querySelectorAll(".arxiv-daily-interest-review__timeline li"))
      .map((item) => item.textContent)).toEqual(["2026-08-03 00:00 — Created"]);
  });

  it("formats confidence, cluster summaries, buffer headings, and timeline labels deterministically", () => {
    expect(formatConfidence(0.7333)).toBe("73%");
    expect(formatConfidence(1)).toBe("100%");
    expect(formatConfidence(0.005)).toBe("1%");
    expect(describeClusterMembers([])).toBeNull();
    expect(describeClusterMembers([
      { paperKey: "arxiv:2608.00001", confidence: 0.9 },
      { paperKey: "arxiv:2608.00002", confidence: 0.8 },
    ])).toBe("Cluster members 2 · avg. confidence 85%");
    expect(bufferPoolHeading(2)).toBe("Unclustered (buffer pool) 2");
    expect(timelineEventLabel("created")).toBe("Created");
    expect(timelineEventLabel("edited")).toBe("Edited");
    expect(timelineEventLabel("members-updated")).toBe("Members updated");
    expect(timelineEventLabel("merged")).toBe("Merged");
    expect(timelineEventLabel("removed")).toBe("Removed");
    expect(unclassifiedBufferPoolPapers(null)).toEqual([]);
  });
});
