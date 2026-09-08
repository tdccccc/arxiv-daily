import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { Modal, type App } from "obsidian";
import {
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
  acceptProposedTopics,
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryCatalogInputManifestFingerprint,
  createPersonalLibraryRepresentativeSetFingerprint,
  normalizeTopic,
  updatePersonalLibraryDirectionCandidate,
  type PersonalLibraryDirectionCandidate,
  type ProposalAcceptanceReceipt,
  type Topic,
} from "@arxiv-daily/core";
import {
  PersonalLibraryInterestProfileModal,
  unclassifiedBufferPoolPapers,
  type InterestProfileReviewController,
  type InterestProfileReviewSnapshot,
} from "../src/library/interest-profile-modal";

// Only the Obsidian DOM extensions are supplied here; the review modal and
// acceptance calculation stay real. Fixtures use complete catalog/snapshot DTOs.
beforeAll(() => {
  type Options = { cls?: string; text?: string; type?: string; value?: string; attr?: Record<string, string> };
  Object.assign(HTMLElement.prototype, {
    addClass(this: HTMLElement, ...classes: string[]) { this.classList.add(...classes); },
    empty(this: HTMLElement) { this.replaceChildren(); },
    createEl(this: HTMLElement, tag: string, options: Options = {}) {
      const element = document.createElement(tag);
      if (options.cls) element.className = options.cls;
      if (options.text !== undefined) element.textContent = options.text;
      if (options.type) element.setAttribute("type", options.type);
      if (options.value !== undefined) (element as HTMLInputElement).value = options.value;
      for (const [key, value] of Object.entries(options.attr ?? {})) element.setAttribute(key, value);
      this.appendChild(element);
      return element;
    },
    createDiv(this: HTMLElement, options: Options = {}) { return this.createEl("div", options); },
    createSpan(this: HTMLElement, options: Options = {}) { return this.createEl("span", options); },
  });
});

beforeEach(() => {
  Modal.opened.length = 0;
  document.body.replaceChildren();
});

const fingerprint = `sha256:${"a".repeat(64)}`;
const evidenceFingerprint = `sha256:${"b".repeat(64)}`;
const at = "2026-09-07T00:00:00.000Z";
const paper = (number: number) => ({ paperKey: `arxiv:2608.0000${number}`, evidenceFingerprint });
const first: PersonalLibraryDirectionCandidate = direction("candidate-1", "Reliable research agents", [1, 2]);
const second: PersonalLibraryDirectionCandidate = direction("candidate-2", "Research systems evaluation", [3, 4]);

function direction(id: string, text: string, numbers: number[]): PersonalLibraryDirectionCandidate {
  const representatives = numbers.map(paper);
  return {
    id, text, discoveryCues: ["agent evidence", "system evaluation"], representatives,
    representativeSetFingerprint: createPersonalLibraryRepresentativeSetFingerprint(representatives),
    lineage: { candidateIds: [id] },
    clusterMembers: representatives.map(({ paperKey }) => ({ paperKey, confidence: 0.9 })),
  };
}

type ReviewSnapshot = InterestProfileReviewSnapshot & {
  settingsTopics: Topic[];
  proposalAcceptance: ProposalAcceptanceReceipt | null;
  acceptanceLoadError: string | null;
};

function snapshot(overrides: Partial<ReviewSnapshot> = {}): ReviewSnapshot {
  const catalogInputPapers = [1, 2, 3, 4].map(paper);
  return {
    catalog: createEmptyPersonalLibraryCatalog(fingerprint, fingerprint, new Date(at)),
    indexedPapers: catalogInputPapers.map(({ paperKey }) => ({ paperKey, title: `Paper ${paperKey}` })),
    proposal: {
      schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
      revision: 0, proposalId: "proposal-1", scopeFingerprint: fingerprint,
      identificationFingerprint: fingerprint,
      catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
        scopeFingerprint: fingerprint, identificationFingerprint: fingerprint, catalogInputPapers,
      }),
      catalogInputPapers, generationContractFingerprint: fingerprint, generatedAt: at,
      topics: [{ id: "proposed-topic", suggestedName: "Research agents", directions: [first, second] }],
    },
    suggestions: null,
    authorization: { kind: "authorized", rootLabel: "papers", grantedAt: at },
    catalogLoadError: null, proposalLoadError: null, suggestionsLoadError: null,
    settingsTopicNames: [], settingsTopics: [], proposalAcceptance: null, acceptanceLoadError: null,
    ...overrides,
  };
}

function settingsTopic(id = "existing-topic", name = "Research agents", directions: Topic["directions"] = []) {
  return normalizeTopic({ id, name, tag: id, detail: false, directions });
}

function receipt(processedCandidateIds = [first.id]): ProposalAcceptanceReceipt {
  return {
    proposalId: "proposal-1", scopeFingerprint: fingerprint,
    topicTargets: { "proposed-topic": "existing-topic" }, processedCandidateIds,
  };
}

function controller(initial = snapshot()) {
  let current = initial;
  const port: InterestProfileReviewController = {
    snapshot: () => current,
    reload: vi.fn(async () => current),
    authorize: vi.fn(async () => true),
    generate: vi.fn(async () => undefined),
    logError: vi.fn(),
    updateProposal: vi.fn(async () => current),
    discardProposal: vi.fn(async () => current),
    renameTopic: vi.fn(async () => current),
    moveDirection: vi.fn(async () => current),
    acceptTopics: vi.fn(async (topicIds, candidateIds) => {
      const proposal = current.proposal!;
      const accepted = acceptProposedTopics({
        proposalId: proposal.proposalId, scopeFingerprint: proposal.scopeFingerprint,
        existingTopics: current.settingsTopics, acceptance: current.proposalAcceptance,
        topics: proposal.topics.filter(({ id }) => topicIds.includes(id)).map((topic) => ({
          ...topic,
          directions: topic.directions.filter(({ id }) => !candidateIds || candidateIds.includes(id)),
        })),
      });
      current = {
        ...current, settingsTopics: accepted.topics,
        settingsTopicNames: accepted.topics.map(({ name }) => name), proposalAcceptance: accepted.acceptance,
      };
      return current;
    }),
  };
  return { port, snapshot: () => current, set(next: ReviewSnapshot) { current = next; return next; } };
}

function open(port: InterestProfileReviewController) {
  const modal = new PersonalLibraryInterestProfileModal({} as App, port);
  modal.modalEl.appendChild(modal.contentEl);
  document.body.appendChild(modal.modalEl);
  modal.open();
  return modal;
}

function button(root: HTMLElement, text: string): HTMLButtonElement {
  const found = Array.from(root.querySelectorAll("button")).find((item) => item.textContent === text);
  expect(found, `button ${text}`).toBeDefined();
  return found!;
}

function choice(root: HTMLElement, text: string): HTMLInputElement {
  const found = Array.from(root.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'))
    .find((item) => item.getAttribute("aria-label") === `Select ${text}`);
  expect(found, `direction ${text}`).toBeDefined();
  return found!;
}

function card(root: HTMLElement, text: string): HTMLElement {
  return choice(root, text).closest<HTMLElement>("article")!;
}

function destination(root: HTMLElement, text: string): HTMLSelectElement {
  const select = card(root, text).querySelector<HTMLSelectElement>(`select[aria-label="Destination for ${text}"]`);
  expect(select, "Each pending direction offers a destination").not.toBeNull();
  return select!;
}

async function ready(root: HTMLElement): Promise<void> {
  await vi.waitFor(() => expect(button(root, "Refresh").disabled).toBe(false));
}

describe("unsaved review drafts", () => {
  function editableController() {
    const ctrl = controller();
    ctrl.port.updateProposal = async ({ candidateId, patch }) => ctrl.set({
      ...ctrl.snapshot(),
      proposal: updatePersonalLibraryDirectionCandidate({
        proposal: ctrl.snapshot().proposal, candidateId, patch,
      }),
    });
    return ctrl;
  }

  it("preserves direction text and representative selection across checkbox changes and refresh", async () => {
    const ctrl = editableController();
    const root = open(ctrl.port).contentEl;
    const row = card(root, first.text);
    row.querySelector<HTMLTextAreaElement>("textarea")!.value = "My corrected research scope";
    const representatives = row.querySelector<HTMLSelectElement>("select[multiple]")!;
    representatives.options[0]!.selected = false;
    choice(root, second.text).click();
    button(root, "Refresh").click();
    await ready(root);
    const restored = card(root, first.text);
    expect(restored.querySelector<HTMLTextAreaElement>("textarea")!.value).toBe("My corrected research scope");
    expect(Array.from(restored.querySelector<HTMLSelectElement>("select[multiple]")!.selectedOptions, ({ value }) => value))
      .toEqual(["arxiv:2608.00002"]);
  });

  it("saves the reviewed text before accepting selected directions", async () => {
    const ctrl = editableController();
    const root = open(ctrl.port).contentEl;
    card(root, first.text).querySelector<HTMLTextAreaElement>("textarea")!.value = "My corrected research scope";
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    expect(ctrl.snapshot().settingsTopics[0]!.directions.map(({ text }) => text))
      .toEqual(["My corrected research scope", second.text]);
  });

  it("retains a failed save for retry and does not accept the stale proposal", async () => {
    const ctrl = editableController();
    const save = ctrl.port.updateProposal;
    ctrl.port.updateProposal = async () => { throw new Error("disk unavailable"); };
    const root = open(ctrl.port).contentEl;
    card(root, first.text).querySelector<HTMLTextAreaElement>("textarea")!.value = "My corrected research scope";
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    expect(ctrl.snapshot().settingsTopics).toEqual([]);
    expect(card(root, first.text).querySelector<HTMLTextAreaElement>("textarea")!.value).toBe("My corrected research scope");
    ctrl.port.updateProposal = save;
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    expect(ctrl.snapshot().settingsTopics[0]!.directions[0]!.text).toBe("My corrected research scope");
  });

  it("does not carry a draft into a replacement proposal reusing candidate IDs", async () => {
    const ctrl = editableController();
    const root = open(ctrl.port).contentEl;
    card(root, first.text).querySelector<HTMLTextAreaElement>("textarea")!.value = "Old proposal draft";
    ctrl.set({ ...ctrl.snapshot(), proposal: { ...ctrl.snapshot().proposal!, proposalId: "replacement" } });
    button(root, "Refresh").click();
    await ready(root);
    expect(card(root, first.text).querySelector<HTMLTextAreaElement>("textarea")!.value).toBe(first.text);
  });

  it("does not accept into another library when the library changes while a draft is saving", async () => {
    const ctrl = editableController();
    ctrl.port.updateProposal = async () => ctrl.set({
      ...ctrl.snapshot(), proposal: { ...ctrl.snapshot().proposal!, scopeFingerprint: `sha256:${"c".repeat(64)}` },
    });
    const root = open(ctrl.port).contentEl;
    card(root, first.text).querySelector<HTMLTextAreaElement>("textarea")!.value = "My corrected scope";
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    expect(ctrl.snapshot().settingsTopics).toEqual([]);
    expect(root.querySelector('[role="alert"]')?.textContent).toMatch(/review changed.*refresh/i);
  });

  it("keeps other direction drafts after saving one direction", async () => {
    const ctrl = editableController();
    const root = open(ctrl.port).contentEl;
    card(root, first.text).querySelector<HTMLTextAreaElement>("textarea")!.value = "First edited scope";
    card(root, second.text).querySelector<HTMLTextAreaElement>("textarea")!.value = "Second edited scope";
    button(card(root, first.text), "Save edits").click();
    await ready(root);
    expect(card(root, second.text).querySelector<HTMLTextAreaElement>("textarea")!.value).toBe("Second edited scope");
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    expect(ctrl.snapshot().settingsTopics[0]!.directions.map(({ text }) => text))
      .toEqual(["First edited scope", "Second edited scope"]);
  });

  it("retains drafts when a failed refresh temporarily makes the proposal unavailable", async () => {
    const ctrl = editableController();
    const original = ctrl.snapshot();
    const root = open(ctrl.port).contentEl;
    card(root, first.text).querySelector<HTMLTextAreaElement>("textarea")!.value = "Recoverable draft";
    ctrl.set({ ...original, proposal: null, proposalLoadError: { kind: "proposal", message: "temporary read failure", code: "unreadable" } });
    button(root, "Refresh").click();
    await ready(root);
    ctrl.set(original);
    button(root, "Refresh").click();
    await ready(root);
    expect(card(root, first.text).querySelector<HTMLTextAreaElement>("textarea")!.value).toBe("Recoverable draft");
  });
});

describe("reviewing a proposal with acceptance history", () => {
  it("reopens partial acceptance with the remaining direction selected and accepts it", async () => {
    const ctrl = controller();
    const modal = open(ctrl.port);
    choice(modal.contentEl, second.text).click();
    button(modal.contentEl, "Accept 1 topic(s) into settings").click();
    await ready(modal.contentEl);
    expect(ctrl.snapshot().settingsTopics[0]!.directions.map(({ text }) => text)).toEqual([first.text]);
    modal.close();

    const root = open(ctrl.port).contentEl;
    expect(choice(root, first.text).checked).toBe(false);
    expect(choice(root, first.text).disabled).toBe(true);
    expect(choice(root, second.text).checked).toBe(true);
    expect(choice(root, second.text).disabled).toBe(false);
    expect(root.querySelector(".arxiv-daily-interest-review__topic-heading")?.textContent).not.toContain("Added");
    expect(button(root, "Accept 1 topic(s) into settings").disabled).toBe(false);
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    expect(ctrl.snapshot().settingsTopics[0]!.directions.map(({ text }) => text)).toEqual([first.text, second.text]);
    expect(root.querySelector(".arxiv-daily-interest-review__topic-heading")?.textContent).toContain("Added");
    expect(button(root, "Accept 0 topic(s) into settings").disabled).toBe(true);
  });

  it("keeps the partial topic selected so its remaining direction can be accepted without reopening", async () => {
    const ctrl = controller();
    const root = open(ctrl.port).contentEl;
    choice(root, second.text).click();
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    choice(root, second.text).click();
    expect(button(root, "Accept 1 topic(s) into settings").disabled).toBe(false);
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    expect(ctrl.snapshot().settingsTopics[0]!.directions.map(({ id }) => id)).toEqual([first.id, second.id]);
  });

  it.each(["edited", "deleted"])("keeps an accepted direction read-only after it is %s in settings", (change) => {
    const existing = settingsTopic("existing-topic", "Renamed research", change === "edited"
      ? [{ id: first.id, text: "My corrected direction", origin: "library" }] : []);
    const ctrl = controller(snapshot({
      settingsTopics: [existing], settingsTopicNames: [existing.name], proposalAcceptance: receipt(),
    }));
    const root = open(ctrl.port).contentEl;
    const accepted = card(root, first.text);
    expect(choice(root, first.text).disabled).toBe(true);
    expect(choice(root, first.text).checked).toBe(false);
    expect(accepted.textContent).toContain("Added");
    expect(accepted.querySelectorAll("textarea, select, button")).toHaveLength(0);
    expect(root.textContent).toContain("Add to existing topic: Renamed research");
    expect(choice(root, second.text).disabled).toBe(false);
  });

  it.each(["proposalId", "scopeFingerprint"] as const)("ignores acceptance history from a different %s", (field) => {
    const existing = settingsTopic();
    const ctrl = controller(snapshot({
      settingsTopics: [existing], settingsTopicNames: [existing.name],
      proposalAcceptance: {
        ...receipt([first.id, second.id]),
        [field]: field === "scopeFingerprint" ? `sha256:${"c".repeat(64)}` : "another-proposal",
      },
    }));
    const root = open(ctrl.port).contentEl;
    expect(choice(root, first.text).disabled).toBe(false);
    expect(choice(root, first.text).checked).toBe(true);
    expect(button(root, "Accept 1 topic(s) into settings").disabled).toBe(false);
  });

  it("shows a same-name manual topic as the destination and permits new direction text", async () => {
    const existing = settingsTopic("existing-topic", "  research AGENTS  ", [
      { id: "manual", text: "My original research direction", origin: "manual" },
    ]);
    const ctrl = controller(snapshot({ settingsTopics: [existing], settingsTopicNames: [existing.name] }));
    const root = open(ctrl.port).contentEl;
    expect(root.querySelector(".arxiv-daily-interest-review__topic-heading")?.textContent).not.toContain("Added");
    expect(root.querySelector(".arxiv-daily-interest-review__topic-heading")?.textContent).toContain("Add to existing topic");
    expect(root.textContent).toContain(`Add to existing topic: ${existing.name}`);
    expect(button(root, "Accept 1 topic(s) into settings").disabled).toBe(false);
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    expect(ctrl.snapshot().settingsTopics).toHaveLength(1);
    expect(ctrl.snapshot().settingsTopics[0]!.directions.map(({ text }) => text))
      .toEqual(["My original research direction", first.text, second.text]);
  });

  it.each(["proposal", "receipt"])("blocks a removed destination recorded in the %s", (source) => {
    const initial = snapshot({ proposalAcceptance: source === "receipt" ? receipt() : null });
    if (source === "proposal") initial.proposal!.topics[0] = { ...initial.proposal!.topics[0]!, targetTopicId: "deleted-topic" };
    const ctrl = controller(initial);
    const root = open(ctrl.port).contentEl;
    expect(root.textContent).toContain("destination topic was removed");
    expect(root.textContent).toContain("Choose a destination");
    expect(button(root, "Accept 1 topic(s) into settings").disabled).toBe(true);
    button(root, "Accept 1 topic(s) into settings").click();
    expect(ctrl.port.acceptTopics).not.toHaveBeenCalled();
    expect(destination(root, second.text).disabled).toBe(false);
  });

  it("blocks acceptance when its persisted receipt cannot be loaded, while offering regeneration", () => {
    const ctrl = controller(snapshot({ acceptanceLoadError: "The saved acceptance record is invalid." }));
    const root = open(ctrl.port).contentEl;
    expect(root.textContent).toContain("acceptance record");
    expect(root.textContent).toMatch(/restore.*saved review state.*reload/iu);
    expect(button(root, "Accept 1 topic(s) into settings").disabled).toBe(true);
    expect(button(root, "Regenerate proposals").disabled).toBe(false);
    button(root, "Accept 1 topic(s) into settings").click();
    expect(ctrl.port.acceptTopics).not.toHaveBeenCalled();
  });
});

describe("changing a pending direction's destination", () => {
  it("unblocks a removed destination after moving the pending direction to a chosen existing topic", async () => {
    const initial = snapshot({
      proposalAcceptance: receipt(), settingsTopics: [settingsTopic("other-topic", "Evaluation")],
      settingsTopicNames: ["Evaluation"],
    });
    const ctrl = controller(initial);
    vi.mocked(ctrl.port.moveDirection!).mockImplementation(async () => ctrl.set({
      ...initial,
      proposal: {
        ...initial.proposal!, revision: 1,
        topics: [
          { id: "proposed-topic", suggestedName: "Research agents", directions: [first] },
          { id: "moved-topic", suggestedName: "Old evaluation label", targetTopicId: "other-topic", directions: [second] },
        ],
      },
    }));
    const root = open(ctrl.port).contentEl;
    expect(button(root, "Accept 1 topic(s) into settings").disabled).toBe(true);
    const select = destination(root, second.text);
    select.value = "other-topic";
    select.dispatchEvent(new Event("change"));
    button(card(root, second.text), "Move direction").click();
    await ready(root);
    expect(ctrl.port.moveDirection).toHaveBeenCalledExactlyOnceWith({ candidateId: second.id, targetTopicId: "other-topic" });
    const heading = choice(root, second.text).closest(".arxiv-daily-interest-review__topic")!
      .querySelector(".arxiv-daily-interest-review__topic-heading");
    expect(heading?.textContent).toContain("Evaluation");
    expect(heading?.textContent).toContain("Add to existing topic");
    expect(choice(root, second.text).checked).toBe(true);
    expect(button(root, "Accept 1 topic(s) into settings").disabled).toBe(false);
    button(root, "Accept 1 topic(s) into settings").click();
    await ready(root);
    expect(ctrl.snapshot().settingsTopics[0]!.directions.map(({ id }) => id)).toEqual([second.id]);
    expect(ctrl.snapshot().settingsTopics).toHaveLength(1);
  });

  it("passes only the chosen candidate and existing topic id to the controller", async () => {
    const ctrl = controller(snapshot({ settingsTopics: [settingsTopic("other-topic", "Evaluation")], settingsTopicNames: ["Evaluation"] }));
    const root = open(ctrl.port).contentEl;
    const select = destination(root, second.text);
    expect(Array.from(select.options, ({ textContent }) => textContent)).toContain("Evaluation");
    select.value = "other-topic";
    select.dispatchEvent(new Event("change"));
    button(card(root, second.text), "Move direction").click();
    await ready(root);
    expect(ctrl.port.moveDirection).toHaveBeenCalledExactlyOnceWith({ candidateId: second.id, targetTopicId: "other-topic" });
    expect(ctrl.port.renameTopic).not.toHaveBeenCalled();
  });

  it("requires a distinct nonempty name before explicitly moving a direction to a new topic", async () => {
    const initial = snapshot({ settingsTopics: [settingsTopic()], settingsTopicNames: ["Research agents"] });
    initial.proposal!.topics[0] = { ...initial.proposal!.topics[0]!, targetTopicId: "existing-topic" };
    const ctrl = controller(initial);
    const root = open(ctrl.port).contentEl;
    const select = destination(root, second.text);
    const newOption = Array.from(select.options).find(({ textContent }) => textContent === "New topic");
    expect(newOption).toBeDefined();
    select.value = newOption!.value;
    select.dispatchEvent(new Event("change"));
    const name = card(root, second.text).querySelector<HTMLInputElement>('input[aria-label="New topic name"]')!;
    expect(name).not.toBeNull();
    expect(name.closest<HTMLElement>("label")!.hidden).toBe(false);
    expect(button(card(root, second.text), "Move direction").disabled).toBe(true);
    name.value = " RESEARCH AGENTS ";
    name.dispatchEvent(new Event("input"));
    expect(button(card(root, second.text), "Move direction").disabled).toBe(true);
    name.value = " Evaluation systems ";
    name.dispatchEvent(new Event("input"));
    expect(button(card(root, second.text), "Move direction").disabled).toBe(false);
    button(card(root, second.text), "Move direction").click();
    await ready(root);
    expect(ctrl.port.moveDirection).toHaveBeenCalledExactlyOnceWith({
      candidateId: second.id, targetTopicId: null, suggestedName: "Evaluation systems",
    });
  });
});

describe("proposal evidence is explanatory", () => {

  it("renders cues read-only while preserving them when direction text is saved", async () => {
    const ctrl = controller();
    const root = open(ctrl.port).contentEl;
    const row = card(root, first.text);
    expect(row.querySelectorAll("textarea")).toHaveLength(1);
    expect(row.textContent).toContain("Evidence hints");
    expect(row.textContent).toContain("Only the direction text drives matching");
    expect(row.textContent).toContain("agent evidence");
    row.querySelector<HTMLTextAreaElement>("textarea")!.value = "A reviewed direction";
    button(row, "Save edits").click();
    await ready(root);
    expect(ctrl.port.updateProposal).toHaveBeenCalledWith({
      candidateId: first.id,
      patch: { text: "A reviewed direction", discoveryCues: ["agent evidence", "system evaluation"] },
      representativePaperKeys: ["arxiv:2608.00001", "arxiv:2608.00002"],
    });
  });

  it("shows fully covered library evidence as success with no new directions", () => {
    const initial = snapshot({ settingsTopics: [settingsTopic("existing", "Research", [
      { id: "followed", text: "Research systems", origin: "manual" },
    ])] });
    initial.proposal = {
      ...initial.proposal!, topics: [], coveredPaperKeys: [1, 2, 3, 4].map((number) => paper(number).paperKey),
    };
    const root = open(controller(initial).port).contentEl;
    expect(root.textContent).toMatch(/Already covered.*4 papers.*no new directions/iu);
    expect(root.textContent).not.toContain("This proposal contains no directions");
    expect(root.querySelector(".arxiv-daily-interest-review__buffer")).toBeNull();
    expect(root.querySelector(".arxiv-daily-interest-review__accept-bar")).toBeNull();
    expect(button(root, "Regenerate proposals").disabled).toBe(false);
  });

  it("excludes existing coverage from the count of papers without a direction", () => {
    const initial = snapshot();
    const catalogInputPapers = [1, 2, 3, 4, 5, 6].map(paper);
    initial.proposal = {
      ...initial.proposal!,
      catalogInputPapers, coveredPaperKeys: [paper(5).paperKey],
      catalogInputFingerprint: createPersonalLibraryCatalogInputManifestFingerprint({
        scopeFingerprint: fingerprint, identificationFingerprint: fingerprint, catalogInputPapers,
      }),
    };
    const root = open(controller(initial).port).contentEl;
    expect(unclassifiedBufferPoolPapers(initial.proposal).map(({ paperKey }) => paperKey))
      .toEqual(["arxiv:2608.00006"]);
    expect(root.querySelector(".arxiv-daily-interest-review__buffer")?.textContent).toContain("1 paper");
    const options = card(root, first.text).querySelector<HTMLSelectElement>("select[multiple]")!.options;
    expect(Array.from(options, ({ value }) => value)).not.toContain("arxiv:2608.00005");
  });
});
