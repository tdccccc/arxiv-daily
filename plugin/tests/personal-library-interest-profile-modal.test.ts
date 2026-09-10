import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Modal, Notice, type App } from "obsidian";
import { DEFAULT_SETTINGS, normalizeTopic, PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION } from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";
import { ArxivDailySettingTab } from "../src/settings/tab";
import { SettingsChangeService } from "../src/settings/change-service";
import { LibraryIndexStatusStore } from "../src/library/index-status";
import {
  PersonalLibraryInterestProfileModal,
  bufferPoolHeading,
  describeClusterMembers,
  formatConfidence,
  normalizeLines,
  safeUserError,
  unclassifiedBufferPoolPapers,
  type InterestProfileReviewController,
  type InterestProfileReviewSnapshot,
} from "../src/library/interest-profile-modal";

beforeAll(() => {
  type Options = { cls?: string; text?: string; type?: string; value?: string; attr?: Record<string, string> };
  const proto = HTMLElement.prototype as any;
  proto.addClass ??= function (...classes: string[]) { this.classList.add(...classes); };
  proto.removeClass ??= function (...classes: string[]) { this.classList.remove(...classes); };
  proto.toggleClass ??= function (name: string, value: boolean) { this.classList.toggle(name, value); };
  proto.empty ??= function () { this.replaceChildren(); };
  proto.detach ??= function () { this.remove(); };
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

beforeEach(() => {
  Modal.opened.length = 0;
  Notice.calls.length = 0;
  document.body.replaceChildren();
});

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
      schemaVersion: PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION, revision: 0, proposalId: "proposal-1", scopeFingerprint: fingerprint,
      identificationFingerprint: fingerprint, catalogInputFingerprint: fingerprint,
      catalogInputPapers: candidate.representatives, generationContractFingerprint: fingerprint,
      generatedAt: "2026-08-03T00:00:00.000Z",
      topics: [{ id: "topic-1", suggestedName: "Research agents", directions: [candidate] }],
    },
    suggestions: null,
    indexedPapers: [],
    authorization: { kind: "authorized", rootLabel: "papers", processingDepth: "metadata-and-abstracts", endpoint: "https://example.test" } as any,
    catalogLoadError: null, proposalLoadError: null, suggestionsLoadError: null,
    settingsTopicNames: [],
    ...overrides,
  };
}

function topicSnapshot(): InterestProfileReviewSnapshot {
  const base = snapshot();
  const paperKey = (number: number): string => `arxiv:2608.${String(number).padStart(5, "0")}`;
  const papers = Array.from({ length: 8 }, (_, index) => ({ paperKey: paperKey(index + 1), evidenceFingerprint: evidence }));
  const topic = (id: string, suggestedName: string, groups: number[][]) => ({
    id,
    suggestedName,
    directions: groups.map((members, index) => ({
      ...candidate,
      id: `${id}-direction-${index + 1}`,
      text: `${suggestedName} direction ${index + 1}`,
      representatives: Array.from(new Set(members), (number) => ({ paperKey: paperKey(number), evidenceFingerprint: evidence })),
      lineage: { candidateIds: [`${id}-direction-${index + 1}`] },
      clusterMembers: members.map((number) => ({
        paperKey: paperKey(number),
        confidence: 1,
      })),
    })),
  });
  return snapshot({
    proposal: {
      ...base.proposal!,
      catalogInputPapers: papers,
      topics: [
        topic("small", "Small coverage", [[1], [1]]),
        topic("tie-z", "First tied coverage", [[2, 3]]),
        topic("tie-a", "Second tied coverage", [[4, 5]]),
        topic("large", "Largest coverage", [[6, 7], [7, 8]]),
      ],
    },
    indexedPapers: papers.map(({ paperKey }) => ({ paperKey, title: `Paper ${paperKey}` })),
  });
}

function controller(initial = snapshot(), options: { granted?: boolean } = {}) {
  let current = initial;
  const update = vi.fn(async () => current);
  const mock: InterestProfileReviewController = {
    snapshot: () => current,
    reload: vi.fn(async () => current), generate: vi.fn(async () => undefined),
    authorize: vi.fn(async () => options.granted ?? true),
    logError: vi.fn(),
    updateProposal: update, discardProposal: update,
    renameTopic: vi.fn(async () => current),
    acceptTopics: vi.fn(async () => current),
  };
  return { mock, set: (next: InterestProfileReviewSnapshot) => { current = next; } };
}

function open(ctrl: InterestProfileReviewController) {
  const modal = new PersonalLibraryInterestProfileModal({} as App, ctrl);
  modal.modalEl.appendChild(modal.contentEl);
  document.body.appendChild(modal.modalEl);
  modal.open();
  return modal;
}

function openPluginReview(
  initial = topicSnapshot(),
  existingTopics: ReturnType<typeof normalizeTopic>[] = [],
) {
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.arxiv.topics = existingTopics;
  const saved: typeof settings[] = [];
  const envelopes: { settings: typeof settings; libraryProposalAcceptances?: unknown }[] = [];
  const saveData = vi.fn(async (data: { settings: typeof settings }) => {
    saved.push(structuredClone(data.settings));
    envelopes.push(structuredClone(data));
  });
  Object.assign(plugin, {
    app: {} as App,
    settings,
    saveData,
    logger: { error: vi.fn() },
    libraryCatalog: initial.catalog,
    libraryProposal: initial.proposal,
    libraryIndexedPapers: initial.indexedPapers,
    librarySuggestions: initial.suggestions,
    libraryCatalogLoadError: null,
    libraryProposalLoadError: null,
    librarySuggestionsLoadError: null,
  });
  wireSettingsTransaction(plugin);
  plugin.openPersonalLibraryDirectionReview();
  const modal = Modal.opened.at(-1)!;
  modal.modalEl.appendChild(modal.contentEl);
  document.body.appendChild(modal.modalEl);
  return { plugin, root: modal.contentEl, saveData, saved, envelopes };
}

function wireSettingsTransaction(plugin: ArxivDailyPlugin): void {
  Object.assign(plugin, { settingsChanges: new SettingsChangeService({
    settings: plugin.settings,
    persistSettings: (candidate) => (plugin as any).enqueueLibraryMutation(() => (plugin as any).persistSettings(candidate)),
  }) });
}

/** A real, already-rendered settings page behind the review modal. */
function openSettingsBehindReview(plugin: ArxivDailyPlugin): ArxivDailySettingTab {
  Object.assign(plugin, {
    stateStore: { snapshot: () => ({}) },
    manifest: { id: "arxiv-daily", version: "0.0.0-test" },
    libraryIndexStatus: new LibraryIndexStatusStore(),
  });
  plugin.settings.arxiv.topics = [normalizeTopic({
    id: "existing-topic", name: "Existing topic", tag: "existing", detail: false,
    directions: [{ id: "existing-direction", text: "An existing research direction", origin: "manual" }],
  })];
  const tab = new ArxivDailySettingTab(plugin.app, plugin);
  // The Obsidian framework renderer is a no-op in the test host. Exercise the
  // actual legacy renderer here, including the actual topic cards and drafts.
  vi.spyOn(tab, "getSettingDefinitions").mockReturnValue([]);
  Object.assign(plugin, { settingsTab: tab });
  document.body.appendChild(tab.containerEl);
  tab.display();
  return tab;
}

function button(root: HTMLElement, text: string): HTMLButtonElement {
  const found = Array.from(root.querySelectorAll("button")).find((item) => item.textContent === text);
  if (!found) throw new Error(`missing button ${text}`);
  return found;
}

function topicChoices(root: HTMLElement): HTMLInputElement[] {
  return Array.from(root.querySelectorAll<HTMLInputElement>('input[aria-label^="Accept "]'));
}

function directionChoice(root: HTMLElement, text: string): HTMLInputElement {
  const choice = Array.from(root.querySelectorAll<HTMLInputElement>('input[aria-label^="Select "]'))
    .find((input) => input.getAttribute("aria-label") === `Select ${text}`);
  expect(choice).toBeDefined();
  return choice!;
}

function expandTopic(root: HTMLElement, name: string): HTMLDetailsElement {
  const section = topicChoices(root)
    .find((choice) => choice.getAttribute("aria-label") === `Accept ${name}`)
    ?.closest<HTMLDetailsElement>("details.arxiv-daily-interest-review__topic");
  expect(section).not.toBeNull();
  expect(section).toBeDefined();
  section!.open = true;
  section!.dispatchEvent(new Event("toggle"));
  return section!;
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

    // This fixture has one representative, so including its direction requires
    // an explicit choice even though the topic itself starts selected.
    expect(button(root, "Add to research topics").disabled).toBe(true);
    directionChoice(root, candidate.text).click();
    expect(button(root, "Add to research topics").disabled).toBe(false);

    const checkbox = root.querySelector<HTMLInputElement>('input[aria-label="Accept Research agents"]')!;
    checkbox.checked = false;
    checkbox.dispatchEvent(new Event("change"));
    expect(button(root, "Add to research topics").disabled).toBe(true);

    const reselected = root.querySelector<HTMLInputElement>('input[aria-label="Accept Research agents"]')!;
    reselected.checked = true;
    reselected.dispatchEvent(new Event("change"));

    const armed = button((modal as any).contentEl, "Add to research topics");
    expect(armed.disabled).toBe(false);
    armed.dispatchEvent(new Event("click"));
    await Promise.resolve();
    await Promise.resolve();
    expect(ctrl.mock.acceptTopics).toHaveBeenCalledWith(["topic-1"], ["candidate-1"]);
  });

  it("orders topics by distinct covered papers and keeps tied topics in proposal order", () => {
    const initial = topicSnapshot();
    const ctrl = controller(initial);
    const root = open(ctrl.mock).contentEl;
    expect(topicChoices(root).map((choice) => choice.getAttribute("aria-label"))).toEqual([
      "Accept Largest coverage",
      "Accept First tied coverage",
      "Accept Second tied coverage",
      "Accept Small coverage",
    ]);
    expect(initial.proposal!.topics.map(({ id }) => id)).toEqual(["small", "tie-z", "tie-a", "large"]);
  });

  it("preselects only the two widest topics and labels the others optional", async () => {
    const ctrl = controller(topicSnapshot());
    const root = open(ctrl.mock).contentEl;
    expect(topicChoices(root).map((choice) => choice.checked)).toEqual([true, true, false, false]);
    const sections = Array.from(root.querySelectorAll(".arxiv-daily-interest-review__topic"));
    expect(sections.map((section) => section.querySelector("summary")?.textContent?.includes("Optional")))
      .toEqual([false, false, true, true]);
    expect(sections[2]!.querySelector("summary")?.textContent).toContain("2 papers");
    expect(sections[3]!.querySelector("summary")?.textContent).toContain("1 paper");
    button(root, "Add to research topics").click();
    await vi.waitFor(() => expect(ctrl.mock.acceptTopics).toHaveBeenCalledWith(
      ["large", "tie-z"], ["large-direction-1", "large-direction-2", "tie-z-direction-1"],
    ));
  });

  it("starts every topic collapsed and summarizes its distinct papers and directions", () => {
    const root = open(controller(topicSnapshot()).mock).contentEl;
    const topics = Array.from(root.querySelectorAll<HTMLDetailsElement>("details.arxiv-daily-interest-review__topic"));
    expect(topics).toHaveLength(4);
    expect(topics.every((topic) => !topic.open)).toBe(true);
    expect(topics.map((topic) => topic.querySelector(".arxiv-daily-interest-review__topic-count")?.textContent))
      .toEqual(["3 papers · 2 directions", "2 papers · 1 direction", "2 papers · 1 direction", "1 paper · 2 directions"]);
  });

  it("preserves reviewed choices across a revision and reseeds only for a different proposal", async () => {
    const initial = topicSnapshot();
    const ctrl = controller(initial);
    const root = open(ctrl.mock).contentEl;
    const largest = topicChoices(root)[0]!;
    largest.checked = false;
    largest.dispatchEvent(new Event("change"));
    const optional = topicChoices(root)[2]!;
    optional.checked = true;
    optional.dispatchEvent(new Event("change"));

    ctrl.set({ ...initial, proposal: { ...initial.proposal!, revision: 1 } });
    button(root, "Refresh").click();
    await vi.waitFor(() => expect(button(root, "Refresh").disabled).toBe(false));
    expect(topicChoices(root).map((choice) => choice.checked)).toEqual([false, true, true, false]);

    ctrl.set({ ...initial, proposal: { ...initial.proposal!, proposalId: "proposal-2" } });
    button(root, "Refresh").click();
    await vi.waitFor(() => expect(button(root, "Refresh").disabled).toBe(false));
    expect(topicChoices(root).map((choice) => choice.checked)).toEqual([true, true, false, false]);
  });

  it("keeps an expanded topic open when its selection or name changes", async () => {
    const initial = topicSnapshot();
    const ctrl = controller(initial);
    vi.mocked(ctrl.mock.renameTopic).mockImplementation(async ({ topicId, suggestedName }) => {
      const next = {
        ...initial,
        proposal: {
          ...initial.proposal!,
          revision: 1,
          topics: initial.proposal!.topics.map((topic) => topic.id === topicId ? { ...topic, suggestedName } : topic),
        },
      };
      ctrl.set(next);
      return next;
    });
    const root = open(ctrl.mock).contentEl;
    const section = expandTopic(root, "Largest coverage");
    const select = section.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    select.click();
    const expanded = root.querySelector<HTMLDetailsElement>("details.arxiv-daily-interest-review__topic")!;
    expect(expanded.open).toBe(true);
    const name = expanded.querySelector<HTMLInputElement>('input[aria-label="Topic name"]')!;
    name.value = "Agent reliability";
    name.dispatchEvent(new Event("change"));
    await vi.waitFor(() => expect(button(root, "Refresh").disabled).toBe(false));
    expect(ctrl.mock.renameTopic).toHaveBeenCalledWith({ topicId: "large", suggestedName: "Agent reliability" });
    expect(root.querySelector<HTMLDetailsElement>("details.arxiv-daily-interest-review__topic")!.open).toBe(true);
    expect(topicChoices(root)[0]!.checked).toBe(false);
    expect(topicChoices(root)[0]!.getAttribute("aria-label")).toBe("Accept Agent reliability");
  });

  it("saves a direction edited inside an expanded topic and reports controller failures", async () => {
    const ctrl = controller();
    const root = open(ctrl.mock).contentEl;
    const section = expandTopic(root, "Research agents");
    const editor = section.querySelector<HTMLDetailsElement>(".arxiv-daily-interest-review__detail")!;
    editor.open = true;
    const fields = Array.from(editor.querySelectorAll("textarea"));
    fields[0]!.value = "Reliable research agents";
    button(editor, "Save edits").click();
    await vi.waitFor(() => expect(ctrl.mock.updateProposal).toHaveBeenCalledWith({
      candidateId: "candidate-1",
      patch: { text: "Reliable research agents", discoveryCues: ["agents"] },
      representativePaperKeys: ["arxiv:2608.00001"],
    }));
    await vi.waitFor(() => expect(button(root, "Refresh").disabled).toBe(false));

    const failure = Object.assign(new Error("stale evidence"), { code: "evidence-mismatch" });
    vi.mocked(ctrl.mock.updateProposal).mockRejectedValueOnce(failure);
    const reopened = expandTopic(root, "Research agents");
    button(reopened, "Save edits").click();
    await vi.waitFor(() => expect(ctrl.mock.logError).toHaveBeenCalledWith("save proposed direction", failure));
    expect(root.querySelector('[role="alert"]')?.textContent).toContain("Representative evidence is missing or stale");
  });

  /**
   * A library of PDFs the scan could not give an arXiv identity has an empty
   * catalog and a full index. Those papers are proposable on the title and
   * abstract the index read, so the button must not be gated on the catalog.
   */
  it("offers generation for a library whose papers only the index can name", () => {
    const ctrl = controller(snapshot({
      catalog: { ...(snapshot().catalog as any), papers: {} },
      proposal: null,
      indexedPapers: [{ paperKey: `file:sha256:${"a".repeat(64)}`, title: "A Local Paper" }],
    }));
    const root = (open(ctrl.mock) as any).contentEl as HTMLElement;
    expect(button(root, "Generate topics").disabled).toBe(false);
  });

  it("refuses generation when neither the catalog nor the index has papers", () => {
    const ctrl = controller(snapshot({
      catalog: { ...(snapshot().catalog as any), papers: {} },
      proposal: null,
      indexedPapers: [],
    }));
    const root = (open(ctrl.mock) as any).contentEl as HTMLElement;
    const generate = button(root, "Generate topics");
    expect(generate.disabled).toBe(true);
    expect(generate.getAttribute("title")).toContain("no indexed metadata-and-abstract papers");
  });

  it("names a fallback-indexed representative from the index instead of calling it missing", () => {
    const paperKey = candidate.representatives[0]!.paperKey;
    const ctrl = controller(snapshot({
      catalog: { ...(snapshot().catalog as any), papers: {} },
      indexedPapers: [{ paperKey, title: "A Local Paper" }],
    }));
    const root = (open(ctrl.mock) as any).contentEl as HTMLElement;
    expect(root.textContent).toContain("A Local Paper");
    expect(root.textContent).not.toContain("missing from current library");
  });

  /**
   * With local embedding no other path in the plugin asks for this grant, so a
   * button disabled on it could never become clickable. Consent is asked on
   * the click instead.
   */
  it("offers generation while unauthorized and asks for consent on the click", async () => {
    const ctrl = controller(snapshot({
      proposal: null,
      authorization: { kind: "connected", rootLabel: "papers" } as any,
    }));
    const root = (open(ctrl.mock) as any).contentEl as HTMLElement;
    const generate = button(root, "Generate topics");
    expect(generate.disabled).toBe(false);
    expect(generate.getAttribute("title")).toContain("confirm what leaves this device");

    generate.dispatchEvent(new Event("click"));
    await vi.waitFor(() => expect(ctrl.mock.authorize).toHaveBeenCalled());
    await vi.waitFor(() => expect(ctrl.mock.generate).toHaveBeenCalled());
  });

  it("generates nothing when the disclosure is declined", async () => {
    const ctrl = controller(snapshot({
      proposal: null,
      authorization: { kind: "connected", rootLabel: "papers" } as any,
    }), { granted: false });
    const root = (open(ctrl.mock) as any).contentEl as HTMLElement;
    button(root, "Generate topics").dispatchEvent(new Event("click"));
    await vi.waitFor(() => expect(ctrl.mock.authorize).toHaveBeenCalled());
    await Promise.resolve();
    expect(ctrl.mock.generate).not.toHaveBeenCalled();
  });

  it("does not re-ask for consent once the grant is recorded", async () => {
    const ctrl = controller(snapshot({ proposal: null }));
    const root = (open(ctrl.mock) as any).contentEl as HTMLElement;
    button(root, "Generate topics").dispatchEvent(new Event("click"));
    await vi.waitFor(() => expect(ctrl.mock.generate).toHaveBeenCalled());
    expect(ctrl.mock.authorize).not.toHaveBeenCalled();
  });

  /**
   * The page can only show a message safe to put on screen, so a failure whose
   * reason reaches neither the log nor the researcher is undiagnosable. Every
   * failure here looked like "Operation failed" and logged nothing.
   */
  it("records the real reason when authorizing fails", async () => {
    const ctrl = controller(snapshot({
      proposal: null,
      authorization: { kind: "connected", rootLabel: "papers" } as any,
    }));
    const boom = Object.assign(new Error("authorization stored but reads back as connected"), {
      code: "authorization-not-recorded",
    });
    (ctrl.mock.authorize as any).mockRejectedValueOnce(boom);
    const root = (open(ctrl.mock) as any).contentEl as HTMLElement;
    button(root, "Generate topics").dispatchEvent(new Event("click"));

    await vi.waitFor(() => expect(ctrl.mock.logError).toHaveBeenCalledWith(
      "authorize personal library processing",
      boom,
    ));
    expect(ctrl.mock.generate).not.toHaveBeenCalled();
  });

  it("tells the researcher which authorization guard refused", () => {
    expect(safeUserError({ code: "authorization-terms-changed" }))
      .toContain("changed while the disclosure was open");
    expect(safeUserError({ code: "authorization-superseded" }))
      .toContain("library changed while authorizing");
    expect(safeUserError({ code: "authorization-not-recorded" }))
      .toContain("was not recorded");
    // Anything unrecognized still refuses to put a raw message on screen.
    expect(safeUserError(new Error("/home/someone/secret/path exploded")))
      .toBe("Operation failed. Refresh and try again.");
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

  /**
   * The researcher found the full title list noisy, so the uncovered-papers
   * section says only how many library papers no proposed direction covers
   * (ADR 0014 §1's "remain visible as uncovered evidence" — the count keeps
   * that fact visible without repeating titles shown elsewhere on the page).
   */
  it("says how many papers the proposal does not cover, without listing them", () => {
    const base = snapshot();
    const uncovered = { paperKey: "arxiv:2608.00002", evidenceFingerprint: evidence };
    const ctrl = controller(snapshot({
      proposal: { ...base.proposal!, catalogInputPapers: [...base.proposal!.catalogInputPapers, uncovered] },
    }));
    const modal = open(ctrl.mock);
    const root = (modal as any).contentEl as HTMLElement;
    expect(unclassifiedBufferPoolPapers(ctrl.mock.snapshot().proposal)).toHaveLength(2);
    const section = root.querySelector(".arxiv-daily-interest-review__buffer");
    expect(section?.textContent).toBe(bufferPoolHeading(2));
    expect(section?.querySelectorAll("li")).toHaveLength(0);
    expect(section?.textContent).not.toContain("Paper <img src=x>");
    // The next action stays visible independently of uncovered-paper counts.
    expect(root.textContent).toContain("rebuild the library index and regenerate proposals");
  });

  it("renders nothing when every paper is covered by a proposed direction", () => {
    const ctrl = controller(topicSnapshot());
    const root = (open(ctrl.mock) as any).contentEl as HTMLElement;
    expect(unclassifiedBufferPoolPapers(ctrl.mock.snapshot().proposal)).toHaveLength(0);
    expect(root.querySelector(".arxiv-daily-interest-review__buffer")).toBeNull();
  });
});

describe("proposed topics already in settings", () => {
  /**
   * A researcher who accepts the same proposal twice must not get a second
   * copy of a topic they already kept. The proposal is left in place so the
   * rest can be accepted later, so the review page has to tell "already
   * added" apart from "not yet reviewed" by name, using settings as the
   * source of truth.
   */
  it("renders an already-added topic as unselectable and excludes it from preselection", () => {
    const ctrl = controller(snapshot({ proposalAcceptance: { proposalId: "proposal-1", scopeFingerprint: fingerprint, topicTargets: {}, processedCandidateIds: ["candidate-1"] } }));
    const root = open(ctrl.mock).contentEl;
    const checkbox = root.querySelector<HTMLInputElement>('input[aria-label="Accept Research agents"]')!;
    expect(checkbox.checked).toBe(false);
    expect(checkbox.disabled).toBe(true);
    expect(checkbox.title).toContain("already processed");
    expect(root.querySelector(".arxiv-daily-interest-review__topic-heading")?.textContent).toContain("Added");
    expect(Array.from(root.querySelectorAll("button"), ({ textContent }) => textContent)).not.toContain("Add to research topics");
    expect(button(root, "Done").disabled).toBe(false);
  });

  it("keeps an accepted direction marked after the settings topic is renamed", () => {
    const ctrl = controller(snapshot({ settingsTopicNames: ["My renamed topic"], proposalAcceptance: { proposalId: "proposal-1", scopeFingerprint: fingerprint, topicTargets: {}, processedCandidateIds: ["candidate-1"] } }));
    const root = open(ctrl.mock).contentEl;
    expect(root.querySelector(".arxiv-daily-interest-review__topic-heading")?.textContent).toContain("Added");
  });

  it("skips an already-added topic when choosing the two widest topics to preselect", async () => {
    const initial = topicSnapshot();
    const ctrl = controller({ ...initial, proposalAcceptance: { proposalId: "proposal-1", scopeFingerprint: fingerprint, topicTargets: {}, processedCandidateIds: ["large-direction-1", "large-direction-2"] } });
    const root = open(ctrl.mock).contentEl;
    expect(topicChoices(root).map((choice) => choice.checked)).toEqual([false, true, true, false]);
    const sections = Array.from(root.querySelectorAll(".arxiv-daily-interest-review__topic"));
    expect(sections[0]!.querySelector("summary")?.textContent).toContain("Added");
    button(root, "Add to research topics").click();
    await vi.waitFor(() => expect(ctrl.mock.acceptTopics).toHaveBeenCalledWith(
      ["tie-z", "tie-a"], ["tie-z-direction-1", "tie-a-direction-1"],
    ));
  });

  it("allows new directions for a same-name topic already in real settings", async () => {
    const initial = topicSnapshot();
    const existing = [normalizeTopic({
      name: "Largest coverage", tag: "largest-coverage", detail: false,
      directions: [{ text: "An existing direction", origin: "manual" }],
    })];
    const { root, saved } = openPluginReview(initial, existing);
    const heading = Array.from(root.querySelectorAll<HTMLElement>(".arxiv-daily-interest-review__topic-heading"))
      .find((element) => element.textContent?.includes("Largest coverage"))!;
    expect(heading.textContent).not.toContain("Added");
    const checkbox = heading.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    expect(checkbox.checked).toBe(true);
    expect(checkbox.disabled).toBe(false);
    button(root, "Add to research topics").click();
    await vi.waitFor(() => expect(saved.length).toBeGreaterThan(0));
    expect(saved[0]!.arxiv.topics.map(({ name }) => name)).toEqual(["Largest coverage", "First tied coverage"]);
    expect(saved[0]!.arxiv.topics[0]!.directions).toHaveLength(3);
  });

  it("reports settings topic names in the profile snapshot", () => {
    const { plugin } = openPluginReview(topicSnapshot(), [normalizeTopic({
      name: "Existing topic", tag: "existing", detail: false,
      directions: [{ text: "A direction", origin: "manual" }],
    })]);
    expect(plugin.getPersonalLibraryProfileSnapshot().settingsTopicNames).toEqual(["Existing topic"]);
  });

  it("appends missing directions to a same-name topic and records the decision", async () => {
    const existing = [normalizeTopic({
      name: "Largest coverage", tag: "largest-coverage", detail: false,
      directions: [{ text: "An existing direction", origin: "manual" }],
    })];
    const { plugin, saveData } = openPluginReview(topicSnapshot(), existing);
    const result = await plugin.acceptPersonalLibraryProposedTopics(["large"]);
    expect(saveData).toHaveBeenCalledOnce();
    expect(result.settingsTopics![0]!.directions.map(({ text }) => text)).toEqual([
      "An existing direction", "Largest coverage direction 1", "Largest coverage direction 2",
    ]);
    expect(result.proposalAcceptance!.processedCandidateIds).toEqual(["large-direction-1", "large-direction-2"]);
  });

  it("reports added directions and new topics for a mixed selection", async () => {
    const existing = [normalizeTopic({
      name: "Largest coverage", tag: "largest-coverage", detail: false,
      directions: [{ text: "An existing direction", origin: "manual" }],
    })];
    const { plugin, saveData, saved } = openPluginReview(topicSnapshot(), existing);
    await plugin.acceptPersonalLibraryProposedTopics(["large", "tie-z"]);
    expect(saveData).toHaveBeenCalledOnce();
    expect(saved[0]!.arxiv.topics.map(({ name }) => name)).toEqual(["Largest coverage", "First tied coverage"]);
    expect(Notice.calls.map(({ message }) => message)).toContain(
      "Added 3 directions to research settings. Created 1 topic.",
    );
  });
});

describe("persisting the reviewed direction selection", () => {
  it("refreshes an already-open settings page after accepting topics", async () => {
    const { plugin, root, saveData, saved } = openPluginReview();
    const tab = openSettingsBehindReview(plugin);
    const names = () => Array.from(tab.containerEl.querySelectorAll(".arxiv-daily-settings__topic-title"),
      (element) => element.textContent);
    expect(names()).toEqual(["Existing topic"]);
    button(root, "Add to research topics").click();
    await vi.waitFor(() => expect(names()).toEqual(["Existing topic", "Largest coverage", "First tied coverage"]));
    expect(saveData).toHaveBeenCalledOnce();
    expect(saved[0]!.arxiv.topics.map(({ name }) => name))
      .toEqual(["Existing topic", "Largest coverage", "First tied coverage"]);
    expect(plugin.logger.error).not.toHaveBeenCalled();
    expect(names()).toEqual(["Existing topic", "Largest coverage", "First tied coverage"]);
    tab.hide();
  });

  it("reports successful acceptance only after the settings write finishes", async () => {
    const { plugin, saveData } = openPluginReview();
    let finish!: () => void;
    const writing = new Promise<void>((resolve) => { finish = resolve; });
    saveData.mockImplementationOnce(() => writing);
    const accepting = plugin.acceptPersonalLibraryProposedTopics(["large"]);
    expect(Notice.calls).toEqual([]);
    finish();
    await accepting;
    expect(Notice.calls.map(({ message }) => message)).toContain("Added 2 directions to research settings. Created 1 topic.");
  });

  it("does not refresh the settings page or report success after a failed write", async () => {
    const { plugin, saveData } = openPluginReview();
    const tab = openSettingsBehindReview(plugin);
    saveData.mockRejectedValueOnce(new Error("disk full"));
    await expect(plugin.acceptPersonalLibraryProposedTopics(["large"])).rejects.toThrow("disk full");
    expect(Array.from(tab.containerEl.querySelectorAll(".arxiv-daily-settings__topic-title"),
      (element) => element.textContent)).toEqual(["Existing topic"]);
    expect(Notice.calls).toEqual([]);
    tab.hide();
  });

  it("keeps a saved acceptance successful when refreshing the settings view fails", async () => {
    const { plugin, saved } = openPluginReview();
    const tab = openSettingsBehindReview(plugin);
    vi.spyOn(tab, "refreshSettings").mockImplementationOnce(() => { throw new Error("view unavailable"); });
    await plugin.acceptPersonalLibraryProposedTopics(["large"]);
    expect(saved[0]!.arxiv.topics.map(({ name }) => name)).toEqual(["Existing topic", "Largest coverage"]);
    expect(Notice.calls.map(({ message }) => message)).toContain(
      "Added 2 directions to research settings. Created 1 topic. Reopen settings to refresh the list.",
    );
    tab.hide();
  });

  it("persists only the kept direction from a selected topic through the review modal", async () => {
    const initial = topicSnapshot();
    const { plugin, root, saveData, saved } = openPluginReview(initial);
    topicChoices(root).find((choice) => choice.getAttribute("aria-label") === "Accept First tied coverage")!.click();
    expandTopic(root, "Largest coverage");
    directionChoice(root, "Largest coverage direction 2").click();
    button(root, "Add to research topics").click();

    await vi.waitFor(() => expect(saveData).toHaveBeenCalledOnce());
    expect(saved[0]!.arxiv.topics.map((topic) => ({ name: topic.name, directions: topic.directions.map(({ text }) => text) })))
      .toEqual([{ name: "Largest coverage", directions: ["Largest coverage direction 1"] }]);
    expect(plugin.settings.arxiv.topics).toEqual(saved[0]!.arxiv.topics);
    expect(plugin.getPersonalLibraryProfileSnapshot().proposal).toEqual(initial.proposal);
  });

  it("leaves a topic with no kept directions out while accepting another selected topic", async () => {
    const { root, saveData, saved } = openPluginReview();
    expandTopic(root, "Largest coverage");
    directionChoice(root, "Largest coverage direction 1").click();
    directionChoice(root, "Largest coverage direction 2").click();
    expect(topicChoices(root)[0]!.checked).toBe(true);
    button(root, "Add to research topics").click();

    await vi.waitFor(() => expect(saveData).toHaveBeenCalledOnce());
    expect(saved[0]!.arxiv.topics.map((topic) => ({ name: topic.name, directions: topic.directions.map(({ text }) => text) })))
      .toEqual([{ name: "First tied coverage", directions: ["First tied coverage direction 1"] }]);
  });

  it("disables acceptance with an explanation when selected topics have no kept directions", () => {
    const { root, saveData, plugin } = openPluginReview();
    directionChoice(root, "Largest coverage direction 1").click();
    directionChoice(root, "Largest coverage direction 2").click();
    directionChoice(root, "First tied coverage direction 1").click();
    const accept = button(root, "Add to research topics");
    expect(accept.disabled).toBe(true);
    expect(root.querySelector(".arxiv-daily-interest-review__accept-bar")?.textContent)
      .toMatch(/Select.*direction/i);
    accept.click();
    expect(saveData).not.toHaveBeenCalled();
    expect(plugin.settings.arxiv.topics).toEqual([]);
  });

  it("does not persist a thin-evidence direction until it is explicitly selected", async () => {
    const { root, saveData, saved } = openPluginReview(snapshot());
    expect(topicChoices(root)[0]!.checked).toBe(true);
    expect(directionChoice(root, candidate.text).checked).toBe(false);
    expect(button(root, "Add to research topics").disabled).toBe(true);
    expect(saveData).not.toHaveBeenCalled();

    expandTopic(root, "Research agents");
    directionChoice(root, candidate.text).click();
    button(root, "Add to research topics").click();
    await vi.waitFor(() => expect(saveData).toHaveBeenCalledOnce());
    expect(saved[0]!.arxiv.topics[0]!.directions.map(({ text }) => text)).toEqual([candidate.text]);
  });

  it("keeps the topic-only host call compatible with accepting all its directions", async () => {
    const { plugin, saved } = openPluginReview();
    await plugin.acceptPersonalLibraryProposedTopics(["large"]);
    expect(saved[0]!.arxiv.topics[0]!.directions.map(({ text }) => text))
      .toEqual(["Largest coverage direction 1", "Largest coverage direction 2"]);
  });

  it("ignores candidates from unselected topics and skips selected topics emptied by filtering", async () => {
    const { plugin, saved } = openPluginReview();
    await plugin.acceptPersonalLibraryProposedTopics(
      ["large", "tie-z"], ["large-direction-1", "tie-a-direction-1"],
    );
    expect(saved[0]!.arxiv.topics.map((topic) => ({ name: topic.name, directions: topic.directions.map(({ text }) => text) })))
      .toEqual([{ name: "Largest coverage", directions: ["Largest coverage direction 1"] }]);
  });

  it("rejects an explicitly empty host selection before changing or persisting settings", async () => {
    const { plugin, saveData } = openPluginReview();
    await expect(plugin.acceptPersonalLibraryProposedTopics(["large"], []))
      .rejects.toThrow("Select at least one proposed direction to accept");
    expect(saveData).not.toHaveBeenCalled();
    expect(plugin.settings.arxiv.topics).toEqual([]);
  });
});

describe("generation progress in the review modal", () => {
  it.each(["success", "failure"])("names the organization phase and clears progress after %s", async (outcome) => {
    const ctrl = controller(snapshot({ proposal: null }));
    let report!: NonNullable<Parameters<InterestProfileReviewController["generate"]>[0]>;
    let release!: () => void;
    const work = new Promise<void>((resolve) => { release = resolve; });
    const failure = Object.assign(new Error("invalid organization"), { code: "proposal-invariant" });
    vi.mocked(ctrl.mock.generate).mockImplementation(async (onProgress) => {
      report = onProgress!;
      await work;
      if (outcome === "failure") throw failure;
    });
    const root = open(ctrl.mock).contentEl;
    button(root, "Generate topics").click();
    expect(ctrl.mock.generate).toHaveBeenCalledOnce();
    report({ phase: "reading", completed: 0, total: 0 });
    expect(button(root, "Reading the index…").disabled).toBe(true);
    report({ phase: "grouping", completed: 0, total: 0 });
    expect(button(root, "Grouping papers…").disabled).toBe(true);
    report({ phase: "organization", completed: 0, total: 1 });
    expect(button(root, "Organizing topics and directions…").disabled).toBe(true);
    release();
    await vi.waitFor(() => expect(button(root, "Generate topics").disabled).toBe(false));
    if (outcome === "failure") {
      expect(ctrl.mock.logError).toHaveBeenCalledWith("generate proposals", failure);
      expect(root.querySelector('[role="alert"]')?.textContent).toContain("generated proposal was invalid");
    } else {
      expect(ctrl.mock.reload).toHaveBeenCalledOnce();
    }
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


describe("atomic proposal acceptance", () => {
  it("keeps live topics private until their receipt and settings are saved together", async () => {
    const { plugin, saveData, envelopes } = openPluginReview();
    let finish!: () => void;
    const gate = new Promise<void>((resolve) => { finish = resolve; });
    const save = saveData.getMockImplementation()!;
    saveData.mockImplementationOnce(async (data) => { await gate; await save(data); });
    const accepting = plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]);
    await vi.waitFor(() => expect(saveData).toHaveBeenCalledOnce());
    expect(plugin.settings.arxiv.topics).toEqual([]);
    finish();
    await accepting;
    expect(envelopes[0]).toMatchObject({
      libraryProposalAcceptances: [{ proposalId: "proposal-1", processedCandidateIds: ["large-direction-1"] }],
    });
    expect(plugin.settings.arxiv.topics[0]!.directions.map(({ id }) => id)).toEqual(["large-direction-1"]);
  });

  it("can retry a failed write without leaving an in-memory acceptance behind", async () => {
    const { plugin, saveData } = openPluginReview();
    saveData.mockRejectedValueOnce(new Error("disk full"));
    await expect(plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]))
      .rejects.toThrow("disk full");
    expect(plugin.settings.arxiv.topics).toEqual([]);
    expect(plugin.getPersonalLibraryProfileSnapshot().proposalAcceptance ?? null).toBeNull();
    await plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]);
    expect(plugin.settings.arxiv.topics).toHaveLength(1);
    expect(plugin.settings.arxiv.topics[0]!.directions).toHaveLength(1);
  });

  it("reloads acceptance identity and preserves a user's edit while accepting the remaining direction", async () => {
    const initial = topicSnapshot();
    const { plugin, envelopes } = openPluginReview(initial);
    await plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]);
    const persisted = structuredClone(envelopes[0]!);
    persisted.settings.arxiv.topics[0]!.name = "My renamed topic";
    persisted.settings.arxiv.topics[0]!.directions[0]!.text = "My refined research question";
    const reloaded = openPluginReview(initial);
    Object.assign(reloaded.plugin, { loadData: vi.fn(async () => structuredClone(persisted)) });
    await (reloaded.plugin as any).loadSettingsAndState();
    wireSettingsTransaction(reloaded.plugin);
    await reloaded.plugin.acceptPersonalLibraryProposedTopics(["large"]);
    expect(reloaded.plugin.settings.arxiv.topics.map(({ name }) => name)).toEqual(["My renamed topic"]);
    expect(reloaded.plugin.settings.arxiv.topics[0]!.directions.map(({ text }) => text))
      .toEqual(["My refined research question", "Largest coverage direction 2"]);
  });

  it("retains a receipt even when all selected text was already present by hand", async () => {
    const existing = normalizeTopic({ id: "manual-topic", name: "Largest coverage", tag: "largest", detail: true,
      directions: [{ id: "manual-direction", text: "Largest coverage direction 1", origin: "manual" }],
    });
    const { plugin, envelopes } = openPluginReview(topicSnapshot(), [existing]);
    await plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]);
    expect(plugin.settings.arxiv.topics).toEqual([existing]);
    expect(envelopes).toHaveLength(1);
    expect(envelopes[0]).toMatchObject({
      libraryProposalAcceptances: [{ processedCandidateIds: ["large-direction-1"] }],
    });
    plugin.settings.arxiv.topics[0]!.directions = [];
    await plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]);
    expect(plugin.settings.arxiv.topics[0]!.directions).toEqual([]);
  });

  it("serializes two different acceptance selections without dropping either", async () => {
    const { plugin } = openPluginReview();
    await Promise.all([
      plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]),
      plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-2"]),
    ]);
    expect(plugin.settings.arxiv.topics).toHaveLength(1);
    expect(plugin.settings.arxiv.topics[0]!.directions.map(({ id }) => id))
      .toEqual(["large-direction-1", "large-direction-2"]);
  });
});

describe("persisted proposal acceptance", () => {
  it("retains library-specific receipts across ordinary settings writes and switching libraries", async () => {
    const initial = topicSnapshot();
    const { plugin, envelopes } = openPluginReview(initial);
    await plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]);
    plugin.settings.arxiv.topics[0]!.directions = [];
    const other = structuredClone(initial.proposal!);
    other.proposalId = "proposal-other-library";
    other.scopeFingerprint = `sha256:${"c".repeat(64)}`;
    other.topics = other.topics.filter(({ id }) => id === "tie-z");
    Object.assign(plugin, { libraryProposal: other });
    await plugin.acceptPersonalLibraryProposedTopics(["tie-z"]);
    await plugin.saveSettings();
    expect(envelopes.at(-1)!.libraryProposalAcceptances).toHaveLength(2);
    Object.assign(plugin, { libraryProposal: initial.proposal });
    await plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]);
    expect(plugin.settings.arxiv.topics.find(({ id }) => id === "large")!.directions).toEqual([]);
  });

  it("blocks acceptance instead of treating an invalid receipt as a never-reviewed proposal", async () => {
    const { plugin, saveData } = openPluginReview();
    Object.assign(plugin, { loadData: vi.fn(async () => ({ settings: plugin.settings, libraryProposalAcceptances: "broken" })) });
    await (plugin as any).loadSettingsAndState();
    wireSettingsTransaction(plugin);
    expect(plugin.getPersonalLibraryProfileSnapshot().acceptanceLoadError).toBeTruthy();
    await expect(plugin.acceptPersonalLibraryProposedTopics(["large"]))
      .rejects.toThrow(/review state|receipt/i);
    expect(saveData).not.toHaveBeenCalled();
  });
});


describe("processed proposal directions", () => {
  it("rejects editing or discarding an already applied direction through a stale controller", async () => {
    const { plugin } = openPluginReview();
    await plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]);
    await expect(plugin.updatePersonalLibraryProposalCandidate({
      candidateId: "large-direction-1", patch: { text: "Replace accepted text" },
    })).rejects.toThrow(/already.*applied|already.*accepted/i);
    await expect(plugin.removePersonalLibraryProposalCandidate("large-direction-1"))
      .rejects.toThrow(/already.*applied|already.*accepted/i);
    expect(plugin.settings.arxiv.topics[0]!.directions[0]!.text).toBe("Largest coverage direction 1");
  });
});

it("requires a valid explicitly chosen destination before moving a candidate", async () => {
  const { plugin } = openPluginReview();
  await expect(plugin.movePersonalLibraryProposalCandidate({
    candidateId: "large-direction-1", targetTopicId: "missing-topic",
  })).rejects.toThrow(/destination|target/i);
  await plugin.acceptPersonalLibraryProposedTopics(["large"], ["large-direction-1"]);
  await expect(plugin.movePersonalLibraryProposalCandidate({
    candidateId: "large-direction-1", targetTopicId: null, suggestedName: "Separate topic",
  })).rejects.toThrow(/already.*applied|already.*accepted/i);
});
