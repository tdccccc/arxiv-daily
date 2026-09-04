import { App, Modal } from "obsidian";
import {
  PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUES,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH,
  PERSONAL_LIBRARY_MAX_REPRESENTATIVES,
  PERSONAL_LIBRARY_MIN_REPRESENTATIVES,
  isThinEvidenceDirectionCandidate,
  type DirectionProposalProgress,
  type DirectionDiffSuggestion,
  type PersonalLibraryCatalog,
  type PersonalLibraryClusterMember,
  type PersonalLibraryDirectionCandidate,
  type PersonalLibraryDirectionProposal,
  type PersonalLibraryDirectionTextPatch,
  type PersonalLibraryRepresentativeEvidence,
} from "@arxiv-daily/core";
import type { PersonalLibraryProfileSnapshot } from "../../main";
import type { LibraryConnectionStatus } from "./connection";
import { chooseModal } from "../services/modal";

export interface InterestProfileReviewSnapshot
  extends Omit<PersonalLibraryProfileSnapshot, "authorization"> {
  authorization: LibraryConnectionStatus;
}

export interface InterestProfileReviewController {
  snapshot(): InterestProfileReviewSnapshot;
  reload(): Promise<InterestProfileReviewSnapshot>;
  generate(onProgress?: (progress: DirectionProposalProgress) => void): Promise<unknown>;
  updateProposal(input: {
    candidateId: string;
    patch: PersonalLibraryDirectionTextPatch;
    representativePaperKeys: string[];
  }): Promise<InterestProfileReviewSnapshot>;
  discardProposal(candidateId: string): Promise<InterestProfileReviewSnapshot>;
  renameTopic(input: { topicId: string; suggestedName: string }): Promise<InterestProfileReviewSnapshot>;
  /** Writes the kept topics into `settings.topics` (ADR 0014 §1). */
  acceptTopics(topicIds: readonly string[]): Promise<InterestProfileReviewSnapshot>;
}

type ReviewTab = "proposed";
type EditableDirection = PersonalLibraryDirectionCandidate;

/** The edited form of one proposed direction, before it is accepted. */
interface ReviewedDirectionDraft {
  text: string;
  discoveryCues: string[];
  representativePaperKeys: string[];
}

interface DirectionFields {
  text: HTMLTextAreaElement;
  cues: HTMLTextAreaElement;
  representatives: HTMLSelectElement;
}

export class PersonalLibraryInterestProfileModal extends Modal {
  private tab: ReviewTab = "proposed";
  private pending = false;
  private closed = false;
  private renderVersion = 0;
  private readonly selectedTopics = new Set<string>();
  private errorMessage = "";
  private selectedProposals = new Set<string>();
  private selectedConfirmed = new Set<string>();
  /** Identity of the proposal whose selection has already been seeded. */
  private preselectedProposal: string | null = null;
  private generationProgress: DirectionProposalProgress | null = null;
  private fields = new Map<string, DirectionFields>();

  constructor(app: App, private readonly controller: InterestProfileReviewController) {
    super(app);
  }

  onOpen(): void {
    this.closed = false;
    this.render();
  }

  onClose(): void {
    this.closed = true;
    this.renderVersion += 1;
    this.contentEl.empty();
  }

  private render(): void {
    if (this.closed) return;
    this.renderVersion += 1;
    this.fields.clear();
    const snapshot = this.controller.snapshot();
    const root = this.contentEl;
    root.empty();
    root.addClass("arxiv-daily-interest-review");
    // Width belongs on the modal box: Obsidian sizes .modal itself, so asking
    // the content element to be wide only makes it overflow and clip.
    this.modalEl.addClass("arxiv-daily-interest-review-modal");
    root.createEl("h2", { text: "Review personal library directions" });
    root.createEl("p", {
      cls: "arxiv-daily-interest-review__disclosure",
      text: "Proposed directions affect nothing until you confirm them. Evidence is metadata and abstracts, never full text.",
    });

    const toolbar = root.createDiv({ cls: "arxiv-daily-interest-review__toolbar" });
    const tabs = toolbar.createDiv({
      cls: "arxiv-daily-interest-review__tabs",
      attr: { role: "tablist", "aria-label": "Direction review sections" },
    });
    this.addTab(tabs, "proposed", "Proposed");
    // Secondary controls share one row with the tabs; each explains itself on
    // hover instead of spending a line of the header on prose.
    const actions = toolbar.createDiv({ cls: "arxiv-daily-interest-review__toolbar-actions" });
    const generation = generationAvailability(snapshot);
    const generate = actions.createEl("button", {
      text: this.generationLabel(snapshot),
      attr: {
        type: "button",
        title: generation.allowed
          ? "Sends bounded catalog metadata and abstracts to your configured model."
          : generation.reason,
      },
    });
    generate.disabled = this.pending || !generation.allowed;
    generate.addEventListener("click", () => void this.generate(snapshot));
    const refresh = actions.createEl("button", {
      text: "Refresh",
      attr: { type: "button", "aria-label": "Refresh personal library directions" },
    });
    refresh.disabled = this.pending;
    refresh.addEventListener("click", () => void this.run("refresh directions", () => this.controller.reload()));

    const error = root.createDiv({
      cls: "arxiv-daily-interest-review__error",
      attr: { role: "alert", "aria-live": "assertive" },
    });
    error.hidden = !this.errorMessage;
    error.textContent = this.errorMessage;

    const panel = root.createEl("section", {
      cls: "arxiv-daily-interest-review__panel",
      attr: {
        role: "tabpanel",
        id: `arxiv-daily-interest-${this.tab}-panel`,
        "aria-labelledby": `arxiv-daily-interest-${this.tab}-tab`,
      },
    });
    this.renderProposed(panel, snapshot);
  }

  private addTab(parent: HTMLElement, tab: ReviewTab, label: string): void {
    const selected = this.tab === tab;
    const button = parent.createEl("button", {
      cls: "arxiv-daily-interest-review__tab",
      text: label,
      attr: {
        type: "button",
        role: "tab",
        id: `arxiv-daily-interest-${tab}-tab`,
        "aria-selected": String(selected),
        "aria-controls": `arxiv-daily-interest-${tab}-panel`,
        tabindex: selected ? "0" : "-1",
      },
    });
    button.disabled = this.pending;
    button.addEventListener("click", () => this.activateTab(tab, false));
    button.addEventListener("keydown", (event) => {
      // One tab left after the confirmed half retired (ADR 0012 / ADR 0014),
      // so arrow/Home/End have nowhere to move.
      const next: ReviewTab | null = null;
      if (!next) return;
      event.preventDefault();
      this.activateTab(next, true);
    });
  }

  private activateTab(tab: ReviewTab, focus: boolean): void {
    this.tab = tab;
    this.errorMessage = "";
    this.render();
    if (focus && !this.closed) {
      this.contentEl.querySelector<HTMLButtonElement>(`#arxiv-daily-interest-${tab}-tab`)?.focus();
    }
  }

  private renderProposed(parent: HTMLElement, snapshot: InterestProfileReviewSnapshot): void {
    this.renderDocumentError(parent, "Proposal", snapshot.proposalLoadError);
    const topics = snapshot.proposal?.topics ?? [];
    const candidates = topics.flatMap(({ directions }) => directions);
    if (!snapshot.proposal && !snapshot.proposalLoadError) {
      parent.createEl("p", { cls: "arxiv-daily-interest-review__empty", text: "No proposal has been generated." });
      return;
    }
    if (snapshot.proposal && candidates.length === 0) {
      parent.createEl("p", { cls: "arxiv-daily-interest-review__empty", text: "This proposal contains no directions." });
      return;
    }
    // A load error can leave no proposal at all while still rendering the tab.
    if (snapshot.proposal) this.preselectProposals(snapshot.proposal, candidates);
    // The accept bar that used to sit here confirmed selected candidates into
    // the interest profile document. Accepting a whole proposed structure into
    // `settings.topics` (ADR 0014 §1) is built in P4's later chunks.
    const allowedKeys = proposalPaperKeys(snapshot);
    // The whole structure is the unit of acceptance (ADR 0014 §1), so the
    // action sits above the list rather than on each row.
    if (topics.length > 0) {
      const bar = parent.createDiv({ cls: "arxiv-daily-interest-review__accept-bar" });
      const accept = bar.createEl("button", {
        text: `Accept ${this.selectedTopics.size} topic(s) into settings`,
        attr: {
          type: "button",
          title: "Adds the selected topics, with their directions, to your research topics. You can edit or remove them there afterwards.",
        },
      });
      accept.addClass("mod-cta");
      accept.disabled = this.pending || this.selectedTopics.size === 0;
      accept.addEventListener("click", () => void this.acceptSelectedTopics());
    }
    for (const topic of topics) {
      const section = parent.createDiv({ cls: "arxiv-daily-interest-review__topic" });
      const heading = section.createDiv({ cls: "arxiv-daily-interest-review__topic-heading" });
      const select = heading.createEl("input", { type: "checkbox" });
      select.checked = this.selectedTopics.has(topic.id);
      select.setAttribute("aria-label", `Accept ${topic.suggestedName}`);
      select.disabled = this.pending;
      select.addEventListener("change", () => {
        if (select.checked) this.selectedTopics.add(topic.id);
        else this.selectedTopics.delete(topic.id);
        this.render();
      });
      // The generated name is a suggestion; the machine tag is derived from
      // whatever it says at acceptance, so it is editable right here.
      const name = heading.createEl("input", { type: "text", value: topic.suggestedName });
      name.setAttribute("aria-label", "Topic name");
      name.disabled = this.pending;
      name.addEventListener("change", () => {
        const next = name.value.trim();
        if (!next || next === topic.suggestedName) return;
        void this.run("rename proposed topic", () =>
          this.controller.renameTopic({ topicId: topic.id, suggestedName: next }));
      });
      heading.createSpan({
        cls: "arxiv-daily-interest-review__topic-count",
        text: `${topic.directions.length} direction(s)`,
      });
      for (const candidate of topic.directions) {
        this.renderDirectionCard(section, candidate, allowedKeys, "proposal", snapshot);
      }
    }
    this.renderBufferPool(parent, snapshot);
    this.renderIncrementalNotice(parent);
  }

  private renderDocumentError(
    parent: HTMLElement,
    label: string,
    error: { message: string } | null | undefined,
  ): void {
    if (!error) return;
    parent.createEl("p", {
      cls: "arxiv-daily-interest-review__document-error",
      attr: { role: "status" },
      text: `${label} could not be loaded: ${error.message}`,
    });
  }

  private async acceptSelectedTopics(): Promise<void> {
    const ids = [...this.selectedTopics];
    if (ids.length === 0) return;
    await this.run("accept proposed topics", async () => {
      const snapshot = await this.controller.acceptTopics(ids);
      this.selectedTopics.clear();
      return snapshot;
    });
  }

  private renderDirectionCard(
    parent: HTMLElement,
    direction: EditableDirection,
    allowedPaperKeys: string[],
    kind: "proposal" | "confirmed",
    snapshot: InterestProfileReviewSnapshot,
  ): void {
    const card = parent.createEl("article", { cls: "arxiv-daily-interest-review__card" });
    const heading = card.createDiv({ cls: "arxiv-daily-interest-review__card-heading" });
    const selected = kind === "proposal" ? this.selectedProposals : this.selectedConfirmed;
    const terminal = !("status" in direction) || direction.status !== "merged";
    const selectLabel = heading.createEl("label", { cls: "arxiv-daily-interest-review__select" });
    const checkbox = selectLabel.createEl("input", { type: "checkbox" });
    checkbox.checked = selected.has(direction.id);
    checkbox.disabled = this.pending || !terminal;
    checkbox.setAttribute("aria-label", kind === "proposal"
      ? `Select ${direction.text}`
      : `Select ${direction.text} for merge`);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selected.add(direction.id);
      else selected.delete(direction.id);
      this.render();
    });
    heading.createEl("strong", { text: direction.text });
    if (kind === "proposal" && isThinEvidenceDirectionCandidate(direction)) {
      heading.createSpan({
        text: "thin evidence",
        cls: "arxiv-daily-interest-review__status is-thin",
        attr: { title: "Only one representative paper. Select it explicitly to include it." },
      });
    }
    if (kind === "proposal") {
      heading.createSpan({
        cls: "arxiv-daily-interest-review__summary",
        text: directionRowSummary(direction),
      });
    }

    // Proposals are reviewed by scanning many rows and deselecting a few, so
    // the editor is collapsed behind the row rather than stacked in front of
    // it; nothing is removed, only folded away until it is wanted.
    const body = kind === "proposal"
      ? this.createCardDetail(card)
      : card;
    const form = body.createDiv({ cls: "arxiv-daily-interest-review__form" });
    const text = this.textArea(form, "Direction (one line)", direction.text, PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH, 2);
    const cues = this.textArea(form, "Discovery cues (one per line)", direction.discoveryCues.join("\n"), undefined, 4);
    cues.maxLength = (PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH + 1) * PERSONAL_LIBRARY_MAX_DISCOVERY_CUES;
    const representatives = this.representativeSelect(form, allowedPaperKeys, direction.representatives.map((item) => item.paperKey));
    this.fields.set(direction.id, { text, cues, representatives });

    const evidence = body.createEl("details", { cls: "arxiv-daily-interest-review__evidence" });
    evidence.createEl("summary", { text: `Evidence: ${direction.representatives.length} representative paper(s), metadata and abstract only` });
    const list = evidence.createEl("ul");
    for (const representative of direction.representatives) {
      const paper = snapshot.catalog?.papers[representative.paperKey];
      list.createEl("li", { text: paper ? `${paper.title} — ${representative.paperKey}` : `${representative.paperKey} — missing from current catalog` });
    }

    if (direction.clusterMembers && direction.clusterMembers.length > 0) {
      this.renderClusterMembers(body, direction.clusterMembers, snapshot);
    }
    const actions = body.createDiv({ cls: "arxiv-daily-interest-review__card-actions" });
    const save = actions.createEl("button", { text: "Save edits", attr: { type: "button" } });
    save.disabled = this.pending || !terminal;
    save.addEventListener("click", () => void this.saveProposal(direction.id));
    this.renderProposalActions(actions, direction.id);
  }

  /**
   * The collapsed half of a proposal row. Rendering it closed on every render
   * is deliberate: a re-render follows selection changes and saves, and a row
   * that reopened itself would fight the scan the list exists for.
   */
  private createCardDetail(card: HTMLElement): HTMLElement {
    const detail = card.createEl("details", { cls: "arxiv-daily-interest-review__detail" });
    detail.createEl("summary", { text: "Edit" });
    return detail;
  }

  private renderProposalActions(parent: HTMLElement, candidateId: string): void {
    const discard = parent.createEl("button", { text: "Discard", attr: { type: "button" } });
    discard.addClass("mod-warning");
    discard.disabled = this.pending;
    discard.addEventListener("click", () => void this.discardProposal(candidateId));
  }

  private renderClusterMembers(
    parent: HTMLElement,
    members: readonly PersonalLibraryClusterMember[],
    snapshot: InterestProfileReviewSnapshot,
  ): void {
    const details = parent.createEl("details", { cls: "arxiv-daily-interest-review__cluster" });
    details.createEl("summary", { text: describeClusterMembers(members) ?? `Cluster members ${members.length}` });
    const list = details.createEl("ul");
    for (const member of members) {
      const paper = snapshot.catalog?.papers[member.paperKey];
      const label = paper ? paper.title : member.paperKey;
      list.createEl("li", { text: `${label} — ${formatConfidence(member.confidence)}` });
    }
  }

  private renderBufferPool(parent: HTMLElement, snapshot: InterestProfileReviewSnapshot): void {
    const buffer = unclassifiedBufferPoolPapers(snapshot.proposal);
    if (buffer.length === 0) return;
    const section = parent.createDiv({ cls: "arxiv-daily-interest-review__buffer" });
    section.createEl("strong", { text: bufferPoolHeading(buffer.length) });
    const list = section.createEl("ul");
    for (const entry of buffer) {
      const paper = snapshot.catalog?.papers[entry.paperKey];
      list.createEl("li", {
        text: paper ? `${paper.title} — ${entry.paperKey}` : `${entry.paperKey} — missing from current catalog`,
      });
    }
    section.createEl("p", {
      cls: "arxiv-daily-interest-review__hint",
      text: "No proposed direction covers these papers. Accepting the proposal will not select them.",
    });
  }

  /**
   * The incremental suggestion flow filed new papers into confirmed directions
   * of the interest profile document, which retired with ADR 0012. It is
   * rebuilt against topics in a later phase; saying so is better than a blank
   * space the researcher reads as a bug.
   */
  private renderIncrementalNotice(parent: HTMLElement): void {
    parent.createEl("p", {
      cls: "arxiv-daily-interest-review__hint",
      attr: { role: "status" },
      text: "Incremental suggestions for papers added after this scan are unavailable while they are rebuilt around topics. Re-running the scan proposes the whole structure again.",
    });
  }

  private textField(parent: HTMLElement, labelText: string, value: string, maxLength: number): HTMLInputElement {
    const label = parent.createEl("label", { cls: "arxiv-daily-interest-review__field" });
    label.createSpan({ text: labelText });
    const input = label.createEl("input", { type: "text" });
    input.value = value;
    input.maxLength = maxLength;
    input.disabled = this.pending;
    return input;
  }

  private textArea(parent: HTMLElement, labelText: string, value: string, maxLength: number | undefined, rows: number): HTMLTextAreaElement {
    const label = parent.createEl("label", { cls: "arxiv-daily-interest-review__field" });
    label.createSpan({ text: labelText });
    const input = label.createEl("textarea");
    input.value = value;
    input.rows = rows;
    if (maxLength !== undefined) input.maxLength = maxLength;
    input.disabled = this.pending;
    return input;
  }

  private representativeSelect(parent: HTMLElement, allowed: string[], selected: string[]): HTMLSelectElement {
    const label = parent.createEl("label", { cls: "arxiv-daily-interest-review__field" });
    label.createSpan({ text: `Representative papers (choose ${PERSONAL_LIBRARY_MIN_REPRESENTATIVES}–${PERSONAL_LIBRARY_MAX_REPRESENTATIVES})` });
    const select = label.createEl("select", { attr: { multiple: "", size: "5" } });
    const selectedSet = new Set(selected);
    for (const paperKey of allowed) {
      const option = select.createEl("option");
      option.value = paperKey;
      option.textContent = paperKey;
      option.selected = selectedSet.has(paperKey);
    }
    select.disabled = this.pending;
    return select;
  }

  private draft(id: string): ReviewedDirectionDraft | null {
    const fields = this.fields.get(id);
    if (!fields) return null;
    const text = fields.text.value.trim().replace(/\s+/gu, " ");
    const discoveryCues = normalizeLines(fields.cues.value);
    const representativePaperKeys = Array.from(fields.representatives.selectedOptions, (option) => option.value).sort(codeUnitCompare);
    const error = validateDraft({ text, discoveryCues, representativePaperKeys });
    if (error) {
      this.errorMessage = error;
      this.renderErrorOnly();
      return null;
    }
    return { text, discoveryCues, representativePaperKeys };
  }

  private patch(draft: ReviewedDirectionDraft): PersonalLibraryDirectionTextPatch {
    return { text: draft.text, discoveryCues: draft.discoveryCues };
  }

  private async generate(snapshot: InterestProfileReviewSnapshot): Promise<void> {
    if (snapshot.proposal) {
      const choice = await chooseModal(this.app, "Regenerate proposed directions", "Replace the current proposal and all unconfirmed edits with newly generated directions?", [
        { label: "Cancel", value: "cancel" },
        { label: "Regenerate", value: "regenerate", warning: true },
      ]);
      if (choice !== "regenerate" || this.closed) return;
    }
    // Progress lands on the button that started it: this modal covers the
    // status bar, so anything reported there would be invisible here.
    this.generationProgress = null;
    try {
      await this.run("generate proposals", async () => {
        await this.controller.generate((progress) => {
          this.generationProgress = progress;
          this.updateGenerationLabel();
        });
        return this.controller.reload();
      });
    } finally {
      this.generationProgress = null;
    }
  }

  private generationLabel(snapshot: InterestProfileReviewSnapshot): string {
    const progress = this.generationProgress;
    if (progress) return `Generating… (${progress.completed}/${progress.total})`;
    return snapshot.proposal ? "Regenerate proposals" : "Generate proposals";
  }

  /**
   * Writes the count into the already-rendered button rather than re-rendering:
   * generation reports once per cluster, and a full re-render would discard
   * whatever the researcher is editing in an open row.
   */
  private updateGenerationLabel(): void {
    if (this.closed) return;
    const button = this.contentEl.querySelector<HTMLButtonElement>(
      ".arxiv-daily-interest-review__toolbar-actions button",
    );
    if (button) button.textContent = this.generationLabel(this.controller.snapshot());
  }

  private saveProposal(id: string): void {
    const draft = this.draft(id);
    if (!draft) return;
    void this.run("save proposed direction", () => this.controller.updateProposal({ candidateId: id, patch: this.patch(draft), representativePaperKeys: draft.representativePaperKeys }));
  }

  private async discardProposal(id: string): Promise<void> {
    const choice = await chooseModal(this.app, "Discard proposed direction", "Discard this proposed direction and its reviewed edits?", [
      { label: "Cancel", value: "cancel" },
      { label: "Discard", value: "discard", warning: true },
    ]);
    if (choice !== "discard" || this.closed) return;
    this.selectedProposals.delete(id);
    await this.run("discard proposed direction", () => this.controller.discardProposal(id));
  }

  /**
   * Seed the selection once per proposal: everything starts selected so that
   * accepting the whole set is one action, except candidates marked thin,
   * which stay out until the researcher opts them in (ADR 0009 §3). Re-seeding
   * only when the proposal identity changes preserves de-selections made since.
   */
  private preselectProposals(
    proposal: PersonalLibraryDirectionProposal,
    candidates: readonly PersonalLibraryDirectionCandidate[],
  ): void {
    const identity = `${proposal.proposalId}:${proposal.revision}`;
    if (this.preselectedProposal === identity) return;
    this.preselectedProposal = identity;
    this.selectedProposals = new Set(
      candidates
        .filter((candidate) => !isThinEvidenceDirectionCandidate(candidate))
        .map(({ id }) => id),
    );
  }

  private async run(action: string, operation: () => Promise<unknown>): Promise<void> {
    void action;
    if (this.pending || this.closed) return;
    this.pending = true;
    this.errorMessage = "";
    const version = ++this.renderVersion;
    this.render();
    try {
      await operation();
      if (!this.closed && version <= this.renderVersion) this.render();
    } catch (error) {
      if (!this.closed) {
        this.errorMessage = safeUserError(error);
        this.render();
      }
    } finally {
      this.pending = false;
      if (!this.closed) this.render();
    }
  }

  private renderErrorOnly(): void {
    const error = this.contentEl.querySelector(".arxiv-daily-interest-review__error");
    if (error instanceof HTMLElement) {
      error.hidden = false;
      error.textContent = this.errorMessage;
    }
  }
}

export function openPersonalLibraryInterestProfileModal(
  app: App,
  controller: InterestProfileReviewController,
): PersonalLibraryInterestProfileModal {
  const modal = new PersonalLibraryInterestProfileModal(app, controller);
  modal.open();
  return modal;
}

/**
 * What a collapsed row has to say for itself: how many cues describe the
 * direction and how much library evidence stands behind it. Both numbers are
 * already on the candidate; nothing is computed or fetched for the row.
 */
export function directionRowSummary(direction: EditableDirection): string {
  const cues = direction.discoveryCues.length;
  const papers = direction.representatives.length;
  return `${cues} ${cues === 1 ? "cue" : "cues"} · ${papers} ${papers === 1 ? "paper" : "papers"}`;
}

export function normalizeLines(value: string): string[] {
  return Array.from(new Set(value.split(/\r?\n/u).map((line) => line.trim()).filter(Boolean))).sort(codeUnitCompare);
}

export function validateDraft(draft: ReviewedDirectionDraft): string | null {
  if (!draft.text) return "Enter the direction as one line.";
  if (draft.text.length > PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH) return `Direction must be at most ${PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH} characters.`;
  if (draft.discoveryCues.length < 1 || draft.discoveryCues.length > PERSONAL_LIBRARY_MAX_DISCOVERY_CUES) return `Enter 1–${PERSONAL_LIBRARY_MAX_DISCOVERY_CUES} non-empty discovery cues.`;
  if (draft.discoveryCues.some((cue) => cue.length > PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH)) return `Each discovery cue must be at most ${PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH} characters.`;
  if (draft.representativePaperKeys.length < PERSONAL_LIBRARY_MIN_REPRESENTATIVES || draft.representativePaperKeys.length > PERSONAL_LIBRARY_MAX_REPRESENTATIVES) return `Choose ${PERSONAL_LIBRARY_MIN_REPRESENTATIVES}–${PERSONAL_LIBRARY_MAX_REPRESENTATIVES} representative papers.`;
  return null;
}


export function formatConfidence(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function describeClusterMembers(members: readonly PersonalLibraryClusterMember[]): string | null {
  if (members.length === 0) return null;
  const average = members.reduce((sum, member) => sum + member.confidence, 0) / members.length;
  return `Cluster members ${members.length} · avg. confidence ${formatConfidence(average)}`;
}

export function bufferPoolHeading(count: number): string {
  return `Unclustered (buffer pool) ${count}`;
}

export function formatTimelineTimestamp(at: string): string {
  const date = new Date(at);
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}`;
}

/**
 * Papers the proposal saw but that no proposed direction covers. Surfacing the
 * count is deliberate: the researcher should be able to see how much of the
 * library this proposal does not speak for.
 */
export function unclassifiedBufferPoolPapers(
  proposal: Pick<PersonalLibraryDirectionProposal, "catalogInputPapers" | "topics"> | null,
): PersonalLibraryRepresentativeEvidence[] {
  if (!proposal) return [];
  const covered = new Set<string>();
  for (const topic of proposal.topics) {
    for (const candidate of topic.directions) {
      for (const member of candidate.clusterMembers ?? []) covered.add(member.paperKey);
    }
  }
  return proposal.catalogInputPapers.filter((entry) => !covered.has(entry.paperKey));
}

/**
 * Content key of one incremental suggestion; must match the plugin's key
 * scheme (kind:directionId:firstPaperKey). Keys are only ever compared
 * against keys computed the same way, never parsed.
 */
export function incrementalSuggestionKey(suggestion: DirectionDiffSuggestion): string {
  switch (suggestion.kind) {
    case "attach":
      return `attach:${suggestion.directionId}:${suggestion.paperKeys[0]}`;
    case "new":
      return `new::${suggestion.paperKeys[0]}`;
    case "split":
      return `split:${suggestion.directionId}:${suggestion.paperKeys[0]}`;
    case "merge":
      return `merge:${suggestion.directionIds[0]}:${suggestion.directionIds[1]}`;
  }
}

export function incrementalSuggestionPaperCount(suggestion: DirectionDiffSuggestion): string {
  if (suggestion.kind === "merge") return "2 directions";
  return `${suggestion.paperKeys.length} paper(s)`;
}

export function truncateReason(reason: string, maximum = 160): string {
  if (reason.length <= maximum) return reason;
  return `${reason.slice(0, maximum).trimEnd()}…`;
}

function generationAvailability(snapshot: InterestProfileReviewSnapshot): { allowed: boolean; reason: string } {
  if (snapshot.authorization.kind !== "authorized") return { allowed: false, reason: "Authorize current personal-library model processing to generate proposals. Local review remains available." };
  if (!snapshot.catalog) return { allowed: false, reason: snapshot.catalogLoadError?.message ? `Load the current catalog first: ${snapshot.catalogLoadError.message}` : "Scan and load the current personal-library catalog first." };
  if (Object.keys(snapshot.catalog.papers).length === 0) return { allowed: false, reason: "The current catalog has no metadata-and-abstract papers to propose from." };
  return { allowed: true, reason: "" };
}

function proposalPaperKeys(snapshot: InterestProfileReviewSnapshot): string[] {
  const manifest = snapshot.proposal?.catalogInputPapers.map((item) => item.paperKey) ?? [];
  const current = new Set(catalogPaperKeys(snapshot.catalog));
  return manifest.filter((key) => current.has(key)).sort(codeUnitCompare);
}

function catalogPaperKeys(catalog: PersonalLibraryCatalog | null): string[] {
  return catalog ? Object.keys(catalog.papers).sort(codeUnitCompare) : [];
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

export function safeUserError(error: unknown): string {
  const code = error && typeof error === "object" && "code" in error
    && typeof (error as { code?: unknown }).code === "string"
    ? (error as { code: string }).code
    : "";
  const messages: Record<string, string> = {
    "invalid-input": "The reviewed direction is invalid. Check its fields and try again.",
    "invalid-document": "The saved review data is invalid. Refresh before trying again.",
    "incompatible-catalog": "The current catalog is not compatible with this review. Refresh the library first.",
    "not-found": "That direction no longer exists. Refresh and try again.",
    conflict: "The review changed elsewhere. Refresh before trying again.",
    stale: "The review changed elsewhere. Refresh before trying again.",
    "partial-confirmation-conflict": "The review changed while saving. Refresh before trying again.",
    "lineage-limit": "These directions have too much merge history to combine.",
    "direction-limit": "The confirmed direction limit has been reached.",
    "merge-relationship": "These directions cannot be changed without breaking merge history.",
    "evidence-mismatch": "Representative evidence is missing or stale. Refresh the catalog and review it again.",
    "catalog-invalid": "The current catalog is invalid. Refresh or rescan the library.",
    "no-evidence": "The current catalog has no eligible metadata-and-abstract evidence.",
    "evidence-too-large": "The selected catalog evidence is too large to process safely.",
    "synthesis-too-large": "The proposed direction synthesis is too large. Reduce the library selection and retry.",
    "output-too-large": "The model response was too large. Retry generation.",
    "proposal-invariant": "The generated proposal was invalid. Retry generation.",
  };
  return messages[code] ?? "Operation failed. Refresh and try again.";
}
