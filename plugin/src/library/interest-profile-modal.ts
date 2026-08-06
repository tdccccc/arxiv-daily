import { App, Modal } from "obsidian";
import {
  PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUES,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH,
  PERSONAL_LIBRARY_MAX_NAME_LENGTH,
  PERSONAL_LIBRARY_MAX_REPRESENTATIVES,
  PERSONAL_LIBRARY_MIN_REPRESENTATIVES,
  type DirectionDiffSuggestion,
  type PersonalLibraryCatalog,
  type PersonalLibraryClusterMember,
  type PersonalLibraryConfirmedDirection,
  type PersonalLibraryDirectionCandidate,
  type PersonalLibraryDirectionProposal,
  type PersonalLibraryDirectionTextPatch,
  type PersonalLibraryDirectionTimelineEvent,
  type PersonalLibraryRepresentativeEvidence,
  type PersonalLibraryReviewedDirectionDraft,
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
  generate(): Promise<unknown>;
  updateProposal(input: {
    candidateId: string;
    patch: PersonalLibraryDirectionTextPatch;
    representativePaperKeys: string[];
  }): Promise<InterestProfileReviewSnapshot>;
  mergeProposals(input: {
    sourceCandidateIds: string[];
    draft: PersonalLibraryReviewedDirectionDraft;
  }): Promise<InterestProfileReviewSnapshot>;
  discardProposal(candidateId: string): Promise<InterestProfileReviewSnapshot>;
  confirmProposal(input: {
    candidateId: string;
    draft: PersonalLibraryReviewedDirectionDraft;
    status: "active" | "disabled";
  }): Promise<InterestProfileReviewSnapshot>;
  updateConfirmed(input: {
    directionId: string;
    patch: PersonalLibraryDirectionTextPatch;
    representativePaperKeys: string[];
  }): Promise<InterestProfileReviewSnapshot>;
  mergeConfirmed(input: {
    sourceDirectionIds: string[];
    draft: PersonalLibraryReviewedDirectionDraft;
    status: "active" | "disabled";
  }): Promise<InterestProfileReviewSnapshot>;
  enable(directionId: string): Promise<InterestProfileReviewSnapshot>;
  disable(directionId: string): Promise<InterestProfileReviewSnapshot>;
  remove(input: {
    directionId: string;
    mode: "restrict" | "cascade";
  }): Promise<InterestProfileReviewSnapshot>;
  applySuggestion(key: string): Promise<InterestProfileReviewSnapshot>;
  dismissSuggestion(key: string): Promise<InterestProfileReviewSnapshot>;
  lock(directionId: string): Promise<InterestProfileReviewSnapshot>;
  unlock(directionId: string): Promise<InterestProfileReviewSnapshot>;
}

type ReviewTab = "proposed" | "confirmed";
type EditableDirection = PersonalLibraryDirectionCandidate | PersonalLibraryConfirmedDirection;

interface DirectionFields {
  name: HTMLInputElement;
  description: HTMLTextAreaElement;
  cues: HTMLTextAreaElement;
  representatives: HTMLSelectElement;
}

export class PersonalLibraryInterestProfileModal extends Modal {
  private tab: ReviewTab = "proposed";
  private pending = false;
  private closed = false;
  private renderVersion = 0;
  private errorMessage = "";
  private selectedProposals = new Set<string>();
  private selectedConfirmed = new Set<string>();
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
    root.createEl("h2", { text: "Review personal library directions" });
    root.createEl("p", {
      cls: "arxiv-daily-interest-review__disclosure",
      text: "Proposed directions do not affect discovery. Only active confirmed directions with compatible, current evidence may become eligible for discovery later. Evidence uses metadata and abstracts, not full text.",
    });

    const toolbar = root.createDiv({ cls: "arxiv-daily-interest-review__toolbar" });
    const tabs = toolbar.createDiv({
      cls: "arxiv-daily-interest-review__tabs",
      attr: { role: "tablist", "aria-label": "Direction review sections" },
    });
    this.addTab(tabs, "proposed", "Proposed");
    this.addTab(tabs, "confirmed", "Confirmed");
    const refresh = toolbar.createEl("button", {
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

    this.renderIncrementalSuggestions(root, snapshot);

    const panel = root.createEl("section", {
      cls: "arxiv-daily-interest-review__panel",
      attr: {
        role: "tabpanel",
        id: `arxiv-daily-interest-${this.tab}-panel`,
        "aria-labelledby": `arxiv-daily-interest-${this.tab}-tab`,
      },
    });
    if (this.tab === "proposed") this.renderProposed(panel, snapshot);
    else this.renderConfirmed(panel, snapshot);
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
      let next: ReviewTab | null = null;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        next = tab === "proposed" ? "confirmed" : "proposed";
      } else if (event.key === "Home") {
        next = "proposed";
      } else if (event.key === "End") {
        next = "confirmed";
      }
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
    const generation = generationAvailability(snapshot);
    const controls = parent.createDiv({ cls: "arxiv-daily-interest-review__section-actions" });
    const generate = controls.createEl("button", {
      text: snapshot.proposal ? "Regenerate proposals" : "Generate proposals",
      attr: { type: "button" },
    });
    generate.disabled = this.pending || !generation.allowed;
    if (!generation.allowed) generate.title = generation.reason;
    generate.addEventListener("click", () => void this.generate(snapshot));
    controls.createSpan({
      cls: "arxiv-daily-interest-review__hint",
      text: generation.allowed ? "Generation sends bounded catalog metadata and abstracts to your configured model." : generation.reason,
    });

    const candidates = snapshot.proposal?.candidates ?? [];
    if (!snapshot.proposal && !snapshot.proposalLoadError) {
      parent.createEl("p", { cls: "arxiv-daily-interest-review__empty", text: "No proposal has been generated." });
      return;
    }
    if (snapshot.proposal && candidates.length === 0) {
      parent.createEl("p", { cls: "arxiv-daily-interest-review__empty", text: "This proposal contains no directions." });
      return;
    }
    const allowedKeys = proposalPaperKeys(snapshot);
    for (const candidate of candidates) {
      this.renderDirectionCard(parent, candidate, allowedKeys, "proposal", snapshot);
    }
    if (candidates.length > 0) {
      const merge = parent.createEl("button", { text: "Merge selected proposals", attr: { type: "button" } });
      merge.disabled = this.pending || this.selectedProposals.size < 2;
      merge.addEventListener("click", () => void this.mergeSelectedProposals(snapshot));
    }
    this.renderBufferPool(parent, snapshot);
  }

  private renderConfirmed(parent: HTMLElement, snapshot: InterestProfileReviewSnapshot): void {
    this.renderDocumentError(parent, "Confirmed profile", snapshot.profileLoadError);
    if (snapshot.eligibility.documentDiagnostics.length > 0) {
      parent.createEl("p", {
        cls: "arxiv-daily-interest-review__diagnostic",
        text: `Profile/catalog compatibility: ${snapshot.eligibility.documentDiagnostics.join(", ")}.`,
      });
    }
    const directions = snapshot.profile?.directions ?? [];
    if (!snapshot.profile && !snapshot.profileLoadError) {
      parent.createEl("p", { cls: "arxiv-daily-interest-review__empty", text: "No confirmed profile exists yet." });
      return;
    }
    if (snapshot.profile && directions.length === 0) {
      parent.createEl("p", { cls: "arxiv-daily-interest-review__empty", text: "No directions have been confirmed." });
      return;
    }
    const allowedKeys = catalogPaperKeys(snapshot.catalog);
    for (const direction of directions) {
      this.renderDirectionCard(parent, direction, allowedKeys, "confirmed", snapshot);
    }
    const merge = parent.createEl("button", { text: "Merge selected confirmed directions", attr: { type: "button" } });
    merge.disabled = this.pending || this.selectedConfirmed.size < 2;
    merge.addEventListener("click", () => void this.mergeSelectedConfirmed());
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
    selectLabel.appendText("Select for merge");
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selected.add(direction.id);
      else selected.delete(direction.id);
      this.render();
    });
    heading.createEl("strong", { text: direction.name });
    if ("status" in direction) heading.createSpan({ text: direction.status, cls: `arxiv-daily-interest-review__status is-${direction.status}` });
    if ("lockedAt" in direction && direction.lockedAt !== undefined) {
      heading.createSpan({ text: "locked", cls: "arxiv-daily-interest-review__status is-locked" });
    }

    const form = card.createDiv({ cls: "arxiv-daily-interest-review__form" });
    const name = this.textField(form, "Name", direction.name, PERSONAL_LIBRARY_MAX_NAME_LENGTH);
    const description = this.textArea(form, "Description", direction.description, PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH, 3);
    const cues = this.textArea(form, "Discovery cues (one per line)", direction.discoveryCues.join("\n"), undefined, 4);
    cues.maxLength = (PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH + 1) * PERSONAL_LIBRARY_MAX_DISCOVERY_CUES;
    const representatives = this.representativeSelect(form, allowedPaperKeys, direction.representatives.map((item) => item.paperKey));
    this.fields.set(direction.id, { name, description, cues, representatives });

    const evidence = card.createEl("details", { cls: "arxiv-daily-interest-review__evidence" });
    evidence.createEl("summary", { text: `Evidence: ${direction.representatives.length} representative paper(s), metadata and abstract only` });
    const list = evidence.createEl("ul");
    for (const representative of direction.representatives) {
      const paper = snapshot.catalog?.papers[representative.paperKey];
      list.createEl("li", { text: paper ? `${paper.title} — ${representative.paperKey}` : `${representative.paperKey} — missing from current catalog` });
    }

    if (direction.clusterMembers && direction.clusterMembers.length > 0) {
      this.renderClusterMembers(card, direction.clusterMembers, snapshot);
    }
    if (kind === "confirmed") {
      this.renderConfirmedDiagnostics(card, direction as PersonalLibraryConfirmedDirection, snapshot);
      this.renderTimeline(card, (direction as PersonalLibraryConfirmedDirection).timeline);
    }
    const actions = card.createDiv({ cls: "arxiv-daily-interest-review__card-actions" });
    const save = actions.createEl("button", { text: "Save edits", attr: { type: "button" } });
    save.disabled = this.pending || !terminal;
    save.addEventListener("click", () => kind === "proposal" ? this.saveProposal(direction.id) : this.saveConfirmed(direction.id));
    if (kind === "proposal") this.renderProposalActions(actions, direction.id);
    else this.renderConfirmedActions(actions, direction as PersonalLibraryConfirmedDirection);
  }

  private renderProposalActions(parent: HTMLElement, candidateId: string): void {
    for (const status of ["active", "disabled"] as const) {
      const button = parent.createEl("button", {
        text: status === "active" ? "Confirm active" : "Confirm disabled",
        attr: { type: "button" },
      });
      button.disabled = this.pending;
      button.addEventListener("click", () => void this.confirmProposal(candidateId, status));
    }
    const discard = parent.createEl("button", { text: "Discard", attr: { type: "button" } });
    discard.addClass("mod-warning");
    discard.disabled = this.pending;
    discard.addEventListener("click", () => void this.discardProposal(candidateId));
  }

  private renderConfirmedActions(parent: HTMLElement, direction: PersonalLibraryConfirmedDirection): void {
    if (direction.status !== "merged") {
      const toggle = parent.createEl("button", { text: direction.status === "active" ? "Disable" : "Enable", attr: { type: "button" } });
      toggle.disabled = this.pending;
      toggle.addEventListener("click", () => void this.toggleConfirmed(direction));
      if (direction.lockedAt !== undefined) {
        const unlock = parent.createEl("button", { text: "Unlock", attr: { type: "button" } });
        unlock.disabled = this.pending;
        unlock.addEventListener("click", () => void this.run("unlock direction", () => this.controller.unlock(direction.id)));
      } else {
        const lock = parent.createEl("button", { text: "Lock", attr: { type: "button" } });
        lock.disabled = this.pending;
        lock.addEventListener("click", () => void this.run("lock direction", () => this.controller.lock(direction.id)));
      }
    }
    const restrict = parent.createEl("button", { text: "Remove", attr: { type: "button" } });
    restrict.addClass("mod-warning");
    restrict.disabled = this.pending;
    restrict.addEventListener("click", () => void this.removeConfirmed(direction.id, "restrict"));
    const cascade = parent.createEl("button", { text: "Cascade remove merge family", attr: { type: "button" } });
    cascade.addClass("mod-warning");
    cascade.disabled = this.pending;
    cascade.addEventListener("click", () => void this.removeConfirmed(direction.id, "cascade"));
  }

  private renderConfirmedDiagnostics(
    parent: HTMLElement,
    direction: PersonalLibraryConfirmedDirection,
    snapshot: InterestProfileReviewSnapshot,
  ): void {
    const diagnostic = snapshot.eligibility.diagnostics.find((item) => item.directionId === direction.id);
    const details = parent.createEl("details", { cls: "arxiv-daily-interest-review__diagnostics" });
    details.createEl("summary", { text: diagnostic?.eligible ? "Eligible with current catalog" : "Eligibility and stale diagnostics" });
    if (direction.status === "merged") details.createEl("p", { text: `Merged into ${direction.mergedIntoDirectionId}.` });
    if (!diagnostic || diagnostic.reasons.length === 0) {
      details.createEl("p", { text: diagnostic?.eligible ? "Active, compatible, and current." : "No direction-level stale evidence was reported." });
      return;
    }
    const list = details.createEl("ul");
    for (const reason of diagnostic.reasons) {
      list.createEl("li", { text: reason.paperKey ? `${reason.reason}: ${reason.paperKey}` : reason.reason });
    }
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

  private renderTimeline(parent: HTMLElement, timeline: readonly PersonalLibraryDirectionTimelineEvent[] | undefined): void {
    if (!timeline || timeline.length === 0) return;
    const section = parent.createDiv({ cls: "arxiv-daily-interest-review__timeline" });
    section.createEl("strong", { text: "Timeline" });
    const list = section.createEl("ul");
    for (const event of timeline.slice(-PERSONAL_LIBRARY_TIMELINE_DISPLAY_LIMIT).reverse()) {
      list.createEl("li", { text: `${formatTimelineTimestamp(event.at)} — ${timelineEventLabel(event.kind)}` });
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
    section.createEl("p", { cls: "arxiv-daily-interest-review__hint", text: "这些论文未进入任何方向草案" });
  }

  private renderIncrementalSuggestions(parent: HTMLElement, snapshot: InterestProfileReviewSnapshot): void {
    this.renderDocumentError(parent, "Incremental suggestions", snapshot.suggestionsLoadError);
    const suggestions = snapshot.suggestions?.suggestions ?? [];
    if (suggestions.length === 0) {
      parent.createEl("p", {
        cls: "arxiv-daily-interest-review__suggestions-empty",
        text: "No incremental suggestions. Run a check for new papers to review suggestions here.",
      });
      return;
    }
    const section = parent.createDiv({ cls: "arxiv-daily-interest-review__suggestions" });
    section.createEl("strong", { text: `Incremental suggestions ${suggestions.length}` });
    for (const suggestion of suggestions) {
      this.renderIncrementalSuggestion(section, suggestion, snapshot);
    }
  }

  private renderIncrementalSuggestion(
    parent: HTMLElement,
    suggestion: DirectionDiffSuggestion,
    snapshot: InterestProfileReviewSnapshot,
  ): void {
    const card = parent.createEl("article", { cls: "arxiv-daily-interest-review__suggestion" });
    const heading = card.createDiv({ cls: "arxiv-daily-interest-review__suggestion-heading" });
    heading.createEl("span", {
      cls: `arxiv-daily-interest-review__suggestion-kind is-${suggestion.kind}`,
      text: suggestion.kind,
    });
    heading.createEl("strong", { text: this.incrementalSuggestionTarget(suggestion, snapshot) });
    heading.createSpan({ text: incrementalSuggestionPaperCount(suggestion) });
    card.createEl("p", {
      cls: "arxiv-daily-interest-review__suggestion-reason",
      text: truncateReason(suggestion.reason),
    });
    const actions = card.createDiv({ cls: "arxiv-daily-interest-review__suggestion-actions" });
    const apply = actions.createEl("button", {
      text: suggestion.kind === "new" ? "Convert to proposal" : "Apply",
      attr: { type: "button" },
    });
    apply.disabled = this.pending;
    apply.addEventListener("click", () => void this.applyIncrementalSuggestion(suggestion));
    const dismiss = actions.createEl("button", { text: "Ignore", attr: { type: "button" } });
    dismiss.disabled = this.pending;
    dismiss.addEventListener("click", () => void this.dismissIncrementalSuggestion(suggestion));
  }

  private incrementalSuggestionTarget(
    suggestion: DirectionDiffSuggestion,
    snapshot: InterestProfileReviewSnapshot,
  ): string {
    switch (suggestion.kind) {
      case "merge": {
        const names = suggestion.directionIds.map((id) => this.directionName(snapshot, id));
        return names.join(" + ");
      }
      case "new":
        return "New direction";
      case "attach":
      case "split":
        return this.directionName(snapshot, suggestion.directionId);
    }
  }

  private directionName(snapshot: InterestProfileReviewSnapshot, directionId: string): string {
    return snapshot.profile?.directions.find((direction) => direction.id === directionId)?.name
      ?? directionId;
  }

  private async applyIncrementalSuggestion(suggestion: DirectionDiffSuggestion): Promise<void> {
    const convertingToProposal = suggestion.kind === "new";
    await this.run("apply incremental suggestion", async () => {
      await this.controller.applySuggestion(incrementalSuggestionKey(suggestion));
      if (convertingToProposal) this.tab = "proposed";
    });
  }

  private async dismissIncrementalSuggestion(suggestion: DirectionDiffSuggestion): Promise<void> {
    await this.run("dismiss incremental suggestion", () =>
      this.controller.dismissSuggestion(incrementalSuggestionKey(suggestion)));
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

  private draft(id: string): PersonalLibraryReviewedDirectionDraft | null {
    const fields = this.fields.get(id);
    if (!fields) return null;
    const name = fields.name.value.trim();
    const description = fields.description.value.trim();
    const discoveryCues = normalizeLines(fields.cues.value);
    const representativePaperKeys = Array.from(fields.representatives.selectedOptions, (option) => option.value).sort(codeUnitCompare);
    const error = validateDraft({ name, description, discoveryCues, representativePaperKeys });
    if (error) {
      this.errorMessage = error;
      this.renderErrorOnly();
      return null;
    }
    return { name, description, discoveryCues, representativePaperKeys };
  }

  private patch(draft: PersonalLibraryReviewedDirectionDraft): PersonalLibraryDirectionTextPatch {
    return { name: draft.name, description: draft.description, discoveryCues: draft.discoveryCues };
  }

  private async generate(snapshot: InterestProfileReviewSnapshot): Promise<void> {
    if (snapshot.proposal) {
      const choice = await chooseModal(this.app, "Regenerate proposed directions", "Replace the current proposal and all unconfirmed edits with newly generated directions?", [
        { label: "Cancel", value: "cancel" },
        { label: "Regenerate", value: "regenerate", warning: true },
      ]);
      if (choice !== "regenerate" || this.closed) return;
    }
    await this.run("generate proposals", async () => {
      await this.controller.generate();
      return this.controller.reload();
    });
  }

  private saveProposal(id: string): void {
    const draft = this.draft(id);
    if (!draft) return;
    void this.run("save proposed direction", () => this.controller.updateProposal({ candidateId: id, patch: this.patch(draft), representativePaperKeys: draft.representativePaperKeys }));
  }

  private saveConfirmed(id: string): void {
    const draft = this.draft(id);
    if (!draft) return;
    void this.run("save confirmed direction", () => this.controller.updateConfirmed({ directionId: id, patch: this.patch(draft), representativePaperKeys: draft.representativePaperKeys }));
  }

  private async confirmProposal(id: string, status: "active" | "disabled"): Promise<void> {
    const draft = this.draft(id);
    if (!draft) return;
    const choice = await chooseModal(this.app, `Confirm ${status} direction`, status === "active"
      ? "Confirm this as active? It is eligible only later when its catalog evidence is compatible and current."
      : "Confirm this direction as disabled? It will remain excluded from discovery until explicitly enabled.", [
      { label: "Cancel", value: "cancel" },
      { label: status === "active" ? "Confirm active" : "Confirm disabled", value: "confirm", cta: true },
    ]);
    if (choice !== "confirm" || this.closed) return;
    await this.run(`confirm ${status} direction`, () => this.controller.confirmProposal({ candidateId: id, draft, status }));
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

  private async mergeSelectedProposals(snapshot: InterestProfileReviewSnapshot): Promise<void> {
    const ids = Array.from(this.selectedProposals).sort(codeUnitCompare);
    if (ids.length < 2) return;
    const draft = this.draft(ids[0]!);
    if (!draft) return;
    const names = ids.map((id) => snapshot.proposal?.candidates.find((candidate) => candidate.id === id)?.name)
      .filter((name): name is string => Boolean(name));
    const boundedNames = names.length === ids.length && names.every(isSafeConfirmationName);
    const subjects = boundedNames ? ` (${names.join(", ")})` : "";
    const choice = await chooseModal(
      this.app,
      "Merge proposed directions",
      `Merge ${ids.length} selected proposed directions${subjects}? This replaces the source proposals with one newly identified proposal.`,
      [
        { label: "Cancel", value: "cancel" },
        { label: "Merge proposals", value: "merge", warning: true },
      ],
    );
    if (choice !== "merge" || this.closed) return;
    await this.run("merge proposed directions", () => this.controller.mergeProposals({ sourceCandidateIds: ids, draft }));
    this.selectedProposals.clear();
  }

  private async mergeSelectedConfirmed(): Promise<void> {
    const ids = Array.from(this.selectedConfirmed).sort(codeUnitCompare);
    if (ids.length < 2) return;
    const draft = this.draft(ids[0]!);
    if (!draft) return;
    const choice = await chooseModal(this.app, "Merge confirmed directions", "Merge the selected terminal directions into one newly identified direction? Source directions become merged history.", [
      { label: "Cancel", value: "cancel" },
      { label: "Merge as disabled", value: "disabled" },
      { label: "Merge as active", value: "active", cta: true },
    ]);
    if ((choice !== "active" && choice !== "disabled") || this.closed) return;
    await this.run("merge confirmed directions", () => this.controller.mergeConfirmed({ sourceDirectionIds: ids, draft, status: choice }));
    this.selectedConfirmed.clear();
  }

  private async toggleConfirmed(direction: PersonalLibraryConfirmedDirection): Promise<void> {
    const enabling = direction.status === "disabled";
    const choice = await chooseModal(this.app, enabling ? "Enable confirmed direction" : "Disable confirmed direction", enabling
      ? "Enable this direction? Enabling can fail if its representative evidence is missing, stale, or incompatible."
      : "Disable this direction? Disabled directions are not eligible for discovery.", [
      { label: "Cancel", value: "cancel" },
      { label: enabling ? "Enable" : "Disable", value: "confirm", cta: enabling },
    ]);
    if (choice !== "confirm" || this.closed) return;
    await this.run(enabling ? "enable direction" : "disable direction", () => enabling ? this.controller.enable(direction.id) : this.controller.disable(direction.id));
  }

  private async removeConfirmed(id: string, mode: "restrict" | "cascade"): Promise<void> {
    const cascade = mode === "cascade";
    const choice = await chooseModal(this.app, cascade ? "Cascade remove merge family" : "Remove confirmed direction", cascade
      ? "Permanently remove this direction and its entire connected merge family, including retained ancestry? This stronger action cannot be undone."
      : "Remove this direction only when doing so does not break retained merge history? This cannot be undone.", [
      { label: "Cancel", value: "cancel" },
      { label: cascade ? "Cascade remove family" : "Remove", value: "remove", warning: true },
    ]);
    if (choice !== "remove" || this.closed) return;
    this.selectedConfirmed.delete(id);
    await this.run(cascade ? "cascade remove direction family" : "remove direction", () => this.controller.remove({ directionId: id, mode }));
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

export function normalizeLines(value: string): string[] {
  return Array.from(new Set(value.split(/\r?\n/u).map((line) => line.trim()).filter(Boolean))).sort(codeUnitCompare);
}

export function validateDraft(draft: PersonalLibraryReviewedDirectionDraft): string | null {
  if (!draft.name) return "Enter a direction name.";
  if (draft.name.length > PERSONAL_LIBRARY_MAX_NAME_LENGTH) return `Name must be at most ${PERSONAL_LIBRARY_MAX_NAME_LENGTH} characters.`;
  if (!draft.description) return "Enter a direction description.";
  if (draft.description.length > PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH) return `Description must be at most ${PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH} characters.`;
  if (draft.discoveryCues.length < 1 || draft.discoveryCues.length > PERSONAL_LIBRARY_MAX_DISCOVERY_CUES) return `Enter 1–${PERSONAL_LIBRARY_MAX_DISCOVERY_CUES} non-empty discovery cues.`;
  if (draft.discoveryCues.some((cue) => cue.length > PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH)) return `Each discovery cue must be at most ${PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH} characters.`;
  if (draft.representativePaperKeys.length < PERSONAL_LIBRARY_MIN_REPRESENTATIVES || draft.representativePaperKeys.length > PERSONAL_LIBRARY_MAX_REPRESENTATIVES) return `Choose ${PERSONAL_LIBRARY_MIN_REPRESENTATIVES}–${PERSONAL_LIBRARY_MAX_REPRESENTATIVES} representative papers.`;
  return null;
}

const PERSONAL_LIBRARY_TIMELINE_DISPLAY_LIMIT = 5 as const;

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

const TIMELINE_EVENT_LABELS: Record<PersonalLibraryDirectionTimelineEvent["kind"], string> = {
  created: "Created",
  edited: "Edited",
  "members-updated": "Members updated",
  merged: "Merged",
  removed: "Removed",
  locked: "Locked",
  unlocked: "Unlocked",
  split: "Split",
};

export function timelineEventLabel(kind: PersonalLibraryDirectionTimelineEvent["kind"]): string {
  return TIMELINE_EVENT_LABELS[kind];
}

export function formatTimelineTimestamp(at: string): string {
  const date = new Date(at);
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}`;
}

export function unclassifiedBufferPoolPapers(
  proposal: Pick<PersonalLibraryDirectionProposal, "catalogInputPapers" | "candidates"> | null,
): PersonalLibraryRepresentativeEvidence[] {
  if (!proposal) return [];
  const covered = new Set<string>();
  for (const candidate of proposal.candidates) {
    for (const member of candidate.clusterMembers ?? []) covered.add(member.paperKey);
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

function isSafeConfirmationName(name: string): boolean {
  return name.length > 0
    && name.length <= 60
    && Array.from(name).every((character) => {
      const code = character.charCodeAt(0);
      return code >= 32 && code !== 127;
    });
}
