import { App, Modal } from "obsidian";
import {
  PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUES,
  PERSONAL_LIBRARY_MAX_DISCOVERY_CUE_LENGTH,
  PERSONAL_LIBRARY_MAX_NAME_LENGTH,
  PERSONAL_LIBRARY_MAX_REPRESENTATIVES,
  PERSONAL_LIBRARY_MIN_REPRESENTATIVES,
  isThinEvidenceDirectionCandidate,
  matchingProposalAcceptance,
  resolveProposedTopicTarget,
  type DirectionProposalProgress,
  type DirectionDiffSuggestion,
  type PersonalLibraryCatalog,
  type PersonalLibraryClusterMember,
  type PersonalLibraryDirectionCandidate,
  type PersonalLibraryDirectionProposal,
  type PersonalLibraryDirectionTextPatch,
  type PersonalLibraryProposedTopic,
  type PersonalLibraryRepresentativeEvidence,
  type Topic,
  type LibraryDirectionPreview,
  topicNameKey,
  directionTextKey,
} from "@arxiv-daily/core";
import type { PersonalLibraryProfileSnapshot } from "../../main";
import type { LibraryConnectionStatus } from "./connection";
import { chooseModal } from "../services/modal";

export interface InterestProfileReviewSnapshot
  extends Omit<PersonalLibraryProfileSnapshot, "authorization"> {
  authorization: LibraryConnectionStatus;
}

export interface InterestProfileReviewOptions {
  /** On the first open, generate only when indexed evidence exists and no saved proposal is available. */
  generateIfMissing?: boolean;
}

export interface InterestProfileReviewController {
  snapshot(): InterestProfileReviewSnapshot;
  reload(): Promise<InterestProfileReviewSnapshot>;
  /** Refresh the folder inventory; arXiv identification sends IDs or titles to arXiv. */
  scan(): Promise<InterestProfileReviewSnapshot>;
  /**
   * Show the disclosure for what generation would send and record the grant.
   * Resolves false when the researcher declines or there is nothing to
   * disclose. Asked in front of generating, the way indexing asks in front of
   * indexing — with local embedding no other path reaches this grant, so a
   * button disabled on it would never become clickable.
   */
  authorize(): Promise<boolean>;
  /**
   * Record why an action failed. The page can only show a message safe to put
   * on screen, so without this the real reason reached nobody — not the log,
   * not the researcher — and every failure looked identical.
   */
  logError(action: string, error: unknown): void;
  generate(
    onProgress?: (progress: DirectionProposalProgress) => void,
    signal?: AbortSignal,
    options?: { regenerateRetired?: boolean },
  ): Promise<unknown>;
  updateProposal(input: {
    candidateId: string;
    patch: PersonalLibraryDirectionTextPatch;
    representativePaperKeys: string[];
  }): Promise<InterestProfileReviewSnapshot>;
  discardProposal(candidateId: string): Promise<InterestProfileReviewSnapshot>;
  renameTopic(input: { topicId: string; suggestedName: string }): Promise<InterestProfileReviewSnapshot>;
  moveDirection?(input: {
    candidateId: string;
    targetTopicId: string | null;
    suggestedName?: string;
  }): Promise<InterestProfileReviewSnapshot>;
  /** Omit candidateIds to accept every direction in the kept topics. */
  acceptTopics(topicIds: readonly string[], candidateIds?: readonly string[]): Promise<InterestProfileReviewSnapshot>;
  previewDirection?(input: { candidateId: string; text: string }): Promise<LibraryDirectionPreview>;
  openPaper?(paperKey: string): Promise<unknown>;
}

type ReviewTab = "proposed" | "overview";
type EditableDirection = PersonalLibraryDirectionCandidate;

/** The edited form of one proposed direction, before it is accepted. */
interface ReviewedDirectionDraft {
  text: string;
  discoveryCues: string[];
  representativePaperKeys: string[];
}

interface DirectionFields {
  text: HTMLTextAreaElement;
  discoveryCues: string[];
  representatives: HTMLSelectElement;
  initial: ReviewedDirectionDraft;
}

interface TopicCoverage {
  topic: PersonalLibraryProposedTopic;
  paperCount: number;
}

interface TopicDestination {
  target: Topic | null;
  error: string | null;
}

export class PersonalLibraryInterestProfileModal extends Modal {
  private tab: ReviewTab = "proposed";
  private pending = false;
  private closed = false;
  private renderVersion = 0;
  private readonly selectedTopics = new Set<string>();
  private readonly expandedTopics = new Set<string>();
  private errorMessage = "";
  private selectedProposals = new Set<string>();
  private selectedConfirmed = new Set<string>();
  /** Identity of the proposal whose selection has already been seeded. */
  private preselectedProposal: string | null = null;
  private generationProgress: DirectionProposalProgress | null = null;
  private fields = new Map<string, DirectionFields>();
  private drafts = new Map<string, ReviewedDirectionDraft>();
  private draftProposalIdentity: string | null = null;
  private previews = new Map<string, LibraryDirectionPreview>();
  private detailElements = new Map<string, HTMLDetailsElement>();
  private expandedDirections = new Set<string>();
  private optionsElement: HTMLDetailsElement | null = null;
  private generating = false;
  private acceptedFeedback = "";
  private automaticGenerationChecked = false;
  private generationRequest: AbortController | null = null;

  constructor(
    app: App,
    private readonly controller: InterestProfileReviewController,
    private readonly options: InterestProfileReviewOptions = {},
  ) {
    super(app);
  }

  onOpen(): void {
    if (this.closed) this.preselectedProposal = null;
    this.closed = false;
    this.modalEl.addClass("arxiv-daily-interest-review-modal");
    this.render();
    if (!this.automaticGenerationChecked) {
      this.automaticGenerationChecked = true;
      const snapshot = this.controller.snapshot();
      if (this.options.generateIfMissing && canGenerateMissingProposal(snapshot)) {
        void this.generate(snapshot, true);
      }
    }
  }

  onClose(): void {
    this.captureDrafts();
    this.closed = true;
    this.generationRequest?.abort("Review window closed");
    this.renderVersion += 1;
    this.contentEl.empty();
  }

  private render(): void {
    if (this.closed) return;
    this.renderVersion += 1;
    const snapshot = this.controller.snapshot();
    if (!snapshot.proposal) this.tab = "proposed";
    const identity = snapshot.proposal
      ? JSON.stringify([snapshot.proposal.scopeFingerprint, snapshot.proposal.proposalId]) : null;
    const unavailable = !snapshot.proposal && snapshot.proposalLoadError !== null
      && snapshot.authorization.kind !== "disconnected";
    if (identity === this.draftProposalIdentity || unavailable) {
      for (const [id, detail] of this.detailElements) {
        if (detail.open) this.expandedDirections.add(id);
        else this.expandedDirections.delete(id);
      }
    } else this.expandedDirections.clear();
    const optionsOpen = this.optionsElement?.open ?? false;
    this.detailElements.clear();
    if (identity === this.draftProposalIdentity || unavailable) this.captureDrafts();
    else {
      this.drafts.clear();
      this.previews.clear();
      this.acceptedFeedback = "";
    }
    if (!unavailable) this.draftProposalIdentity = identity;
    this.fields.clear();
    const pendingIds = new Set(snapshot.proposal?.topics.flatMap(({ directions }) =>
      directions.map(({ id }) => id)) ?? []);
    const processed = processedCandidateIds(snapshot);
    for (const id of this.drafts.keys()) {
      if (!unavailable && (!pendingIds.has(id) || processed.has(id))) this.drafts.delete(id);
    }
    const root = this.contentEl;
    root.empty();
    root.addClass("arxiv-daily-interest-review");
    // Width belongs on the modal box: Obsidian sizes .modal itself, so asking
    // the content element to be wide only makes it overflow and clip. The
    // modal class itself is added once in onOpen (it also sets the dialog's
    // enlarged base size); here only the narrower "summary" variant toggles.
    this.modalEl.classList.toggle("arxiv-daily-interest-review-modal--summary",
      this.tab === "proposed" && ![...pendingIds].some((id) => !processed.has(id)));
    root.createEl("h2", {
      text: this.tab === "overview" ? "Library overview" : "Topics from your library",
      attr: { id: "arxiv-daily-interest-title", tabindex: "-1" },
    });
    root.createEl("p", {
      cls: "arxiv-daily-interest-review__disclosure",
      text: snapshot.proposal
        ? "Open a topic, choose directions to follow, then add them to your research topics."
        : "Find research topics in your papers, then choose what to follow.",
    });

    const toolbar = root.createDiv({ cls: "arxiv-daily-interest-review__toolbar" });
    if (this.tab === "overview") {
      root.querySelector(".arxiv-daily-interest-review__disclosure")?.remove();
      const back = toolbar.createEl("button", { text: "Back to review", attr: { type: "button" } });
      back.disabled = this.pending;
      back.addEventListener("click", () => this.activateTab("proposed"));
    }
    const options = toolbar.createEl("details", { cls: "arxiv-daily-interest-review__options" });
    options.open = optionsOpen;
    options.createEl("summary", { text: "More options" });
    this.optionsElement = options;
    const actions = options.createDiv({ cls: "arxiv-daily-interest-review__toolbar-actions" });
    if (snapshot.proposal && this.tab === "proposed") {
      const overview = actions.createEl("button", { text: "Library overview", attr: { type: "button" } });
      overview.disabled = this.pending;
      overview.addEventListener("click", () => this.activateTab("overview"));
    }
    if ((snapshot.proposalLoadError && !needsRetiredProposalRegeneration(snapshot))
      || (snapshot.proposal && (this.tab === "overview" || !needsCoverageUpdate(snapshot)))) {
      this.renderGenerateButton(actions, snapshot);
    }
    const refresh = actions.createEl("button", {
      text: "Refresh",
      attr: { type: "button", "aria-label": "Refresh personal library directions" },
    });
    refresh.disabled = this.pending;
    refresh.addEventListener("click", () => this.reloadReview());
    options.createEl("p", {
      cls: "arxiv-daily-interest-review__hint",
      text: "Suggestions use paper titles and abstracts. Only directions you add affect your daily reports.",
    });
    if (snapshot.proposal) this.renderIncrementalNotice(options);

    const error = root.createDiv({
      cls: "arxiv-daily-interest-review__error",
      attr: { role: "alert", "aria-live": "assertive" },
    });
    error.hidden = !this.errorMessage;
    error.textContent = this.errorMessage;
    if (this.generating) {
      root.createEl("p", {
        cls: "arxiv-daily-interest-review__generation-status",
        attr: { role: "status" }, text: this.generationLabel(snapshot),
      });
    }
    if (this.acceptedFeedback) {
      root.createEl("p", {
        cls: "arxiv-daily-interest-review__feedback",
        attr: { role: "status" }, text: this.acceptedFeedback,
      });
    }

    const panel = root.createEl("section", {
      cls: "arxiv-daily-interest-review__panel",
      attr: {
        role: "region",
        id: `arxiv-daily-interest-${this.tab}-panel`,
        "aria-labelledby": "arxiv-daily-interest-title",
      },
    });
    if (this.tab === "overview") this.renderOverview(panel, snapshot);
    else this.renderProposed(panel, snapshot);
    if (this.tab === "proposed") {
      this.renderCoverage(options, snapshot);
      this.renderBufferPool(options, snapshot);
      root.appendChild(toolbar);
    }
  }

  private renderGenerateButton(
    parent: HTMLElement,
    snapshot: InterestProfileReviewSnapshot,
    primary = false,
  ): void {
    const regenerateRetired = needsRetiredProposalRegeneration(snapshot);
    const generation = generationAvailability(snapshot, regenerateRetired);
    const button = parent.createEl("button", {
      cls: "arxiv-daily-interest-review__generate",
      text: this.generationLabel(snapshot),
      attr: {
        type: "button",
        title: generation.allowed
          ? snapshot.authorization.kind === "authorized"
            ? "Suggests topics from titles and abstracts using your configured model."
            : "Asks you to confirm what leaves this device, then generates topics."
          : generation.reason,
      },
    });
    if (primary) button.addClass("mod-cta");
    button.disabled = this.pending || this.generationRequest !== null || !generation.allowed;
    button.addEventListener("click", () => void this.generate(snapshot, false, regenerateRetired));
    if (primary && !generation.allowed) {
      parent.createEl("p", { attr: { role: "status" }, text: generation.reason });
    }
    if (snapshot.authorization.kind !== "disconnected"
      && (!snapshot.catalog || proposablePaperKeys(snapshot).size === 0)) {
      const scan = parent.createEl("button", { text: "Scan library", attr: { type: "button" } });
      scan.disabled = this.pending;
      scan.addEventListener("click", () => void this.scanLibrary());
    }
  }

  private reloadReview(): void {
    this.previews.clear();
    void this.run("refresh directions", () => this.controller.reload());
  }

  private activateTab(tab: ReviewTab): void {
    this.tab = tab;
    this.errorMessage = "";
    this.render();
    if (!this.closed) {
      this.contentEl.scrollTop = 0;
      this.contentEl.querySelector<HTMLElement>("h2")?.focus({ preventScroll: true });
    }
  }

  private renderProposed(parent: HTMLElement, snapshot: InterestProfileReviewSnapshot): void {
    if (snapshot.acceptanceLoadError) {
      parent.createEl("p", {
        cls: "arxiv-daily-interest-review__document-error",
        attr: { role: "status" },
        text: acceptanceLoadMessage,
      });
    }
    const topics = topicsByCoverage(snapshot.proposal?.topics ?? []);
    const candidates = topics.flatMap(({ topic }) => topic.directions);
    const processed = processedCandidateIds(snapshot);
    const added = addedTopicIds(snapshot.proposal?.topics ?? [], processed);
    if (!snapshot.proposal) {
      if (needsRetiredProposalRegeneration(snapshot)) {
        // Reloading a retired proposal just reads the same incompatible file
        // again; only an explicit regenerate actually recovers it, and the
        // old file is preserved by the controller until that succeeds.
        const state = this.renderState(parent, "Suggestions need to be regenerated",
          "Suggestions from an older version need to be regenerated. The old file will be preserved.");
        this.renderGenerateButton(state, snapshot, true);
      } else if (snapshot.proposalLoadError) {
        const state = this.renderState(parent, "Couldn't open your suggestions", "Try loading your saved review again.");
        this.renderDocumentError(state, "Saved review", snapshot.proposalLoadError);
        const retry = state.createEl("button", { text: "Try again", cls: "mod-cta", attr: { type: "button" } });
        retry.disabled = this.pending;
        retry.addEventListener("click", () => this.reloadReview());
      } else {
        const papers = proposablePaperKeys(snapshot).size;
        const state = this.renderState(parent, "Find topics to follow", papers > 0
          ? `Use your ${papers} library ${papers === 1 ? "paper" : "papers"} to suggest research directions.`
          : "Prepare your library in settings to get started.");
        this.renderGenerateButton(state, snapshot, true);
      }
      return;
    }
    this.renderDocumentError(parent, "Saved review", snapshot.proposalLoadError);
    if (snapshot.proposal && candidates.length === 0) {
      const coveredCount = new Set(snapshot.proposal.coveredPaperKeys ?? []).size;
      const coverage = currentCoverage(snapshot);
      if (coverage.changed + coverage.unverified > 0) {
        const state = this.renderState(parent, "This review needs an update",
          "Check your library against the research topics you follow now.");
        this.renderGenerateButton(state, snapshot, true);
      } else {
        this.renderCompletion(parent, coveredCount > 0
          ? `Your current directions already cover ${coveredCount} ${coveredCount === 1 ? "paper" : "papers"}.`
          : "You can add a direction in Research topics whenever you need one.",
        "No new directions to add");
      }
      return;
    }
    // A load error can leave no proposal at all while still rendering the tab.
    if (snapshot.proposal) this.preselectProposals(snapshot.proposal, topics, candidates, added, processed);
    const allowedKeys = proposalPaperKeys(snapshot);
    // Selection is reviewed across topics; the shared action follows the list.
    let topicsParent = parent;
    let acceptBar: HTMLElement | undefined;
    const complete = candidates.length > 0 && candidates.every(({ id }) => processed.has(id))
      && !snapshot.acceptanceLoadError;
    if (complete) {
      this.renderCompletion(parent, "You can edit your directions in Research topics.");
      const reviewed = parent.createEl("details", { cls: "arxiv-daily-interest-review__reviewed" });
      this.rememberDetail(["reviewed"], reviewed);
      reviewed.createEl("summary", { text: "Reviewed topics" });
      topicsParent = reviewed;
    } else if (topics.length > 0) {
      const reviewed = this.reviewedTopics();
      const directionCount = reviewed.reduce((count, { directions }) => count + directions.length, 0);
      const blocked = acceptanceBlockReason(snapshot, reviewed);
      const bar = parent.createDiv({ cls: "arxiv-daily-interest-review__accept-bar" });
      acceptBar = bar;
      const accept = bar.createEl("button", {
        text: "Add to research topics",
        attr: {
          type: "button",
          title: "Adds the selected topics with only their checked directions to your research topics. You can edit or remove them there afterwards.",
        },
      });
      accept.addClass("mod-cta");
      accept.disabled = this.pending || reviewed.length === 0 || blocked !== null;
      accept.addEventListener("click", () => void this.acceptSelectedTopics());
      bar.createSpan({
        cls: "arxiv-daily-interest-review__selection-summary",
        attr: { role: "status" },
        text: directionCount === 0 ? "Select directions to follow."
          : `${directionCount} ${directionCount === 1 ? "direction" : "directions"} selected`,
      });
      if (blocked) {
        if (!snapshot.acceptanceLoadError) {
          bar.createSpan({ cls: "arxiv-daily-interest-review__hint", attr: { role: "status" }, text: blocked });
        }
      }
    }
    for (const [index, { topic, paperCount }] of topics.entries()) {
      const section = topicsParent.createEl("details", { cls: "arxiv-daily-interest-review__topic" });
      section.open = this.expandedTopics.has(topic.id);
      const version = this.renderVersion;
      section.addEventListener("toggle", () => {
        if (version !== this.renderVersion) return;
        if (section.open) this.expandedTopics.add(topic.id);
        else this.expandedTopics.delete(topic.id);
      });
      const heading = section.createEl("summary", { cls: "arxiv-daily-interest-review__topic-heading" });
      const isAdded = added.has(topic.id);
      const destination = proposedTopicDestination(topic, snapshot);
      const name = destination.target?.name ?? topic.suggestedName;
      const select = heading.createEl("input", { type: "checkbox" });
      select.checked = !isAdded && this.selectedTopics.has(topic.id);
      select.setAttribute("aria-label", `Accept ${name}`);
      select.disabled = this.pending || isAdded;
      if (isAdded) {
        select.setAttribute("title", "All proposed directions were already processed. Later edits and deletions in settings are preserved.");
      }
      select.addEventListener("click", (event) => event.stopPropagation());
      select.addEventListener("change", () => {
        if (select.checked) this.selectedTopics.add(topic.id);
        else this.selectedTopics.delete(topic.id);
        this.render();
      });
      heading.createEl("strong", { text: name });
      heading.createSpan({
        cls: "arxiv-daily-interest-review__topic-count",
        text: `${paperCount} ${paperCount === 1 ? "paper" : "papers"} · ${topic.directions.length} ${topic.directions.length === 1 ? "direction" : "directions"}`,
      });
      if (isAdded) {
        heading.createSpan({ cls: "arxiv-daily-interest-review__status", text: "Added" });
      } else if (destination.target) {
        heading.createSpan({ cls: "arxiv-daily-interest-review__status", text: "Add to existing topic" });
      }
      if (index >= 2) {
        heading.createSpan({ cls: "arxiv-daily-interest-review__status", text: "Optional" });
      }
      const body = section.createDiv({ cls: "arxiv-daily-interest-review__topic-body" });
      let rename: HTMLDetailsElement | undefined;
      if (destination.target) {
        body.createEl("p", {
          cls: "arxiv-daily-interest-review__hint",
          text: `${isAdded ? "Added to topic" : "Add to existing topic"}: ${destination.target.name}`,
        });
      } else if (destination.error && !isAdded) {
        body.createEl("p", {
          cls: "arxiv-daily-interest-review__document-error",
          attr: { role: "status" }, text: destination.error,
        });
      } else if (!isAdded) {
        rename = body.createEl("details", { cls: "arxiv-daily-interest-review__rename" });
        this.rememberDetail([topic.id, "rename"], rename);
        rename.createEl("summary", { text: "Rename topic" });
        const nameField = rename.createEl("label", { cls: "arxiv-daily-interest-review__field" });
        nameField.createSpan({ text: "Topic name" });
        const nameInput = nameField.createEl("input", { type: "text", value: topic.suggestedName });
        nameInput.setAttribute("aria-label", "Topic name");
        nameInput.disabled = this.pending;
        nameInput.addEventListener("change", () => {
          const next = nameInput.value.trim();
          if (!next || next === topic.suggestedName) return;
          void this.run("rename proposed topic", () =>
            this.controller.renameTopic({ topicId: topic.id, suggestedName: next }));
        });
      }
      for (const candidate of topic.directions) {
        this.renderDirectionCard(body, candidate, allowedKeys, "proposal", snapshot, topic, processed.has(candidate.id));
      }
      if (rename) body.appendChild(rename);
    }
    if (acceptBar) parent.appendChild(acceptBar);
  }

  private renderState(parent: HTMLElement, title: string, message: string): HTMLElement {
    this.contentEl.querySelector(".arxiv-daily-interest-review__disclosure")?.remove();
    const state = parent.createDiv({ cls: "arxiv-daily-interest-review__state" });
    state.createEl("h3", { text: title });
    state.createEl("p", { text: message, attr: { role: "status" } });
    return state;
  }

  private renderCompletion(parent: HTMLElement, message: string, title = "Review complete"): void {
    const completion = this.renderState(parent, title, message);
    completion.addClass("arxiv-daily-interest-review__completion");
    const done = completion.createEl("button", { cls: "mod-cta", text: "Done", attr: { type: "button" } });
    done.disabled = this.pending;
    done.addEventListener("click", () => this.close());
  }

  private renderOverview(parent: HTMLElement, snapshot: InterestProfileReviewSnapshot): void {
    this.renderDocumentError(parent, "Proposal", snapshot.proposalLoadError);
    const proposal = snapshot.proposal;
    if (!proposal) {
      parent.createEl("p", { text: "No library analysis yet." });
      return;
    }
    const indexedKeys = new Set(snapshot.indexedPapers.map(({ paperKey }) => paperKey));
    const analyzedKeys = new Set(proposal.catalogInputPapers.map(({ paperKey }) => paperKey));
    const outside = [...indexedKeys].filter((key) => !analyzedKeys.has(key)).length;
    parent.createEl("p", { text: `${analyzedKeys.size} papers analyzed on ${new Date(proposal.generatedAt).toLocaleDateString()}.${outside ? ` ${outside} currently indexed papers were not in this analysis.` : ""}` });
    const coverage = currentCoverage(snapshot);
    const processed = processedCandidateIds(snapshot);
    const rows = new Map<string, { name: string; directions: Array<{ text: string; keys: string[]; status: string }> }>();
    const append = (id: string, name: string, text: string, keys: string[], status: string) => {
      const row = rows.get(id) ?? { name, directions: [] };
      row.directions.push({ text, keys, status });
      rows.set(id, row);
    };
    for (const item of coverage.items) {
      append(item.evidence.topicId, item.topicName, item.evidence.directionText, item.evidence.paperKeys,
        item.valid ? "Current coverage" : "Direction changed; regenerate to verify");
    }
    for (const topic of proposal.topics) {
      const destination = proposedTopicDestination(topic, snapshot);
      for (const direction of topic.directions) {
        const accepted = processed.has(direction.id);
        const current = destination.target?.directions.find(({ id }) => id === direction.id)
          ?? destination.target?.directions.find(({ text }) => directionTextKey(text) === directionTextKey(direction.text));
        const unchanged = current !== undefined && directionTextKey(current.text) === directionTextKey(direction.text);
        append(destination.target?.id ?? topic.id, destination.target?.name ?? topic.suggestedName,
          direction.text, (direction.clusterMembers ?? direction.representatives).map(({ paperKey }) => paperKey),
          accepted ? unchanged ? "Accepted" : "Accepted direction changed or removed" : "Proposed");
      }
    }
    for (const [topicId, row] of rows) {
      const section = parent.createEl("section", { cls: "arxiv-daily-interest-review__overview-topic" });
      const count = new Set(row.directions.flatMap(({ keys }) => keys)).size;
      section.createEl("h3", { text: `${row.name} (${count} papers)` });
      for (const direction of row.directions) {
        const details = section.createEl("details");
        this.rememberDetail(["overview", topicId, direction.text], details);
        details.createEl("summary", { text: `${direction.text} (${direction.keys.length} papers; ${direction.status})` });
        const list = details.createEl("ul");
        for (const key of direction.keys) this.renderPaperLink(list.createEl("li"), key, snapshot);
      }
    }
    if (coverage.unverified) parent.createEl("p", { text: `${coverage.unverified} papers have unverified legacy coverage. Regenerate proposals.` });
    const uncovered = unclassifiedBufferPoolPapers(proposal);
    if (uncovered.length) {
      const details = parent.createEl("details", { cls: "arxiv-daily-interest-review__uncovered" });
      this.rememberDetail(["uncovered"], details);
      details.createEl("summary", { text: `${uncovered.length} ${uncovered.length === 1 ? "paper" : "papers"} without a direction` });
      const list = details.createEl("ul");
      for (const { paperKey } of uncovered) this.renderPaperLink(list.createEl("li"), paperKey, snapshot);
    }
  }

  private renderPaperLink(parent: HTMLElement, key: string, snapshot: InterestProfileReviewSnapshot): void {
    const title = paperTitle(snapshot, key) ?? key;
    if (!this.controller.openPaper) {
      parent.createSpan({ text: title });
      return;
    }
    const button = parent.createEl("button", {
      cls: "arxiv-daily-interest-review__paper-link", text: title,
      attr: { type: "button", title: "Open library PDF" },
    });
    button.dataset.paperKey = key;
    button.disabled = this.pending;
    button.addEventListener("click", () => {
      const document = button.ownerDocument;
      const focused = document.activeElement === button;
      let focusMoved = false;
      const trackFocus = (event: FocusEvent) => {
        if (event.target !== button && event.target !== document.body) focusMoved = true;
      };
      if (focused) document.addEventListener("focusin", trackFocus);
      const identity = this.draftProposalIdentity;
      const links = () => Array.from(this.contentEl.querySelectorAll<HTMLButtonElement>(
        ".arxiv-daily-interest-review__paper-link",
      )).filter((link) => link.dataset.paperKey === key);
      const index = links().indexOf(button);
      void this.run("open library paper", () => this.controller.openPaper!(key)).then(() => {
        // Preserve keyboard position if repainting lost it. A PDF view or a
        // user-selected control that took focus must keep it.
        if (focused && !focusMoved && !this.closed && this.draftProposalIdentity === identity
          && document.activeElement === document.body) {
          links()[index]?.focus({ preventScroll: true });
        }
      }).finally(() => document.removeEventListener("focusin", trackFocus));
    });
  }

  private renderCoverage(parent: HTMLElement, snapshot: InterestProfileReviewSnapshot): void {
    const coverage = currentCoverage(snapshot);
    if (coverage.current + coverage.changed + coverage.unverified === 0) return;
    const details = parent.createEl("details", { cls: "arxiv-daily-interest-review__coverage" });
    this.rememberDetail(["coverage"], details);
    details.createEl("summary", {
      text: `Existing coverage: ${coverage.current} current, ${coverage.changed} changed, ${coverage.unverified} unverified`,
    });
    for (const item of coverage.items) {
      const group = details.createEl("details");
      this.rememberDetail(["coverage", item.evidence.topicId, item.evidence.directionId], group);
      group.createEl("summary", {
        text: `${item.topicName}: ${item.evidence.directionText} (${item.evidence.paperKeys.length} papers${item.valid ? "" : "; direction changed"})`,
      });
      const papers = group.createEl("ul");
      for (const key of item.evidence.paperKeys) this.renderPaperLink(papers.createEl("li"), key, snapshot);
    }
    if (coverage.changed + coverage.unverified > 0) {
      details.createEl("p", { text: "Regenerate proposals to verify coverage against current directions." });
    }
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
    const reviewed = this.reviewedTopics();
    const ids = reviewed.map(({ id }) => id);
    if (ids.length === 0 || acceptanceBlockReason(this.controller.snapshot(), reviewed)) return;
    const candidateIds = reviewed.flatMap(({ directions }) => directions.map(({ id }) => id));
    this.captureDrafts();
    const edits = candidateIds.flatMap((id) => {
      const draft = this.drafts.get(id);
      return draft ? [{ id, draft }] : [];
    });
    for (const { draft } of edits) {
      const error = validateDraft(draft);
      if (error) {
        this.errorMessage = error;
        this.renderErrorOnly();
        return;
      }
    }
    const proposal = this.controller.snapshot().proposal;
    const assertSameProposal = (): void => {
      const current = this.controller.snapshot().proposal;
      if (!current || current.proposalId !== proposal?.proposalId
        || current.scopeFingerprint !== proposal.scopeFingerprint
        || current.identificationFingerprint !== proposal.identificationFingerprint) {
        throw Object.assign(new Error("The proposal changed. Review the new proposal before accepting."), { code: "conflict" });
      }
    };
    await this.run("accept proposed topics", async () => {
      for (const { id, draft } of edits) {
        assertSameProposal();
        await this.persistDraft(id, draft);
      }
      assertSameProposal();
      const before = processedCandidateIds(this.controller.snapshot());
      const result = await this.controller.acceptTopics(ids, candidateIds);
      const processed = processedCandidateIds(result);
      const count = candidateIds.filter((id) => processed.has(id) && !before.has(id)).length;
      this.acceptedFeedback = count > 0
        ? `Confirmed ${count} ${count === 1 ? "direction" : "directions"} in Research topics.` : "";
      return result;
    });
  }

  /**
   * Reads the live snapshot rather than taking a proposal, because it also
   * has to exclude processed directions even when acceptance happened after
   * the last render. The accept count and payload use the same selection.
   */
  private reviewedTopics(): PersonalLibraryProposedTopic[] {
    const snapshot = this.controller.snapshot();
    const processed = processedCandidateIds(snapshot);
    const topics = new Map(snapshot.proposal?.topics.map((topic) => [topic.id, topic]) ?? []);
    return [...this.selectedTopics].flatMap((id) => {
      const topic = topics.get(id);
      if (!topic) return [];
      const directions = topic.directions.filter((direction) =>
        this.selectedProposals.has(direction.id) && !processed.has(direction.id));
      return directions.length > 0 ? [{ ...topic, directions }] : [];
    });
  }

  private renderDirectionCard(
    parent: HTMLElement,
    direction: EditableDirection,
    allowedPaperKeys: string[],
    kind: "proposal" | "confirmed",
    snapshot: InterestProfileReviewSnapshot,
    topic: PersonalLibraryProposedTopic,
    processed: boolean,
  ): void {
    const card = parent.createEl("article", { cls: "arxiv-daily-interest-review__card" });
    const heading = card.createDiv({ cls: "arxiv-daily-interest-review__card-heading" });
    const selected = kind === "proposal" ? this.selectedProposals : this.selectedConfirmed;
    const terminal = !("status" in direction) || direction.status !== "merged";
    const selectLabel = heading.createEl("label", { cls: "arxiv-daily-interest-review__select" });
    const checkbox = selectLabel.createEl("input", { type: "checkbox" });
    checkbox.checked = !processed && selected.has(direction.id);
    checkbox.disabled = this.pending || !terminal || processed;
    checkbox.setAttribute("aria-label", kind === "proposal"
      ? `Select ${direction.text}`
      : `Select ${direction.text} for merge`);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selected.add(direction.id);
      else selected.delete(direction.id);
      this.render();
    });
    heading.createEl("strong", { text: direction.text });
    if (processed) heading.createSpan({ cls: "arxiv-daily-interest-review__status", text: "Added" });
    if (kind === "proposal" && isThinEvidenceDirectionCandidate(direction)) {
      heading.createSpan({
        text: "thin evidence",
        cls: "arxiv-daily-interest-review__status is-thin",
        attr: { title: "Only one representative paper. Select it explicitly to include it." },
      });
    }
    // Proposals are reviewed by scanning many rows and deselecting a few, so
    // the editor is collapsed behind the row rather than stacked in front of
    // it; nothing is removed, only folded away until it is wanted.
    const detail = kind === "proposal" ? this.createCardDetail(card, processed ? "Details" : "Edit") : null;
    const body = detail ?? card;
    if (detail) {
      this.rememberDetail([direction.id, "editor"], detail);
    }
    const evidence = body.createEl("details", { cls: "arxiv-daily-interest-review__evidence" });
    this.rememberDetail([direction.id, "evidence"], evidence);
    evidence.createEl("summary", {
      text: `Evidence: ${direction.representatives.length} representative ${direction.representatives.length === 1 ? "paper" : "papers"}`,
    });
    evidence.createEl("p", { text: "Based on paper titles and abstracts." });
    let representativeEditor: HTMLDetailsElement | null = null;
    if (!processed) {
      const form = body.createDiv({ cls: "arxiv-daily-interest-review__form" });
      body.insertBefore(form, evidence);
      const initial: ReviewedDirectionDraft = {
        text: direction.text, discoveryCues: [...direction.discoveryCues],
        representativePaperKeys: direction.representatives.map(({ paperKey }) => paperKey),
      };
      const draft = this.drafts.get(direction.id) ?? initial;
      const text = this.textArea(form, "Direction (one line)", draft.text, PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH, 2);
      if (this.controller.moveDirection) {
        const destination = form.createEl("details", { cls: "arxiv-daily-interest-review__destination" });
        this.rememberDetail([direction.id, "destination"], destination);
        destination.createEl("summary", { text: "Change topic" });
        this.renderDirectionDestination(destination, direction, topic, snapshot);
      }
      representativeEditor = evidence.createEl("details", { cls: "arxiv-daily-interest-review__representatives" });
      this.rememberDetail([direction.id, "representatives"], representativeEditor);
      representativeEditor.createEl("summary", { text: "Change representative papers" });
      const representatives = this.representativeSelect(representativeEditor,
        [...new Set([...allowedPaperKeys, ...draft.representativePaperKeys])].sort(codeUnitCompare),
        draft.representativePaperKeys, snapshot);
      this.fields.set(direction.id, { text, discoveryCues: [...direction.discoveryCues], representatives, initial });
      if (this.controller.previewDirection) {
        const previewArea = form.createDiv({ cls: "arxiv-daily-interest-review__preview" });
        const renderPreview = () => {
          previewArea.empty();
          const preview = this.previews.get(direction.id);
          if (!preview) return;
          if (preview.directionText !== text.value.trim().replace(/\s+/gu, " ")
            || (snapshot.arxivCategories !== undefined
              && JSON.stringify(preview.categories) !== JSON.stringify(snapshot.arxivCategories))) {
            previewArea.createEl("p", { text: "Preview is out of date. Preview matches again." });
            return;
          }
          previewArea.createEl("p", { text: `Library sample: ${preview.papers.length} papers; ${preview.papers.filter(({ matched }) => matched).length} matches.` });
          previewArea.createEl("p", { text: `Current arXiv categories: ${preview.categories.join(", ")}` });
          if (preview.missingCategories.length) previewArea.createEl("p", {
            text: `Matching sample papers outside current categories: ${preview.missingCategories.join(", ")}. Review Paper categories in settings.`,
          });
          const list = previewArea.createEl("ul");
          for (const paper of preview.papers) list.createEl("li", {
            text: `${paper.title}: ${paper.matched ? `matches ${preview.directionText}` : "not selected"}${paper.categoryCoverage === "unknown" ? "; arXiv category unknown" : ""}`,
          });
        };
        text.addEventListener("input", renderPreview);
        renderPreview();
        const previewButton = form.createEl("button", { text: "Preview matches", attr: { type: "button" } });
        previewButton.disabled = this.pending;
        previewButton.addEventListener("click", () => {
          const draft = this.draft(direction.id);
          if (!draft) return;
          this.previews.delete(direction.id);
          void this.run("preview direction matches", async () => {
            if (this.controller.snapshot().authorization.kind !== "authorized" && !await this.controller.authorize()) return;
            const result = await this.controller.previewDirection!({ candidateId: direction.id, text: draft.text });
            this.previews.set(direction.id, result);
          });
        });
      }
    }

    const hints = evidence.createDiv({ cls: "arxiv-daily-interest-review__hint" });
    hints.createEl("strong", { text: "Evidence hints" });
    hints.createEl("p", { text: "Only the direction text drives matching; these hints explain the library evidence." });
    const cues = hints.createEl("ul");
    for (const cue of direction.discoveryCues) cues.createEl("li", { text: cue });

    const list = evidence.createEl("ul");
    for (const representative of direction.representatives) {
      const title = paperTitle(snapshot, representative.paperKey);
      const item = list.createEl("li");
      if (title) this.renderPaperLink(item, representative.paperKey, snapshot);
      else item.createSpan({ text: `${representative.paperKey} — missing from current library` });
    }

    if (direction.clusterMembers && direction.clusterMembers.length > 0) {
      this.renderClusterMembers(evidence, direction, snapshot);
    }
    if (representativeEditor) evidence.appendChild(representativeEditor);
    if (processed) return;
    const actions = body.createDiv({ cls: "arxiv-daily-interest-review__card-actions" });
    const save = actions.createEl("button", { text: "Save edits", attr: { type: "button" } });
    save.disabled = this.pending || !terminal;
    save.addEventListener("click", () => void this.saveProposal(direction.id));
    this.renderProposalActions(actions, direction.id);
  }

  /** Details start collapsed; the current edit stays open through async work. */
  private createCardDetail(card: HTMLElement, label: string): HTMLDetailsElement {
    const detail = card.createEl("details", { cls: "arxiv-daily-interest-review__detail" });
    detail.createEl("summary", { text: label });
    return detail;
  }

  private rememberDetail(identity: readonly string[], detail: HTMLDetailsElement): void {
    const key = JSON.stringify(identity);
    detail.open = this.expandedDirections.has(key);
    this.detailElements.set(key, detail);
  }

  private renderDirectionDestination(
    parent: HTMLElement,
    direction: PersonalLibraryDirectionCandidate,
    topic: PersonalLibraryProposedTopic,
    snapshot: InterestProfileReviewSnapshot,
  ): void {
    if (!this.controller.moveDirection) return;
    const destination = proposedTopicDestination(topic, snapshot);
    const topics = snapshot.settingsTopics ?? [];
    const label = parent.createEl("label", { cls: "arxiv-daily-interest-review__field" });
    label.createSpan({ text: "Destination" });
    const select = label.createEl("select", { attr: { "aria-label": `Destination for ${direction.text}` } });
    // The empty value is reserved for explicitly creating a topic; settings
    // topic ids are nonempty. Keep the current unaccepted group as an option
    // so accepting a newly proposed topic never requires a Move first.
    const keepValue = "__keep-proposed-topic__";
    if (!destination.target) {
      select.createEl("option", {
        text: destination.error ? "Choose a destination" : `New topic: ${topic.suggestedName}`,
        value: keepValue,
      });
    }
    for (const existing of topics) {
      select.createEl("option", { text: existing.name, value: existing.id });
    }
    const createNew = select.createEl("option", { text: "New topic" });
    createNew.value = "";
    select.value = destination.target?.id ?? keepValue;
    select.disabled = this.pending;

    const nameLabel = parent.createEl("label", { cls: "arxiv-daily-interest-review__field" });
    nameLabel.createSpan({ text: "New topic name" });
    const name = nameLabel.createEl("input", {
      type: "text", attr: { "aria-label": "New topic name", maxlength: String(PERSONAL_LIBRARY_MAX_NAME_LENGTH) },
    });
    name.disabled = this.pending;
    const hint = parent.createEl("p", { cls: "arxiv-daily-interest-review__hint", attr: { role: "status" } });
    const move = parent.createEl("button", { text: "Move direction", attr: { type: "button" } });
    const update = (): void => {
      const createNew = select.value === "";
      const proposedName = name.value.trim();
      const duplicate = topics.some((existing) => topicNameKey(existing.name) === topicNameKey(proposedName));
      const validName = proposedName.length > 0 && proposedName.length <= PERSONAL_LIBRARY_MAX_NAME_LENGTH && !duplicate;
      nameLabel.hidden = !createNew;
      hint.hidden = !createNew || validName;
      hint.textContent = duplicate
        ? "This topic already exists. Select it in Destination to add this direction there."
        : "Enter a name for the new topic.";
      move.disabled = this.pending || select.value === keepValue
        || select.value === destination.target?.id || (createNew && !validName);
    };
    select.addEventListener("change", update);
    name.addEventListener("input", update);
    move.addEventListener("click", () => {
      if (move.disabled) return;
      const targetTopicId = select.value || null;
      const input = {
        candidateId: direction.id, targetTopicId,
        ...(targetTopicId === null ? { suggestedName: name.value.trim() } : {}),
      };
      const wasSelected = this.selectedTopics.has(topic.id);
      const wasExpanded = this.expandedTopics.has(topic.id);
      void this.run("move proposed direction", async () => {
        const next = await this.controller.moveDirection!(input);
        const movedTopic = next.proposal?.topics.find((item) => item.directions.some(({ id }) => id === direction.id));
        if (movedTopic && wasSelected) this.selectedTopics.add(movedTopic.id);
        if (movedTopic && wasExpanded) this.expandedTopics.add(movedTopic.id);
        return next;
      });
    });
    update();
  }

  private renderProposalActions(parent: HTMLElement, candidateId: string): void {
    const discard = parent.createEl("button", { text: "Discard", attr: { type: "button" } });
    discard.addClass("mod-warning");
    discard.disabled = this.pending;
    discard.addEventListener("click", () => void this.discardProposal(candidateId));
  }

  private renderClusterMembers(
    parent: HTMLElement,
    direction: PersonalLibraryDirectionCandidate,
    snapshot: InterestProfileReviewSnapshot,
  ): void {
    const members = direction.clusterMembers ?? [];
    const details = parent.createEl("details", { cls: "arxiv-daily-interest-review__cluster" });
    this.rememberDetail([direction.id, "members"], details);
    details.createEl("summary", { text: `All ${members.length} supporting papers` });
    const list = details.createEl("ul");
    for (const member of members) {
      this.renderPaperLink(list.createEl("li"), member.paperKey, snapshot);
    }
  }

  /**
   * The researcher found the full title list noisy, so this reports only how
   * many library papers no proposed direction covers (ADR 0014 §1's "remain
   * visible as uncovered evidence"). The count still says how much of the
   * library this proposal does not speak for; the titles remain available in
   * Library overview when the researcher wants to inspect them.
   */
  private renderBufferPool(parent: HTMLElement, snapshot: InterestProfileReviewSnapshot): void {
    const count = unclassifiedBufferPoolPapers(snapshot.proposal).length;
    if (count === 0) return;
    const section = parent.createDiv({ cls: "arxiv-daily-interest-review__buffer" });
    section.createEl("p", { text: bufferPoolHeading(count) });
  }

  /** Explain the available path from newly added papers to reviewed changes. */
  private renderIncrementalNotice(parent: HTMLElement): void {
    parent.createEl("p", {
      cls: "arxiv-daily-interest-review__hint",
      attr: { role: "status" },
      text: "After adding papers, rebuild the library index and regenerate proposals to review new directions. Your accepted directions stay in research settings.",
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

  private representativeSelect(
    parent: HTMLElement,
    allowed: string[],
    selected: string[],
    snapshot: InterestProfileReviewSnapshot,
  ): HTMLSelectElement {
    const label = parent.createEl("label", { cls: "arxiv-daily-interest-review__field" });
    label.createSpan({ text: `Representative papers (choose ${PERSONAL_LIBRARY_MIN_REPRESENTATIVES}–${PERSONAL_LIBRARY_MAX_REPRESENTATIVES})` });
    const select = label.createEl("select", { attr: { multiple: "", size: "5" } });
    const selectedSet = new Set(selected);
    for (const paperKey of allowed) {
      const option = select.createEl("option");
      option.value = paperKey;
      option.textContent = paperTitle(snapshot, paperKey) ?? paperKey;
      option.selected = selectedSet.has(paperKey);
    }
    select.disabled = this.pending;
    return select;
  }

  private draft(id: string): ReviewedDirectionDraft | null {
    const fields = this.fields.get(id);
    if (!fields) return null;
    const text = fields.text.value.trim().replace(/\s+/gu, " ");
    const discoveryCues = [...fields.discoveryCues];
    const representativePaperKeys = Array.from(fields.representatives.selectedOptions, (option) => option.value).sort(codeUnitCompare);
    const error = validateDraft({ text, discoveryCues, representativePaperKeys });
    if (error) {
      this.errorMessage = error;
      this.renderErrorOnly();
      return null;
    }
    return { text, discoveryCues, representativePaperKeys };
  }

  private captureDrafts(): void {
    for (const [id, fields] of this.fields) {
      const draft: ReviewedDirectionDraft = {
        text: fields.text.value.trim().replace(/\s+/gu, " "),
        discoveryCues: [...fields.discoveryCues],
        representativePaperKeys: Array.from(fields.representatives.selectedOptions, ({ value }) => value).sort(codeUnitCompare),
      };
      if (JSON.stringify(draft) !== JSON.stringify(fields.initial)) this.drafts.set(id, draft);
      else this.drafts.delete(id);
    }
  }

  private async persistDraft(id: string, draft: ReviewedDirectionDraft): Promise<void> {
    await this.controller.updateProposal({
      candidateId: id, patch: this.patch(draft), representativePaperKeys: draft.representativePaperKeys,
    });
    this.drafts.delete(id);
    this.fields.delete(id);
  }

  private patch(draft: ReviewedDirectionDraft): PersonalLibraryDirectionTextPatch {
    return { text: draft.text, discoveryCues: draft.discoveryCues };
  }

  private async generate(snapshot: InterestProfileReviewSnapshot, onlyIfMissing = false, regenerateRetired = false): Promise<void> {
    if (this.closed || this.pending || this.generationRequest) return;
    if (snapshot.proposalLoadError && !(regenerateRetired && needsRetiredProposalRegeneration(snapshot))) return;
    const request = new AbortController();
    this.generationRequest = request;
    this.render();
    try {
      // Confirm only when regenerating would discard something: an unreviewed
      // direction still awaiting a decision, or an edit that was never saved.
      // A proposal that is already fully reviewed (or has none yet) can
      // regenerate without asking.
      const processed = processedCandidateIds(snapshot);
      const hasUnreviewedDirections = snapshot.proposal?.topics.some(({ directions }) =>
        directions.some(({ id }) => !processed.has(id)));
      if (hasUnreviewedDirections || this.drafts.size > 0) {
        const choice = await chooseModal(this.app, "Regenerate proposed directions", "Replace the current proposal and all unconfirmed edits with newly generated directions?", [
          { label: "Cancel", value: "cancel" },
          { label: "Regenerate", value: "regenerate", warning: true },
        ]);
        if (choice !== "regenerate" || request.signal.aborted || this.closed) return;
      }
      // Consent is asked here rather than gating the button, so declining leaves
      // the page usable and nothing has been sent.
      if (snapshot.authorization.kind !== "authorized") {
        let granted = false;
        try {
          granted = await this.controller.authorize();
        } catch (error) {
          if (request.signal.aborted || this.closed) return;
          this.controller.logError("authorize personal library processing", error);
          this.errorMessage = safeUserError(error);
          return;
        }
        if (!granted || request.signal.aborted || this.closed) return;
      }
      // Authorization can outlive a refresh or another review window. Opening
      // suggestions must never replace a proposal that appeared while waiting.
      if (onlyIfMissing && !canGenerateMissingProposal(this.controller.snapshot())) return;
      if (regenerateRetired && !needsRetiredProposalRegeneration(this.controller.snapshot())) return;
      // Progress lands on the button that started it: this modal covers the
      // status bar, so anything reported there would be invisible here.
      this.generationProgress = null;
      this.generating = true;
      try {
        await this.run("generate proposals", async () => {
          try {
            const report = (progress: DirectionProposalProgress) => {
              if (request.signal.aborted) return;
              this.generationProgress = progress;
              this.updateGenerationLabel();
            };
            if (regenerateRetired) await this.controller.generate(report, request.signal, { regenerateRetired: true });
            else await this.controller.generate(report, request.signal);
            if (request.signal.aborted || this.closed) return;
            return await this.controller.reload();
          } catch (error) {
            if (!request.signal.aborted) throw error;
          }
        });
      } finally {
        this.generating = false;
        this.generationProgress = null;
        this.render();
      }
    } finally {
      if (this.generationRequest === request) this.generationRequest = null;
      this.render();
    }
  }

  private generationLabel(snapshot: InterestProfileReviewSnapshot): string {
    const progress = this.generationProgress;
    // The phases before any topic exists carry no counts, and they are the
    // slow ones on a real library — saying what is happening beats a count
    // that cannot move yet.
    if (progress?.phase === "reading") return "Reading the index…";
    if (progress?.phase === "grouping") return "Grouping papers…";
    if (progress?.phase === "organization") return "Organizing topics and directions…";
    if (progress) return `Generating… (${progress.completed}/${progress.total})`;
    if (this.generating) return "Generating topics…";
    if (needsRetiredProposalRegeneration(snapshot)) return "Regenerate suggestions";
    if (needsCoverageUpdate(snapshot)) return "Update suggestions";
    return snapshot.proposal ? "Generate again" : "Generate topics";
  }

  /**
   * Updates the already-rendered button without disturbing open review rows.
   */
  private updateGenerationLabel(): void {
    if (this.closed) return;
    for (const element of Array.from(this.contentEl.querySelectorAll<HTMLElement>(
      ".arxiv-daily-interest-review__generate, .arxiv-daily-interest-review__generation-status",
    ))) element.textContent = this.generationLabel(this.controller.snapshot());
  }

  private async scanLibrary(): Promise<void> {
    await this.run(
      "scan personal library",
      () => this.controller.scan(),
      "Personal library scan failed. Try again.",
    );
  }

  private saveProposal(id: string): void {
    if (processedCandidateIds(this.controller.snapshot()).has(id)) return;
    const draft = this.draft(id);
    if (!draft) return;
    void this.run("save proposed direction", () => this.persistDraft(id, draft));
  }

  private async discardProposal(id: string): Promise<void> {
    if (processedCandidateIds(this.controller.snapshot()).has(id)) return;
    const choice = await chooseModal(this.app, "Discard proposed direction", "Discard this proposed direction and its reviewed edits?", [
      { label: "Cancel", value: "cancel" },
      { label: "Discard", value: "discard", warning: true },
    ]);
    if (choice !== "discard" || this.closed || processedCandidateIds(this.controller.snapshot()).has(id)) return;
    this.selectedProposals.delete(id);
    await this.run("discard proposed direction", () => this.controller.discardProposal(id));
  }

  /**
   * Start with the two topics covering the most papers, among those not
   * fully processed. Review edits keep the researcher's choices; opening a
   * proposal again seeds only its remaining candidates.
   */
  private preselectProposals(
    proposal: PersonalLibraryDirectionProposal,
    topics: readonly TopicCoverage[],
    candidates: readonly PersonalLibraryDirectionCandidate[],
    added: ReadonlySet<string>,
    processed: ReadonlySet<string>,
  ): void {
    const available = new Set(topics.map(({ topic }) => topic.id));
    for (const id of this.selectedTopics) {
      if (!available.has(id) || added.has(id)) this.selectedTopics.delete(id);
    }
    const availableCandidates = new Set(candidates.map(({ id }) => id));
    for (const id of this.selectedProposals) {
      if (!availableCandidates.has(id) || processed.has(id)) this.selectedProposals.delete(id);
    }
    const identity = JSON.stringify([proposal.scopeFingerprint, proposal.proposalId]);
    if (this.preselectedProposal === identity) return;
    this.preselectedProposal = identity;
    this.selectedTopics.clear();
    const selectable = topics.filter(({ topic }) => !added.has(topic.id));
    for (const { topic } of selectable.slice(0, 2)) this.selectedTopics.add(topic.id);
    this.expandedTopics.clear();
    this.selectedProposals = new Set(
      candidates
        .filter((candidate) => !processed.has(candidate.id) && !isThinEvidenceDirectionCandidate(candidate))
        .map(({ id }) => id),
    );
  }

  private async run(
    action: string,
    operation: () => Promise<unknown>,
    errorFallback?: string,
  ): Promise<void> {
    if (this.pending || this.closed) return;
    this.pending = true;
    this.errorMessage = "";
    const version = ++this.renderVersion;
    this.render();
    try {
      await operation();
      if (!this.closed && version <= this.renderVersion) this.render();
    } catch (error) {
      this.controller.logError(action, error);
      if (!this.closed) {
        this.errorMessage = safeUserError(error, errorFallback);
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
  options: InterestProfileReviewOptions = {},
): PersonalLibraryInterestProfileModal {
  const modal = new PersonalLibraryInterestProfileModal(app, controller, options);
  modal.open();
  return modal;
}

/** A whole topic is Added only after every one of its candidates was processed. */
function addedTopicIds(
  proposalTopics: readonly PersonalLibraryProposedTopic[],
  processed: ReadonlySet<string>,
): Set<string> {
  return new Set(
    proposalTopics.filter((topic) => topic.directions.length > 0
      && topic.directions.every(({ id }) => processed.has(id))).map(({ id }) => id),
  );
}

function processedCandidateIds(snapshot: InterestProfileReviewSnapshot): Set<string> {
  const receipt = snapshot.proposal
    ? matchingProposalAcceptance(snapshot.proposal, snapshot.proposalAcceptance) : null;
  return new Set(receipt?.processedCandidateIds ?? []);
}

function needsCoverageUpdate(snapshot: InterestProfileReviewSnapshot): boolean {
  if (!snapshot.proposal || snapshot.proposal.topics.some(({ directions }) => directions.length > 0)) return false;
  const coverage = currentCoverage(snapshot);
  return coverage.changed + coverage.unverified > 0;
}

function currentCoverage(snapshot: InterestProfileReviewSnapshot) {
  const proposal = snapshot.proposal;
  const items = (proposal?.coverageEvidence ?? []).map((evidence) => {
    const topic = snapshot.settingsTopics?.find(({ id }) => id === evidence.topicId);
    const direction = topic?.directions.find(({ id }) => id === evidence.directionId);
    return { evidence, valid: direction?.text === evidence.directionText, topicName: topic?.name ?? "Removed topic" };
  });
  return {
    items,
    current: items.filter(({ valid }) => valid).reduce((count, { evidence }) => count + evidence.paperKeys.length, 0),
    changed: items.filter(({ valid }) => !valid).reduce((count, { evidence }) => count + evidence.paperKeys.length, 0),
    unverified: proposal?.coverageEvidence === undefined ? new Set(proposal?.coveredPaperKeys ?? []).size : 0,
  };
}

function proposedTopicDestination(
  topic: PersonalLibraryProposedTopic,
  snapshot: InterestProfileReviewSnapshot,
): TopicDestination {
  const receipt = snapshot.proposal
    ? matchingProposalAcceptance(snapshot.proposal, snapshot.proposalAcceptance) : null;
  try {
    return { target: resolveProposedTopicTarget(topic, snapshot.settingsTopics ?? [], receipt), error: null };
  } catch (error) {
    return { target: null, error: safeUserError(error) };
  }
}

const acceptanceLoadMessage = "The saved acceptance record could not be loaded. Restore the saved review state and reload before accepting directions.";

function acceptanceBlockReason(
  snapshot: InterestProfileReviewSnapshot,
  selected: readonly PersonalLibraryProposedTopic[],
): string | null {
  if (snapshot.acceptanceLoadError) return acceptanceLoadMessage;
  for (const topic of selected) {
    const destination = proposedTopicDestination(topic, snapshot);
    if (destination.error) return destination.error;
  }
  return null;
}

function topicsByCoverage(topics: readonly PersonalLibraryProposedTopic[]): TopicCoverage[] {
  return topics.map((topic) => ({
    topic,
    paperCount: new Set(topic.directions.flatMap((direction) =>
      (direction.clusterMembers ?? []).map(({ paperKey }) => paperKey),
    )).size,
  })).sort((left, right) => right.paperCount - left.paperCount);
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
  return `No proposed direction covers ${count} ${count === 1 ? "paper" : "papers"} in your library.`;
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
  proposal: Pick<PersonalLibraryDirectionProposal, "catalogInputPapers" | "topics" | "coveredPaperKeys"> | null,
): PersonalLibraryRepresentativeEvidence[] {
  if (!proposal) return [];
  const covered = new Set(proposal.coveredPaperKeys ?? []);
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

function canGenerateMissingProposal(snapshot: InterestProfileReviewSnapshot): boolean {
  return !snapshot.proposal && !snapshot.proposalLoadError
    && snapshot.indexedPapers.length > 0 && generationAvailability(snapshot).allowed;
}

function needsRetiredProposalRegeneration(snapshot: InterestProfileReviewSnapshot): boolean {
  return !snapshot.proposal && snapshot.proposalLoadError?.code === "regeneration-required";
}

function generationAvailability(snapshot: InterestProfileReviewSnapshot, regenerateRetired = false): { allowed: boolean; reason: string } {
  if (snapshot.proposalLoadError && !(regenerateRetired && needsRetiredProposalRegeneration(snapshot))) {
    return { allowed: false, reason: "The saved suggestions could not be loaded. Refresh or resolve the file error before generating." };
  }
  // Being unauthorized is deliberately not a reason to disable: the grant is
  // asked for on the first click. Disabling on it stranded every local-embedding
  // library, because no other path in the plugin asks for that grant.
  if (snapshot.authorization.kind === "disconnected") {
    return { allowed: false, reason: "Choose a personal library folder in settings first." };
  }
  if (!snapshot.catalog) return { allowed: false, reason: snapshot.catalogLoadError?.message ? `Load the current catalog first: ${snapshot.catalogLoadError.message}` : "Scan and load the current personal-library catalog first." };
  // A library of files the scan could not identify has an empty catalog and a
  // full index; those papers are proposable on the title and abstract the
  // index read, so counting only the catalog would disable the button forever.
  if (proposablePaperKeys(snapshot).size === 0) {
    return { allowed: false, reason: "The current library has no indexed metadata-and-abstract papers to propose from." };
  }
  return { allowed: true, reason: "" };
}

/** Every paper a proposal may draw on, however it was identified. */
function proposablePaperKeys(snapshot: InterestProfileReviewSnapshot): Set<string> {
  const keys = new Set(catalogPaperKeys(snapshot.catalog));
  for (const { paperKey } of snapshot.indexedPapers) keys.add(paperKey);
  return keys;
}

/** Title for display, from whichever source knows the paper. */
export function paperTitle(
  snapshot: InterestProfileReviewSnapshot,
  paperKey: string,
): string | undefined {
  return snapshot.catalog?.papers[paperKey]?.title
    ?? snapshot.indexedPapers.find((paper) => paper.paperKey === paperKey)?.title;
}

function proposalPaperKeys(snapshot: InterestProfileReviewSnapshot): string[] {
  const manifest = snapshot.proposal?.catalogInputPapers.map((item) => item.paperKey) ?? [];
  const current = proposablePaperKeys(snapshot);
  const covered = new Set(snapshot.proposal?.coveredPaperKeys ?? []);
  return manifest.filter((key) => current.has(key) && !covered.has(key)).sort(codeUnitCompare);
}

function catalogPaperKeys(catalog: PersonalLibraryCatalog | null): string[] {
  return catalog ? Object.keys(catalog.papers).sort(codeUnitCompare) : [];
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

export function safeUserError(error: unknown, fallback = "Operation failed. Refresh and try again."): string {
  const code = error && typeof error === "object" && "code" in error
    && typeof (error as { code?: unknown }).code === "string"
    ? (error as { code: string }).code
    : "";
  const messages: Record<string, string> = {
    "invalid-input": "The reviewed direction is invalid. Check its fields and try again.",
    "invalid-document": "The saved review data is invalid. Refresh before trying again.",
    "incompatible-catalog": "The current catalog is not compatible with this review. Refresh the library first.",
    "not-found": "That direction no longer exists. Refresh and try again.",
    "target-missing": "The destination topic was removed. Choose a destination for each remaining direction before accepting it.",
    "target-ambiguous": "Several topics have this name. Choose a destination for each remaining direction before accepting it.",
    conflict: "The review changed elsewhere. Refresh before trying again.",
    stale: "The review changed elsewhere. Refresh before trying again.",
    "partial-confirmation-conflict": "The review changed while saving. Refresh before trying again.",
    "lineage-limit": "These directions have too much merge history to combine.",
    "direction-limit": "The confirmed direction limit has been reached.",
    "merge-relationship": "These directions cannot be changed without breaking merge history.",
    "evidence-mismatch": "Representative evidence is missing or stale. Refresh the catalog and review it again.",
    "authorization-terms-changed": "What would be sent changed while the disclosure was open. Refresh and review it again.",
    "authorization-superseded": "The library changed while authorizing. Refresh and try again.",
    "authorization-not-recorded": "The authorization was not recorded. Refresh and try again.",
    "catalog-invalid": "The current catalog is invalid. Refresh or rescan the library.",
    "no-evidence": "The current catalog has no eligible metadata-and-abstract evidence.",
    "evidence-too-large": "The selected catalog evidence is too large to process safely.",
    "synthesis-too-large": "The proposed direction synthesis is too large. Reduce the library selection and retry.",
    "output-too-large": "The model response was too large. Retry generation.",
    "proposal-invariant": "The generated proposal was invalid. Retry generation.",
  };
  return messages[code] ?? fallback;
}
