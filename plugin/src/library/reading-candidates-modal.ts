import { App, Modal, setIcon } from "obsidian";
import {
  personalNoveltyDifferenceTypeLabel,
  type ReadingCandidateDecisionKind,
  type ReadingCandidateRecord,
  type ReadingCandidatesDocument,
} from "@arxiv-daily/core";

export interface ReadingCandidatesModalOptions {
  getCandidates: () => ReadingCandidatesDocument | null;
  decide: (paperKey: string, kind: ReadingCandidateDecisionKind) => Promise<boolean>;
  remove: (paperKey: string) => Promise<boolean>;
  onError?: (action: string, error: unknown) => void;
}

export class ReadingCandidatesModal extends Modal {
  constructor(app: App, private readonly options: ReadingCandidatesModalOptions) {
    super(app);
  }

  onOpen(): void {
    renderReadingCandidatesModal(this.contentEl, this.options);
  }

  onClose(): void {
    this.contentEl.empty();
  }
}

export function renderReadingCandidatesModal(
  contentEl: HTMLElement,
  options: ReadingCandidatesModalOptions,
): void {
  contentEl.empty();
  contentEl.addClass("arxiv-daily-reading-candidates-modal");
  contentEl.createEl("h2", { text: "Reading candidates" });

  const document = options.getCandidates();
  const records = Object.values(document?.candidates ?? {});
  const pending = records.filter((record) => !record.decision);
  const decided = records.length - pending.length;
  contentEl.createEl("p", {
    cls: "arxiv-daily-reading-candidates-modal__counts",
    text: `${pending.length} pending · ${decided} decided`,
  });

  if (pending.length === 0) {
    contentEl.createEl("p", {
      cls: "arxiv-daily-reading-candidates-modal__empty",
      text: "No reading candidates yet. Save discovered papers from the dashboard to review them here.",
    });
    return;
  }

  const refresh = () => renderReadingCandidatesModal(contentEl, options);
  for (const group of groupPendingCandidates(pending)) {
    contentEl.createEl("h3", {
      cls: "arxiv-daily-reading-candidates-modal__group",
      text: group.label,
    });
    for (const candidate of group.candidates) {
      renderCandidateRow(contentEl, candidate, options, refresh);
    }
  }
}

export function groupPendingCandidates(
  candidates: readonly ReadingCandidateRecord[],
): Array<{ label: string; candidates: ReadingCandidateRecord[] }> {
  const groups = new Map<string, { label: string; candidates: ReadingCandidateRecord[] }>();
  for (const candidate of candidates) {
    const direction = candidate.source.directions[0];
    const topic = candidate.source.manualTopics[0];
    const key = direction ? `direction:${direction.id}` : topic ? `topic:${topic.tag}` : "other";
    const label = direction
      ? direction.name
      : topic
        ? topic.name ?? topic.tag
        : "Other";
    const group = groups.get(key) ?? { label, candidates: [] };
    group.candidates.push(candidate);
    groups.set(key, group);
  }
  return [...groups.values()]
    .sort((left, right) => left.label.localeCompare(right.label))
    .map((group) => ({
      label: group.label,
      candidates: group.candidates.sort((left, right) => {
        const bySaved = right.savedAt.localeCompare(left.savedAt);
        return bySaved !== 0 ? bySaved : left.paperKey.localeCompare(right.paperKey);
      }),
    }));
}

function renderCandidateRow(
  container: HTMLElement,
  candidate: ReadingCandidateRecord,
  options: ReadingCandidatesModalOptions,
  refresh: () => void,
): void {
  const row = container.createDiv({
    cls: "arxiv-daily-reading-candidates-modal__row",
  });
  row.createDiv({
    cls: "arxiv-daily-reading-candidates-modal__title",
    text: candidate.title,
  });
  row.createDiv({
    cls: "arxiv-daily-reading-candidates-modal__meta",
    text: [
      candidate.authors,
      candidate.topic,
      `Report ${candidate.source.reportDate}`,
    ].filter(Boolean).join(" · "),
  });
  const priorWorks = candidate.relatedPriorWorks
    .map((work) => work.title)
    .filter(Boolean);
  if (priorWorks.length > 0) {
    row.createDiv({
      cls: "arxiv-daily-reading-candidates-modal__related",
      text: `Related: ${priorWorks.join("; ")}`,
    });
  }
  if (candidate.provisionalNovelty) {
    row.createDiv({
      cls: "arxiv-daily-reading-candidates-modal__novelty",
      text: [
        personalNoveltyDifferenceTypeLabel(candidate.provisionalNovelty.differenceType),
        candidate.provisionalNovelty.explanation,
      ].join(" — "),
    });
  }

  const actions = row.createDiv({
    cls: "arxiv-daily-reading-candidates-modal__actions",
  });
  const decisions: Array<{ kind: ReadingCandidateDecisionKind; label: string; icon: string }> = [
    { kind: "read-closely", label: "Read closely", icon: "book-open" },
    { kind: "skim", label: "Skim", icon: "scan-text" },
    { kind: "dismiss", label: "Dismiss", icon: "check" },
  ];
  for (const decision of decisions) {
    const button = actions.createEl("button", {
      cls: "arxiv-daily-reading-candidates-modal__action",
      attr: { type: "button", "aria-label": `${decision.label} ${candidate.title}` },
    });
    setIcon(button, decision.icon);
    button.createSpan({ text: decision.label });
    button.addEventListener("click", () => {
      void (async () => {
        button.disabled = true;
        try {
          await options.decide(candidate.paperKey, decision.kind);
          refresh();
        } catch (error) {
          options.onError?.(`decide ${decision.kind}`, error);
          button.disabled = false;
        }
      })();
    });
  }
  const removeButton = actions.createEl("button", {
    cls: "arxiv-daily-reading-candidates-modal__remove",
    attr: { type: "button", "aria-label": `Remove ${candidate.title}` },
  });
  setIcon(removeButton, "trash");
  removeButton.addEventListener("click", () => {
    void (async () => {
      removeButton.disabled = true;
      try {
        await options.remove(candidate.paperKey);
        refresh();
      } catch (error) {
        options.onError?.("remove", error);
        removeButton.disabled = false;
      }
    })();
  });
}
