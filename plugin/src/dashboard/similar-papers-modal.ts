import { Modal, setIcon, type App } from "obsidian";
import type { PaperIndexEntry, PaperSearchResult } from "@arxiv-daily/core";

export interface SimilarPapersModalCallbacks {
  openDetail(entry: PaperIndexEntry): void | Promise<void>;
  openDaily(entry: PaperIndexEntry): void | Promise<void>;
  openArxiv(entry: PaperIndexEntry): void | Promise<void>;
  openPdf(entry: PaperIndexEntry): void | Promise<void>;
  onActionError?(error: unknown, action: string, entry: PaperIndexEntry): void;
}

export interface SimilarPapersModalOptions extends SimilarPapersModalCallbacks {
  source: PaperIndexEntry;
  results: readonly PaperSearchResult[];
}

export class SimilarPapersModal extends Modal {
  constructor(app: App, private readonly options: SimilarPapersModalOptions) {
    super(app);
  }

  onOpen(): void {
    renderSimilarPapersModal(this.contentEl, this.options);
  }

  onClose(): void {
    this.contentEl.empty();
  }
}

export function renderSimilarPapersModal(
  contentEl: HTMLElement,
  options: SimilarPapersModalOptions,
): void {
  contentEl.empty();
  contentEl.addClass("arxiv-daily-similar-modal");
  contentEl.createEl("h2", { text: "Similar papers" });
  contentEl.createEl("p", {
    cls: "arxiv-daily-similar-modal__source",
    text: `Based on ${options.source.arxivId} · ${options.source.title}`,
  });
  if (options.results.length === 0) {
    contentEl.createEl("p", {
      cls: "arxiv-daily-similar-modal__empty",
      text: "No similar papers were found in the local paper index.",
    });
    return;
  }

  const list = contentEl.createEl("ol", {
    cls: "arxiv-daily-similar-modal__list",
    attr: { "aria-label": "Similar local papers" },
  });
  for (const result of options.results.slice(0, 10)) {
    const item = list.createEl("li", { cls: "arxiv-daily-similar-modal__item" });
    item.createEl("div", {
      cls: "arxiv-daily-similar-modal__title",
      text: result.entry.title,
    });
    item.createEl("div", {
      cls: "arxiv-daily-similar-modal__meta",
      text: `${result.entry.arxivId} · ${result.entry.authors.join(", ") || "Unknown authors"}`,
    });
    item.createEl("div", {
      cls: "arxiv-daily-similar-modal__context",
      text: [
        result.entry.primaryTopic || result.entry.category || "Uncategorized",
        result.entry.published || result.entry.updated || "Date unavailable",
        resourceSummary(result.entry),
      ].join(" · "),
    });
    const reason = result.reasons.slice(0, 2).map((value) => value.text).join(" · ");
    item.createEl("div", {
      cls: "arxiv-daily-similar-modal__reason",
      text: reason || "Shared indexed terms",
    });
    const actions = item.createEl("div", {
      cls: "arxiv-daily-similar-modal__actions",
      attr: { "aria-label": `Actions for ${result.entry.arxivId}` },
    });
    addAction(actions, "file-text", "Open detail", result.entry, options.openDetail, options.onActionError, Boolean(result.entry.detail && result.entry.paperPath));
    addAction(actions, "calendar", "Open daily report", result.entry, options.openDaily, options.onActionError, result.entry.dailyReports.length > 0);
    addAction(actions, "external-link", "Open arXiv", result.entry, options.openArxiv, options.onActionError);
    addAction(actions, "file-down", "Open PDF", result.entry, options.openPdf, options.onActionError);
  }
}

function resourceSummary(entry: PaperIndexEntry): string {
  const resources = [
    entry.detail && entry.paperPath ? "Detail available" : "No detail",
    entry.dailyReports.length > 0 ? "Daily report available" : "No daily report",
    entry.pdfPath ? "PDF saved" : "PDF online",
  ];
  return resources.join(" · ");
}

function addAction(
  parent: HTMLElement,
  icon: string,
  label: string,
  entry: PaperIndexEntry,
  callback: (entry: PaperIndexEntry) => void | Promise<void>,
  onError?: (error: unknown, action: string, entry: PaperIndexEntry) => void,
  available = true,
): void {
  const unavailableLabel = `${label} unavailable`;
  const button = parent.createEl("button", {
    cls: "clickable-icon",
    attr: {
      type: "button",
      "aria-label": available ? label : unavailableLabel,
      title: available ? label : unavailableLabel,
      ...(available ? {} : { disabled: "", "aria-disabled": "true" }),
    },
  });
  setIcon(button, icon);
  if (!available) return;
  button.addEventListener("click", () => {
    try {
      Promise.resolve(callback(entry)).catch((error) => reportActionError(onError, error, label, entry));
    } catch (error) {
      reportActionError(onError, error, label, entry);
    }
  });
}

function reportActionError(
  onError: SimilarPapersModalCallbacks["onActionError"],
  error: unknown,
  action: string,
  entry: PaperIndexEntry,
): void {
  try {
    onError?.(error, action, entry);
  } catch {
    // Action failures must not escape modal event handlers.
  }
}
