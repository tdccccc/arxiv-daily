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
  for (const result of options.results) {
    const item = list.createEl("li", { cls: "arxiv-daily-similar-modal__item" });
    item.createEl("div", {
      cls: "arxiv-daily-similar-modal__title",
      text: result.entry.title,
    });
    item.createEl("div", {
      cls: "arxiv-daily-similar-modal__meta",
      text: `${result.entry.arxivId} · ${result.entry.authors.join(", ") || "Unknown authors"}`,
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
    addAction(actions, "file-text", "Open detail", result.entry, options.openDetail, options.onActionError);
    addAction(actions, "calendar", "Open daily report", result.entry, options.openDaily, options.onActionError);
    addAction(actions, "external-link", "Open arXiv", result.entry, options.openArxiv, options.onActionError);
    addAction(actions, "file-down", "Open PDF", result.entry, options.openPdf, options.onActionError);
  }
}

function addAction(
  parent: HTMLElement,
  icon: string,
  label: string,
  entry: PaperIndexEntry,
  callback: (entry: PaperIndexEntry) => void | Promise<void>,
  onError?: (error: unknown, action: string, entry: PaperIndexEntry) => void,
): void {
  const button = parent.createEl("button", {
    cls: "clickable-icon",
    attr: { type: "button", "aria-label": label, title: label },
  });
  setIcon(button, icon);
  button.addEventListener("click", () => {
    try {
      Promise.resolve(callback(entry)).catch((error) =>
        onError?.(error, label, entry),
      );
    } catch (error) {
      onError?.(error, label, entry);
    }
  });
}
