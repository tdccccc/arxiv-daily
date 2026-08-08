import { Modal, setIcon, type App } from "obsidian";
import type {
  KnowledgeBaseChunkHit,
  PaperIndexEntry,
  PaperSearchResult,
} from "@arxiv-daily/core";

export interface LibrarySimilarMatch {
  paperKey: string;
  title: string;
  score: number;
  hits: readonly KnowledgeBaseChunkHit[];
}

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
  /**
   * Optional library full-text similarity: enables the "Library" tab, which
   * loads its matches asynchronously on first open. Omit to render only the
   * existing daily-report similarity list.
   */
  library?: {
    query: string;
    load: () => Promise<readonly LibrarySimilarMatch[]>;
  };
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

  const library = options.library;
  if (!library) {
    renderDailyPanel(contentEl, options);
    return;
  }

  // Tab bar: library full-text similarity (default) + daily-report lexical
  // similarity. The library tab loads asynchronously on first open.
  const tabs = contentEl.createDiv({
    cls: "arxiv-daily-similar-modal__tabs",
    attr: { role: "tablist", "aria-label": "Similar papers sections" },
  });
  const libraryButton = tabs.createEl("button", {
    cls: "arxiv-daily-similar-modal__tab",
    text: "Library similar",
    attr: { type: "button", role: "tab", "aria-selected": "true" },
  });
  const dailyButton = tabs.createEl("button", {
    cls: "arxiv-daily-similar-modal__tab",
    text: "Daily similar",
    attr: { type: "button", role: "tab", "aria-selected": "false" },
  });
  const libraryPanel = contentEl.createDiv({
    cls: "arxiv-daily-similar-modal__panel",
    attr: { role: "tabpanel" },
  });
  const dailyPanel = contentEl.createDiv({
    cls: "arxiv-daily-similar-modal__panel",
    attr: { role: "tabpanel", hidden: "" },
  });

  let libraryLoaded = false;
  const select = (tab: "library" | "daily"): void => {
    const isLibrary = tab === "library";
    libraryButton.setAttribute("aria-selected", String(isLibrary));
    dailyButton.setAttribute("aria-selected", String(!isLibrary));
    libraryPanel.toggleAttribute("hidden", !isLibrary);
    dailyPanel.toggleAttribute("hidden", isLibrary);
    if (isLibrary && !libraryLoaded) {
      libraryLoaded = true;
      void loadLibraryPanel(libraryPanel, library);
    }
  };
  libraryButton.addEventListener("click", () => select("library"));
  dailyButton.addEventListener("click", () => select("daily"));
  select("library");

  renderDailyPanel(dailyPanel, options);
}

async function loadLibraryPanel(
  panel: HTMLElement,
  library: NonNullable<SimilarPapersModalOptions["library"]>,
): Promise<void> {
  panel.empty();
  panel.createEl("p", {
    cls: "arxiv-daily-similar-modal__status",
    text: "Searching your library…",
  });
  try {
    const matches = await library.load();
    panel.empty();
    if (matches.length === 0) {
      panel.createEl("p", {
        cls: "arxiv-daily-similar-modal__empty",
        text: "No similar papers found in your library.",
      });
      return;
    }
    const list = panel.createEl("ol", {
      cls: "arxiv-daily-similar-modal__list",
      attr: { "aria-label": "Similar library papers" },
    });
    for (const match of matches.slice(0, 10)) {
      const item = list.createEl("li", { cls: "arxiv-daily-similar-modal__item" });
      item.createDiv({
        cls: "arxiv-daily-similar-modal__title",
        text: match.title,
      });
      item.createDiv({
        cls: "arxiv-daily-similar-modal__meta",
        text: `${match.paperKey} · similarity ${match.score.toFixed(3)}`,
      });
      for (const hit of match.hits.slice(0, 2)) {
        item.createDiv({
          cls: "arxiv-daily-similar-modal__hit",
          text: `p.${hit.page} · ${hit.text.slice(0, 160).replace(/\s+/g, " ")}…`,
        });
      }
    }
  } catch (error) {
    panel.empty();
    panel.createEl("p", {
      cls: "arxiv-daily-similar-modal__error",
      text: `Library full-text search unavailable: ${describeError(error)}`,
    });
  }
}

function renderDailyPanel(
  panel: HTMLElement,
  options: SimilarPapersModalOptions,
): void {
  if (options.results.length === 0) {
    panel.createEl("p", {
      cls: "arxiv-daily-similar-modal__empty",
      text: "No similar papers were found in the local paper index.",
    });
    return;
  }

  const list = panel.createEl("ol", {
    cls: "arxiv-daily-similar-modal__list",
    attr: { "aria-label": "Similar local papers" },
  });
  for (const result of options.results.slice(0, 10)) {
    const item = list.createEl("li", { cls: "arxiv-daily-similar-modal__item" });
    item.createDiv({
      cls: "arxiv-daily-similar-modal__title",
      text: result.entry.title,
    });
    item.createDiv({
      cls: "arxiv-daily-similar-modal__meta",
      text: `${result.entry.arxivId} · ${result.entry.authors.join(", ") || "Unknown authors"}`,
    });
    item.createDiv({
      cls: "arxiv-daily-similar-modal__context",
      text: [
        result.entry.primaryTopic || result.entry.category || "Uncategorized",
        result.entry.published || result.entry.updated || "Date unavailable",
        resourceSummary(result.entry),
      ].join(" · "),
    });
    const reason = result.reasons.slice(0, 2).map((value) => value.text).join(" · ");
    item.createDiv({
      cls: "arxiv-daily-similar-modal__reason",
      text: reason || "Shared indexed terms",
    });
    const actions = item.createDiv({
      cls: "arxiv-daily-similar-modal__actions",
      attr: { "aria-label": `Actions for ${result.entry.arxivId}` },
    });
    const openDetail = (entry: PaperIndexEntry) => options.openDetail(entry);
    const openDaily = (entry: PaperIndexEntry) => options.openDaily(entry);
    const openArxiv = (entry: PaperIndexEntry) => options.openArxiv(entry);
    const openPdf = (entry: PaperIndexEntry) => options.openPdf(entry);
    const onActionError = options.onActionError
      ? (error: unknown, action: string, entry: PaperIndexEntry) =>
          options.onActionError?.(error, action, entry)
      : undefined;
    addAction(actions, "file-text", "Open detail", result.entry, openDetail, onActionError, Boolean(result.entry.detail && result.entry.paperPath));
    addAction(actions, "calendar", "Open daily report", result.entry, openDaily, onActionError, result.entry.dailyReports.length > 0);
    addAction(actions, "external-link", "Open arXiv", result.entry, openArxiv, onActionError);
    addAction(actions, "file-down", "Open PDF", result.entry, openPdf, onActionError);
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

function describeError(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error);
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
