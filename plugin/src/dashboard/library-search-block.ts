import { setIcon } from "obsidian";
import type { LibraryFullTextMatch } from "../library/fulltext-results";

/**
 * Dashboard "Library matches" block: the second result surface of the single
 * search box (ADR 0006). The dashboard's own search box keeps filtering the
 * daily-report rows lexically; this block renders the async full-text
 * knowledge-base matches for the same query. Pure rendering — the view owns
 * the async orchestration (debounce, staleness token) and calls this with a
 * state snapshot.
 *
 * Passage hits may exist on the match objects; this surface lists papers and
 * opens the PDF as a whole until passage quality has a measured bar.
 */

export type LibrarySearchMatch = LibraryFullTextMatch;

export interface LibrarySearchBlockOptions {
  openLibraryPdf?: (match: LibrarySearchMatch) => void | Promise<void>;
  onActionError?: (error: unknown, action: string) => void;
}

export type LibrarySearchState =
  | { kind: "loading" }
  | { kind: "matches"; matches: readonly LibrarySearchMatch[] }
  | { kind: "empty" }
  | { kind: "error"; message: string };

export function renderLibrarySearchBlock(
  container: HTMLElement,
  state: LibrarySearchState,
  options: LibrarySearchBlockOptions = {},
): void {
  container.empty();
  container.addClass("arxiv-daily-dashboard__library-results");
  if (state.kind === "loading") {
    container.createDiv({
      cls: "arxiv-daily-dashboard__library-status",
      text: "Searching your library…",
    });
    return;
  }
  container.createDiv({
    cls: "arxiv-daily-dashboard__library-heading",
    text: "Library matches",
  });
  if (state.kind === "empty") {
    container.createDiv({
      cls: "arxiv-daily-dashboard__library-empty",
      text: "No library matches for this query.",
    });
    return;
  }
  if (state.kind === "error") {
    container.createDiv({
      cls: "arxiv-daily-dashboard__library-error",
      text: `Library search unavailable: ${state.message}`,
    });
    return;
  }
  const list = container.createEl("ol", {
    cls: "arxiv-daily-dashboard__library-list",
    attr: { "aria-label": "Library matches" },
  });
  for (const match of state.matches) {
    const item = list.createEl("li", {
      cls: "arxiv-daily-dashboard__library-item",
    });
    const header = item.createDiv({
      cls: "arxiv-daily-dashboard__library-header",
    });
    header.createDiv({
      cls: "arxiv-daily-dashboard__library-title",
      text: match.title,
    });
    if (match.filePath && options.openLibraryPdf) {
      const action = "Open PDF";
      const button = header.createEl("button", {
        cls: "clickable-icon arxiv-daily-dashboard__library-open",
        attr: { type: "button", "aria-label": action, title: action },
      });
      setIcon(button, "file-down");
      button.addEventListener("click", () => {
        try {
          void Promise.resolve(options.openLibraryPdf?.(match)).catch((error) => {
            reportActionError(options, error, action);
          });
        } catch (error) {
          reportActionError(options, error, action);
        }
      });
    }
    item.createDiv({
      cls: "arxiv-daily-dashboard__library-meta",
      text: match.filePath ?? match.paperKey,
    });
  }
}

function reportActionError(
  options: LibrarySearchBlockOptions,
  error: unknown,
  action: string,
): void {
  try {
    options.onActionError?.(error, action);
  } catch {
    // Action failures must not escape DOM event handlers.
  }
}
