/**
 * Dashboard "Library matches" block: the second result surface of the single
 * search box (ADR 0006). The dashboard's own search box keeps filtering the
 * daily-report rows lexically; this block renders the async full-text
 * knowledge-base matches for the same query. Pure rendering — the view owns
 * the async orchestration (debounce, staleness token) and calls this with a
 * state snapshot.
 */

export interface LibrarySearchMatch {
  paperKey: string;
  title: string;
  score: number;
  /** Relative library path for fallback-indexed files; unset for arXiv papers. */
  filePath?: string;
}

export type LibrarySearchState =
  | { kind: "loading" }
  | { kind: "matches"; matches: readonly LibrarySearchMatch[] }
  | { kind: "empty" }
  | { kind: "error"; message: string };

export function renderLibrarySearchBlock(
  container: HTMLElement,
  state: LibrarySearchState,
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
    item.createDiv({
      cls: "arxiv-daily-dashboard__library-title",
      text: match.title,
    });
    item.createDiv({
      cls: "arxiv-daily-dashboard__library-meta",
      text: `${match.filePath ?? match.paperKey} · similarity ${match.score.toFixed(3)}`,
    });
  }
}
