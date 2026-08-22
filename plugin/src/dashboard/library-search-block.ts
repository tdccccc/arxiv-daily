import type { KnowledgeBaseChunkHit } from "@arxiv-daily/core";
import type { LibraryFullTextMatch } from "../library/fulltext-results";

/**
 * Dashboard "Library matches" block: the second result surface of the single
 * search box (ADR 0006). The dashboard's own search box keeps filtering the
 * daily-report rows lexically; this block renders the async full-text
 * knowledge-base matches for the same query. Pure rendering — the view owns
 * the async orchestration (debounce, staleness token) and calls this with a
 * state snapshot.
 */

export type LibrarySearchMatch = LibraryFullTextMatch;

export interface LibrarySearchBlockOptions {
  openEvidence?: (
    match: LibrarySearchMatch,
    hit: KnowledgeBaseChunkHit,
  ) => void | Promise<void>;
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
    item.createDiv({
      cls: "arxiv-daily-dashboard__library-title",
      text: match.title,
    });
    item.createDiv({
      cls: "arxiv-daily-dashboard__library-meta",
      text: `${match.filePath ?? match.paperKey} · ${formatEvidenceScore(match)}`,
    });
    for (const hit of match.hits) {
      renderEvidence(item, match, hit, options);
    }
  }
}

export function formatEvidenceScore(match: Pick<LibrarySearchMatch, "score" | "scoreKind">): string {
  return match.scoreKind === "cosine"
    ? `best semantic evidence ${match.score.toFixed(3)}`
    : "lexical match";
}

function renderEvidence(
  parent: HTMLElement,
  match: LibrarySearchMatch,
  hit: KnowledgeBaseChunkHit,
  options: LibrarySearchBlockOptions,
): void {
  const evidence = parent.createDiv({ cls: "arxiv-daily-dashboard__library-evidence" });
  if (hit.headings.length > 0) {
    evidence.createDiv({
      cls: "arxiv-daily-dashboard__library-section",
      text: hit.headings.join(" / "),
    });
  }
  evidence.createDiv({
    cls: "arxiv-daily-dashboard__library-passage",
    text: evidenceSnippet(hit.text),
  });
  const actions = evidence.createDiv({ cls: "arxiv-daily-dashboard__library-evidence-actions" });
  actions.createSpan({
    cls: "arxiv-daily-dashboard__library-page",
    text: `Page ${hit.page}`,
  });
  if (!match.filePath || !options.openEvidence) return;
  const action = `Open PDF at page ${hit.page}`;
  const button = actions.createEl("button", {
    cls: "arxiv-daily-dashboard__library-open-evidence",
    text: action,
    attr: { type: "button", "aria-label": action },
  });
  button.addEventListener("click", () => {
    try {
      void Promise.resolve(options.openEvidence?.(match, hit)).catch((error) => {
        reportActionError(options, error, action);
      });
    } catch (error) {
      reportActionError(options, error, action);
    }
  });
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

function evidenceSnippet(text: string): string {
  const normalized = text.replace(/\s+/gu, " ").trim();
  const maxCodeUnits = 360;
  return normalized.length <= maxCodeUnits
    ? normalized
    : `${normalized.slice(0, maxCodeUnits).trimEnd()}...`;
}
