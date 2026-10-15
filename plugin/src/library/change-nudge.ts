import { shouldNudgeLibraryChange } from "@arxiv-daily/core";

/** The one action the nudge offers; it opens the direction review. */
export const LIBRARY_CHANGE_NUDGE_ACTION = "Update direction suggestions";

/** "1 new paper" / "3 new papers": shared by the nudge and the review page. */
export function describeNewLibraryPapers(count: number): string {
  return `${count} new ${count === 1 ? "paper" : "papers"}`;
}

/**
 * The live line on the review page. Unlike the nudge it also speaks for 1 or 2
 * papers: the page is already open, so there is nothing to interrupt.
 */
export function describeLibraryChangeLine(count: number): string {
  return `${describeNewLibraryPapers(count)} in your library ${count === 1 ? "isn't" : "aren't"} reflected in your direction suggestions yet.`;
}

/**
 * What to tell the researcher when their library has drifted from the current
 * direction suggestions (ADR 0015 section 3). `null` below the nudge count,
 * where a notice would be noise. The count comes from a local manifest read;
 * nothing here implies a model call has happened or will happen unasked.
 */
export function describeLibraryChangeNudge(count: number): string | null {
  if (!shouldNudgeLibraryChange(count)) return null;
  return describeLibraryChangeLine(count);
}

/**
 * Body of the Obsidian notice: the sentence and one button. A notice takes a
 * fragment, which is how it carries a button.
 * `null` when the count does not warrant a nudge.
 */
export function buildLibraryChangeNudgeFragment(
  count: number,
  onAction: () => void,
): DocumentFragment | null {
  const text = describeLibraryChangeNudge(count);
  if (text === null) return null;
  return createFragment((fragment) => {
    fragment.createDiv({ text });
    fragment.createEl("button", { text: LIBRARY_CHANGE_NUDGE_ACTION }).addEventListener("click", onAction);
  });
}
