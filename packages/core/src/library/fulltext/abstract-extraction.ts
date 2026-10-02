/**
 * Abstract extraction from a paper's leading pages (ADR 0013).
 *
 * The index covers a paper's title and abstract instead of its full text, and
 * the abstract has to come from the PDF itself: measured on the frozen
 * baseline corpus (212 files), only 1.4% of papers yield an arXiv ID, 45.3%
 * carry a DOI but no arXiv ID, and 53.3% expose no metadata at all. No
 * identification-dependent route reaches even half the library.
 *
 * `extractAbstractFromPages` is pure, deterministic and side-effect-free: it
 * performs no I/O and reads only the leading pages handed to it. Three routes
 * are tried in order, each scoped to a single page so an abstract never bleeds
 * into the body across a page break:
 *
 *  1. `marker` — an `Abstract` / `ABSTRACT` / letter-spaced `A B S T R A C T`
 *     line. Covers 77.8% of the corpus. Handles both the marker alone on its
 *     line (A&A) and the marker starting the body inline (`Abstract. Random
 *     forests are …`, Kluwer).
 *  2. `dated` — REVTeX papers print no `Abstract` word at all; the body simply
 *     follows the `(Dated: …)` line.
 *  3. `leading-text` — nothing recognizable, so take bounded leading prose.
 *
 * A page whose text layer is missing (scanned papers — 4.2% of the corpus,
 * often just a bibcode watermark) yields `none` rather than a fallback: a
 * watermark embedded as if it were an abstract is worse than no vector.
 *
 * Every route is bounded by `MAX_ABSTRACT_CHARS`. This bound is what keeps
 * the index at one or two chunks per paper — without it a single odd layout
 * could restore full-text volume, which is the cost ADR 0013 exists to remove.
 */

/** Upper bound on extracted abstract text. Roughly 600 tokens. */
export const MAX_ABSTRACT_CHARS = 2_400;

/**
 * Pages beyond this are never read: a long author list can push the abstract to
 * page 2, but no further.
 *
 * Exported because the indexer must pass the same bound to the extractor as
 * `PdfExtractionOptions.maxPages`. If the extractor were given a smaller bound
 * than this function reads, papers whose abstract sits on page 2 would silently
 * degrade to the `leading-text` fallback with nothing reporting the loss.
 */
export const MAX_LEADING_PAGES = 2;

/**
 * Below this, the leading pages carry no usable text layer at all. Checked
 * only before the `leading-text` fallback — a marker hit is itself proof of a
 * text layer, regardless of how little text the page holds.
 */
const MIN_USABLE_CHARS = 200;

/** A marker hit shorter than this is a table-of-contents entry, not an abstract. */
const MIN_ABSTRACT_CHARS = 40;

/** How the abstract was located. Recorded so extraction quality stays measurable. */
export type AbstractRoute = "marker" | "dated" | "leading-text" | "none";

export interface AbstractExtraction {
  /** Extracted abstract text, or undefined when no usable text exists. */
  abstract?: string;
  route: AbstractRoute;
}

/** `Abstract`, `ABSTRACT`, or letter-spaced `A B S T R A C T`, at a line start. */
const ABSTRACT_MARKER = /^[ \t]*(?:A[ \t]?B[ \t]?S[ \t]?T[ \t]?R[ \t]?A[ \t]?C[ \t]?T|Abstract)[ \t]*[.:—–-]?[ \t]*/m;

/** REVTeX prints the submission date immediately above the abstract body. */
const DATED_MARKER = /^[ \t]*\(Dated:[^)]*\)[ \t]*/m;

/**
 * Where an abstract ends: a keyword list, a subject classification, or the
 * first numbered/roman section heading of the body.
 */
const ABSTRACT_TERMINATOR = new RegExp(
  [
    String.raw`^[ \t]*Key[ \t]?words?\b`,
    String.raw`^[ \t]*Subject[ \t]headings\b`,
    String.raw`^[ \t]*(?:PACS|MSC)\b`,
    String.raw`^[ \t]*\d{1,2}\.?[ \t]+[A-Z]`,
    String.raw`^[ \t]*[IVX]{1,5}\.[ \t]+[A-Z]`,
    String.raw`^[ \t]*Section[ \t]+\d`,
  ].join("|"),
  "m",
);

/** Journal furniture that should not open a `leading-text` fallback. */
const HEADER_NOISE = /^(?:https?:\/\/|doi:|DOI:|©|c\s+\d{4}|arXiv:|\d{4}[A-Za-z&.]*\.{2,})/;

/**
 * Extract a paper's abstract from its leading pages.
 *
 * @param pages page texts in document order (index 0 is page 1)
 * @returns the abstract and the route that found it; `none` when the leading
 *          pages carry no usable text
 */
export function extractAbstractFromPages(pages: readonly string[]): AbstractExtraction {
  const leading = pages.slice(0, MAX_LEADING_PAGES).map(normalizePage);

  for (const page of leading) {
    const found = afterMarker(page, ABSTRACT_MARKER);
    if (found) return { abstract: found, route: "marker" };
  }
  for (const page of leading) {
    const found = afterMarker(page, DATED_MARKER);
    if (found) return { abstract: found, route: "dated" };
  }

  // Only the fallback needs the text-layer guard: a marker hit already proves
  // the page carries real text, however short the page happens to be.
  const usableChars = leading.reduce((sum, page) => sum + page.length, 0);
  if (usableChars < MIN_USABLE_CHARS) return { route: "none" };

  const fallback = leadingProse(leading[0] ?? "");
  return fallback ? { abstract: fallback, route: "leading-text" } : { route: "none" };
}

/**
 * Text following `marker` on the same page, cut at the first terminator and
 * bounded. Returns undefined when the marker is absent or the text behind it
 * is too short to be an abstract.
 */
function afterMarker(page: string, marker: RegExp): string | undefined {
  const match = marker.exec(page);
  if (!match) return undefined;
  const body = page.slice(match.index + match[0].length);
  const end = ABSTRACT_TERMINATOR.exec(body);
  const text = collapse(end ? body.slice(0, end.index) : body);
  return text.length >= MIN_ABSTRACT_CHARS ? bound(text) : undefined;
}

/** Bounded leading prose, skipping journal furniture lines at the top of the page. */
function leadingProse(page: string): string | undefined {
  const lines = page.split("\n").filter((line) => !HEADER_NOISE.test(line));
  const text = collapse(lines.join("\n"));
  return text.length >= MIN_ABSTRACT_CHARS ? bound(text) : undefined;
}

/** Trim lines and drop empty ones, preserving line structure for marker anchoring. */
function normalizePage(page: string): string {
  return page
    .split("\n")
    .map((line) => line.trim().replace(/[ \t ]+/g, " "))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** Fold the page's hard line breaks into spaces — embedding input is prose, not layout. */
function collapse(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/** Cut to `MAX_ABSTRACT_CHARS` at a word boundary when one is available. */
function bound(text: string): string {
  if (text.length <= MAX_ABSTRACT_CHARS) return text;
  const head = text.slice(0, MAX_ABSTRACT_CHARS);
  const space = head.lastIndexOf(" ");
  return (space > MAX_ABSTRACT_CHARS * 0.8 ? head.slice(0, space) : head).trimEnd();
}
