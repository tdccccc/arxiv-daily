/**
 * Fallback title extraction for papers without catalog metadata.
 *
 * Used at index time for fallback-indexed files (papers without catalog
 * metadata, e.g. `Mucesh2021.pdf`): the title feeds lexical query fusion and
 * result display. arXiv papers get their title from the catalog instead.
 *
 * Three paths, in priority order:
 *
 * 1. Document metadata (primary when present) — the host returns
 *    `PdfExtractionResult.metadataTitle` (`info.Title`), a machine-readable
 *    title when the producer wrote one. Garbage metadata (paths, file names,
 *    page references, arXiv stamps, LaTeX residue) is rejected; HTML entities
 *    are decoded. A typographic result whose tokens already cover the
 *    metadata and is longer wins over a metadata that dropped a character.
 * 2. Typographic — the host provides the first page's line layout
 *    (text + font size + vertical position, `PdfExtractionResult.layout`).
 *    The title is the most prominent text block on the page, so lines are
 *    ranked by font size with structural guards:
 *    - top-margin running heads (baseline above the page box), arXiv stamps,
 *      preprint report numbers and DOIs are excluded per line
 *    - a max band inside the top strip followed by a nearly-as-large band
 *      below it is a journal logo ("Astronomy & Astrophysics") and is skipped
 *    - candidate runs join lines at one font size (subscripts are skipped);
 *      author lines break the run (old papers set title and authors in one
 *      face); candidates must read as titles (length, uppercase/letter start,
 *      not a section heading, journal citation, date or address line)
 *    - after a candidate is chosen, continuation lines (lowercase starts,
 *      roman-numeral parts, long phrases without author initials) extend it,
 *      e.g. "Euclid preparation" + "VII. Forecast validation …"
 * 3. Text-only (fallback) — line heuristics over the plain first page for
 *    hosts without typographic layout. This keeps the pre-v4 rules.
 *
 * Known limitations (validated against the 376-file personal library): a
 * title and author line in one face with no initial or marker stays joined
 * ("BAYESIAN … ESTIMATION NARCISO BENIç TEZ"); an article-type label at the
 * title's font stays prefixed ("Review article A survey of …"); a
 * capitalized second continuation line can truncate a series title
 * ("… in the Western" instead of "… Western Galactic Hemisphere").
 */

import type { PdfLayoutLine } from "./ports";

/** The vertical band at the top of the page where journal logos sit, in % of page height. */
const STRIP_TOP = 13;

/** A band at ≥ 93% of the max font below the strip makes the max band a masthead. */
const MASTHEAD_RATIO = 0.93;

/** Merge line fonts within this difference into one band. */
const BAND_TOLERANCE = 0.3;

const MIN_TITLE_LEN = 10;
const MAX_TITLE_LEN = 300;

const JOURNAL_REFERENCE = /^(mnras|apj|apjl|apjs|a&a|aap|aj|nat|nature|science|space sci rev|jcap|prd|prl|pnas|pasj|jhep|aas|phys\.?\s*rev\.?)\b/i;

/** arXiv stamps ("arXiv:1208.0605v2 [astro-ph.CO] 18 Apr 2014"). */
const ARXIV_STAMP = /^arxiv:/i;

/** Section headings ("1. INTRODUCTION", "1.1. Introduction"). */
const SECTION_HEADING = /^\d+\s*[.)]/;

const BARE_ABSTRACT = /^a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*$/i;

const ALL_DIGITS_PUNCT = /^[\d\s.…\-—–()]+$/;

/** Collab/lab preprint report numbers and DOIs ("DES-2019-0442",
 * "DES 2015-0146", "FERMILAB-PUB-19-513-AE", "Fermilab PUB-16-012-E-PPD",
 * "DOI 10.3847/0067-0049/224/1/1"). */
const DOCUMENT_IDENTIFIER = /^(?:[Dd][Oo][Ii][\s:]|[A-Z][A-Za-z0-9&]{1,}(?:[\s-][A-Z0-9]+)*[\s-]\d{2,4}[A-Z0-9-]*$)/;

/** Publication date lines ("Received 2002 March 11; accepted 2002 October 31"). */
const DATE_LINE = /^(?:received|accepted|published|submitted|revised)\b/i;

/** Old-LaTeX email convention ("benitezn=mars.berkeley.edu") marks affiliations. */
const EMAIL_CONVENTION = /[a-z0-9._-]+=[a-z0-9.-]+\.[a-z]{2,}/i;

/**
 * A line that reads as an author list ("Wayne Hu and Andrey V. Kravtsov",
 * "Michael A. Strauss,", "THOMAS H. REIPRICH1 AND HANS BOéHRINGER") — never a
 * title line, breaks a same-font run (old papers set title and authors in one
 * face).
 */
const AUTHOR_LINE = /^(?:[A-Z][\p{L}.'-]+\d*(?: [A-Z]\.)+\d? [A-Z][\p{L}.'-]+\d*[,;]?|[A-Z][\p{L}.'-]+\d*(?: (?!\b[Aa][Nn][Dd]\b)[A-Z][\p{L}.'-]+\d*){0,2}(?:,? (?:[Aa][Nn][Dd]|&)) [A-Z][\p{L}.'-]+\d*(?: [A-Z][\p{L}.'-]+\d*){0,3}[,;]?|and [A-Z][\p{L}.'-]+\d*(?: [A-Z][\p{L}.'-]+\d*){0,2})$/u;

/** A name initial ("V." in "Andrey V. Kravtsov"); author lines carry one. */
const NAME_INITIAL = /(?:^|\s)[A-Z]\./;

/** A two-word capitalized line ("Raul Jimenez") that can start an author pair. */
const TWO_WORD_NAME = /^[A-Z][\p{L}.'-]+\d* [A-Z][\p{L}.'-]+\d*[,;]?$/u;

/** Two-word and "Name and Name" shapes that read as authors (no initials
 * required here: this also guards candidates when the max band is a fragment). */
const NAME_PAIR_CANDIDATE = /^(?:[A-Z][\p{L}.'-]+\d* [A-Z][\p{L}.'-]+\d*[,;]?|[A-Z][\p{L}.'-]+\d*(?: (?!\b[Aa][Nn][Dd]\b)[A-Z][\p{L}.'-]+\d*){0,2}(?:,? (?:[Aa][Nn][Dd]|&)) [A-Z][\p{L}.'-]+\d*(?: [A-Z][\p{L}.'-]+\d*){0,3}[,;]?)$/u;

/** A bare name-pair list ("Jacob Devlin Ming-Wei Chang Kenton Lee",
 * "Wayne Xin Zhao, Kun Zhou*, Junyi Li*, …") is an author block, not a title
 * continuation. Two shapes: middle-dot separated pairs and comma-separated
 * full-name pairs (arXiv-style, with optional affiliation stars and a
 * trailing "and"). */
const NAME_PAIR_LIST = /^(?:[A-Z][\p{L}.'-]+\d* [A-Z][\p{L}.'-]+\d*)(?: [·•] [A-Z][\p{L}.'-]+\d* [A-Z][\p{L}.'-]+\d*)+(?: [·•])?[,;]?$|^(?:[A-Z][\p{L}.'-]+\d*(?: [A-Z][\p{L}.'-]+\d*){1,2}\*?, )+(?:and )?[A-Z][\p{L}.'-]+\d*(?: [A-Z][\p{L}.'-]+\d*){0,2}\*?,?$/u;

/**
 * Title continuation cues: a lowercase start, a roman-numeral part
 * ("XXVIII. …"), a subscript continuation ("0 using …") or a long uppercase
 * phrase without author initials ("Optical identification and properties of…"
 * after "The SRG/eROSITA All-Sky Survey").
 */
const CONTINUATION = /^(?:[ivxlcdm]+\.\s|\d+\s+[a-z]|[a-z]|(?=[A-Z])[A-Z][a-zA-Z ,:'–-]{39,})/;

/** Function words never start a title ("and …", "with …"); lowercase proper
 * nouns ("redMaPPer – III. …", "dustmaps: …") do. */
const FUNCTION_WORD_START = /^(?:and|or|et|with|from|in|of|by|using|use|for|at|the|a|an|to|on|as|but|also|only|via|per|into|among|between|through|under|over|towards?|after|before|during|within|without|including|excluding|based|compared|relative|according|results)$/i;

/**
 * Extract a title from the first page of extracted PDF text, or `null` when
 * no plausible title is found. Selection order: the document metadata title
 * (machine-readable, when the host provides a usable one) first, then the
 * typographic layout when the host provides one, then plain-text line
 * heuristics.
 */
export function extractTitleFromFirstPage(
  pages: readonly string[],
  layout?: readonly (readonly PdfLayoutLine[])[],
  metadataTitle?: string,
): string | null {
  const decodedMetadata = decodeMetadataTitle(metadataTitle);
  if (decodedMetadata !== null) {
    // A typographic title wins over the metadata only when its BASE title
    // (before continuation extension) is a STRICT token superset and longer
    // (metadata dropped a character, e.g. a missing "z"). Equal token sets —
    // or an extended font result that only adds author lines — mean the
    // metadata is the same title in better shape: the metadata wins.
    const firstPageLayout = layout?.[0];
    if (firstPageLayout && firstPageLayout.length > 0) {
      const fontTitle = extractFontTitle(firstPageLayout);
      if (fontTitle !== null && strictlyCovers(fontTitle.base, decodedMetadata)
        && fontTitle.base.length > decodedMetadata.length) {
        return fontTitle.title;
      }
    }
    return decodedMetadata;
  }
  const firstPageLayout = layout?.[0];
  if (firstPageLayout && firstPageLayout.length > 0) {
    const title = extractFontTitle(firstPageLayout);
    if (title !== null) return title.title;
  }
  return extractTextTitle(pages);
}

/**
 * Validate and decode the document metadata title. Returns `null` when the
 * metadata is missing or unusable (paths, file names, page references, arXiv
 * stamps, LaTeX residue); pdf.js leaves HTML entities in `info.Title`
 * (`&ndash;`, `&#x00D7;`), which are decoded here.
 */
function decodeMetadataTitle(metadataTitle?: string): string | null {
  const title = metadataTitle?.trim();
  if (!title || title.length < MIN_TITLE_LEN || title.length > MAX_TITLE_LEN) return null;
  if (ARXIV_STAMP.test(title)) return null;
  if (METADATA_GARBAGE.test(title)) return null;
  return decodeHtmlEntities(title);
}

/** Paths, file names, page references and LaTeX residue in metadata titles. */
const METADATA_GARBAGE = /(?:\\|\.(?:eps|dvi|ps|tp|pdf|doc)$|^\d{4,6}\s+\d+\.\.\d+$|microsoft word|\$)/i;

const HTML_ENTITIES: Readonly<Record<string, string>> = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&apos;": "'",
  "&nbsp;": " ",
  "&ndash;": "–",
  "&mdash;": "—",
  "&hellip;": "…",
  "&times;": "×",
  "&minus;": "−",
};

/** Decode the numeric and common named HTML entities pdf.js leaves in metadata. */
function decodeHtmlEntities(text: string): string {
  return text.replace(/&#x([0-9a-f]+);|&#(\d+);|&[a-z]+;/gi, (match, hex: string | undefined, dec: string | undefined) => {
    if (hex) return String.fromCodePoint(parseInt(hex, 16));
    if (dec) return String.fromCodePoint(parseInt(dec, 10));
    return HTML_ENTITIES[match.toLowerCase()] ?? match;
  });
}

/** Normalized lowercase token set, used for the metadata/font coverage check. */
function tokens(text: string): Set<string> {
  return new Set(text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().split(/\s+/).filter(Boolean));
}

/** Whether every token of `needle` also appears in `haystack`. */
function coversAll(haystack: string, needle: string): boolean {
  const haystackTokens = tokens(haystack);
  for (const token of tokens(needle)) {
    if (!haystackTokens.has(token)) return false;
  }
  return true;
}

/** Whether the haystack's token set strictly contains the needle's (the
 * needle is missing at least one token the haystack has). */
function strictlyCovers(haystack: string, needle: string): boolean {
  const haystackTokens = tokens(haystack);
  const needleTokens = tokens(needle);
  if (needleTokens.size === 0 || haystackTokens.size <= needleTokens.size) return false;
  for (const token of needleTokens) {
    if (!haystackTokens.has(token)) return false;
  }
  return true;
}

/** Cut a candidate line at the first author marker, if any. */
function cutAtAuthorMarker(line: string): string {
  const markers = ["∗", "†", "@"];
  let cut = line.length;
  for (const marker of markers) {
    const index = line.indexOf(marker);
    if (index !== -1 && index < cut) cut = index;
  }
  const abstractIndex = line.indexOf("Abstract");
  if (abstractIndex !== -1 && abstractIndex < cut) cut = abstractIndex;
  return line.slice(0, cut).trim();
}

function isPlausibleTitle(text: string): boolean {
  if (text.length < MIN_TITLE_LEN || text.length > MAX_TITLE_LEN) return false;
  if (/(?:www\.|https?:\/\/)/i.test(text)) return false;
  if (DATE_LINE.test(text)) return false;
  if (EMAIL_CONVENTION.test(text)) return false;
  // Lowercase starts are author/affiliation continuations ("and X", "with
  // results…"); lowercase proper-noun titles ("redMaPPer – III. …",
  // "dustmaps: …") pass because their first word is not a function word.
  const firstWord = text.split(/\s+/)[0] ?? "";
  if (/^[a-z]/.test(text) && FUNCTION_WORD_START.test(firstWord)) return false;
  // A single lowercase letter start is a math symbol ("z ∼ 1.1 …"), not a title.
  if (/^[a-z](?:\s|$)/.test(text)) return false;
  const first = text.codePointAt(0);
  const isLetter = first !== undefined && /^\p{L}$/u.test(String.fromCodePoint(first));
  const isDigit = first !== undefined && /^\d$/.test(String.fromCodePoint(first));
  if (!isLetter && !isDigit) return false;
  if (ARXIV_STAMP.test(text)) return false;
  if (DOCUMENT_IDENTIFIER.test(text)) return false;
  if (SECTION_HEADING.test(text)) return false;
  if (BARE_ABSTRACT.test(text)) return false;
  if (ALL_DIGITS_PUNCT.test(text)) return false;
  if (JOURNAL_REFERENCE.test(text) && /\d{4}/.test(text)) return false;
  return true;
}

function isContinuation(line: string): boolean {
  if (DATE_LINE.test(line)) return false;
  if (EMAIL_CONVENTION.test(line)) return false;
  if (/(?:www\.|https?:\/\/)/i.test(line)) return false;
  if (NAME_PAIR_LIST.test(line)) return false;
  // Author lines carry affiliation markers ("R. Beck, 1,2 <", "…, 3,4,2");
  // titles never do. This catches host text-extraction variants where name
  // patterns fail ("R' obert Beck, 1,2 <" from Obsidian's pdf.js).
  if (AFFILIATION_MARKER.test(line)) return false;
  if (CONTINUATION.test(line)) return true;
  // Long uppercase phrases without author initials continue series titles
  // ("Optical identification and properties of…" after "The SRG/eROSITA
  // All-Sky Survey").
  if (line.length >= 40 && line.includes(" ") && !NAME_INITIAL.test(line)) return true;
  return false;
}

/** Comma-separated affiliation numbers ("1,2 <", ", 3,4,2") mark author lines. */
const AFFILIATION_MARKER = /,\s*\d+\s*[,<]/;

/** Join run lines with spaces; rejoin hyphenation and subscript digits. */
function joinRun(run: readonly PdfLayoutLine[]): string {
  let text = "";
  for (const line of run) {
    const trimmed = line.text.trim();
    if (!text) {
      text = trimmed;
    } else if (text.endsWith("-")) {
      text += trimmed;
    } else {
      text += ` ${trimmed}`;
    }
  }
  return text;
}

/** Typographic title selection over a first-page line layout. Returns the
 * base candidate (before continuation extension) and the extended title;
 * callers use the base for the metadata coverage check so that extension
 * noise (author lines) never outranks a correct metadata title. */
function extractFontTitle(
  layout: readonly PdfLayoutLine[],
): { base: string; title: string } | null {
  const usable = layout.filter((line) => {
    const text = line.text.trim();
    if (!text || line.fontSize <= 0) return false;
    const topPct = line.topFraction * 100;
    // Running heads in the top margin, arXiv stamps and preprint report
    // numbers are never titles; exclude per line.
    if (topPct < -5) return false;
    if (ARXIV_STAMP.test(text)) return false;
    if (DOCUMENT_IDENTIFIER.test(text)) return false;
    return true;
  });
  if (usable.length === 0) return null;
  const maxFont = Math.max(...usable.map((line) => line.fontSize));

  // Masthead guard: a max band that starts inside the top strip, followed by a
  // nearly-as-large band below the strip, is a journal logo (A&A "Astronomy &
  // Astrophysics", Elsevier "Astroparticle Physics"); skip to the next band.
  const maxBandInStrip = usable.some(
    (line) => Math.abs(line.fontSize - maxFont) <= BAND_TOLERANCE && line.topFraction * 100 <= STRIP_TOP,
  );
  let nextBand = -1;
  for (const line of usable) {
    if (line.fontSize < maxFont - BAND_TOLERANCE && line.fontSize > nextBand && line.text.trim().length > 4) {
      nextBand = line.fontSize;
    }
  }
  const skipMax = maxBandInStrip
    && nextBand >= maxFont * MASTHEAD_RATIO
    && usable.some(
      (line) => Math.abs(line.fontSize - nextBand) <= BAND_TOLERANCE && line.topFraction * 100 > STRIP_TOP,
    );
  const minBand = skipMax ? nextBand : maxFont;

  const bandFonts = [...new Set(usable.map((line) => line.fontSize))]
    .filter((font) => font <= minBand + BAND_TOLERANCE)
    .sort((left, right) => right - left);

  // The max band may be a short fragment of the title rendered at a larger
  // font ("surveys" in "COSMOLIKE … photometric galaxy surveys"); in that
  // case two-word name-like candidates (author lines) are not titles.
  const maxBandRunLength = (() => {
    let length = 0;
    for (const line of usable) {
      if (Math.abs(line.fontSize - maxFont) <= BAND_TOLERANCE) length += line.text.trim().length;
      else if (length > 0) break;
    }
    return length;
  })();

  let chosen: string | null = null;
  let chosenEnd = -1;
  for (const band of bandFonts) {
    // Assemble runs in reading order; short lines with a much smaller font
    // (subscripts) are skipped and do not break the run.
    let run: PdfLayoutLine[] = [];
    for (let index = 0; index < usable.length; index += 1) {
      const line = usable[index]!;
      const text = line.text.trim();
      const inBand = Math.abs(line.fontSize - band) <= BAND_TOLERANCE;
      const subscript = text.length <= 4 && line.fontSize < band * 0.8;
      let nextSubstantial = index + 1;
      while (nextSubstantial < usable.length && usable[nextSubstantial]!.text.trim().length <= 4) {
        nextSubstantial += 1;
      }
      const nextStartsAnd = nextSubstantial < usable.length
        && /^and\b/i.test(usable[nextSubstantial]!.text.trim());
      const authorLine = run.length > 0
        && ((AUTHOR_LINE.test(text) && NAME_INITIAL.test(text))
          || (TWO_WORD_NAME.test(text) && nextStartsAnd)
          || AFFILIATION_MARKER.test(text));
      if ((!inBand && !subscript) || authorLine) {
        if (run.length > 0) {
          const cleaned = cutAtAuthorMarker(joinRun(run));
          if (isPlausibleTitle(cleaned)
            && !(maxBandRunLength < MIN_TITLE_LEN && NAME_PAIR_CANDIDATE.test(cleaned))) {
            chosen = cleaned;
            chosenEnd = index;
            break;
          }
          run = [];
        }
      } else if (inBand) {
        run.push(line);
      }
    }
    if (chosen) break;
    if (run.length > 0) {
      const cleaned = cutAtAuthorMarker(joinRun(run));
      if (isPlausibleTitle(cleaned)
        && !(maxBandRunLength < MIN_TITLE_LEN && NAME_PAIR_CANDIDATE.test(cleaned))) {
        chosen = cleaned;
        chosenEnd = usable.length;
        break;
      }
    }
  }
  if (chosen === null) return null;

  // Extend the chosen title with continuation lines after the run (lowercase
  // starts, roman numeral parts), e.g. "Euclid preparation" + "VII. …".
  // Lines already present in the title (duplicate renderings) and short
  // fragments are skipped.
  const parts = [chosen];
  for (let index = chosenEnd; index < usable.length; index += 1) {
    const line = usable[index]!.text.trim();
    if (!line) continue;
    if (!isContinuation(line)) break;
    if (line.length <= 4) continue;
    const cleaned = cutAtAuthorMarker(line);
    if (!cleaned) continue;
    if (parts.join(" ").includes(cleaned)) continue;
    parts.push(cleaned);
  }
  let joined = "";
  for (const part of parts) {
    if (!joined) joined = part;
    else if (joined.endsWith("-") || /^\d/.test(part)) joined += part;
    else joined += ` ${part}`;
  }
  return { base: chosen, title: joined };
}

/**
 * Text-only fallback: scan the first page line by line, skipping blank lines,
 * arXiv/reprint headers, lines starting lowercase (title continuations of a
 * permission notice), and lines that are too short/long to be a title. The
 * first remaining line is the candidate; author lists that share the title
 * line are cut at the first author marker (`∗`, `†`, `@`) or the literal
 * "Abstract".
 */
const HEADER_PREFIXES = [
  "arxiv:",
  "submitted to",
  "submitted for",
  "draft version",
  "preprint typeset",
  "published as",
  "published in",
  "presented at",
  "this paper has been",
  "provided proper attribution",
  "proceedings of",
  "space sci rev",
  "advance access publication",
  "advance access",
  "©",
  "doi:",
  "http",
  "www.",
];

const ADVANCE_ACCESS_BANNER = /^advance\s+access\b/i;

function extractTextTitle(pages: readonly string[]): string | null {
  const firstPage = pages[0];
  if (!firstPage) return null;
  for (const rawLine of firstPage.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.length < 10 || line.length > 200) continue;
    const lowered = line.toLowerCase();
    if (HEADER_PREFIXES.some((prefix) => lowered.startsWith(prefix))) continue;
    if (ADVANCE_ACCESS_BANNER.test(line)) continue;
    // Author/affiliation continuation lines (" , Martin Landriau 2 , ...").
    if (/^[,.;:—–-]/.test(line)) continue;
    // Journal references are not titles.
    if (JOURNAL_REFERENCE.test(line) && /\d{4}/.test(line)) continue;
    const firstChar = line.charCodeAt(0);
    // Titles start with an uppercase letter or a digit-free symbol; lowercase
    // starts are almost always continuation lines of a preceding block.
    if (firstChar >= 97 && firstChar <= 122) continue;
    if (/^[\d\s.…-]+$/.test(line)) continue;
    const title = cutAtAuthorMarker(line);
    if (title.length >= 5) return title;
  }
  return null;
}
