/**
 * Deterministic full-text chunking for the personal library knowledge base.
 *
 * `chunkFullText` is a pure, deterministic, side-effect-free function: the same
 * input pages and options always produce the same chunks. It performs no I/O
 * and uses no randomness.
 *
 * Pipeline:
 *  1. Whitespace normalization — every page is split into lines; each line is
 *     trimmed and internal runs of whitespace (tabs, CR, multiple spaces,
 *     non-breaking spaces, …) are collapsed to single spaces.
 *  2. Paragraph aggregation — a paragraph is a maximal run of consecutive
 *     non-empty lines within a page (paragraphs never span page boundaries,
 *     since each page is an independent text block from the extractor). Lines
 *     of a paragraph are joined with "\n", preserving line structure.
 *  3. Noise filtering — paragraphs shorter than `minChunkChars` (default 16
 *     chars, a small value meant to drop page numbers, headers and other
 *     noise) are discarded.
 *  4. Greedy chunk assembly — paragraphs are accumulated into a chunk while
 *     the estimated token count stays within `targetTokens`. Paragraph
 *     boundaries are always preferred; a single paragraph that does not fit is
 *     hard-split instead of overflowing the cap.
 *  5. Overlap — between consecutive chunks the tail of the previous chunk is
 *     carried forward, duplicated, at the start of the next chunk so that
 *     context straddling a boundary is not lost. Duplication is intentional
 *     and does not count as content loss.
 *
 * Token estimation is the English approximation `ceil(chars / 4)`; caps are
 * converted to characters via `targetChars = targetTokens * 4`, which keeps
 * `ceil(chars / 4) <= targetTokens` for every emitted chunk.
 */

import type { ParsedDocument, ParserCapability } from "../../documents/parsed-document";
import {
  CHUNK_DERIVATION_VERSIONS,
  createEvidenceChunkId,
  type EvidenceChunk,
  type ParserProvenance,
} from "./evidence-chunk";
import type { FullTextChunk } from "./knowledge-base";

/** Options controlling full-text chunking. */
export interface ChunkingOptions {
  /** Target chunk size in tokens (estimated as ceil(chars / 4)). Default 512. */
  targetTokens?: number;
  /**
   * Overlap between consecutive chunks in tokens. Default is ~10% of
   * `targetTokens` (rounded, at least 1). 0 disables overlap.
   */
  overlapTokens?: number;
  /**
   * Minimum paragraph length in characters; shorter paragraphs are dropped as
   * noise. Default 16. 0 keeps every non-empty paragraph.
   */
  minChunkChars?: number;
}

const DEFAULT_TARGET_TOKENS = 512;
const DEFAULT_MIN_CHUNK_CHARS = 16;
const CHARS_PER_TOKEN = 4;

/** One atomic unit of chunking: a paragraph (or a piece of a hard-split one). */
interface Unit {
  text: string;
  /** One-based page of the unit's first character. */
  page: number;
}

/**
 * Chunk page-texts into a deterministic sequence of `FullTextChunk`s.
 *
 * The chunk's `page` is the one-based page of its first character; when a
 * chunk starts with carried-over overlap text, that text keeps the page of
 * the unit it came from. `index` is always 0-based and continuous.
 *
 * @param pages page texts in document order (index 0 is page 1)
 * @param options optional tuning knobs, see `ChunkingOptions`
 * @returns contiguous chunks; empty input yields an empty array
 */
export function chunkParsedDocument(
  document: ParsedDocument,
  capabilities: readonly ParserCapability[],
  parser: ParserProvenance,
  options?: ChunkingOptions,
): EvidenceChunk[] {
  if (!capabilities.includes("document-structure")) {
    const pages = document.blocks
      .filter((block) => block.kind === "page")
      .map((block) => block.text);
    return chunkFullText(pages, options).map((legacy) => evidenceFromLegacy(legacy, parser));
  }

  const targetTokens = requirePositiveInteger(options?.targetTokens, "targetTokens", DEFAULT_TARGET_TOKENS);
  const overlapTokens = requireNonNegativeInteger(
    options?.overlapTokens,
    "overlapTokens",
    Math.max(1, Math.round(targetTokens / 10)),
  );
  const minChunkChars = requireNonNegativeInteger(options?.minChunkChars, "minChunkChars", DEFAULT_MIN_CHUNK_CHARS);
  const sections = structuredSections(document, minChunkChars);
  const chunks: EvidenceChunk[] = [];
  for (const section of sections) {
    for (const built of chunkStructuredSection(
      section.units,
      targetTokens * CHARS_PER_TOKEN,
      overlapTokens * CHARS_PER_TOKEN,
    )) {
      const first = built[0]!;
      const last = built[built.length - 1]!;
      const locator = {
        pageStart: first.page,
        pageEnd: last.page,
        ...(first.block === undefined ? {} : { blockStart: first.block }),
        ...(last.block === undefined ? {} : { blockEnd: last.block }),
      };
      const identity = {
        text: built.map((unit) => unit.text).join("\n"),
        headings: [...section.headings],
        locator,
        derivation: { parser, ...CHUNK_DERIVATION_VERSIONS },
      };
      chunks.push({
        id: createEvidenceChunkId(identity),
        index: chunks.length,
        page: locator.pageStart,
        ...identity,
      });
    }
  }
  return chunks;
}

interface StructuredUnit extends Unit {
  block?: number;
}

interface StructuredSection {
  headings: string[];
  units: StructuredUnit[];
}

function structuredSections(document: ParsedDocument, minChunkChars: number): StructuredSection[] {
  const sections: StructuredSection[] = [];
  const headingStack: Array<{ level: number; text: string }> = [];
  let current: StructuredSection = { headings: [], units: [] };
  const flush = (): void => {
    if (current.units.length > 0) sections.push(current);
    current = { headings: headingStack.map((heading) => heading.text), units: [] };
  };
  for (const block of document.blocks) {
    if (block.kind === "heading") {
      const text = normalizeBlockText(block.text);
      if (!text) continue;
      flush();
      const level = Number.isSafeInteger(block.headingLevel) && block.headingLevel! > 0 ? block.headingLevel! : 1;
      while (headingStack.length > 0 && headingStack[headingStack.length - 1]!.level >= level) headingStack.pop();
      headingStack.push({ level, text });
      current = { headings: headingStack.map((heading) => heading.text), units: [] };
      continue;
    }
    const text = normalizeBlockText(block.text);
    const page = block.locator.page;
    if (!text || page === undefined) continue;
    current.units.push({ text, page, block: block.locator.block });
  }
  flush();
  // Drop a section only when all of its blocks together are noise. This lets
  // adjacent short structured blocks form useful evidence.
  return sections.filter((section) => section.units.reduce((sum, unit) => sum + unit.text.length, 0) >= minChunkChars);
}

function chunkStructuredSection(
  sourceUnits: readonly StructuredUnit[],
  targetChars: number,
  overlapChars: number,
): StructuredUnit[][] {
  const units = sourceUnits.flatMap((unit) => splitStructuredUnit(unit, targetChars));
  const chunks: StructuredUnit[][] = [];
  let previous: readonly StructuredUnit[] = [];
  let cursor = 0;
  while (cursor < units.length) {
    const carried = structuredOverlap(previous, overlapChars, targetChars);
    const parts = [...carried];
    let chars = parts.length === 0 ? 0 : totalChars(parts) + parts.length - 1;
    while (cursor < units.length) {
      const unit = units[cursor]!;
      const next = chars === 0 ? unit.text.length : chars + 1 + unit.text.length;
      if (next > targetChars) {
        if (parts.length === carried.length && carried.length > 0) {
          const budget = targetChars - chars - 1;
          const cut = boundaryCut(unit.text, budget);
          parts.push({ ...unit, text: unit.text.slice(0, cut) });
          units[cursor] = { ...unit, text: unit.text.slice(cut) };
        }
        break;
      }
      parts.push(unit);
      chars = next;
      cursor += 1;
    }
    if (parts.length === carried.length) {
      // A carried suffix left no room for new content; discard it and always
      // make progress with the already hard-split next unit.
      parts.length = 0;
      parts.push(units[cursor]!);
      cursor += 1;
    }
    chunks.push(parts);
    previous = parts;
  }
  return chunks;
}

function splitStructuredUnit(unit: StructuredUnit, targetChars: number): StructuredUnit[] {
  const parts: StructuredUnit[] = [];
  let remaining = unit.text;
  while (remaining.length > targetChars) {
    const cut = boundaryCut(remaining, targetChars);
    parts.push({ ...unit, text: remaining.slice(0, cut) });
    remaining = remaining.slice(cut);
  }
  if (remaining.length > 0) parts.push({ ...unit, text: remaining });
  return parts;
}

function structuredOverlap(
  previous: readonly StructuredUnit[],
  overlapChars: number,
  targetChars: number,
): StructuredUnit[] {
  if (previous.length === 0 || overlapChars <= 0) return [];
  const carried: StructuredUnit[] = [];
  let chars = 0;
  for (let index = previous.length - 1; index >= 0; index -= 1) {
    const unit = previous[index]!;
    const next = chars === 0 ? unit.text.length : chars + 1 + unit.text.length;
    if (next > overlapChars) break;
    carried.unshift(unit);
    chars = next;
  }
  if (carried.length === 0) {
    carried.push(previous[previous.length - 1]!);
    chars = carried[0]!.text.length;
  }
  return chars < targetChars - 1 ? carried : [];
}

function evidenceFromLegacy(chunk: FullTextChunk, parser: ParserProvenance): EvidenceChunk {
  const identity = {
    text: chunk.text,
    headings: [] as string[],
    locator: { pageStart: chunk.page },
    derivation: { parser, ...CHUNK_DERIVATION_VERSIONS },
  };
  return {
    id: createEvidenceChunkId(identity),
    index: chunk.index,
    page: chunk.page,
    ...identity,
  };
}

function normalizeBlockText(text: string): string {
  return text.split("\n").map((line) => line.trim().replace(/\s+/g, " ")).filter(Boolean).join("\n");
}

export function chunkFullText(pages: readonly string[], options?: ChunkingOptions): FullTextChunk[] {
  const targetTokens = requirePositiveInteger(options?.targetTokens, "targetTokens", DEFAULT_TARGET_TOKENS);
  const overlapTokens = requireNonNegativeInteger(
    options?.overlapTokens,
    "overlapTokens",
    Math.max(1, Math.round(targetTokens / 10)),
  );
  const minChunkChars = requireNonNegativeInteger(options?.minChunkChars, "minChunkChars", DEFAULT_MIN_CHUNK_CHARS);

  const targetChars = targetTokens * CHARS_PER_TOKEN;
  const overlapChars = overlapTokens * CHARS_PER_TOKEN;

  const units = extractParagraphUnits(pages, minChunkChars);
  const chunks: FullTextChunk[] = [];
  let prevUnits: readonly Unit[] = [];
  let cursor = 0;

  while (cursor < units.length) {
    // Overlap: duplicate the trailing units of the previous chunk (paragraph
    // granularity — whole units only, never mid-paragraph cuts), targeting
    // ~overlapChars. Overshoots by at most one unit when the last paragraph
    // alone exceeds the budget, and is skipped when it would leave no room
    // for new content.
    const carried = carriedOverlap(prevUnits, overlapChars, targetChars);
    const parts: Unit[] = [...carried];
    let charsSoFar = parts.length === 0 ? 0 : totalChars(parts) + parts.length - 1;

    while (cursor < units.length) {
      const unit = units[cursor]!;
      const nextChars = charsSoFar === 0 ? unit.text.length : charsSoFar + 1 + unit.text.length;
      if (nextChars <= targetChars) {
        parts.push(unit);
        charsSoFar = nextChars;
        cursor += 1;
        continue;
      }
      if (parts.length > carried.length) {
        // Chunk is full and the next paragraph is on the other side of a
        // paragraph boundary: flush, and let the next chunk start with it.
        break;
      }
      // Only carried overlap content so far and the paragraph does not fit:
      // hard-split it to fill the remaining budget without exceeding the cap.
      // The remainder stays at `units[cursor]` for the next chunk (no `cursor`
      // advance here — the head piece above is consumed instead).
      const budget = targetChars - (charsSoFar === 0 ? 0 : charsSoFar + 1);
      const cut = boundaryCut(unit.text, budget);
      parts.push({ text: unit.text.slice(0, cut), page: unit.page });
      units[cursor] = { text: unit.text.slice(cut), page: unit.page };
      break;
    }

    chunks.push({
      index: chunks.length,
      page: parts[0]!.page,
      text: parts.map((part) => part.text).join("\n"),
    });
    prevUnits = parts;
  }
  return chunks;
}

/**
 * Normalize a page into paragraph units: trim and collapse internal whitespace
 * runs per line, group consecutive non-empty lines into paragraphs, and drop
 * paragraphs shorter than `minChunkChars`.
 */
function extractParagraphUnits(pages: readonly string[], minChunkChars: number): Unit[] {
  const units: Unit[] = [];
  pages.forEach((pageText, pageIndex) => {
    const page = pageIndex + 1;
    let lines: string[] = [];
    const flush = (): void => {
      if (lines.length === 0) return;
      const text = lines.join("\n");
      if (text.length >= minChunkChars) units.push({ text, page });
      lines = [];
    };
    for (const rawLine of pageText.split("\n")) {
      const line = rawLine.trim().replace(/\s+/g, " ");
      if (line.length === 0) flush();
      else lines.push(line);
    }
    flush();
  });
  return units;
}

/**
 * Pick the overlap for the next chunk: the longest trailing suffix of the
 * previous chunk's units whose total text length is within `overlapChars`
 * (measured in characters, i.e. tokens x 4). If even the last unit alone
 * exceeds the budget it is carried whole anyway (a full paragraph of context
 * beats a tighter but mid-paragraph cut; the overshoot is at most one unit).
 * Returns an empty array when there is no previous chunk, overlap is
 * disabled, or carrying the tail would leave no room for at least one new
 * character in the next chunk.
 */
function carriedOverlap(prevUnits: readonly Unit[], overlapChars: number, targetChars: number): Unit[] {
  if (prevUnits.length === 0 || overlapChars <= 0) return [];
  let carried: Unit[] = [];
  let total = 0;
  for (let index = prevUnits.length - 1; index >= 0; index -= 1) {
    const unit = prevUnits[index]!;
    if (total + unit.text.length > overlapChars) break;
    carried.unshift(unit);
    total += unit.text.length;
  }
  if (carried.length === 0) {
    carried = [prevUnits[prevUnits.length - 1]!];
    total = carried[0]!.text.length;
  }
  // Account for the "\n" separators inside the carried text; leave budget for
  // at least one new character so the next chunk always makes progress.
  const charsSoFar = total + carried.length - 1;
  if (charsSoFar >= targetChars - 1) return [];
  return carried;
}

/**
 * Find where to cut `text` so the head is at most `maxChars`: prefer the last
 * line break, then the last space, inside the window; fall back to an exact
 * character cut (mid-word) so pieces never exceed the token cap.
 */
function boundaryCut(text: string, maxChars: number): number {
  if (text.length <= maxChars) return text.length;
  const newline = text.lastIndexOf("\n", maxChars - 1);
  if (newline >= 0) return newline + 1;
  const space = text.lastIndexOf(" ", maxChars - 1);
  if (space >= 0) return space + 1;
  return maxChars;
}

function totalChars(parts: readonly Unit[]): number {
  let total = 0;
  for (const part of parts) total += part.text.length;
  return total;
}

function requirePositiveInteger(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new TypeError(`chunkFullText: ${name} must be a positive integer, got ${JSON.stringify(value)}`);
  }
  return value;
}

function requireNonNegativeInteger(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TypeError(`chunkFullText: ${name} must be a non-negative integer, got ${JSON.stringify(value)}`);
  }
  return value;
}
