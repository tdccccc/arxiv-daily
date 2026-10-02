import { sha256Hex } from "../../utils/digest";

export interface ParserProvenance {
  /** Stable parser implementation identifier, independent of its host. */
  readonly id: string;
  /** Parser output-contract/implementation version. */
  readonly version: string;
}

export interface NormalizedBoundingBox {
  readonly left: number;
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
}

export interface EvidenceLocator {
  readonly pageStart: number;
  /** Present only when the end page is known from source structure. */
  readonly pageEnd?: number;
  readonly blockStart?: number;
  readonly blockEnd?: number;
  /** Optional page-normalized geometry in [0, 1]. */
  readonly bbox?: NormalizedBoundingBox;
}

export interface EvidenceDerivation {
  readonly parser: ParserProvenance;
  readonly chunkerVersion: number;
  readonly embeddingInputVersion: number;
}

/**
 * Derivation versions of the current indexing rules. Both take part in the
 * reuse decision, so raising either rebuilds every stored paper (ADR 0013 §3
 * chooses rebuilding over migration).
 *
 * `chunkerVersion` 3 — chunks come from one assembled title-and-abstract block
 * with the short-paragraph noise filter off, replacing per-page blocks fed
 * through the full-text path (2 was structured section-aware chunking).
 *
 * `embeddingInputVersion` 2 — the embedded text is a paper's title and
 * abstract rather than its full text.
 *
 * Both are raised explicitly even though the extractor provenance change
 * already forces a rebuild on its own: a rebuild that happens only as a side
 * effect stops happening the moment someone restores that provenance, and
 * nothing would report the loss.
 */
export const CHUNK_DERIVATION_VERSIONS = {
  chunkerVersion: 3,
  embeddingInputVersion: 2,
} as const;

export interface EvidenceChunk {
  readonly id: string;
  /** Compatibility vector-row position. */
  readonly index: number;
  /** Compatibility alias for locator.pageStart. */
  readonly page: number;
  readonly text: string;
  readonly headings: readonly string[];
  readonly locator: EvidenceLocator;
  readonly derivation: EvidenceDerivation;
}

export type EvidenceChunkIdentityInput = Pick<EvidenceChunk, "text" | "headings" | "locator" | "derivation">;

/**
 * Derive a stable host-neutral identity from canonical evidence content and its
 * source/derivation metadata. Paper keys, paths, row positions and randomness
 * are deliberately absent.
 */
export function createEvidenceChunkId(input: EvidenceChunkIdentityInput): string {
  const bbox = input.locator.bbox;
  const fields = [
    "evidence-chunk-id-v1",
    input.text,
    String(input.headings.length),
    ...input.headings,
    String(input.locator.pageStart),
    optionalNumber(input.locator.pageEnd),
    optionalNumber(input.locator.blockStart),
    optionalNumber(input.locator.blockEnd),
    bbox ? String(bbox.left) : "",
    bbox ? String(bbox.top) : "",
    bbox ? String(bbox.right) : "",
    bbox ? String(bbox.bottom) : "",
    input.derivation.parser.id,
    input.derivation.parser.version,
    String(input.derivation.chunkerVersion),
    String(input.derivation.embeddingInputVersion),
  ];
  return `sha256:${sha256Hex(fields.map(lengthPrefixed).join(""))}`;
}

function lengthPrefixed(value: string): string {
  return `${value.length}:${value}`;
}

function optionalNumber(value: number | undefined): string {
  return value === undefined ? "" : String(value);
}
