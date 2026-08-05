/**
 * Full-text knowledge base host ports.
 *
 * Core defines the contract and orchestration; hosts implement the pieces that
 * need their runtime: PDF full-text extraction (Obsidian built-in pdf.js in
 * the plugin host) and local embedding inference (transformers.js with
 * multilingual-e5-small q8). Neither port may leak host APIs into core.
 */

export interface PdfExtractionResult {
  /** Page texts in document order. Page 0 is the first page. */
  readonly pages: readonly string[];
}

export interface PdfExtractionOptions {
  signal?: AbortSignal;
}

export interface PdfTextExtractor {
  /**
   * Extract full text from PDF bytes. Must resolve pages in document order and
   * never throw for extractable content — malformed pages degrade to empty
   * strings; hard I/O or runtime failures throw.
   */
  extractPdfText(bytes: Uint8Array, options?: PdfExtractionOptions): Promise<PdfExtractionResult>;
}

export interface EmbeddingOptions {
  signal?: AbortSignal;
}

export interface EmbeddingModel {
  /** Stable identifier of the loaded model (family + quantization), e.g. `multilingual-e5-small-q8`. */
  readonly modelId: string;
  /** Vector dimension of the model, e.g. 384. */
  readonly dimension: number;
  /**
   * Embed texts in batch order. The caller applies the prefix policy before
   * calling (see `applyEmbeddingPrefix`); the host treats input as final.
   */
  embed(texts: readonly string[], options?: EmbeddingOptions): Promise<readonly Float32Array[]>;
}

/**
 * e5 family prefix policy. Core owns the policy so both sides always agree:
 * passages are embedded with `passage: ` at index time, queries with `query: `
 * at retrieval time.
 */
export const E5_QUERY_PREFIX = "query: ";
export const E5_PASSAGE_PREFIX = "passage: ";

export type EmbeddingTextKind = "query" | "passage";

export function applyEmbeddingPrefix(kind: EmbeddingTextKind, text: string): string {
  return kind === "query" ? `${E5_QUERY_PREFIX}${text}` : `${E5_PASSAGE_PREFIX}${text}`;
}
