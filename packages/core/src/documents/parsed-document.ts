export const DOCUMENT_PARSER_CAPABILITIES = [
  "page-text",
  "text-layout",
  "document-metadata",
  "document-structure",
] as const;

export type ParserCapability = (typeof DOCUMENT_PARSER_CAPABILITIES)[number];

export type ParsedBlockKind =
  | "page"
  | "heading"
  | "paragraph"
  | "list-item"
  | "table"
  | "figure"
  | "caption"
  | "equation"
  | "code"
  | "unknown";

export interface SourceLocator {
  /** 1-based source page number. */
  readonly page?: number;
  /** 0-based ordinal in the document's top-level reading-order blocks. */
  readonly block?: number;
  /** Inclusive UTF-16 offset within the block text. */
  readonly charStart?: number;
  /** Exclusive UTF-16 offset within the block text. */
  readonly charEnd?: number;
}

export interface ParsedTextLayoutLine {
  readonly text: string;
  readonly fontSize: number;
  readonly topFraction: number;
}

export interface ParsedBlock {
  readonly kind: ParsedBlockKind;
  readonly text: string;
  readonly locator: SourceLocator;
  readonly layout?: readonly ParsedTextLayoutLine[];
}

export interface ParsedDocumentMetadata {
  readonly title?: string;
}

export interface ParsedDocument {
  readonly mediaType: string;
  /** Top-level blocks in the parser's default reading order. */
  readonly blocks: readonly ParsedBlock[];
  readonly metadata?: ParsedDocumentMetadata;
}

export interface ParseDocumentOptions {
  readonly signal?: AbortSignal;
}

export interface DocumentParser {
  readonly capabilities: readonly ParserCapability[];
  parse(bytes: Uint8Array, options?: ParseDocumentOptions): Promise<ParsedDocument>;
}
