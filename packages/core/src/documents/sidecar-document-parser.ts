import {
  DOCUMENT_PARSER_CAPABILITIES,
  type DocumentParser,
  type ParsedBlock,
  type ParsedDocument,
  type ParsedTextLayoutLine,
  type ParserCapability,
} from "./parsed-document";

export const SIDECAR_PARSER_PROTOCOL_VERSION = 1 as const;
export const MAX_SIDECAR_REQUEST_BYTES = 25 * 1024 * 1024;
export const MAX_SIDECAR_RESPONSE_BYTES = 16 * 1024 * 1024;

const MAX_PARSER_FIELD_LENGTH = 128;
const MAX_DOCUMENT_BLOCKS = 100_000;
const MAX_BLOCK_TEXT_LENGTH = 256 * 1024;
const MAX_DOCUMENT_TEXT_LENGTH = 8 * 1024 * 1024;
const MAX_PAGE_NUMBER = 100_000;
const MAX_BLOCK_ORDINAL = 1_000_000;

export interface SidecarParserCapabilities {
  readonly protocolVersion: typeof SIDECAR_PARSER_PROTOCOL_VERSION;
  readonly parser: DocumentParser["provenance"];
  readonly capabilities: readonly ParserCapability[];
  readonly maxRequestBytes: number;
  readonly maxResponseBytes: number;
}

export interface SidecarParseResponse {
  readonly protocolVersion: typeof SIDECAR_PARSER_PROTOCOL_VERSION;
  readonly parser: DocumentParser["provenance"];
  readonly document: ParsedDocument;
}

/** Reject anything other than an unauthenticated literal loopback HTTP endpoint. */
export function requireLoopbackSidecarUrl(value: string): URL {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new TypeError("sidecar endpoint must be an http loopback URL");
  }
  if (
    url.protocol !== "http:"
    || (url.hostname !== "127.0.0.1" && url.hostname !== "[::1]")
    || url.username
    || url.password
    || url.search
    || url.hash
  ) {
    throw new TypeError("sidecar endpoint must be an http loopback URL");
  }
  return url;
}

export function decodeSidecarParserCapabilities(value: unknown): SidecarParserCapabilities {
  const object = exactObject(value, [
    "protocolVersion", "parser", "capabilities", "maxRequestBytes", "maxResponseBytes",
  ], "sidecar capability document");
  return {
    protocolVersion: protocolVersion(object.protocolVersion),
    parser: parserProvenance(object.parser),
    capabilities: parserCapabilities(object.capabilities),
    maxRequestBytes: boundedByteLimit(object.maxRequestBytes, "maxRequestBytes", MAX_SIDECAR_REQUEST_BYTES),
    maxResponseBytes: boundedByteLimit(object.maxResponseBytes, "maxResponseBytes", MAX_SIDECAR_RESPONSE_BYTES),
  };
}

export function decodeSidecarParseResponse(value: unknown): SidecarParseResponse {
  const object = exactObject(value, ["protocolVersion", "parser", "document"], "sidecar parse response");
  return {
    protocolVersion: protocolVersion(object.protocolVersion),
    parser: parserProvenance(object.parser),
    document: parsedDocument(object.document),
  };
}

function protocolVersion(value: unknown): typeof SIDECAR_PARSER_PROTOCOL_VERSION {
  if (value !== SIDECAR_PARSER_PROTOCOL_VERSION) {
    throw new TypeError(`sidecar protocolVersion must be ${SIDECAR_PARSER_PROTOCOL_VERSION}`);
  }
  return value;
}

function parserProvenance(value: unknown): DocumentParser["provenance"] {
  const object = exactObject(value, ["id", "version"], "sidecar parser");
  return {
    id: boundedString(object.id, "sidecar parser.id", MAX_PARSER_FIELD_LENGTH),
    version: boundedString(object.version, "sidecar parser.version", MAX_PARSER_FIELD_LENGTH),
  };
}

function parserCapabilities(value: unknown): ParserCapability[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > DOCUMENT_PARSER_CAPABILITIES.length) {
    throw new TypeError("sidecar capabilities must be a non-empty bounded array");
  }
  const allowed = new Set<string>(DOCUMENT_PARSER_CAPABILITIES);
  const result: ParserCapability[] = [];
  for (const capability of value) {
    if (typeof capability !== "string" || !allowed.has(capability) || result.includes(capability as ParserCapability)) {
      throw new TypeError("sidecar capability is unsupported or duplicated");
    }
    result.push(capability as ParserCapability);
  }
  if (!result.includes("page-text")) throw new TypeError("sidecar capabilities must include page-text");
  return result;
}

function parsedDocument(value: unknown): ParsedDocument {
  const object = exactObject(value, ["mediaType", "blocks"], "sidecar document", ["metadata"]);
  if (object.mediaType !== "application/pdf") throw new TypeError("sidecar document mediaType must be application/pdf");
  if (!Array.isArray(object.blocks) || object.blocks.length > MAX_DOCUMENT_BLOCKS) {
    throw new TypeError("sidecar document blocks exceed the allowed limit");
  }
  let totalText = 0;
  const blocks = object.blocks.map((block, index) => {
    const parsed = parsedBlock(block, index);
    totalText += parsed.text.length;
    if (totalText > MAX_DOCUMENT_TEXT_LENGTH) throw new TypeError("sidecar document text exceeds the allowed limit");
    return parsed;
  });
  const metadata = object.metadata === undefined ? undefined : parsedMetadata(object.metadata);
  return {
    mediaType: "application/pdf",
    blocks,
    ...(metadata ? { metadata } : {}),
  };
}

function parsedMetadata(value: unknown): ParsedDocument["metadata"] {
  const object = exactObject(value, ["title"], "sidecar document metadata");
  return { title: boundedString(object.title, "sidecar document metadata.title", MAX_BLOCK_TEXT_LENGTH) };
}

function parsedBlock(value: unknown, index: number): ParsedBlock {
  const object = exactObject(value, ["kind", "text", "locator"], "sidecar document block", ["headingLevel", "layout"]);
  const kinds = new Set<ParsedBlock["kind"]>([
    "page", "heading", "paragraph", "list-item", "table", "figure", "caption", "equation", "code", "unknown",
  ]);
  if (typeof object.kind !== "string" || !kinds.has(object.kind as ParsedBlock["kind"])) {
    throw new TypeError(`sidecar document block ${index} has an unsupported kind`);
  }
  const kind = object.kind as ParsedBlock["kind"];
  const headingLevel = object.headingLevel === undefined
    ? undefined
    : boundedInteger(object.headingLevel, `sidecar document block ${index}.headingLevel`, 1, 16);
  if (kind === "heading" && headingLevel === undefined) {
    throw new TypeError(`sidecar document block ${index} headingLevel is required for headings`);
  }
  if (kind !== "heading" && headingLevel !== undefined) {
    throw new TypeError(`sidecar document block ${index} headingLevel is only valid for headings`);
  }
  const layout = object.layout === undefined ? undefined : parsedLayout(object.layout, index);
  return {
    kind,
    text: boundedString(object.text, `sidecar document block ${index}.text`, MAX_BLOCK_TEXT_LENGTH, true),
    locator: parsedLocator(object.locator, index),
    ...(headingLevel === undefined ? {} : { headingLevel }),
    ...(layout === undefined ? {} : { layout }),
  };
}

function parsedLocator(value: unknown, index: number): ParsedBlock["locator"] {
  const object = exactObject(value, [], `sidecar document block ${index}.locator`, ["page", "block", "charStart", "charEnd"]);
  const page = boundedInteger(object.page, `sidecar document block ${index}.locator.page`, 1, MAX_PAGE_NUMBER);
  const block = object.block === undefined ? undefined : boundedInteger(object.block, `sidecar document block ${index}.locator.block`, 0, MAX_BLOCK_ORDINAL);
  const charStart = object.charStart === undefined ? undefined : boundedInteger(object.charStart, `sidecar document block ${index}.locator.charStart`, 0, MAX_DOCUMENT_TEXT_LENGTH);
  const charEnd = object.charEnd === undefined ? undefined : boundedInteger(object.charEnd, `sidecar document block ${index}.locator.charEnd`, 0, MAX_DOCUMENT_TEXT_LENGTH);
  if (charStart !== undefined && charEnd !== undefined && charEnd < charStart) {
    throw new TypeError(`sidecar document block ${index}.locator.charEnd must not precede charStart`);
  }
  return { page, ...(block === undefined ? {} : { block }), ...(charStart === undefined ? {} : { charStart }), ...(charEnd === undefined ? {} : { charEnd }) };
}

function parsedLayout(value: unknown, blockIndex: number): ParsedTextLayoutLine[] {
  if (!Array.isArray(value) || value.length > 10_000) throw new TypeError(`sidecar document block ${blockIndex}.layout exceeds the allowed limit`);
  return value.map((entry, index) => {
    const object = exactObject(entry, ["text", "fontSize", "topFraction"], `sidecar document block ${blockIndex}.layout[${index}]`);
    const fontSize = finiteNumber(object.fontSize, `sidecar document block ${blockIndex}.layout[${index}].fontSize`, 0, 10_000);
    const topFraction = finiteNumber(object.topFraction, `sidecar document block ${blockIndex}.layout[${index}].topFraction`, 0, 1);
    return {
      text: boundedString(object.text, `sidecar document block ${blockIndex}.layout[${index}].text`, MAX_BLOCK_TEXT_LENGTH, true),
      fontSize,
      topFraction,
    };
  });
}

function exactObject(
  value: unknown,
  required: readonly string[],
  name: string,
  optional: readonly string[] = [],
): Record<string, unknown> {
  if (!isPlainObject(value)) throw new TypeError(`${name} must be an object`);
  const allowed = new Set([...required, ...optional]);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) throw new TypeError(`${name} contains unknown field ${key}`);
  }
  for (const key of required) {
    if (!(key in value)) throw new TypeError(`${name} is missing required field ${key}`);
  }
  return value;
}

function boundedString(value: unknown, name: string, maxLength: number, allowEmpty = false): string {
  if (typeof value !== "string" || value.length > maxLength || (!allowEmpty && value.length === 0)) {
    throw new TypeError(`${name} must be a bounded ${allowEmpty ? "string" : "non-empty string"}`);
  }
  return value;
}

function boundedByteLimit(value: unknown, name: string, max: number): number {
  return boundedInteger(value, name, 1, max);
}

function boundedInteger(value: unknown, name: string, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min || value > max) {
    throw new TypeError(`${name} must be an integer in range`);
  }
  return value;
}

function finiteNumber(value: unknown, name: string, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) throw new TypeError(`${name} must be a finite number in range`);
  return value;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}
