import { inflate } from "pako";

/**
 * Lightweight PDF identification-evidence extraction (identification v2).
 *
 * Extracts just enough evidence from a PDF to identify the paper behind an
 * arbitrarily named file — it never parses the full document structure and
 * never returns body text:
 *   - arXiv IDs from the Info dict /arXivID (the submission system's own
 *     identity claim), decompressed content-stream headers, or XMP
 *     identifiers — in that trust order;
 *   - a usable document title from the Info dict /Title (literal, hex, or
 *     UTF-16) or XMP dc:title, used only as a title-search candidate.
 *
 * Scanned PDFs without a text layer produce no evidence (caller keeps the
 * file unresolved); custom-encoded fonts may produce garbage titles, which
 * the title-usage filter rejects.
 */
import {
  extractArxivIdsFromText,
  modernArxivIdFromText,
  type ArxivIdCandidate,
} from "./pdf-text-utils";

export interface PdfIdentificationEvidence {
  /** Canonical arXiv ID found in the PDF (content-stream header or XMP). */
  arxivId?: string;
  /** Usable document title (decoded and filtered) or undefined. */
  title?: string;
}

/** Cache version for content-based identification evidence rules. */
export const PDF_IDENTIFICATION_EVIDENCE_VERSION = 3 as const;

const META_REGION_BYTES = 256 * 1024;
const STREAM_REGEX = /(?:\r?\n)stream\r?\n/g;
const MAX_INFLATED_STREAM_BYTES = 8 * 1024 * 1024;
const MAX_TOTAL_INFLATED_BYTES = 32 * 1024 * 1024;
const STREAM_TEXT_PREFIX_CHARS = 512;
const TITLE_MIN_LENGTH = 10;
const TITLE_MAX_LENGTH = 300;

export function extractPdfIdentificationEvidence(bytes: Uint8Array): PdfIdentificationEvidence {
  if (bytes.length === 0) return {};
  const latin = latin1Decode(bytes);
  if (!latin.startsWith("%PDF")) return {};

  const inflated = inflateFlateStreams(bytes, latin);
  const streamTexts = inflated
    .map((text) => extractTextLiterals(text))
    .filter((text): text is string => typeof text === "string" && text.length > 0);

  // arXiv IDs: the Info dict's /arXivID is the submission system's own
  // identity claim and outranks any stream text — content streams can cite
  // other papers' IDs in reference lists ("… arXiv:0912.0201 …"), which would
  // otherwise misidentify the file. Stream headers (arXiv page headers) come
  // next, then XMP identifiers.
  const headerText = streamTexts.map((text) => text.slice(0, STREAM_TEXT_PREFIX_CHARS)).join("\n");
  const arxivId = infoDictArxivId(latin)
    ?? extractArxivIdsFromText(headerText)[0]?.canonicalId
    ?? xmpArxivId(inflated.join("\n"));

  const title = findUsableTitle(latin, inflated.join("\n"));
  const evidence: PdfIdentificationEvidence = {};
  if (arxivId) evidence.arxivId = arxivId;
  if (title) evidence.title = title;
  return evidence;
}

/** Decompress every FlateDecode stream, bounded, tolerant of malformed data. */
function inflateFlateStreams(bytes: Uint8Array, latin: string): string[] {
  const outputs: string[] = [];
  let total = 0;
  let match: RegExpExecArray | null;
  STREAM_REGEX.lastIndex = 0;
  while ((match = STREAM_REGEX.exec(latin)) !== null) {
    const start = match.index + match[0].length;
    const end = latin.indexOf("endstream", start);
    if (end < 0) break;
    if (end - start > MAX_INFLATED_STREAM_BYTES) continue;
    let text: string | undefined;
    try {
      const inflated = inflate(bytes.subarray(start, end));
      if (inflated.byteLength > MAX_INFLATED_STREAM_BYTES) continue;
      total += inflated.byteLength;
      if (total > MAX_TOTAL_INFLATED_BYTES) break;
      text = latin1Decode(inflated);
    } catch {
      // Non-Flate streams (e.g. raw) are skipped; identification is best-effort.
    }
    if (text) outputs.push(text);
  }
  return outputs;
}

/** Extract text operators `(...)` and `<hex>` string literals in stream order. */
function extractTextLiterals(text: string): string | undefined {
  let out = "";
  for (const match of text.matchAll(/\(((?:\\.|[^()\\]){1,400})\)|(<[0-9A-Fa-f]{2,}>)/g)) {
    if (match[1] !== undefined) {
      out += decodeLiteral(match[1]);
    } else {
      out += decodeHexLiteral(match[2]!);
    }
    out += " ";
  }
  return out.trim() ? out : undefined;
}

function decodeLiteral(value: string): string {
  return value
    .replace(/\\([nrtbf])/g, (_, code: string) => (
      { n: "\n", r: "\r", t: "\t", b: "\b", f: "\f" } as Record<string, string>
    )[code]!)
    .replace(/\\(.)/g, "$1");
}

function decodeHexLiteral(value: string): string {
  // Callers pass the captured hex content (no angle brackets).
  const hex = value;
  if (/^(feff|fffe)/i.test(hex)) {
    const bytes = hexToBytes(hex);
    const utf16le = bytes[0] === 0xff && bytes[1] === 0xfe;
    const codeUnits: number[] = [];
    for (let index = 2; index + 1 < bytes.length; index += 2) {
      codeUnits.push(utf16le ? bytes[index]! | (bytes[index + 1]! << 8) : (bytes[index]! << 8) | bytes[index + 1]!);
    }
    return utf16Decode(codeUnits);
  }
  return latin1Decode(hexToBytes(hex));
}

function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

function utf16Decode(codeUnits: number[]): string {
  // Strip a leading UTF-16 BOM if present.
  const start = codeUnits[0] === 0xfeff ? 1 : 0;
  let out = "";
  for (let index = start; index < codeUnits.length; index += 1) {
    const unit = codeUnits[index]!;
    if (unit >= 0xd800 && unit <= 0xdbff && codeUnits[index + 1] !== undefined) {
      const low = codeUnits[index + 1]!;
      if (low >= 0xdc00 && low <= 0xdfff) {
        out += String.fromCharCode(unit, low);
        index += 1;
        continue;
      }
    }
    out += String.fromCharCode(unit);
  }
  return out;
}

/** XMP dc:identifier carrying an arXiv URL. */
function xmpArxivId(allInflatedText: string): string | undefined {
  const identifier = allInflatedText.match(/<dc:identifier>(?:<rdf:li[^>]*>)?([^<]{5,200})/i);
  if (!identifier) return undefined;
  const candidate = identifier[1]!;
  return modernArxivIdFromText(candidate);
}

/**
 * The Info dict's /arXivID (or /arXiv) — written by the arXiv submission
 * system — is the file's own identity claim, more authoritative than any
 * stream text (references can cite other papers). Matched in the head
 * metadata region, which is where Info dicts live.
 */
function infoDictArxivId(latin: string): string | undefined {
  const region = latin.slice(0, META_REGION_BYTES);
  const match = region.match(/\/arXivID\s*\(([^)]{5,200})\)/i)
    ?? region.match(/\/arXiv\s*\(([^)]{5,200})\)/i);
  return match ? modernArxivIdFromText(match[1]!) : undefined;
}

/** /Title from the head/tail metadata region (literal, hex, or UTF-16), filtered for usability. */
function findUsableTitle(latin: string, inflatedText: string): string | undefined {
  const region = latin.slice(0, META_REGION_BYTES) + latin.slice(Math.max(0, latin.length - META_REGION_BYTES));
  const literal = region.match(/\/Title\s*\(((?:\\.|[^()\\]){1,400})\)/i);
  const hex = region.match(/\/Title\s*<([0-9A-Fa-f]{2,})>/i);
  let decoded: string | undefined;
  if (literal) decoded = decodeLiteral(literal[1]!);
  else if (hex) decoded = decodeHexLiteral(hex[1]!);
  if (!decoded) {
    // Some writers only carry the title in XMP dc:title.
    const xmpTitle = inflatedText.match(/<dc:title>(?:<rdf:li[^>]*>)?([^<]{5,300})/i);
    if (xmpTitle) decoded = xmpTitle[1]!;
  }
  if (!decoded) return undefined;
  const title = decoded.replace(/\s+/g, " ").trim();
  if (!isUsableTitle(title)) return undefined;
  return title;
}

function isUsableTitle(title: string): boolean {
  if (title.length < TITLE_MIN_LENGTH || title.length > TITLE_MAX_LENGTH) return false;
  if (!/[A-Za-z]{2}/.test(title)) return false;
  if (/(\.eps|\.ps|\.fig|\.png|\.pdf|\.tex|\.bib)(\s|$)/i.test(title)) return false;
  if (/^(fig(ure)?|table|page|appendix|section)[.\s]*\d/i.test(title)) return false;
  // Section-numbered headings (e.g. "2.2. Cluster-Finding Algorithm").
  if (/^\d+(\.\d+)*[\s.]/.test(title)) return false;
  // Identifier-like single tokens (e.g. "pipeline_diagram") are not titles.
  if (!/\s/.test(title) && title.length < 40) return false;
  if (/\b(pgplot|placeholder|lorem ipsum)\b/i.test(title)) return false;
  if (/^(draft|manuscript|preprint|untitled)\s*$/i.test(title)) return false;
  if (/^[\d\s.,:;!?-]+$/.test(title)) return false;
  return true;
}

function latin1Decode(bytes: Uint8Array): string {
  return new TextDecoder("iso-8859-1").decode(bytes);
}
