/**
 * Hidden carrier for the "sources the summary drew on" text of one paper.
 *
 * The daily report no longer shows this line, but the paper index and search
 * still need it, so it travels as one HTML comment line right under the paper
 * heading. The text derives from untrusted arXiv section titles, so it is
 * base64url-encoded UTF-8: the payload alphabet cannot terminate the comment
 * or start a new Markdown line.
 */
export const SOURCE_SECTIONS_MARKER_PREFIX = "arxiv-daily-source-sections";
export const SOURCE_SECTIONS_MARKER_MAX_CODE_UNITS = 48_000 as const;
/** Keeps the encoded marker (at most 4/3 of 3 bytes per unit) well inside the marker bound. */
const SOURCE_SECTIONS_MAX_TEXT_CODE_UNITS = 8_000;

const MARKER_RE = new RegExp(`^<!-- ${SOURCE_SECTIONS_MARKER_PREFIX}:v1:([A-Za-z0-9_-]+) -->$`);

/** Render the marker line for already-normalized, single-line text; blank text renders nothing. */
export function renderSourceSectionsMarker(sourceSections: string): string {
  let text = sourceSections.replace(/\s+/gu, " ").trim();
  if (!text) return "";
  if (text.length > SOURCE_SECTIONS_MAX_TEXT_CODE_UNITS) {
    text = text.slice(0, SOURCE_SECTIONS_MAX_TEXT_CODE_UNITS);
    // Never cut a surrogate pair in half.
    if (/[\uD800-\uDBFF]$/.test(text)) text = text.slice(0, -1);
    text = text.trimEnd();
  }
  return `<!-- ${SOURCE_SECTIONS_MARKER_PREFIX}:v1:${encodeBase64Url(text)} -->`;
}

/** Read one marker line back as compacted text; anything not exactly a marker is null. */
export function parseSourceSectionsMarker(line: string): string | null {
  if (line.length > SOURCE_SECTIONS_MARKER_MAX_CODE_UNITS) return null;
  const encoded = MARKER_RE.exec(line)?.[1];
  if (!encoded || encoded.length % 4 === 1) return null;
  let text: string;
  try {
    text = decodeBase64Url(encoded);
  } catch {
    return null;
  }
  text = text.replace(/\s+/gu, " ").trim();
  return text || null;
}

function encodeBase64Url(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function decodeBase64Url(value: string): string {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - value.length % 4) % 4);
  const binary = atob(padded);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
}
