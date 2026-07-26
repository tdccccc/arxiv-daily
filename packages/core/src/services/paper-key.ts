import { modernArxivResources } from "../utils/arxiv";

/** Source segment of a paperKey: lowercase `[a-z0-9_]+`. */
const SOURCE_RE = /^[a-z0-9_]+$/;

/**
 * Stable paper identity: `source:externalId` (e.g. `arxiv:2606.12345`).
 * Disk note/PDF paths use the short externalId stem only — never this full key.
 */
export type PaperKeyParts = {
  source: string;
  externalId: string;
};

export class PaperKeyError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "PaperKeyError";
  }
}

export function isValidSource(source: string): boolean {
  return SOURCE_RE.test(source);
}

/** Format a canonical paperKey. Source is lowercased; externalId is not re-normalized here. */
export function formatPaperKey(source: string, externalId: string): string {
  const normalizedSource = source.trim().toLowerCase();
  const normalizedExternalId = externalId.trim();
  if (!normalizedSource || !isValidSource(normalizedSource)) {
    throw new PaperKeyError(
      `invalid paper source: ${JSON.stringify(source)} (expected lowercase [a-z0-9_]+)`,
    );
  }
  if (!normalizedExternalId) {
    throw new PaperKeyError("invalid paper externalId: empty");
  }
  if (normalizedExternalId.includes(":")) {
    throw new PaperKeyError(
      `invalid paper externalId: ${JSON.stringify(externalId)} (must not contain ':')`,
    );
  }
  return `${normalizedSource}:${normalizedExternalId}`;
}

/** Parse `source:externalId`. Throws PaperKeyError on invalid input. */
export function parsePaperKey(key: string): PaperKeyParts {
  const parsed = tryParsePaperKey(key);
  if (!parsed) {
    throw new PaperKeyError(`invalid paperKey: ${JSON.stringify(key)}`);
  }
  return parsed;
}

/** Parse `source:externalId`, or null if invalid. */
export function tryParsePaperKey(key: string): PaperKeyParts | null {
  const trimmed = key.normalize("NFC").trim();
  if (!trimmed) return null;
  const colon = trimmed.indexOf(":");
  if (colon <= 0 || colon === trimmed.length - 1) return null;
  const source = trimmed.slice(0, colon).toLowerCase();
  const externalId = trimmed.slice(colon + 1).trim();
  if (!isValidSource(source) || !externalId || externalId.includes(":")) {
    return null;
  }
  // Reject uppercase source in the original (source must be lowercase in stored keys).
  if (trimmed.slice(0, colon) !== source) return null;
  return { source, externalId };
}

/**
 * Build an arXiv paperKey from a bare modern id, `arxiv:…`, or trusted arXiv URL.
 * externalId uses the same modern-id normalization as `modernArxivResources`.
 */
export function paperKeyFromArxivId(bareOrUrl: string): string {
  const resources = modernArxivResources(bareOrUrl);
  if (!resources) {
    throw new PaperKeyError(`invalid arXiv ID: ${JSON.stringify(bareOrUrl)}`);
  }
  return formatPaperKey("arxiv", resources.id);
}

/**
 * Resolve a store lookup input to a canonical paperKey.
 * Accepts a full paperKey or a bare modern arXiv id / arXiv URL (compat).
 */
export function resolvePaperLookupKey(input: string): string {
  const trimmed = input.normalize("NFC").trim();
  if (!trimmed) {
    throw new PaperKeyError("invalid paper lookup key: empty");
  }

  const asPaperKey = tryParsePaperKey(trimmed);
  if (asPaperKey) {
    if (asPaperKey.source === "arxiv") {
      const resources = modernArxivResources(asPaperKey.externalId);
      if (!resources) {
        throw new PaperKeyError(
          `invalid arXiv paperKey: ${JSON.stringify(trimmed)}`,
        );
      }
      return formatPaperKey("arxiv", resources.id);
    }
    return formatPaperKey(asPaperKey.source, asPaperKey.externalId);
  }

  const resources = modernArxivResources(trimmed);
  if (resources) {
    return formatPaperKey("arxiv", resources.id);
  }

  throw new PaperKeyError(
    `invalid paper lookup key: ${JSON.stringify(input)} (expected paperKey or bare arXiv id)`,
  );
}

/**
 * Short path stem for notes/PDFs. Always the source-local externalId — never
 * a paperKey string (which contains `:` and must not appear in filenames).
 */
export function paperPathStem(source: string, externalId: string): string {
  if (source === "arxiv") {
    const resources = modernArxivResources(externalId);
    if (!resources) {
      throw new PaperKeyError(`invalid arXiv externalId for path stem: ${JSON.stringify(externalId)}`);
    }
    return resources.id;
  }
  const stem = externalId.trim();
  if (!stem || stem.includes("/") || stem.includes(":") || stem.includes("\\")) {
    throw new PaperKeyError(
      `invalid path stem externalId: ${JSON.stringify(externalId)}`,
    );
  }
  return stem;
}
