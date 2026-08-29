import {
  buildChatCompletionsUrl,
  buildCheckpointEndpointDigest,
  sha256Hex,
  type LibraryInventory,
} from "@arxiv-daily/core";

export const LIBRARY_CONNECTION_SCHEMA_VERSION = 1 as const;
/** Processing depths a library connection can be authorized at (ADR 0008). */
export const LIBRARY_PROCESSING_DEPTHS = ["metadata-and-abstracts", "full-text"] as const;
export type LibraryProcessingDepth = (typeof LIBRARY_PROCESSING_DEPTHS)[number];
export const LIBRARY_PROCESSING_DEPTH = LIBRARY_PROCESSING_DEPTHS[0];
export const LIBRARY_ELIGIBLE_EXTENSIONS = [".pdf"] as const;

/**
 * What a library authorization grant covers: the LLM endpoint, plus the
 * embedding endpoint when remote embedding is enabled (ADR 0008). Changing
 * any endpoint or the processing depth invalidates the grant.
 */
export interface LibraryAuthorizationScope {
  /** LLM chat endpoint base URL. */
  llmBaseUrl: string;
  /** Embedding endpoint base URL when remote embedding is enabled. */
  embeddingEndpoint?: { baseUrl: string };
}

export interface PersistedLibraryAuthorization {
  fingerprint: string;
  grantedAt: string;
}

export interface PersistedLibraryConnection {
  schemaVersion: typeof LIBRARY_CONNECTION_SCHEMA_VERSION;
  selectedRoot: string;
  rootIdentity: string;
  eligibleExtensions: string[];
  processingDepth: LibraryProcessingDepth;
  authorization?: PersistedLibraryAuthorization;
}

export type LibraryConnectionStatus =
  | { kind: "disconnected" }
  | { kind: "authorization-required"; rootLabel: string }
  | { kind: "authorization-invalidated"; rootLabel: string }
  | { kind: "authorized"; rootLabel: string; grantedAt: string };

export interface LibraryAuthorizationDisclosure {
  selectedRoot: string;
  eligibleExtensions: string[];
  processingDepth: LibraryProcessingDepth;
  endpoint: string;
  /** Present when remote embedding is enabled (full-text depth). */
  embeddingEndpoint?: string;
  authorizationFingerprint: string;
}

export interface LibraryInventoryPreview {
  eligible: Array<{ path: string; size?: number }>;
  ignored: Array<{ path: string; reason: string }>;
  folders: number;
  truncated: boolean;
}

export function createLibraryConnection(
  selectedRoot: string,
  rootIdentity: string,
): PersistedLibraryConnection {
  return {
    schemaVersion: LIBRARY_CONNECTION_SCHEMA_VERSION,
    selectedRoot,
    rootIdentity,
    eligibleExtensions: [...LIBRARY_ELIGIBLE_EXTENSIONS],
    processingDepth: LIBRARY_PROCESSING_DEPTH,
  };
}

export function decodeLibraryConnection(value: unknown): PersistedLibraryConnection | undefined {
  if (!isRecord(value) || value.schemaVersion !== LIBRARY_CONNECTION_SCHEMA_VERSION) {
    return undefined;
  }
  if (
    typeof value.selectedRoot !== "string"
    || !value.selectedRoot.trim()
    || typeof value.rootIdentity !== "string"
    || !/^\d+:\d+$/.test(value.rootIdentity)
    || !isLibraryProcessingDepth(value.processingDepth)
    || !Array.isArray(value.eligibleExtensions)
    || !value.eligibleExtensions.every((entry) => typeof entry === "string")
  ) {
    return undefined;
  }
  const eligibleExtensions = normalizeExtensions(value.eligibleExtensions);
  if (eligibleExtensions.length === 0) return undefined;
  const authorization = decodeAuthorization(value.authorization);
  return {
    schemaVersion: LIBRARY_CONNECTION_SCHEMA_VERSION,
    selectedRoot: value.selectedRoot,
    rootIdentity: value.rootIdentity,
    eligibleExtensions,
    processingDepth: value.processingDepth,
    ...(authorization ? { authorization } : {}),
  };
}

export function libraryConnectionStatus(
  connection: PersistedLibraryConnection | undefined,
  scope: LibraryAuthorizationScope,
): LibraryConnectionStatus {
  if (!connection) return { kind: "disconnected" };
  const rootLabel = libraryRootLabel(connection.selectedRoot);
  if (!connection.authorization) {
    return { kind: "authorization-required", rootLabel };
  }
  // Status evaluation must be total: an endpoint that can no longer be
  // digested (invalid URL shape after a settings change or external edit)
  // invalidates the grant instead of throwing into settings and run paths.
  let fingerprint: string;
  try {
    fingerprint = libraryAuthorizationFingerprint(connection, scope);
  } catch {
    return { kind: "authorization-invalidated", rootLabel };
  }
  if (connection.authorization.fingerprint !== fingerprint) {
    return { kind: "authorization-invalidated", rootLabel };
  }
  return {
    kind: "authorized",
    rootLabel,
    grantedAt: connection.authorization.grantedAt,
  };
}

export function authorizeLibraryConnection(
  connection: PersistedLibraryConnection,
  scope: LibraryAuthorizationScope,
  now = new Date(),
): PersistedLibraryConnection {
  return {
    ...connection,
    // Remote embedding processes full text, so its grants are full-text
    // depth (ADR 0008); local-only grants stay at metadata and abstracts.
    processingDepth: scope.embeddingEndpoint ? "full-text" : LIBRARY_PROCESSING_DEPTH,
    authorization: {
      fingerprint: libraryAuthorizationFingerprint(connection, scope),
      grantedAt: now.toISOString(),
    },
  };
}

export function revokeLibraryConnection(
  connection: PersistedLibraryConnection,
): PersistedLibraryConnection {
  const { authorization: _authorization, ...withoutAuthorization } = connection;
  return withoutAuthorization;
}

export function libraryAuthorizationDisclosure(
  connection: PersistedLibraryConnection,
  scope: LibraryAuthorizationScope,
): LibraryAuthorizationDisclosure {
  return {
    selectedRoot: connection.selectedRoot,
    eligibleExtensions: [...connection.eligibleExtensions],
    processingDepth: connection.processingDepth,
    endpoint: displayChatEndpoint(scope.llmBaseUrl),
    ...(scope.embeddingEndpoint
      ? { embeddingEndpoint: displayChatEndpoint(scope.embeddingEndpoint.baseUrl) }
      : {}),
    authorizationFingerprint: libraryAuthorizationFingerprint(connection, scope),
  };
}

export function libraryAuthorizationFingerprint(
  connection: PersistedLibraryConnection,
  scope: LibraryAuthorizationScope,
): string {
  const input = {
    version: 1,
    rootDigest: `sha256:${sha256Hex(
      `${connection.selectedRoot}\0${connection.rootIdentity}`,
    )}`,
    eligibleExtensions: normalizeExtensions(connection.eligibleExtensions),
    processingDepth: scope.embeddingEndpoint ? "full-text" : connection.processingDepth,
    endpointDigest: buildCheckpointEndpointDigest(scope.llmBaseUrl),
    ...(scope.embeddingEndpoint
      ? { embeddingEndpointDigest: buildCheckpointEndpointDigest(scope.embeddingEndpoint.baseUrl) }
      : {}),
  };
  return `sha256:${sha256Hex(JSON.stringify(input))}`;
}

export function buildLibraryInventoryPreview(
  inventory: LibraryInventory,
  eligibleExtensions: readonly string[],
): LibraryInventoryPreview {
  const allowed = new Set(normalizeExtensions(eligibleExtensions));
  const preview: LibraryInventoryPreview = {
    eligible: [],
    ignored: [],
    folders: 0,
    truncated: inventory.truncated,
  };
  for (const entry of inventory.entries) {
    if (entry.type === "folder") {
      preview.folders += 1;
      continue;
    }
    if (entry.type === "ignored") {
      preview.ignored.push({
        path: entry.path,
        reason: entry.ignoredReason === "symbolic-link"
          ? "Symbolic link"
          : "Unsupported filesystem entry",
      });
      continue;
    }
    const extension = fileExtension(entry.path);
    if (allowed.has(extension)) {
      preview.eligible.push({
        path: entry.path,
        ...(entry.size === undefined ? {} : { size: entry.size }),
      });
    } else {
      preview.ignored.push({ path: entry.path, reason: "Unsupported file type" });
    }
  }
  return preview;
}

function displayChatEndpoint(baseUrl: string): string {
  const url = new URL(buildChatCompletionsUrl(baseUrl));
  url.username = "";
  url.password = "";
  for (const key of Array.from(url.searchParams.keys())) {
    url.searchParams.set(key, "[redacted]");
  }
  url.hash = "";
  return url.toString();
}

function normalizeExtensions(extensions: readonly string[]): string[] {
  return Array.from(new Set(extensions.map((entry) => entry.trim().toLowerCase())))
    .filter((entry) => /^\.[a-z0-9]+$/.test(entry))
    .sort();
}

function fileExtension(path: string): string {
  const name = path.split("/").at(-1) ?? "";
  const index = name.lastIndexOf(".");
  return index > 0 ? name.slice(index).toLowerCase() : "";
}

function libraryRootLabel(selectedRoot: string): string {
  return selectedRoot.replace(/[\\/]+$/, "").split(/[\\/]/).at(-1) || "Selected folder";
}

function decodeAuthorization(value: unknown): PersistedLibraryAuthorization | undefined {
  if (!isRecord(value)) return undefined;
  if (
    typeof value.fingerprint !== "string"
    || !/^sha256:[0-9a-f]{64}$/.test(value.fingerprint)
    || typeof value.grantedAt !== "string"
    || !Number.isFinite(Date.parse(value.grantedAt))
  ) {
    return undefined;
  }
  return { fingerprint: value.fingerprint, grantedAt: value.grantedAt };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isLibraryProcessingDepth(value: unknown): value is LibraryProcessingDepth {
  return value === "metadata-and-abstracts" || value === "full-text";
}
