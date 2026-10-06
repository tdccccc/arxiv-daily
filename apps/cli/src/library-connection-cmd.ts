import * as fs from "node:fs/promises";
import * as path from "node:path";
import { parse, stringify } from "smol-toml";
import {
  authorizeLibraryConnection,
  createLibraryConnection,
  libraryAuthorizationDisclosure,
  libraryConnectionStatus,
  revokeLibraryConnection,
  sha256Hex,
  type LibraryAuthorizationDisclosure,
  type LibraryAuthorizationScope,
  type LibraryConnectionStatus,
  type PersistedLibraryConnection,
} from "@arxiv-daily/core";
import { NodeStorageAdapter, NodeFileLock, openScopedLibrarySource } from "@arxiv-daily/node-runtime";
import { CliConfigError, loadCliConfig, type CliRuntimeConfig } from "./config";

export interface CliLibraryConnectionInspection {
  status: LibraryConnectionStatus;
  disclosure: LibraryAuthorizationDisclosure | null;
  diagnostic?: string;
}

export async function connectCliLibrary(config: CliRuntimeConfig, selectedRoot: string): Promise<CliRuntimeConfig> {
  return mutateConnection(config, async () => {
    const source = await openScopedLibrarySource(selectedRoot);
    return createLibraryConnection(source.canonicalRoot, source.rootIdentity);
  });
}

export async function authorizeCliLibrary(config: CliRuntimeConfig, fingerprint: string): Promise<CliRuntimeConfig> {
  return mutateConnection(config, async (current) => {
    const connection = requireConnection(current);
    const inspection = inspectCliLibraryConnection(current);
    if (!inspection.disclosure) {
      throw new CliConfigError("Library model endpoint is invalid; fix its settings before authorizing processing.");
    }
    if (inspection.disclosure.authorizationFingerprint !== fingerprint) {
      throw new CliConfigError("Library disclosure fingerprint changed or does not match; inspect the current disclosure before authorizing.");
    }
    const source = await openScopedLibrarySource(connection.selectedRoot);
    if (source.canonicalRoot !== connection.selectedRoot || source.rootIdentity !== connection.rootIdentity) {
      throw new CliConfigError("Library folder identity changed; reconnect the library before authorizing processing.");
    }
    return authorizeLibraryConnection(connection, authorizationScope(current));
  });
}

export async function revokeCliLibrary(config: CliRuntimeConfig): Promise<CliRuntimeConfig> {
  return mutateConnection(config, async (current) => revokeLibraryConnection(requireConnection(current)));
}

export function inspectCliLibraryConnection(config: CliRuntimeConfig): CliLibraryConnectionInspection {
  const scope = authorizationScope(config);
  const status = libraryConnectionStatus(config.libraryConnection, scope);
  const diagnostic = config.libraryConnectionError;
  if (!config.libraryConnection) {
    return { status, disclosure: null, ...(diagnostic ? { diagnostic } : {}) };
  }
  try {
    return {
      status,
      disclosure: libraryAuthorizationDisclosure(config.libraryConnection, scope),
      ...(diagnostic ? { diagnostic } : {}),
    };
  } catch {
    return { status, disclosure: null, diagnostic: "Library model endpoint is invalid; fix its settings before authorizing processing." };
  }
}

function authorizationScope(config: CliRuntimeConfig): LibraryAuthorizationScope {
  return {
    llmBaseUrl: config.settings.llm.baseUrl,
    ...(config.settings.embedding.mode === "remote"
      ? { embeddingEndpoint: { baseUrl: config.settings.embedding.baseUrl } }
      : {}),
  };
}

function requireConnection(config: CliRuntimeConfig): PersistedLibraryConnection {
  if (!config.libraryConnection) throw new CliConfigError("Connect a library before changing its authorization.");
  return config.libraryConnection;
}

async function mutateConnection(
  config: CliRuntimeConfig,
  update: (current: CliRuntimeConfig) => Promise<PersistedLibraryConnection>,
): Promise<CliRuntimeConfig> {
  if (!config.configRevision) {
    throw new CliConfigError("Reload the CLI configuration to obtain its revision before changing the library connection.");
  }
  const target = await fs.realpath(config.configPath);
  const directory = path.dirname(target);
  const fileName = path.basename(target);
  const locks = new NodeFileLock(directory, { lockRoot: path.join(directory, ".arxiv-daily-config-locks") });
  const lease = await locks.acquire(`cli-config:${fileName}`, { wait: true });
  if (!lease) throw new CliConfigError("CLI configuration is being changed; reload and try again.");
  try {
    const raw = await fs.readFile(target, "utf8");
    assertRevision(raw, config.configRevision);
    // Model settings and the prior connection come from the locked durable
    // document, never from mutable caller-owned configuration objects.
    const current = await loadCliConfig({ configPath: config.configPath, readText: async () => raw });
    const next = await update(current);
    const content = stringify({ ...parse(raw), library: next });
    // Also catch an editor changing the file while directory validation ran.
    assertRevision(await fs.readFile(target, "utf8"), config.configRevision);
    await new NodeStorageAdapter(directory).writeTextAtomic(fileName, content, 0o600);
    return await loadCliConfig({ configPath: config.configPath });
  } finally {
    await lease.release();
  }
}

function assertRevision(raw: string, expected: string): void {
  if (`sha256:${sha256Hex(raw)}` !== expected) {
    throw new CliConfigError("CLI configuration changed since it was loaded; reload it before changing the library connection.");
  }
}
