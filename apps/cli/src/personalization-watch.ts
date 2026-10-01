import * as fs from "node:fs/promises";
import { watch, type FSWatcher } from "node:fs";
import * as path from "node:path";
import {
  createPersonalLibraryScopeFingerprint, createPersonalLibraryIdentificationFingerprint,
  derivePersonalLibraryCatalogPaths, derivePersonalLibraryInterestProfileStorePaths,
  sha256Hex, type StorageAdapter, type Logger, type PersonalizedDailyDiscoverySnapshot,
} from "@arxiv-daily/core";
import type { CliRuntimeConfig } from "./config";
import { loadCliPersonalizedDiscovery } from "./library-cmd";

/** Keep the independent daily task's captured library authorization/evidence current. */
export async function watchCliPersonalization(
  config: CliRuntimeConfig,
  storage: StorageAdapter,
  snapshot: PersonalizedDailyDiscoverySnapshot,
  controller: AbortController,
  logger: Logger,
): Promise<() => void> {
  const connection = config.libraryConnection!;
  const scope = createPersonalLibraryScopeFingerprint(connection);
  const identity = createPersonalLibraryIdentificationFingerprint(connection.eligibleExtensions);
  const profilePaths = derivePersonalLibraryInterestProfileStorePaths(storage, config.settings.output, scope, identity);
  const catalogPaths = derivePersonalLibraryCatalogPaths(storage, config.settings.output);
  const watchedFiles = [
    await fs.realpath(config.configPath),
    path.resolve(config.vaultRoot, profilePaths.profile.documentPath),
    path.resolve(config.vaultRoot, catalogPaths.documentPath),
  ];
  const directories = new Map<string, Set<string>>();
  for (const file of watchedFiles) {
    const directory = path.dirname(file);
    if (!directories.has(directory)) directories.set(directory, new Set());
    directories.get(directory)!.add(path.basename(file));
  }
  const watchers: FSWatcher[] = [];
  const fingerprint = JSON.stringify(snapshot);
  let disposed = false;
  let checking = false;
  let dirty = false;
  const close = () => { disposed = true; for (const watcher of watchers) watcher.close(); };
  const check = async () => {
    dirty = true;
    if (checking || disposed) return;
    checking = true;
    try {
      while (dirty && !disposed && !controller.signal.aborted) {
        dirty = false;
        if (config.configRevision && `sha256:${sha256Hex(await fs.readFile(config.configPath, "utf8"))}` !== config.configRevision) {
          throw new Error("Personalized daily task cancelled because its configuration or authorization changed");
        }
        const current = await loadCliPersonalizedDiscovery(config, storage, logger);
        if (JSON.stringify(current) !== fingerprint) throw new Error("Personalized daily task cancelled because its library evidence changed");
      }
    } catch (error) {
      if (!disposed) controller.abort(error);
    } finally { checking = false; }
  };
  try {
    for (const [directory, names] of directories) {
      const watcher = watch(directory, { persistent: false }, (_event, filename) => {
        if (filename === null || names.has(String(filename))) void check();
      });
      watcher.on("error", error => { if (!disposed) controller.abort(error); });
      watchers.push(watcher);
    }
    await check();
    return close;
  } catch (error) {
    close();
    controller.abort(error);
    return close;
  }
}
