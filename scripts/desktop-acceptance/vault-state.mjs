import fsPromises from "node:fs/promises";
import path from "node:path";

const INTERRUPT_SIGNALS = { SIGINT: 130, SIGTERM: 143 };

/**
 * The mutable state a launch rewrites: the plugin's settings store and
 * Obsidian's workspace layout. Historical build backups sitting beside the
 * settings store are deliberately not listed — the harness must not read or
 * rewrite them.
 */
export function vaultStatePaths(vaultPath, { pluginId }) {
  const configDir = path.join(vaultPath, ".obsidian");
  return [
    path.join(configDir, "plugins", pluginId, "data.json"),
    path.join(configDir, "workspace.json"),
  ];
}

/**
 * Captures the protected files before a run and puts them back afterwards on
 * every exit path — success, throw, and interrupting signal. A file that was
 * absent at capture time is removed again rather than left behind.
 */
export function createVaultStateGuard({
  vaultPath,
  pluginId,
  fs = fsPromises,
  process: proc = process,
  additionalPaths = [],
}) {
  if (typeof vaultPath !== "string" || !path.isAbsolute(vaultPath)) {
    throw new TypeError(`vault path must be absolute: ${String(vaultPath)}`);
  }
  const vaultRoot = path.resolve(vaultPath);
  const extra = additionalPaths.map((candidate) => {
    if (typeof candidate !== "string" || !path.isAbsolute(candidate)) {
      throw new TypeError(`protected path must be absolute: ${String(candidate)}`);
    }
    const resolved = path.resolve(candidate);
    if (!resolved.startsWith(`${vaultRoot}${path.sep}`)) {
      throw new Error(`protected path must be inside the vault: ${resolved}`);
    }
    return resolved;
  });
  const paths = [...vaultStatePaths(vaultPath, { pluginId }), ...extra];
  let snapshot = null;
  let installed = null;

  async function capture() {
    const entries = [];
    for (const target of paths) {
      try {
        entries.push([target, await fs.readFile(target)]);
      } catch (error) {
        if (error?.code !== "ENOENT") throw error;
        entries.push([target, null]);
      }
    }
    snapshot = entries;
    return snapshot;
  }

  async function restore() {
    if (snapshot === null) {
      throw new Error("vault state was not captured; refusing to restore");
    }
    for (const [target, content] of snapshot) {
      if (content === null) await fs.rm(target, { force: true });
      else await fs.writeFile(target, content);
    }
  }

  function removeSignalRestore() {
    if (installed === null) return;
    for (const [signal, handler] of installed) proc.off(signal, handler);
    installed = null;
  }

  function installSignalRestore() {
    if (installed !== null) return;
    installed = Object.entries(INTERRUPT_SIGNALS).map(([signal, exitCode]) => {
      const handler = async () => {
        // A signal arrives once; restore best-effort so an interrupted run can
        // never leave the vault rewritten, then stop being a handler.
        try {
          await restore();
        } catch {
          /* the exit code below already reports an abnormal end */
        }
        removeSignalRestore();
        proc.exit(exitCode);
      };
      proc.on(signal, handler);
      return [signal, handler];
    });
  }

  async function protect(body) {
    await capture();
    installSignalRestore();
    let bodyError;
    let result;
    try {
      result = await body();
    } catch (error) {
      bodyError = error;
    } finally {
      removeSignalRestore();
    }
    try {
      await restore();
    } catch (restoreError) {
      // A restore failure must not hide why the run actually failed.
      if (bodyError === undefined) throw restoreError;
    }
    if (bodyError !== undefined) throw bodyError;
    return result;
  }

  return { capture, restore, protect, installSignalRestore, removeSignalRestore, paths };
}
