import { createHash } from "node:crypto";
import path from "node:path";

/**
 * Obsidian identifies a vault by an opaque key in its app-level config. Deriving
 * it from the vault path keeps repeated harness runs pointing at one entry
 * instead of accumulating duplicates.
 */
function vaultId(normalizedPath) {
  return createHash("sha256").update(normalizedPath).digest("hex").slice(0, 16);
}

function normalizeVaultPath(vaultPath) {
  if (typeof vaultPath !== "string" || vaultPath.length === 0) {
    throw new TypeError("vault path must be a non-empty string");
  }
  if (!path.isAbsolute(vaultPath)) {
    throw new TypeError(`vault path must be absolute: ${vaultPath}`);
  }
  return path.resolve(vaultPath);
}

/**
 * Build the app-level config that mounts exactly one vault. The harness writes
 * this into a disposable XDG config home so the real vault list is never read
 * or rewritten.
 */
export function composeVaultConfig({ vaultPath, timestamp }) {
  const normalized = normalizeVaultPath(vaultPath);
  if (typeof timestamp !== "number" || !Number.isFinite(timestamp)) {
    throw new TypeError("timestamp must be a finite number");
  }
  return {
    vaults: {
      [vaultId(normalized)]: { path: normalized, ts: timestamp, open: true },
    },
  };
}

/**
 * Refuse to run against anything that overlaps the user's real Obsidian
 * configuration — including a parent directory, which would put the real config
 * inside the harness's disposable tree.
 */
export function assertIsolatedConfigHome(configHome, { realConfigHome }) {
  if (typeof configHome !== "string" || configHome.length === 0) {
    throw new TypeError("config home must be a non-empty string");
  }
  if (!path.isAbsolute(configHome)) {
    throw new TypeError(`config home must be absolute: ${configHome}`);
  }
  const candidate = path.resolve(configHome);
  const real = path.resolve(realConfigHome);
  const overlaps =
    candidate === real ||
    candidate.startsWith(`${real}${path.sep}`) ||
    real.startsWith(`${candidate}${path.sep}`);
  if (overlaps) {
    throw new Error(
      `refusing to use ${candidate}: it overlaps the real Obsidian configuration at ${real}`,
    );
  }
  return candidate;
}
