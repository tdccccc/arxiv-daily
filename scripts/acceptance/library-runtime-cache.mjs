// P4 Chunk 1 of docs/helm/2026-10-13-exploration-safety-and-coverage: the Reading workbench's
// personal-library tasks need a prepared PDF runtime before a model ever touches the workbench, so
// `library-build` never spawns `npm ci` against the real registry at model time (research.md
// "Reading workbench — P4 intake"). This module owns exactly that: a persistent, test-only,
// version-keyed cache OUTSIDE the repo and outside the user's real `~/.config/arxiv-daily` /
// `~/.cache/arxiv-daily`, prepared once with network via the real built CLI's `library prepare` in
// remote-embedding mode (so `@huggingface/transformers` is never installed here), then copied — real
// bytes, `.prepared` marker included — into one fixture's own `cache_dir` before its workbench server
// starts. Reused by `exploration.mjs` for the exploration workbench host only; the obsidian host and
// the fixed `workbench.mjs` suite never call `library-build`, so neither needs this.
import { randomBytes, createHash } from "node:crypto";
import { spawn } from "node:child_process";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
export const DEFAULT_CLI_PATH = path.join(repoRoot, "apps/cli/dist/arxiv-daily-cli.cjs");
// Must track packages/node-runtime/src/library-runtime.ts's `nodeLibraryRuntimePaths` fixed leaf name
// (not version-keyed there — the version keying happens one level up, in the cache root this module
// owns) so the directory this copies from/into is exactly the one that runtime prepares and reads.
const PDF_RUNTIME_LEAF = "pdfjs-5.4.624";

async function readJson(file) {
  return JSON.parse(await fs.readFile(file, "utf8"));
}

/** Same fingerprint shape the runtime itself hashes (manifest + lockfile), used here only to key the cache directory name. */
export async function pdfRuntimeFingerprint() {
  const [manifest, lockfile] = await Promise.all([
    readJson(path.join(repoRoot, "tools/node-library-runtime/pdf/package.json")),
    readJson(path.join(repoRoot, "tools/node-library-runtime/pdf/package-lock.json")),
  ]);
  const version = manifest.dependencies?.["pdfjs-dist"];
  if (typeof version !== "string" || !version.trim()) throw new Error("Pinned pdfjs-dist version is missing from tools/node-library-runtime/pdf/package.json");
  const hash = createHash("sha256").update(JSON.stringify({ manifest, lockfile })).digest("hex").slice(0, 16);
  return { version, hash };
}

/**
 * `${XDG_CACHE_HOME:-~/.cache}/arxiv-daily-acceptance/runtimes/pdfjs-<version>-<lockfile hash>` — a
 * directory this module passes as `cache_dir` to a throwaway CLI config, so the real prepared runtime
 * lands one level under it at `.../runtimes/pdfjs-5.4.624` (the fixed leaf above). Deliberately reads
 * `env` (defaulting to the real `process.env`, not any fixture's) so a fixture's own isolated
 * `XDG_CACHE_HOME` never redirects this cache into a throwaway temp root that gets deleted every run.
 */
export async function persistentLibraryRuntimeCacheRoot({ env = process.env } = {}) {
  const base = env.XDG_CACHE_HOME?.trim() || path.join(os.homedir(), ".cache");
  const { version, hash } = await pdfRuntimeFingerprint();
  return path.join(base, "arxiv-daily-acceptance", "runtimes", `pdfjs-${version}-${hash}`);
}

/**
 * Spawns the real built CLI's `library prepare` against an isolated, throwaway config whose
 * `cache_dir` is the persistent cache root and whose `[embedding]` mode is `"remote"` (so the runtime
 * never installs the local embedding model). This is the only call in the exploration path allowed to
 * reach the network, and only on a cache miss: `prepareNodeLibraryRuntime`'s own `.prepared` fingerprint
 * check (packages/node-runtime/src/library-runtime.ts:47-50) makes every later call a cheap no-op.
 */
export async function runLibraryPrepare({ cacheDir, cliPath = DEFAULT_CLI_PATH, signal } = {}) {
  await fs.access(cliPath).catch(() => { throw new Error("CLI build is missing; run: npm run build --workspace apps/cli"); });
  const configHome = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-daily-runtime-prepare-"));
  try {
    const vaultRoot = path.join(configHome, "vault");
    await fs.mkdir(path.join(configHome, "arxiv-daily"), { recursive: true });
    await fs.mkdir(vaultRoot, { recursive: true });
    const config = `schema_version = 1\nvault_root = ${JSON.stringify(vaultRoot)}\ncache_dir = ${JSON.stringify(cacheDir)}\n\n[embedding]\nmode = "remote"\n`;
    await fs.writeFile(path.join(configHome, "arxiv-daily", "config.toml"), config, { mode: 0o600 });
    await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, [cliPath, "library", "prepare"], {
        env: { ...process.env, XDG_CONFIG_HOME: configHome }, stdio: ["ignore", "pipe", "pipe"], signal,
      });
      let stderr = "";
      child.stderr.setEncoding("utf8").on("data", chunk => { stderr = (stderr + chunk).slice(-4000); });
      child.once("error", reject);
      child.once("close", code => (code === 0 ? resolve() : reject(new Error(`library prepare failed (${code}): ${stderr}`))));
    });
  } finally {
    await fs.rm(configHome, { recursive: true, force: true });
  }
}

/**
 * Ensures the persistent, version-keyed cache holds a prepared pdfjs runtime and returns its directory
 * (the one to copy from). `runPrepare` is a test seam: real callers get the real CLI spawn above; tests
 * inject a fake that never touches the network, so creation, reuse and rebuild-on-mismatch are all
 * provable without downloading anything. Concurrency: two overlapping calls for the *same* cache root
 * both delegate to the same `runLibraryPrepare`, which shares `prepareNodeLibraryRuntime`'s own
 * `NodeFileLock`-guarded install (packages/node-runtime/src/library-runtime.ts:43-57) — the second caller
 * waits for the first's install, then re-reads the now-matching `.prepared` marker instead of reinstalling.
 */
export async function ensurePersistentPdfRuntime({ env, signal, runPrepare = runLibraryPrepare } = {}) {
  const cacheRoot = await persistentLibraryRuntimeCacheRoot({ env });
  await fs.mkdir(cacheRoot, { recursive: true, mode: 0o700 });
  await runPrepare({ cacheDir: cacheRoot, signal });
  const pdfDir = path.join(cacheRoot, "runtimes", PDF_RUNTIME_LEAF);
  await fs.access(path.join(pdfDir, ".prepared")).catch(() => {
    throw new Error("Library runtime preparation did not leave a `.prepared` runtime at " + pdfDir);
  });
  return pdfDir;
}

/**
 * Copies the prepared runtime (real bytes, `.prepared` included — never a symlink; the runtime itself
 * refuses a symlinked leaf) into `fixture.cacheDir/runtimes/pdfjs-5.4.624`, so `library-build` on that
 * fixture's workbench finds it already installed. Copies into a sibling temp directory first and
 * `rename()`s it into place, so a process killed mid-copy never leaves a partial `pdfjs-5.4.624` that a
 * later `installed()` check would wrongly treat as real — the runtime's own `.prepared`/`installed()`
 * check (library-runtime.ts:50,64-73) is what actually detects and rebuilds a corrupted/partial cache;
 * this only makes sure the fixture-side copy itself is all-or-nothing.
 */
export async function seedWorkbenchLibraryRuntime(fixture, { env, signal, ensure = ensurePersistentPdfRuntime } = {}) {
  if (!fixture?.cacheDir) throw new Error("A fixture with its own cache_dir is required to seed the library runtime");
  const sourceDir = await ensure({ env, signal });
  const runtimesDir = path.join(fixture.cacheDir, "runtimes");
  const dest = path.join(runtimesDir, PDF_RUNTIME_LEAF);
  await fs.mkdir(runtimesDir, { recursive: true, mode: 0o700 });
  const tmp = path.join(runtimesDir, `.${PDF_RUNTIME_LEAF}.tmp-${process.pid}-${randomBytes(4).toString("hex")}`);
  await fs.rm(tmp, { recursive: true, force: true });
  try {
    await fs.cp(sourceDir, tmp, { recursive: true });
    await fs.rm(dest, { recursive: true, force: true });
    await fs.rename(tmp, dest);
  } finally {
    await fs.rm(tmp, { recursive: true, force: true });
  }
  return dest;
}

/**
 * The environment overlay for a spawned workbench server during exploration: npm offline so any
 * accidental `npm ci` (a prepare that somehow misses the seeded cache) fails fast instead of reaching
 * the real registry. `npm_config_offline` is npm's own env form of `--offline`; `fetch_retries=0` keeps
 * a forced failure fast rather than retried.
 */
export function npmOfflineEnv(env = process.env) {
  return { ...env, npm_config_offline: "true", npm_config_fetch_retries: "0" };
}
