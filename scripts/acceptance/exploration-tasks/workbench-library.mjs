/**
 * P4 Chunk 2 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/04-workbench-library-tasks.md):
 * a small, reusable workbench personal-library fixture seed, mirroring exploration-tasks/library.mjs's
 * Obsidian counterpart (`startLibraryFixture`/`seedClusterableLibraryPaper`) but through the real built
 * CLI (`library connect`/`status`/`authorize`/`prepare`/`scan`/`index`) against the fixture's own
 * isolated XDG config — never the model's own restricted connect-by-path/embedding controls
 * (research.md "Reading workbench — P4 intake": "Connecting by typing a path stays out of scope ...
 * the embedding fields, including mode, stay restricted on the workbench"), and never a hand-written
 * catalog/search-index shape for anything the real CLI itself computes (the authorization fingerprint,
 * the on-disk index).
 *
 * The fixture's own TOML has no `[embedding]` (local mode by default); remote embedding is required so
 * indexing talks to the fixture's own `/embeddings` handler (fixtures.mjs — 3-dimensional vectors, never
 * a ~130 MB local-model download). `library-build`'s own server action always runs prepare/scan/index
 * together (research.md), so the real CLI equivalent here does the same three steps in the same order —
 * proven against the real built CLI binary in the P4 Chunk 1 real-host probe.
 */
import { spawn } from "node:child_process";
import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { writeConfigPatch } from "../exploration-browser.mjs";
import { DEFAULT_CLI_PATH, npmOfflineEnv } from "../library-runtime-cache.mjs";
import { buildFixturePdf } from "../fixtures.mjs";

/** Matches the fixture's own `/embeddings` handler (fixtures.mjs): always 3-dimensional vectors. */
export const WORKBENCH_LIBRARY_EMBEDDING_DIMENSION = 3;
/**
 * The embedding API key field is restricted from the model (exploration-browser.mjs's
 * `classifyControlRestriction` credentials rule), but `validateEmbeddingConfig`
 * (packages/core/src/settings/validation.ts) still rejects a remote config with an empty key before any
 * request is sent — so a real key-shaped fixture value is required here, never the model's own claim of
 * one (the same reasoning exploration-tasks/library.mjs already documents for the Obsidian fixture).
 */
const WORKBENCH_LIBRARY_EMBEDDING_API_KEY = "fixture-embedding-key";
const WORKBENCH_LIBRARY_EMBEDDING_MODEL = "fixture-embedding-model";

/**
 * Spawns the real built CLI with one `library <command>` subcommand against `env` (the fixture's own
 * isolated `XDG_CONFIG_HOME`/`XDG_CACHE_HOME`), parsing its one-line JSON stdout (every `library`
 * subcommand prints exactly one JSON object via its own `print()`, apps/cli/src/library-cmd.ts /
 * library-connection-cmd.ts). This is the only seam `startWorkbenchLibraryFixture` has for actually
 * touching a process or the filesystem outside `writeConfigPatch`, so a test can inject a fake here
 * (`runCli`) and prove the whole seed sequence — ordering, the fingerprint round trip, the TOML it
 * writes, the resync — without ever spawning a real process.
 */
export async function runWorkbenchLibraryCli({ args, env, cliPath = DEFAULT_CLI_PATH, signal } = {}) {
  await access(cliPath).catch(() => { throw new Error("CLI build is missing; run: npm run build --workspace apps/cli"); });
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [cliPath, ...args], { env, stdio: ["ignore", "pipe", "pipe"], signal });
    let stdout = "", stderr = "";
    child.stdout.setEncoding("utf8").on("data", chunk => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", chunk => { stderr = (stderr + chunk).slice(-4000); });
    child.once("error", reject);
    child.once("close", code => {
      if (code !== 0) return reject(new Error(`library ${args.join(" ")} failed (${code}): ${stderr}`));
      try { resolve(JSON.parse(stdout)); }
      catch (error) { reject(new Error(`library ${args.join(" ")} did not print one JSON object: ${error.message}`)); }
    });
  });
}

/**
 * Writes one more synthetic library PDF — the same "measurement"-keyed fixture-vector trick as
 * exploration-tasks/library.mjs's `seedClusterableLibraryPaper` (fixtures.mjs's `/embeddings` handler
 * keys on the substring "measurement") — so a real clustering run groups it with the fixture's own
 * PAPERS[0] into a real two-member group instead of every paper landing in its own singleton outlier
 * (the same no-evidence failure P3 Chunk 3's real-host run hit with only two dissimilar fixture papers).
 */
async function seedClusterableWorkbenchLibraryPaper(fixture, {
  id = "9912.00099",
  title = "Repeated measurement protocols for controlled research",
  abstract = "This study repeats the same controlled measurement protocol to validate a reliable research measurement outcome.",
} = {}) {
  const libraryDir = path.join(fixture.vaultRoot, "library");
  await mkdir(libraryDir, { recursive: true });
  await writeFile(path.join(libraryDir, `${id}.pdf`), buildFixturePdf({ id, title, abstract }));
  return { id, title, abstract };
}

/**
 * Seeds a connected (optionally authorized, optionally scanned, optionally indexed) personal library on
 * `driver`'s own fixture, through the real CLI — never the model's own actions. Order: patch `[embedding]`
 * to remote (so the very first `library connect` already sees a valid, fingerprint-able scope), `library
 * connect` the fixture's own `vault/library` folder, `library status` for the real disclosure fingerprint,
 * then (when `authorized`) `library authorize --fingerprint` against that exact fingerprint, then (when
 * `scanned` or `indexed`) `library prepare`/`scan` with npm forced offline (the pre-installed-runtime
 * contract Chunk 1 already sets up for the real exploration run; here it is only a fail-fast backstop,
 * since this helper never installs anything itself), then (when `indexed`, which requires `authorized` —
 * remote-embedding indexing needs granted processing consent, research.md) `library index`. `scanned`
 * deliberately does not require `authorized`: the real `library scan` CLI command never calls
 * `assertAuthorized` (apps/cli/src/library-cmd.ts) — scanning a library's own folder for a catalog is the
 * one personal-library action the workbench itself documents as not calling the model
 * (apps/cli/src/workbench/web/library.ts's own "浏览目录不调用模型"), so P4 Chunk 3's catalog-filter task
 * seeds a scanned-but-unauthorized library on purpose, to exercise that exact unrestricted path. Ends by
 * resyncing the running workbench server's in-memory config to the disk this just wrote out-of-band
 * (`driver.resyncSettings()` — the same path reset() itself uses), so the next real UI settings save does
 * not 409 against a stale revision.
 */
export async function startWorkbenchLibraryFixture(driver, { authorized = false, scanned = false, indexed = false, clusterable = false, runCli = runWorkbenchLibraryCli, signal } = {}) {
  if (indexed && !authorized) throw new Error("startWorkbenchLibraryFixture: indexed requires authorized — remote-embedding indexing needs granted processing consent");
  const fixture = driver?.fixture;
  if (!fixture?.vaultRoot || !fixture?.configPath || !fixture?.env) throw new Error("startWorkbenchLibraryFixture: driver.fixture with vaultRoot/configPath/env is required");
  const libraryRoot = path.join(fixture.vaultRoot, "library");
  let clusterablePaper;
  if (clusterable) clusterablePaper = await seedClusterableWorkbenchLibraryPaper(fixture);

  await writeConfigPatch(fixture, table => {
    table.embedding = {
      ...table.embedding, mode: "remote",
      base_url: `${fixture.server.url}/v1`, api_key: WORKBENCH_LIBRARY_EMBEDDING_API_KEY,
      model: WORKBENCH_LIBRARY_EMBEDDING_MODEL, dimension: WORKBENCH_LIBRARY_EMBEDDING_DIMENSION,
    };
  });

  await runCli({ args: ["library", "connect", libraryRoot], env: fixture.env, signal });
  const status = await runCli({ args: ["library", "status"], env: fixture.env, signal });
  const fingerprint = status?.disclosure?.authorizationFingerprint;
  if (!fingerprint) throw new Error("startWorkbenchLibraryFixture: `library status` reported no disclosure fingerprint to authorize against");

  if (authorized) {
    await runCli({ args: ["library", "authorize", "--fingerprint", fingerprint], env: fixture.env, signal });
  }
  if (scanned || indexed) {
    const offlineEnv = npmOfflineEnv(fixture.env);
    await runCli({ args: ["library", "prepare"], env: offlineEnv, signal });
    await runCli({ args: ["library", "scan"], env: offlineEnv, signal });
    if (indexed) await runCli({ args: ["library", "index"], env: offlineEnv, signal });
  }

  await driver.resyncSettings();
  return { libraryRoot, fingerprint, ...(clusterablePaper ? { clusterablePaper } : {}) };
}
