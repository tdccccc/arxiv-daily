// P4 Chunk 1 of docs/helm/2026-10-13-exploration-safety-and-coverage: the Reading workbench's
// personal-library tasks must never let a model's actions spawn `npm ci` against the real registry.
// This covers the baseline problem (a fresh exploration fixture has no prepared runtime and an
// unrestricted npm environment), the cache helper that fixes it (scripts/acceptance/
// library-runtime-cache.mjs), and exploration.mjs's opt-in wiring of that helper for the workbench
// host only.
import assert from "node:assert/strict";
import { access, lstat, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { createFixtureEnvironment } from "../acceptance/fixtures.mjs";
import { runExploration } from "../acceptance/exploration.mjs";
import {
  ensurePersistentPdfRuntime, npmOfflineEnv, persistentLibraryRuntimeCacheRoot, seedWorkbenchLibraryRuntime,
} from "../acceptance/library-runtime-cache.mjs";

async function temporary(t, prefix = "acceptance-library-runtime-") {
  const root = await mkdtemp(path.join(tmpdir(), prefix));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function exists(file) {
  try { await access(file); return true; }
  catch (error) { if (error.code === "ENOENT") return false; throw error; }
}

/**
 * Stands in for the real `library prepare` CLI spawn: shaped like
 * packages/node-runtime/src/library-runtime.ts's own `prepare()` (a `.prepared` fingerprint check
 * plus an `installed()`-style sentinel), so "reuse" and "rebuild on mismatch" are provable without
 * ever installing anything real or touching the network. `fingerprintRef` is a mutable box so a test
 * can change "what a correct install looks like" between calls, simulating a pin bump.
 */
function fakeRunPrepare(fingerprintRef, installs = []) {
  return async ({ cacheDir }) => {
    const pdfDir = path.join(cacheDir, "runtimes", "pdfjs-5.4.624");
    const marker = path.join(pdfDir, ".prepared");
    const sentinel = path.join(pdfDir, "node_modules", ".installed");
    const current = await readFile(marker, "utf8").catch(() => "");
    if (current === fingerprintRef.current && await exists(sentinel)) return;
    installs.push(cacheDir);
    await rm(pdfDir, { recursive: true, force: true });
    await mkdir(path.join(pdfDir, "node_modules"), { recursive: true });
    await writeFile(sentinel, "ok");
    await writeFile(path.join(pdfDir, "marker-source.txt"), `built for ${fingerprintRef.current}`);
    await writeFile(marker, fingerprintRef.current);
  };
}

// ---------------------------------------------------------------------------
// Red: the baseline fixture and environment, before any opt-in wiring.
// ---------------------------------------------------------------------------

test("baseline: a fresh exploration fixture has no prepared PDF runtime in its cache_dir", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  assert.equal(await exists(path.join(fixture.cacheDir, "runtimes", "pdfjs-5.4.624", ".prepared")), false,
    "nothing pre-seeds the fixture's own cache_dir; that is the opt-in exploration wiring's job, not the fixture's");
});

test("baseline: the fixture environment does not force npm offline", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  assert.notEqual(fixture.env.npm_config_offline, "true",
    "a spawned npm ci could still reach the real registry through this environment alone");
});

// ---------------------------------------------------------------------------
// Green: scripts/acceptance/library-runtime-cache.mjs — persistent cache keying.
// ---------------------------------------------------------------------------

test("the persistent cache root is version-keyed, outside the repo, and ignores a fixture's own XDG_CACHE_HOME", async t => {
  const outside = await temporary(t, "acceptance-outside-cache-");
  const root = await persistentLibraryRuntimeCacheRoot({ env: { XDG_CACHE_HOME: outside } });
  assert.ok(root.startsWith(outside + path.sep), "the cache root lives under the given XDG_CACHE_HOME");
  assert.match(root, /arxiv-daily-acceptance[/\\]runtimes[/\\]pdfjs-5\.4\.624-[0-9a-f]{16}$/);
  // The fixture's own cache_dir is a throwaway temp directory inside its own root; the persistent
  // cache must never land there, or it would be deleted (and re-downloaded) on every single fixture.
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  assert.ok(!root.startsWith(fixture.cacheDir), "the persistent cache is not nested inside a fixture's disposable cache_dir");
});

// ---------------------------------------------------------------------------
// Green: ensurePersistentPdfRuntime — creates once, reuses, rebuilds on mismatch.
// ---------------------------------------------------------------------------

test("ensurePersistentPdfRuntime creates the runtime once and reuses it without reinstalling", async t => {
  const cacheHome = await temporary(t, "acceptance-persistent-cache-");
  const fingerprintRef = { current: "fingerprint-v1" };
  const installs = [];
  const runPrepare = fakeRunPrepare(fingerprintRef, installs);
  const first = await ensurePersistentPdfRuntime({ env: { XDG_CACHE_HOME: cacheHome }, runPrepare });
  assert.equal(installs.length, 1, "first call installs");
  assert.equal(await readFile(path.join(first, ".prepared"), "utf8"), "fingerprint-v1");

  const second = await ensurePersistentPdfRuntime({ env: { XDG_CACHE_HOME: cacheHome }, runPrepare });
  assert.equal(second, first, "the same directory is returned on reuse");
  assert.equal(installs.length, 1, "a second call with a matching fingerprint does not reinstall");
});

test("ensurePersistentPdfRuntime rebuilds when the fingerprint no longer matches (a simulated pin bump)", async t => {
  const cacheHome = await temporary(t, "acceptance-persistent-cache-");
  const fingerprintRef = { current: "fingerprint-v1" };
  const installs = [];
  const runPrepare = fakeRunPrepare(fingerprintRef, installs);
  await ensurePersistentPdfRuntime({ env: { XDG_CACHE_HOME: cacheHome }, runPrepare });
  assert.equal(installs.length, 1);

  fingerprintRef.current = "fingerprint-v2";
  const rebuilt = await ensurePersistentPdfRuntime({ env: { XDG_CACHE_HOME: cacheHome }, runPrepare });
  assert.equal(installs.length, 2, "a changed fingerprint triggers exactly one rebuild");
  assert.equal(await readFile(path.join(rebuilt, ".prepared"), "utf8"), "fingerprint-v2");
});

test("ensurePersistentPdfRuntime rebuilds a corrupted/partial cache even when the fingerprint still matches", async t => {
  const cacheHome = await temporary(t, "acceptance-persistent-cache-");
  const fingerprintRef = { current: "fingerprint-v1" };
  const installs = [];
  const runPrepare = fakeRunPrepare(fingerprintRef, installs);
  const pdfDir = await ensurePersistentPdfRuntime({ env: { XDG_CACHE_HOME: cacheHome }, runPrepare });
  assert.equal(installs.length, 1);

  // Simulate a partial/corrupted install: the `.prepared` marker is still there and still correct,
  // but the installed sentinel (stand-in for the runtime's real installed() dependency check) is gone.
  await rm(path.join(pdfDir, "node_modules", ".installed"));
  const rebuilt = await ensurePersistentPdfRuntime({ env: { XDG_CACHE_HOME: cacheHome }, runPrepare });
  assert.equal(installs.length, 2, "a missing dependency behind a stale-but-matching marker still triggers a rebuild");
  assert.equal(await exists(path.join(rebuilt, "node_modules", ".installed")), true);
});

test("ensurePersistentPdfRuntime raises a clear error if the prepare step leaves no .prepared runtime", async t => {
  const cacheHome = await temporary(t, "acceptance-persistent-cache-");
  await assert.rejects(
    ensurePersistentPdfRuntime({ env: { XDG_CACHE_HOME: cacheHome }, runPrepare: async () => {} }),
    /did not leave a `\.prepared` runtime/,
  );
});

// ---------------------------------------------------------------------------
// Green: seedWorkbenchLibraryRuntime — real copy into a fixture's cache_dir, temp-dir-then-rename.
// ---------------------------------------------------------------------------

test("seedWorkbenchLibraryRuntime copies the prepared runtime (not a symlink) into the fixture's cache_dir", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const source = await temporary(t, "acceptance-source-runtime-");
  await mkdir(path.join(source, "node_modules", "pdfjs-dist"), { recursive: true });
  await writeFile(path.join(source, "node_modules", "pdfjs-dist", "package.json"), "{}");
  await writeFile(path.join(source, ".prepared"), "source-fingerprint");

  const dest = await seedWorkbenchLibraryRuntime(fixture, { ensure: async () => source });
  assert.equal(dest, path.join(fixture.cacheDir, "runtimes", "pdfjs-5.4.624"));
  assert.equal(await readFile(path.join(dest, ".prepared"), "utf8"), "source-fingerprint");
  assert.equal((await lstat(dest)).isSymbolicLink(), false, "the fixture's runtime directory is a real copy, not a symlink to the source");

  // A real copy, not a shared reference: mutating the source afterwards must not affect the fixture's copy.
  await writeFile(path.join(source, ".prepared"), "source-fingerprint-changed");
  assert.equal(await readFile(path.join(dest, ".prepared"), "utf8"), "source-fingerprint",
    "the fixture's copy is independent of the source directory");

  // No leftover temp directory from the rename.
  const entries = await readdir(path.join(fixture.cacheDir, "runtimes"));
  assert.deepEqual(entries, ["pdfjs-5.4.624"]);
});

test("seedWorkbenchLibraryRuntime is idempotent: seeding twice leaves one clean copy and no stray temp directories", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const source = await temporary(t, "acceptance-source-runtime-");
  await writeFile(path.join(source, ".prepared"), "fingerprint-a");
  await seedWorkbenchLibraryRuntime(fixture, { ensure: async () => source });

  await writeFile(path.join(source, ".prepared"), "fingerprint-b");
  await seedWorkbenchLibraryRuntime(fixture, { ensure: async () => source });

  const entries = await readdir(path.join(fixture.cacheDir, "runtimes"));
  assert.deepEqual(entries, ["pdfjs-5.4.624"]);
  assert.equal(await readFile(path.join(fixture.cacheDir, "runtimes", "pdfjs-5.4.624", ".prepared"), "utf8"), "fingerprint-b");
});

test("seedWorkbenchLibraryRuntime requires a fixture with its own cache_dir", async () => {
  await assert.rejects(seedWorkbenchLibraryRuntime({}), /cache_dir/);
});

// ---------------------------------------------------------------------------
// Green: npmOfflineEnv — the environment overlay forced on the workbench server.
// ---------------------------------------------------------------------------

test("npmOfflineEnv forces npm offline without mutating the given environment", () => {
  const base = Object.freeze({ PATH: "/usr/bin", FOO: "bar" });
  const overlay = npmOfflineEnv(base);
  assert.equal(overlay.npm_config_offline, "true");
  assert.equal(overlay.npm_config_fetch_retries, "0");
  assert.equal(overlay.PATH, "/usr/bin");
  assert.equal(overlay.FOO, "bar");
  assert.equal(base.npm_config_offline, undefined, "the input environment object is not mutated");
});

// ---------------------------------------------------------------------------
// Green: exploration.mjs wires this in for the workbench host only, before the server starts.
// ---------------------------------------------------------------------------

test("exploration host=workbench seeds the fixture's library runtime and forces npm offline before starting the session", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const seen = [];
  const fakeDriver = {
    observe: async () => ({ id: "s1", controls: [], text: "", title: "", layout: { rootMounted: true } }),
    execute: async () => ({ ok: true }), capture: async () => undefined,
  };
  const goalTask = { id: "goal-a", title: "t", goal: "g", async verify() { return { passed: true, checks: [{ label: "x", passed: true }] }; } };
  let sessionFixtureEnv, driverFixtureEnv;
  await runExploration({
    fixture, host: "workbench", artifactDir: path.join(await temporary(t), "exploration"),
    goalTasks: [goalTask], personaTasks: [], model: async () => '{"type":"done"}',
  }, {
    startWorkbenchSession: async ({ fixture: sessionFixture }) => { seen.push("session"); sessionFixtureEnv = sessionFixture.env; return {}; },
    createExplorationDriver: async ({ fixture: driverFixture }) => { seen.push("driver"); driverFixtureEnv = driverFixture.env; return fakeDriver; },
    libraryRuntimeCache: {
      seedWorkbenchLibraryRuntime: async seededFixture => { seen.push("seed:" + seededFixture.cacheDir); },
      npmOfflineEnv: env => ({ ...env, npm_config_offline: "true" }),
    },
  });
  assert.deepEqual(seen, ["seed:" + fixture.cacheDir, "session", "driver"], "the runtime is seeded before the session (and its driver) starts");
  assert.equal(sessionFixtureEnv.npm_config_offline, "true", "the session's own fixture env is npm-offline");
  assert.equal(driverFixtureEnv.npm_config_offline, "true", "the driver sees the same npm-offline fixture env");
});

test("exploration host=obsidian never touches the library runtime cache", async t => {
  const seen = [];
  const fakeDriver = {
    observe: async () => ({ id: "s1", controls: [], text: "", title: "", layout: { rootMounted: true } }),
    execute: async () => ({ ok: true }), capture: async () => undefined,
    reset: async () => {}, dispose: async () => {},
  };
  const goalTask = { id: "goal-a", title: "t", goal: "g", async verify() { return { passed: true, checks: [{ label: "x", passed: true }] }; } };
  await runExploration({
    fixture: { vaultRoot: "/tmp/x", server: { url: "http://127.0.0.1:1" } }, host: "obsidian",
    artifactDir: path.join(await temporary(t), "exploration"), goalTasks: [goalTask], personaTasks: [], model: async () => '{"type":"done"}',
  }, {
    startObsidianExplorationSession: async () => ({ session: { evaluate: async () => {}, client: {}, diagnostics: { entries: () => [] } }, stop: async () => {} }),
    createObsidianExplorationDriver: async () => fakeDriver,
    libraryRuntimeCache: { seedWorkbenchLibraryRuntime: async () => { seen.push("seed"); }, npmOfflineEnv: env => env },
  });
  assert.deepEqual(seen, [], "the obsidian host never calls the workbench-only runtime cache helper");
});
