// P4 Chunk 2 of docs/helm/2026-10-13-exploration-safety-and-coverage: the Reading workbench's
// personal-library tasks need a deterministic seed (connected, optionally authorized, optionally
// indexed) through the real CLI, and a reset() that actually clears library state back to the fixture
// baseline (TOML [library]/[embedding], library-origin directions, `.index/personal-library-*`, extra
// library PDFs) so one library task's leftovers never leak into the next task or persona — the same
// leak class P3 Chunk 8 fixed for Obsidian (d0e3fdc).
import assert from "node:assert/strict";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { createFixtureEnvironment } from "../acceptance/fixtures.mjs";
import { createExplorationDriver, settings, writeConfigPatch } from "../acceptance/exploration-browser.mjs";
import {
  runWorkbenchLibraryCli, startWorkbenchLibraryFixture, WORKBENCH_LIBRARY_EMBEDDING_DIMENSION,
} from "../acceptance/exploration-tasks/workbench-library.mjs";
import { DEFAULT_CLI_PATH } from "../acceptance/library-runtime-cache.mjs";

// ---------------------------------------------------------------------------
// Test seam: a fake Playwright page/context, exactly like acceptance-exploration.test.mjs's own
// `fakeResetPage` — no real browser is needed to exercise reset()'s own fixture-file bookkeeping.
// ---------------------------------------------------------------------------
function fakeResetPage(url, { revision = "sha256:current" } = {}) {
  const calls = [];
  let currentUrl = url;
  return {
    calls,
    url: () => currentUrl,
    on() {}, off() {},
    goto: async target => { currentUrl = String(target); },
    locator(selector) {
      if (selector === "dialog[open]") return { count: async () => 0, waitFor: async () => {}, locator: () => ({ first: () => ({ click: async () => {} }) }) };
      return { waitFor: async () => {} };
    },
    request: {
      get: async target => { calls.push(["GET", String(target)]); return { ok: () => true, json: async () => ({ revision, setupRequired: revision === null, values: {} }) }; },
      post: async (target, options) => { calls.push(["POST", String(target), options?.data]); return { ok: () => true, json: async () => ({}) }; },
    },
  };
}

async function makeDriver(fixture, { revision = "sha256:current" } = {}) {
  const url = "http://127.0.0.1:3900/session/";
  const page = fakeResetPage(url, { revision });
  const context = { on() {}, off() {}, route: async () => {}, unroute: async () => {} };
  return createExplorationDriver({ page, context, url, fixture, artifactDir: path.join(fixture.root, "artifacts") });
}

// ---------------------------------------------------------------------------
// Red/Green: reset() must restore TOML [library]/[embedding] and strip library-origin directions.
// ---------------------------------------------------------------------------

test("reset() removes the fixture TOML's [library] and [embedding] tables added by a library seed", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await writeConfigPatch(fixture, table => {
    table.library = { schemaVersion: 1, selectedRoot: fixture.libraryRoot, rootIdentity: "dev:1", eligibleExtensions: [".pdf"], processingDepth: "metadata-and-abstracts", authorization: { fingerprint: "fp", grantedAt: new Date().toISOString() } };
    table.embedding = { mode: "remote", base_url: "https://example.invalid/v1", api_key: "k", model: "m", dimension: WORKBENCH_LIBRARY_EMBEDDING_DIMENSION };
  });
  const before = await settings(fixture);
  assert.ok(before.library, "precondition: [library] exists before reset()");
  assert.ok(before.embedding, "precondition: [embedding] exists before reset()");

  const driver = await makeDriver(fixture);
  t.after(() => driver.dispose());
  await driver.reset();

  const after = await settings(fixture);
  assert.equal(after.library, undefined, "reset() must remove the [library] table back to the fixture baseline (no library section)");
  assert.equal(after.embedding, undefined, "reset() must remove the [embedding] table back to the fixture baseline (no embedding section)");
});

test("reset() removes library-origin directions but keeps the fixture's own manual direction", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await writeConfigPatch(fixture, table => {
    table.arxiv.topics[0].directions.push({ id: "library-direction-1", text: "Accepted from the library", origin: "library" });
  });
  const before = await settings(fixture);
  assert.equal(before.arxiv.topics[0].directions.length, 2, "precondition: a library-origin direction was accepted");

  const driver = await makeDriver(fixture);
  t.after(() => driver.dispose());
  await driver.reset();

  const after = await settings(fixture);
  assert.deepEqual(
    after.arxiv.topics[0].directions.map(direction => direction.origin),
    ["manual"],
    "reset() must drop every origin:\"library\" direction and keep the fixture's own manual one",
  );
});

test("reset() leaves settings-persistence's own saved value alone (only library state is restored to baseline)", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await writeConfigPatch(fixture, table => { table.output.max_daily_papers = 13; table.library = { schemaVersion: 1, selectedRoot: fixture.libraryRoot, rootIdentity: "dev:1", eligibleExtensions: [".pdf"], processingDepth: "metadata-and-abstracts" }; });
  const driver = await makeDriver(fixture);
  t.after(() => driver.dispose());
  await driver.reset();
  const after = await settings(fixture);
  assert.equal(after.library, undefined, "library state is still cleared");
  assert.equal(after.output.max_daily_papers, 13, "a non-library saved setting must survive reset(), matching the existing settings-persistence contract");
});

// ---------------------------------------------------------------------------
// Red/Green: reset() must remove personal-library-* index state but leave unrelated index entries.
// ---------------------------------------------------------------------------

test("reset() removes personal-library-* entries under the vault's .index directory but keeps unrelated entries", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const indexDir = path.join(fixture.vaultRoot, "arxiv-daily", ".index");
  await mkdir(path.join(indexDir, "personal-library-search-index", "ab", "cd"), { recursive: true });
  await writeFile(path.join(indexDir, "personal-library-search-index", "ab", "cd", "current.json"), "{}");
  await writeFile(path.join(indexDir, "personal-library-catalog.json"), "{}");
  await writeFile(path.join(indexDir, "papers.json"), '{"papers":{}}');

  const driver = await makeDriver(fixture);
  t.after(() => driver.dispose());
  await driver.reset();

  const remaining = await readdir(indexDir);
  assert.ok(!remaining.some(entry => entry.startsWith("personal-library-")), `every personal-library-* entry must be removed, found: ${remaining}`);
  assert.ok(remaining.includes("papers.json"), "an unrelated index entry must survive reset()");
});

// ---------------------------------------------------------------------------
// Red/Green: reset() must remove extra library PDFs (added after the driver started) but keep the
// fixture's own baseline PDFs.
// ---------------------------------------------------------------------------

test("reset() removes library PDFs added after the driver started but keeps the fixture's own baseline PDFs", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const baselineBefore = await readdir(fixture.libraryRoot);
  assert.ok(baselineBefore.length >= 2, "precondition: the fixture ships its own baseline library PDFs");

  const driver = await makeDriver(fixture);
  t.after(() => driver.dispose());
  await writeFile(path.join(fixture.libraryRoot, "9999.99999.pdf"), "not a real pdf, just extra");

  await driver.reset();

  const remaining = await readdir(fixture.libraryRoot);
  assert.deepEqual(remaining.sort(), baselineBefore.sort(), "only the extra PDF added after driver creation must be removed");
});

// ---------------------------------------------------------------------------
// Red/Green: startWorkbenchLibraryFixture (CLI step injectable for a fast, no-network unit test).
// ---------------------------------------------------------------------------

function fakeCliRecorder(responses = {}) {
  const calls = [];
  const runCli = async ({ args, env }) => {
    calls.push({ args, env });
    const key = args.slice(0, 2).join(" ");
    if (responses[key]) return responses[key];
    return {};
  };
  return { calls, runCli };
}

test("startWorkbenchLibraryFixture connects, authorizes and builds the index through the real CLI in order, and resyncs the server", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const { calls, runCli } = fakeCliRecorder({ "library status": { disclosure: { authorizationFingerprint: "fp-123" } } });
  let resynced = false;
  const fakeDriver = { fixture, resyncSettings: async () => { resynced = true; } };

  const result = await startWorkbenchLibraryFixture(fakeDriver, { authorized: true, indexed: true, clusterable: true, runCli });

  assert.deepEqual(calls.map(call => call.args.slice(0, 2).join(" ")), [
    "library connect", "library status", "library authorize", "library prepare", "library scan", "library index",
  ], "the seed must connect, read the disclosure, authorize against its fingerprint, then prepare/scan/index in order");
  assert.equal(calls[2].args[3], "fp-123", "authorize must use the real fingerprint returned by `library status`, never a hand-written one");
  assert.equal(result.fingerprint, "fp-123");
  assert.equal(resynced, true, "the running server's in-memory config must be resynced after the out-of-band seed");

  const config = await settings(fixture);
  assert.equal(config.embedding.mode, "remote");
  assert.equal(config.embedding.dimension, WORKBENCH_LIBRARY_EMBEDDING_DIMENSION);
  assert.equal(config.embedding.base_url, `${fixture.server.url}/v1`);
  assert.ok(config.embedding.api_key?.length > 0, "a real key-shaped fixture value, never empty (validateEmbeddingConfig rejects an empty one)");

  const libraryFiles = await readdir(fixture.libraryRoot);
  assert.ok(libraryFiles.includes("9912.00099.pdf"), "clusterable:true seeds one more fixture PDF for a real two-member cluster");

  const offlineCalls = calls.filter(call => ["library prepare", "library scan", "library index"].includes(call.args.slice(0, 2).join(" ")));
  assert.ok(offlineCalls.every(call => call.env.npm_config_offline === "true"), "prepare/scan/index must run with npm forced offline");
});

test("startWorkbenchLibraryFixture without authorized only connects (no authorize/prepare/scan/index)", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const { calls, runCli } = fakeCliRecorder({ "library status": { disclosure: { authorizationFingerprint: "fp-999" } } });
  const fakeDriver = { fixture, resyncSettings: async () => {} };

  await startWorkbenchLibraryFixture(fakeDriver, { runCli });

  assert.deepEqual(calls.map(call => call.args.slice(0, 2).join(" ")), ["library connect", "library status"]);
});

test("startWorkbenchLibraryFixture rejects indexed without authorized", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const { runCli } = fakeCliRecorder();
  const fakeDriver = { fixture, resyncSettings: async () => {} };
  await assert.rejects(
    startWorkbenchLibraryFixture(fakeDriver, { indexed: true, runCli }),
    /indexed requires authorized/,
  );
});

test("startWorkbenchLibraryFixture requires a driver exposing its own fixture", async () => {
  await assert.rejects(startWorkbenchLibraryFixture({}), /driver\.fixture/);
});

// ---------------------------------------------------------------------------
// Real-CLI integration (cheap: no browser, no runtime prepare/scan/index — just connect+authorize
// against the real built CLI binary), mirroring library-runtime-cache.mjs's own real-CLI tests.
// ---------------------------------------------------------------------------

test("startWorkbenchLibraryFixture against the real built CLI connects and authorizes a fixture library", async t => {
  try { await readFile(DEFAULT_CLI_PATH); }
  catch { t.skip("CLI build is missing; run: npm run build --workspace apps/cli"); return; }

  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  let resynced = false;
  const fakeDriver = { fixture, resyncSettings: async () => { resynced = true; } };

  const result = await startWorkbenchLibraryFixture(fakeDriver, { authorized: true, runCli: runWorkbenchLibraryCli });

  assert.equal(resynced, true);
  const config = await settings(fixture);
  assert.equal(config.library?.authorization?.fingerprint, result.fingerprint, "the real CLI authorized against its own real fingerprint");
  assert.equal(config.embedding.mode, "remote");
});
