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
import { createFixtureEnvironment, PAPERS } from "../acceptance/fixtures.mjs";
import { createExplorationDriver, settings, writeConfigPatch } from "../acceptance/exploration-browser.mjs";
import {
  runWorkbenchLibraryCli, startWorkbenchLibraryFixture, WORKBENCH_LIBRARY_EMBEDDING_DIMENSION,
} from "../acceptance/exploration-tasks/workbench-library.mjs";
import { goalTasks as workbenchGoalTasks } from "../acceptance/exploration-tasks/workbench.mjs";
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

test("reset() removes a topic auto-created by library-accept-direction once its only (library-origin) direction is gone, but keeps a topic that still has a direction of its own", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await writeConfigPatch(fixture, table => {
    // Mirrors what acceptProposedTopics (packages/core/src/settings/accept-proposed-topics.ts) actually
    // writes for a brand-new proposed topic: it holds only the one library-origin direction it was
    // created for, unlike the fixture's own baseline "Research" topic (one manual direction, left alone).
    table.arxiv.topics.push({ id: "auto-created-topic", name: "Research evidence 1", tag: "research-evidence-1", detail: false,
      directions: [{ id: "library-direction-1", text: "Accepted from the library", origin: "library" }] });
    // A topic that never had directions is not reset()'s to remove.
    table.arxiv.topics.push({ id: "empty-manual-topic", name: "Empty manual", tag: "empty-manual", detail: false, directions: [] });
  });
  const before = await settings(fixture);
  assert.equal(before.arxiv.topics.length, 3, "precondition: library-accept-direction auto-created a second topic");

  const driver = await makeDriver(fixture);
  t.after(() => driver.dispose());
  await driver.reset();

  const after = await settings(fixture);
  assert.deepEqual(after.arxiv.topics.map(topic => topic.id), ["acceptance-topic", "empty-manual-topic"],
    "reset() must remove the auto-created topic entirely once its only direction is gone (never leave an empty shell whose name could collide with the next real `library propose` call)");
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

// ---------------------------------------------------------------------------
// P4 Chunk 3: `scanned` runs `library prepare`/`scan` (no authorize, no index) — the catalog-filter task's
// own precondition. Scanning never requires authorization (apps/cli/src/library-cmd.ts's `scan` branch
// never calls `assertAuthorized`), unlike `indexed`.
// ---------------------------------------------------------------------------

test("startWorkbenchLibraryFixture with scanned:true (and no authorized) connects and scans, without authorizing or indexing", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const { calls, runCli } = fakeCliRecorder({ "library status": { disclosure: { authorizationFingerprint: "fp-321" } } });
  const fakeDriver = { fixture, resyncSettings: async () => {} };

  await startWorkbenchLibraryFixture(fakeDriver, { scanned: true, runCli });

  assert.deepEqual(calls.map(call => call.args.slice(0, 2).join(" ")), ["library connect", "library status", "library prepare", "library scan"],
    "scanned:true must prepare/scan but never authorize or index when authorized is left false");
  const offlineCalls = calls.filter(call => ["library prepare", "library scan"].includes(call.args.slice(0, 2).join(" ")));
  assert.ok(offlineCalls.every(call => call.env.npm_config_offline === "true"), "prepare/scan must run with npm forced offline");
});

test("startWorkbenchLibraryFixture with indexed:true still scans before indexing (scanned is implied, not a separate call)", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const { calls, runCli } = fakeCliRecorder({ "library status": { disclosure: { authorizationFingerprint: "fp-654" } } });
  const fakeDriver = { fixture, resyncSettings: async () => {} };

  await startWorkbenchLibraryFixture(fakeDriver, { authorized: true, indexed: true, runCli });

  assert.deepEqual(calls.map(call => call.args.slice(0, 2).join(" ")), [
    "library connect", "library status", "library authorize", "library prepare", "library scan", "library index",
  ], "indexed:true's own order must be unchanged by the scanned option existing");
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

// ---------------------------------------------------------------------------
// P4 Chunk 3 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/04-workbench-library-tasks.md):
// the four personal-library goal tasks' own verify() — untouched state, partial actions, and state
// present WITHOUT the model's own observed action must all fail; only the full, in-order action
// sequence together with the matching on-disk/fixture state passes. Each test reads the task straight
// out of goalTasks(fixture) (the module the exploration runner itself imports), never a parallel copy
// of the task's own matching logic, and builds its own real-ish fixture (createFixtureEnvironment, this
// file's own convention) rather than a hand-rolled minimal one, so the index-file path derivation
// (path.dirname(daily_dir)) matches the real fixture config exactly.
// ---------------------------------------------------------------------------

async function findWorkbenchTask(fixture, id) {
  const tasks = await workbenchGoalTasks(fixture);
  const task = tasks.find(candidate => candidate.id === id);
  assert.ok(task, `task ${id} must exist in goalTasks()`);
  return task;
}

function fakeServerDriver(requests = []) {
  return { fixture: { server: { requests } } };
}

async function libraryAuthorizationPatch(fixture) {
  await writeConfigPatch(fixture, table => {
    table.library = {
      schemaVersion: 1, selectedRoot: fixture.libraryRoot, rootIdentity: "dev:1",
      eligibleExtensions: [".pdf"], processingDepth: "metadata-and-abstracts",
      authorization: { fingerprint: "fp", grantedAt: new Date().toISOString() },
    };
  });
}

async function writeLibrarySearchIndexFile(fixture) {
  const dir = path.join(fixture.vaultRoot, "arxiv-daily", ".index", "personal-library-search-index", "ab", "cd");
  await mkdir(dir, { recursive: true });
  await writeFile(path.join(dir, "current.json"), "{}");
}

test("library-build-index: verify fails on the untouched state (no click at all)", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-build-index");
  const result = await task.verify({ driver: fakeServerDriver(), actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false, "must not pass without an observed click on Retry preparation");
});

test("library-build-index: verify fails after clicking Retry preparation alone, before the disclosure is confirmed", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-build-index");
  const actions = [{ type: "click", control: { dataAction: "library-build" } }];
  const result = await task.verify({ driver: fakeServerDriver(), actions });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, true);
  assert.equal(result.checks[1].passed, false, "must not pass without an observed click on the confirm-and-build button");
});

test("library-build-index: verify fails on leftover state alone (authorization, index file and an embeddings request already present) when no action is observed", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-build-index");
  await libraryAuthorizationPatch(fixture);
  await writeLibrarySearchIndexFile(fixture);
  const result = await task.verify({ driver: fakeServerDriver([{ kind: "embeddings" }]), actions: [] });
  assert.equal(result.passed, false, "leftover state with no observed click must not pass");
  assert.equal(result.checks[0].passed, false);
});

test("library-build-index: verify passes once both clicks are observed and disk/fixture state agree; fails if authorization, the index file or the embeddings request is missing", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-build-index");
  const actions = [
    { type: "click", control: { dataAction: "library-build" } },
    { type: "click", control: { dataAction: "confirm-library-build" } },
  ];
  const driverWithEmbeddings = fakeServerDriver([{ kind: "embeddings" }]);

  // Mutation 1: neither authorization nor the index file exist yet.
  const beforeAnyState = await task.verify({ driver: driverWithEmbeddings, actions });
  assert.equal(beforeAnyState.passed, false);

  await libraryAuthorizationPatch(fixture);
  // Mutation 2: authorized, but the index file does not exist yet.
  const withoutIndex = await task.verify({ driver: driverWithEmbeddings, actions });
  assert.equal(withoutIndex.passed, false);

  await writeLibrarySearchIndexFile(fixture);
  // Mutation 3: authorized and indexed, but the fixture never actually saw an embeddings request.
  const withoutEmbeddings = await task.verify({ driver: fakeServerDriver([]), actions });
  assert.equal(withoutEmbeddings.passed, false);

  const result = await task.verify({ driver: driverWithEmbeddings, actions });
  assert.equal(result.passed, true, JSON.stringify(result));
});

test("library-revoke: verify fails on the untouched state (no click at all)", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await libraryAuthorizationPatch(fixture);
  const task = await findWorkbenchTask(fixture, "library-revoke");
  const result = await task.verify({ actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false, "must not pass without an observed click on Revoke");
});

test("library-revoke: verify fails on leftover state alone (authorization already absent) when no action is observed", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  // No [library] table at all: the same "revoked" shape the real action leaves behind.
  const task = await findWorkbenchTask(fixture, "library-revoke");
  const result = await task.verify({ actions: [] });
  assert.equal(result.passed, false, "state alone (authorization absent) with no observed click must not pass");
  assert.equal(result.checks[0].passed, false);
});

test("library-revoke: verify passes once the Revoke click is observed and authorization is actually gone from disk; fails while it is still present", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await libraryAuthorizationPatch(fixture);
  const task = await findWorkbenchTask(fixture, "library-revoke");
  const actions = [{ type: "click", control: { dataAction: "library-revoke" } }];

  // Mutation: the click is observed, but authorization is still on disk (a plausible wrong outcome:
  // the click did not actually revoke it).
  const stillAuthorized = await task.verify({ actions });
  assert.equal(stillAuthorized.passed, false);

  await writeConfigPatch(fixture, table => { delete table.library; });
  const result = await task.verify({ actions });
  assert.equal(result.passed, true, JSON.stringify(result));
});

test("library-catalog-filter: verify fails on the untouched state (no fill, no browse click)", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-catalog-filter");
  const result = await task.verify({ observation: { text: "" }, actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false, "must not pass without an observed fill of the keyword field");
});

test("library-catalog-filter: verify fails after filling the keyword alone, before the browse button is clicked", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-catalog-filter");
  const actions = [{ type: "fill", control: { name: "文献库关键词" }, value: "measurement" }];
  const result = await task.verify({ observation: { text: "" }, actions });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, true);
  assert.equal(result.checks[1].passed, false, "must not pass without an observed click on 筛选目录");
});

test("library-catalog-filter: verify fails on leftover result text alone (目录 · 1 篇论文 already shown) when no action is observed", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-catalog-filter");
  const observation = { text: `目录 · 1 篇论文\n${PAPERS[0].title}` };
  const result = await task.verify({ observation, actions: [] });
  assert.equal(result.passed, false, "leftover result text with no observed action must not pass");
  assert.equal(result.checks[0].passed, false);
});

test("library-catalog-filter: verify passes once fill and browse are observed in order and the result text shows exactly the matching paper; fails on a wrong result count or paper", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-catalog-filter");
  const actions = [
    { type: "fill", control: { name: "文献库关键词" }, value: "measurement" },
    { type: "click", control: { name: "筛选目录" } },
  ];

  // Mutation 1: the result text still shows the untouched two-paper catalog.
  const beforeFilter = await task.verify({ observation: { text: "目录 · 2 篇论文" }, actions });
  assert.equal(beforeFilter.passed, false);

  // Mutation 2: exactly one result, but it is the wrong paper (not PAPERS[0]).
  const wrongPaper = await task.verify({ observation: { text: `目录 · 1 篇论文\n${PAPERS[1].title}` }, actions });
  assert.equal(wrongPaper.passed, false);

  const result = await task.verify({ observation: { text: `目录 · 1 篇论文\n${PAPERS[0].title}` }, actions });
  assert.equal(result.passed, true, JSON.stringify(result));

  // Order matters: a browse click observed before the fill must not count.
  const reordered = [actions[1], actions[0]];
  const reorderedResult = await task.verify({ observation: { text: `目录 · 1 篇论文\n${PAPERS[0].title}` }, actions: reordered });
  assert.equal(reorderedResult.passed, false, "a browse click before the keyword fill must not count as filtering");
});

test("library-index-search: verify fails on the untouched state (no select, fill or search click)", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-index-search");
  const result = await task.verify({ driver: fakeServerDriver(), observation: { text: "" }, actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false, "must not pass without an observed select of the dense search mode");
});

test("library-index-search: verify fails after selecting the mode and filling the keyword alone, before the search button is clicked", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-index-search");
  const actions = [
    { type: "select", control: { name: "索引检索方式" }, value: "dense" },
    { type: "fill", control: { name: "文献库关键词" }, value: "measurement" },
  ];
  const result = await task.verify({ driver: fakeServerDriver(), observation: { text: "" }, actions });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, true);
  assert.equal(result.checks[1].passed, true);
  assert.equal(result.checks[2].passed, false, "must not pass without an observed click on 检索索引");
});

test("library-index-search: verify fails on leftover result text and a leftover embeddings request alone, when no action is observed", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-index-search");
  const observation = { text: `检索结果 · 2 篇论文\n${PAPERS[0].title}` };
  const result = await task.verify({ driver: fakeServerDriver([{ kind: "embeddings" }]), observation, actions: [] });
  assert.equal(result.passed, false, "leftover result text and request with no observed action must not pass");
  assert.equal(result.checks[0].passed, false);
});

test("library-index-search: verify passes once the mode select, keyword fill and search click are all observed in order and the fixture saw a fresh embeddings request; fails if the request is missing or the result list is empty", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-index-search");
  const actions = [
    { type: "select", control: { name: "索引检索方式" }, value: "dense" },
    { type: "fill", control: { name: "文献库关键词" }, value: "measurement" },
    { type: "click", control: { name: "检索索引" } },
  ];
  const observation = { text: `检索结果 · 2 篇论文\n${PAPERS[0].title}` };

  // Mutation 1: the search ran, but the fixture never actually saw an embeddings request after this
  // task started (a plausible wrong outcome: the model picked a mode that never calls the embedding
  // endpoint, or the click never actually triggered a search).
  const withoutEmbeddings = await task.verify({ driver: fakeServerDriver([]), observation, actions });
  assert.equal(withoutEmbeddings.passed, false);

  // Mutation 2: an embeddings request happened, but no result was actually shown.
  const withoutResults = await task.verify({ driver: fakeServerDriver([{ kind: "embeddings" }]), observation: { text: "检索结果 · 0 篇论文" }, actions });
  assert.equal(withoutResults.passed, false);

  const result = await task.verify({ driver: fakeServerDriver([{ kind: "embeddings" }]), observation, actions });
  assert.equal(result.passed, true, JSON.stringify(result));

  // Order matters: selecting the mode after the search click must not count.
  const reordered = [actions[2], actions[0], actions[1]];
  const reorderedResult = await task.verify({ driver: fakeServerDriver([{ kind: "embeddings" }]), observation, actions: reordered });
  assert.equal(reorderedResult.passed, false, "a search click before the mode select and keyword fill must not count");
});

// ---------------------------------------------------------------------------
// P4 Chunk 4 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/04-workbench-library-tasks.md):
// Direction candidate review tasks — same untouched/partial/leftover/full-pass convention as the Chunk 3
// tests above, hand-writing TOML/`direction-proposal.json` state directly rather than running the real
// CLI's `library propose` (prepare()'s own job, exercised separately by the real-host run).
// ---------------------------------------------------------------------------

const NEW_TOPIC_NAME = "工作台候选主题（已重命名）";

/** Writes a minimal two-topic, one-candidate-each `direction-proposal.json` at the real derived path
 * (any hash subdirectories work: readLibraryDirectionProposal() finds the file by its own name, never
 * by a predicted hash), matching the shape library-edit-candidate's own verify() reads. */
async function writeLibraryDirectionProposalFile(fixture, topics) {
  const dir = path.join(fixture.vaultRoot, "arxiv-daily", ".index", "personal-library-profiles", "ab", "cd");
  await mkdir(dir, { recursive: true });
  await writeFile(path.join(dir, "direction-proposal.json"), JSON.stringify({ topics }));
}
const twoFixtureTopics = () => [
  { id: "topic-0", suggestedName: "Research evidence 1", directions: [{ id: "candidate-0" }] },
  { id: "topic-1", suggestedName: "Research evidence 2", directions: [{ id: "candidate-1" }] },
];

test("library-accept-direction: verify fails on the untouched state (no click at all)", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-accept-direction");
  const result = await task.verify({ driver: fakeServerDriver(), actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false, "must not pass without an observed click on 方向审核");
});

test("library-accept-direction: verify fails after the nav/generate/confirm clicks alone, before accepting", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-accept-direction");
  const actions = [
    { type: "click", control: { name: "方向审核" } },
    { type: "click", control: { name: "生成候选方向" } },
    { type: "click", control: { name: "确认" } },
  ];
  const result = await task.verify({ driver: fakeServerDriver(), actions });
  assert.equal(result.passed, false);
  assert.equal(result.checks[2].passed, true);
  assert.equal(result.checks[3].passed, false, "must not pass without an observed click on 接受所选方向");
});

test("library-accept-direction: verify fails on leftover state alone (direction and proposal request already present) when no action is observed", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await writeConfigPatch(fixture, table => { table.arxiv.topics[0].directions.push({ id: "library-direction-1", text: "Accepted from the library", origin: "library" }); });
  const task = await findWorkbenchTask(fixture, "library-accept-direction");
  const result = await task.verify({ driver: fakeServerDriver([{ kind: "proposal" }]), actions: [] });
  assert.equal(result.passed, false, "leftover state with no observed click must not pass");
  assert.equal(result.checks[0].passed, false);
});

test("library-accept-direction: verify passes once the full nav/generate/confirm/accept/confirm sequence is observed and TOML/fixture state agree; fails if the direction or the proposal request is missing", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-accept-direction");
  const actions = [
    { type: "click", control: { name: "方向审核" } },
    { type: "click", control: { name: "生成候选方向" } },
    { type: "click", control: { name: "确认" } },
    { type: "click", control: { name: "接受所选方向" } },
    { type: "click", control: { name: "确认" } },
  ];
  const driverWithProposal = fakeServerDriver([{ kind: "proposal" }]);

  // Mutation 1: neither the direction nor the proposal request exist yet.
  const beforeAnyState = await task.verify({ driver: driverWithProposal, actions });
  assert.equal(beforeAnyState.passed, false);

  // Mutation 2: the fixture never actually saw a "proposal" kind request after this task started.
  const withoutProposalRequest = await task.verify({ driver: fakeServerDriver([]), actions });
  assert.equal(withoutProposalRequest.passed, false);

  await writeConfigPatch(fixture, table => { table.arxiv.topics[0].directions.push({ id: "library-direction-1", text: "Accepted from the library", origin: "library" }); });
  const result = await task.verify({ driver: driverWithProposal, actions });
  assert.equal(result.passed, true, JSON.stringify(result));
});

test("library-edit-candidate: verify fails on the untouched state (no click at all)", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await writeLibraryDirectionProposalFile(fixture, twoFixtureTopics());
  const task = await findWorkbenchTask(fixture, "library-edit-candidate");
  const result = await task.verify({ actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false, "must not pass without an observed click on 方向审核");
});

test("library-edit-candidate: verify fails after renaming alone, before discarding the other topic's candidate", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await writeLibraryDirectionProposalFile(fixture, twoFixtureTopics());
  const task = await findWorkbenchTask(fixture, "library-edit-candidate");
  const actions = [
    { type: "click", control: { name: "方向审核" } },
    { type: "fill", control: { name: "建议主题名称" }, value: NEW_TOPIC_NAME },
    { type: "click", control: { name: "保存主题名称" } },
  ];
  const result = await task.verify({ actions });
  assert.equal(result.passed, false);
  assert.equal(result.checks[2].passed, true, "the rename click must be observed");
  assert.equal(result.checks[3].passed, false, "must not pass without an observed click on 删除候选");
});

test("library-edit-candidate: verify fails after clicking 删除候选 alone, before confirming the discard", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await writeLibraryDirectionProposalFile(fixture, twoFixtureTopics());
  const task = await findWorkbenchTask(fixture, "library-edit-candidate");
  const actions = [
    { type: "click", control: { name: "方向审核" } },
    { type: "click", control: { name: "删除候选" } },
  ];
  const result = await task.verify({ actions });
  assert.equal(result.passed, false);
  assert.equal(result.checks[3].passed, true);
  assert.equal(result.checks[4].passed, false, "must not pass without an observed confirm click on the discard dialog");
});

test("library-edit-candidate: verify fails on leftover state alone (file already shows one renamed topic) when no action is observed", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  await writeLibraryDirectionProposalFile(fixture, [{ id: "topic-0", suggestedName: NEW_TOPIC_NAME, directions: [{ id: "candidate-0" }] }]);
  const task = await findWorkbenchTask(fixture, "library-edit-candidate");
  const result = await task.verify({ actions: [] });
  assert.equal(result.passed, false, "leftover state with no observed click must not pass");
  assert.equal(result.checks[0].passed, false);
});

test("library-edit-candidate: verify passes once the full rename-and-discard action sequence is observed and the proposal file agrees (one topic left, renamed); fails if the file still shows two topics or the wrong name", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  const task = await findWorkbenchTask(fixture, "library-edit-candidate");
  const actions = [
    { type: "click", control: { name: "方向审核" } },
    { type: "fill", control: { name: "建议主题名称" }, value: NEW_TOPIC_NAME },
    { type: "click", control: { name: "保存主题名称" } },
    { type: "click", control: { name: "删除候选" } },
    { type: "click", control: { name: "确认" } },
  ];

  // Mutation 1: the file still shows both original topics (neither rename nor discard actually landed).
  await writeLibraryDirectionProposalFile(fixture, twoFixtureTopics());
  const beforeAnyState = await task.verify({ actions });
  assert.equal(beforeAnyState.passed, false);

  // Mutation 2: the discard landed (one topic left) but it was never actually renamed.
  await writeLibraryDirectionProposalFile(fixture, [{ id: "topic-0", suggestedName: "Research evidence 1", directions: [{ id: "candidate-0" }] }]);
  const withoutRename = await task.verify({ actions });
  assert.equal(withoutRename.passed, false);

  // Mutation 3: fully correct — one topic left, renamed.
  await writeLibraryDirectionProposalFile(fixture, [{ id: "topic-0", suggestedName: NEW_TOPIC_NAME, directions: [{ id: "candidate-0" }] }]);
  const result = await task.verify({ actions });
  assert.equal(result.passed, true, JSON.stringify(result));
});
