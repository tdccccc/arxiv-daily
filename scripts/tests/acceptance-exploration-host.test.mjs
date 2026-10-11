import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";
import { Window } from "happy-dom";
import { evaluate as cdpEvaluate } from "../desktop-acceptance/cdp.mjs";
import { describeAcceptanceSuites, parseArgs } from "../acceptance/run.mjs";
import { runExploration, runExplorationLoop } from "../acceptance/exploration.mjs";
import {
  createObsidianExplorationDriver,
  dailyReportFile,
  HISTORY_FILE,
  INDEX_FILE,
  obsidianFocusInScopeExpression,
  paperNoteFile,
  SETTINGS_FILE,
  STATE_FILE,
  startObsidianExplorationSession,
} from "../acceptance/exploration-obsidian.mjs";
import { ACCEPTANCE_DATE, PAPERS } from "../acceptance/fixtures.mjs";
import * as obsidianTaskLibrary from "../acceptance/exploration-tasks/obsidian.mjs";
import * as workbenchTaskLibrary from "../acceptance/exploration-tasks/workbench.mjs";

// ---------------------------------------------------------------------------
// --host arg parsing and threading (run.mjs)
// ---------------------------------------------------------------------------

test("parseArgs defaults host to workbench when --host is not given", () => {
  const options = parseArgs(["--suite", "exploration"]);
  assert.equal(options.host, "workbench");
});

test("parseArgs threads an explicit --host obsidian through", () => {
  const options = parseArgs(["--suite", "exploration", "--host", "obsidian"]);
  assert.equal(options.host, "obsidian");
});

test("parseArgs rejects an unknown --host before anything else is parsed", () => {
  assert.throws(() => parseArgs(["--suite", "exploration", "--host", "bogus"]), /Unknown exploration host: bogus/);
});

test("parseArgs accepts --host=obsidian (equals form), same as every other flag", () => {
  const options = parseArgs(["--suite", "exploration", "--host=obsidian"]);
  assert.equal(options.host, "obsidian");
});

// ---------------------------------------------------------------------------
// runExploration: unknown host, and host -> driver/session/task-library
// selection, exercised with fakes (the `dependencies` test seam) so this
// never launches a real browser or Obsidian.
// ---------------------------------------------------------------------------

test("an unknown exploration host is rejected before any model, session or driver work happens", async () => {
  const touched = [];
  const result = await runExploration({ host: "bogus", fixture: { root: "/x" }, artifactDir: "/tmp/does-not-matter" }, {
    startObsidianExplorationSession: async () => { touched.push("obsidian-session"); },
    createObsidianExplorationDriver: async () => { touched.push("obsidian-driver"); },
    startWorkbenchSession: async () => { touched.push("workbench-session"); },
    createExplorationDriver: async () => { touched.push("workbench-driver"); },
  });
  assert.equal(result.status, "blocked");
  assert.match(result.errors.join(" "), /Unknown exploration host: bogus/);
  assert.deepEqual(touched, [], "no session or driver factory may run for an unknown host");
});

function fakeModel(reply) {
  return async () => reply;
}

test("host=obsidian selects the Obsidian session/driver/task library, never the workbench ones", async () => {
  const seen = [];
  const fakeTask = { id: "fake-obsidian-task", title: "fake", goal: "fake goal", async verify() { return { passed: true, checks: [{ label: "x", passed: true }] }; } };
  const fakeDriver = {
    observe: async () => ({ id: "s1", controls: [], text: "", title: "", layout: { rootMounted: true, view: "dashboard" } }),
    execute: async () => ({ ok: true }),
    capture: async () => undefined,
    reset: async () => { seen.push("reset"); },
    dispose: async () => { seen.push("dispose"); },
  };
  const result = await runExploration({
    host: "obsidian", fixture: { vaultRoot: "/tmp/x", server: { url: "http://127.0.0.1:1" } },
    artifactDir: path.join(tmpdir(), "exploration-host-test-obsidian"), tasks: [fakeTask], model: fakeModel('{"type":"wait","milliseconds":1}'),
  }, {
    startObsidianExplorationSession: async () => { seen.push("obsidian-session"); return { session: { evaluate: async () => {}, client: {}, diagnostics: { entries: () => [] } }, stop: async () => { seen.push("obsidian-stop"); } }; },
    createObsidianExplorationDriver: async () => { seen.push("obsidian-driver"); return fakeDriver; },
    startWorkbenchSession: async () => { seen.push("workbench-session"); return {}; },
    createExplorationDriver: async () => { seen.push("workbench-driver"); return fakeDriver; },
  });
  assert.ok(seen.includes("obsidian-session"));
  assert.ok(seen.includes("obsidian-driver"));
  assert.ok(!seen.includes("workbench-session"));
  assert.ok(!seen.includes("workbench-driver"));
  assert.equal(result.scenarios[0].id, "fake-obsidian-task");
});

test("host=workbench (the default) still selects the workbench session/driver, not Obsidian's", async () => {
  const seen = [];
  const fakeTask = { id: "fake-workbench-task", title: "fake", goal: "fake goal", async verify() { return { passed: true, checks: [{ label: "x", passed: true }] }; } };
  const fakeDriver = {
    observe: async () => ({ id: "s1", controls: [], text: "", title: "", layout: { rootMounted: true } }),
    execute: async () => ({ ok: true }),
    capture: async () => undefined,
  };
  await runExploration({
    fixture: {}, artifactDir: path.join(tmpdir(), "exploration-host-test-workbench"), tasks: [fakeTask], goalTasks: [fakeTask], model: fakeModel('{"type":"wait","milliseconds":1}'),
  }, {
    startObsidianExplorationSession: async () => { seen.push("obsidian-session"); },
    createObsidianExplorationDriver: async () => { seen.push("obsidian-driver"); },
    startWorkbenchSession: async () => { seen.push("workbench-session"); return {}; },
    createExplorationDriver: async () => { seen.push("workbench-driver"); return fakeDriver; },
  });
  assert.ok(seen.includes("workbench-session"));
  assert.ok(seen.includes("workbench-driver"));
  assert.ok(!seen.includes("obsidian-session"));
  assert.ok(!seen.includes("obsidian-driver"));
});

test("--release runs every persona alongside the default goal tasks when --persona was not explicitly given", async () => {
  const fakeDriver = {
    observe: async () => ({ id: "s1", controls: [], text: "", title: "", layout: { rootMounted: true } }),
    execute: async () => ({ ok: true }), capture: async () => undefined,
  };
  const goalTask = { id: "goal-a", title: "t", goal: "g", async verify() { return { passed: true, checks: [{ label: "x", passed: true }] }; } };
  const personaTask = { id: "persona-a", title: "p", kind: "persona", goal: "g" };
  const result = await runExploration({
    fixture: {}, artifactDir: path.join(tmpdir(), "exploration-release-test"),
    goalTasks: [goalTask], personaTasks: [personaTask], release: true, model: fakeModel('{"type":"done"}'),
  }, {
    startWorkbenchSession: async () => ({}), createExplorationDriver: async () => fakeDriver,
  });
  assert.deepEqual(result.scenarios.map(s => s.id), ["goal-a", "persona-a"]);
  assert.equal(result.scenarios[1].status, "explored");
});

test("an explicit --persona still wins over --release (release only fills the default when nothing was chosen)", async () => {
  const fakeDriver = {
    observe: async () => ({ id: "s1", controls: [], text: "", title: "", layout: { rootMounted: true } }),
    execute: async () => ({ ok: true }), capture: async () => undefined,
  };
  const goalTask = { id: "goal-a", title: "t", goal: "g", async verify() { return { passed: true, checks: [{ label: "x", passed: true }] }; } };
  const personaA = { id: "persona-a", title: "p", kind: "persona", goal: "g" };
  const personaB = { id: "persona-b", title: "p", kind: "persona", goal: "g" };
  const result = await runExploration({
    fixture: {}, artifactDir: path.join(tmpdir(), "exploration-release-explicit-test"),
    goalTasks: [goalTask], personaTasks: [personaA, personaB], personaIds: ["persona-b"], release: true, model: fakeModel('{"type":"done"}'),
  }, {
    startWorkbenchSession: async () => ({}), createExplorationDriver: async () => fakeDriver,
  });
  assert.deepEqual(result.scenarios.map(s => s.id), ["goal-a", "persona-b"]);
});

// ---------------------------------------------------------------------------
// Real-host fix: task.prepare() gets a longer per-call ceiling than the other
// per-step host calls, because a real product operation (e.g. the Obsidian
// star task's daily-report generation, measured at ~12s on a real host) can
// legitimately exceed the 10s per-step ceiling used for observation/
// verification/a single action/host reset. This test uses a real ~10.3s
// sleep (deliberately just over the OLD 10000ms ceiling) rather than a mocked
// clock, so it is intentionally one of the slower tests in this suite.
// ---------------------------------------------------------------------------

test("task.prepare() may run longer than 10s (a real product operation can), without hitting the per-step ceiling used for observation/verification/actions/reset", async t => {
  t.timeout?.(20000);
  const driver = {
    observe: async () => ({ id: "s1", controls: [], text: "", title: "", layout: { rootMounted: true } }),
    execute: async () => ({ ok: true }),
    capture: async () => undefined,
  };
  const prepareTask = {
    id: "slow-prepare", title: "slow prepare", goal: "goal",
    async prepare() { await new Promise(resolve => setTimeout(resolve, 10300)); },
    async verify() { return { passed: true, checks: [{ label: "x", passed: true }] }; },
  };
  const result = await runExplorationLoop({
    driver, model: async () => '{"type":"wait","milliseconds":1}', tasks: [prepareTask],
    artifactDir: path.join(tmpdir(), "exploration-host-test-slow-prepare"), maxDurationMs: 30000,
  });
  assert.equal(result.scenarios[0].status, "blocked", "the fake driver's verify() needs at least one non-wait action; what matters here is that prepare itself was not aborted by a 10s timeout");
  assert.ok(!/Task preparation timeout/.test(result.scenarios[0].error ?? ""), `prepare must not be aborted at 10s: ${result.scenarios[0].error}`);
});

// ---------------------------------------------------------------------------
// startObsidianExplorationSession: missing Obsidian/Xvfb/xauth -> blocked,
// never launched (the fixed Obsidian suite's own preflight/blocked handling,
// reused via the `check` seam).
// ---------------------------------------------------------------------------

test("a failing preflight blocks the Obsidian exploration session and never launches anything", async () => {
  let launched = false;
  await assert.rejects(
    () => startObsidianExplorationSession({
      fixture: { vaultRoot: "/tmp/vault", server: { url: "http://127.0.0.1:1" } },
      check: async () => ({ ok: false, blockers: [{ message: "Obsidian was not found at /opt/Obsidian/obsidian", remedy: "install Obsidian" }] }),
      runSession: async () => { launched = true; },
    }),
    error => {
      assert.equal(error.name, "ObsidianExplorationBlockedError");
      assert.match(error.message, /Obsidian was not found/);
      return true;
    },
  );
  assert.equal(launched, false);
});

test("a preflight that itself throws is reported as blocked, not a crash", async () => {
  await assert.rejects(
    () => startObsidianExplorationSession({
      fixture: { vaultRoot: "/tmp/vault", server: { url: "http://127.0.0.1:1" } },
      check: async () => { throw new Error("inotify probe failed"); },
      runSession: async () => { throw new Error("must never be called"); },
    }),
    error => {
      assert.equal(error.name, "ObsidianExplorationBlockedError");
      assert.match(error.message, /preflight could not complete.*inotify probe failed/s);
      return true;
    },
  );
});

test("a passing preflight starts the session, hands back the renderer session, and stop() releases runDesktopSession's own body/teardown", async () => {
  let bodyRan = false, stopped = false;
  const fakeSessionInfo = { session: { evaluate: async () => {}, client: {} }, port: 1234 };
  const result = await startObsidianExplorationSession({
    fixture: { vaultRoot: "/tmp/vault", server: { url: "http://127.0.0.1:1" } },
    check: async () => ({ ok: true }),
    runSession: async ({ body }) => {
      bodyRan = true;
      await body(fakeSessionInfo);
      stopped = true;
      return "teardown-complete";
    },
  });
  assert.equal(result.session, fakeSessionInfo.session);
  assert.equal(result.port, 1234);
  assert.equal(bodyRan, true);
  assert.equal(stopped, false, "runSession's body is still blocking until stop() is called");
  const teardown = await result.stop();
  assert.equal(stopped, true);
  assert.equal(teardown, "teardown-complete");
});

test("an abort signal releases a held-open Obsidian session so its own teardown still runs", async () => {
  const controller = new AbortController();
  let released = false;
  const result = await startObsidianExplorationSession({
    fixture: { vaultRoot: "/tmp/vault", server: { url: "http://127.0.0.1:1" } },
    check: async () => ({ ok: true }),
    signal: controller.signal,
    runSession: async ({ body }) => { await body({ session: {} }); released = true; },
  });
  assert.equal(released, false);
  controller.abort(new Error("cancelled"));
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(released, true, "aborting released the held body without an explicit stop() call");
  await result.stop();
});

// ---------------------------------------------------------------------------
// exploration-tasks/obsidian.mjs: shape (program verify, no persona
// default tasks, Chinese goals), moved out of exploration-obsidian.mjs so
// the per-host driver (observation/execution) and per-host task library
// (what to do, how to tell it passed) live in separate, independently
// growable files (P3 Chunk 1 of the task-library phase).
// ---------------------------------------------------------------------------

test("the default Obsidian goal tasks have ids, titles, Chinese goals and a program verify, matching the exported id list", async () => {
  const tasks = await obsidianTaskLibrary.goalTasks();
  assert.deepEqual(tasks.map(t => t.id), obsidianTaskLibrary.GOAL_TASK_IDS);
  // Ids are unchanged by the move (phase contract: "keeping existing task ids/behavior unchanged"); P3
  // Chunk 2 appends topic/direction/category tasks after the original two, P3 Chunk 3 appends the
  // four run tasks after those, and P3 Chunk 4 appends the five reading tasks, keeping every earlier id
  // stable.
  assert.deepEqual(obsidianTaskLibrary.GOAL_TASK_IDS, [
    "daily-paper-limit", "star-paper-from-report",
    "add-topic-direction", "remove-topic-direction", "add-arxiv-category",
    "run-for-date", "cancel-then-rerun", "auth-failure-retry", "completed-date-idempotent",
    "dashboard-tabs-paging", "mark-to-read", "create-paper-note", "summarize-by-id", "open-today-report",
  ]);
  for (const t of tasks) {
    assert.equal(typeof t.title, "string");
    assert.ok(t.title.length > 0);
    assert.ok(/[一-鿿]/.test(t.goal), `goal should be in Chinese: ${t.goal}`);
    assert.equal(typeof t.verify, "function");
    assert.notEqual(t.kind, "persona");
  }
  assert.equal(typeof tasks[1].prepare, "function", "the star task needs prepare() to deterministically generate a daily report");
});

// ---------------------------------------------------------------------------
// Task-library structure (P3 Chunk 1): scripts/acceptance/exploration-tasks/
// {obsidian,workbench}.mjs each export GOAL_TASK_IDS, PERSONA_TASK_IDS,
// goalTasks(fixture|ctx), and personaTasks() — the single place chunks 2-5
// add tasks/personas to, and the single place `--list` and task selection
// read ids from.
// ---------------------------------------------------------------------------

test("both host task-library modules export the documented shape", () => {
  for (const [name, lib] of [["obsidian", obsidianTaskLibrary], ["workbench", workbenchTaskLibrary]]) {
    assert.ok(Array.isArray(lib.GOAL_TASK_IDS) && lib.GOAL_TASK_IDS.length > 0, `${name}: GOAL_TASK_IDS must be a non-empty array`);
    assert.ok(Array.isArray(lib.PERSONA_TASK_IDS), `${name}: PERSONA_TASK_IDS must be an array (empty is fine: none yet)`);
    assert.equal(typeof lib.goalTasks, "function", `${name}: goalTasks must be exported`);
    assert.equal(typeof lib.personaTasks, "function", `${name}: personaTasks must be exported`);
  }
});

test("ids are unique across goal tasks and personas combined, per host", async () => {
  for (const lib of [obsidianTaskLibrary, workbenchTaskLibrary]) {
    const ids = [...lib.GOAL_TASK_IDS, ...lib.PERSONA_TASK_IDS];
    assert.equal(new Set(ids).size, ids.length, `duplicate ids within one host's task library: ${ids.join(", ")}`);
  }
});

test("the workbench goal task ids are unchanged by the move and match goalTasks()'s actual ids", async () => {
  // P3 Chunk 5 appends three more ids after the original three, keeping every earlier id stable.
  assert.deepEqual(workbenchTaskLibrary.GOAL_TASK_IDS, [
    "settings-persistence", "sidebar-persistence", "generate-and-read",
    "star-filter-restore", "generate-paper-note", "external-config-conflict",
  ]);
  const root = await mkdtemp(path.join(tmpdir(), "exploration-task-library-workbench-"));
  try {
    const vaultRoot = path.join(root, "vault");
    const configPath = path.join(root, "config.toml");
    await mkdir(vaultRoot);
    await writeFile(configPath, `vault_root = ${JSON.stringify(vaultRoot)}\n[output]\nmax_daily_papers = 2\n`);
    const tasks = await workbenchTaskLibrary.goalTasks({ root, vaultRoot, configPath });
    assert.deepEqual(tasks.map(t => t.id), workbenchTaskLibrary.GOAL_TASK_IDS);
  } finally { await rm(root, { recursive: true, force: true }); }
});

// P3 Chunk 5: both hosts now share the same three persona ids (exploration-tasks/personas.mjs).
test("both hosts expose the same three persona ids, each a real kind:\"persona\" task with no verify and a worded-for-that-host Chinese brief", () => {
  assert.deepEqual(obsidianTaskLibrary.PERSONA_TASK_IDS, ["first-time-user", "impatient-clicker", "keyboard-only"]);
  assert.deepEqual(workbenchTaskLibrary.PERSONA_TASK_IDS, obsidianTaskLibrary.PERSONA_TASK_IDS);
  for (const [host, lib] of [["obsidian", obsidianTaskLibrary], ["workbench", workbenchTaskLibrary]]) {
    const personas = lib.personaTasks();
    assert.deepEqual(personas.map(p => p.id), lib.PERSONA_TASK_IDS, `${host}: personaTasks() order must match PERSONA_TASK_IDS`);
    for (const p of personas) {
      assert.equal(p.kind, "persona", `${host}/${p.id}: must be kind "persona"`);
      assert.equal(p.verify, undefined, `${host}/${p.id}: persona tasks must omit verify`);
      assert.equal(typeof p.title, "string"); assert.ok(p.title.length > 0);
      assert.ok(/[一-鿿]/.test(p.goal), `${host}/${p.id}: goal should be in Chinese`);
      assert.ok(Number.isInteger(p.maxSteps) && p.maxSteps >= 1 && p.maxSteps <= 20, `${host}/${p.id}: maxSteps must be a sane positive integer, got ${p.maxSteps}`);
    }
  }
  // keyboard-only's brief is the one persona with a hard behavioral constraint: it must actually say so.
  for (const lib of [obsidianTaskLibrary, workbenchTaskLibrary]) {
    const keyboardOnly = lib.personaTasks().find(p => p.id === "keyboard-only");
    assert.match(keyboardOnly.goal, /press/);
    assert.match(keyboardOnly.goal, /wait/);
    assert.match(keyboardOnly.goal, /click/);
  }
  // Same ids, different wording per host (so the brief actually names the right surface).
  const obsidianBriefs = obsidianTaskLibrary.personaTasks().map(p => p.goal);
  const workbenchBriefs = workbenchTaskLibrary.personaTasks().map(p => p.goal);
  for (let i = 0; i < obsidianBriefs.length; i++) assert.notEqual(obsidianBriefs[i], workbenchBriefs[i], "each persona's brief must be worded per host, not identical copy-paste");
});

test("--list (describeAcceptanceSuites) lists both hosts' goal task ids and personas", async () => {
  const output = await describeAcceptanceSuites();
  assert.match(output, /obsidian/);
  assert.match(output, /workbench/);
  for (const id of obsidianTaskLibrary.GOAL_TASK_IDS) assert.ok(output.includes(id), `--list must mention Obsidian task id ${id}`);
  for (const id of workbenchTaskLibrary.GOAL_TASK_IDS) assert.ok(output.includes(id), `--list must mention workbench task id ${id}`);
  // P3 Chunk 5: both hosts now have personas; --list must mention each persona id, and the output must
  // explain --release.
  for (const id of obsidianTaskLibrary.PERSONA_TASK_IDS) assert.ok(output.includes(id), `--list must mention persona id ${id}`);
  assert.match(output, /--release/);
});

// ---------------------------------------------------------------------------
// createObsidianExplorationDriver.reset(): call ordering against a fake CDP
// session (happy-dom + a recording fake client, same convention as
// acceptance-exploration-obsidian.test.mjs's buildSession), and failure ->
// throws when the measured duration exceeds the 30s phase-abort budget.
// ---------------------------------------------------------------------------

function buildFakeObsidianSession({ modalOpen = false, settingsMarker = false, diagnosticsEntries = [] } = {}) {
  const window = new Window();
  window.document.body.innerHTML = `
    ${modalOpen ? '<div class="modal-container"><div class="modal arxiv-daily-modal"><button type="button">Close</button></div></div>' : ""}
    <div class="view-content arxiv-daily-dashboard">
      <div class="arxiv-daily-dashboard__tab is-active" role="tab" tabindex="0">All</div>
    </div>
    ${settingsMarker ? '<input class="arxiv-daily-settings__model-input" />' : ""}
  `;
  const calls = [];
  const reloadCalls = [];
  const httpClient = { request: async () => ({}) };
  const context = {
    document: window.document,
    getComputedStyle: (...args) => window.getComputedStyle(...args),
    innerWidth: 1200, innerHeight: 800,
    app: {
      setting: { isOpen: modalOpen, activeTab: undefined, open() { calls.push("settings.open"); this.isOpen = true; }, close() { calls.push("settings.close"); this.isOpen = false; }, openTabById() { calls.push("settings.openTabById"); } },
      // disablePlugin/enablePlugin mock a full plugin reload the way the fixed suite's own
      // reloadPlugin() (obsidian-ui.mjs) drives it — the real fix for the settings-desync bug found on
      // the real host (see exploration-obsidian.mjs's reset() comment): a bare loadSettingsAndState()
      // call desyncs SettingsChangeService's captured settings reference from the live plugin.settings.
      plugins: {
        plugins: { "arxiv-daily": { getHttpClient: () => httpClient } },
        disablePlugin: async () => { reloadCalls.push("disable"); },
        enablePlugin: async () => { reloadCalls.push("enable"); },
      },
      commands: { executeCommandById: () => true },
    },
  };
  const client = {
    calls,
    async send(method, params = {}) {
      if (method === "Runtime.evaluate") {
        let value;
        try { value = vm.runInNewContext(params.expression, context); }
        catch (error) { return { result: {}, exceptionDetails: { text: "Uncaught", exception: { description: error.message } } }; }
        return { result: { value } };
      }
      calls.push({ method, params });
      if (method === "Input.dispatchMouseEvent" && params.type === "mousePressed") window.document.elementFromPoint(params.x, params.y)?.focus?.();
      return {};
    },
  };
  const evaluate = expression => cdpEvaluate(client, expression);
  // `diagnosticsEntries` is the same mutable array the caller pushed into (not a copy): observe() calls
  // `session.diagnostics.entries()` fresh every time and slices off what it already consumed
  // (`diagnosticsSeen`), so a test can push new entries between two observe() calls, exactly like a real
  // CDP session accumulating console/pageerror entries over time.
  return { window, calls, reloadCalls, diagnosticsEntries, session: { client, evaluate, diagnostics: { entries: () => diagnosticsEntries } } };
}

async function withTempFixture(run) {
  const root = await mkdtemp(path.join(tmpdir(), "exploration-obsidian-driver-test-"));
  const vaultRoot = path.join(root, "vault");
  await mkdir(path.join(vaultRoot, ".obsidian", "plugins", "arxiv-daily"), { recursive: true });
  const artifactDir = path.join(root, "artifacts");
  await mkdir(artifactDir, { recursive: true });
  const fixture = { vaultRoot, server: { url: "http://127.0.0.1:1" } };
  try { return await run({ fixture, artifactDir }); }
  finally { await rm(root, { recursive: true, force: true }); }
}

test("reset() closes any plugin modal, closes settings, rewrites data.json, fully reloads the plugin, and opens the dashboard, in that order", async () => {
  await withTempFixture(async ({ fixture, artifactDir }) => {
    const { calls, reloadCalls, session } = buildFakeObsidianSession({ modalOpen: true });
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    const before = calls.length;
    const { durationMs } = await driver.reset();
    assert.ok(Number.isFinite(durationMs) && durationMs >= 0);
    const after = calls.slice(before);
    // Escape (keyDown/keyUp) to close the plugin modal happens first, before any settings/data.json work.
    const firstKeyIndex = after.findIndex(c => c.method === "Input.dispatchKeyEvent");
    assert.ok(firstKeyIndex === 0, "the modal-closing Escape is the first thing reset() does");
    assert.equal(after[firstKeyIndex].params.key, "Escape");
    // A full disable/enable reload (not a bare settings re-read) — the real-host fix.
    assert.deepEqual(reloadCalls, ["disable", "enable"], "reset reloads the whole plugin, once, in order");
    const written = JSON.parse(await (await import("node:fs/promises")).readFile(path.join(fixture.vaultRoot, ".obsidian/plugins/arxiv-daily/data.json"), "utf8"));
    assert.equal(written.settings.output.maxDailyPapers, 20, "data.json was rewritten back to the fixture baseline");
  });
});

test("reset() leaves keyboard focus inside the dashboard's own surface, so the keyboard-only persona's first press is not rejected as out-of-scope", async () => {
  await withTempFixture(async ({ fixture, artifactDir }) => {
    const { window, session } = buildFakeObsidianSession();
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    await driver.reset();
    assert.equal(window.document.activeElement?.className, "arxiv-daily-dashboard__tab is-active", "the dashboard's first tab must hold real keyboard focus after reset(), not body or nothing");
    const scoped = await session.evaluate(obsidianFocusInScopeExpression());
    assert.equal(scoped, true, "obsidianFocusInScopeExpression must agree focus is inside the plugin surface right after reset()");
  });
});

test("reset() throws when the measured duration exceeds the 30s phase-abort budget", async () => {
  await withTempFixture(async ({ fixture, artifactDir }) => {
    const { session } = buildFakeObsidianSession();
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    const realNow = Date.now;
    let calls = 0;
    Date.now = () => { calls += 1; return calls === 1 ? realNow() : realNow() + 40000; };
    try {
      await assert.rejects(() => driver.reset(), /30s|phase-abort/);
    } finally { Date.now = realNow; }
  });
});

test("observe() rejects a stale observation id at execute() without sending any input", async () => {
  await withTempFixture(async ({ fixture, artifactDir }) => {
    const { calls, session } = buildFakeObsidianSession();
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    const stale = await driver.observe();
    await driver.observe(); // advances the driver's own "latest" observation
    const before = calls.length;
    const result = await driver.execute({ type: "click", ref: stale.controls[0]?.ref ?? "s1-1" }, stale);
    assert.equal(result.ok, false);
    assert.match(result.error, /[Ss]tale/);
    assert.equal(calls.length, before, "no CDP input call was sent for a stale observation");
  });
});

// ---------------------------------------------------------------------------
// P3 Chunk 3: driver.expectAuthorizationFailure()'s allowance must be narrow
// — only the task that requested it, only authorization-failure diagnostics
// (never a blanket "ignore every console error" switch), and not carried
// over once reset() moves to the next task. Exercised against observe()
// itself (not just classifyObsidianDiagnostics in isolation), since the bug
// this guards against is observe() wiring the carve-out in at all.
// ---------------------------------------------------------------------------

const CONTROLLED_401_TEXT = "paper-filter: LLM call failed: Controlled acceptance authentication failure";

test("expectAuthorizationFailure(): the controlled 401 console error is suppressed, but an unrelated console error in the very same task still surfaces", async () => {
  await withTempFixture(async ({ fixture, artifactDir }) => {
    fixture.server.requests = [{ method: "POST", kind: "model", status: 401 }];
    const diagnosticsEntries = [];
    const { session } = buildFakeObsidianSession({ diagnosticsEntries });
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    driver.expectAuthorizationFailure();
    diagnosticsEntries.push(
      { source: "console", level: "error", text: CONTROLLED_401_TEXT },
      { source: "console", level: "error", text: "Something else broke" },
    );
    const snapshot = await driver.observe();
    assert.ok(!snapshot.errors.some(text => text.includes(CONTROLLED_401_TEXT)), "the controlled authentication failure must be allowed, not surfaced as a product error");
    assert.ok(snapshot.errors.some(text => text.includes("Something else broke")), "an unrelated console error in the same task must still be treated as unexpected");
  });
});

test("expectAuthorizationFailure(): without the fixture actually having returned a 401, the same console text is not suppressed", async () => {
  await withTempFixture(async ({ fixture, artifactDir }) => {
    fixture.server.requests = []; // no 401 was ever actually returned
    const diagnosticsEntries = [];
    const { session } = buildFakeObsidianSession({ diagnosticsEntries });
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    driver.expectAuthorizationFailure();
    diagnosticsEntries.push({ source: "console", level: "error", text: CONTROLLED_401_TEXT });
    const snapshot = await driver.observe();
    assert.ok(snapshot.errors.some(text => text.includes(CONTROLLED_401_TEXT)), "opting in alone must not suppress the error without a matching real 401 fixture response");
  });
});

test("expectAuthorizationFailure(): the allowance does not carry over once reset() moves on to the next task", async () => {
  await withTempFixture(async ({ fixture, artifactDir }) => {
    fixture.server.requests = [{ method: "POST", kind: "model", status: 401 }];
    const diagnosticsEntries = [];
    const { session } = buildFakeObsidianSession({ diagnosticsEntries });
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    driver.expectAuthorizationFailure();
    diagnosticsEntries.push({ source: "console", level: "error", text: CONTROLLED_401_TEXT });
    const during = await driver.observe();
    assert.ok(!during.errors.some(text => text.includes(CONTROLLED_401_TEXT)), "allowed while the task that requested it is still active");

    await driver.reset(); // simulates moving on to the next task
    diagnosticsEntries.push({ source: "console", level: "error", text: CONTROLLED_401_TEXT });
    const after = await driver.observe();
    assert.ok(after.errors.some(text => text.includes(CONTROLLED_401_TEXT)), "reset() must clear the allowance: the next task did not opt in, so the same text is unexpected again");
  });
});

// ---------------------------------------------------------------------------
// P3 Chunk 2: driver additions the new Obsidian goal tasks rely on —
// applySettingsPatch(mutate) (a task's prepare() seeding data.json directly,
// never through the model) and reloadThenReopenSettings() (proof a saved
// setting survives a full plugin reload, not only re-opening the settings
// tab). Exercised against the same fake CDP session/fixture convention as
// reset() above, not a fake driver, since these are real driver behavior.
// ---------------------------------------------------------------------------

test("driver.applySettingsPatch writes the mutated settings to data.json and fully reloads the plugin", async () => {
  await withTempFixture(async ({ fixture, artifactDir }) => {
    const { reloadCalls, session } = buildFakeObsidianSession();
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    const dataPath = path.join(fixture.vaultRoot, SETTINGS_FILE);
    await mkdir(path.dirname(dataPath), { recursive: true });
    await writeFile(dataPath, JSON.stringify({ settings: { output: { maxDailyPapers: 20 } } }));
    await driver.applySettingsPatch(current => { current.settings.output.maxDailyPapers = 42; });
    const written = JSON.parse(await (await import("node:fs/promises")).readFile(dataPath, "utf8"));
    assert.equal(written.settings.output.maxDailyPapers, 42, "the mutation must be persisted to data.json");
    assert.deepEqual(reloadCalls, ["disable", "enable"], "applySettingsPatch must fully reload the plugin, once, so the write actually takes effect");
  });
});

test("driver.reloadThenReopenSettings reloads the plugin before (re)opening settings, not after", async () => {
  await withTempFixture(async ({ fixture, artifactDir }) => {
    const { calls, reloadCalls, session } = buildFakeObsidianSession({ settingsMarker: true });
    const driver = await createObsidianExplorationDriver({ session, fixture, artifactDir });
    const before = calls.length;
    await driver.reloadThenReopenSettings();
    const after = calls.slice(before);
    assert.deepEqual(reloadCalls, ["disable", "enable"]);
    const openIndex = after.findIndex(c => c === "settings.open");
    assert.ok(openIndex >= 0, "settings must be (re)opened");
    // reloadPlugin() itself closes settings first (helper.reloadPlugin), then reloadThenReopenSettings
    // opens them again — the open must come after the close the reload performed.
    const closeIndex = after.findIndex(c => c === "settings.close");
    assert.ok(closeIndex >= 0 && openIndex > closeIndex, "settings must close (as part of the reload) before it is reopened");
  });
});

// ---------------------------------------------------------------------------
// P3 Chunk 2: the four new/extended Obsidian goal tasks' verify() functions,
// exercised against a fake driver exposing only jsonFile/applySettingsPatch/
// closeThenReopenSettings/reloadThenReopenSettings/read — never a real CDP
// session — per the phase's "prefer unit tests with a fake driver exposing
// jsonFile/read results" acceptance bar. Each task is checked both on the
// untouched fixture baseline (verify must NOT pass — the negative evidence
// the phase requires) and once the real, independent on-disk/UI state
// reflects the completed goal (verify must pass).
// ---------------------------------------------------------------------------

/** A fake driver whose jsonFile/read/applySettingsPatch all read and write the same in-memory "disk". */
function createFakeSettingsDriver(initialSettings) {
  let disk = structuredClone(initialSettings);
  return {
    async jsonFile(relative) {
      assert.equal(relative, SETTINGS_FILE, `task verify() must read the real settings file, not ${relative}`);
      return structuredClone(disk);
    },
    async applySettingsPatch(mutate) { mutate(disk); },
    async closeThenReopenSettings() {},
    async reloadThenReopenSettings() {},
    async openDashboard() {},
    async read(expression) {
      const topic = disk.settings.arxiv.topics.find(item => item.id === obsidianTaskLibrary.FIXTURE_TOPIC_ID);
      if (expression.includes("topic-direction-input")) return (topic?.directions ?? []).map(direction => direction.text);
      if (expression.includes(obsidianTaskLibrary.TARGET_CATEGORY_ID)) return disk.settings.arxiv.categories ?? [];
      if (expression.includes("Daily paper limit")) return String(disk.settings.output.maxDailyPapers);
      throw new Error(`Unhandled fake read() expression: ${expression}`);
    },
  };
}

function fixtureSettingsBaseline() {
  return {
    settings: {
      output: { maxDailyPapers: 20 },
      arxiv: {
        categories: ["cs.AI"],
        topics: [{
          id: obsidianTaskLibrary.FIXTURE_TOPIC_ID, name: obsidianTaskLibrary.FIXTURE_TOPIC_NAME, tag: "research", detail: false,
          directions: [{ id: "acceptance-reliable", text: obsidianTaskLibrary.FIXTURE_DIRECTION_TEXT, origin: "manual" }],
        }],
      },
    },
  };
}

test("add-topic-direction: verify fails on the untouched fixture and passes once the new direction is saved and visible after reopening", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "add-topic-direction");
  assert.ok(task, "add-topic-direction must exist in the Obsidian goal task library");
  const driver = createFakeSettingsDriver(fixtureSettingsBaseline());
  const untouched = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model has added anything");

  await driver.applySettingsPatch(current => {
    current.settings.arxiv.topics[0].directions.push({ id: "new", text: obsidianTaskLibrary.NEW_DIRECTION_TEXT, origin: "manual" });
  });
  const done = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(done.passed, true, `verify should pass once the new direction is saved and visible: ${JSON.stringify(done)}`);
});

test("add-topic-direction: verify still fails if the existing direction was overwritten instead of a new one being added", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "add-topic-direction");
  const driver = createFakeSettingsDriver(fixtureSettingsBaseline());
  await driver.applySettingsPatch(current => {
    current.settings.arxiv.topics[0].directions[0].text = obsidianTaskLibrary.NEW_DIRECTION_TEXT;
  });
  const result = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(result.passed, false, "replacing the original direction instead of adding a new one must not pass");
});

test("remove-topic-direction: prepare() deterministically seeds a second direction (not via the model); verify fails until it is removed and the original is kept", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "remove-topic-direction");
  assert.ok(task, "remove-topic-direction must exist in the Obsidian goal task library");
  assert.equal(typeof task.prepare, "function");
  const driver = createFakeSettingsDriver(fixtureSettingsBaseline());

  await task.prepare({ driver });
  const afterPrepare = await driver.jsonFile(SETTINGS_FILE);
  const topicAfterPrepare = afterPrepare.settings.arxiv.topics.find(item => item.id === obsidianTaskLibrary.FIXTURE_TOPIC_ID);
  assert.equal(topicAfterPrepare.directions.length, 2, "prepare must seed exactly one extra direction");
  assert.ok(topicAfterPrepare.directions.some(direction => direction.text === obsidianTaskLibrary.SEEDED_DIRECTION_TEXT));
  assert.ok(topicAfterPrepare.directions.some(direction => direction.text === obsidianTaskLibrary.FIXTURE_DIRECTION_TEXT));

  const beforeRemoval = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(beforeRemoval.passed, false, "verify must not pass while the seeded direction is still present (prepare alone must not satisfy the task)");

  await driver.applySettingsPatch(current => {
    const topic = current.settings.arxiv.topics.find(item => item.id === obsidianTaskLibrary.FIXTURE_TOPIC_ID);
    topic.directions = topic.directions.filter(direction => direction.text !== obsidianTaskLibrary.SEEDED_DIRECTION_TEXT);
  });
  const afterRemoval = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(afterRemoval.passed, true, `verify should pass once only the seeded direction is removed: ${JSON.stringify(afterRemoval)}`);
});

test("remove-topic-direction: verify fails if the original direction was removed instead of the seeded one", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "remove-topic-direction");
  const driver = createFakeSettingsDriver(fixtureSettingsBaseline());
  await task.prepare({ driver });
  await driver.applySettingsPatch(current => {
    const topic = current.settings.arxiv.topics.find(item => item.id === obsidianTaskLibrary.FIXTURE_TOPIC_ID);
    topic.directions = topic.directions.filter(direction => direction.text !== obsidianTaskLibrary.FIXTURE_DIRECTION_TEXT);
  });
  const result = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(result.passed, false, "removing the wrong direction must not pass");
});

test("add-arxiv-category: verify fails on the untouched fixture and passes once the target category is added and cs.AI is kept", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "add-arxiv-category");
  assert.ok(task, "add-arxiv-category must exist in the Obsidian goal task library");
  const driver = createFakeSettingsDriver(fixtureSettingsBaseline());
  const untouched = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model has added the category");

  await driver.applySettingsPatch(current => { current.settings.arxiv.categories.push(obsidianTaskLibrary.TARGET_CATEGORY_ID); });
  const done = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(done.passed, true, `verify should pass once the target category is present alongside cs.AI: ${JSON.stringify(done)}`);
});

test("add-arxiv-category: verify fails if cs.AI was replaced rather than kept alongside the new category", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "add-arxiv-category");
  const driver = createFakeSettingsDriver(fixtureSettingsBaseline());
  await driver.applySettingsPatch(current => { current.settings.arxiv.categories = [obsidianTaskLibrary.TARGET_CATEGORY_ID]; });
  const result = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(result.passed, false, "losing the original category must not pass, even though the target category is present");
});

test("daily-paper-limit: verify's new reload proof actually reloads the plugin once persistence is confirmed, and only once per task run", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "daily-paper-limit");
  const settings = fixtureSettingsBaseline();
  settings.settings.output.maxDailyPapers = 7;
  let reloadCalls = 0;
  const driver = createFakeSettingsDriver(settings);
  driver.reloadThenReopenSettings = async () => { reloadCalls += 1; };
  const actions = [{ type: "fill", control: { name: "Daily paper limit" }, value: "7" }];

  const result = await task.verify({ driver, observation: {}, actions });
  assert.equal(result.passed, true, JSON.stringify(result));
  assert.equal(reloadCalls, 1, "verify must reload the plugin to prove the setting survives a reload, not only a settings reopen");
  assert.ok(result.checks.some(check => /重载/.test(check.label)), "a check must specifically name the full-plugin-reload proof");

  await task.verify({ driver, observation: {}, actions }); // verify is polled repeatedly by the real loop
  assert.equal(reloadCalls, 1, "the expensive reload must happen at most once per task run (memoized)");
});

test("daily-paper-limit: verify fails (and never reloads the plugin) while the on-disk value does not match the target yet", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "daily-paper-limit");
  const settings = fixtureSettingsBaseline(); // maxDailyPapers stays at the fixture default (20)
  let reloadCalls = 0;
  const driver = createFakeSettingsDriver(settings);
  driver.reloadThenReopenSettings = async () => { reloadCalls += 1; };
  const actions = [{ type: "fill", control: { name: "Daily paper limit" }, value: "7" }];

  const result = await task.verify({ driver, observation: {}, actions });
  assert.equal(result.passed, false);
  assert.equal(reloadCalls, 0, "a plugin reload is only worth paying for once persistence on disk is already confirmed");
});

// P3 Chunk 7 negative-evidence audit: star-paper-from-report had no dedicated "verify fails on the
// untouched state" unit test anywhere in the suite before this — added here, same convention as every
// other Obsidian goal task's own untouched-state test above/below.
test("star-paper-from-report: verify fails on the untouched state (no star click yet), never touching the driver", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "star-paper-from-report");
  assert.ok(task, "star-paper-from-report must exist in the Obsidian goal task library");
  let driverTouched = false;
  const driver = new Proxy({}, { get() { driverTouched = true; throw new Error("verify must not need the driver before a star click is observed"); } });
  const result = await task.verify({ driver, actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false);
  assert.equal(driverTouched, false);
});

// ---------------------------------------------------------------------------
// P3 Chunk 3: the four new Obsidian run goal tasks (run-for-date,
// cancel-then-rerun, auth-failure-retry, completed-date-idempotent). A fake
// driver stands in for the real CDP session, exposing exactly the surface
// these tasks' prepare()/verify() use (jsonFile/file/fixture.server/
// clearDateState/expectAuthorizationFailure/openDashboard/idle/
// ensureDailyReport) — never a real Obsidian/CDP session, per the phase's
// "prefer a fake driver" acceptance bar. Each task is checked on the
// untouched state (must not pass), on at least one plausible wrong outcome
// (must not pass), and on the fully correct state (must pass).
// ---------------------------------------------------------------------------

function createFakeRunDriver({ state = {}, index = {}, historyLines = [], reports = {} } = {}) {
  const disk = { state: structuredClone({ runState: {}, ...state }), index: structuredClone({ papers: {}, ...index }), historyLines: structuredClone(historyLines), reports: structuredClone(reports) };
  const requests = [];
  const calls = { setMode: [], releaseHeld: 0, clearDateState: [], expectAuthorizationFailure: 0 };
  const notFound = () => { const error = new Error("not found"); error.code = "ENOENT"; throw error; };
  return {
    disk, requests, calls,
    async jsonFile(relative) {
      if (relative === STATE_FILE) return structuredClone(disk.state);
      if (relative === INDEX_FILE) return structuredClone(disk.index);
      throw new Error(`Unhandled fake jsonFile: ${relative}`);
    },
    async file(relative) {
      if (relative === HISTORY_FILE) return disk.historyLines.length ? `${disk.historyLines.map(line => JSON.stringify(line)).join("\n")}\n` : "";
      const date = Object.keys(disk.reports).find(d => relative === dailyReportFile(d));
      return date ? disk.reports[date] : notFound();
    },
    fixture: { server: {
      requests,
      setMode(mode) { calls.setMode.push(mode); },
      releaseHeld() { calls.releaseHeld += 1; },
    } },
    async clearDateState(date) {
      calls.clearDateState.push(date);
      delete disk.state.runState[date];
      delete disk.reports[date];
      for (const entry of Object.values(disk.index.papers)) {
        if (Array.isArray(entry.dailyReports)) entry.dailyReports = entry.dailyReports.filter(p => !p.endsWith(`${date}.md`));
      }
    },
    expectAuthorizationFailure() { calls.expectAuthorizationFailure += 1; },
    async openDashboard() {},
    async idle() {},
    async ensureDailyReport(date) { if (!disk.reports[date]) disk.reports[date] = `# daily ${date}\n`; },
  };
}

function fillDate(value) { return { type: "fill", control: { name: "Date" }, value }; }
function clickNamed(name) { return { type: "click", control: { name } }; }

function indexWithReports(date) {
  return { papers: Object.fromEntries(PAPERS.map((paper, i) => [`p${i}`, { arxivId: paper.id, dailyReports: [dailyReportFile(date)] }])) };
}

function reportMarkdown(date) {
  return `# arXiv Daily ${date}\n\n${PAPERS.map(paper => `## ${paper.title}\n\narXiv:${paper.id}\n`).join("\n")}`;
}

test("run-for-date: verify fails on the untouched state, fails while the run has not reached completed, and passes once run-state/report/index all agree", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "run-for-date");
  assert.ok(task, "run-for-date must exist in the Obsidian goal task library");
  const date = obsidianTaskLibrary.RUN_FOR_DATE_DATE;

  const untouchedDriver = createFakeRunDriver({});
  const untouched = await task.verify({ driver: untouchedDriver, actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model submits anything");

  const stuckDriver = createFakeRunDriver({ state: { runState: { [date]: { status: "failed_permanent" } } } });
  const stuck = await task.verify({ driver: stuckDriver, actions: [fillDate(date)] });
  assert.equal(stuck.passed, false, "verify must not pass while the run has not completed (a plausible wrong outcome: a failed run)");

  const doneDriver = createFakeRunDriver({
    state: { runState: { [date]: { status: "completed" } } },
    index: indexWithReports(date),
    reports: { [date]: reportMarkdown(date) },
  });
  const done = await task.verify({ driver: doneDriver, actions: [fillDate(date)] });
  assert.equal(done.passed, true, `verify should pass once the date is completed with a real report and index: ${JSON.stringify(done)}`);
});

test("cancel-then-rerun: verify requires an actual Cancel click (not just a fill), a recorded cancellation, and a later successful rerun; releasing the held fixture request is gated on the model's own cancel click", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "cancel-then-rerun");
  assert.ok(task, "cancel-then-rerun must exist in the Obsidian goal task library");
  const date = obsidianTaskLibrary.CANCEL_THEN_RERUN_DATE;

  const untouchedDriver = createFakeRunDriver({});
  const untouched = await task.verify({ driver: untouchedDriver, actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model submits anything");

  const noCancelDriver = createFakeRunDriver({});
  const noCancel = await task.verify({ driver: noCancelDriver, actions: [fillDate(date)] });
  assert.equal(noCancel.passed, false, "verify must not pass without a Cancel active tasks click (a plausible wrong outcome: just waiting for the run to finish)");
  assert.equal(noCancelDriver.calls.releaseHeld, 0, "the held fixture request must not be released before the model's own cancel click");

  const fullDriver = createFakeRunDriver({
    state: { runState: { [date]: { status: "completed" } } },
    index: indexWithReports(date),
    historyLines: [{ date, resultKind: "cancelled" }],
  });
  const actions = [fillDate(date), clickNamed("Cancel active tasks"), fillDate(date)];
  const done = await task.verify({ driver: fullDriver, actions });
  assert.equal(done.passed, true, `verify should pass once cancellation is recorded and the rerun completed without duplicates: ${JSON.stringify(done)}`);
  assert.equal(fullDriver.calls.releaseHeld, 1, "the held request is released exactly once, after the model's cancel click");
  assert.deepEqual(fullDriver.calls.setMode, [{ llm: "normal" }], "the fixture is restored to normal exactly once");

  const duplicateDriver = createFakeRunDriver({
    state: { runState: { [date]: { status: "completed" } } },
    index: { papers: { p0: { arxivId: PAPERS[0].id, dailyReports: [dailyReportFile(date), dailyReportFile(date)] } } },
    historyLines: [{ date, resultKind: "cancelled" }],
  });
  const duplicate = await task.verify({ driver: duplicateDriver, actions });
  assert.equal(duplicate.passed, false, "a duplicated paper-index report entry must not pass, even though the date completed");
});

test("cancel-then-rerun: verify re-polled twice does not release the held request a second time", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "cancel-then-rerun");
  const date = obsidianTaskLibrary.CANCEL_THEN_RERUN_DATE;
  const driver = createFakeRunDriver({
    state: { runState: { [date]: { status: "completed" } } },
    index: indexWithReports(date),
    historyLines: [{ date, resultKind: "cancelled" }],
  });
  const actions = [fillDate(date), clickNamed("Cancel active tasks"), fillDate(date)];
  await task.verify({ driver, actions });
  await task.verify({ driver, actions });
  assert.equal(driver.calls.releaseHeld, 1, "releasing the held request must be memoized across repeated verify() polls");
});

test("auth-failure-retry: verify cannot pass without the model reviewing the Run History tab; the fixture is only restored to normal once that review happened", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "auth-failure-retry");
  assert.ok(task, "auth-failure-retry must exist in the Obsidian goal task library");
  const date = obsidianTaskLibrary.AUTH_FAILURE_RETRY_DATE;

  const untouchedDriver = createFakeRunDriver({});
  const untouched = await task.verify({ driver: untouchedDriver, actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model submits anything");

  const noFailureDriver = createFakeRunDriver({});
  const noFailure = await task.verify({ driver: noFailureDriver, actions: [fillDate(date)] });
  assert.equal(noFailure.passed, false, "verify must not pass until the authentication failure is actually recorded on disk");

  const failedDriver = createFakeRunDriver({ state: { runState: { [date]: { status: "failed_permanent" } } } });
  const noReview = await task.verify({ driver: failedDriver, actions: [fillDate(date)] });
  assert.equal(noReview.passed, false, "verify must not pass without the model reviewing the Run History tab (a plausible wrong outcome: retrying blindly)");
  assert.equal(failedDriver.calls.setMode.filter(mode => mode.llm === "normal").length, 0, "the fixture must stay unauthorized until the model actually reviews the failure");

  const reviewedNoForceDriver = createFakeRunDriver({ state: { runState: { [date]: { status: "failed_permanent" } } } });
  const reviewedNoForce = await task.verify({ driver: reviewedNoForceDriver, actions: [fillDate(date), clickNamed("Run History")] });
  assert.equal(reviewedNoForce.passed, false, "verify must not pass before Force run for date is actually used to retry");
  assert.equal(reviewedNoForceDriver.calls.setMode.filter(mode => mode.llm === "normal").length, 1, "reviewing the failure restores the fixture exactly once, enabling (but not substituting for) the retry");

  // The full happy path is exercised across two verify() calls, matching how the real exploration loop
  // actually observes it: the first call sees the failure while it is still the current run-state (and
  // must memoize that fact — see production code's `authFailureSeen`), and only the second call, after
  // the model's force-retry, sees the current run-state having moved on to "completed" (overwriting
  // "failed_permanent"). A single-call test with state already "completed" from the start would not
  // actually exercise that memoization, and would pass for the wrong reason.
  // Fresh task instance (fresh memoization closures): the scenarios above already exercised and
  // consumed the shared "fixture restored" memoization on their own `task`, so the full happy path gets
  // its own task/closures here, exactly as a real single task run would start from scratch.
  const freshTask = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "auth-failure-retry");
  const doneDriver = createFakeRunDriver({ state: { runState: { [date]: { status: "failed_permanent" } } } });
  const firstActions = [fillDate(date), clickNamed("Run History")];
  const mid = await freshTask.verify({ driver: doneDriver, actions: firstActions });
  assert.equal(mid.passed, false, "not yet passed: Force run has not happened");
  assert.deepEqual(doneDriver.calls.setMode, [{ llm: "normal" }], "reviewing the failure restored the fixture");

  doneDriver.disk.state.runState[date] = { status: "completed" };
  doneDriver.disk.historyLines.push({ date, event: "completed", trigger: "force" });
  doneDriver.disk.reports[date] = reportMarkdown(date);
  const actions = [...firstActions, clickNamed("Force run for date…"), fillDate(date)];
  const done = await freshTask.verify({ driver: doneDriver, actions });
  assert.equal(done.passed, true, `verify should pass once the model reviewed the failure and force-retried to completion: ${JSON.stringify(done)}`);
});

test("auth-failure-retry: prepare() clears the date, switches the fixture to unauthorized, and opts the driver into the authorization-failure diagnostics carve-out", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "auth-failure-retry");
  const date = obsidianTaskLibrary.AUTH_FAILURE_RETRY_DATE;
  const driver = createFakeRunDriver({ state: { runState: { [date]: { status: "completed" } } }, reports: { [date]: "stale" } });
  await task.prepare({ driver });
  assert.deepEqual(driver.calls.clearDateState, [date]);
  assert.deepEqual(driver.calls.setMode, [{ llm: "unauthorized" }]);
  assert.equal(driver.calls.expectAuthorizationFailure, 1);
  assert.equal(driver.disk.state.runState[date], undefined, "clearDateState must actually clear the stale completed state");
});

test("completed-date-idempotent: verify fails on the untouched state, fails on a force-run-shaped outcome (new request, changed content, no skip record), fails on a duplicated index entry, and passes on a true skip", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "completed-date-idempotent");
  assert.ok(task, "completed-date-idempotent must exist in the Obsidian goal task library");
  const date = ACCEPTANCE_DATE;
  const originalContent = reportMarkdown(date);

  const driver = createFakeRunDriver({ reports: { [date]: originalContent } });
  await task.prepare({ driver });

  const untouched = await task.verify({ driver, actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model submits anything");

  const forceRunShaped = createFakeRunDriver({ reports: { [date]: originalContent } });
  await task.prepare({ driver: forceRunShaped });
  forceRunShaped.disk.reports[date] = `${originalContent}\nregenerated`;
  forceRunShaped.requests.push({ model: "fixture-model-2" });
  const forceOutcome = await task.verify({ driver: forceRunShaped, actions: [fillDate(date)] });
  assert.equal(forceOutcome.passed, false, "verify must not pass when the outcome looks like a force run (content changed, a new model request) instead of a skip");

  const duplicateDriver = createFakeRunDriver({ reports: { [date]: originalContent }, index: { papers: { p0: { arxivId: PAPERS[0].id, dailyReports: [dailyReportFile(date), dailyReportFile(date)] } } } });
  await task.prepare({ driver: duplicateDriver });
  duplicateDriver.disk.historyLines.push({ date, event: "skipped", resultKind: "skipped", reason: "already done" });
  const duplicateOutcome = await task.verify({ driver: duplicateDriver, actions: [fillDate(date)] });
  assert.equal(duplicateOutcome.passed, false, "verify must not pass when the paper index has a duplicated report entry for the date, even with a skip recorded");

  const skippedDriver = createFakeRunDriver({ reports: { [date]: originalContent }, index: indexWithReports(date) });
  await task.prepare({ driver: skippedDriver });
  skippedDriver.disk.historyLines.push({ date, event: "skipped", resultKind: "skipped", reason: "already done" });
  const skipped = await task.verify({ driver: skippedDriver, actions: [fillDate(date)] });
  assert.equal(skipped.passed, true, `verify should pass once a genuine "already done" skip is recorded and nothing else changed: ${JSON.stringify(skipped)}`);
});

// ---------------------------------------------------------------------------
// P3 Chunk 4: the five new Obsidian reading goal tasks (dashboard-tabs-paging,
// mark-to-read, create-paper-note, summarize-by-id, open-today-report). A
// fake driver stands in for the real CDP session/vault, exposing exactly the
// surface these tasks' prepare()/verify() use (jsonFile/file/read/
// applyIndexPatch/removeVaultFile/writeVaultFile/ensureDailyReport/
// openDashboard) — never a real Obsidian/CDP session. Each task is checked
// on the untouched state (must not pass), on at least one plausible wrong
// outcome (must not pass), and on the fully correct state (must pass).
// ---------------------------------------------------------------------------

function createFakeReadingDriver({ index = {}, notes = {}, reports = {}, dom = {} } = {}) {
  const disk = {
    index: structuredClone({ papers: {}, ...index }),
    notes: structuredClone(notes),
    reports: structuredClone(reports),
    dom: { activeTab: "false", pageLabel: "", tableText: "", activeFile: null, ...dom },
  };
  const calls = { applyIndexPatch: 0, removeVaultFile: [], writeVaultFile: [], ensureDailyReport: [], reloadPlugin: 0 };
  const notFound = () => { const error = new Error("not found"); error.code = "ENOENT"; throw error; };
  return {
    disk, calls,
    async jsonFile(relative) {
      if (relative === INDEX_FILE) return structuredClone(disk.index);
      throw new Error(`Unhandled fake jsonFile: ${relative}`);
    },
    async file(relative) {
      if (relative in disk.notes) return disk.notes[relative];
      const date = Object.keys(disk.reports).find(d => relative === dailyReportFile(d));
      if (date) return disk.reports[date];
      return notFound();
    },
    async read(expression) {
      if (expression.includes('data-tab="all"')) return disk.dom.activeTab;
      if (expression.includes("pagination-label")) return disk.dom.pageLabel;
      if (expression.includes("__table")) return disk.dom.tableText;
      if (expression.includes("getActiveFile")) return disk.dom.activeFile;
      throw new Error(`Unhandled fake read: ${expression}`);
    },
    async openDashboard() {},
    async reloadPlugin() { calls.reloadPlugin += 1; },
    async ensureDailyReport(date) { calls.ensureDailyReport.push(date); disk.reports[date] ??= reportMarkdown(date); },
    async applyIndexPatch(mutate) { calls.applyIndexPatch += 1; mutate(disk.index); },
    async removeVaultFile(relative) { calls.removeVaultFile.push(relative); delete disk.notes[relative]; },
    async writeVaultFile(relative, content) { calls.writeVaultFile.push(relative); disk.notes[relative] = content; },
  };
}

function fillNamed(name, value) { return { type: "fill", control: { name }, value }; }

function detailNoteMarkdown(arxivId) {
  const sections = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "适用边界"];
  return `# Paper ${arxivId}\n\n- **arXiv**: https://arxiv.org/abs/${arxivId}\n\n${sections.map(s => `## ${s}\n\nFixture body for ${arxivId}.\n`).join("\n")}`;
}

test("dashboard-tabs-paging: verify fails on the untouched state, fails without reaching page 2, and passes once All/page-2 state is confirmed on the live DOM", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "dashboard-tabs-paging");
  assert.ok(task, "dashboard-tabs-paging must exist in the Obsidian goal task library");
  const first = obsidianTaskLibrary.PAGINATION_FIRST_PAPER_ID;
  const last = obsidianTaskLibrary.PAGINATION_LAST_PAPER_ID;

  const untouched = await task.verify({ driver: createFakeReadingDriver({}), actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model switches tabs");

  const noNext = await task.verify({ driver: createFakeReadingDriver({}), actions: [clickNamed("All32")] });
  assert.equal(noNext.passed, false, "verify must not pass without a Next page click (a plausible wrong outcome: staying on page 1)");

  const stalePageDriver = createFakeReadingDriver({ dom: { activeTab: "true", pageLabel: "Page 1 / 2", tableText: `${first} ${last}` } });
  const stalePage = await task.verify({ driver: stalePageDriver, actions: [clickNamed("All32"), clickNamed("Next page")] });
  assert.equal(stalePage.passed, false, "verify must not pass when the live DOM still reports page 1 regardless of the recorded click (a plausible wrong outcome: the click had no effect)");

  // Isolated from the pageLabel/activeTab checks above: tab and page indicator both already correct, but
  // the visible table still shows the page-1-only paper (and lacks the page-2-only one) — this must fail
  // on its own, not only in combination with a wrong tab/page.
  const wrongPapersDriver = createFakeReadingDriver({ dom: { activeTab: "true", pageLabel: "Page 2 / 2", tableText: first } });
  const wrongPapers = await task.verify({ driver: wrongPapersDriver, actions: [clickNamed("All32"), clickNamed("Next page")] });
  assert.equal(wrongPapers.passed, false, "verify must not pass when the tab/page indicator are correct but the table still shows the first page's paper instead of the second page's (a plausible wrong outcome: the page indicator advanced without the rows actually changing)");

  const doneDriver = createFakeReadingDriver({ dom: { activeTab: "true", pageLabel: "Page 2 / 2", tableText: last } });
  const done = await task.verify({ driver: doneDriver, actions: [clickNamed("All32"), clickNamed("Next page")] });
  assert.equal(done.passed, true, `verify should pass once the live DOM shows the All tab active, page 2, and the page-2-only paper: ${JSON.stringify(done)}`);
});

test("dashboard-tabs-paging: prepare() writes a real daily-report Markdown file (with an arxiv.org/abs link per synthetic paper) plus one real Paper note per synthetic id, then reloads the plugin so the vault scan picks them up", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "dashboard-tabs-paging");
  const driver = createFakeReadingDriver({});
  await task.prepare({ driver });
  assert.equal(driver.calls.reloadPlugin, 1, "a direct papers.json write alone does not survive a reload (confirmed on a real host); the fix must reload after writing real files");
  const reportPath = dailyReportFile(obsidianTaskLibrary.PAGINATION_REPORT_DATE);
  const reportContent = driver.disk.notes[reportPath];
  assert.ok(reportContent, "prepare() must write a real daily-report file the vault scan can discover");
  assert.match(reportContent, new RegExp(`arxiv\\.org/abs/${obsidianTaskLibrary.PAGINATION_FIRST_PAPER_ID}`));
  assert.match(reportContent, new RegExp(`arxiv\\.org/abs/${obsidianTaskLibrary.PAGINATION_LAST_PAPER_ID}`));
  assert.equal((reportContent.match(/arxiv\.org\/abs\//g) || []).length, obsidianTaskLibrary.PAGINATION_PAPER_COUNT, "the report must link every synthetic paper, matching history-sync.ts's extractArxivIds pattern");
  assert.ok(driver.disk.notes[paperNoteFile(obsidianTaskLibrary.PAGINATION_FIRST_PAPER_ID)], "each synthetic paper also gets a real Paper note, so its row never ambiguously shows 'Summarize by arXiv ID'");
  assert.ok(driver.disk.notes[paperNoteFile(obsidianTaskLibrary.PAGINATION_LAST_PAPER_ID)]);
});

test("mark-to-read: verify fails on the untouched state, fails when the submit happened but the on-disk status never changed, and passes once the target paper's status is to_read", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "mark-to-read");
  assert.ok(task, "mark-to-read must exist in the Obsidian goal task library");
  const id = obsidianTaskLibrary.MARK_TO_READ_ID;

  const untouchedIndex = { papers: { p0: { arxivId: id, status: "inbox" } } };
  const untouched = await task.verify({ driver: createFakeReadingDriver({ index: untouchedIndex }), actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model fills the Paper ID field");

  const noDiskChangeDriver = createFakeReadingDriver({ index: { papers: { p0: { arxivId: id, status: "inbox" } } } });
  const noDiskChange = await task.verify({ driver: noDiskChangeDriver, actions: [fillNamed("Paper ID or URL", id), clickNamed("Set mark")] });
  assert.equal(noDiskChange.passed, false, "verify must not pass when the fill/click happened but the on-disk status never actually changed (a plausible wrong outcome: the submit failed silently)");

  const doneDriver = createFakeReadingDriver({ index: { papers: { p0: { arxivId: id, status: "to_read" } } } });
  const done = await task.verify({ driver: doneDriver, actions: [fillNamed("Paper ID or URL", id), clickNamed("Set mark")] });
  assert.equal(done.passed, true, `verify should pass once the target paper's status is to_read on disk: ${JSON.stringify(done)}`);
});

test("mark-to-read: prepare() ensures the daily report exists so the target paper is indexed", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "mark-to-read");
  const driver = createFakeReadingDriver({});
  await task.prepare({ driver });
  assert.deepEqual(driver.calls.ensureDailyReport, [ACCEPTANCE_DATE]);
});

test("create-paper-note: verify fails on the untouched state, fails when the click happened but no note file was ever written, and passes once the note/index/active-file all agree", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "create-paper-note");
  assert.ok(task, "create-paper-note must exist in the Obsidian goal task library");
  const id = obsidianTaskLibrary.CREATE_PAPER_NOTE_ID;
  const notePath = paperNoteFile(id);

  const untouched = await task.verify({ driver: createFakeReadingDriver({}), actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model clicks the row action");

  const noFileDriver = createFakeReadingDriver({ index: { papers: { p0: { arxivId: id, detail: false, paperPath: null } } } });
  const noFile = await task.verify({ driver: noFileDriver, actions: [clickNamed("Summarize by arXiv ID")] });
  assert.equal(noFile.passed, false, "verify must not pass when the click happened but no note file was ever written (a plausible wrong outcome: the summarize call failed)");

  // Isolated from the index check below: the note file exists with the right structure, but the index
  // entry was never updated — must fail on its own.
  const indexNotUpdatedDriver = createFakeReadingDriver({
    index: { papers: { p0: { arxivId: id, detail: false, paperPath: null } } },
    notes: { [notePath]: detailNoteMarkdown(id) },
    dom: { activeFile: notePath },
  });
  const indexNotUpdated = await task.verify({ driver: indexNotUpdatedDriver, actions: [clickNamed("Summarize by arXiv ID")] });
  assert.equal(indexNotUpdated.passed, false, "verify must not pass when the note file was written but the paper index was never updated to point at it (a plausible wrong outcome: the write succeeded but the index commit failed)");

  // Isolated from the structure check above: the index already claims a note exists, but the file itself
  // was never actually written — must fail on its own.
  const contentMissingDriver = createFakeReadingDriver({
    index: { papers: { p0: { arxivId: id, detail: true, paperPath: notePath } } },
    dom: { activeFile: notePath },
  });
  const contentMissing = await task.verify({ driver: contentMissingDriver, actions: [clickNamed("Summarize by arXiv ID")] });
  assert.equal(contentMissing.passed, false, "verify must not pass when the index claims a note exists but the file was never actually written (a plausible wrong outcome: the index was updated without the write succeeding)");

  const doneDriver = createFakeReadingDriver({
    index: { papers: { p0: { arxivId: id, detail: true, paperPath: notePath } } },
    notes: { [notePath]: detailNoteMarkdown(id) },
    dom: { activeFile: notePath },
  });
  const done = await task.verify({ driver: doneDriver, actions: [clickNamed("Summarize by arXiv ID")] });
  assert.equal(done.passed, true, `verify should pass once the note file, index entry and active file all agree: ${JSON.stringify(done)}`);
});

test("create-paper-note: prepare() clears the target's own note and seeds the fixture's other paper with one, so exactly one row can show the action", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "create-paper-note");
  const targetId = obsidianTaskLibrary.CREATE_PAPER_NOTE_ID;
  const otherId = obsidianTaskLibrary.CREATE_PAPER_NOTE_OTHER_ID;
  const driver = createFakeReadingDriver({
    notes: { [paperNoteFile(targetId)]: "stale note" },
    index: {
      papers: {
        p0: { arxivId: targetId, detail: true, paperPath: paperNoteFile(targetId) },
        p1: { arxivId: otherId, detail: false, paperPath: null },
      },
    },
  });
  await task.prepare({ driver });
  assert.deepEqual(driver.calls.removeVaultFile, [paperNoteFile(targetId)]);
  assert.ok(driver.calls.writeVaultFile.includes(paperNoteFile(otherId)));
  const targetEntry = Object.values(driver.disk.index.papers).find(e => e.arxivId === targetId);
  const otherEntry = Object.values(driver.disk.index.papers).find(e => e.arxivId === otherId);
  assert.equal(targetEntry.detail, false);
  assert.equal(targetEntry.paperPath, null);
  assert.equal(otherEntry.detail, true);
  assert.equal(otherEntry.paperPath, paperNoteFile(otherId));
});

test("summarize-by-id: verify fails on the untouched state, fails without a Summarize click, fails when submitted but no note file was ever written, and passes once the note/index agree", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "summarize-by-id");
  assert.ok(task, "summarize-by-id must exist in the Obsidian goal task library");
  const id = obsidianTaskLibrary.SUMMARIZE_BY_ID_TARGET;
  const notePath = paperNoteFile(id);

  const untouched = await task.verify({ driver: createFakeReadingDriver({}), actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model fills the Paper ID field");

  const noSubmit = await task.verify({ driver: createFakeReadingDriver({}), actions: [fillNamed("Paper ID or URL", id)] });
  assert.equal(noSubmit.passed, false, "verify must not pass without clicking Summarize (a plausible wrong outcome: filled but never submitted)");

  const noFileDriver = createFakeReadingDriver({});
  const noFile = await task.verify({ driver: noFileDriver, actions: [fillNamed("Paper ID or URL", id), clickNamed("Summarize")] });
  assert.equal(noFile.passed, false, "verify must not pass when submit happened but no note file was ever written (a plausible wrong outcome: the fetch failed)");

  // Isolated from the index check below: the note file exists with the right structure, but the index
  // entry was never updated — must fail on its own.
  const indexNotUpdated = await task.verify({
    driver: createFakeReadingDriver({ notes: { [notePath]: detailNoteMarkdown(id) } }),
    actions: [fillNamed("Paper ID or URL", id), clickNamed("Summarize")],
  });
  assert.equal(indexNotUpdated.passed, false, "verify must not pass when the note file was written but the paper index was never updated (a plausible wrong outcome: the write succeeded but the index commit failed)");

  // Isolated from the structure check above: the index already claims a note exists, but the file itself
  // was never actually written — must fail on its own.
  const contentMissing = await task.verify({
    driver: createFakeReadingDriver({ index: { papers: { p0: { arxivId: id, detail: true, paperPath: notePath } } } } ),
    actions: [fillNamed("Paper ID or URL", id), clickNamed("Summarize")],
  });
  assert.equal(contentMissing.passed, false, "verify must not pass when the index claims a note exists but the file was never actually written (a plausible wrong outcome: the index was updated without the write succeeding)");

  const doneDriver = createFakeReadingDriver({
    index: { papers: { p0: { arxivId: id, detail: true, paperPath: notePath } } },
    notes: { [notePath]: detailNoteMarkdown(id) },
  });
  const done = await task.verify({ driver: doneDriver, actions: [fillNamed("Paper ID or URL", id), clickNamed("Summarize")] });
  assert.equal(done.passed, true, `verify should pass once the note file and index entry agree: ${JSON.stringify(done)}`);
});

test("summarize-by-id: prepare() clears any stale note/index entry for the target id", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "summarize-by-id");
  const id = obsidianTaskLibrary.SUMMARIZE_BY_ID_TARGET;
  const driver = createFakeReadingDriver({
    notes: { [paperNoteFile(id)]: "stale" },
    index: { papers: { [`arxiv:${id}`]: { arxivId: id, detail: true, paperPath: paperNoteFile(id) } } },
  });
  await task.prepare({ driver });
  assert.deepEqual(driver.calls.removeVaultFile, [paperNoteFile(id)]);
  assert.equal(driver.disk.index.papers[`arxiv:${id}`], undefined);
});

test("open-today-report: verify fails on the untouched state, fails when a different note ends up active, and passes once ACCEPTANCE_DATE's report is the active file", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "open-today-report");
  assert.ok(task, "open-today-report must exist in the Obsidian goal task library");
  const label = `${ACCEPTANCE_DATE}: open daily report, 2 indexed papers`;

  const untouched = await task.verify({ driver: createFakeReadingDriver({}), actions: [] });
  assert.equal(untouched.passed, false, "verify must not pass before the model clicks the calendar cell");

  const wrongFileDriver = createFakeReadingDriver({ dom: { activeFile: "arxiv-daily/daily/2026-10-02.md" } });
  const wrongFile = await task.verify({ driver: wrongFileDriver, actions: [clickNamed(label)] });
  assert.equal(wrongFile.passed, false, "verify must not pass when a different date's report ended up active (a plausible wrong outcome: the wrong calendar cell)");

  const doneDriver = createFakeReadingDriver({ dom: { activeFile: dailyReportFile(ACCEPTANCE_DATE) } });
  const done = await task.verify({ driver: doneDriver, actions: [clickNamed(label)] });
  assert.equal(done.passed, true, `verify should pass once the active file is ACCEPTANCE_DATE's report: ${JSON.stringify(done)}`);
});

test("open-today-report: prepare() ensures the ACCEPTANCE_DATE report exists", async () => {
  const task = (await obsidianTaskLibrary.goalTasks()).find(t => t.id === "open-today-report");
  const driver = createFakeReadingDriver({});
  await task.prepare({ driver });
  assert.deepEqual(driver.calls.ensureDailyReport, [ACCEPTANCE_DATE]);
});
