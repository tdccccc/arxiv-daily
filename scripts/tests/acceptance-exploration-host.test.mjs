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
  SETTINGS_FILE,
  startObsidianExplorationSession,
} from "../acceptance/exploration-obsidian.mjs";
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
  // Chunk 2 appends topic/direction/category tasks after the original two, keeping their ids stable.
  assert.deepEqual(obsidianTaskLibrary.GOAL_TASK_IDS, [
    "daily-paper-limit", "star-paper-from-report",
    "add-topic-direction", "remove-topic-direction", "add-arxiv-category",
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
  assert.deepEqual(workbenchTaskLibrary.GOAL_TASK_IDS, ["settings-persistence", "sidebar-persistence", "generate-and-read"]);
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

test("neither host has any personas yet, but personaTasks() returns a usable (empty) array, not undefined", () => {
  assert.deepEqual(obsidianTaskLibrary.personaTasks(), []);
  assert.deepEqual(workbenchTaskLibrary.personaTasks(), []);
  assert.deepEqual(obsidianTaskLibrary.PERSONA_TASK_IDS, []);
  assert.deepEqual(workbenchTaskLibrary.PERSONA_TASK_IDS, []);
});

test("--list (describeAcceptanceSuites) lists both hosts' goal task ids and personas", async () => {
  const output = await describeAcceptanceSuites();
  assert.match(output, /obsidian/);
  assert.match(output, /workbench/);
  for (const id of obsidianTaskLibrary.GOAL_TASK_IDS) assert.ok(output.includes(id), `--list must mention Obsidian task id ${id}`);
  for (const id of workbenchTaskLibrary.GOAL_TASK_IDS) assert.ok(output.includes(id), `--list must mention workbench task id ${id}`);
  assert.match(output, /none yet/, "no personas exist yet for either host");
});

// ---------------------------------------------------------------------------
// createObsidianExplorationDriver.reset(): call ordering against a fake CDP
// session (happy-dom + a recording fake client, same convention as
// acceptance-exploration-obsidian.test.mjs's buildSession), and failure ->
// throws when the measured duration exceeds the 30s phase-abort budget.
// ---------------------------------------------------------------------------

function buildFakeObsidianSession({ modalOpen = false, settingsMarker = false } = {}) {
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
  return { window, calls, reloadCalls, session: { client, evaluate, diagnostics: { entries: () => [] } } };
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
