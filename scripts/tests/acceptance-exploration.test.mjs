import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";
import { Window } from "happy-dom";
import { classifyControlRestriction, createExplorationDriver } from "../acceptance/exploration-browser.mjs";
import { goalTasks as createDefaultExplorationTasks } from "../acceptance/exploration-tasks/workbench.mjs";
import { collapseOptionSets, loadExplorationModel, parseExplorationAction, runExplorationLoop, selectExplorationTasks } from "../acceptance/exploration.mjs";
import { ACCEPTANCE_DATE, PAPERS } from "../acceptance/fixtures.mjs";

const observation = (version = 1) => ({
  id: `snapshot-${version}`, text: "Settings", url: "/", controls: [
    { ref: `s${version}-1`, tag: "button", role: "button", name: "Settings", disabled: false },
    { ref: `s${version}-2`, tag: "input", type: "number", name: "maxDailyPapers", value: "3", disabled: false },
    { ref: `s${version}-3`, tag: "input", type: "password", name: "apiKey", value: "", sensitive: true, restricted: "secret" },
    { ref: `s${version}-4`, tag: "input", type: "text", name: "vaultRoot", value: "/temporary", restricted: "filesystem path" },
  ],
});

function fixtureDriver() {
  let version = 0;
  const executed = [];
  return {
    executed,
    observe: async () => observation(++version),
    execute: async action => { executed.push(action); return { ok: true }; },
    capture: async () => undefined,
  };
}

function task(verify = () => ({ passed: false, checks: [{ label: "saved after reload", passed: false }] })) {
  return { id: "settings", title: "Save settings", goal: "Save, reload and inspect", verify };
}

function persona(overrides = {}) {
  return { id: "new-user", title: "第一次使用的新用户", kind: "persona", goal: "扮演第一次使用的新用户，自由探索当前页面并记录可疑交互", ...overrides };
}

/** A model shaped like the real HTTP adapter: `.next()` plus cumulative `.metrics()` usage. */
function usageModel(reply, { apiCallsPerStep = 1, inputTokensPerStep = 0, outputTokensPerStep = 0 } = {}) {
  const usage = { apiCalls: 0, inputTokens: 0, outputTokens: 0, totalTokens: 0 };
  return {
    async next(request) {
      usage.apiCalls += apiCallsPerStep;
      usage.inputTokens += inputTokensPerStep;
      usage.outputTokens += outputTokensPerStep;
      usage.totalTokens += inputTokensPerStep + outputTokensPerStep;
      return typeof reply === "function" ? reply(request) : reply;
    },
    metrics: () => ({ ...usage }),
  };
}

async function temporary(t) {
  const root = await mkdtemp(path.join(tmpdir(), "acceptance-exploration-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test("an unsupported done claim is blocked but later tasks still run (rewritten: this used to assert the whole run stopped after the first non-passed task; the new contract runs every task regardless, subject only to run-wide budgets)", async t => {
  const driver = fixtureDriver();
  const result = await runExplorationLoop({ driver, model: async () => '{"type":"done","reason":"Everything works"}',
    tasks: [task(), { ...task(), id: "reading" }], artifactDir: await temporary(t), maxSteps: 5 });
  assert.equal(result.status, "blocked");
  assert.deepEqual(result.scenarios.map(value => value.status), ["blocked", "blocked"]);
  assert.match(result.scenarios[0].error, /evidence|判据/i);
  assert.match(result.scenarios[1].error, /evidence|判据/i);
  assert.equal(driver.executed.length, 0);
});

test("a failed task does not stop a later passing task", async t => {
  const driver = fixtureDriver();
  // The first observe() call reports a product error so task "settings" fails immediately
  // without ever calling the model; subsequent observe() calls are clean for task "reading".
  let observeCalls = 0;
  driver.observe = async () => { observeCalls += 1; return observeCalls === 1 ? { ...observation(observeCalls), errors: ["Uncaught rendering error"] } : observation(observeCalls); };
  const result = await runExplorationLoop({ driver, model: async request => JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [task(), { ...task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "observed saved state", passed: actions.length === 1 }] })), id: "reading" }],
    artifactDir: await temporary(t), maxSteps: 5 });
  assert.deepEqual(result.scenarios.map(value => value.status), ["failed", "passed"]);
  assert.match(result.scenarios[0].error, /rendering/);
  assert.equal(result.scenarios[1].assertions[0].passed, true);
});

test("a blocked task (invalid actions exhausted) does not stop a later passing task", async t => {
  const driver = fixtureDriver();
  let calls = 0;
  const result = await runExplorationLoop({ driver, model: async request => {
    calls += 1;
    return calls <= 3 ? "not json" : JSON.stringify({ type: "click", ref: request.observation.controls[0].ref });
  }, tasks: [task(), { ...task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "observed saved state", passed: actions.length === 1 }] })), id: "reading" }],
    artifactDir: await temporary(t), maxSteps: 10, maxInvalidActions: 3 });
  assert.deepEqual(result.scenarios.map(value => value.status), ["blocked", "passed"]);
  assert.equal(driver.executed.length, 1);
});

test("the host is reset before every task, including the first", async t => {
  const driver = fixtureDriver();
  driver.reset = async () => { driver.resets = (driver.resets ?? 0) + 1; };
  const result = await runExplorationLoop({ driver, model: async request => JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] })),
      { ...task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] })), id: "reading" }],
    artifactDir: await temporary(t), maxSteps: 5 });
  assert.deepEqual(result.scenarios.map(value => value.status), ["passed", "passed"]);
  assert.equal(driver.resets, 2);
});

test("a throwing reset blocks only that task; the loop tries reset again for the next task", async t => {
  const driver = fixtureDriver();
  let attempt = 0;
  driver.reset = async () => { attempt += 1; if (attempt === 1) throw new Error("dialog would not close"); };
  const result = await runExplorationLoop({ driver, model: async request => JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] })),
      { ...task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] })), id: "reading" }],
    artifactDir: await temporary(t), maxSteps: 5 });
  assert.deepEqual(result.scenarios.map(value => value.status), ["blocked", "passed"]);
  assert.match(result.scenarios[0].error, /dialog would not close/);
  assert.equal(attempt, 2);
});

// Rewritten for chunk 3: a task's own step budget is always clamped to the run's own, so when the
// run budget is this tiny every task's own cap equals it too. The first task now exhausts its own
// (non-fatal) cap at exactly the run limit; the run-wide cap itself only becomes the active, fatal
// blocker once a next task is attempted with no run budget left, which still leaves later tasks not-run.
test("run-wide step budget exhaustion is fatal and leaves a task after it not-run, even once a per-task cap absorbs the first task", async t => {
  const driver = fixtureDriver();
  const result = await runExplorationLoop({ driver, model: async () => '{"type":"wait","milliseconds":1}',
    tasks: [task(), { ...task(), id: "reading" }, { ...task(), id: "third" }], artifactDir: await temporary(t), maxSteps: 2 });
  assert.equal(result.scenarios[0].status, "blocked");
  assert.match(result.scenarios[0].error, /task budget exhausted/i);
  assert.equal(result.scenarios[1].status, "blocked");
  assert.match(result.scenarios[1].error, /step|步数/i);
  assert.equal(result.scenarios[2].status, "not-run");
});

test("a task's own step budget never exceeds the run-wide one even when it asks for more", async t => {
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async () => '{"type":"wait","milliseconds":1}',
    tasks: [{ ...task(), maxSteps: 50 }], artifactDir: await temporary(t), maxSteps: 3 });
  assert.equal(result.scenarios[0].status, "blocked");
  assert.equal(result.scenarios[0].usage.modelCalls, 3);
});

test("parent cancellation ends the run and leaves later tasks not-run", async t => {
  const controller = new AbortController(); controller.abort("owner cancelled");
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async () => '{"type":"done"}',
    tasks: [task(), { ...task(), id: "reading" }], artifactDir: await temporary(t), signal: controller.signal });
  assert.equal(result.scenarios[0].status, "blocked");
  assert.equal(result.scenarios[1].status, "not-run");
});

test("only program checks after an actual action can pass a task", async t => {
  const driver = fixtureDriver();
  const requests = [];
  const result = await runExplorationLoop({ driver, model: async request => {
    requests.push(request);
    return JSON.stringify({ type: "click", ref: request.observation.controls[0].ref });
  }, tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "observed saved state", passed: actions.length === 1 }] }))],
    artifactDir: await temporary(t), maxSteps: 5 });
  assert.equal(result.status, "passed");
  assert.equal(result.scenarios[0].assertions[0].passed, true);
  assert.equal(driver.executed.length, 1);
  assert.equal(requests.length, 1);
});

test("a vacuous verifier cannot produce a passing result", async t => {
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async request => JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [task(() => ({ passed: true, checks: [] }))], artifactDir: await temporary(t), maxSteps: 1 });
  assert.equal(result.status, "blocked");
  assert.match(result.scenarios[0].error, /budget|预算|判据/i);
});

test("invalid actions and stale refs never reach the browser and stop after bounded recovery", async t => {
  const driver = fixtureDriver();
  const replies = ["not json", '{"type":"click","ref":"s0-1"}', '{"type":"shell","command":"touch /tmp/no"}'];
  const result = await runExplorationLoop({ driver, model: async () => replies.shift(), tasks: [task()],
    artifactDir: await temporary(t), maxSteps: 10, maxInvalidActions: 3 });
  assert.equal(result.status, "blocked");
  assert.equal(driver.executed.length, 0);
  assert.equal(result.metrics.modelCalls, 3);
  assert.equal(result.scenarios[0].steps.length, 3);
});

test("step exhaustion records incomplete work and never passes", async t => {
  const driver = fixtureDriver();
  const result = await runExplorationLoop({ driver, model: async () => '{"type":"wait","milliseconds":1}', tasks: [task()],
    artifactDir: await temporary(t), maxSteps: 2 });
  assert.equal(result.status, "blocked");
  assert.match(result.scenarios[0].error, /step|步数/i);
  assert.equal(result.metrics.modelCalls, 2);
});

test("a model that never resolves is aborted within the call deadline", async t => {
  let receivedSignal;
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async request => {
    receivedSignal = request.signal;
    return new Promise(() => {});
  }, tasks: [task()], artifactDir: await temporary(t), maxDurationMs: 200, modelCallTimeoutMs: 20 });
  assert.equal(result.status, "blocked");
  assert.equal(receivedSignal.aborted, true);
  assert.match(result.scenarios[0].error, /timeout|超时/i);
});

test("API failures are blockers while independently observed product failures are failed", async t => {
  const artifactDir = await temporary(t);
  const unavailable = await runExplorationLoop({ driver: fixtureDriver(), model: async () => { throw new Error("API unavailable"); }, tasks: [task()], artifactDir });
  assert.equal(unavailable.status, "blocked");
  const driver = fixtureDriver();
  driver.observe = async () => ({ ...observation(), errors: ["Uncaught rendering error"] });
  const broken = await runExplorationLoop({ driver, model: async () => '{"type":"done"}', tasks: [task()], artifactDir });
  assert.equal(broken.status, "failed");
  assert.match(broken.scenarios[0].error, /rendering/);
});

test("findings keep model suspicions separate from verified failures and preserve evidence", async t => {
  const artifactDir = await temporary(t);
  let calls = 0;
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async request => {
    calls += 1;
    return calls === 1 ? '{"type":"finding","summary":"The button label may be unclear","evidence":"Settings"}'
      : JSON.stringify({ type: "click", ref: request.observation.controls[0].ref });
  }, tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "Saved state", passed: actions.length === 1 }] }))], artifactDir });
  assert.equal(result.status, "passed");
  assert.equal(result.findings[0].status, "needs-review");
  assert.equal(result.findings[0].step, 1);
  assert.ok(result.artifacts.some(value => value.endsWith("exploration-events.jsonl")));
  const records = (await readFile(path.join(artifactDir, "exploration-events.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
  assert.equal(records[0].observation.id, "snapshot-1");
  assert.equal(records[0].action.type, "finding");
});

test("P3 Chunk 5: a finding carries the same step's own screenshot, not a later or earlier step's", async t => {
  const artifactDir = await temporary(t);
  let version = 0, calls = 0;
  const driver = {
    observe: async () => observation(++version),
    execute: async action => ({ ok: true, action }),
    capture: async name => `/artifacts/${name}.png`,
  };
  const result = await runExplorationLoop({ driver, model: async request => {
    calls += 1;
    return calls === 1 ? '{"type":"finding","summary":"Odd state","evidence":"Settings"}'
      : JSON.stringify({ type: "click", ref: request.observation.controls[0].ref });
  }, tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "Saved state", passed: actions.length === 1 }] }))], artifactDir });
  assert.equal(result.findings[0].step, 1);
  assert.equal(result.findings[0].screenshot, "/artifacts/settings-step-1.png");
  assert.ok(result.scenarios[0].artifacts.includes("/artifacts/settings-step-1.png"));
});

test("a finding keeps no screenshot field at all when the driver never captures one", async t => {
  const artifactDir = await temporary(t);
  let calls = 0;
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async () => {
    calls += 1;
    return calls === 1 ? '{"type":"finding","summary":"Odd state","evidence":"Settings"}' : '{"type":"done"}';
  }, tasks: [persona()], artifactDir, maxSteps: 10 });
  assert.equal(result.findings[0].screenshot, undefined);
});

// P3 Chunk 7: hardening navigation during exploration. An allowed action (e.g. a click or a keyboard
// Enter) can cause an in-scope navigation/reload that destroys the page's execution context right as
// the harness's own observe()/capture() calls race it — a real driver's wording for this is Playwright's
// "Execution context was destroyed, most likely because of a navigation" (see chunk5's finding #2: Enter
// on a navigating link -> capture() throws this -> the whole task used to go "blocked"). These tests use
// fake drivers that throw that exact message to drive the fix without a real browser.
const CONTEXT_DESTROYED = new Error("Execution context was destroyed, most likely because of a navigation");

test("P3 Chunk 7: a transient context-destroyed observe() failure recovers and the task still passes", async t => {
  const driver = fixtureDriver();
  let observeCalls = 0;
  const realObserve = driver.observe;
  driver.observe = async () => { observeCalls += 1; if (observeCalls === 2) throw CONTEXT_DESTROYED; return realObserve(); };
  const result = await runExplorationLoop({ driver, model: async request => JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] }))],
    artifactDir: await temporary(t), maxSteps: 5 });
  assert.equal(result.scenarios[0].status, "passed");
  assert.ok(observeCalls > 2, "observe() must have been retried, not just failed once");
});

test("P3 Chunk 7: a transient context-destroyed capture() failure after a successful action does not block the task (chunk5 finding #2 repro)", async t => {
  const artifactDir = await temporary(t);
  let captureCalls = 0;
  const driver = {
    observe: (() => { let version = 0; return async () => observation(++version); })(),
    execute: async action => ({ ok: true, action }),
    capture: async name => { captureCalls += 1; if (captureCalls === 1) throw CONTEXT_DESTROYED; return `/artifacts/${name}.png`; },
  };
  const result = await runExplorationLoop({ driver, model: async request => JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] }))],
    artifactDir, maxSteps: 5 });
  assert.equal(result.scenarios[0].status, "passed");
  assert.ok(captureCalls > 1, "capture() must have been retried after the context-destroyed failure");
});

test("P3 Chunk 7: a persistently context-destroyed observe() still blocks after the bounded wait, with a clear reason", async t => {
  const driver = fixtureDriver();
  driver.observe = async () => { throw CONTEXT_DESTROYED; };
  const result = await runExplorationLoop({ driver, model: async () => '{"type":"wait","milliseconds":1}', tasks: [task()],
    artifactDir: await temporary(t), maxDurationMs: 400 });
  assert.equal(result.scenarios[0].status, "blocked");
  assert.match(result.scenarios[0].error, /execution context was destroyed|timeout/i);
});

test("P3 Chunk 7: a persistently context-destroyed capture() never blocks the task; it is recorded as step evidence instead", async t => {
  const driver = fixtureDriver();
  driver.capture = async () => { throw CONTEXT_DESTROYED; };
  const result = await runExplorationLoop({ driver, model: async request => JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] }))],
    artifactDir: await temporary(t), maxDurationMs: 4000, maxSteps: 5 });
  assert.equal(result.scenarios[0].status, "passed");
  // The bounded wait's own timeout (once it outraces the last retry) is what the task actually sees, so
  // the recorded reason may read either as the original context-destroyed message or as that bounded
  // wait's own "... capture timeout after Nms" wording — both are a clear reason, never a silent blocked.
  assert.ok(result.scenarios[0].steps.some(step => typeof step.captureError === "string" && /execution context was destroyed|capture timeout/i.test(step.captureError)));
});

test("P3 Chunk 7: an out-of-scope navigation error is a distinct guard, not retried, and still blocks immediately", async t => {
  const driver = fixtureDriver();
  let observeCalls = 0;
  driver.observe = async () => { observeCalls += 1; throw new Error("Browser left the isolated workbench scope"); };
  const result = await runExplorationLoop({ driver, model: async () => '{"type":"wait","milliseconds":1}', tasks: [task()],
    artifactDir: await temporary(t), maxDurationMs: 5000 });
  assert.equal(result.scenarios[0].status, "blocked");
  assert.match(result.scenarios[0].error, /left the isolated workbench scope/i);
  assert.equal(observeCalls, 1, "an out-of-scope error must not be retried");
});

test("a persona task without a verifier ends on done as explored, never passed, and keeps findings", async t => {
  const driver = fixtureDriver();
  let calls = 0;
  const result = await runExplorationLoop({ driver, model: async () => {
    calls += 1;
    return calls === 1 ? '{"type":"finding","summary":"Button label is unclear","evidence":"Settings"}' : '{"type":"done","reason":"explored enough"}';
  }, tasks: [persona()], artifactDir: await temporary(t), maxSteps: 10 });
  assert.equal(result.scenarios[0].status, "explored");
  assert.notEqual(result.scenarios[0].status, "passed");
  assert.equal(result.findings.length, 1);
  assert.equal(result.scenarios[0].findings.length, 1);
});

test("a goal task without a verifier is still rejected at entry", async t => {
  await assert.rejects(runExplorationLoop({ driver: fixtureDriver(), model: async () => '{"type":"done"}',
    tasks: [{ id: "no-verify", title: "x", goal: "y" }], artifactDir: await temporary(t) }), /verif/i);
});

test("a persona task that supplies verify is rejected as an ambiguous shape", async t => {
  await assert.rejects(runExplorationLoop({ driver: fixtureDriver(), model: async () => '{"type":"done"}',
    tasks: [{ ...persona(), verify: () => ({ passed: true, checks: [] }) }], artifactDir: await temporary(t) }), /verif|persona/i);
});

test("a persona task that never stops is ended as explored at its own step cap, and the next task still runs", async t => {
  const driver = fixtureDriver();
  const result = await runExplorationLoop({ driver, model: async request => request.task.kind === "persona"
    ? '{"type":"wait","milliseconds":1}'
    : JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [{ ...persona(), maxSteps: 2 }, task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] }))],
    artifactDir: await temporary(t), maxSteps: 50 });
  assert.equal(result.scenarios[0].status, "explored");
  assert.equal(result.scenarios[1].status, "passed");
});

test("a persona task that observes a product error is still failed, not explored", async t => {
  const driver = fixtureDriver();
  driver.observe = async () => ({ ...observation(), errors: ["Uncaught rendering error"] });
  const result = await runExplorationLoop({ driver, model: async () => '{"type":"done"}', tasks: [persona()], artifactDir: await temporary(t) });
  assert.equal(result.scenarios[0].status, "failed");
  assert.match(result.scenarios[0].error, /rendering/);
});

test("a suite with a passed goal task and an explored persona is overall passed", async t => {
  const driver = fixtureDriver();
  const result = await runExplorationLoop({ driver, model: async request => request.task.kind === "persona"
    ? '{"type":"done"}'
    : JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] })), persona()],
    artifactDir: await temporary(t), maxSteps: 10 });
  assert.equal(result.status, "passed");
  assert.deepEqual(result.scenarios.map(s => s.status), ["passed", "explored"]);
});

test("a suite made only of personas that all explore is still overall passed", async t => {
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async () => '{"type":"done"}',
    tasks: [persona(), { ...persona(), id: "impatient" }], artifactDir: await temporary(t) });
  assert.equal(result.status, "passed");
  assert.deepEqual(result.scenarios.map(s => s.status), ["explored", "explored"]);
});

test("selectExplorationTasks picks only the chosen ids, in library order, not the selection order", () => {
  const goalTasks = [{ id: "a" }, { id: "b" }, { id: "c" }];
  const personaTasks = [{ id: "x" }, { id: "y" }];
  const selected = selectExplorationTasks({ goalTasks, personaTasks, taskIds: ["c", "a"], personaIds: ["y"] });
  assert.deepEqual(selected.map(value => value.id), ["a", "c", "y"]);
});

test("selectExplorationTasks defaults to every goal task and no personas when nothing is selected", () => {
  const goalTasks = [{ id: "a" }, { id: "b" }], personaTasks = [{ id: "x" }];
  assert.deepEqual(selectExplorationTasks({ goalTasks, personaTasks }).map(value => value.id), ["a", "b"]);
});

test("selectExplorationTasks rejects an unknown id and lists the ids that do exist", () => {
  const goalTasks = [{ id: "a" }, { id: "b" }];
  assert.throws(() => selectExplorationTasks({ goalTasks, taskIds: ["zzz"] }), /zzz/);
  assert.throws(() => selectExplorationTasks({ goalTasks, taskIds: ["zzz"] }), /a, b/);
  const personaTasks = [{ id: "x" }];
  assert.throws(() => selectExplorationTasks({ goalTasks, personaTasks, personaIds: ["nope"] }), /nope.*\bx\b/s);
});

test("selectExplorationTasks selects goal tasks and personas independently", () => {
  const goalTasks = [{ id: "a" }, { id: "b" }], personaTasks = [{ id: "x" }, { id: "y" }];
  assert.deepEqual(selectExplorationTasks({ goalTasks, personaTasks, taskIds: ["b"] }).map(value => value.id), ["b"]);
  assert.deepEqual(selectExplorationTasks({ goalTasks, personaTasks, taskIds: [], personaIds: ["y"] }).map(value => value.id), ["y"]);
});

test("a goal task that exhausts its own step budget is blocked, not fatal; the next task still runs", async t => {
  const driver = fixtureDriver();
  const result = await runExplorationLoop({ driver, model: async () => '{"type":"wait","milliseconds":1}',
    tasks: [{ ...task(), maxSteps: 3 }, { ...task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] })), id: "reading" }],
    artifactDir: await temporary(t), maxSteps: 50 });
  assert.equal(result.scenarios[0].status, "blocked");
  assert.match(result.scenarios[0].error, /task budget exhausted/i);
  assert.equal(result.scenarios[0].usage.modelCalls, 3);
  // The second task gets its own fresh model behaviour (clicks) and can still pass.
  const second = await runExplorationLoop({ driver: fixtureDriver(), model: async request => JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }),
    tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] }))], artifactDir: await temporary(t), maxSteps: 5 });
  assert.equal(second.scenarios[0].status, "passed");
});

test("goal tasks default to a 20-step budget and persona tasks to 15", async t => {
  const driver = fixtureDriver();
  const goal = await runExplorationLoop({ driver, model: async () => '{"type":"wait","milliseconds":1}', tasks: [task()], artifactDir: await temporary(t), maxSteps: 100 });
  assert.equal(goal.scenarios[0].status, "blocked");
  assert.equal(goal.scenarios[0].usage.modelCalls, 20);
  const explored = await runExplorationLoop({ driver: fixtureDriver(), model: async () => '{"type":"wait","milliseconds":1}', tasks: [persona()], artifactDir: await temporary(t), maxSteps: 100 });
  assert.equal(explored.scenarios[0].status, "explored");
  assert.equal(explored.scenarios[0].usage.modelCalls, 15);
});

test("--task-max-steps overrides the default per-task step budget for tasks without their own maxSteps", async t => {
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async () => '{"type":"wait","milliseconds":1}',
    tasks: [task()], artifactDir: await temporary(t), maxSteps: 50, taskMaxSteps: 4 });
  assert.equal(result.scenarios[0].status, "blocked");
  assert.equal(result.scenarios[0].usage.modelCalls, 4);
});

test("a goal task that exhausts its own model API-call budget is blocked, not fatal; a later task still passes", async t => {
  const driver = fixtureDriver();
  const model = usageModel(request => request.task.id === "settings" ? '{"type":"wait","milliseconds":1}' : JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }), { apiCallsPerStep: 5 });
  const result = await runExplorationLoop({
    driver, model,
    tasks: [{ ...task(), maxApiCalls: 12, maxSteps: 50 },
      { ...task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] })), id: "reading" }],
    artifactDir: await temporary(t), maxSteps: 50,
  });
  assert.equal(result.scenarios[0].status, "blocked");
  assert.match(result.scenarios[0].error, /task budget exhausted.*api/i);
  assert.ok(result.scenarios[0].usage.apiCalls >= 12);
  assert.equal(result.scenarios[1].status, "passed");
});

test("run-wide model API-call budget exhaustion (the adapter's own cap) is fatal; later tasks are not-run", async t => {
  const driver = fixtureDriver();
  let calls = 0;
  const model = {
    async next() {
      calls += 1;
      if (calls > 2) { const error = new Error("Exploration model API call budget exhausted (2)"); error.name = "ExplorationBudgetError"; throw error; }
      return '{"type":"wait","milliseconds":1}';
    },
    metrics: () => ({ apiCalls: calls, inputTokens: 0, outputTokens: 0, totalTokens: 0 }),
  };
  const result = await runExplorationLoop({ driver, model, tasks: [task(), { ...task(), id: "reading" }, { ...task(), id: "third" }], artifactDir: await temporary(t), maxSteps: 50 });
  assert.equal(result.scenarios[0].status, "blocked");
  assert.match(result.scenarios[0].error, /budget exhausted/i);
  assert.deepEqual(result.scenarios.slice(1).map(value => value.status), ["not-run", "not-run"]);
});

test("per-scenario usage is recorded and the run totals equal the sum", async t => {
  const driver = fixtureDriver();
  const model = usageModel(request => request.task.id === "settings" ? JSON.stringify({ type: "click", ref: request.observation.controls[0].ref }) : '{"type":"done"}',
    { apiCallsPerStep: 1, inputTokensPerStep: 100, outputTokensPerStep: 10 });
  const result = await runExplorationLoop({
    driver, model,
    tasks: [task(({ actions }) => ({ passed: actions.length === 1, checks: [{ label: "ok", passed: actions.length === 1 }] })), persona()],
    artifactDir: await temporary(t), maxSteps: 10,
  });
  assert.equal(result.scenarios[0].status, "passed");
  assert.equal(result.scenarios[1].status, "explored");
  assert.equal(result.scenarios[0].usage.modelCalls, 1);
  assert.equal(result.scenarios[1].usage.modelCalls, 1);
  const totals = result.metrics.usage;
  assert.equal(totals.apiCalls, result.scenarios[0].usage.apiCalls + result.scenarios[1].usage.apiCalls);
  assert.equal(totals.inputTokens, result.scenarios[0].usage.inputTokens + result.scenarios[1].usage.inputTokens);
  assert.equal(totals.apiCalls, 2);
  assert.equal(totals.inputTokens, 200);
});

test("a task's own remaining step budget is sent to the model alongside the run's", async t => {
  const requests = [];
  await runExplorationLoop({ driver: fixtureDriver(), model: async request => { requests.push(request); return '{"type":"wait","milliseconds":1}'; },
    tasks: [{ ...task(), maxSteps: 3 }], artifactDir: await temporary(t), maxSteps: 50 });
  assert.deepEqual(requests.map(request => request.remaining.taskSteps), [3, 2, 1]);
});

test("action validation only accepts current refs and bounded browser operations", () => {
  const current = observation();
  assert.deepEqual(parseExplorationAction('{"type":"fill","ref":"s1-2","value":"5"}', current), { type: "fill", ref: "s1-2", value: "5" });
  for (const command of [
    { type: "click", ref: "s0-1" }, { type: "navigate", url: "https://example.com" },
    { type: "evaluate", code: "document.cookie" }, { type: "fill", ref: "s1-3", value: "secret" },
    { type: "fill", ref: "s1-4", value: "/real-vault" }, { type: "wait", milliseconds: 600000 },
    { type: "click", ref: "s1-1", unexpected: true }, { type: "fill", ref: "s1-1", value: "text" },
  ]) assert.throws(() => parseExplorationAction(JSON.stringify(command), current));
});

test("a host-agnostic keyboard press is allowed only from a strict allowlist and never carries a ref", () => {
  const current = observation();
  assert.deepEqual(parseExplorationAction('{"type":"press","key":"Enter"}', current), { type: "press", key: "Enter" });
  assert.deepEqual(parseExplorationAction('{"type":"press","key":"Shift+Tab"}', current), { type: "press", key: "Shift+Tab" });
  assert.deepEqual(parseExplorationAction('{"type":"press","key":"ArrowDown","reason":"move to the next option"}', current), { type: "press", key: "ArrowDown", reason: "move to the next option" });
  for (const command of [
    { type: "press", key: "a" }, { type: "press", key: "Control+C" }, { type: "press", key: "F5" },
    { type: "press" }, { type: "press", key: "" }, { type: "press", key: "enter" },
    { type: "press", key: "Enter", ref: current.controls[0].ref },
  ]) assert.throws(() => parseExplorationAction(JSON.stringify(command), current));
});

// P4 Chunk 2 Fix 2: a host-agnostic recovery action, usable with no ref, so a model that has wandered
// outside the explorable plugin/app surface (empty controls) still has a safe way back.
test("home is a valid action with no ref, and is a browser action routed through driver.execute", async t => {
  const current = observation();
  assert.deepEqual(parseExplorationAction('{"type":"home"}', current), { type: "home" });
  assert.deepEqual(parseExplorationAction('{"type":"home","reason":"left the plugin surface"}', current), { type: "home", reason: "left the plugin surface" });
  assert.throws(() => parseExplorationAction('{"type":"home","ref":"s1-1"}', current), "home never carries a ref");
  const driver = fixtureDriver();
  const result = await runExplorationLoop({ driver, model: async () => '{"type":"home"}', tasks: [{ ...task(), maxSteps: 1 }], artifactDir: await temporary(t), maxSteps: 3 });
  assert.deepEqual(driver.executed, [{ type: "home" }]);
  assert.equal(result.scenarios[0].status, "blocked"); // never satisfies the task's own checks, same as any other single action
});

// P4 Chunk 2 Fix 3: many identical <select> option lists (e.g. ~41 arXiv categories repeated once per
// settings row) must collapse to one shared observation.optionSets entry, never one inline copy per
// control, while parseExplorationAction's select validation keeps working either way.
test("select validation accepts an enabled option via either inline options or a shared optionsRef/optionSets, and rejects a disabled or absent one", () => {
  const OPTIONS = [{ value: "cs.AI", label: "Artificial Intelligence", disabled: false }, { value: "cs.LG", label: "Machine Learning", disabled: false }, { value: "cs.CV", label: "Computer Vision", disabled: true }];
  const inline = { id: "snap-inline", controls: [{ ref: "s1-1", tag: "select", options: OPTIONS }] };
  const shared = { id: "snap-ref", optionSets: { o1: OPTIONS }, controls: [{ ref: "s1-1", tag: "select", optionsRef: "o1" }] };
  for (const current of [inline, shared]) {
    assert.deepEqual(parseExplorationAction('{"type":"select","ref":"s1-1","value":"cs.LG"}', current), { type: "select", ref: "s1-1", value: "cs.LG" });
    assert.throws(() => parseExplorationAction('{"type":"select","ref":"s1-1","value":"cs.CV"}', current), /enabled option/);
    assert.throws(() => parseExplorationAction('{"type":"select","ref":"s1-1","value":"astro-ph"}', current), /enabled option/);
  }
});

test("collapseOptionSets: identical select option lists collapse to one shared entry; distinct lists get their own", () => {
  const CATEGORY_OPTIONS = Array.from({ length: 41 }, (_, index) => ({ value: `cat.${index}`, label: `Category ${index}`, disabled: false }));
  const OTHER_OPTIONS = [{ value: "zh", label: "Chinese", disabled: false }, { value: "en", label: "English", disabled: false }];
  const controls = [
    { ref: "s1-1", tag: "select", options: CATEGORY_OPTIONS },
    { ref: "s1-2", tag: "select", options: CATEGORY_OPTIONS.map(option => ({ ...option })) }, // a fresh array, same content
    { ref: "s1-3", tag: "select", options: OTHER_OPTIONS },
    { ref: "s1-4", tag: "button" }, // never touched: not a select
  ];
  const optionSets = collapseOptionSets(controls);
  assert.equal(Object.keys(optionSets).length, 2, "two distinct option lists, regardless of how many controls share one");
  assert.equal(controls[0].optionsRef, controls[1].optionsRef, "identical option lists share one ref");
  assert.notEqual(controls[0].optionsRef, controls[2].optionsRef);
  assert.deepEqual(optionSets[controls[0].optionsRef], CATEGORY_OPTIONS);
  assert.deepEqual(optionSets[controls[2].optionsRef], OTHER_OPTIONS);
  for (const control of controls.slice(0, 3)) assert.ok(!("options" in control), "the inline copy is replaced, not kept alongside the ref");
  assert.ok(!("optionsRef" in controls[3]));
});

// P4 Chunk 2 Fix 3 regression: a settings page with ~20 arXiv-category rows (each a real ~41-option
// select) used to serialize one full inline options array per row and, in the real P4 Chunk 1 run, blew
// past exploration-model.ts's 100,000-character model-input budget. After collapseOptionSets, the whole
// observation (plus the task/checks/history/remaining envelope exploration-model.ts actually sends) must
// stay well under that budget.
test("a settings page with ~20 identical-option category rows stays well under the 100000-character model payload budget", () => {
  const CATEGORY_OPTIONS = Array.from({ length: 41 }, (_, index) => ({ value: `group.cat-${index}`, label: `Some fairly long category label ${index}`, disabled: false }));
  const controls = Array.from({ length: 20 }, (_, row) => ({
    ref: `s1-${row + 1}`, tag: "select", name: `Category ${row + 1}`, field: "", type: "", role: "",
    disabled: false, readOnly: false, options: CATEGORY_OPTIONS.map(option => ({ ...option })),
    rect: { x: 0, y: row * 40, width: 300, height: 32 }, inViewport: true,
  }));
  const optionSets = collapseOptionSets(controls);
  const payload = JSON.stringify({ task: { id: "add-arxiv-category", title: "x", goal: "x" }, observation: { id: "s99", controls, optionSets }, checks: [], history: [], remaining: {} });
  assert.ok(payload.length < 20000, `expected a small deduplicated payload, got ${payload.length} characters`);
  assert.equal(Object.keys(optionSets).length, 1, "every row shares the same option list");
});

test("press is a browser action routed through driver.execute like click or fill", async t => {
  const driver = fixtureDriver();
  const result = await runExplorationLoop({
    driver, model: async () => '{"type":"press","key":"Tab"}',
    tasks: [task(({ actions }) => ({ passed: actions.some(action => action.type === "press"), checks: [{ label: "pressed a key", passed: actions.some(action => action.type === "press") }] }))],
    artifactDir: await temporary(t), maxSteps: 5,
  });
  assert.equal(result.scenarios[0].status, "passed");
  assert.deepEqual(driver.executed.map(action => action.type), ["press"]);
});

async function localModel(t, respond = (_body, response) => {
  response.writeHead(200, { "content-type": "text/event-stream" });
  response.end(`data: ${JSON.stringify({ choices: [{ delta: { content: '{"type":"done"}' } }], usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 } })}\n\ndata: [DONE]\n\n`);
}) {
  const requests = [];
  const server = createServer((request, response) => {
    let raw = "";
    request.on("data", chunk => { raw += chunk; });
    request.on("end", () => {
      const body = JSON.parse(raw);
      requests.push({ path: request.url, body });
      respond(body, response);
    });
  });
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  return { requests, url: `http://127.0.0.1:${server.address().port}/v1` };
}

async function modelConfig(t, url, extra = "") {
  const root = await temporary(t);
  const configPath = path.join(root, "config.toml");
  // Deliberately no vault_root or business settings: the explorer only reads [llm].
  await writeFile(configPath, `[llm]\nprovider = "custom"\nbase_url = ${JSON.stringify(url)}\napi_key = "test-model-key-private"\nmodel = "controlled-ui-model"\nthinking_mode = false\n${extra}`);
  return configPath;
}

test("configured exploration uses the real model client while reading only CLI llm fields", async t => {
  const server = await localModel(t);
  const configPath = await modelConfig(t, server.url);
  const model = await loadExplorationModel({ configPath, maxApiCalls: 4 });
  const answer = await model.next({ task: { id: "test", title: "Test", goal: "Check controls" }, observation: observation(), history: [], checks: [] });
  assert.equal(answer, '{"type":"done"}');
  assert.equal(server.requests[0].path, "/v1/chat/completions");
  assert.equal(server.requests[0].body.model, "controlled-ui-model");
  assert.equal(server.requests[0].body.max_tokens, 4096);
  assert.equal(server.requests[0].body.stream, true);
  assert.match(server.requests[0].body.messages[0].content, /ref/);
  assert.ok(!JSON.stringify(server.requests).includes("test-model-key-private"));
  assert.ok(!JSON.stringify(model.describe).includes("test-model-key-private"));
  assert.equal(model.metrics().apiCalls, 1);
  assert.equal(model.metrics().inputTokens, 12);
  assert.equal(model.redact("test-model-key-private"), "[REDACTED]");
});

test("missing model configuration or key is a clear blocker with no model request", async t => {
  const root = await temporary(t);
  await assert.rejects(loadExplorationModel({ configPath: path.join(root, "missing.toml") }), /config|配置/i);
  const file = path.join(root, "no-key.toml");
  await writeFile(file, '[llm]\nbase_url = "https://example.invalid/v1"\nmodel = "fake"\n');
  await assert.rejects(loadExplorationModel({ configPath: file }), /api.key|密钥/i);
});

test("all HTTP attempts, including stream-option fallback, consume the model budget", async t => {
  const server = await localModel(t, (_body, response) => {
    response.writeHead(400, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: { message: "stream_options include_usage unsupported" } }));
  });
  const model = await loadExplorationModel({ configPath: await modelConfig(t, server.url), maxApiCalls: 1 });
  await assert.rejects(model.next({ task: task(), observation: observation(), history: [], checks: [] }), /budget|预算/i);
  assert.equal(server.requests.length, 1);
  assert.equal(model.metrics().apiCalls, 1);
});

test("provider failure text is redacted before it reaches a report", async t => {
  const server = await localModel(t, (_body, response) => {
    response.writeHead(401, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: { message: "Rejected test-model-key-private" } }));
  });
  const model = await loadExplorationModel({ configPath: await modelConfig(t, server.url) });
  const artifactDir = await temporary(t);
  const result = await runExplorationLoop({ model, driver: fixtureDriver(), tasks: [task()], artifactDir });
  assert.equal(result.status, "blocked");
  const events = await readFile(path.join(artifactDir, "exploration-events.jsonl"), "utf8");
  const report = await readFile(path.join(artifactDir, "exploration.json"), "utf8");
  assert.ok(!events.includes("test-model-key-private"));
  assert.ok(!report.includes("test-model-key-private"));
  assert.match(report, /REDACTED/);
});

test("the real HTTP adapter aborts a stalled model within the exploration deadline", async t => {
  const server = await localModel(t, () => {});
  const model = await loadExplorationModel({ configPath: await modelConfig(t, server.url) });
  const result = await runExplorationLoop({ model, driver: fixtureDriver(), tasks: [task()], artifactDir: await temporary(t), modelCallTimeoutMs: 30 });
  assert.equal(result.status, "blocked");
  assert.match(result.errors.join(" "), /timeout/);
  assert.equal(server.requests.length, 1);
});

test('browser controls cannot escape the fixture through secrets, paths or external navigation', () => {
  const url = 'http://127.0.0.1:3456/session/';
  for (const control of [
    {tag:'input',type:'password',field:'apiKey'}, {tag:'input',type:'text',field:'vaultRoot'},
    {tag:'input',type:'text',field:'dailyDir'}, {tag:'input',type:'text',field:'embedding.baseUrl'},
    {tag:'button',dataAction:'email-test',name:'Send test'}, {tag:'button',dataAction:'library-connect',name:'Choose folder'},
    {tag:'a',href:'https://example.com'}, {tag:'a',href:'javascript:alert(1)'}, {tag:'a',href:'file:///tmp/file'},
    {tag:'a',href:'http://127.0.0.1:3456/other/'}, {tag:'input',type:'file',field:'papers'},
  ]) assert.ok(classifyControlRestriction(control,url),JSON.stringify(control));
  for (const control of [{tag:'button',name:'Settings'}, {tag:'input',type:'number',field:'maxDailyPapers'},
    {tag:'select',field:'appearance.theme'}, {tag:'a',href:'./?paper=2610.10001'}]) {
    assert.equal(classifyControlRestriction(control,url),undefined);
  }
});


test("a pre-aborted parent signal prevents any model or browser operation", async t => {
  const controller = new AbortController(); controller.abort("owner cancelled"); let calls = 0;
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async () => { calls++; return '{"type":"done"}'; }, tasks: [task()], artifactDir: await temporary(t), signal: controller.signal });
  assert.equal(calls, 0); assert.equal(result.status, "blocked"); assert.match(result.errors.join(" "), /owner cancelled/);
});

test("parent cancellation aborts an in-flight model request and records the reason", async t => {
  const controller = new AbortController(); let received;
  const result = await runExplorationLoop({ driver: fixtureDriver(), model: async request => { received = request.signal; setTimeout(() => controller.abort("owner cancelled"), 5); return new Promise(() => {}); }, tasks: [task()], artifactDir: await temporary(t), signal: controller.signal, modelCallTimeoutMs: 60 });
  assert.equal(received.aborted, true); assert.equal(result.status, "blocked"); assert.match(result.errors.join(" "), /owner cancelled/);
});

test('a failed fixture run or missing report after completed is a verified failure', async t => {
  const root=await mkdtemp(path.join(tmpdir(),'exploration-verifier-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const fixture={root,vaultRoot:path.join(root,'vault'),configPath:path.join(root,'config.toml')};
  await mkdir(fixture.vaultRoot);
  await writeFile(fixture.configPath,`vault_root = ${JSON.stringify(fixture.vaultRoot)}\n[output]\nmax_daily_papers = 2\n`);
  const task=(await createDefaultExplorationTasks(fixture)).find(task=>task.id==='generate-and-read');
  const startedAt=Date.now()-10;
  for (const status of ['failed','completed']) {
    const answer=await task.verify({driver:{dailyEvidence:async()=>({report:false,run:{date:'2026-10-01',status,startedAt:new Date().toISOString()}})},actions:[{type:'click',control:{dataAction:'generate'}}],observation:{layout:{view:'list'},text:''},startedAt});
    assert.equal(answer.failed,true,status);
    assert.equal(answer.passed,false);
  }
});

test('the actual settings secret-disclosure action is outside exploration scope',()=>{
  assert.ok(classifyControlRestriction({tag:'button',dataAction:'show-secret',name:'显示 API 密钥'},'http://127.0.0.1:3000/session/'));
});

test('completion evidence reads the report after observing the completed run', async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'exploration-completion-race-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const fixture = { root, vaultRoot: path.join(root, 'vault'), configPath: path.join(root, 'config.toml') };
  await mkdir(fixture.vaultRoot);
  await writeFile(fixture.configPath, `vault_root = ${JSON.stringify(fixture.vaultRoot)}\n[output]\ndaily_dir = "daily"\n`);
  const url = 'http://127.0.0.1:3000/session/';
  const page = {
    url: () => url, on() {}, off() {},
    locator: () => ({ waitFor: async () => {} }),
    request: { get: async () => {
      // The operation completes between polls, writing its report before announcing success.
      await mkdir(path.join(fixture.vaultRoot, 'daily'));
      await writeFile(path.join(fixture.vaultRoot, 'daily', '2026-10-01.md'), '# Report\n2610.10001\n');
      return { ok: () => true, json: async () => ({ run: { date: '2026-10-01', status: 'completed' } }) };
    } },
  };
  const context = { on() {}, off() {}, route: async () => {}, unroute: async () => {} };
  const driver = await createExplorationDriver({ page, context, url, fixture, artifactDir: path.join(root, 'artifacts') });
  t.after(() => driver.dispose());
  const evidence = await driver.dailyEvidence('2026-10-01');
  assert.equal(evidence.run.status, 'completed');
  assert.equal(evidence.report, true, 'completed must be paired with a fresh file observation');
});

// P4 Chunk 2 Fix 1: after external-config-conflict rewrites the fixture's CLI TOML out-of-band
// (writeConfigPatch, bypassing the server's own save path), the server's in-memory config falls out of
// sync with disk and every later task/persona in the same session sees the "配置文件已在外部修改..." 409
// guard leak in (observed in the real P4 Chunk 1 run). reset() must resync the server's held config to
// disk before every task, not just after that one task.
function fakeResetPage(url, { revision = 'sha256:current' } = {}) {
  const calls = [];
  let currentUrl = url;
  return {
    calls,
    url: () => currentUrl,
    on() {}, off() {},
    goto: async target => { currentUrl = String(target); },
    locator(selector) {
      if (selector === 'dialog[open]') return { count: async () => 0, waitFor: async () => {}, locator: () => ({ first: () => ({ click: async () => {} }) }) };
      return { waitFor: async () => {} };
    },
    request: {
      get: async target => { calls.push(['GET', String(target)]); return { ok: () => true, json: async () => ({ revision, setupRequired: revision === null, values: {} }) }; },
      post: async (target, options) => { calls.push(['POST', String(target), options?.data]); return { ok: () => true, json: async () => ({}) }; },
    },
  };
}

test("reset() resyncs the workbench config revision with disk before every task, so a write after an earlier external config edit is not rejected as conflicting", async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'exploration-resync-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const fixture = { root, vaultRoot: path.join(root, 'vault'), configPath: path.join(root, 'config.toml') };
  await mkdir(fixture.vaultRoot);
  await writeFile(fixture.configPath, `vault_root = ${JSON.stringify(fixture.vaultRoot)}\n`);
  const url = 'http://127.0.0.1:3100/session/';
  const page = fakeResetPage(url, { revision: 'sha256:current-disk-revision' });
  const context = { on() {}, off() {}, route: async () => {}, unroute: async () => {} };
  const driver = await createExplorationDriver({ page, context, url, fixture, artifactDir: path.join(root, 'artifacts') });
  t.after(() => driver.dispose());
  await driver.reset();
  assert.deepEqual(page.calls.map(call => call[0]), ['GET', 'POST'], "reset() must read, then resave, the config through the server's own settings API, every task boundary");
  assert.equal(page.calls[1][1], new URL('api/settings', url).href);
  assert.equal(page.calls[1][2].revision, 'sha256:current-disk-revision', "the resync save must use the revision actually on disk right now, not a stale one");
  assert.deepEqual(page.calls[1][2].values, {}, "a resync save changes nothing on disk -- it only lets the server's held config catch up to the current disk content");
});

test("reset() skips the config resync when settings have never been configured (revision null), instead of sending an invalid save", async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'exploration-resync-unconfigured-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const fixture = { root, vaultRoot: path.join(root, 'vault'), configPath: path.join(root, 'config.toml') };
  await mkdir(fixture.vaultRoot);
  await writeFile(fixture.configPath, `vault_root = ${JSON.stringify(fixture.vaultRoot)}\n`);
  const url = 'http://127.0.0.1:3101/session/';
  const page = fakeResetPage(url, { revision: null });
  const context = { on() {}, off() {}, route: async () => {}, unroute: async () => {} };
  const driver = await createExplorationDriver({ page, context, url, fixture, artifactDir: path.join(root, 'artifacts') });
  t.after(() => driver.dispose());
  await driver.reset();
  assert.deepEqual(page.calls.map(call => call[0]), ['GET'], 'no settings exist yet; there is nothing to resync and no save should be attempted');
});

// ---------------------------------------------------------------------------
// P1 Chunk 2 (docs/helm/2026-10-13-exploration-safety-and-coverage): `press` must refuse to dispatch a
// key when the live document.activeElement is restricted/sensitive/disabled or has drifted outside the
// workbench surface — the same policy click/fill/select already re-check against the live DOM (lines
// 162-167 above) before acting, which `press` previously skipped entirely (scripts/acceptance/
// exploration-browser.mjs:183 before this chunk: a bare `page.keyboard.press(action.key)`, no checks of
// any kind). `page`/`context` here are hand-built fakes, same convention as `fakeResetPage` above — no
// real browser — so `evaluate()` just hands back whatever focus descriptor the scenario wants to pretend
// is currently focused, keyed off the evaluated function's own name (`browserObservation` for the
// observe() this test drives first, `liveControlDescriptor` for the focus check under test).
// ---------------------------------------------------------------------------

function fakePressPage(url, { focus, dialogOpen = false } = {}) {
  const calls = [];
  return {
    calls,
    url: () => url,
    on() {}, off() {},
    locator(selector) {
      if (selector === "dialog[open]") return { count: async () => (dialogOpen ? 1 : 0), waitFor: async () => {}, locator: () => ({ first: () => ({ click: async () => {} }) }) };
      return { waitFor: async () => {} };
    },
    async evaluate(fn) {
      calls.push(["evaluate", fn.name]);
      if (fn.name === "browserObservation") {
        return { id: "s1", title: "Workbench", text: "", controls: [],
          layout: { rootMounted: true, view: "list", dialogOpen, sidebarCollapsed: false, viewport: { width: 1024, height: 768 } } };
      }
      if (fn.name === "liveControlDescriptor") return focus;
      throw new Error(`Unhandled fake evaluate(): ${fn.name}`);
    },
    keyboard: { press: async key => calls.push(["press", key]) },
  };
}

async function makePressDriver(t, { focus, dialogOpen } = {}) {
  const root = await mkdtemp(path.join(tmpdir(), 'exploration-press-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const fixture = { root, vaultRoot: path.join(root, 'vault'), configPath: path.join(root, 'config.toml') };
  await mkdir(fixture.vaultRoot);
  await writeFile(fixture.configPath, `vault_root = ${JSON.stringify(fixture.vaultRoot)}\n`);
  const url = 'http://127.0.0.1:3102/session/';
  const page = fakePressPage(url, { focus, dialogOpen });
  const context = { on() {}, off() {}, route: async () => {}, unroute: async () => {} };
  const driver = await createExplorationDriver({ page, context, url, fixture, artifactDir: path.join(root, 'artifacts') });
  t.after(() => driver.dispose());
  const observation = await driver.observe();
  return { driver, page, observation };
}

const notFocused = { tag: 'body', type: '', field: '', name: '', href: '', dataAction: '', disabled: false, readOnly: false, outsideDialog: false, outsideWorkbench: false };

test('press Space is refused when focus is on the schedule.enabled switch, and no key is sent', async t => {
  const { driver, page, observation } = await makePressDriver(t, { focus: { ...notFocused, tag: 'input', type: 'checkbox', field: 'schedule.enabled', name: '启用定时任务' } });
  await assert.rejects(driver.execute({ type: 'press', key: 'Space' }, observation), /ExplorationActionError/);
  assert.ok(!page.calls.some(call => call[0] === 'press'), 'the scheduler switch must never receive the key event');
});

test('press Enter is refused when focus is on the email.enabled switch, and no key is sent', async t => {
  const { driver, page, observation } = await makePressDriver(t, { focus: { ...notFocused, tag: 'input', type: 'checkbox', field: 'email.enabled', name: '发送邮件' } });
  await assert.rejects(driver.execute({ type: 'press', key: 'Enter' }, observation), /ExplorationActionError/);
  assert.ok(!page.calls.some(call => call[0] === 'press'), 'the email switch must never receive the key event');
});

test('press is refused when the focused credential field is sensitive, and no key is sent', async t => {
  const { driver, page, observation } = await makePressDriver(t, { focus: { ...notFocused, tag: 'input', type: 'password', field: 'apiKey', name: 'LLM API Key' } });
  await assert.rejects(driver.execute({ type: 'press', key: 'Space' }, observation), /ExplorationActionError/);
  assert.ok(!page.calls.some(call => call[0] === 'press'), 'a credential field must never receive the key event');
});

test('Tab, Shift+Tab and Escape still move away from a restricted focused control', async t => {
  const { driver, page, observation } = await makePressDriver(t, { focus: { ...notFocused, tag: 'input', type: 'checkbox', field: 'schedule.enabled', name: '启用定时任务' } });
  for (const key of ['Tab', 'Shift+Tab', 'Escape']) assert.deepEqual(await driver.execute({ type: 'press', key }, observation), { ok: true });
  assert.deepEqual(page.calls.filter(call => call[0] === 'press'), [['press', 'Tab'], ['press', 'Shift+Tab'], ['press', 'Escape']]);
});

test('Tab is still refused when focus has left the workbench surface', async t => {
  const { driver, page, observation } = await makePressDriver(t, { focus: { ...notFocused, tag: 'a', name: 'External', href: 'https://example.com/', outsideWorkbench: true } });
  await assert.rejects(driver.execute({ type: 'press', key: 'Tab' }, observation), /ExplorationActionError/);
  assert.ok(!page.calls.some(call => call[0] === 'press'));
});

test('press Enter is refused when focus has drifted outside the workbench surface, and no key is sent', async t => {
  const { driver, page, observation } = await makePressDriver(t, { focus: { ...notFocused, tag: 'a', name: 'External', href: 'https://example.com/', outsideWorkbench: true } });
  await assert.rejects(driver.execute({ type: 'press', key: 'Enter' }, observation), /ExplorationActionError/);
  assert.ok(!page.calls.some(call => call[0] === 'press'), 'focus outside the workbench must never receive the key event');
});

test('an ordinary press on an unrestricted, in-scope, enabled control still dispatches the key', async t => {
  const { driver, page, observation } = await makePressDriver(t, { focus: notFocused });
  const result = await driver.execute({ type: 'press', key: 'ArrowDown' }, observation);
  assert.deepEqual(result, { ok: true });
  assert.deepEqual(page.calls.filter(call => call[0] === 'press'), [['press', 'ArrowDown']], 'ordinary keyboard navigation must still work');
});

// ---------------------------------------------------------------------------
// P3 Chunk 5: workbench reading-journey and settings-conflict goal tasks.
// Same convention as the tests above: a real temporary fixture directory
// (vaultRoot + configPath on disk), task.verify()/task.prepare() called
// directly with hand-built observations/actions/fake driver methods — never a
// real browser or model here (that is the real-host evidence in the chunk
// report, not this file).
// ---------------------------------------------------------------------------

async function makeWorkbenchFixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), 'exploration-chunk5-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const fixture = { root, vaultRoot: path.join(root, 'vault'), configPath: path.join(root, 'config.toml') };
  await mkdir(fixture.vaultRoot);
  await writeFile(fixture.configPath, `vault_root = ${JSON.stringify(fixture.vaultRoot)}\n[llm]\nmodel = "fixture-model"\n[output]\ndaily_dir = "arxiv-daily/daily"\npapers_dir = "arxiv-daily/papers"\nmax_daily_papers = 2\n`);
  return fixture;
}
async function writeWorkbenchPaperIndex(fixture, papers) {
  const file = path.join(fixture.vaultRoot, 'arxiv-daily/.index/papers.json');
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify({ papers }));
}

// P3 Chunk 7 negative-evidence audit: settings-persistence, sidebar-persistence and generate-and-read
// (the original 3 workbench goal tasks, P1) had no dedicated "verify fails on the untouched state" unit
// test anywhere in the suite before this — added here, same convention as every reading-journey task's
// own untouched-state test below.
test('settings-persistence: verify fails on the untouched state (no fill, no reload)', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'settings-persistence');
  const driver = { inspect: async () => ({ maxDailyPapers: 2, preferences: {} }) };
  const result = await task.verify({ driver, observation: { controls: [] }, actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false, 'must not pass without an observed fill of maxDailyPapers');
});

test('sidebar-persistence: verify fails on the untouched state (no toggle, no reload)', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'sidebar-persistence');
  const driver = { inspect: async () => ({ preferences: { sidebarCollapsed: false } }) };
  const result = await task.verify({ driver, observation: { layout: {} }, actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false, 'must not pass without an observed sidebar-toggle click');
});

test('generate-and-read: verify fails on the fully untouched state (no action at all)', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'generate-and-read');
  const driver = { dailyEvidence: async () => ({ report: false, run: null }) };
  const result = await task.verify({ driver, observation: { layout: { view: 'list' }, text: '' }, actions: [], startedAt: Date.now() });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false, 'must not pass without an observed click on the generate entry point');
});

test('star-filter-restore: verify fails on the untouched state (no star click yet)', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'star-filter-restore');
  const result = await task.verify({ observation: { controls: [] }, actions: [] });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, false);
});

test('star-filter-restore: verify fails after the star click alone, before the "收藏" scope filter is used', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'star-filter-restore');
  const actions = [{ type: 'click', control: { name: `收藏 · ${PAPERS[1].title}` } }];
  const result = await task.verify({ observation: { controls: [] }, actions });
  assert.equal(result.passed, false);
  assert.equal(result.checks[0].passed, true);
  assert.equal(result.checks[1].passed, false);
});

test('star-filter-restore: verify passes once star, filter, open and back are all observed, in order, and the disk/UI state agree', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'star-filter-restore');
  const paper = PAPERS[1], other = PAPERS[0];
  const starKey = `arxiv:${paper.id}`, otherKey = `arxiv:${other.id}`;
  const actions = [
    { type: 'click', control: { name: `收藏 · ${paper.title}` } },
    { type: 'click', control: { name: '收藏12' } },
    { type: 'click', control: { paperKey: starKey, name: paper.title } },
    { type: 'click', control: { name: '← 返回列表' } },
  ];
  await writeWorkbenchPaperIndex(fixture, { [starKey]: { priority: 'high' } });
  const result = await task.verify({ observation: { controls: [{ paperKey: starKey }] }, actions });
  assert.equal(result.passed, true, JSON.stringify(result));
  // Mutation: the other (unstarred) paper still showing means the filter never really restored.
  const stillShowingOther = await task.verify({ observation: { controls: [{ paperKey: starKey }, { paperKey: otherKey }] }, actions });
  assert.equal(stillShowingOther.passed, false);
  // Mutation: the on-disk mark was never actually "high".
  await writeWorkbenchPaperIndex(fixture, { [starKey]: { priority: 'normal' } });
  const notMarked = await task.verify({ observation: { controls: [{ paperKey: starKey }] }, actions });
  assert.equal(notMarked.passed, false);
});

test('star-filter-restore: prepare() ensures the daily report exists and resets the target paper to unstarred', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'star-filter-restore');
  const calls = [];
  await task.prepare({ driver: { ensureDailyReport: async date => calls.push(['ensureDailyReport', date]), ensurePaperUnstarred: async key => calls.push(['ensurePaperUnstarred', key]) } });
  assert.deepEqual(calls, [['ensureDailyReport', ACCEPTANCE_DATE], ['ensurePaperUnstarred', `arxiv:${PAPERS[1].id}`]]);
});

test('generate-paper-note: verify fails on the untouched state, and still fails after all three clicks while the note file does not exist yet', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'generate-paper-note');
  const key = `arxiv:${PAPERS[0].id}`;
  const result0 = await task.verify({ observation: { controls: [] }, actions: [] });
  assert.equal(result0.passed, false);
  const actions = [
    { type: 'click', control: { paperKey: key } },
    { type: 'click', control: { name: '生成详细总结' } },
    { type: 'click', control: { name: '开始生成' } },
  ];
  const result1 = await task.verify({ observation: { controls: [] }, actions });
  assert.equal(result1.passed, false, 'must not pass before the note file exists');
});

test('generate-paper-note: verify passes once the note file, index entry and UI link all agree; fails if the index was never updated', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'generate-paper-note');
  const paper = PAPERS[0], key = `arxiv:${paper.id}`;
  const actions = [
    { type: 'click', control: { paperKey: key } },
    { type: 'click', control: { name: '生成详细总结' } },
    { type: 'click', control: { name: '开始生成' } },
  ];
  const notePath = path.join(fixture.vaultRoot, 'arxiv-daily/papers', `${paper.id}.md`);
  await mkdir(path.dirname(notePath), { recursive: true });
  await writeFile(notePath, `---\narxiv_id: "${paper.id}"\n---\n\nReal generated content, well over a hundred characters long, so the length check in verify() actually exercises a real body rather than a stub.\n`);
  const observation = { controls: [{ name: '阅读详细总结' }] };
  // Mutation: the file exists but the index was never updated to reflect it.
  const withoutIndex = await task.verify({ observation, actions });
  assert.equal(withoutIndex.passed, false);
  await writeWorkbenchPaperIndex(fixture, { [key]: { detail: true, paperPath: `arxiv-daily/papers/${paper.id}.md` } });
  const result = await task.verify({ observation, actions });
  assert.equal(result.passed, true, JSON.stringify(result));
});

test('generate-paper-note: prepare() ensures the daily report exists and clears any existing note for the target', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const paper = PAPERS[0];
  const notePath = path.join(fixture.vaultRoot, 'arxiv-daily/papers', `${paper.id}.md`);
  await mkdir(path.dirname(notePath), { recursive: true });
  await writeFile(notePath, 'stale note');
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'generate-paper-note');
  const calls = [];
  await task.prepare({ driver: { ensureDailyReport: async date => calls.push(date) } });
  assert.deepEqual(calls, [ACCEPTANCE_DATE]);
  await assert.rejects(readFile(notePath));
});

test('external-config-conflict: verify never writes the external change until the model field is actually observed rendered', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'external-config-conflict');
  await task.prepare();
  const before = await readFile(fixture.configPath, 'utf8');
  const result = await task.verify({ observation: { controls: [{ name: 'Settings' }] }, actions: [] });
  assert.equal(result.passed, false);
  const after = await readFile(fixture.configPath, 'utf8');
  assert.equal(after, before, 'no model field observed yet; no external write should happen');
});

test('external-config-conflict: verify writes the external value to disk exactly once, the first time the model field is observed rendered', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'external-config-conflict');
  await task.prepare();
  const observationWithModelField = { controls: [{ field: 'model', value: 'fixture-model' }] };
  await task.verify({ observation: observationWithModelField, actions: [] });
  const config1 = await readFile(fixture.configPath, 'utf8');
  assert.match(config1, /workbench-conflict-external-model/);
  await task.verify({ observation: observationWithModelField, actions: [] });
  const config2 = await readFile(fixture.configPath, 'utf8');
  assert.equal(config2, config1, 'the external write only happens once per task run');
});

test('external-config-conflict: verify passes once discard-then-reopen is fully observed and disk/UI agree on the external value; fails if the draft ever reached disk', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'external-config-conflict');
  await task.prepare();
  await task.verify({ observation: { controls: [{ field: 'model', value: 'fixture-model' }] }, actions: [] });
  const actions = [
    { type: 'fill', control: { field: 'model' }, value: 'workbench-conflict-draft-model' },
    { type: 'click', control: { name: '放弃未保存修改并关闭' } },
    { type: 'click', control: { name: '放弃修改并关闭' } },
    { type: 'click', control: { name: '设置' } },
  ];
  const result = await task.verify({ observation: { controls: [{ field: 'model', value: 'workbench-conflict-external-model' }] }, actions });
  assert.equal(result.passed, true, JSON.stringify(result));
  const config = await readFile(fixture.configPath, 'utf8');
  assert.match(config, /workbench-conflict-external-model/);
  assert.doesNotMatch(config, /workbench-conflict-draft-model/);
});

test('external-config-conflict: verify fails if "设置" is clicked again before the discard was actually confirmed', async t => {
  const fixture = await makeWorkbenchFixture(t);
  const task = (await createDefaultExplorationTasks(fixture)).find(x => x.id === 'external-config-conflict');
  await task.prepare();
  await task.verify({ observation: { controls: [{ field: 'model', value: 'fixture-model' }] }, actions: [] });
  const actions = [
    { type: 'fill', control: { field: 'model' }, value: 'workbench-conflict-draft-model' },
    { type: 'click', control: { name: '放弃未保存修改并关闭' } },
    { type: 'click', control: { name: '设置' } },
  ];
  const result = await task.verify({ observation: { controls: [{ field: 'model', value: 'fixture-model' }] }, actions });
  assert.equal(result.passed, false);
});

// ---------------------------------------------------------------------------
// P7 Chunk 1 (docs/helm/2026-10-13-exploration-safety-and-coverage/phases/07-workbench-observer-parity.md):
// the workbench observer must report a <summary>'s parent <details> open state as `expanded`, and name
// links and summaries by their own text before their `title` tooltip, mirroring the Obsidian observer
// after d600a2e. The real browserObservation (exploration-browser.mjs, not exported) runs here against a
// happy-dom document: a fake page's evaluate() hands its own source to a vm context whose globals are that
// document, through the real driver.observe() path. happy-dom's always-zero getBoundingClientRect is
// replaced so the visibility gate lets the fixture's controls through.
// ---------------------------------------------------------------------------

async function observeWorkbenchFixture(t, bodyHtml) {
  const root = await mkdtemp(path.join(tmpdir(), 'exploration-observer-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const fixture = { root, vaultRoot: path.join(root, 'vault'), configPath: path.join(root, 'config.toml') };
  await mkdir(fixture.vaultRoot);
  await writeFile(fixture.configPath, `vault_root = ${JSON.stringify(fixture.vaultRoot)}\n`);
  const url = 'http://127.0.0.1:3102/session/';
  const window = new Window();
  window.document.body.innerHTML = `<div class="workbench" data-view="settings">${bodyHtml}</div>`;
  window.Element.prototype.getBoundingClientRect = () => ({ x: 0, y: 0, width: 100, height: 20, top: 0, left: 0, right: 100, bottom: 20 });
  const context = { document: window.document, getComputedStyle: (...args) => window.getComputedStyle(...args), innerWidth: 1200, innerHeight: 800 };
  const page = {
    url: () => url,
    on() {}, off() {},
    locator: () => ({ waitFor: async () => {} }),
    async evaluate(fn, arg) { return JSON.parse(JSON.stringify(vm.runInNewContext(`(${fn.toString()})`, context)(arg))); },
    keyboard: { press: async () => {} },
  };
  const browserContext = { on() {}, off() {}, route: async () => {}, unroute: async () => {} };
  const driver = await createExplorationDriver({ page, context: browserContext, url, fixture, artifactDir: path.join(root, 'artifacts') });
  t.after(() => driver.dispose());
  return driver.observe();
}

test("P7 workbench observer: a <summary> inside an open <details> is reported expanded: true", async t => {
  const observation = await observeWorkbenchFixture(t, '<details open><summary>Advanced</summary><p>body</p></details>');
  const summary = observation.controls.find(control => control.tag === 'summary');
  assert.ok(summary, 'the summary must be observed as a control');
  assert.equal(summary.expanded, true);
});

test("P7 workbench observer: a <summary> inside a closed <details> is reported expanded: false", async t => {
  const observation = await observeWorkbenchFixture(t, '<details><summary>Advanced</summary><p>body</p></details>');
  const summary = observation.controls.find(control => control.tag === 'summary');
  assert.ok(summary, 'the summary must be observed as a control');
  assert.equal(summary.expanded, false);
});

test("P7 workbench observer: a titled link is named by its own text, not its title tooltip", async t => {
  const observation = await observeWorkbenchFixture(t, '<a href="#report" title="Tooltip for the report">Full report</a>');
  const link = observation.controls.find(control => control.tag === 'a');
  assert.ok(link, 'the link must be observed as a control');
  assert.equal(link.name, 'Full report');
});

test("P7 workbench observer: a titled summary is named by its own text, not its title tooltip", async t => {
  const observation = await observeWorkbenchFixture(t, '<details><summary title="Tooltip for the section">Advanced options</summary><p>body</p></details>');
  const summary = observation.controls.find(control => control.tag === 'summary');
  assert.ok(summary, 'the summary must be observed as a control');
  assert.equal(summary.name, 'Advanced options');
});

test("P7 workbench observer guard: aria-label still wins over own text and title for a button and a link", async t => {
  const observation = await observeWorkbenchFixture(t, '<button aria-label="Explicit button label" title="Tooltip">Own button text</button><a href="#x" aria-label="Explicit link label" title="Tooltip">Own link text</a>');
  assert.equal(observation.controls.find(control => control.tag === 'button').name, 'Explicit button label');
  assert.equal(observation.controls.find(control => control.tag === 'a').name, 'Explicit link label');
});

test("P7 workbench observer guard: a textless titled button still falls back to its title", async t => {
  const observation = await observeWorkbenchFixture(t, '<button title="Only a tooltip"></button>');
  const button = observation.controls.find(control => control.tag === 'button');
  assert.ok(button, 'the button must be observed as a control');
  assert.equal(button.name, 'Only a tooltip');
});

// P7 Chunk 1 L1: the workbench settings rows that hold an external link ("Report a bug" and similar) render
// a plain "Open" link. A link keeps its settings-row name before its own text, so those rows stay
// distinguishable; its own text still wins over a title tooltip (test above).
test("P7 workbench observer: a link inside a settings row keeps the row name instead of its generic own text", async t => {
  const observation = await observeWorkbenchFixture(t, '<div data-setting-name="Report a bug"><a href="https://example.invalid/issues">Open</a></div><div data-setting-name="Documentation"><a href="https://example.invalid/docs">Open</a></div>');
  assert.deepEqual(observation.controls.filter(control => control.tag === 'a').map(control => control.name), ['Report a bug', 'Documentation']);
});

for (const key of ['Enter', 'Space']) test(`setup scheduler activation blocks ${key}`, async t => {
  const { driver, page, observation } = await makePressDriver(t, { focus: { ...notFocused, tag: 'button', name: '开启自动日报', dataAction: 'enable' } });
  await assert.rejects(driver.execute({ type: 'press', key }, observation), /ExplorationActionError/);
  assert.ok(!page.calls.some(call => call[0] === 'press'));
});
test('setup scheduler observation preserves its action and restriction', async t => {
  const observation = await observeWorkbenchFixture(t, '<button data-setup-action="enable">开启自动日报</button>');
  assert.equal(observation.controls[0].dataAction, 'enable');
  assert.match(classifyControlRestriction(observation.controls[0]), /scheduling/);
});
