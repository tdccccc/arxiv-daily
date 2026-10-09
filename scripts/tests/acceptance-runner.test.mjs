import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { parseArgs, runAcceptance } from "../acceptance/run.mjs";

function passed(suite) { return { suite, status: "passed", scenarios: [{ id: `${suite}.task`, title: "User task", status: "passed", assertions: [{ label: "Observed expected state", passed: true }] }] }; }
async function dependencies(t, overrides = {}) {
  const root = await mkdtemp(join(tmpdir(), "acceptance-runner-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const events = [], reports = [];
  let id = 0;
  const base = {
    createFixture: async () => { const n = ++id; events.push(`create:${n}`); return { root: join(root, String(n)), env: { XDG_CONFIG_HOME: `fixture-${n}` }, dispose: async () => events.push(`dispose:${n}`) }; },
    build: async suites => { events.push(`build:${suites.join(",")}`); },
    runners: Object.fromEntries(["workbench", "obsidian", "exploration"].map(name => [name, async ({ fixture }) => { events.push(`run:${name}:${fixture.env.XDG_CONFIG_HOME}`); return passed(name); }])),
    writeReport: async (_directory, report) => { reports.push(report); },
    ...overrides,
  };
  return { deps: base, events, reports, output: join(root, "report") };
}

test("ordinary acceptance selects both products without model exploration", () => {
  const options = parseArgs([]);
  assert.deepEqual(options.suites, ["workbench", "obsidian"]);
  assert.equal(options.build, true);
  assert.equal(options.suites.includes("exploration"), false);
});

test("command explicitly selects suites and bounds optional model exploration", () => {
  const options = parseArgs(["--suite", "workbench", "--explore", "--max-steps", "12", "--max-api-calls=15", "--max-duration-ms", "60000", "--model-config", "/tmp/model.toml", "--skip-build"]);
  assert.deepEqual(options.suites, ["workbench", "exploration"]);
  assert.equal(options.maxSteps, 12);
  assert.equal(options.maxApiCalls, 15);
  assert.equal(options.maxDurationMs, 60000);
  assert.equal(options.configPath, "/tmp/model.toml");
  assert.equal(options.build, false);
  assert.throws(() => parseArgs(["--suite", "typo"]), /suite/i);
  assert.throws(() => parseArgs(["--max-steps", "0"]), /max-steps/);
  assert.throws(() => parseArgs(["--what"]), /Unknown/);
});

test("each suite receives its own fixture and cleanup runs despite a failed suite", async t => {
  const f = await dependencies(t);
  f.deps.runners.workbench = async () => { f.events.push("fail:workbench"); throw new Error("Product assertion failed"); };
  const report = await runAcceptance({ ...parseArgs([]), output: f.output }, f.deps);
  assert.equal(report.status, "failed");
  assert.deepEqual(f.events, ["build:workbench,obsidian", "create:1", "fail:workbench", "dispose:1", "create:2", "run:obsidian:fixture-2", "dispose:2"]);
  assert.equal(report.suites.find(s => s.suite === "obsidian").status, "passed");
  assert.ok(f.reports.length >= 1);
});

test("model config stays separate from the product fixture and model only runs when selected", async t => {
  const f = await dependencies(t);
  f.deps.runners.exploration = async options => {
    assert.equal(options.configPath, "/tmp/real-model-config.toml");
    assert.equal(options.fixture.env.XDG_CONFIG_HOME, "fixture-2");
    assert.equal(options.maxApiCalls, 6);
    return passed("exploration");
  };
  const options = parseArgs(["--suite", "workbench", "--explore", "--model-config", "/tmp/real-model-config.toml", "--max-api-calls", "6"]);
  const report = await runAcceptance({ ...options, output: f.output }, f.deps);
  assert.equal(report.status, "passed");
  assert.deepEqual(report.selectedSuites, ["workbench", "exploration"]);
});

test("build failure prevents product runs and writes a failure report", async t => {
  const f = await dependencies(t, { build: async () => { throw new Error("Build failed"); } });
  const report = await runAcceptance({ ...parseArgs([]), output: f.output }, f.deps);
  assert.equal(report.exitCode, 1);
  assert.equal(f.events.length, 0);
  assert.ok(f.reports.at(-1).suites.every(s => s.status === "failed"));
});

test("cleanup failure cannot be hidden behind successful UI assertions", async t => {
  const f = await dependencies(t, { createFixture: async () => ({ env: {}, dispose: async () => { throw new Error("Owned fixture could not be cleaned"); } }) });
  const report = await runAcceptance({ ...parseArgs(["--suite", "workbench", "--skip-build"]), output: f.output }, f.deps);
  assert.equal(report.status, "failed");
  assert.match(JSON.stringify(report), /could not be cleaned/);
});

test("aborted acceptance never starts further suites and reports them not-run", async t => {
  const controller = new AbortController();
  controller.abort();
  const f = await dependencies(t);
  const report = await runAcceptance({ ...parseArgs([]), output: f.output, signal: controller.signal }, f.deps);
  assert.equal(report.status, "blocked");
  assert.deepEqual(f.events, []);
  assert.ok(report.suites.every(s => s.status === "not-run"));
});

test("a thrown suite preserves request evidence before its fixture is removed", async t => {
  const f = await dependencies(t, { createFixture: async () => ({ env: {}, server: { requests: [{ kind: "model", status: 401 }] }, dispose: async () => {} }) });
  f.deps.runners.workbench = async () => { throw new Error("Could not inspect error state"); };
  const report = await runAcceptance({ ...parseArgs(["--suite", "workbench", "--skip-build"]), output: f.output }, f.deps);
  const artifact = report.suites[0].artifacts?.find(path => path.endsWith("fixture-requests.json"));
  assert.ok(artifact, "Request log must survive a thrown UI scenario");
  assert.equal(JSON.parse(await readFile(artifact, "utf8"))[0].status, 401);
});

test("build failures link their logs from the saved report", async t => {
  const error = Object.assign(new Error("CLI build failed"), { artifacts: ["build-apps-cli.log"] });
  const f = await dependencies(t, { build: async () => { throw error; } });
  const report = await runAcceptance({ ...parseArgs(["--suite", "workbench"]), output: f.output }, f.deps);
  assert.deepEqual(report.suites[0].artifacts, ["build-apps-cli.log"]);
});
