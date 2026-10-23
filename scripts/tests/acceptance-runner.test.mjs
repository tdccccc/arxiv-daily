import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { parseArgs, runAcceptance, npmBuildCommand, workspacesToBuild } from "../acceptance/run.mjs";

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

test("--task and --persona accumulate across repeated and comma-separated flags; per-task budget flags parse", () => {
  const options = parseArgs(["--suite", "exploration", "--task", "a,b", "--task", "c", "--persona", "x", "--task-max-steps", "5", "--task-max-api-calls", "8"]);
  assert.deepEqual(options.taskIds, ["a", "b", "c"]);
  assert.deepEqual(options.personaIds, ["x"]);
  assert.equal(options.taskMaxSteps, 5);
  assert.equal(options.taskMaxApiCalls, 8);
  assert.throws(() => parseArgs(["--task", ""]), /--task/);
  assert.equal(parseArgs([]).taskIds, undefined);
});

test("--release is a value-less flag, off by default", () => {
  assert.equal(parseArgs([]).release, undefined);
  assert.equal(parseArgs(["--suite", "exploration", "--release"]).release, true);
  assert.throws(() => parseArgs(["--release=workbench"]), /--release/);
});

test("ordinary runs keep the small run-wide budget defaults", () => {
  const options = parseArgs(["--suite", "exploration"]);
  assert.equal(options.maxSteps, 90);
  assert.equal(options.maxDurationMs, 900000);
  assert.equal(options.maxApiCalls, undefined); // exploration.mjs falls back to 2x the effective step budget
});

test("--release without explicit budget flags raises the run-wide defaults with headroom over measured usage", () => {
  const options = parseArgs(["--suite", "exploration", "--host", "obsidian", "--release"]);
  // Measured real-model release runs (2026-10-12, docs/helm/2026-10-10-model-driven-release-exploration):
  // obsidian used up to 106 model calls / ~536s; workbench up to 80 calls / ~359s. Defaults below give
  // roughly 2x headroom over the larger (obsidian) figure while staying within the hard maxima
  // (maxSteps <= 300, maxDurationMs <= 3600000).
  assert.equal(options.maxSteps, 220);
  assert.equal(options.maxDurationMs, 1200000);
  assert.equal(options.maxApiCalls, undefined); // still falls back to 2x maxSteps = 440, well under the 1000 hard cap
});

test("--release still lets explicit budget flags win, regardless of flag order", () => {
  const before = parseArgs(["--suite", "exploration", "--max-steps", "50", "--max-duration-ms", "60000", "--max-api-calls", "70", "--release"]);
  assert.equal(before.maxSteps, 50);
  assert.equal(before.maxDurationMs, 60000);
  assert.equal(before.maxApiCalls, 70);
  const after = parseArgs(["--suite", "exploration", "--release", "--max-steps", "50", "--max-duration-ms", "60000", "--max-api-calls", "70"]);
  assert.equal(after.maxSteps, 50);
  assert.equal(after.maxDurationMs, 60000);
  assert.equal(after.maxApiCalls, 70);
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

test("task/persona selection and per-task budget options reach the exploration runner", async t => {
  const f = await dependencies(t);
  f.deps.runners.exploration = async options => {
    assert.deepEqual(options.taskIds, ["generate-and-read"]);
    assert.deepEqual(options.personaIds, ["new-user"]);
    assert.equal(options.taskMaxSteps, 10);
    assert.equal(options.taskMaxApiCalls, 20);
    assert.equal(options.release, true);
    return passed("exploration");
  };
  const options = parseArgs(["--suite", "workbench", "--explore", "--task", "generate-and-read", "--persona", "new-user", "--task-max-steps", "10", "--task-max-api-calls", "20", "--release"]);
  const report = await runAcceptance({ ...options, output: f.output }, f.deps);
  assert.equal(report.status, "passed");
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

test("Windows builds invoke npm through Node and never execute a cmd file directly", () => {
  assert.deepEqual(npmBuildCommand("win32", { npm_execpath: "C:/npm/npm-cli.js" }), { command: process.execPath, prefix: ["C:/npm/npm-cli.js"] });
  assert.throws(() => npmBuildCommand("win32", {}), /through npm/);
});

test("exploration build receives the selected host including mixed suites", async t => {
  for (const argv of [["--suite", "exploration", "--host", "obsidian"], ["--suite", "workbench", "--explore", "--host", "obsidian"]]) {
    let received;
    const f = await dependencies(t, { build: async (_suites, options) => { received = options.host; } });
    await runAcceptance({ ...parseArgs(argv), output: f.output }, f.deps);
    assert.equal(received, "obsidian");
  }
});

test("skip-build still records fingerprints of the exact existing artifacts", async t => {
  const f = await dependencies(t);
  const report = await runAcceptance({ ...parseArgs(["--suite", "workbench", "--skip-build"]), output: f.output }, f.deps);
  assert.ok(Array.isArray(report.metadata.buildArtifacts));
  assert.ok(report.metadata.buildArtifacts.some(item => item.path.endsWith("arxiv-daily-cli.cjs")));
  assert.ok(!f.events.some(event => event.startsWith("build:")));
});

test("build selection follows both fixed suites and exploration host", () => {
  assert.deepEqual(workspacesToBuild(["exploration"], "obsidian"), ["obsidian-arxiv-daily"]);
  assert.deepEqual(workspacesToBuild(["workbench", "exploration"], "obsidian"), ["apps/cli", "obsidian-arxiv-daily"]);
  assert.deepEqual(workspacesToBuild(["exploration"], "workbench"), ["apps/cli"]);
  assert.deepEqual(workspacesToBuild(["obsidian"], "workbench"), ["obsidian-arxiv-daily"]);
});
