import assert from "node:assert/strict";
import { access, mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { startWorkbenchSession } from "../acceptance/workbench-session.mjs";

async function environment(t, source) {
  const root = await mkdtemp(join(tmpdir(), "acceptance-workbench-contract-"));
  const cleanups = [];
  t.after(async () => { for (const cleanup of cleanups.reverse()) await cleanup(); await rm(root, { recursive: true, force: true }); });
  const artifactDir = join(root, "evidence");
  const cliPath = join(root, "fake-cli.cjs");
  const configHome = join(root, "config");
  await mkdir(configHome);
  if (source) await writeFile(cliPath, source);
  const fixture = { root, configHome, env: { ...process.env, XDG_CONFIG_HOME: configHome, ACCEPTANCE_PID_FILE: join(root, "pid") } };
  return { fixture, artifactDir, cliPath, onCleanup: action => cleanups.push(action) };
}

function browserDouble() {
  const calls = { urls: [], closed: false, tracing: false, route: null };
  const page = { on() {}, setDefaultTimeout() {}, setDefaultNavigationTimeout() {}, async goto(url) { calls.urls.push(url); }, isClosed: () => calls.closed };
  const context = {
    on() {}, async newPage() { return page; }, async route(_glob, callback) { calls.route = callback; },
    tracing: { async start() { calls.tracing = true; }, async stop({ path }) { await writeFile(path, "trace evidence"); } },
    async close() { calls.closed = true; },
  };
  const browser = { async newContext() { return context; }, async close() { calls.closed = true; } };
  return { calls, launchBrowser: async () => browser };
}

const realServer = `
const fs = require('node:fs');
const http = require('node:http');
fs.writeFileSync(process.env.ACCEPTANCE_PID_FILE, String(process.pid));
const server = http.createServer((_req, res) => res.end('fixture'));
server.listen(0, '127.0.0.1', () => {
  console.log('isolated config: ' + process.env.XDG_CONFIG_HOME);
  console.log('Workbench: http://127.0.0.1:' + server.address().port + '/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/');
});
process.on('SIGINT', () => server.close(() => process.exit(0)));
process.on('SIGTERM', () => server.close(() => process.exit(0)));
`;

test("missing CLI is blocked before launching a browser", async t => {
  const options = await environment(t);
  let launched = false;
  await assert.rejects(startWorkbenchSession({ ...options, launchBrowser: async () => { launched = true; } }), error => {
    assert.equal(error.code, "WORKBENCH_BLOCKED");
    assert.match(error.message, /build|构建/i);
    return true;
  });
  assert.equal(launched, false);
});

test("session starts the real child with isolated config, preserves launch URL and closes idempotently", async t => {
  const options = await environment(t, realServer);
  const browser = browserDouble();
  const session = await startWorkbenchSession({ ...options, launchBrowser: browser.launchBrowser });
  options.onCleanup(() => session.stop());
  assert.match(session.url, /^http:\/\/127\.0\.0\.1:\d+\/a{48}\/$/);
  assert.deepEqual(browser.calls.urls, [session.url]);
  assert.equal(browser.calls.tracing, true);
  assert.ok(session.page && session.context && session.browser);
  const pid = Number(await readFile(options.fixture.env.ACCEPTANCE_PID_FILE, "utf8"));
  assert.doesNotThrow(() => process.kill(pid, 0));
  await session.stop();
  await session.stop();
  assert.equal(browser.calls.closed, true);
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
  assert.match(await readFile(join(options.artifactDir, "workbench.stdout.log"), "utf8"), new RegExp(options.fixture.configHome));
  await access(join(options.artifactDir, "workbench.trace.zip"));
  await access(options.fixture.root);
});

test("a foreign launch URL fails closed and cleans up its child before rejection", async t => {
  const options = await environment(t, `
    require('node:fs').writeFileSync(process.env.ACCEPTANCE_PID_FILE, String(process.pid));
    console.log('Workbench: https://example.invalid/');
    setInterval(() => {}, 1000);
  `);
  let launched = false;
  await assert.rejects(startWorkbenchSession({ ...options, startupTimeoutMs: 1000, launchBrowser: async () => { launched = true; } }), /loopback|本机|工作台链接|launch URL/i);
  assert.equal(launched, false);
  const pid = Number(await readFile(options.fixture.env.ACCEPTANCE_PID_FILE, "utf8"));
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
  assert.match(await readFile(join(options.artifactDir, "workbench.stdout.log"), "utf8"), /example\.invalid/);
});

test("a child that never announces readiness times out and is not left running", async t => {
  const options = await environment(t, `
    require('node:fs').writeFileSync(process.env.ACCEPTANCE_PID_FILE, String(process.pid));
    console.log('starting');
    setInterval(() => {}, 1000);
  `);
  await assert.rejects(startWorkbenchSession({ ...options, startupTimeoutMs: 250, launchBrowser: browserDouble().launchBrowser }), /timed out|超时/i);
  const pid = Number(await readFile(options.fixture.env.ACCEPTANCE_PID_FILE, "utf8"));
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
});

test("browser startup failure is blocked, retaining child output and stopping the service", async t => {
  const options = await environment(t, realServer);
  await assert.rejects(startWorkbenchSession({ ...options, launchBrowser: async () => { throw new Error("browser binary unavailable"); } }), error => {
    assert.equal(error.code, "WORKBENCH_BLOCKED");
    assert.match(error.message, /browser binary unavailable/);
    assert.ok(error.artifacts?.includes(join(options.artifactDir, "workbench.stdout.log")), "startup failures must link retained CLI diagnostics");
    return true;
  });
  const pid = Number(await readFile(options.fixture.env.ACCEPTANCE_PID_FILE, "utf8"));
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
  assert.match(await readFile(join(options.artifactDir, "workbench.stdout.log"), "utf8"), /Workbench:/);
});

test("a pre-aborted acceptance signal cannot start the CLI or browser", async t => {
  const options = await environment(t);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(startWorkbenchSession({ ...options, signal: controller.signal, launchBrowser: browserDouble().launchBrowser }), { name: "AbortError" });
});

test("aborting acceptance stops its owned browser and CLI", async t => {
  const options = await environment(t, realServer), browser = browserDouble();
  const controller = new AbortController();
  const session = await startWorkbenchSession({ ...options, signal: controller.signal, launchBrowser: browser.launchBrowser });
  options.onCleanup(() => session.stop());
  const pid = Number(await readFile(options.fixture.env.ACCEPTANCE_PID_FILE, "utf8"));
  controller.abort();
  const deadline = Date.now() + 1000;
  while (!browser.calls.closed && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(browser.calls.closed, true, "abort must close the owned browser without waiting for fixture disposal");
  await session.stop();
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
});

test("the browser cannot navigate or fetch outside the owned workbench origin", async t => {
  const options = await environment(t, realServer), browser = browserDouble();
  const session = await startWorkbenchSession({ ...options, launchBrowser: browser.launchBrowser });
  options.onCleanup(() => session.stop());
  const decisions = [];
  const route = url => ({
    request: () => ({ url: () => url }),
    continue: async () => decisions.push("continue"),
    abort: async reason => decisions.push(reason),
  });
  await browser.calls.route(route(new URL("api/status", session.url).href));
  await browser.calls.route(route("https://example.invalid/private"));
  await browser.calls.route(route("http://127.0.0.1:1/unrelated-service"));
  assert.deepEqual(decisions, ["continue", "blockedbyclient", "blockedbyclient"]);
  assert.equal(session.diagnostics.blockedRequests.length, 2);
});

test("an unavailable isolated session produces blocked evidence and does not claim unrun journeys passed", async t => {
  const { runWorkbenchAcceptance, WORKBENCH_SCENARIOS } = await import("../acceptance/workbench.mjs");
  const options = await environment(t);
  const suite = await runWorkbenchAcceptance({
    ...options,
    fixture: { ...options.fixture, env: {}, server: { requests: [], releaseHeld() {} } },
  });
  assert.equal(suite.status, "blocked");
  assert.equal(suite.scenarios.length, WORKBENCH_SCENARIOS.length);
  assert.equal(suite.scenarios[0].status, "blocked");
  assert.ok(suite.scenarios.slice(1).every(scenario => scenario.status === "not-run"));
  assert.ok(suite.scenarios.every(scenario => scenario.status !== "passed"));
});

test("an aborted suite marks every remaining journey not-run", async t => {
  const { runWorkbenchAcceptance, WORKBENCH_SCENARIOS } = await import("../acceptance/workbench.mjs");
  const options = await environment(t);
  const controller = new AbortController(); controller.abort();
  const suite = await runWorkbenchAcceptance({
    ...options, signal: controller.signal,
    fixture: { ...options.fixture, server: { requests: [], releaseHeld() {} } },
  });
  assert.equal(suite.status, "blocked");
  assert.equal(suite.scenarios.length, WORKBENCH_SCENARIOS.length);
  assert.ok(suite.scenarios.every(scenario => scenario.status === "not-run"));
});

test("console-error classifier allows the settings-save conflict text only for the scenarios that deliberately provoke it", async () => {
  const { classifyWorkbenchConsoleErrors } = await import("../acceptance/workbench.mjs");
  const conflict = "Failed to load resource: the server responded with a status of 409 (Conflict)";
  for (const scenarioId of ["workbench.cancel-retry", "workbench.settings-conflict"]) {
    assert.deepEqual(classifyWorkbenchConsoleErrors([conflict], { scenarioId }), { expected: [conflict], unexpected: [] });
  }
});

test("console-error classifier does not swallow an unexpected error even inside a scenario with a known allowance", async () => {
  const { classifyWorkbenchConsoleErrors } = await import("../acceptance/workbench.mjs");
  const conflict = "Failed to load resource: the server responded with a status of 409 (Conflict)";
  const unrelated = "TypeError: Cannot read properties of undefined (reading 'foo')";
  const result = classifyWorkbenchConsoleErrors([conflict, unrelated], { scenarioId: "workbench.settings-conflict" });
  assert.deepEqual(result, { expected: [conflict], unexpected: [unrelated] });
});

test("console-error classifier treats the same conflict text as unexpected outside its known scenarios", async () => {
  const { classifyWorkbenchConsoleErrors } = await import("../acceptance/workbench.mjs");
  const conflict = "Failed to load resource: the server responded with a status of 409 (Conflict)";
  assert.deepEqual(classifyWorkbenchConsoleErrors([conflict], { scenarioId: "workbench.reading" }), { expected: [], unexpected: [conflict] });
  assert.deepEqual(classifyWorkbenchConsoleErrors([conflict]), { expected: [], unexpected: [conflict] });
});

test("completed daily outcome is acknowledged by its saved-report UI status", async t => {
  const { createServer } = await import("node:http");
  const { waitForWorkbenchRun } = await import("../acceptance/workbench.mjs");
  const run = { id: "saved-daily", status: "completed", outcome: "papers_written", exitCode: 0 };
  const server = createServer((_request, response) => {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ run }));
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const session = {
    url: "http://127.0.0.1:" + server.address().port + "/",
    page: { locator: () => ({
      innerText: async () => "日报已保存",
      getAttribute: async name => name === "class" ? "run-state completed" : null,
      isVisible: async () => true,
    }) },
  };
  assert.deepEqual(await waitForWorkbenchRun(session, { id: run.id }), run);
});
