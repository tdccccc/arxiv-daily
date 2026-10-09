import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { classifyControlRestriction, createDefaultExplorationTasks, createExplorationDriver } from "../acceptance/exploration-browser.mjs";
import { loadExplorationModel, parseExplorationAction, runExplorationLoop } from "../acceptance/exploration.mjs";

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

async function temporary(t) {
  const root = await mkdtemp(path.join(tmpdir(), "acceptance-exploration-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test("an unsupported done claim is blocked and remaining tasks are not run", async t => {
  const driver = fixtureDriver();
  const result = await runExplorationLoop({ driver, model: async () => '{"type":"done","reason":"Everything works"}',
    tasks: [task(), { ...task(), id: "reading" }], artifactDir: await temporary(t), maxSteps: 5 });
  assert.equal(result.status, "blocked");
  assert.deepEqual(result.scenarios.map(value => value.status), ["blocked", "not-run"]);
  assert.match(result.scenarios[0].error, /evidence|判据/i);
  assert.equal(driver.executed.length, 0);
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
