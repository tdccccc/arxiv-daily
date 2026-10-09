import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { Window } from "happy-dom";
import { classifyObsidianDiagnostics, createObsidianSettings, installHttpProxyExpression, runObsidianAcceptance } from "../acceptance/obsidian.mjs";
import { dateSubmitTargetExpression, pollJsonFile, waitForStableTarget } from "../acceptance/obsidian-ui.mjs";

function dateModalFixture() {
  const window = new Window();
  window.document.body.innerHTML = '<div class="modal closing-settings"><button class="mod-cta">Unrelated settings action</button></div><div class="modal date-dialog"><input type="date"><button class="mod-cta">Run</button></div>';
  const previous = window.document.querySelector(".closing-settings button");
  const run = window.document.querySelector(".date-dialog button");
  previous.getBoundingClientRect = () => ({ x: 5, y: 5, width: 100, height: 40 });
  run.getBoundingClientRect = () => ({ x: 350, y: 240, width: 100, height: 40 });
  window.document.elementFromPoint = () => run;
  window.document.querySelector("input").value = "2026-10-01";
  return { window, run };
}

test("date submission targets the date dialog while a closing settings dialog still has a CTA", () => {
  const { window } = dateModalFixture();
  const observed = vm.runInNewContext(dateSubmitTargetExpression("2026-10-01"), { document: window.document });
  assert.equal(observed.ready, true);
  assert.equal(observed.label, "Run");
  assert.equal(observed.x, 400);
  assert.equal(observed.y, 260);
});

test("date submission waits for the submitted date to be validated and the Run control to be enabled", () => {
  const { window, run } = dateModalFixture();
  const observe = date => vm.runInNewContext(dateSubmitTargetExpression(date), { document: window.document });
  assert.equal(observe("2026-10-02").ready, false);
  run.disabled = true;
  assert.equal(observe("2026-10-01").ready, false);
});

test("a visible date button behind another element is not a safe pointer target", () => {
  const { window } = dateModalFixture();
  window.document.elementFromPoint = () => window.document.querySelector(".closing-settings button");
  const observed = vm.runInNewContext(dateSubmitTargetExpression("2026-10-01"), { document: window.document });
  assert.equal(observed.hitTarget, false);
});

test("pointer submission waits for consecutive stable geometry and confirmed hits", async () => {
  const target = (x, hitTarget = true) => ({ ready: true, hitTarget, x, y: 400, width: 100, height: 30 });
  const samples = [target(450, false), target(450), target(460), target(470), target(470), target(470)];
  let calls = 0;
  const result = await waitForStableTarget(async () => samples[Math.min(calls++, samples.length - 1)], "date submit", { timeoutMs: 1000, intervalMs: 1, stableSamples: 3 });
  assert.equal(result.x, 470);
  assert.equal(calls, 6);
});

test("a persistently covered target fails with the last hit-test evidence", async () => {
  await assert.rejects(waitForStableTarget(async () => ({ ready: true, hitTarget: false, x: 400, y: 300, width: 100, height: 30, hit: { text: "overlay" } }), "date submit", { timeoutMs: 15, intervalMs: 1 }), /Timed out.*overlay/);
});

test("settings readiness tolerates only transient missing or partially written JSON", async () => {
  const samples = [Object.assign(new Error("not yet created"), { code: "ENOENT" }), "", '{"settings":', '{"settings":{"model":"old"}}', '{"settings":{"model":"new"}}'];
  let reads = 0;
  const result = await pollJsonFile(async () => {
    const sample = samples[reads++];
    if (sample instanceof Error) throw sample;
    return JSON.parse(sample);
  }, data => data.settings.model === "new", "saved settings", { timeoutMs: 1000, intervalMs: 1 });
  assert.equal(result.settings.model, "new");
  assert.equal(reads, 5);
});

test("persistently corrupt settings fail within the deadline and retain the last parse error", async () => {
  await assert.rejects(pollJsonFile(async () => JSON.parse('{"settings":'), () => true, "saved settings", { timeoutMs: 15, intervalMs: 1 }), /Timed out waiting for saved settings; last observation:.*SyntaxError/);
});

test("settings readiness does not retry a permission error or a broken acceptance predicate", async () => {
  let reads = 0;
  const denied = Object.assign(new Error("permission denied"), { code: "EACCES" });
  await assert.rejects(pollJsonFile(async () => { reads++; throw denied; }, () => true, "saved settings"), error => error === denied);
  assert.equal(reads, 1);
  const badPredicate = new SyntaxError("assertion code is broken");
  await assert.rejects(pollJsonFile(async () => ({}), () => { throw badPredicate; }, "saved settings"), error => error === badPredicate);
});

test("Obsidian settings keep scheduled work and email inert in a fresh vault", () => {
  const data = createObsidianSettings({ endpoint: "http://127.0.0.1:41234/v1" });
  assert.equal(data.settings.schedule.enabled, false);
  assert.equal(data.settings.email.enabled, false);
  assert.equal(data.settings.llm.baseUrl, "http://127.0.0.1:41234/v1");
  assert.equal(data.settings.llm.model, "fixture-model");
  assert.equal(data.settings.arxiv.topics[0].tag, "research");
  assert.deepEqual(data.settings.arxiv.categories, ["cs.AI"]);
  assert.equal(data.settings.onboarding.guideCompleted, false);
});

test("the transport wrapper routes every request to the owned service and preserves cancellation", async () => {
  const requests = [];
  const controller = new AbortController();
  const http = { request(request) { requests.push(request); return Promise.resolve({ status: 200 }); } };
  const context = vm.createContext({ URL, app: { plugins: { plugins: { "arxiv-daily": { getHttpClient: () => http } } } } });
  const expression = installHttpProxyExpression("http://127.0.0.1:41234");
  vm.runInContext(expression, context);
  vm.runInContext(expression, context);
  await http.request({ url: "https://arxiv.org/list/cs.AI/recent?skip=0&show=2000", method: "GET", signal: controller.signal, timeoutMs: 1200 });
  await http.request({ url: "http://127.0.0.1:41234/v1/chat/completions", method: "POST", body: '{"model":"fixture-model"}', headers: { "content-type": "application/json" } });
  assert.equal(requests.length, 2);
  const redirected = new URL(requests[0].url);
  assert.equal(redirected.origin, "http://127.0.0.1:41234");
  assert.equal(redirected.pathname, "/proxy");
  assert.equal(redirected.searchParams.get("url"), "https://arxiv.org/list/cs.AI/recent?skip=0&show=2000");
  assert.equal(requests[0].signal, controller.signal);
  assert.equal(requests[0].timeoutMs, 1200);
  assert.equal(requests[1].body, '{"model":"fixture-model"}');
  assert.equal(requests[1].headers["content-type"], "application/json");
});

test("the proxy seam refuses an endpoint outside loopback", () => {
  assert.throws(() => installHttpProxyExpression("https://example.com"), /loopback/i);
});

test("observed fixture 401 responses allow the real controlled message without a numeric status", () => {
  const expected = { source: "console", level: "error", text: "[arxiv-daily] paper-filter: LLM call failed Error: Controlled acceptance authentication failure" };
  const permanent = { source: "console", level: "error", text: "[arxiv-daily] arXiv 2026-10-01 permanent: paper filter LLM failed: Controlled acceptance authentication failure" };
  const unrelated = { source: "console", level: "error", text: "settings: write failed" };
  const exception = { source: "pageerror", level: "error", text: "Controlled acceptance authentication failure" };
  const result = classifyObsidianDiagnostics([expected, permanent, unrelated, exception], {
    authorizationFailureExpected: true,
    fixtureRequests: [{ method: "POST", kind: "model", status: 401 }],
  });
  assert.deepEqual(result.expected, [expected, permanent]);
  assert.deepEqual(result.unexpected, [unrelated, exception]);
  assert.deepEqual(classifyObsidianDiagnostics([expected]).unexpected, [expected]);
});

test("controlled authentication text is not ignored without this scenario's model 401 evidence", () => {
  const error = { source: "console", level: "error", text: "Controlled acceptance authentication failure" };
  for (const fixtureRequests of [[], [{ method: "POST", kind: "model", status: 200 }], [{ method: "GET", kind: "recent", status: 401 }]]) {
    assert.deepEqual(classifyObsidianDiagnostics([error], { authorizationFailureExpected: true, fixtureRequests }).unexpected, [error]);
  }
  assert.deepEqual(classifyObsidianDiagnostics([error], { fixtureRequests: [{ method: "POST", kind: "model", status: 401 }] }).unexpected, [error]);
});

test("a missing desktop environment produces blocked outcomes and never starts Obsidian", async () => {
  let launched = false;
  const result = await runObsidianAcceptance({ fixture: {}, artifactDir: "/tmp/unused-obsidian-contract" }, {
    preflight: async () => ({ ok: false, blockers: [{ message: "Obsidian missing", remedy: "Install Obsidian" }] }),
    runDesktopSession: async () => { launched = true; },
  });
  assert.equal(launched, false);
  assert.equal(result.status, "blocked");
  assert.ok(result.scenarios.length > 0);
  assert.ok(result.scenarios.every((scenario) => scenario.status === "blocked"));
  assert.match(JSON.stringify(result), /Obsidian missing/);
});
