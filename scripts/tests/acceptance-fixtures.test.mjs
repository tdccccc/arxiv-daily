import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile, stat, access } from "node:fs/promises";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { ACCEPTANCE_DATE, PAPERS, createFixtureEnvironment, startFixtureServer } from "../acceptance/fixtures.mjs";

async function serverFor(t) {
  const server = await startFixtureServer();
  t.after(() => server.close());
  return server;
}
function proxy(server, url, init) {
  return fetch(`${server.url}/proxy?url=${encodeURIComponent(url)}`, init);
}
const filterRequest = {
  model: "fixture-model-2", stream: false,
  messages: [{ role: "system", content: "为所有命中论文评分\n- new-topic:\n  - new-topic#1: Changed research direction" },
    { role: "user", content: PAPERS.map(p => `ID: ${p.id}\nTitle: ${p.title}`).join("\n") }],
};

test("fixture serves the real arXiv and model wire formats and records effective settings", async t => {
  const server = await serverFor(t);
  const recent = await proxy(server, "https://arxiv.org/list/cs.AI/recent?skip=0&show=2000");
  assert.equal(recent.status, 200);
  assert.match(await recent.text(), /Thu, 1 Oct 2026/);
  const atom = await proxy(server, `https://export.arxiv.org/api/query?id_list=${PAPERS[0].id}`);
  assert.match(await atom.text(), new RegExp(PAPERS[0].id));
  const response = await proxy(server, "https://fixture-model.test/v1/chat/completions", {
    method: "POST", headers: { authorization: "Bearer never-log-this", "content-type": "application/json" }, body: JSON.stringify(filterRequest),
  });
  const answer = await response.json();
  assert.equal(answer.choices[0].finish_reason, "stop");
  const parsed = JSON.parse(answer.choices[0].message.content);
  assert.deepEqual(parsed.papers.map(p => p.id), PAPERS.map(p => p.id));
  assert.equal(parsed.papers[0].category, "new-topic");
  assert.deepEqual(parsed.papers[0].directions, ["new-topic#1"]);
  const recorded = server.requests.find(r => r.kind === "filter");
  assert.equal(recorded.model, "fixture-model-2");
  assert.match(recorded.prompt, /Changed research direction/);
  assert.doesNotMatch(JSON.stringify(server.requests), /never-log-this/);
});

test("unknown external requests fail locally and are visible in evidence", async t => {
  const server = await serverFor(t);
  const response = await proxy(server, "https://unexpected.invalid/no-real-network");
  assert.equal(response.status, 502);
  assert.match(await response.text(), /Unexpected fixture request/);
  assert.equal(server.requests.at(-1).kind, "unexpected");
});

test("fixture modes deterministically produce errors and release held model calls", async t => {
  const server = await serverFor(t);
  server.setMode({ llm: "unauthorized" });
  const request = () => proxy(server, "https://fixture-model.test/v1/chat/completions", { method: "POST", body: JSON.stringify(filterRequest) });
  assert.equal((await request()).status, 401);
  server.setMode({ llm: "hold" });
  let settled = false;
  const pending = request().then(response => { settled = true; return response; });
  await new Promise(resolve => setTimeout(resolve, 40));
  assert.equal(settled, false);
  server.releaseHeld();
  assert.equal((await pending).status, 200);
  server.setMode({ arxiv: "empty" });
  const empty = await proxy(server, "https://arxiv.org/list/cs.AI/recent");
  assert.doesNotMatch(await empty.text(), /Title:/);
});

test("aborted model requests do not leave held connections or prevent cleanup", async t => {
  const server = await serverFor(t);
  server.setMode({ llm: "hold" });
  const controller = new AbortController();
  const pending = proxy(server, "https://fixture-model.test/v1/chat/completions", {
    method: "POST", body: JSON.stringify(filterRequest), signal: controller.signal,
  });
  await new Promise(resolve => setTimeout(resolve, 40));
  controller.abort();
  await assert.rejects(pending, { name: "AbortError" });
  await server.close();
});

test("fixture environment owns config, data and PDFs and transports child requests without real network", async t => {
  const fixture = await createFixtureEnvironment();
  t.after(() => fixture.dispose());
  assert.ok(fixture.vaultRoot.startsWith(fixture.root + "/"));
  assert.ok(fixture.configPath.startsWith(fixture.configHome + "/"));
  const config = await readFile(fixture.configPath, "utf8");
  assert.ok(config.includes(fixture.vaultRoot));
  assert.match(config, /fixture-model/);
  assert.match(config, /enabled = false/);
  const pdf = await readFile(join(fixture.libraryRoot, `${PAPERS[0].id}.pdf`));
  assert.ok(pdf.subarray(0, 8).toString().startsWith("%PDF-1."));
  assert.match(pdf.toString(), /\/Count 4/);
  assert.ok((await stat(join(fixture.vaultRoot, ".obsidian"))).isDirectory());
  const output = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["-e", "fetch('https://fixture-model.test/v1/models').then(r=>r.json()).then(j=>console.log(JSON.stringify(j)))"], { env: fixture.env, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.on("data", data => { stdout += data; }); child.stderr.on("data", data => { stderr += data; });
    child.once("error", reject); child.once("exit", code => code === 0 ? resolve(stdout) : reject(new Error(stderr)));
  });
  assert.match(output, /fixture-model-2/);
  assert.equal(fixture.server.requests.at(-1).kind, "models");
  await fixture.dispose();
  await assert.rejects(access(fixture.root), { code: "ENOENT" });
});

test("unconfigured fixture leaves first-run configuration absent", async t => {
  const fixture = await createFixtureEnvironment({ configured: false });
  t.after(() => fixture.dispose());
  await assert.rejects(access(fixture.configPath), { code: "ENOENT" });
  assert.equal(ACCEPTANCE_DATE, "2026-10-01");
});
