import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BlockedError, runScenario, buildReport, writeReport } from "../acceptance/report.mjs";

test("scenario records user steps, assertions and failure captures", async () => {
  const captured = [];
  const result = await runScenario({ id: "browser.save", title: "Save settings", captureFailure: async () => { captured.push(true); return "failure.png"; } }, async t => {
    t.step("Edit model and reopen settings");
    t.check(true, "Setting remains visible");
    t.check(false, "Stored settings match the screen", "old value");
  });
  assert.equal(result.status, "failed");
  assert.equal(result.assertions.length, 2);
  assert.equal(result.steps.length, 1);
  assert.deepEqual(result.artifacts, ["failure.png"]);
  assert.equal(captured.length, 1);
});

test("zero assertions cannot turn a completed callback into a passing scenario", async () => {
  const result = await runScenario({ id: "empty", title: "Empty" }, async t => t.step("Opened page"));
  assert.equal(result.status, "failed");
  assert.match(result.error, /assertion/i);
});

test("environment blockers remain different from failed assertions", async () => {
  const result = await runScenario({ id: "unavailable", title: "Missing browser" }, async () => { throw new BlockedError("Browser is missing"); });
  assert.equal(result.status, "blocked");
  assert.match(result.error, /Browser is missing/);
});

test("selected suites missing from the evidence are reported not-run and cannot pass", () => {
  const report = buildReport({ selectedSuites: ["workbench", "obsidian"], suites: [{ suite: "workbench", status: "passed", scenarios: [{ id: "workbench.save", title: "Save", status: "passed", assertions: [{ label: "Saved", passed: true }] }] }] });
  assert.equal(report.status, "blocked");
  assert.equal(report.exitCode, 2);
  assert.equal(report.suites.find(s => s.suite === "obsidian").status, "not-run");
});

test("suite status cannot hide failed or assertion-free scenarios", () => {
  const report = buildReport({ selectedSuites: ["workbench"], suites: [{ suite: "workbench", status: "passed", scenarios: [{ id: "broken", title: "Broken", status: "failed", error: "persist failed" }] }] });
  assert.equal(report.status, "failed");
  assert.equal(report.exitCode, 1);
  const empty = buildReport({ selectedSuites: ["workbench"], suites: [{ suite: "workbench", status: "passed", scenarios: [] }] });
  assert.notEqual(empty.status, "passed");
});

test("report writes inspectable JSON and escaped HTML without credentials", async t => {
  const dir = await mkdtemp(join(tmpdir(), "arxiv-acceptance-report-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const report = buildReport({ selectedSuites: ["workbench"], suites: [{ suite: "workbench", status: "passed", scenarios: [{ id: "safe", title: "<script>alert(1)</script>", status: "passed", assertions: [{ label: "Stored", passed: true }], details: { apiKey: "do-not-publish", note: "Authorization: Bearer private-token" } }] }] });
  assert.equal(report.exitCode, 0);
  await writeReport(dir, report);
  const json = await readFile(join(dir, "report.json"), "utf8");
  const html = await readFile(join(dir, "report.html"), "utf8");
  assert.doesNotMatch(json + html, /do-not-publish|private-token/);
  assert.doesNotMatch(html, /<script>alert/);
  assert.match(html, /&lt;script&gt;/);
});

test("model findings remain visible for review even when the trusted task checks pass", async t => {
  const dir = await mkdtemp(join(tmpdir(), "arxiv-acceptance-findings-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const report = buildReport({ selectedSuites: ["exploration"], suites: [{ suite: "exploration", status: "passed", scenarios: [{ id: "settings", title: "Settings", status: "passed", assertions: [{ label: "Saved", passed: true }] }], findings: [{ status: "needs-review", taskId: "settings", summary: "Unclear save feedback", evidence: "Repeated click needed <script>" }] }] });
  assert.equal(report.status, "passed");
  assert.equal(report.reviewCount, 1);
  await writeReport(dir, report);
  const html = await readFile(join(dir, "report.html"), "utf8");
  assert.match(html, /待复核/);
  assert.match(html, /Unclear save feedback/);
  assert.match(html, /Repeated click needed &lt;script&gt;/);
});
