import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { parse } from "yaml";

async function workflow() {
  try { return parse(await readFile(new URL("../../.github/workflows/ui-acceptance.yml", import.meta.url), "utf8"), { uniqueKeys: true }); }
  catch (error) { if (error.code === "ENOENT") return {}; throw error; }
}

function assertAcceptanceWorkflow(value) {
  assert.ok(value.on && Object.hasOwn(value.on, "pull_request") && Object.hasOwn(value.on, "workflow_dispatch"), "Browser acceptance runs on pull requests and explicit dispatch");
  assert.deepEqual(value.on.push?.branches, ["main"]);
  assert.deepEqual(value.permissions, { contents: "read" });
  const job = value.jobs?.workbench;
  assert.ok(job, "A real workbench acceptance job is required");
  assert.ok(job["timeout-minutes"] > 0 && job["timeout-minutes"] <= 20);
  assert.notEqual(job["continue-on-error"], true);
  const steps = job.steps;
  assert.ok(steps.some(step => step.run?.includes("playwright install --with-deps chromium")), "CI provisions an actual browser");
  const acceptance = steps.find(step => step.run?.includes("npm run test:acceptance -- --suite workbench"));
  assert.ok(acceptance, "CI invokes the public fixed-browser acceptance command");
  assert.doesNotMatch(acceptance.run, /\|\||--explore|--skip-build/);
  assert.notEqual(acceptance["continue-on-error"], true);
  const upload = steps.find(step => step.uses?.startsWith("actions/upload-artifact@"));
  assert.ok(upload, "Failures retain reviewable evidence");
  assert.equal(upload.if, "always()");
  assert.match(upload.with.path, /^output\/playwright\/acceptance\/?$/);
  for (const step of steps) {
    if (step.uses) assert.match(step.uses, /^[^@\s]+@[0-9a-f]{40}$/, "Actions must be pinned to an exact commit");
    if (step.run?.includes("npm run test:acceptance")) {
      assert.doesNotMatch(step.run, /\|\||--explore|--skip-build/);
      assert.notEqual(step["continue-on-error"], true);
    }
    assert.doesNotMatch(JSON.stringify(step.env ?? {}), /api_key|apiKey|OPENAI_API_KEY|ANTHROPIC_API_KEY/);
  }
}

test("CI executes fixed real-browser acceptance and publishes evidence without model credentials", async () => {
  assertAcceptanceWorkflow(await workflow());
});

test("CI cannot turn failed acceptance into green or omit failure artifacts", async () => {
  const current = await workflow();
  assertAcceptanceWorkflow(current);
  const ignored = structuredClone(current);
  ignored.jobs.workbench.steps.find(step => step.run?.includes("npm run test:acceptance")).run += " || true";
  assert.throws(() => assertAcceptanceWorkflow(ignored));
  const missingEvidence = structuredClone(current);
  missingEvidence.jobs.workbench.steps.find(step => step.uses?.startsWith("actions/upload-artifact@")).if = "success()";
  assert.throws(() => assertAcceptanceWorkflow(missingEvidence));
});
