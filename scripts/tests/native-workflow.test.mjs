import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { parse } from "yaml";

function workflow(name) {
  const path = resolve(`.github/workflows/${name}.yml`);
  assert.ok(existsSync(path), `${name} workflow must exist`);
  return parse(readFileSync(path, "utf8"), { schema: "core", uniqueKeys: true });
}
const matrix = [
  { runner: "ubuntu-22.04", target: "linux-x64" },
  { runner: "ubuntu-24.04-arm", target: "linux-arm64" },
  { runner: "macos-15-intel", target: "darwin-x64" },
  { runner: "macos-15", target: "darwin-arm64" },
  { runner: "windows-2022", target: "win32-x64" },
  { runner: "windows-11-arm", target: "win32-arm64" },
];

function assertNativeWorkflow(value) {
  assert.deepEqual(value.permissions, { contents: "read" });
  assert.ok(value.on.workflow_call);
  assert.ok(Object.hasOwn(value.on, "pull_request"));
  assert.ok(!Object.hasOwn(value.on, "pull_request_target"));
  const job = value.jobs.build;
  assert.equal(job["runs-on"], "${{ matrix.runner }}");
  assert.equal(job.strategy["fail-fast"], false);
  assert.deepEqual(job.strategy.matrix.include, matrix);
  assert.equal(job.env.ARXIV_DAILY_TEST_NATIVE, "1");
  assert.ok(!job["continue-on-error"]);
  for (const step of job.steps) {
    if (step.uses) assert.match(step.uses, /^[^@\s]+@[0-9a-f]{40}$/);
    assert.ok(!step["continue-on-error"]);
    if (step.run && /test|smoke/.test(step.run)) assert.equal(step.if, undefined);
  }
  const commands = job.steps.map(step => step.run ?? "").join("\n");
  for (const command of ["native-sdk.mjs", "native-build.mjs", "native/tests/storage.test.cjs", "tests/native-private-storage.test.ts", "tests/native-storage-loader.test.ts", "native-package-smoke.mjs", "native-assets.mjs export"]) {
    assert.ok(commands.includes(command), `missing native verification: ${command}`);
  }
  const upload = job.steps.find(step => step.name === "Upload native binary");
  assert.equal(upload.with["if-no-files-found"], "error");
  assert.equal(upload.with.name, "native-storage-${{ matrix.target }}");
  assert.equal(upload.with.overwrite, false);
}

test("native CI uses real OS/architecture runners and mandatory capability tests", () => {
  assertNativeWorkflow(workflow("native-storage"));
});

test("native CI forwards Vitest filters and report arguments without npm shell parsing", () => {
  const steps = workflow("native-storage").jobs.build.steps;
  for (const name of ["Test private storage and shared locks", "Test Node delivery composition", "Test desktop delivery composition"]) {
    const step = steps.find(value => value.name === name);
    assert.match(step.run, /^node (?:\.\.\/)+node_modules\/vitest\/vitest\.mjs run /);
    assert.ok(step["working-directory"]);
    assert.match(step.run, /--reporter=json --outputFile=/);
  }
  assert.match(steps.find(value => value.name === "Test Node delivery composition").run, /-t "native delivery composition"/);
  assert.match(steps.find(value => value.name === "Test desktop delivery composition").run, /-t "uses native delivery storage"/);
});

test("native CI rejects skipped tests, tolerated failures, and a fake single-platform matrix", () => {
  const original = workflow("native-storage");
  const skipped = structuredClone(original);
  skipped.jobs.build.steps.find(step => step.run?.includes("native/tests/storage.test.cjs")).if = "${{ false }}";
  assert.throws(() => assertNativeWorkflow(skipped));
  const tolerated = structuredClone(original);
  tolerated.jobs.build["continue-on-error"] = true;
  assert.throws(() => assertNativeWorkflow(tolerated));
  const fake = structuredClone(original);
  fake.jobs.build.strategy.matrix.include[2].runner = "ubuntu-22.04";
  assert.throws(() => assertNativeWorkflow(fake));
});

test("both publishing workflows require same-run native artifacts before building releases", () => {
  for (const [name, jobName] of [["release", "release"], ["publish-cli", "publish"]]) {
    const value = workflow(name);
    assert.equal(value.jobs.native.uses, "./.github/workflows/native-storage.yml");
    assert.deepEqual(value.jobs.native.permissions, { contents: "read" });
    const job = value.jobs[jobName];
    assert.ok([].concat(job.needs ?? []).includes("native"));
    assert.equal(job.env.ARXIV_DAILY_NATIVE_RELEASE, "1");
    const download = job.steps.find(step => step.uses?.startsWith("actions/download-artifact@"));
    assert.ok(download);
    assert.match(download.uses, /@[0-9a-f]{40}$/);
    assert.equal(download.with.pattern, "native-storage-*");
    assert.equal(download.with["merge-multiple"], true);
    assert.equal(download.with["digest-mismatch"], "error");
    assert.equal(download.with["run-id"], undefined);
    assert.equal(download.with["github-token"], undefined);
    assert.equal(download.with.path, "packages/node-runtime/native/prebuilds");
  }
});
