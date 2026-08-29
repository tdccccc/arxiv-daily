import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { root } from "../release-utils.mjs";
import { runRootTests } from "../run-root-tests.mjs";

function recordingSpawn(result = { status: 0 }) {
  const calls = [];
  return {
    calls,
    spawn(command, args, options) {
      calls.push({ command, args, options });
      return result;
    },
  };
}

test("a default root test runs every workspace through the explicit full-suite script", () => {
  const recorder = recordingSpawn();

  const status = runRootTests([], {
    spawn: recorder.spawn,
    npmExecutable: "/fixture/npm-cli.js",
  });

  assert.equal(status, 0);
  assert.deepEqual(recorder.calls, [{
    command: process.execPath,
    args: ["/fixture/npm-cli.js", "run", "test:workspaces"],
    options: { stdio: "inherit" },
  }]);
});

test("explicit root test arguments target one Core invocation unchanged", () => {
  const recorder = recordingSpawn();
  const argv = [
    "--reporter=dot",
    "-t",
    "returns failed_transient when /recent misses the date",
    "tests/pipeline.test.ts",
  ];

  const status = runRootTests(argv, {
    spawn: recorder.spawn,
    npmExecutable: "/fixture/npm-cli.js",
  });

  assert.equal(status, 0);
  assert.equal(recorder.calls.length, 1);
  assert.deepEqual(recorder.calls[0], {
    command: process.execPath,
    args: [
      "/fixture/npm-cli.js",
      "run",
      "test",
      "--workspace",
      "@arxiv-daily/core",
      "--",
      ...argv,
    ],
    options: { stdio: "inherit" },
  });
});

test("a root test child without an exit status fails closed", () => {
  const status = runRootTests([], {
    spawn: () => ({ status: null, error: new Error("spawn failed") }),
    npmExecutable: "npm",
  });

  assert.equal(status, 1);
});

test("the real root npm entry sends a Core focus to Core only", () => {
  const result = spawnSync(
    "npm",
    [
      "test",
      "--",
      "--reporter=dot",
      "-t",
      "returns failed_transient when /recent misses the date",
      "tests/pipeline.test.ts",
    ],
    { cwd: root, encoding: "utf8", timeout: 120_000 },
  );

  assert.equal(
    result.status,
    0,
    `focused root test failed\nstdout=${result.stdout}\nstderr=${result.stderr}`,
  );
  // Vitest names the package it ran in; unlike npm's "> pkg@version script"
  // banner, that line survives a caller running us under --silent, which sets
  // npm_config_loglevel=silent for every child npm.
  assert.match(result.stdout, /packages[/\\]core/);
  assert.doesNotMatch(result.stdout, /@arxiv-daily\/node-runtime|arxiv-daily@0\.4\.1 test|obsidian-arxiv-daily/);
  assert.doesNotMatch(`${result.stdout}\n${result.stderr}`, /No test files found/);
});
