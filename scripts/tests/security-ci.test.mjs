import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { parse } from "yaml";
import { auditLockfiles, trackedLockfiles } from "../audit-lockfiles.mjs";
import { root } from "../release-utils.mjs";

const codeqlPath = `${root}/.github/workflows/codeql.yml`;
const dependabotPath = `${root}/.github/dependabot.yml`;

test("CodeQL workflow is PR, main, scheduled, and manually runnable", async () => {
  const source = await readFile(codeqlPath, "utf8");
  const workflow = parse(source, { schema: "core", uniqueKeys: true });
  assert.deepEqual(workflow.on.push, { branches: ["main"] });
  assert.deepEqual(workflow.on.pull_request, null);
  assert.deepEqual(workflow.on.schedule, [{ cron: "23 3 * * 1" }]);
  assert.deepEqual(workflow.on.workflow_dispatch, null);
  assert.deepEqual(workflow.permissions, {
    contents: "read",
    "security-events": "write",
    actions: "read",
  });
  const steps = workflow.jobs.analyze.steps;
  assert.equal(steps[1].uses, "github/codeql-action/init@6f2ac81f5cd81fe351a197ba0b00d91d03cd2aee");
  assert.equal(steps[2].uses, "github/codeql-action/analyze@6f2ac81f5cd81fe351a197ba0b00d91d03cd2aee");
  assert.equal(steps[1].with.languages, "javascript-typescript");
  for (const step of steps) {
    if ("uses" in step) assert.match(step.uses, /^[^@\s]+@[0-9a-f]{40}$/);
  }
});

test("Dependabot covers every lockfile and GitHub Actions", async () => {
  const config = parse(await readFile(dependabotPath, "utf8"), {
    schema: "core",
    uniqueKeys: true,
  });
  assert.equal(config.version, 2);
  const locks = execFileSync('git', ['ls-files', '-z', '*package-lock.json'], { cwd: root, encoding: 'utf8' }).split('\0').filter(Boolean);
  const covered = new Set(config.updates.filter(update => update['package-ecosystem'] === 'npm').map(update => update.directory));
  for (const lock of locks) {
    const directory = path.posix.dirname(lock);
    assert.ok(covered.has(directory === '.' ? '/' : `/${directory}`), `Dependabot missing ${directory}`);
  }
  assert.ok(config.updates.some(update => update['package-ecosystem'] === 'github-actions' && update.directory === '/'));
  for (const update of config.updates) {
    assert.equal(update.schedule.interval, "weekly");
    assert.equal(update.schedule.day, "monday");
    assert.ok(Number.isInteger(update["open-pull-requests-limit"]));
  }
});

test("CI audits every tracked lockfile and relay retains its own gate", async () => {
  const rootWorkflow = await readFile(`${root}/.github/workflows/lint.yml`, "utf8");
  const relayWorkflow = await readFile(`${root}/.github/workflows/email-relay.yml`, "utf8");
  assert.match(rootWorkflow, /node scripts\/audit-lockfiles\.mjs/);
  assert.match(relayWorkflow, /npm audit --audit-level=moderate/);
});

test('audit runner includes every tracked lockfile and newly added directories', () => {
  const locks = [...trackedLockfiles(), 'tools/new-runtime/package-lock.json'];
  const calls = [];
  auditLockfiles({ lockfiles: locks, log() {}, run(command, args) { calls.push({ command, args }); return { status: 0 }; } });
  assert.equal(calls.length, locks.length);
  assert.deepEqual(calls.map(call => call.args.at(-1)), locks.map(lock => path.resolve(root, path.dirname(lock))));
  assert.ok(calls.every(call => call.command === 'npm' && call.args.includes('--audit-level=moderate') && call.args.includes('--package-lock-only')));
});
test('a failing runtime audit cannot hide behind a passing root audit', () => {
  assert.throws(() => auditLockfiles({ lockfiles: ['package-lock.json', 'tools/new-runtime/package-lock.json'], log() {}, run(_command, args) { return { status: args.at(-1).endsWith('new-runtime') ? 1 : 0 }; } }), /tools\/new-runtime\/package-lock.json/);
  assert.throws(() => auditLockfiles({ lockfiles: [] }), /No tracked lockfiles/);
});

test('audit retries transient registry errors but does not bypass vulnerabilities', () => {
  let calls = 0;
  auditLockfiles({ lockfiles: ['package-lock.json'], log() {}, run() { return ++calls < 3 ? { status: 1, stderr: 'audit endpoint returned an error' } : { status: 0 }; } });
  assert.equal(calls, 3);
  calls = 0;
  assert.throws(() => auditLockfiles({ lockfiles: ['package-lock.json'], log() {}, run() { calls++; return { status: 1, stdout: '1 high severity vulnerability' }; } }));
  assert.equal(calls, 1);
});
