import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { parse } from "yaml";
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
  assert.equal(steps[1].uses, "github/codeql-action/init@db488ddef3bf6cb639b32c2e9a7c0a7ea8271d28");
  assert.equal(steps[2].uses, "github/codeql-action/analyze@db488ddef3bf6cb639b32c2e9a7c0a7ea8271d28");
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
  assert.deepEqual(
    config.updates.map(({ "package-ecosystem": ecosystem, directory }) => [ecosystem, directory]),
    [
      ["npm", "/"],
      ["npm", "/services/email-relay"],
      ["npm", "/extensions/vscode-arxiv-daily"],
      ["github-actions", "/"],
    ],
  );
  for (const update of config.updates) {
    assert.equal(update.schedule.interval, "weekly");
    assert.equal(update.schedule.day, "monday");
    assert.ok(Number.isInteger(update["open-pull-requests-limit"]));
  }
});

test("root and relay workflows require moderate-or-higher audit gates", async () => {
  const rootWorkflow = await readFile(`${root}/.github/workflows/lint.yml`, "utf8");
  const relayWorkflow = await readFile(`${root}/.github/workflows/email-relay.yml`, "utf8");
  assert.match(rootWorkflow, /npm audit --audit-level=moderate/);
  assert.match(relayWorkflow, /npm audit --audit-level=moderate/);
});
