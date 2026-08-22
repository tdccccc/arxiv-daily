import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { parse } from "yaml";
import { root } from "../release-utils.mjs";

const workflowPath = `${root}/.github/workflows/lint.yml`;
const expectedWorkflow = {
  name: "Root verification",
  on: {
    push: { branches: ["main"] },
    pull_request: null,
  },
  permissions: { contents: "read" },
  jobs: {
    verify: {
      "runs-on": "ubuntu-latest",
      steps: [
        {
          name: "Check out repository",
          uses: "actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd",
          with: { "persist-credentials": false },
        },
        {
          name: "Set up Node.js",
          uses: "actions/setup-node@a0853c24544627f65ddf259abe73b1d18a591444",
          with: {
            "node-version": "22.17.0",
            cache: "npm",
            "cache-dependency-path": "package-lock.json",
          },
        },
        { name: "Install dependencies", run: "npm ci" },
        { name: "Audit dependencies", run: "npm audit --audit-level=moderate" },
        { name: "Test release tools", run: "npm run test:release-tools" },
        { name: "Check package boundaries", run: "npm run check:boundaries" },
        { name: "Lint Obsidian plugin", run: "npm run lint" },
        { name: "Typecheck workspaces", run: "npm run typecheck" },
        {
          name: "Test workspaces",
          run: "NODE_OPTIONS=--max-old-space-size=8192 npm run test:workspaces -- --maxWorkers=1",
        },
        {
          name: "Test settings UI regression",
          run: "npm --workspace obsidian-arxiv-daily exec vitest -- run --config vitest.config.mts tests/settings-declarative-tab.test.ts",
        },
        { name: "Build workspaces", run: "npm run build" },
        {
          name: "Check Obsidian submission requirements",
          run: "npm run check:obsidian-submission",
        },
        { name: "Smoke test build artifacts", run: "npm run smoke:build" },
        { name: "Smoke test CLI package installation", run: "npm run smoke:install" },
      ],
    },
    compatibility: {
      name: "Node ${{ matrix.node-version }} compatibility",
      "runs-on": "ubuntu-latest",
      strategy: {
        "fail-fast": false,
        matrix: { "node-version": ["20.19.0", "22.17.0"] },
      },
      steps: [
        {
          name: "Check out repository",
          uses: "actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd",
          with: { "persist-credentials": false },
        },
        {
          name: "Set up Node.js",
          uses: "actions/setup-node@a0853c24544627f65ddf259abe73b1d18a591444",
          with: {
            "node-version": "${{ matrix.node-version }}",
            cache: "npm",
            "cache-dependency-path": "package-lock.json",
          },
        },
        { name: "Install dependencies", run: "npm ci" },
        { name: "Typecheck workspaces", run: "npm run typecheck" },
        { name: "Build workspaces", run: "npm run build" },
        { name: "Smoke test build artifacts", run: "npm run smoke:build" },
      ],
    },
  },
};

function parseWorkflow(source) {
  const workflow = parse(source, {
    schema: "core",
    uniqueKeys: true,
  });
  if (
    workflow?.on?.pull_request &&
    typeof workflow.on.pull_request === "object" &&
    !Array.isArray(workflow.on.pull_request) &&
    Object.keys(workflow.on.pull_request).length === 0
  ) {
    workflow.on.pull_request = null;
  }
  return workflow;
}

export function assertRootVerificationWorkflow(source) {
  const workflow = parseWorkflow(source);
  assert.deepEqual(
    workflow,
    expectedWorkflow,
    "root verification workflow semantics must match the release-equivalent contract",
  );

  for (const job of Object.values(workflow.jobs)) {
    for (const step of job.steps) {
      if ("uses" in step) {
        assert.match(
          step.uses,
          /^[^@\s]+@[0-9a-f]{40}$/,
          `${step.uses} must use a full commit SHA`,
        );
      }
    }
  }
}

async function currentWorkflow() {
  return readFile(workflowPath, "utf8");
}

test("root workflow structurally matches the release-equivalent contract", async () => {
  const source = await currentWorkflow();
  assert.doesNotThrow(() => assertRootVerificationWorkflow(source));
});

test("equivalent YAML formatting does not change workflow acceptance", async () => {
  const source = (await currentWorkflow())
    .replace(/^on:$/m, '"on":')
    .replace('branches: ["main"]', "branches: [main]")
    .replace(/^  pull_request:$/m, "  pull_request: {}");

  assert.doesNotThrow(() => assertRootVerificationWorkflow(source));
});

test("PR and push filters cannot narrow or expand the required events", async () => {
  const source = await currentWorkflow();
  const pullRequestsOpenedOnly = source.replace(
    /^  pull_request:$/m,
    "  pull_request:\n    types: [opened]",
  );
  const tagPush = source.replace(
    /^    branches: \["main"\]$/m,
    '    branches: ["main"]\n    "tags": ["*.*.*"]',
  );

  assert.throws(() => assertRootVerificationWorkflow(pullRequestsOpenedOnly));
  assert.throws(() => assertRootVerificationWorkflow(tagPush));
});

test("all action steps must remain expected and pinned by immutable SHA", async () => {
  const source = await currentWorkflow();
  const mutableExtraAction = source.replace(
    /^      - name: Install dependencies$/m,
    "      - name: Mutable cache\n        uses: actions/cache@v4 # mutable tag\n      - name: Install dependencies",
  );
  const mutableCheckout = source.replace(
    /actions\/checkout@[0-9a-f]{40}/,
    "actions/checkout@v5",
  );

  assert.throws(() => assertRootVerificationWorkflow(mutableExtraAction));
  assert.throws(() => assertRootVerificationWorkflow(mutableCheckout));
});

test("required steps cannot be disabled or redirected", async () => {
  const source = await currentWorkflow();
  const disabledTests = source.replace(
    /^      - name: Test workspaces$/m,
    "      - name: Test workspaces\n        if: ${{ false }}",
  );
  const redirectedBuild = source.replace(
    /^      - name: Build workspaces$/m,
    "      - name: Build workspaces\n        working-directory: extensions/vscode-arxiv-daily",
  );
  const toleratedFailure = source.replace(
    /^      - name: Smoke test build artifacts$/m,
    "      - name: Smoke test build artifacts\n        continue-on-error: true",
  );

  assert.throws(() => assertRootVerificationWorkflow(disabledTests));
  assert.throws(() => assertRootVerificationWorkflow(redirectedBuild));
  assert.throws(() => assertRootVerificationWorkflow(toleratedFailure));
});

test("root verification remains limited to the root release group", async () => {
  const source = (await currentWorkflow()).toLowerCase();
  assert.doesNotMatch(source, /email-relay|services\/|companion|vscode|extensions\//);
});
