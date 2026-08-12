import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { parse } from "yaml";
import { root } from "../release-utils.mjs";

export const relayPaths = [
  "services/email-relay/**",
  "packages/core/src/delivery/hosted.ts",
  "packages/core/src/delivery/deliver-email.ts",
  "packages/core/src/delivery/types.ts",
  "packages/core/tests/delivery/hosted-deliver.test.ts",
  ".github/workflows/email-relay.yml",
  "product-units.json",
  "scripts/check-product-units.mjs",
  "scripts/tests/product-units.test.mjs",
  "scripts/tests/independent-product-workflows.test.mjs",
];

export const companionPaths = [
  "extensions/vscode-arxiv-daily/**",
  "contracts/companion-cli-commands.json",
  "apps/cli/package.json",
  "apps/cli/src/main.ts",
  "apps/cli/tests/cli-main.test.ts",
  ".github/workflows/vscode-companion.yml",
  "product-units.json",
  "scripts/check-product-units.mjs",
  "scripts/tests/product-units.test.mjs",
  "scripts/tests/independent-product-workflows.test.mjs",
];

const checkoutStep = {
  name: "Check out repository",
  uses: "actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd",
  with: { "persist-credentials": false },
};

const expectedRelayWorkflow = {
  name: "Email relay verification",
  on: {
    push: { branches: ["main"], paths: relayPaths },
    pull_request: { paths: relayPaths },
  },
  permissions: { contents: "read" },
  jobs: {
    verify: {
      "runs-on": "ubuntu-latest",
      defaults: { run: { "working-directory": "services/email-relay" } },
      steps: [
        checkoutStep,
        {
          name: "Set up Node.js",
          uses: "actions/setup-node@a0853c24544627f65ddf259abe73b1d18a591444",
          with: {
            "node-version": "22.17.0",
            cache: "npm",
            "cache-dependency-path": "services/email-relay/package-lock.json",
          },
        },
        { name: "Install relay dependencies", run: "npm ci" },
        { name: "Typecheck relay", run: "npm run typecheck" },
        { name: "Test relay", run: "npm test" },
        {
          name: "Verify preflight script is read-only",
          run: "node scripts/cutover-preflight.mjs --check-readonly",
        },
        {
          name: "Dry-run relay bundle",
          run: 'npm exec -- wrangler deploy src/index.ts --dry-run --config wrangler.toml --outdir "$RUNNER_TEMP/email-relay-wrangler"',
        },
      ],
    },
  },
};

const expectedCompanionWorkflow = {
  name: "VS Code companion verification",
  on: {
    push: { branches: ["main"], paths: companionPaths },
    pull_request: { paths: companionPaths },
  },
  permissions: { contents: "read" },
  jobs: {
    verify: {
      "runs-on": "ubuntu-latest",
      defaults: {
        run: { "working-directory": "extensions/vscode-arxiv-daily" },
      },
      steps: [
        checkoutStep,
        {
          name: "Set up Node.js",
          uses: "actions/setup-node@a0853c24544627f65ddf259abe73b1d18a591444",
          with: {
            "node-version": "22.17.0",
            cache: "npm",
            "cache-dependency-path": "extensions/vscode-arxiv-daily/package-lock.json",
          },
        },
        { name: "Install companion dependencies", run: "npm ci" },
        { name: "Build companion", run: "npm run build" },
        { name: "Test companion", run: "npm test" },
        { name: "Smoke test companion", run: "npm run smoke" },
        {
          name: "Verify temporary VSIX package",
          run: 'output="$RUNNER_TEMP/arxiv-daily-vscode.vsix"\nnpm run vsix:package -- --out "$output"\ntest -s "$output"\n',
        },
      ],
    },
  },
};

function parseWorkflow(source) {
  return parse(source, { schema: "core", uniqueKeys: true });
}

function assertPinnedActions(workflow) {
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

function assertNoReleaseOrCredentialBypass(source) {
  assert.doesNotMatch(source, /\b(?:npm|vsce)\s+publish\b/i);
  assert.doesNotMatch(source, /\bwrangler\s+(?:publish|versions upload)\b/i);
  assert.doesNotMatch(source, /\b(?:secrets\.|github\.token|GH_TOKEN|CLOUDFLARE_API_TOKEN)\b/i);
}

export function assertRelayWorkflow(source) {
  const workflow = parseWorkflow(source);
  assert.deepEqual(workflow, expectedRelayWorkflow);
  assertPinnedActions(workflow);
  assertNoReleaseOrCredentialBypass(source);
  const dryRun = workflow.jobs.verify.steps.at(-1).run;
  assert.match(dryRun, /wrangler deploy src\/index\.ts/);
  assert.match(dryRun, /--dry-run/);
  assert.match(dryRun, /--config wrangler\.toml/);
  assert.match(dryRun, /--outdir "\$RUNNER_TEMP\/email-relay-wrangler"/);
}

export function assertCompanionWorkflow(source) {
  const workflow = parseWorkflow(source);
  assert.deepEqual(workflow, expectedCompanionWorkflow);
  assertPinnedActions(workflow);
  assertNoReleaseOrCredentialBypass(source);
  const packageStep = workflow.jobs.verify.steps.at(-1).run;
  assert.match(packageStep, /npm run vsix:package -- --out "\$output"/);
  assert.match(packageStep, /test -s "\$output"/);
  assert.doesNotMatch(packageStep, /(?:^|\s)(?:dist\/|[^$\s]*\.vsix)/m);
}

async function currentWorkflow(name) {
  return readFile(`${root}/.github/workflows/${name}`, "utf8");
}

test("relay workflow has exact independent verification semantics", async () => {
  const source = await currentWorkflow("email-relay.yml");
  assert.doesNotThrow(() => assertRelayWorkflow(source));
});

test("companion workflow has exact independent verification semantics", async () => {
  const source = await currentWorkflow("vscode-companion.yml");
  assert.doesNotThrow(() => assertCompanionWorkflow(source));
});

test("PR events and compatibility paths cannot be narrowed", async () => {
  const relay = await currentWorkflow("email-relay.yml");
  const companion = await currentWorkflow("vscode-companion.yml");

  assert.throws(() =>
    assertRelayWorkflow(
      relay
        .replace(/^  pull_request:$/m, "  pull_request:\n    types: [opened]")
        .replace(/^      - packages\/core\/src\/delivery\/hosted\.ts\n/m, ""),
    ),
  );
  assert.throws(() =>
    assertCompanionWorkflow(
      companion.replace(
        /^      - contracts\/companion-cli-commands\.json\n/m,
        "",
      ),
    ),
  );
});

test("actions, cache locks, working directories, and commands stay exact", async () => {
  const relay = await currentWorkflow("email-relay.yml");
  const companion = await currentWorkflow("vscode-companion.yml");

  assert.throws(() =>
    assertRelayWorkflow(
      relay.replace(/actions\/checkout@[0-9a-f]{40}/, "actions/checkout@v5"),
    ),
  );
  assert.throws(() =>
    assertRelayWorkflow(
      relay.replace(
        "services/email-relay/package-lock.json",
        "package-lock.json",
      ),
    ),
  );
  assert.throws(() =>
    assertCompanionWorkflow(
      companion.replace(
        "extensions/vscode-arxiv-daily",
        "services/email-relay",
      ),
    ),
  );
  assert.throws(() =>
    assertCompanionWorkflow(companion.replace("npm run smoke", "npm test")),
  );
});

test("required verification cannot be disabled or tolerated", async () => {
  const relay = await currentWorkflow("email-relay.yml");
  const companion = await currentWorkflow("vscode-companion.yml");

  assert.throws(() =>
    assertRelayWorkflow(
      relay.replace(
        /^      - name: Test relay$/m,
        "      - name: Test relay\n        if: ${{ false }}",
      ),
    ),
  );
  assert.throws(() =>
    assertCompanionWorkflow(
      companion.replace(
        /^      - name: Smoke test companion$/m,
        "      - name: Smoke test companion\n        continue-on-error: true",
      ),
    ),
  );
});

test("production deploy, VSIX publish, and credential access are rejected", async () => {
  const relay = await currentWorkflow("email-relay.yml");
  const companion = await currentWorkflow("vscode-companion.yml");

  assert.throws(() => assertRelayWorkflow(relay.replace(" --dry-run", "")));
  assert.throws(() =>
    assertCompanionWorkflow(
      companion.replace(
        'npm run vsix:package -- --out "$output"',
        'npx vsce publish --packagePath "$output"',
      ),
    ),
  );
  assert.throws(() =>
    assertRelayWorkflow(
      relay.replace(
        /^        run: npm ci$/m,
        "        env:\n          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}\n        run: npm ci",
      ),
    ),
  );
});
