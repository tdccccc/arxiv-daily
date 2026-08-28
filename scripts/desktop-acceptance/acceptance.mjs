#!/usr/bin/env node
// Desktop acceptance run: drives a real, isolated Obsidian through the
// scenarios that could previously only be checked by hand.
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  pdfPageLocationScenario,
  runScenarios,
  settingsMigrationScenario,
  sidecarDisabledScenario,
  sidecarFallbackScenario,
} from "./scenarios.mjs";
import { runDesktopSession } from "./session.mjs";
import { installSettingsFixture, legacySettingsFixture } from "./settings-fixture.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const vaultPath = process.env.OBSIDIAN_TEST_VAULT;

if (!vaultPath) {
  console.error(
    "OBSIDIAN_TEST_VAULT is not set.\n\n" +
      "Point it at a disposable Obsidian vault holding at least one PDF:\n" +
      "  OBSIDIAN_TEST_VAULT=/path/to/test_vault npm run test:desktop\n\n" +
      "The harness deploys this branch's build into that vault and leaves it there;\n" +
      "the plugin settings store and workspace layout are captured first and restored\n" +
      "on every exit path, including Ctrl-C.",
  );
  process.exit(2);
}

// Port 1 is privileged and never listening, so a probe against it fails the way
// an absent sidecar provider would.
const UNREACHABLE_PORT = 1;

let outcome;
try {
  outcome = await runDesktopSession({
    vaultPath,
    pluginId: process.env.OBSIDIAN_TEST_PLUGIN_ID ?? "arxiv-daily",
    sourceDir: path.join(repoRoot, "plugin"),
    obsidianPath: process.env.OBSIDIAN_BINARY ?? "/opt/Obsidian/obsidian",
    beforeLaunch: ({ vaultPath: vault, pluginId, fs }) =>
      installSettingsFixture({ vaultPath: vault, pluginId, fs, data: legacySettingsFixture() }),
    async body({ session }) {
      const results = await runScenarios([
        () => settingsMigrationScenario({ session }),
        () => sidecarDisabledScenario({ session, requests: session.requests.networkUrls() }),
        () => pdfPageLocationScenario({ session }),
        () => sidecarFallbackScenario({ session, unreachablePort: UNREACHABLE_PORT }),
      ]);
      return {
        results,
        pluginVersion: session.pluginVersion,
        diagnosticsComplete: session.diagnosticsComplete,
        errors: session.diagnostics.errors(),
        observedRequests: session.requests.networkUrls(),
      };
    },
  });
} catch (error) {
  console.error(`desktop acceptance could not run: ${error.message}`);
  process.exit(1);
}

console.log(`build under test   ${outcome.pluginVersion}`);
console.log(
  `diagnostics        ${outcome.diagnosticsComplete ? "complete" : "INCOMPLETE — the plugin was already running when we attached"}`,
);
console.log(`network requests   ${outcome.observedRequests.length}`);
console.log("");
for (const scenario of outcome.results.scenarios) {
  console.log(`${scenario.passed ? "PASS" : "FAIL"}  ${scenario.name}`);
  console.log(`      ${scenario.detail}`);
}
console.log("");
if (outcome.errors.length > 0) {
  console.log(`renderer errors    ${outcome.errors.length}`);
  for (const entry of outcome.errors) console.log(`  [${entry.source}] ${entry.text.slice(0, 200)}`);
}

const clean = outcome.results.passed && outcome.errors.length === 0 && outcome.diagnosticsComplete;
console.log(clean ? "desktop acceptance PASSED" : "desktop acceptance FAILED");
process.exitCode = clean ? 0 : 1;
