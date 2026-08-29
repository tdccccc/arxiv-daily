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
import { describeBlockers, preflight } from "./preflight.mjs";
import { startProbeListener } from "./probe-listener.mjs";
import { runDesktopSession } from "./session.mjs";
import { installSettingsFixture, legacySettingsFixture } from "./settings-fixture.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const vaultPath = process.env.OBSIDIAN_TEST_VAULT;
const pluginId = process.env.OBSIDIAN_TEST_PLUGIN_ID ?? "arxiv-daily";
const sourceDir = path.join(repoRoot, "plugin");
const obsidianPath = process.env.OBSIDIAN_BINARY ?? "/opt/Obsidian/obsidian";

// Exit code 2 means "this environment cannot run the acceptance"; exit code 1
// means "the acceptance ran and something failed". Conflating them would let a
// missing dependency read as a product defect, and vice versa.
const EXIT_BLOCKED = 2;

const environment = await preflight({ vaultPath, obsidianPath, sourceDir });
if (!environment.ok) {
  console.error("desktop acceptance cannot run in this environment:\n");
  console.error(describeBlockers(environment.blockers));
  console.error(
    "\nOnce those are resolved:\n" +
      "  OBSIDIAN_TEST_VAULT=/path/to/test_vault npm run test:desktop\n\n" +
      "The harness deploys this branch's build into that vault and leaves it there;\n" +
      "the plugin settings store and workspace layout are captured first and restored\n" +
      "on every exit path, including Ctrl-C.",
  );
  process.exit(EXIT_BLOCKED);
}

// A listener we own is the only way to observe the plugin's HTTP: it goes out
// through Obsidian's requestUrl in the Electron main process, invisible to the
// renderer's debugging protocol. It refuses every request, which is exactly the
// condition the probe-failure fallback exists for.
const listener = await startProbeListener();

let outcome;
try {
  outcome = await runDesktopSession({
    vaultPath,
    pluginId,
    sourceDir,
    obsidianPath,
    beforeLaunch: ({ vaultPath: vault, pluginId, fs }) =>
      installSettingsFixture({ vaultPath: vault, pluginId, fs, data: legacySettingsFixture() }),
    async body({ session }) {
      const results = await runScenarios([
        () => settingsMigrationScenario({ session }),
        () => sidecarDisabledScenario({ session, listener }),
        () => pdfPageLocationScenario({ session }),
        () => sidecarFallbackScenario({ session, listener }),
      ]);
      return {
        results,
        pluginVersion: session.pluginVersion,
        diagnosticsComplete: session.diagnosticsComplete,
        errors: session.diagnostics.errors(),
      };
    },
  });
} catch (error) {
  await listener.close();
  console.error(`desktop acceptance could not run: ${error.message}`);
  process.exit(1);
}
await listener.close();

console.log(`build under test   ${outcome.pluginVersion}`);
console.log(
  `diagnostics        ${outcome.diagnosticsComplete ? "complete" : "INCOMPLETE — the plugin was already running when we attached"}`,
);
console.log(`sidecar requests   ${listener.requests().length} (to the harness's own listener)`);
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
