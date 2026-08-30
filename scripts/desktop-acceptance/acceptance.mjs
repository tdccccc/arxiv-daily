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
import { librarySettingsScenarios } from "./library-settings.mjs";
import { describeBlockers, preflight } from "./preflight.mjs";
import { startProbeListener } from "./probe-listener.mjs";
import { runDesktopSession } from "./session.mjs";
import { createScreenshotWriter, resolveScreenshotDir } from "./screenshots.mjs";
import {
  connectedLibraryFixture,
  installSettingsFixture,
  legacySettingsFixture,
  readRootIdentity,
  resolveLibraryRoot,
} from "./settings-fixture.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const vaultPath = process.env.OBSIDIAN_TEST_VAULT;
const pluginId = process.env.OBSIDIAN_TEST_PLUGIN_ID ?? "arxiv-daily";
const sourceDir = path.join(repoRoot, "plugin");
const obsidianPath = process.env.OBSIDIAN_BINARY ?? "/opt/Obsidian/obsidian";
const screenshotDir = resolveScreenshotDir({ repoRoot });

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

const sessions = [];
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
  sessions.push({ name: "sidecar, migration and PDF location", ...outcome });

  // The personal library settings page needs a library already connected, which
  // the legacy fixture above deliberately does not have. A second session is
  // the honest way to hold two different persisted states: the state guard
  // brackets each run on its own.
  const libraryRoot = await resolveLibraryRoot({ vaultPath });
  const rootIdentity = await readRootIdentity(libraryRoot);
  const libraryOutcome = await runDesktopSession({
    vaultPath,
    pluginId,
    sourceDir,
    obsidianPath,
    beforeLaunch: ({ vaultPath: vault, pluginId, fs }) =>
      installSettingsFixture({
        vaultPath: vault,
        pluginId,
        fs,
        data: connectedLibraryFixture({ libraryRoot, rootIdentity }),
      }),
    async body({ session }) {
      const screenshots = await createScreenshotWriter({
        client: session.client,
        evaluate: session.evaluate,
        outputDir: screenshotDir,
      });
      const results = await runScenarios([
        () => librarySettingsScenarios({ session, screenshots }),
      ]);
      return {
        results,
        libraryRoot,
        screenshots: screenshots.written(),
        pluginVersion: session.pluginVersion,
        diagnosticsComplete: session.diagnosticsComplete,
        errors: session.diagnostics.errors(),
      };
    },
  });
  sessions.push({ name: "personal library settings page", ...libraryOutcome });
} catch (error) {
  await listener.close();
  console.error(`desktop acceptance could not run: ${error.message}`);
  process.exit(1);
}
await listener.close();

console.log(`build under test   ${sessions[0].pluginVersion}`);
console.log(`sidecar requests   ${listener.requests().length} (to the harness's own listener)`);
const libraryRun = sessions.find((run) => run.screenshots);
if (libraryRun) {
  console.log(`library folder     ${libraryRun.libraryRoot}`);
  console.log(`screenshots        ${screenshotDir}`);
  for (const entry of libraryRun.screenshots) console.log(`  ${entry.name}.png`);
}
console.log("");

let clean = true;
for (const run of sessions) {
  console.log(
    `session ${run.name}: diagnostics ${run.diagnosticsComplete ? "complete" : "INCOMPLETE — the plugin was already running when we attached"}`,
  );
  for (const scenario of run.results.scenarios) {
    console.log(`${scenario.passed ? "PASS" : "FAIL"}  ${scenario.name}`);
    console.log(`      ${scenario.detail}`);
  }
  if (run.errors.length > 0) {
    console.log(`renderer errors    ${run.errors.length}`);
    for (const entry of run.errors) console.log(`  [${entry.source}] ${entry.text.slice(0, 200)}`);
  }
  console.log("");
  clean &&= run.results.passed && run.errors.length === 0 && run.diagnosticsComplete;
}

console.log(clean ? "desktop acceptance PASSED" : "desktop acceptance FAILED");
process.exitCode = clean ? 0 : 1;
