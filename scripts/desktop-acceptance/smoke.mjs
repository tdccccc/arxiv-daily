#!/usr/bin/env node
// Launch a real, isolated Obsidian on the designated test vault, attach a CDP
// session, and report what the renderer said. P3 attaches the actual acceptance
// scenarios to this same session.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runDesktopSession } from "./session.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

const vaultPath = process.env.OBSIDIAN_TEST_VAULT;
if (!vaultPath) {
  console.error(
    "OBSIDIAN_TEST_VAULT is not set.\n" +
      "Point it at a disposable Obsidian vault, e.g.\n" +
      "  OBSIDIAN_TEST_VAULT=/path/to/test_vault npm run test:desktop\n" +
      "The harness deploys this branch's build into that vault; the build is left in place,\n" +
      "while the plugin settings store and workspace layout are restored afterwards.",
  );
  process.exit(2);
}

const result = await runDesktopSession({
  vaultPath,
  pluginId: process.env.OBSIDIAN_TEST_PLUGIN_ID ?? "arxiv-daily",
  sourceDir: path.join(repoRoot, "plugin"),
  obsidianPath: process.env.OBSIDIAN_BINARY ?? "/opt/Obsidian/obsidian",
  async body({ port, session }) {
    return {
      port,
      pluginVersion: session.pluginVersion,
      trustPromptAccepted: session.trustPromptAccepted,
      diagnosticsComplete: session.diagnosticsComplete,
      vaultName: await session.evaluate("app.vault.getName()"),
      commands: await session.evaluate(
        'app.commands.listCommands().filter(c => c.id.startsWith("arxiv-daily:")).length',
      ),
      entries: session.diagnostics.entries(),
      errors: session.diagnostics.errors(),
    };
  },
});

console.log(`vault              ${result.vaultName}`);
console.log(`plugin version     ${result.pluginVersion}`);
console.log(`trust prompt       ${result.trustPromptAccepted ? "accepted" : "not shown"}`);
console.log(`plugin commands    ${result.commands}`);
console.log(`diagnostics        ${result.diagnosticsComplete ? "complete (plugin started after attach)" : "INCOMPLETE (plugin was already running)"}`);
console.log(`console entries    ${result.entries.length}`);
for (const entry of result.entries) {
  console.log(`  [${entry.source}/${entry.level}] ${entry.text.slice(0, 160)}`);
}
console.log(`errors             ${result.errors.length}`);
process.exitCode = result.errors.length === 0 ? 0 : 1;
