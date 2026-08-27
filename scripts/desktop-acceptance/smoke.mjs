#!/usr/bin/env node
// P1 skeleton: launch a real, isolated Obsidian on the designated test vault,
// prove the CDP endpoint answers, then reclaim the process group cleanly.
// Driving the renderer is P2's job; this entry only establishes the environment.
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
      "The harness deploys this branch's build into that vault and restores it afterwards.",
  );
  process.exit(2);
}

const result = await runDesktopSession({
  vaultPath,
  pluginId: process.env.OBSIDIAN_TEST_PLUGIN_ID ?? "arxiv-daily",
  sourceDir: path.join(repoRoot, "plugin"),
  obsidianPath: process.env.OBSIDIAN_BINARY ?? "/opt/Obsidian/obsidian",
  async body({ port, browser, expectedVersion }) {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    return {
      port,
      userAgent: browser["User-Agent"],
      expectedVersion,
      pages: targets.filter((t) => t.type === "page").map((t) => t.url),
    };
  },
});

console.log(`CDP port           ${result.port}`);
console.log(`build under test   ${result.expectedVersion}`);
console.log(`renderer           ${result.userAgent}`);
console.log(`pages              ${JSON.stringify(result.pages)}`);
