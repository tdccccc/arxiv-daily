/**
 * P6 T5 — proposal-generation diagnostic (2026-08-04).
 *
 * Reproduces the plugin's direction-proposal generation against the real
 * configured endpoint and the real plugin_test catalog, without the Obsidian
 * host, to surface the exact failure class. The API key is read from the
 * plugin data but never printed.
 *
 * Usage:
 *   PLUGIN_TEST=/path/to/plugin_test node scripts/personalization/run-diagnose-proposal.mjs
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  decodePersonalLibraryCatalog,
  LlmClient,
  proposePersonalLibraryDirections,
  type PersonalLibraryDirectionProposal,
} from "@arxiv-daily/core";
import { NodeHttpClient } from "@arxiv-daily/node-runtime";

const PLUGIN_TEST = process.env.PLUGIN_TEST ?? "/home/tiandc/Documents/code/arxiv-daily/plugin_test";

const logger = {
  warn: (...args: unknown[]) => console.warn("[warn]", ...args),
  error: (...args: unknown[]) => console.error("[error]", ...args),
  info: (...args: unknown[]) => console.info("[info]", ...args),
  setSensitiveValues: () => undefined,
};

async function main(): Promise<void> {
  const indexPath = join(PLUGIN_TEST, "arxiv-daily", ".index");
  const data = JSON.parse(
    await readFile(join(PLUGIN_TEST, ".obsidian", "plugins", "arxiv-daily", "data.json"), "utf8"),
  );
  const llmSettings = data.settings?.llm;
  if (!llmSettings?.baseUrl || !llmSettings?.apiKey) {
    console.log("FAIL: no llm settings (baseUrl/apiKey) in plugin data");
    return;
  }
  console.log("endpoint:", llmSettings.baseUrl, "| model:", llmSettings.model);

  const catalogRaw = await readFile(join(indexPath, "personal-library-catalog.json"), "utf8");
  const catalog = decodePersonalLibraryCatalog(JSON.parse(catalogRaw));
  if (!catalog) {
    console.log("FAIL: catalog cannot be decoded");
    return;
  }
  const ready = Object.values(catalog.files).filter((f) => f.status === "ready").length;
  console.log("catalog: papers", Object.keys(catalog.papers).length, "ready files", ready);

  let proposal: PersonalLibraryDirectionProposal | undefined;
  let errorText = "";
  try {
    proposal = await proposePersonalLibraryDirections({
      catalog,
      llm: new LlmClient(llmSettings, logger, new NodeHttpClient()),
      signal: undefined,
      createId: () => `diag-${Math.random().toString(36).slice(2, 10)}`,
      now: () => new Date(),
    });
  } catch (error) {
    errorText = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  }
  if (proposal) {
    console.log("OK: proposal generated with", proposal.candidates.length, "candidates");
    for (const candidate of proposal.candidates) {
      const reps = "representatives" in candidate
        ? (candidate as { representatives: Array<{ paperKey: string }> }).representatives
        : (candidate as { representativePaperKeys: string[] }).representativePaperKeys;
      console.log(" -", candidate.name, "| reps:", reps.length);
    }
  } else {
    console.log("FAIL:", errorText);
  }
}

main().catch((error) => {
  console.error("FAIL:", error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
