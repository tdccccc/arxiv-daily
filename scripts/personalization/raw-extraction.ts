/**
 * P6 T5 — raw extraction output diagnostic (2026-08-04).
 *
 * Sends the exact extraction request the proposal generator would send
 * (system prompt + rendered paper batch) and prints the raw model output,
 * to see why strict validation keeps failing with cues-invalid.
 *
 * Usage:
 *   PLUGIN_TEST=/path/to/plugin_test node scripts/personalization/run-raw-extraction.mjs
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  decodePersonalLibraryCatalog,
  LlmClient,
  renderPersonalLibraryExtractionUserMessage,
  selectPersonalLibraryDirectionPapers,
} from "@arxiv-daily/core";
import { NodeHttpClient } from "@arxiv-daily/node-runtime";

const PLUGIN_TEST = process.env.PLUGIN_TEST ?? "/home/tiandc/Documents/code/arxiv-daily/plugin_test";
const PROMPT_PATH = join(
  process.cwd(),
  "packages",
  "core",
  "src",
  "prompts",
  "personal-library-direction-extraction.system.md",
);

const logger = {
  warn: () => undefined,
  error: () => undefined,
  info: () => undefined,
  setSensitiveValues: () => undefined,
};

async function main(): Promise<void> {
  const indexPath = join(PLUGIN_TEST, "arxiv-daily", ".index");
  const data = JSON.parse(
    await readFile(join(PLUGIN_TEST, ".obsidian", "plugins", "arxiv-daily", "data.json"), "utf8"),
  );
  const llmSettings = data.settings?.llm;
  const catalog = decodePersonalLibraryCatalog(
    JSON.parse(await readFile(join(indexPath, "personal-library-catalog.json"), "utf8")),
  );
  if (!catalog) throw new Error("catalog invalid");
  const papers = selectPersonalLibraryDirectionPapers(catalog);
  console.log("batch papers:", papers.length, "| prompt:", PROMPT_PATH);

  const system = await readFile(PROMPT_PATH, "utf8");
  const user = renderPersonalLibraryExtractionUserMessage(papers.slice(0, 20));
  const client = new LlmClient(llmSettings, logger, new NodeHttpClient());
  const raw = await client.call(
    [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    { maxOutputCodeUnits: 64_000, maxCompletionTokens: 4_096 },
  );
  console.log("=== RAW MODEL OUTPUT (first 10000 chars) ===");
  console.log(raw.slice(0, 10_000));
  console.log("=== END (total length", raw.length, ") ===");
}

main().catch((error) => {
  console.error("FAIL:", error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
