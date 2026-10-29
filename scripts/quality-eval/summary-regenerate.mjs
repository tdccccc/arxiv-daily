#!/usr/bin/env node
import { loadQualityEval } from "./load.mjs";

const USAGE = `Usage: node scripts/quality-eval/summary-regenerate.mjs --sources <dir> --out <dir> [options]

Writes the sampled papers' summaries again with the current generator model and prompts, on the cached source
text (summary-sources.mjs), through the daily pipeline's own functions. The result is what the judge
(summary-judge.mjs --summaries) evaluates. This calls the generator model and costs money.
Without --yes it only prints the plan and calls nothing.

  --sources <dir>        Cached sources from summary-sources.mjs (required)
  --out <dir>            Where summaries/ and run.json go (required)
  --config <file>        CLI config.toml: the generator model ([llm]), topic names, summary language, vault layout (default: the CLI's own)
  --vault <dir>          Vault root (default: vault_root of the CLI config)
  --daily-dir <dir>      Daily reports, relative to the vault
  --papers-dir <dir>     Detail notes, relative to the vault
  --daily <n>            Daily entries to regenerate, spread over the months in proportion (default 20)
  --notes <n>            Detail notes to regenerate (default 5); only notes written under the current prompt
  --include-older-notes  Also sample notes written under earlier prompts
  --seed <text>          Sampling seed; the same seed as the judge gives the same papers
  --ids <a,b,...>        Exactly these papers (their entry and note); overrides sampling
  --from <date>, --to <date>  Only reports in this range (YYYY-MM-DD)
  --regenerate           Generate again even when a result for the same model, prompt and source text exists
  --yes                  Make the model calls
`;

const qe = await loadQualityEval();
try {
  const args = qe.parseSummaryRegenerateArgs(process.argv.slice(2));
  if (args.help) { process.stdout.write(USAGE); process.exit(0); }
  const result = await qe.runSummaryRegenerate(args, { log: line => process.stdout.write(`${line}\n`) });
  if (result.stopped) process.exitCode = 1;
} catch (error) {
  process.stderr.write(`error: ${error.message}\n\n${USAGE}`);
  process.exitCode = 1;
}
