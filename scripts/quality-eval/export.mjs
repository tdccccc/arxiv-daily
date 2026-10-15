#!/usr/bin/env node
import { loadQualityEval } from "./load.mjs";

const USAGE = `Usage: node scripts/quality-eval/export.mjs --out <dir> [options]

Exports a label sheet from the filter decision records kept by daily runs.
No model is called; abstracts are fetched from arXiv (cached under <out>/abstract-cache).

  --out <dir>        Where labels.csv, labels.guide.md and labels.meta.json go (required)
  --vault <dir>      Vault root holding arxiv-daily/.index (default: vault_root of the CLI config)
  --config <file>    CLI config.toml supplying vault root and output layout (default: the CLI's own)
  --from <date>      First report date to include (YYYY-MM-DD)
  --to <date>        Last report date to include (YYYY-MM-DD)
  --skipped <n>      How many skipped papers to add to every selected one (default 50)
  --seed <text>      Sampling seed; the same seed and records give the same rows (default quality-eval)
  --force            Replace an existing labels.csv (it may contain labels)
`;

const qe = await loadQualityEval();
try {
  const args = qe.parseExportArgs(process.argv.slice(2));
  if (args.help) { process.stdout.write(USAGE); process.exit(0); }
  const result = await qe.runExport(args, {
    onProgress: message => process.stderr.write(`${message}\n`),
    onWarning: message => process.stderr.write(`warning: ${message}\n`),
  });
  process.stdout.write(`wrote ${result.rows} rows (${result.selected} selected, ${result.skippedSampled} skipped, ${result.withoutAbstract} without abstract) to ${result.outDir}\n`);
} catch (error) {
  process.stderr.write(`error: ${error.message}\n\n${USAGE}`);
  process.exitCode = 1;
}
