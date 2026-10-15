#!/usr/bin/env node
import { loadQualityEval } from "./load.mjs";

const USAGE = `Usage: node scripts/quality-eval/evaluate.mjs --sheet <labels.csv> --out <dir> --recorded [options]

Scores a filled label sheet and writes report.md and report.json to <out>.
--recorded scores the filter decisions saved with the sheet (no model call).

  --sheet <file>   The filled labels.csv from export.mjs (required)
  --meta <file>    labels.meta.json (default: the one beside the sheet)
  --out <dir>      Where report.md and report.json go (required)
  --recorded       Score the recorded decisions without calling a model
`;

const qe = await loadQualityEval();
try {
  const args = qe.parseEvaluateArgs(process.argv.slice(2));
  if (args.help) { process.stdout.write(USAGE); process.exit(0); }
  const result = await qe.runEvaluate(args);
  const { sheet } = result.report;
  process.stdout.write(`scored ${sheet.labelled} labelled rows (${sheet.unlabelled} unlabelled left out); wrote ${result.files.join(" and ")}\n`);
} catch (error) {
  process.stderr.write(`error: ${error.message}\n\n${USAGE}`);
  process.exitCode = 1;
}
