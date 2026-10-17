#!/usr/bin/env node
import { loadQualityEval } from "./load.mjs";

const USAGE = `Usage: node scripts/quality-eval/evaluate.mjs --sheet <labels.csv> --out <dir> (--recorded | [--runs N] [--yes]) [options]

Scores a filled label sheet and writes report.md and report.json to <out>.

  --recorded            Score the filter decisions saved with the sheet. No model is called.
  (default)             Rerun the current filter on the labelled papers with the CLI config's model.
                        Without --yes it only prints how many model calls it would make, then stops.

  --sheet <file>        The filled labels.csv from export.mjs (required)
  --meta <file>         labels.meta.json (default: the one beside the sheet)
  --out <dir>           Where report.md and report.json go (required)
  --runs <n>            How many times to rerun the filter, 1 to 20 (default 1)
  --yes                 Make the model calls (they cost money)
  --force               Replace an existing report.md/report.json in --out
  --config <file>       CLI config.toml supplying the model settings (default: the CLI's own)
  --topics-from-config  Use the config's topics instead of the direction list the sheet was labelled with
`;

const qe = await loadQualityEval();
try {
  const args = qe.parseEvaluateArgs(process.argv.slice(2));
  if (args.help) { process.stdout.write(USAGE); process.exit(0); }
  const result = await qe.runEvaluate(args, { log: line => process.stdout.write(`${line}\n`) });
  if (!result.dryRun) {
    const { sheet } = result.report;
    process.stdout.write(`scored ${sheet.labelled} labelled rows (${sheet.unlabelled} unlabelled left out); wrote ${result.files.join(" and ")}\n`);
    if (result.allRunsFailed) {
      process.stderr.write("error: every model run failed; see the report for the reasons\n");
      process.exitCode = 1;
    }
  }
} catch (error) {
  process.stderr.write(`error: ${error.message}\n\n${USAGE}`);
  process.exitCode = 1;
}
