#!/usr/bin/env node
import { loadQualityEval } from "./load.mjs";

const USAGE = `Usage: node scripts/quality-eval/summary-check.mjs --out <dir> [options]

Checks the daily summaries and detail notes already written in a vault, offline.
No model is called and nothing is fetched. Writes report.md and findings.json to <out>.

  --out <dir>         Where report.md and findings.json go (required)
  --vault <dir>       Vault root (default: vault_root of the CLI config)
  --config <file>     CLI config.toml supplying vault root and output layout (default: the CLI's own)
  --daily-dir <dir>   Daily reports, relative to the vault (default: the config's, else arxiv-daily/daily)
  --papers-dir <dir>  Detail notes, relative to the vault (default: the config's, else arxiv-daily/papers)
  --from <date>       First report date to include (YYYY-MM-DD)
  --to <date>         Last report date to include (YYYY-MM-DD)
  --force             Replace an existing report.md/findings.json in --out
`;

const qe = await loadQualityEval();
try {
  const args = qe.parseSummaryCheckArgs(process.argv.slice(2));
  if (args.help) { process.stdout.write(USAGE); process.exit(0); }
  const { report, files } = await qe.runSummaryCheck(args);
  const { counts } = report.corpus;
  process.stdout.write(`checked ${counts.entries} daily entries (${counts.fallbacks} fallbacks) and ${counts.notes} detail notes; wrote ${files.join(" and ")}\n`);
} catch (error) {
  process.stderr.write(`error: ${error.message}\n\n${USAGE}`);
  process.exitCode = 1;
}
