#!/usr/bin/env node
import { loadQualityEval } from "./load.mjs";

const USAGE = `Usage: node scripts/quality-eval/summary-sources.mjs --sources <dir> [options]

Fetches and caches the source text the summaries were written from (the daily pipeline's own fetch:
rendered HTML, then LaTeX source, then the abs page), one <arXiv id>.json per paper in <dir>.
This makes many requests to arXiv and calls no model. Without --yes it only prints the plan.

  --sources <dir>     Where the cache goes (required)
  --vault <dir>       Vault root (default: vault_root of the CLI config)
  --config <file>     CLI config.toml supplying vault root, output layout and the section/paper budgets (default: the CLI's own)
  --daily-dir <dir>   Daily reports, relative to the vault (default: the config's, else arxiv-daily/daily)
  --papers-dir <dir>  Detail notes, relative to the vault (default: the config's, else arxiv-daily/papers)
  --from <date>       First report date to include (YYYY-MM-DD)
  --to <date>         Last report date to include (YYYY-MM-DD)
  --fallbacks         Also fetch the papers whose daily summary fell back to the abstract
  --limit <n>         Fetch at most n papers in this run (for a small trial)
  --refresh           Fetch papers again even when they are already cached
  --yes               Make the requests
`;

const qe = await loadQualityEval();
try {
  const args = qe.parseSummarySourcesArgs(process.argv.slice(2));
  if (args.help) { process.stdout.write(USAGE); process.exit(0); }
  const result = await qe.runSummarySources(args, { onProgress: line => process.stderr.write(`${line}\n`) });
  if (result.dryRun) {
    process.stdout.write(`${qe.describeSourcesPlan(result.plan)}\n`);
  } else {
    const { byPath } = result;
    process.stdout.write(`fetched ${result.fetched} (html ${byPath.html}, latex ${byPath.latex}, abs only ${byPath.abs}), already cached ${result.skipped}, failed ${result.failed}, not attempted ${result.remaining}; cache in ${result.sourcesDir}\n`);
    if (result.stopped === "rate-limit") process.stderr.write(`stopped: arXiv asked for a later retry. Run the same command again later to resume (${result.remaining} not attempted).\n`);
    if (result.stopped === "network") process.stderr.write(`stopped after repeated failures; see failures.json in ${result.sourcesDir}. Run the same command again to resume.\n`);
    if (result.stopped) process.exitCode = 1;
  }
} catch (error) {
  process.stderr.write(`error: ${error.message}\n\n${USAGE}`);
  process.exitCode = 1;
}
