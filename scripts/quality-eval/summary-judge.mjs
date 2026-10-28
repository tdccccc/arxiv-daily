#!/usr/bin/env node
import { loadQualityEval } from "./load.mjs";

const USAGE = `Usage: node scripts/quality-eval/summary-judge.mjs --judge-config <toml> --sources <dir> --out <dir> [options]
       node scripts/quality-eval/summary-judge.mjs --report-only --out <dir> [--offline <findings.json>]

Has a separately configured model judge a sample of daily entries and detail notes against the source text
the generator saw (cached by summary-sources.mjs): accuracy, completeness, clarity, with quoted evidence.
This calls a model and costs money. Without --yes it only prints the plan and calls nothing.

  --judge-config <file>  [llm]-only TOML for the judge model: api_key, base_url, model, optional provider,
                         thinking_mode, reasoning_effort (required). Must name a different model from the generator.
  --sources <dir>        Cached sources from summary-sources.mjs (required)
  --out <dir>            Where judgments/ and run.json go (required)
  --config <file>        CLI config.toml: the generator's model (for the different-model check), topic names, vault layout
  --generator-model <n>  The generator's model name, when no CLI config can say
  --vault <dir>          Vault root (default: vault_root of the CLI config)
  --daily-dir <dir>      Daily reports, relative to the vault
  --papers-dir <dir>     Detail notes, relative to the vault
  --daily <n>            Daily entries to judge, spread over the months in proportion (default 20)
  --notes <n>            Detail notes to judge (default 5); only notes written under the current prompt
  --include-older-notes  Also sample notes written under earlier prompts
  --seed <text>          Sampling seed; the same seed gives the same sample
  --ids <a,b,...>        Judge exactly these papers (their entry and note); overrides sampling
  --from <date>, --to <date>  Only reports in this range (YYYY-MM-DD)
  --offline <file>       findings.json of summary-check.mjs run with --sources: the report compares its unfound numbers with the judgments
  --report-only          Rebuild report.md and judgments.json from <out> without a judge config, a model or any request
  --rejudge              Judge again even when a judgment for the same judge, prompt and source exists
  --yes                  Make the model calls
`;

const qe = await loadQualityEval();
try {
  const args = qe.parseSummaryJudgeArgs(process.argv.slice(2));
  if (args.help) { process.stdout.write(USAGE); process.exit(0); }
  if (args.reportOnly) {
    const { files } = await qe.runJudgeReportOnly(args);
    process.stdout.write(`wrote ${files.join(" and ")}\n`);
  } else {
    const result = await qe.runSummaryJudge(args, { log: line => process.stdout.write(`${line}\n`) });
    if (result.stopped) process.exitCode = 1;
  }
} catch (error) {
  process.stderr.write(`error: ${error.message}\n\n${USAGE}`);
  process.exitCode = 1;
}
