#!/usr/bin/env node
import { loadQualityEval } from "./load.mjs";

const USAGE = `Usage: node scripts/quality-eval/summary-meta-review.mjs --judged <dir> --sources <dir> --config <cli toml> --reviewer-model <name> --reviewer-effort <effort> --out <dir> [options]
       node scripts/quality-eval/summary-meta-review.mjs --report-only --out <dir>

Has a second model re-check every issue the judge raised (summary-judge.mjs output): is the problem real, is the
severity right. One request per judged item that has issues. This calls a model and costs money.
Without --yes it only prints the plan and calls nothing.

  --judged <dir>           Output directory of summary-judge.mjs (required)
  --sources <dir>          Cached sources from summary-sources.mjs (required)
  --config <file>          CLI config.toml: its [llm] endpoint and key are used for the reviewer (required)
  --reviewer-model <name>  The reviewer's model name; must differ from the judge's (required)
  --reviewer-effort <e>    Reasoning effort, passed through as given, e.g. low, medium, high, xhigh (required)
  --vault <dir>            Vault root; needed only to rebuild the input of judgments of historical summaries
  --daily-dir <dir>, --papers-dir <dir>  Vault layout, relative to the vault
  --out <dir>              Where meta/, run.json, meta-review.md and meta-review.json go (required)
  --rereview               Review again even when the same reviewer, effort, prompt and judgment were reviewed
  --report-only            Rebuild meta-review.md and meta-review.json from <out> without any model
  --yes                    Make the model calls
`;

const qe = await loadQualityEval();
try {
  const args = qe.parseSummaryMetaReviewArgs(process.argv.slice(2));
  if (args.help) { process.stdout.write(USAGE); process.exit(0); }
  if (args.reportOnly) {
    const { files } = await qe.runMetaReportOnly(args);
    process.stdout.write(`wrote ${files.join(" and ")}\n`);
  } else {
    const result = await qe.runSummaryMetaReview(args, { log: line => process.stdout.write(`${line}\n`) });
    if (result.stopped) process.exitCode = 1;
  }
} catch (error) {
  process.stderr.write(`error: ${error.message}\n\n${USAGE}`);
  process.exitCode = 1;
}
