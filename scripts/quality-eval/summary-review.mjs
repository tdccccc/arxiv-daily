#!/usr/bin/env node
import { loadQualityEval } from "./load.mjs";

const USAGE = `Usage: node scripts/quality-eval/summary-review.mjs --judgments <dir> --out <dir> [options]
       node scripts/quality-eval/summary-review.mjs --import <results.json> --out <dir>

Makes a self-contained Chinese web page (review.html) where a person agrees or disagrees with the judge's issues,
and later turns the exported answers into an agreement rate. No model is called and nothing is fetched.

  --judgments <dir>   Output directory of summary-judge.mjs; issues are taken from here first (required unless --meta)
  --meta <dir>        Output directory of summary-meta-review.mjs: a dispute page with the issues the second model disagreed with or
                      was unsure about (all of them unless --count), plus a small spot-check of the ones it agreed with
  --dispute-sample <n> With --meta: show only n disputes: all accuracy disputes and all on judge-major issues, the rest shared out over the
                      other dimensions in proportion (seeded); the calibration then extrapolates per stratum
  --agreed-sample <n> With --meta: how many agreed issues to add as a spot-check (default 3; 0 for none)
  --extra <dir>       Another judge output directory, used to fill up to --count when --judgments has fewer issues
  --sources <dir>     Cached sources (summary-sources.mjs): shows about 300 characters of source around each quote
  --vault <dir>       Vault root, only to show the full text of historical summaries the judgment did not keep
  --count <n>         How many issues to show (default 12); majors first, spread over papers and dimensions
  --seed <text>       Selection seed (default "review")
  --out <dir>         Where review.html and review.json go (required)
  --import <file>     The results file exported from the page: writes calibration.md and calibration.json to --out
`;

const qe = await loadQualityEval();
try {
  const args = qe.parseSummaryReviewArgs(process.argv.slice(2));
  if (args.help) { process.stdout.write(USAGE); process.exit(0); }
  if (args.importPath) {
    const { files, calibration } = await qe.runReviewImport({ resultsPath: args.importPath, outDir: args.outDir });
    const rate = calibration.overall.rate;
    process.stdout.write(`agreement ${rate === null ? "n/a" : `${(rate * 100).toFixed(1)}%`} (${calibration.overall.agree} agree, ${calibration.overall.disagree} disagree, ${calibration.overall.unsure} unsure, ${calibration.unanswered} unanswered); wrote ${files.join(" and ")}\n`);
  } else {
    const result = await qe.runSummaryReview({ ...(args.metaDir ? {} : { count: 12 }), seed: "review", ...args });
    process.stdout.write(`wrote ${result.cards} cards${result.shortfall ? ` (${result.shortfall} fewer than asked: not enough issues)` : ""}: ${result.files.join(" and ")}\n`);
  }
} catch (error) {
  process.stderr.write(`error: ${error.message}\n\n${USAGE}`);
  process.exitCode = 1;
}
