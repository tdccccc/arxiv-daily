# Journal — summary quality evaluation

## 2026-10-28 — note

- evidence: Chunk 3's real-vault run flagged `$6:1$`, `$8:1:1$` and `$g \in (0, 0.15]$`, which render fine in Obsidian. A probe of core's `canonicalizeScientificMarkdownMath` confirms the two forms. A digit-leading colon ratio gives `unclosed-delimiter`, because a `$` before a digit is read as currency. A half-open interval gives `unbalanced-parentheses`. `$0.5$`, `$10^{10}\,M_{\odot}$`, `$0.8$–$0.95$`, `$1.2\pm0.3$` and `$3\sigma$` pass. Daily summaries use the same validator, so a summary that writes either form is rejected, retried and may fall back.
- change: none to this initiative. The offline report labels these patterns as known core-rule false positives and does not count them as defects.
- disposition: product follow-up, outside this initiative: relax the validator for ratios and half-open intervals, with tests.
- next: P1 chunk 4 (report and command).

## 2026-10-28 — L1 adjust

- evidence: The offline trial shows mainResult far over the 80–180 guideline in 2026-08 and 2026-09 (median 432 and 364), and back within range in 2026-10 (median 110, 94.9% inside). The researcher said: "摘要偏长问题不大，只要能确保各个条目写的准确清楚完整".
- change: Length becomes reference data in the offline report and is no longer listed as a problem. The goal's judge criterion now names three dimensions explicitly: accuracy, completeness and clarity.
- disposition: Keep the length measures and the per-month table. Drop the length list from the report's problem list (chunk 4 tweak).
- next: Finish P1 chunk 4, then chunks 5–6.

## 2026-10-28 — note (P1 phase verification)

- evidence: The source fetch got all 472 papers (html 460, latex 3, abs only 9) with 0 failures. The formal offline report is in `~/quality-eval/summary-2026-10-28-offline/`. The owner spot-checked 10 numbers drawn with a fixed seed from 517 unfound ones:
  - 2 were normalisation misses: `.028` without a leading zero, and `4,454,600.` broken by the comma-list rule at a sentence end. Both are now fixed, with tests.
  - 2 are likely real errors: 2608.19669 says VLPO gained 1.8 points where the source says +0.9, and 2607.01175 says 87–89% where the source says 87%.
  - 6 are absent from the model's input, with no nearby value. They are probably table values, which extraction drops, or numbers the model derived or invented.
  - The earlier partial trial also confirmed 2605.20779, which says 11–300 GeV where the source says 1–300 GeV.
  - The false-positive rate is 2 in 10, well under the abort trigger of about half.
- change: none; the check stays a metric, with its list marked for human review.
- disposition: keep. After the fixes, 97.1% of daily numbers and 96.0% of note numbers are found; 84 of 462 daily entries and 18 of 46 notes have at least one number not found.
- next: close P1 and plan P2 (the judge). P2 needs the researcher's judge model config file.
