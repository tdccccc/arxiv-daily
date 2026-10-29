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

## 2026-10-29 — note (real judge trial; steer pending)

- evidence: The researcher approved a judge trial on claude-opus-5-5 (anthropic provider via an OpenAI-compatible proxy; omitting `provider` fails with "requires adaptive thinking"). It judged 4 items: daily 2605.20779, 2608.19669 and 2607.01175, and note 2607.01175. All 4 passed validation on the first try, using about 46–51k input and 2–4k output tokens each. Output is in `~/quality-eval/summary-judge-trial/`.
  - The judge caught all three errors known from P1, plus many major ones (17× written as 17%, a 60% increase written as a 30% decrease, a wrong benchmark name). The owner confirmed these against the cached source.
  - The same-input assumption breaks for historical outputs. 2608.19669 had v1 on 2026-08-20, was reported 2026-08-21, and got a heavily revised v2 on 2026-09-29; the cache holds v2, so most of its "errors" are probably revision differences. 2605.20779's v2 is dated the report day. 2607.01175 has only v1, but its note says its input lost digits, so the extraction then differed from now.
  - The cache holds versions v2 78, v3 14 and v4 4, about 20% of 472.
  - Judging fixed historical outputs also cannot give the before/after-prompt comparison named in Intent.
- change: none yet. Proposed L3: regenerate the sampled summaries with the current generator on the cached input, then judge those, so the input is identical and prompt or model changes can be compared. Historical judging becomes reference only, with multi-version papers flagged. Waiting on the researcher.
- disposition: keep all P2 code. The trial artifacts stay outside the repo.
- next: the researcher's decision on regeneration.

## 2026-10-29 — L3 steer

- evidence: See the previous entry. The researcher chose option A: "A吧".
- change:
  - Intent and a new success criterion now require regenerating the sampled summaries on the cached input with the current generator, then judging those. The constraint on same input became "the baseline judges regenerated summaries; historical judging is reference only, and multi-version papers are flagged".
  - P2 is done: tool complete, real trial verified. The old P3 never started, so it was re-phased in place: P3 is now regenerate and judge regenerated summaries, and P4 is calibration plus baseline.
- disposition: keep all code. Historical judging stays available as reference.
- next: P3 chunk 1, the regenerate command.

## 2026-10-29 — note (P3 phase verification)

- evidence: With the researcher's approval, the 4 trial items were regenerated with gpt-6.1-sol on the cached input: 4 generated, 1 request each, no fallbacks. Output is in `~/quality-eval/summary-regen-trial/`. Claude-opus-5-5 then judged them: 4 of 4 valid on the first try (`~/quality-eval/summary-judge-regen-trial/`). Verdicts, historical then regenerated:

  | Item | Historical | Regenerated |
  | --- | --- | --- |
  | daily 2605.20779 | 7 accuracy-major + 1 completeness-major | accuracy ok, 1 completeness-minor + 1 clarity-minor |
  | daily 2607.01175 | 3 accuracy-major + 5 minor | accuracy ok, 3 minor |
  | daily 2608.19669 | 4 accuracy-major + 4 accuracy-minor | no issues |
  | note 2607.01175 | 5 accuracy-major + 1 completeness-major | no issues |

  The owner checked regenerated 2608.19669 by hand: 9.5, 19 and 5.6 points all match the source.
- change: none.
- disposition: keep. The historical errors cannot be attributed to a single cause. Generation then used gpt-6-luna until 2026-10-15, older prompts and older extraction; the current setup is accurate on these 4. n=4 and the papers were chosen for their known errors, so the P4 baseline must confirm this.
- next: close P3 and start P4 (baseline run, then calibration on its issues).

## 2026-10-29 — note (baseline run)

- evidence:
  - Regeneration: 40 daily entries and 10 notes, seed "summary-judge", sampled in proportion by month, regenerated with gpt-6.1-sol: 50 generated, 0 fallbacks, 0 errors.
  - Judging with claude-opus-5-5: the first pass judged 29 and stopped on 11 errors. All 11 were "HTTP network failure" about 24 s into a request, from the proxy, not the client, whose idle timeout is 120 s. Resuming judged the other 21 with 0 errors. In total 50 of 50 are valid (47 first try, 3 after retry).
  - Daily, 40 entries:

    | Dimension | ok | minor | major |
    | --- | --- | --- | --- |
    | accuracy | 37 | 3 | 0 |
    | completeness | 13 | 25 | 2 |
    | clarity | 23 | 17 | 0 |

  - Notes, 10:

    | Dimension | ok | minor | major |
    | --- | --- | --- | --- |
    | accuracy | 8 | 2 | 0 |
    | completeness | 6 | 4 | 0 |
    | clarity | 4 | 6 | 0 |

  - Completeness issues sit almost entirely in mainResult (36 minor, 2 major). Clarity issues are mostly undefined abbreviations.
  - The owner read all 7 accuracy and major issues; they are well grounded, and the severities are sometimes strict.
  - Report: `~/quality-eval/summary-baseline-2026-10-29/judged/report.md`.
- change: none.
- disposition: keep. The review page has 15 cards, at `~/quality-eval/summary-baseline-2026-10-29/review/review.html`.
- next: the researcher reviews the page, then results are imported for the agreement rate.
