# P3 — first-baseline

goal_ref: ../goal.md
created: 2026-10-15T12:35:00+08:00
updated: 2026-10-18T11:35:00+08:00
revision: 2

## Outcome

The researcher has labelled at least 30 papers from real filter decision records, and the first baseline report exists for both the recorded decisions and a real-model rerun.

## Assumptions

- The researcher's real interests are the two topics used in the trial runs (Photo-z; Galaxy Cluster, one direction each), on astro-ph.CO + astro-ph.GA — confirmed by the user 2026-10-15.
- Backfilling five announcement days with real CLI runs in a temp vault (user consented to the cost of about five daily runs) yields roughly 10–20 selected papers; with 40 skipped samples that is a sheet of 50–60 rows.
- arXiv may keep throttling the query API, so backfilled records are likely titles-only (`hadAbstract=false`); the rerun with sheet abstracts then doubles as a titles-vs-abstracts comparison.

## Approach

Backfill 2026-10-08, 09, 12, 13 into the trial vault that already holds 2026-10-14, export one sheet per day range (single direction snapshot, so rerun works by default), hand it to the researcher, then run `evaluate --recorded` and a `--runs 3` rerun after the user confirms the cost line.

## Chunks

### Chunk 1 — backfill records and export the sheet

- change: five days of filter decision records exist in the trial vault; one sheet with every selected candidate and 40 sampled skipped ones is exported with abstracts.
- kind: non-behavioral (operation)
- first signal: n/a
- checks: each run completes; record count per day; sheet row count, non-empty abstracts, single direction snapshot.
- L1: 2026-10-13 is not an announcement day in /recent (available: 10-08, 09, 12, 14, 15), so 10-15 replaced it.
- [x] accepted on observed evidence — records: 10-08 40 (1 in report), 10-09 52 (2), 10-12 32 (1), 10-14 100 (2), 10-15 68 (2); 10-14 and 10-15 titles-only (API 429 / network failure), the others had abstracts. Sheet `/home/tiandc/quality-eval/2026-10-15-baseline/labels.csv`: 48 rows (8 selected + 40 of 284 skipped, seed `baseline-1`), 0 without abstract (abs-page fallback), one direction snapshot across all five days; source records copied beside it

### Chunk 2 — researcher labels

- change: the researcher fills `label` / `label_direction` for at least 30 rows.
- kind: non-behavioral (user work)
- first signal: n/a
- checks: `evaluate --recorded` accepts the sheet without row errors; labelled count ≥ 30.
- [x] accepted on observed evidence — 2026-10-18: the researcher labelled all 48 rows in a local web labeller (`~/quality-eval/labeler.html`, outside the repo) and pasted the result. The owner applied it with the labeller's own CSV writer: fingerprint `eeb0e8cb` matched, 48/48 row/id pairs matched, and only the label cells changed (+51 bytes). The unlabelled original is kept as `labels.unlabelled.csv`. `evaluate --recorded` scored 48 rows with 0 errors (3 y, all direction 2; 45 n).

### Chunk 3 — baseline reports

- change: `report-recorded` and a `--runs 3` rerun report exist for the labelled sheet; findings summarised for the user.
- kind: non-behavioral (operation, real model cost — confirm the printed cost line with the user first)
- first signal: n/a
- checks: both reports written; numbers spot-checked against the sheet.
- [x] accepted on observed evidence — 2026-10-18, user-approved (3 model calls; the dry run's cost line was shown first). Both reports are in `~/quality-eval/2026-10-15-baseline/`:
  - **Recorded decisions** (`report-recorded/`; gpt-6-luna, 28 of 48 rows filtered on title only): precision 37.5% (3/8), recall 100% (3/3), direction 100%. The 5 false positives are 2610.10901, 12322, 12595, 19079 and 19317.
  - **Rerun** (`report-run-1/`; gpt-6.1-sol, abstracts for all rows, `--runs 3`): precision 60% (3/5), recall 100%, direction 100%. The 3 runs agreed on all 48 rows. The false positives left are 2610.10901 (score 95, galaxy-cluster) and 2610.12322 (score 95, photo-z).
  - **Rows that had an abstract at filter time** (20): precision 25% recorded vs 33% rerun. Most of the gain comes from having abstracts, not from the model change.
  - **Observation:** runs 1–2 answered only 5 of 48 papers, and the evaluator treated the 43 unmentioned papers as skip; run 3 listed all 48. The outcome was the same. Whether the daily filter handles such omissions the same way was not checked.
  - **Caveat:** only 3 positives, all galaxy-cluster, and none for photo-z. The baseline is a comparison point, not an accuracy estimate.

## Phase verification

- Goal success criterion 5 holds: ≥ 30 labelled papers and a first baseline report from a real-model run.
  - Result (2026-10-18): holds. 48 rows are labelled, and both the recorded-decision report and the `--runs 3` real-model rerun report exist (see Chunks 2–3).

## Abort / reshape triggers

- If backfilled days still give fewer than about 8 selected papers in total, widen the date range before handing over the sheet rather than asking for labels on a sheet with too few positives.
