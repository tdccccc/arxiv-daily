# P3 — first-baseline

goal_ref: ../goal.md
created: 2026-10-15T12:35:00+08:00
updated: 2026-10-15T12:35:00+08:00
revision: 1

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
- [ ] accepted on observed evidence

### Chunk 3 — baseline reports

- change: `report-recorded` and a `--runs 3` rerun report exist for the labelled sheet; findings summarised for the user.
- kind: non-behavioral (operation, real model cost — confirm the printed cost line with the user first)
- first signal: n/a
- checks: both reports written; numbers spot-checked against the sheet.
- [ ] accepted on observed evidence

## Phase verification

- Goal success criterion 5 holds: ≥ 30 labelled papers and a first baseline report from a real-model run.

## Abort / reshape triggers

- If backfilled days still give fewer than about 8 selected papers in total, widen the date range before handing over the sheet rather than asking for labels on a sheet with too few positives.
