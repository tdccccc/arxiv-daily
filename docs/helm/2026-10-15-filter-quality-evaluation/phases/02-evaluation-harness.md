# P2 — evaluation-harness

goal_ref: ../goal.md
created: 2026-10-15T11:45:00+08:00
updated: 2026-10-15T11:50:00+08:00
revision: 2

## Outcome

A developer can export a label sheet from a vault's filter decision records, a researcher can fill it in a spreadsheet or text editor, and one command scores either the recorded decisions (no model call) or a fresh real-model filter run against those labels, writing a Markdown + JSON report that is comparable across prompt and model changes.

## Assumptions

- The dev-script precedent holds: TypeScript under `scripts/`, bundled with esbuild at run time from a small `.mjs` entry (as `scripts/acceptance/exploration.mjs` does for `exploration-model.ts`), importing `@arxiv-daily/core` / `@arxiv-daily/node-runtime` and reading LLM settings and topics from the CLI config via `apps/cli/src/config-path.ts` + the CLI config loader. Verify before building.
- Core already exposes what a rerun needs: `buildPaperFilterRequest`, `decodePaperFilterRecords`, `classifiableTopics`, `LlmClient`; no core change is needed. If one is, it must be a small export, not a parallel implementation.
- arXiv abstracts can be fetched by id in batches (`export.arxiv.org/api/query?id_list=`), but the endpoint rate-limits (HTTP 429 seen twice on 2026-10-15); export needs backoff and a local cache so a retry does not refetch.
- CSV (UTF-8 with BOM, RFC 4180 quoting) opens correctly in common spreadsheets and survives Chinese text and multi-line abstracts.
- The filter's positive class is "category is not skip" (in_report, cut_by_limit and excluded_ignored all count as selected by the filter); the daily cap and ignore marks are not filter decisions.

## Approach

`scripts/quality-eval/` with two commands. `export` reads `<vault>/.index/filter-decisions/*.json` for a date range, samples (every selected candidate plus a seeded random sample of skipped ones up to a size), fetches abstracts, and writes `labels.csv` plus `labels.guide.md` (Chinese instructions and the numbered direction legend) and `labels.meta.json` (direction snapshot, source dates, prompt/result contract versions, models, seed). The researcher fills `label` (y/n) and `label_direction` (legend number) per row. `evaluate` reads the filled sheet and either scores the recorded `model_*` columns (`--recorded`) or reruns the real filter on the labelled papers with the current CLI config (`--runs N` for repeat runs), then writes `report.md` + `report.json`.

## Chunks

### Chunk 1 — label sheet export

- change: `export` produces the three files from a vault's records; sampling is deterministic for a seed and includes every selected candidate; abstracts come from arXiv with backoff and an on-disk cache, and rows whose abstract cannot be fetched are kept with an empty abstract and a warning; CSV round-trips commas, quotes, newlines and Chinese text; the guide lists directions numbered from the snapshot.
- kind: behavior
- first signal: Red — new script tests (fixture vault with v2 records, fake fetch) fail because the export module does not exist.
- checks: focused tests Green without network; the repo's script test command stays Green.
- [x] accepted on observed evidence — delegate Red (`ERR_MODULE_NOT_FOUND` for load.mjs) → 12/12 Green; `npm run test:release-tools` (CI "Test release tools") runs the new tests, 853 pass; boundaries, lint, product-units Green; owner re-ran the focused tests 12/12. Real smoke wrote a correct 42-row sheet but every abstract was empty: export.arxiv.org API answered 429/503 (likely IP/proxy-level throttle) while arxiv.org/abs answered 200 — follow-up in Chunk 5

### Chunk 2 — score recorded decisions

- change: `evaluate --recorded` reads a filled sheet and reports precision, recall and F1 of "selected", direction accuracy among true positives, per-topic breakdown, precision/recall at score thresholds (selected iff score ≥ t, for a fixed list of t), counts of unlabelled rows and of `hadAbstract=false` rows (reported separately and also excluded in a second metric line), and a disagreement list (false positives and false negatives with title and score). Rejects unknown label values and unknown direction numbers with a row-numbered error.
- kind: behavior
- first signal: Red — metric tests on a hand-computed fixture fail because the module does not exist.
- checks: focused tests Green; script test command Green.
- [ ] accepted on observed evidence

### Chunk 3 — rerun with the real filter

- change: `evaluate` without `--recorded` builds the filter request through core for the labelled papers (with the sheet's abstracts) using topics and LLM settings from the CLI config, calls the model `--runs N` times, decodes with core's validator (omitted papers count as skip), and reports the same metrics per run plus run-to-run agreement; the report header names prompt contract version, model, provider, date and sheet; if the config's directions differ from the sheet's snapshot (by text), the report says so prominently and direction accuracy is withheld. Never prints the API key. A cost line states the number of model calls before they are made, and `--yes` is required to make them.
- kind: behavior
- first signal: Red — tests with a fake LLM client fail because the rerun path does not exist.
- checks: focused tests Green; script test command Green; no network or model calls in tests.
- [ ] accepted on observed evidence

### Chunk 4 — usage guide

- change: `scripts/quality-eval/README.md` (Chinese) explains export → label → evaluate, where files go, what the metrics mean, the cost of rerun mode and that it is not wired into CI.
- kind: non-behavioral
- first signal: n/a
- checks: read-through; commands in the guide match the implemented flags.
- [ ] accepted on observed evidence

### Chunk 5 — abstract fallback through the abs page (L1, from the real smoke)

- change: export fills abstracts the API batch could not supply by fetching `arxiv.org/abs/<id>` one at a time through core's existing `ArxivFetcher.fetchPaperAbsPage` (spacing and cache included) and core's abstract extraction, exposed from core as a small pure helper rather than copied; cached results make reruns free; rows still empty after both routes keep the warning.
- why: on 2026-10-15 the export API rejected even single requests for 20+ minutes while the abs page answered; without abstracts the sheet cannot be labelled.
- kind: behavior
- first signal: Red — export test where the fake API fails and the fake abs page returns a known abstract expects that abstract in the CSV.
- checks: focused tests Green; core tests touching paper-content Green; core typecheck; real smoke against the 2026-10-14 record yields non-empty abstracts.
- [x] accepted on observed evidence — delegate Red (3 script cases; `extractAbsPageAbstract is not a function` ×3) → script 15/15, core helper files 33 passed; core suite 2491, typecheck, lint, boundaries, release tools 856 Green; owner re-ran script tests, core helper files, CLI 457 and plugin 1048 Green. Real smoke: 42/42 abstracts via abs pages in ~3 min with the API still timing out; cached rerun 0.23 s, no network

## Phase verification

- Export against the real trial vault record (2026-10-14, 100 candidates) fetches abstracts from arXiv and writes the three files; the CSV opens with correct columns (spot-check by parsing it back).
- `evaluate --recorded` on that sheet with owner-filled placeholder labels (clearly marked as not researcher labels, discarded afterwards) produces a report whose counts match a hand count.
- Rerun mode is exercised with the fake client only; the first real rerun belongs to P3 with the user's consent.

## Abort / reshape triggers

- If the rerun cannot reuse core's request builder and validator without copying them, stop and decide on a small core export (L1) rather than duplicating prompt logic.
- If arXiv rate limits make export impractical even with backoff and cache, reshape (L2): e.g. capture abstracts at filter time instead.
