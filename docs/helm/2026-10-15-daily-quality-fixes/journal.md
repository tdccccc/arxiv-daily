# Journal — daily quality fixes

## 2026-10-15T16:55+08:00 — decision: no system-prompt escape hint (P1)

- Context: the investigation showed a prompt-only "double every backslash"
  hint made the model drop LaTeX (`28±7%` instead of `$28\pm7$%`). With the
  pre-parse repair, first attempts should already parse.
- Decision (user, option A): do not change `daily-paper-summary.system.md` /
  `.en.system.md`; keep the prompt contract at 2 so cached checkpoints stay
  valid. The retry correction text carries the escape rule and asks to keep
  LaTeX.
- Effect: P1 Chunk 3 dropped (L1 plan edit).

## 2026-10-15T17:05+08:00 — decision: daily model switched to gpt-6.1-sol

- User asked to switch the real model; endpoint `/v1/models` lists
  `gpt-6.1-sol`. User chose option C: the daily CLI config
  (`~/.config/arxiv-daily/config.toml`, backup `config.toml.bak-2026-10-15`)
  now uses `gpt-6.1-sol`, and the P1 real-model acceptance runs 2610.16741 on
  both `gpt-6.1-sol` and `gpt-6-luna` (the root-cause model), each with
  consent before running.

## 2026-10-15T18:30+08:00 — decisions: P2 shape and a new P3 (RSS)

- Recon (`tmp/helm-reports/p2-recon.md`) found one process-wide request
  coordinator with a single cooldown for export.arxiv.org and arxiv.org, so an
  abs-page fallback would stall behind a query-API Retry-After cooldown.
- User decisions: per-host cooldowns (pacing stays shared); header-less 429
  on the query API → no retries, fail fast to the fallback; on arxiv.org ≥ 30 s
  backoff; no cross-process cooldown this time; fallback cap 500 network
  fetches per run (asked about token cost: page fetches use none; filter input
  returns to normal-day size).
- User asked whether their IP causes the throttling. Evidence: one query per
  day, compliant pacing; both direct and proxied egress were throttled on
  10-14 while abs pages stayed 200; a probe at 18:20 returned 200. Likely
  service-side or shared-egress throttling (inferred). User agreed to add P3:
  use arXiv RSS (one request per category, different host, latest day only)
  as the abstract source for today's run, with the query API and abs pages as
  gap fillers / backfill path. Added to goal as pending; success criteria for
  P3 are written when it starts.
