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
