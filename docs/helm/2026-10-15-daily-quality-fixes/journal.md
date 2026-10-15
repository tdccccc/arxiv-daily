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
