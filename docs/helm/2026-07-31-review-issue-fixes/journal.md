## 2026-07-31 — L1 adjust

- evidence: deleting the `dateWindowNote` option from
  `MarkdownWriter.writeDaily` broke markdown-writer.test.ts:351
  ("writeDaily includes submitted-date fallback notes"); initial grep had
  excluded tests, so the consumer was hidden
- change: reverted markdown-writer.ts to original; T1 now only removes the
  dead `const dateWindowNote = undefined` binding in pipeline.ts and the
  `dateWindowNote` property from the writeDaily call
- disposition: writer option + render hook + test kept (tested dormant
  capability; consider wiring it up or removing it in a later phase)
- next: re-run checks, commit P1

## 2026-07-31 — note

- evidence: P1 completed (PR #3, commits 60bccfe..a125cf6, all checks green)
- change: phase 01-cleanup → done; P2 → blocked
- disposition: P2 waits for the bug-fix branch to merge into main (conflict
  surface: pipeline.ts, view.ts); P3 waits for a product decision
- next: when the bug-fix PR merges, rebase P2 worktree and start view.ts split

## 2026-07-31 — note (P2 dependencies + P3 evidence)

- evidence: user named the bug-fix branches — `fix/arxiv-failure-recovery`
  and `fix/arxiv-request-resilience`; also closed the two stale active goals
  (email-delivery-beta done; cli-product-config done with waived criterion)
- change: goal.md constraint now names both fix branches as the P2 gate;
  while closing cli-product-config, its 07-28 journal revealed the P3
  decision already exists: TOML deliberately omits detail_selection/profile
  (balanced + per-topic `detail`)
- disposition: P3 likely closes as documented-as-fixed pending user
  confirmation
- next: P2 starts after the fix branches merge; P3 confirm with user
