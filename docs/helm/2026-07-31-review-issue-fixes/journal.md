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
