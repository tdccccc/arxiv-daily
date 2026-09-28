# Journal

## 2026-09-24 — note

- evidence: P5's three planned chunks landed (frontmatter YAML quoting, frontmatter refresh keeps user properties, dates older than /recent fail permanently). The last one changes a pinned contract: two pipeline tests expected `failed_transient` for such dates; with the submittedDate fallback long gone nothing can fetch them, so they now expect `failed_permanent`.
- change: P5 blocked while the review agent's core / CLI / relay batches are still outstanding; P6 (low-priority plugin items) started meanwhile.
- disposition: the `extra_body` thinking-parameter finding (review) is not changed without real provider calls; it goes to the user as a decision. The idle-timeout and surrogate-truncation findings are recorded in the review and not fixed here.
- next: P6 chunk 1 (enable + Run today does not block); resume P5 when the review agent reports.

## 2026-09-25 — L1 adjust

- evidence: the review agent delivered batches A and B, then died on API 503 before batch C (transcript `ad7dc58f…/subagents/agent-acore-reviewer-*`). Not yet handled from A/B, re-checked against code: a filter answer in a ```` ```json ```` fence fails the day (`paper-filter.ts` strict `JSON.parse`); a crash-left `running` date is recovered as `failed_permanent` and never retried automatically (`state-store.ts` `recoverStaleRunning`); automatic email needs exclusive create, which both the plugin and node adapters only provide on Linux, while the test send skips that check. Cross-process gaps (in-memory `RunLock`, `PaperIndexStore` without readback) are real but need cross-process locking. Settings migration passing a non-empty `topics` array through is only PLAUSIBLE.
- change: P5 unblocked and extended with chunks 4–7 (crash retry, single-fence filter answers, visible email gap, in-session batch C review). Goal gets a non-goal for cross-platform email and a 2026-09-25 decision line.
- disposition: no code discarded. The 08-10 strict-JSON contract is loosened only by one outer fence (user decision). Cross-process locking and the migration item are recorded in the review, not fixed. Cross-platform email is the next initiative.
- next: P5 chunk 4 — failing state-store test for crash recovery.

## 2026-09-28 — resume / owner handoff

- evidence: user asked to finish the latest Helm. Branch `fix/onboarding-review` is clean and the earlier session has paused after c55e53f. P1–P4 and P6 are accepted; P5 chunks 4–6 have commits (8b4b2a6, 25889d7 + b6a06ce, c55e53f) with recorded Red/Green evidence but unchecked plan entries.
- change: ownership passes to `codex-root-session-2026-09-28`; continue P5 in this session without subagents, preserving all previous user decisions and the no-push constraint.
- disposition: retain existing code and tests. Revalidate the committed fixes; do not recreate their historical Red. No live provider calls or cross-platform email implementation are added.
- next: review batch C (CLI, node-runtime, relay), record findings and any required fix chunks, run the complete acceptance checks, then reconcile review dispositions and close the initiative if all criteria are met.
