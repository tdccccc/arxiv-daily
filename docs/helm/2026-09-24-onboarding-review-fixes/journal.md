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

## 2026-09-28 — L1 adjust P5 for CLI credential permissions

- evidence: batch C found that `runInit` uses default `fs.writeFile` permissions for config.toml containing LLM and email credentials. With a usual 022 umask this creates 0644 files; overwriting or keeping an existing 0644 file also preserves broad access. Root suites currently pass (core 2052 + 2 skipped, node-runtime 45, CLI 75, plugin 751), relay tests 141 pass, and root lint/typecheck/build/boundaries/submission pass. Relay standalone typecheck cannot yet find its Cloudflare type dependency.
- change: add P5 chunk 8 (F28) with real-filesystem failing-first tests and private config persistence; outcome and initiative scope remain unchanged. Lower-priority batch C findings will be recorded without expanding implementation, per chunk 7.
- disposition: retain all existing changes. Reuse the already tested Node atomic writer; no delivery protocol or provider changes. Install the relay's locked dependencies to resolve the standalone verification environment.
- next: observe F28 Red, implement the narrow fix, observe Green and relevant regressions, then close review dispositions and Helm.

## 2026-09-28 — P5 accepted / initiative done

- evidence: P5 chunks 4–6 are accepted from their recorded Red/Green commits (8b4b2a6, 25889d7 + b6a06ce, c55e53f) and this session's complete regression checks. Batch C is recorded in the main review: F28 is the only new High, fixed in f6ef773. The three real-filesystem permission tests failed first (new 0664, existing 0644; expected 0600); after the fix all six focused tests pass, as do the full CLI 78 and node-runtime 45 tests. Keep-existing testing was corrected to assert retained settings rather than exact bytes because the existing merge function appends a comment; no merge behavior changed.
- verification: `NODE_OPTIONS=--max-old-space-size=8192 npm test` passed before F28 (core 2052 + 2 skipped, node-runtime 45, CLI 75, plugin 751). F28 only changes CLI init, so the later full CLI/node-runtime regressions cover its affected paths. Final root lint (0 errors / 21 existing warnings), typecheck, build, boundaries and Obsidian submission checks pass. Relay standalone typecheck and 141 tests pass after installing its existing locked dependencies. No dependency or lockfile changes. Core's two pre-existing skips remain; no live provider/arXiv/mail, real cron, production cutover, Windows ACL or fresh Obsidian UI checks were performed in this continuation.
- checkpoint: On track. The five already checked onboarding criteria retain their accepted P1–P4/P6 evidence, including the real Obsidian 1.13.7 checks. Every planned behavioral fix now has its own isolated commit and recorded failing-first evidence. No new behavior is attributed to the b6a06ce type-safety supplement; its evidence is Green typecheck and filter regression. Documentation-only transitions use diff and schema checks.
- dispositions / scoped waivers for the sixth success criterion (fix or explicitly waive review findings):
  - F5: no 1.13+ quick-start templates, per the user's explicit non-goal; F10's schedule step is implemented.
  - F19: waive implementation in this initiative as already prescribed by P5; changing provider thinking parameters requires the separate provider decision and live-contract verification. It remains a confirmed open issue, not a repaired provider integration.
  - F23/F24: retain the earlier deferral; incremental HTTP streaming is outside these local fixes, and surrogate truncation remains a low-impact plausible boundary issue without a required fix chunk.
  - Batch B cross-process RunLock/PaperIndexStore consistency: retain the planned deferral to shared locking work; no cross-process safety guarantee is claimed. Non-empty-topic migration remains PLAUSIBLE, with no confirmed regression accepted for fixing.
  - F27: accept the visibility fix; actual macOS/Windows automatic email remains the explicitly requested follow-up initiative.
  - F29–F32: waive implementation here under P5 chunk 7's rule that lower-priority batch C findings are recorded. Cron path quoting, invalid schedule windows, async command error handling and JSON-null validation each have concrete local evidence in the review for later work.
  - C2's remaining window-close variants and the additional theme/width UI matrix were not part of a confirmed fix chunk; their optional manual verification remains documented. Existing accepted real-Obsidian evidence is not replaced by an unperformed check.
- change: all P5 chunks checked, P5 and goal set to done; all seven goal success criteria are checked, with the scoped review waivers above. Branch remains `fix/onboarding-review`; no push, PR, deployment or change to local main.
- architecture-map result: status: updated; report: `docs/architecture-map.html`; scope: CLI init config persistence only; summary: documented private temporary-file replacement and POSIX modes without changing the renderer; evidence: `runCli` → `runInit` → `NodeStorageAdapter.writeTextAtomic` and `loadCliConfig`; validation: `--validate` valid, `--check` current. No claim of a full-map audit or new browser review.
- next: this initiative has no remaining required work. The previously chosen next initiative is cross-platform automatic email; it has not been started implicitly.
