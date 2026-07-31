# arXiv failure recovery journal

## 2026-07-30 — initiative opened

- Created `fix/arxiv-failure-recovery` and its isolated worktree from `fix/arxiv-request-resilience` at `1debd02`.
- Locked the goal around transport recovery, atomic workflows, metadata single-flight, monotonic retry policy, and end-to-end state consistency.
- Preserved the explicit constraint not to commit or push without user instruction.

## 2026-07-30 — P1 transport deadlines

- Added host-neutral typed network/timeout failures and strict retry classification.
- Added Node physical abort plus logical deadlines covering body reads.
- Added Obsidian logical timeout/cancellation races with late settlement consumption.
- Added a defensive core watchdog so non-conforming adapters cannot permanently retain the process-wide arXiv queue.

## 2026-07-30 — P2 atomic workflows

- Made multi-category discovery atomic and prevented enrichment or pipeline mutation after partial discovery.
- Added verified-detail, replaceable-stub, user-conflict, and identity-mismatch handling across manual and daily repair paths.
- Added network-free reconciliation of existing verified detail notes and content fallback from HTML to source and `/abs`.

## 2026-07-30 — P3 metadata concurrency

- Added process-wide per-canonical-ID metadata single-flight, including overlapping sets and multi-batch partial-success semantics.
- Hardened persistent metadata cache validation and added storage-object/root-scoped serialization.
- Added bounded operation leases and cancellation-aware logical waits so one hung storage operation cannot block unrelated roots or future progress permanently.
- A storage adapter that ignores cancellation may leave orphan physical I/O after lease expiry; this is an intentional host limitation, while logical callers and queues remain recoverable.

## 2026-07-30 — P4 retry and scheduler consistency

- Made request spacing and cooldown deadlines monotonic; wall time is used only to convert a received HTTP-date once.
- Preserved long Retry-After minimums without hour-scale timers by retaining process-local cooldown state and returning a typed transient deferral.
- Normalized retry exhaustion so returned result, persisted state, history, progress, logs, and plugin UI agree.

## 2026-07-30 — P5 adversarial review and closure

- Review L1 adjustments:
  - changed Obsidian logical timeout to transient but not immediately retryable, preventing overlap with an uncancellable native request in the same operation;
  - made successful rendered HTML usable even when cache persistence fails;
  - replaced multi-write manual detail repair with atomic `PaperIndexStore.reconcileManualDetail` while keeping frontmatter refresh separately retryable;
  - made generated YAML scalar decoding the exact inverse of writer escaping;
  - made cache operation queues root/storage scoped with bounded recoverable leases and cancellation-aware waits;
  - tightened Retry-After parsing to exact IMF-fixdate grammar.
- A final independent re-review found one remaining medium parser edge case: JavaScript normalized impossible calendar dates and accepted weekday mismatches. Canonical `toUTCString()` round-trip validation and regression tests now reject both.
- Obsidian `requestUrl` cannot be physically cancelled. A timed-out native request may continue as an orphan; the current invocation does not immediately retry it, late settlement is consumed, and the logical queue recovers. Cross-invocation and cross-process physical coordination remain explicit non-goals.
- Verification after the final fix:
  - focused arXiv fetcher suite: 36 passed;
  - full tests: core 853, node runtime 11, CLI 27, plugin 248 passed;
  - all workspace typechecks passed;
  - lint passed with 0 errors and 53 pre-existing warnings;
  - workspace boundaries passed;
  - release-tool tests 5 passed and release metadata 0.3.5 passed;
  - workspace build and smoke build passed;
  - `git diff --check` passed.
- No unresolved blocking or medium correctness findings remain. All goal criteria and phases are done; the worktree remains uncommitted pending user direction.
