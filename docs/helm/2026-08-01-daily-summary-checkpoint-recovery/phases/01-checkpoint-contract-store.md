# P1 — checkpoint contract and store

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

Core has a tested, versioned, host-neutral store that can atomically preserve and retrieve compatible per-paper structured-summary checkpoint entries without changing pipeline behavior yet.

## Assumptions

- The existing `StorageAdapter` primitives are sufficient for a core-owned store used by both Node and Obsidian hosts.
- A date-scoped checkpoint document containing independently keyed per-paper entries allows valid work to survive selection changes without treating the whole run as one all-or-nothing cache key.
- Compatibility can be decided before reuse from canonical persisted inputs plus explicit prompt/result contract versions; secrets and raw provider responses are unnecessary.
- Validated results and typed fallbacks are all worth recording, but result kind is part of reuse policy: validated results and exact-match validation-exhausted fallbacks are reusable, while transport-exhausted fallbacks are retried on resume by default.
- Process-local mutation serialization is sufficient because cross-process and cross-device checkpoint coordination is outside this initiative.

## Approach

Introduce a dedicated core checkpoint store under the configured index path rather than expanding `RunStateEntry`. Persist one versioned document per report date with entries keyed by stable paper identity; each entry carries a deterministic compatibility fingerprint, the fingerprint input/version metadata needed to reject stale data, and the parsed `DailyPaperResult`. Reuse the repository's established atomic-write, backup-recovery, corruption-tolerance, and process-local mutation-queue patterns. Keep this phase below the summarizer so storage correctness and invalidation rules are settled before orchestration changes.

The contract will distinguish three concepts:

1. **Document schema version** — whether the stored JSON can be decoded safely.
2. **Prompt/result contract version** — whether an old result still obeys the current structured-summary semantics.
3. **Entry compatibility fingerprint** — whether the paper source content and all effective generation inputs for this run match the stored result.

A mismatch is a cache miss, never a best-effort reuse. A compatible validated result or validation-exhausted fallback is reusable; a transport-exhausted fallback remains inspectable but lookup returns a miss so a recovered run retries the provider. Unknown or corrupt entries are ignored while valid sibling entries remain recoverable where the document can be decoded safely. Mutation failure must preserve the previous valid checkpoint and surface an error to the caller.

## Settled P1 contract

- Path: `<indexDir>/daily-summary-checkpoints/YYYY-MM-DD.json`, with `<document>.bak` as the last valid recovery copy. The configured hidden index directory is derived by the existing `derivePaperInboxPaths` rule.
- Schema: document schema version `1`, exact top-level and entry keys, date binding, ISO timestamps, and independently decoded entries keyed by canonical `paperKey` (`arxiv:<id>`). The map key, entry `paperKey`, fingerprint paper identity, and bare result summary ID must agree. `DailyPaperResult` structured values reuse generation's trim/scientific-math validator; fallback attempts are integers from `1` through `DAILY_PAPER_SUMMARY_MAX_ATTEMPTS`. Persisted prompt/result contract versions must equal the current versions. Invalid entries are skipped with an observable warning while valid siblings survive; an invalid document falls back to a valid backup, then to an empty document.
- Compatibility: SHA-256 over a fixed-order version-1 effective-input object containing canonical paper identity, title/authors, trusted fallback abstract, prompt-trimmed abstract/conclusion and nonblank full sections, normalized summary language, provider, sanitized/effective endpoint, model, provider-specific generation mode, and prompt/result contract versions. Non-thinking mode records effective temperature only; thinking mode excludes ignored temperature and records either Anthropic's effective budget or the effective reasoning effort. API keys and redundant raw inputs are excluded; endpoint userinfo, query, and hash are removed before hashing or persistence.
- Reuse: `structured` and exact-fingerprint `validation-exhausted` results are reusable. `transport-exhausted` is retained for diagnostics but `lookupReusable` returns a miss.
- Mutation: same normalized document paths share one process-local queue across store instances. Stores prefer `StorageAdapter.writeTextAtomic`; otherwise they replace through `.tmp` and `.bak`, restoring the prior valid primary if replacement fails. A failed mutation rejects and does not allow a later queued mutation to inherit a poisoned queue.

## Tasks

- [x] Inventory the exact `DailyPaperResult` variants and effective summarization inputs, then define a canonical, versioned checkpoint schema and compatibility-fingerprint contract without credentials or unstable object ordering.
- [x] Define a date-scoped path under the configured index area and implement a core `DailySummaryCheckpointStore` over `StorageAdapter`, including load, compatible lookup, atomic upsert, and removal primitives needed by later phases.
- [x] Reuse or extract the existing atomic replacement and same-path process-local serialization pattern so failed writes retain the last valid main or backup document.
- [x] Validate decoded documents and entries defensively: reject incompatible versions/fingerprints, isolate invalid entries where safe, and recover from a valid backup when the main document is unreadable.
- [x] Add focused tests for round trips, deterministic fingerprints, every invalidation input, mixed valid/stale entries, corrupt/schema-invalid data, backup recovery, concurrent local mutations, and failed atomic replacement.
- [x] Document the settled path, schema, compatibility inputs, fallback retention, and corruption policy in this phase before activating P2.
- [x] Run focused core tests, core typecheck, and `git diff --check`.

## Verification

- Focused store tests demonstrate that an entry survives store reconstruction and is returned only when paper identity, source content, summary language, model/provider selection, generation parameters, prompt contract, and result schema contract match.
- Changing any compatibility input produces a miss for that entry without discarding compatible sibling entries.
- Both validated and typed-fallback `DailyPaperResult` variants round-trip through strict decoding; malformed or unknown variants are never returned.
- Concurrent process-local upserts preserve all entries, and an injected write/rename failure leaves the previous valid checkpoint readable.
- A corrupt primary document recovers from a valid backup; unrecoverable state degrades to no reusable entries with an observable warning/error policy rather than misusing data.
- No pipeline or summarizer behavior changes in P1.
- Focused tests, `npm --prefix packages/core run typecheck`, and `git diff --check` pass.

## Abort / reshape triggers

- If the effective model/provider identity or generation parameters cannot be obtained in core without leaking secrets or coupling to one host, stop and reshape the fingerprint boundary before implementing persistence.
- If independently reusable entries cannot preserve trusted paper identity and deterministic final ordering, reshape to a stricter run-level contract rather than accepting ambiguous reuse.
- If `StorageAdapter` cannot provide equivalent recoverable replacement semantics in both hosts, stop and define the smallest shared storage capability instead of introducing host-specific business behavior.
- If preserving typed fallbacks would make transient exhaustion permanent across materially different retry/provider conditions not captured by the fingerprint, revise fallback compatibility before P2.
