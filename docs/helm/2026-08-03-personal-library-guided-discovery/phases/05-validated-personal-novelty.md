# P5 — validated abstract-level personal novelty

goal_ref: ../goal.md
updated: 2026-08-04

## Outcome

Every library-derived daily entry can carry validated abstract-level personal novelty: a bounded difference type, a named representative prior-paper comparison basis, an explicit metadata-and-abstract evidence depth, and a bounded explanation, deterministically rendered in the daily report, persisted per committed occurrence with strict machine-readable markers, recovered after interruption, and shown in Dashboard, without changing manual-only entries, summary/detail/email generation, or the existing authority and consent boundaries.

## Assumptions

- Novelty applies only to entries whose discovery provenance contains at least one confirmed direction. Manual-only papers and manual topics never receive novelty evidence.
- One novelty statement per paper compares it to the complete union of representative prior papers across its matched directions. If that complete basis cannot fit the bounded call contract, the paper deterministically receives no novelty; comparison evidence is never silently truncated.
- Novelty is best-effort additive evidence, not a gate: transport, output-limit, validation-exhaustion, checkpoint, or plan-too-large conditions degrade to "no novelty for this paper" and never fail, block, or rewrite the reliable daily run.
- A valid novelty result is strict: exact bounded difference-type enum, a non-empty unique subset of the supplied representative `paperKey`s as comparison basis, exact `metadata-and-abstract` evidence depth, and a bounded explanation that only describes abstract-level difference without unsupported full-text implications such as challenges, supersedes, or proof claims.
- Sending representative metadata-and-abstract evidence to the configured endpoint is model processing and requires the same current endpoint-bound authorization that gates personalized discovery. Revocation or any eligibility/root/output/endpoint transition must make further novelty calls impossible.
- Novelty never enters daily summary prompts or the summary checkpoint fingerprint. Detailed-report generation, digest/email projection, scheduler semantics, and P4 union/provenance behavior remain unchanged; the goal-level email open question stays open for P6.

## Approach

Add strict host-neutral novelty DTOs and a bounded generator in Core that validates every structured model result against the supplied canonical representative evidence, retries logically exactly three times, and never promotes partial or invented references. Persist validated results in a strict date-scoped novelty checkpoint with exact fingerprinting, and run novelty as a deterministic best-effort pipeline stage after filtering for library-derived papers only. Render visible and machine-readable novelty at the daily commit boundary, persist it per committed occurrence in the Paper Index, and rebuild it from committed Markdown during existing-daily repair and Dashboard history sync. The plugin derives the novelty input from the same currently authorized eligibility snapshot that gates personalized discovery, so the existing revision/identity guards and combined abort signal cover novelty without new transition surface.

## Tasks

- [x] Define and adversarially test strict bounded Core novelty input and result contracts, deterministic complete comparison basis selection, whole-run and per-call bounds, exactly three logical validation attempts, prompt-injection containment, cancellation, metrics, and privacy payload minimization while leaving manual-only and summary behavior unchanged.
- [x] Extend the daily filter checkpoint store with a strict date-scoped novelty document (exact branded prepared snapshot, fingerprint recomputation, backup recovery, atomic private writes, per-path queues, committed-report cleanup) and wire a deterministic best-effort pipeline novelty stage after filtering that degrades to no-novelty on every failure class without changing `PipelineResult` semantics.
- [x] Carry validated novelty through filtered papers and deterministic structured/fallback daily rendering with a localized plain-text line, a strict versioned machine-readable novelty marker bound to its paper block, per-report novelty persistence in the Paper Index, and non-destructive existing-daily repair and Dashboard history recovery.
- [x] Project the latest applicable committed novelty as separate Dashboard metadata, text-only and accessible, distinct from discovery provenance and query-time match reasons, without new filters, ranking, search, or fabricated novelty claims.
- [ ] Extend the plugin's immutable daily discovery snapshot to derive novelty input from the same currently authorized eligibility plus catalog representative evidence, with the existing revision/identity guards and combined abort signal, without changing manual single-paper fetch or proposal-only review.
- [ ] Prove manual-only compatibility, authorized-eligibility-only novelty, complete checkpoint/recovery behavior, privacy and authorization boundaries, and no summary/detail/email regression; run full repository quality suites and independent correctness/security reviews.
- [ ] Complete every accepted implementation chunk's independent technical-report handoff, mark P5 done while keeping P6 pending and the goal active, and create staged local commits without pushing.

## Verification

- Core novelty tests prove enum/depth/basis/explanation strictness, unknown or invented basis rejection, duplicate and malformed output rejection, exactly three logical attempts, deterministic complete basis or plan-too-large, per-call and whole-run bounds, cancellation before/after every call, injection containment for every free-text field, and no partial promotion.
- Checkpoint tests prove exact reuse only under identical rendered calls, basis evidence, direction identity/text, generation identity, and contract versions; unrelated catalog entries and file-path-only changes do not invalidate; corrupt/unreadable primary and backup, atomic promotion, per-path serialization, and committed-report cleanup behave like the existing filter checkpoint.
- Pipeline tests prove novelty attaches only to library-derived papers, every failure class degrades to no-novelty without changing `PipelineResult` kinds, summary prompts/checkpoints, detail reports, digest/email, and manual-only behavior remain unchanged, and cancellation is honored between calls.
- Rendering/parser/index tests prove Chinese and English novelty is identical for structured and fallback entries, hostile metadata cannot forge or reorder markers, invalid novelty never damages discovery provenance or other index state, legacy Markdown and reports stay valid, and committed occurrence novelty survives restart, repair, repeated reports, and history rebuild.
- Dashboard tests prove novelty is visible without search, text-only and fully accessible, distinct from provenance and `matchReasons`, with legacy rows unchanged.
- Plugin lifecycle tests prove novelty calls occur only under current authorization and eligibility, revocation and every transition abort captured runs with zero further model calls, all daily entry paths converge on one immutable snapshot, and manual fetch and proposal-only review are unaffected.
- Instrumented regressions prove no roots, paths, PDF bytes, fingerprints, authorization state, credentials, or unrelated catalog records are sent or persisted by the novelty path.
- Complete Core and plugin suites with the established Core heap allowance; node-runtime and CLI regressions; workspace typecheck/build; plugin production build; ESLint; boundary check; `git diff --check`; and independent correctness/security reviews pass.

## Abort / reshape triggers

- If representative metadata-and-abstract evidence cannot fit bounded novelty calls without silently dropping comparison papers, stop and L2 reshape the comparison-basis contract instead of truncating evidence.
- If structured novelty output cannot be strictly validated only against the supplied daily paper and representative evidence, omit novelty and narrow the contract rather than accepting repaired, inferred, or model-authored comparison identity.
- If novelty failures begin to fail or block daily runs, change summary/detail/email behavior, or weaken the metadata-versus-full-text boundary, move that behavior out of P5 instead of widening this phase.
- If per-paper single-novelty proves semantically misleading across heterogeneous matched directions, stop and L2 reshape to per-direction novelty with the same bounds and authority boundaries before shipping.
- If committed Markdown cannot be the sole recovery authority for novelty without unbounded or prose-inferred markers, stop and L2 define a smaller versioned marker grammar.
