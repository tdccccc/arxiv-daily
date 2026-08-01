# P1 — filter contract and durable store

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

Core can construct, persist, decode, and look up a validated report-date paper-filter batch under an exact request and generation fingerprint, independently of the per-paper summary document.

## Assumptions

- The current filter request is one batch whose paper and topic order affects the rendered prompt.
- Existing filter response semantics permit omitted paper IDs and preserve response-record order.
- Filter and summary stores may share low-level durability mechanics but must retain independent schemas and corruption boundaries.
- A strictly valid empty filter record list is a reusable result; malformed responses are not.

## Approach

Extract one request builder and one strict filter-record decoder from the live filter path so checkpoint compatibility cannot drift from generation. Define versioned filter fingerprint and document contracts around the exact rendered messages plus effective LLM identity. Persist one batch document per report date under `<indexDir>/filter-checkpoints`, using the same host-neutral strict read, backup recovery, fail-closed mutation, temporary-file replacement, and process-local serialization guarantees established for structured-summary checkpoints.

## Tasks

- [x] Add filter-checkpoint terminology to the project glossary.
- [x] Extract and test the exact paper-filter request builder and reusable strict record decoder without changing live filter behavior.
- [x] Define versioned filter compatibility, fingerprint, result, document, path, and error contracts.
- [x] Implement exact-compatible lookup, strict save, remove-all, corruption isolation, backup recovery, and fail-closed mutation.
- [x] Reuse effective endpoint/generation identity and exclude API keys, plaintext endpoints, host paths, and raw responses.
- [x] Add focused fingerprint, strict-decoder, durability, concurrency, and fault-injection tests.
- [x] Run focused Core typecheck/tests and `git diff --check`, then reconcile and commit P1.

## Verification

- Exact messages, provider/model/effective mode, endpoint digest, and contract versions determine compatibility.
- Changes that do not affect the live request, including API key and currently unused paper metadata, do not invalidate reuse.
- Persisted decisions are strictly tied to known IDs and valid topic tags and retain record order.
- A corrupt or unreadable document cannot silently become a reusable result or be overwritten from stale state.
- Filter and summary checkpoint files remain independently recoverable.

## Recorded evidence

- Core typecheck passed.
- Paper-filter 23, filter-checkpoint 49, and summary-checkpoint regression 74 tests passed — 146 focused tests total.
- `git diff --check` passed.
- Independent final correctness/security review reported no findings after structured request identity and live temperature identity fixes.

## Abort / reshape triggers

- If exact rendered messages cannot be shared with the live LLM path, stop rather than maintain duplicate prompt identity logic.
- If extracting common durability code destabilizes the proven summary store, keep independent stores temporarily and record the duplication for later refactoring.
- If current filter parser semantics are ambiguous, preserve existing behavior and version it rather than tightening production classification in this phase.
