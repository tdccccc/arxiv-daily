# P3 — rescue assembler

goal_ref: ../goal.md
status: done

## Outcome

An unexpected runtime failure in the normal deterministic assembler invokes a bounded Rescue LLM that can produce a strictly validated complete daily report from compact paired slots.

## Assumptions

- Normal assembly input preflight remains outside the rescue boundary, so invalid trusted inputs fail before generation rather than being rescued.
- Compact paired slots contain enough trusted metadata and structured/fallback content to reconstruct the report without source full text.

## Approach

Catch only unexpected runtime failures from the post-preflight deterministic assembly step, send compact paired slots to a dedicated Rescue LLM prompt, and validate the returned report with strict postflight checks. Retry only postflight validation failures, with at most three total rescue logical calls.

## Tasks

- [x] Define a compact rescue-slot projection containing trusted display metadata plus each typed structured/fallback result, excluding abstracts/conclusions, full sections, and other source full text.
- [x] Isolate the deterministic assembler runtime boundary after preflight so invalid inputs, cancellation, and permanent provider errors cannot enter rescue.
- [x] Add a dedicated Rescue LLM prompt and validation-only retry path with maximum total three logical calls and corrective guidance after invalid responses.
- [x] Implement strict postflight validation for complete paper coverage, exact IDs/order/category placement, trusted metadata, fallback markers/counts, and generated-summary extraction semantics.
- [x] Forward cancellation, metrics, and permanent errors unchanged; do not add application transport retries or rescue typed transient transport exhaustion.
- [x] Add focused tests for runtime-trigger gating, compact/no-full-text payloads, first/third valid rescue, three invalid responses, strict postflight rejection, exclusions, and no-rescue paths.
- [x] Run focused tests, core typecheck, and `git diff --check`.

## Verification

- A deterministic assembler runtime exception after valid preflight triggers one Rescue LLM logical call and returns a strict complete report when valid.
- Invalid rescue responses receive at most two corrective retries, for three total logical calls.
- Rescue prompts contain compact paired slots and no abstract/conclusion/full-section source text.
- Preflight failures, cancellation, permanent 4xx, and typed transient transport exhaustion do not invoke or continue rescue.
- `npm run typecheck --workspace @arxiv-daily/core`
- `git diff --check`

## Abort / reshape triggers

- If strict postflight cannot prove exact trusted paper coverage and fallback identity without parsing localized prose, stop and strengthen machine-readable rescue markers/contracts first.
- If the runtime boundary cannot distinguish deterministic assembler failures from preflight/input failures, stop and isolate the assembler API before adding rescue.
