# P4 — emergency report and repository verification

goal_ref: ../goal.md
status: done

## Outcome

Rescue postflight exhaustion or typed transient rescue failure produces a minimal deterministic complete emergency report, while pipeline commit, index/search/repair behavior and the full repository remain verified.

## Assumptions

- P3 typed rescue exhaustion and `LlmTransientExhaustedError` are the only failures eligible for emergency assembly.
- The compact preflighted paired slots are sufficient to render every paper safely without another provider call.
- Existing daily write, paper-index, search, and repair boundaries can consume the emergency Markdown without treating fallback abstracts as structured summaries.

## Approach

Add a narrowly gated minimal deterministic renderer after P3 rescue failure, reuse trusted paired slots and stable fallback markers, then verify durable daily commit and downstream index/search/repair semantics across the repository.

## Tasks

- [x] Define and share a rendering-safety contract across normal, rescue, and emergency Markdown, including canonical detail links, exact scalar round-trips, assembly preflight, and parser-projection checks.
- [x] Harden rescue serialization delimiters while retaining the approved fallback original abstract and exact locally trusted postflight rendering.
- [x] Route production and injected renderer failures through the same typed post-preflight runtime boundary, without rescuing preflight or unrelated failures.
- [x] Restore marker-confirmed fallback abstracts during dashboard history rebuild without indexing localized unavailable placeholders, and verify BM25 and legacy substring discovery.
- [x] Add stable typed-error discriminators and safe cross-realm guards with focused gating tests.
- [x] Run focused core tests, the full core suite and typecheck, then full repository lint/tests/typechecks/boundaries/build checks.
- [x] Inspect final diff/status, run `git diff --check`, and close the initiative only when all success criteria are satisfied.

## Verification

- Rescue validation exhaustion and typed transient exhaustion each yield a complete deterministic report with every selected paper exactly once.
- Cancellation, permanent provider failures, preflight failures, and unrelated programming errors never enter emergency assembly.
- Pipeline daily write, paper index, search, existing-file rerun, and repair tests pass for mixed emergency content.
- Core and full repository verification commands pass.
- `git diff --check`

## Abort / reshape triggers

- If an emergency report cannot remain parser-compatible without weakening structured/fallback separation, stop and strengthen stable markers/parser contracts first.
- If durable write/index behavior requires changing existing-file commit semantics, stop and reshape the phase before implementation.
