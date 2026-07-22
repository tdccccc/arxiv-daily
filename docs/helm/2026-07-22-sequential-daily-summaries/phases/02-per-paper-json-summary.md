# P2 — per-paper JSON summary

goal_ref: ../goal.md
status: done

## Outcome

One trusted paper can be summarized through a language-specific prompt and a strictly validated JSON response into the structured daily-summary contract.

## Assumptions

- The five structured fields from P1 capture the existing daily summary quality contract without model-authored metadata.
- A minimal structural paper input can remain compatible with `DailyPaperWithContent` without importing `summarizer.ts` at runtime.
- Source-section labels can remain deterministic and preserve the current fallback semantics.

## Approach

Add an isolated per-paper summarizer with Chinese and English prompt templates, a one-paper escaped data wrapper, direct strict JSON parsing, and a reusable trusted source-section helper. Keep the existing daily orchestration and prompts unchanged until P3.

## Tasks

- [x] Add language-specific one-paper JSON prompt templates with quality, grounding, fallback, and injection rules.
- [x] Implement the one-paper request with cancellation, metrics, signal, and deterministic temperature options.
- [x] Strictly validate exact response keys, trusted ID, and trimmed non-empty semantic fields.
- [x] Export a deterministic source-section helper preserving existing extraction and fallback semantics.
- [x] Add focused prompt, request, parser, option, cancellation-adjacent, and source-section tests.
- [x] Run focused tests, core typecheck, and `git diff --check`.

## Verification

- Run the new per-paper summary tests plus P1 assembler/parser tests.
- Run `npm --prefix packages/core run typecheck` and `git diff --check` from this worktree.
- Every invalid response shape rejects with a clear error; valid Chinese and English requests produce trimmed `StructuredPaperSummary` values.

## Abort / reshape triggers

- If the contract needs model-controlled title, author, category, link, or source-section data, stop and reshape rather than expanding trusted metadata into the response.
- If implementation requires changing `summarizeDaily` orchestration or evidence-budget defaults, defer that work to P3 instead of widening this phase.
