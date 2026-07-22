# P2 — user-visible regressions

goal_ref: ../goal.md
status: done

## Outcome

Writer, adapter, pipeline, and history paths have user-visible regression coverage proving scientific Markdown survives storage and rerun workflows without marker misclassification.

## Assumptions

- Core P1 rendering and parsing behavior is the intended contract for downstream writers and adapters.
- Existing uncommitted fallback feature changes provide the relevant pipeline and history seams without requiring production redesign.

## Approach

Exercise generated scientific Markdown through existing writer, storage-adapter, pipeline, and dashboard/history test seams, adding only regression coverage and minimal fixes required by observed downstream behavior.

## Tasks

- [x] Trace daily Markdown from summarizer output through writer and storage adapter boundaries.
- [x] Add writer/adapter regression coverage for raw MathJax and ordinary comparison punctuation.
- [x] Add pipeline regression coverage for multiline hostile prose and exact standalone machine markers.
- [x] Add history/dashboard regression coverage for preserved scientific Markdown and fallback/emergency identity.
- [x] Run affected core and plugin tests plus relevant typechecks; fix observed downstream regressions.
- [x] Checkpoint whether P3 end-to-end validation remains necessary or can be narrowed.

## Verification

- Focused writer, adapter, pipeline, and history/dashboard tests pass.
- Relevant core/plugin typechecks pass.
- Stored user-visible Markdown contains representative MathJax exactly as emitted by P1.

## Abort / reshape triggers

- If downstream code intentionally transforms Markdown through a documented format boundary, stop and reshape the expected round-trip contract rather than bypassing it.
- If coverage requires changing canonical IDs, `safeDetailLink`, or structured/fallback separation, stop and steer before implementation.
