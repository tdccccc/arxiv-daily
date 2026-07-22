# P1 — rendered paper count

goal_ref: ../goal.md
status: done

## Outcome

Generated daily metadata and diagnostics are derived from unique rendered paper blocks rather than raw ID occurrences or pre-generation input size.

## Assumptions

- A valid paper entry is represented by a level-three heading block containing an arXiv link or ID.
- If the model omits a paper, reporting the smaller rendered count is preferable to claiming unseen content exists.

## Approach

Parse one arXiv ID per rendered paper block after normalization, use that inventory for the canonical count line and drift diagnostics, and deduplicate filter-model classifications upstream.

## Tasks

- [x] Add rendered-block inventory and canonical count normalization.
- [x] Correct missing and duplicate diagnostics.
- [x] Deduplicate filter output by arXiv ID.
- [x] Add focused regression tests.

## Verification

- Run core summarizer and paper-filter tests.
- Missing model output lowers the count while a normal Markdown link does not trigger a duplicate warning.

## Abort / reshape triggers

- If valid generated reports lack stable level-three paper blocks, stop and reuse the broader daily-history parser instead.
- If downstream consumers require selected rather than rendered counts, separate the two metrics instead of silently changing both.
