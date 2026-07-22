# Daily report rerun consistency

status: done
updated: 2026-07-22

## Intent

Keep generated daily-report metadata consistent with the rendered papers and make a deleted recent report straightforward to regenerate from the dashboard.

## Success criteria

- [x] The daily count line reflects unique paper blocks actually rendered by the model.
- [x] A recent completed date becomes runnable when its daily file is missing.
- [x] Dashboard Run/Force date actions reliably open, report progress, and refresh after completion.
- [x] Core and plugin regression tests pass.

## Non-goals

- Changing arXiv API retry policy for 429/503 responses.
- Overwriting a daily report that still exists.

## Constraints

- Preserve scheduler durability and no-overwrite semantics for existing Markdown files.
- Preserve terminal no-update handling for zero-paper, skipped, and permanent-failure dates.

## Phases

1. P1 — Generated report counts and duplicate diagnostics match rendered paper blocks — status: done
2. P2 — Missing recent reports are runnable from the calendar — status: done
3. P3 — Dashboard date-run controls reliably open and refresh — status: done
4. P4 — Regression verification passes and the initiative is closed — status: done
