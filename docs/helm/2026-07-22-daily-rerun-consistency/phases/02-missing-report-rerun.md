# P2 — missing report rerun

goal_ref: ../goal.md
status: done

## Outcome

A completed recent date whose daily Markdown file is missing becomes a green runnable calendar cell when it otherwise satisfies calendar eligibility.

## Assumptions

- A non-zero completed state without a report represents a deleted or externally moved report.
- Existing reports continue to take precedence and remain protected from overwrite.

## Approach

Allow non-zero completed state through the calendar whitelist when no report exists, then resolve it as runnable only when all date eligibility checks pass; otherwise retain the report-missing fallback.

## Tasks

- [x] Refine terminal-state blocking for calendar reruns.
- [x] Prefer runnable over report-missing for eligible dates.
- [x] Update calendar regression tests.

## Verification

- Run plugin calendar-state tests.
- A missing completed report is runnable inside `/recent` but report-missing outside the eligible window.

## Abort / reshape triggers

- If scheduler manual runs still reject completed state, reshape to clear the date state before calendar execution.
- If deleted reports cannot be distinguished from storage failures, retain the report-missing label outside explicit eligibility only.
