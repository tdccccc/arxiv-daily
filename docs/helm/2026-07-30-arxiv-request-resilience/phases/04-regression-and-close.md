# P4 — regression and close

goal_ref: ../goal.md
status: done

## Outcome

Repository-wide verification and final review demonstrate that the request, note-safety, UI, cache, and host wiring changes are ready, with all Helm success criteria supported by evidence.

## Assumptions

- Focused phase verification has already covered the riskiest behavioral paths.
- Any unrelated baseline failure can be separated from regressions with reproducible output.

## Approach

Review the cumulative diff for integration mistakes, run repository checks from the isolated worktree, fix only genuine regressions, then update the Helm artifacts with exact evidence and close the initiative.

## Tasks

- [x] Review the cumulative diff for correctness, compatibility, and accidental scope growth.
- [x] Run release-tool checks, boundaries, lint, workspace typechecks, and all tests.
- [x] Run workspace builds and smoke build.
- [x] Run final diff/status hygiene checks.
- [x] Record verification evidence and close all success criteria.

## Verification

- `npm run test:release-tools`
- `npm run check:release-version -- 0.3.5`
- `npm run check:boundaries`
- `npm run lint`
- `npm run typecheck`
- `npm test`
- `npm run build`
- `npm run smoke:build`
- `git diff --check`
- `git status --short --branch`

## Abort / reshape triggers

- If a full-suite failure exposes a design flaw in a completed phase, reopen or supersede that phase rather than documenting it away.
- If smoke build fails only because of a proven unrelated stale baseline assertion, record the exact evidence and do not weaken feature tests.
