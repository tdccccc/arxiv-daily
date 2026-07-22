# P3 — dashboard date controls

goal_ref: ../goal.md
status: done

## Outcome

Dashboard Run/Force date controls open reliably after menu dismissal and refresh report state after scheduler completion or failure.

## Assumptions

- Deferring modal opening by one event loop avoids Obsidian menu teardown races.
- Dashboard-local scheduler calls are preferable to command indirection for awaiting completion.

## Approach

Share the date-picker modal, defer menu actions, invoke the selected scheduler path locally, and refresh in a finally block.

## Tasks

- [x] Extract a reusable date-picker modal.
- [x] Defer date actions until after menu teardown.
- [x] Await normal/force scheduler calls and refresh afterward.
- [x] Add focused regression tests.

## Verification

- Plugin typecheck passes.
- Commands, dashboard view, and calendar state tests pass.

## Abort / reshape triggers

- If Obsidian still dismisses the deferred modal, replace the menu entries with a dashboard-owned modal launcher outside Menu lifecycle.
- If scheduler errors prevent refresh, retain the finally-based refresh and surface the original error notice.
