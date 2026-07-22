# P4 — regression migration and verification

goal_ref: ../goal.md
status: done

## Outcome

Regression migration, dead-code cleanup, documentation consistency, and full repository verification complete the initiative without changing compatibility defaults or atomic sequential behavior.

## Assumptions

- Legacy multi-paper prompts and snapshots are removable only when repository-wide references prove they are outside current documentation and build contracts.
- Strict structured-validation failures remain transient through the pipeline, while provider 4xx and cancellation retain their existing permanent/cancelled classifications.
- Existing detail-summary tests can be compared with the branch base to identify valuable coverage lost during test simplification.

## Approach

Audit the complete feature diff and repository references, minimize public exports and dead assets, restore only meaningful regression coverage, run every repository verification command, then inspect the complete final diff before closing the goal.

## Tasks

- [x] Audit and remove unused legacy prompts, helpers, snapshots, imports, and exports.
- [x] Review pipeline error classifications and atomic write behavior with focused regression tests.
- [x] Verify compatibility defaults, output parsing, labels, links, ordering, counts, and consumers.
- [x] Compare historical detail-summary coverage and restore relevant behavioral tests.
- [x] Run lint, typecheck, tests, boundaries, build, and `git diff --check` from this worktree.
- [x] Inspect final status, diff stat, and complete diff for unrelated or generated artifacts.

## Verification

- Repository-wide reference searches and API-consumer checks show no stale contracts or unnecessary public surface.
- Full repository verification passes; test totals and expected injected-failure stderr are recorded.
- `plugin/main.js` remains ignored and uncommitted, and no unrelated worktree is touched.

## Abort / reshape triggers

- If cleanup would remove a documented or externally consumed compatibility contract, retain it and document the operational distinction instead.
- If any full-suite verification fails from the feature, keep P4 and the goal active until corrected and rerun.
