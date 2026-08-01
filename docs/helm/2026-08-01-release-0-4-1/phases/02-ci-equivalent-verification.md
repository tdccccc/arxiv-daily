# P2 — CI-equivalent verification

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

Every command in the corrected release workflow passes locally against the exact 0.4.1 candidate, including the formerly failing workspace test command.

## Assumptions

- Local Node 22 is sufficiently aligned with workflow Node 22.17.0 for this release gate.
- Running the commands in workflow order exposes any state or generated-artifact dependency.
- The fixed test invocation preserves all 1,372 tests.

## Approach

Run clean install and every workflow verification command in order, inspect generated artifacts and package contents, then record exact evidence before advancing main.

## Tasks

- [ ] Run clean install, release tools, version, boundaries, lint, and typecheck.
- [ ] Run the exact fixed workspace test command and confirm all test counts.
- [ ] Build and smoke-test production artifacts.
- [ ] Verify CLI version, npm package contents, plugin assets, audit, and clean tracked scope.
- [ ] Commit evidence and activate the main/publish checkpoint.

## Verification

- Commands exactly mirror `.github/workflows/release.yml` through `smoke:build`.
- `npm audit --omit=dev` reports no production vulnerabilities.
- `git diff --check` passes and generated products remain ignored.

## Abort / reshape triggers

- If the exact test command still exhausts memory, do not increase limits blindly; profile or split the Core suite.
- If any count or product artifact differs from the verified 0.4.0 candidate unexpectedly, investigate before publication.
