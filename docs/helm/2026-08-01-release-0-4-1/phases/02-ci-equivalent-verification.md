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

- [x] Run clean install, release tools, version, boundaries, lint, and typecheck.
- [x] Run the exact fixed workspace test command and confirm all test counts.
- [x] Build and smoke-test production artifacts.
- [x] Verify CLI version, npm package contents, plugin assets, audit, and clean tracked scope.
- [x] Commit evidence and activate the main/publish checkpoint.

## Recorded evidence

- The complete corrected workflow command chain exited successfully from clean `npm ci` through `smoke:build`.
- Release tools 5, metadata, boundaries, lint (0 errors/52 warnings), and all workspace typechecks passed.
- Exact CI test invocation passed 1,372 tests across 94 files: Core 1,012; Node Runtime 13; CLI 59; Plugin 288.
- Build and production smoke passed.
- Both CLI bundles report 0.4.1 and share SHA-256 `220c77a3ed396c44876ec56d670599542434029b0072eb487e62fbad1015450f`.
- Plugin `main.js` remains SHA-256 `76a78057bfca88d3300e6fcc255b4be173928a7996afc92541bf1523aea84246`.
- Production dependency audit reports 0 vulnerabilities; generated artifacts remain ignored; `git diff --check` passes.

## Verification

- Commands exactly mirror `.github/workflows/release.yml` through `smoke:build`.
- `npm audit --omit=dev` reports no production vulnerabilities.
- `git diff --check` passes and generated products remain ignored.

## Abort / reshape triggers

- If the exact test command still exhausts memory, do not increase limits blindly; profile or split the Core suite.
- If any count or product artifact differs from the verified 0.4.0 candidate unexpectedly, investigate before publication.
