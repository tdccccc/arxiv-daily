# P2 — release verification

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

The complete repository release gate and explicit product artifact checks pass against the exact 0.4.0 release candidate.

## Assumptions

- The repository's documented release commands cover the same source and build paths used by the tag workflow.
- Serial tests with an 8 GiB heap are an acceptable operational adjustment for the known local Core test memory requirement.
- Generated bundles are release evidence but remain ignored build products rather than committed files.

## Approach

Install from the canonical lockfile, run every documented release gate, inspect the generated CLI and plugin artifacts, and record exact evidence. Fix only release-blocking correctness issues and rerun the affected plus complete gate before accepting the phase.

## Tasks

- [x] Install dependencies from the canonical root lockfile.
- [x] Run release-tool tests, version check, boundaries, lint, and all workspace typechecks.
- [x] Run the complete workspace test suite serially with the required heap allowance.
- [x] Build all products and pass the production smoke build.
- [x] Verify embedded CLI versions and the exact three plugin release assets.
- [x] Confirm release notes, branch scope, clean diff, and generated-artifact policy.
- [x] Record evidence, update Helm status, and commit P2.

## Recorded evidence

- Clean `npm ci` completed; `npm audit --omit=dev` reported 0 production vulnerabilities. The install's two high-severity advisories are confined to the development dependency tree.
- Release-tool tests passed: 5 tests.
- Version consistency, workspace boundaries, and all workspace typechecks passed.
- Lint passed with 0 errors and 52 warnings, below the configured maximum of 60.
- Full serial workspace tests passed: Core 1,012; Node Runtime 13; CLI 59; Plugin 288 — 1,372 tests across 94 files.
- Workspace build and production smoke build passed.
- Canonical and plugin-copied CLI bundles both report `Version: 0.4.0` and have identical SHA-256 `64feb51a9e0f086df948f6d8b1a8dfaab7bd5c622e803524cea97a09eaa94fc9`.
- Plugin release assets exist exactly as `manifest.json`, `main.js`, and `styles.css`; `main.js` SHA-256 is `76a78057bfca88d3300e6fcc255b4be173928a7996afc92541bf1523aea84246`.
- npm dry-run produced `arxiv-daily@0.4.0` with only package metadata, LICENSE, README, and the canonical CLI bundle.
- Generated bundles and installed dependencies remain ignored; `git diff --check` passed and tracked branch scope contains only release metadata, notes, and Helm records.

## Verification

- `npm ci`
- `npm run test:release-tools`
- `npm run check:release-version -- 0.4.0`
- `npm run check:boundaries`
- `npm run lint`
- `npm run typecheck`
- `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1`
- `npm run build`
- `npm run smoke:build`
- `git diff --check`

## Abort / reshape triggers

- If any release gate fails because the candidate is incorrect, stop release progression and fix forward on the release branch.
- If generated CLI/plugin versions differ from metadata, do not tag until the build/version source is reconciled.
- If tests require semantic weakening or skipped coverage, stop and reshape rather than declaring the candidate verified.
