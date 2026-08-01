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

- [ ] Install dependencies from the canonical root lockfile.
- [ ] Run release-tool tests, version check, boundaries, lint, and all workspace typechecks.
- [ ] Run the complete workspace test suite serially with the required heap allowance.
- [ ] Build all products and pass the production smoke build.
- [ ] Verify embedded CLI versions and the exact three plugin release assets.
- [ ] Confirm release notes, branch scope, clean diff, and generated-artifact policy.
- [ ] Record evidence, update Helm status, and commit P2.

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
