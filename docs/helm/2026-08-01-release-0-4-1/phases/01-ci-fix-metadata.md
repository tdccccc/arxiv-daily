# P1 — CI fix and metadata

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

A new 0.4.1 release candidate fixes and regression-tests the CI memory constraint while preserving the product contents and documenting the failed 0.4.0 tag accurately.

## Assumptions

- The CI failure is exclusively Core Vitest default-concurrency heap exhaustion, as shown by the failed job log.
- The locally proven 8 GiB heap and single-worker invocation is suitable for GitHub's runner and does not weaken test coverage.
- No 0.4.0 GitHub Release or npm package exists, so users can move directly from 0.3.5 to 0.4.1.

## Approach

Make the workflow invoke the exact safe test command, lock that command with the existing workflow regression test, synchronize all metadata to 0.4.1, and adapt the curated release notes without changing product claims.

## Tasks

- [x] Replace workflow `npm test` with the exact memory-safe serial invocation.
- [x] Add a release-tool regression assertion for the exact invocation.
- [x] Synchronize all release metadata to 0.4.1.
- [x] Curate 0.4.1 notes and record that 0.4.0 failed before publication.
- [x] Run release-tool, version, focused test-command, and diff checks.
- [x] Commit P1 and activate complete verification.

## Recorded evidence

- Failed run 30693899943 exhausted the Core worker heap at approximately 4 GiB; all later workspaces passed, and asset/release/npm steps never ran.
- Workflow now runs `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1`.
- Release-tool tests assert the exact safe command and all 5 tests pass.
- 0.4.1 metadata consistency and `git diff --check` pass.
- No local/remote tag, npm version, or GitHub Release occupies 0.4.1.

## Verification

- `npm run test:release-tools`
- `npm run check:release-version -- 0.4.1`
- `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1`
- `git diff --check`

## Abort / reshape triggers

- If CI logs reveal any cause beyond Core heap exhaustion, stop and address the complete failure before retagging.
- If 0.4.1 is already occupied in any target, select the next unused version.
- If serialization hides or skips tests, reject the approach rather than weaken verification.
