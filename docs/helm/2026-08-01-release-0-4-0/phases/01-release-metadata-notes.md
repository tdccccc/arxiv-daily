# P1 — release metadata and notes

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

A clean release branch contains accurate curated 0.4.0 notes and internally consistent CLI, plugin, workspace, manifest, version-map, and lockfile metadata.

## Assumptions

- The checkpoint recovery, portability, resilience, and cancellation work since 0.3.5 is additive and warrants a pre-1.0 minor release.
- CLI and plugin both receive material behavior changes in this release, so sharing 0.4.0 is semantically accurate.
- The existing release synchronization and checking tools remain the authoritative metadata path.

## Approach

Describe the release in user-facing terms, synchronize all version-bearing files with the repository tool, then inspect and validate the resulting release inputs before committing them as one preparation phase.

## Tasks

- [x] Create the release Helm initiative in the clean release worktree.
- [x] Curate `docs/releases/0.4.0.md` with highlights, upgrade steps, compatibility, and data-handling guidance.
- [x] Synchronize all release metadata with `sync:release-version`.
- [x] Check version consistency and inspect every metadata/release-note change.
- [x] Confirm manifest minimum-app mappings, CLI package surface, and expected release assets remain intact.
- [x] Run `git diff --check` and commit P1.

## Recorded evidence

- `npm run check:release-version -- 0.4.0` passed.
- `git diff --check` passed.
- Local and remote tag, npm, and GitHub Release checks found no existing `0.4.0`.
- Both manifests identify `0.4.0` with unchanged `minAppVersion: 1.4.0`; both version maps contain the same mapping.
- CLI `bin` remains `./dist/arxiv-daily-cli.cjs` and its published file allowlist remains unchanged.
- GitHub Actions exposes the `NPM_TOKEN` secret name required by the unified release workflow.

## Verification

- `npm run check:release-version -- 0.4.0`
- `git diff --check`
- The diff contains only expected release preparation and Helm files.
- Both manifests and version maps agree on 0.4.0 and the existing minimum Obsidian version.

## Abort / reshape triggers

- If CLI and plugin no longer share material release content, stop and reshape around independent versions rather than forcing lockstep.
- If synchronization changes product packaging or minimum compatibility unexpectedly, stop and investigate before committing.
- If any existing 0.4.0 tag, npm version, or GitHub Release is found, choose a new version rather than reuse it.
