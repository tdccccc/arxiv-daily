# Release 0.4.1

status: done
updated: 2026-08-01
owner: sess_4ab50bf8-a3f3-40cd-bf05-66eb503e2ed9

## Intent

Fix forward from the immutable failed 0.4.0 tag by making CI use the repository's proven memory-safe workspace test invocation, then publish the same product release as 0.4.1 without creating a partial or overwritten release.

## Success criteria

- [x] The release workflow runs all workspace tests serially with the required heap allowance, protected by a release-tool regression test.
- [x] All product and release metadata and curated notes consistently identify 0.4.1 and explain the 0.4.0 tag-only failure.
- [x] The exact workflow command and complete release gate pass locally.
- [x] The verified fix-forward commit reaches local and remote main without disturbing existing worktree edits.
- [x] No 0.4.1 tag is pushed without explicit authorization after the new release checkpoint.

## Non-goals

- Moving, deleting, or reusing the immutable 0.4.0 tag.
- Creating a GitHub Release or npm package for 0.4.0.
- Changing product behavior beyond the already verified 0.4.0 candidate.
- Decoupling CLI and plugin versioning in this recovery release.

## Constraints

- Preserve evidence from failed Actions run 30693899943: Core OOM occurred before build, assets, GitHub Release, or npm publication.
- Use 0.4.1 as the next stable version and the same bare SemVer tag convention.
- Match local verification exactly to the fixed workflow test command.
- Keep the main worktree's seven pre-existing uncommitted documentation edits intact.
- Do not push a new tag without a separate explicit authorization.

## Phases

1. P1 — CI-safe tests, 0.4.1 metadata, and curated fix-forward notes form a new candidate — status: done
2. P2 — the exact CI-equivalent complete gate verifies the 0.4.1 candidate — status: done
3. P3 — verified main is pushed and stops before the separately authorized 0.4.1 tag — status: done
