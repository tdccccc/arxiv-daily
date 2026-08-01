# Release 0.4.0

status: done
updated: 2026-08-01
owner: sess_4ab50bf8-a3f3-40cd-bf05-66eb503e2ed9

## Intent

Prepare one verified 0.4.0 release commit for the CLI and Obsidian plugin, shipping their shared recovery and resilience improvements through the repository's existing unified release workflow.

## Success criteria

- [x] All CLI, plugin, shared workspace, manifest, version-map, and lockfile metadata consistently identify version 0.4.0.
- [x] Curated 0.4.0 notes accurately describe the user-visible changes, upgrade path, compatibility, and checkpoint data handling.
- [x] The complete documented release gate passes from a clean release worktree.
- [x] The verified release commit is fast-forwarded into local `main` without disturbing its pre-existing uncommitted documentation edits.
- [x] The exact commit, tag, assets, npm prerequisite, and verification evidence are reported before any external publication.

## Non-goals

- Decoupling CLI and plugin product versioning, tags, release notes, or workflows in this release.
- Backfilling missing 0.3.4 or 0.3.5 GitHub releases.
- Pushing commits or tags, creating a GitHub Release, or publishing npm without a separate explicit authorization.

## Constraints

- Use version 0.4.0 for both products because this release changes both CLI and plugin behavior.
- Work from a clean release worktree based on commit `34545cb`; do not touch the main worktree's existing uncommitted files.
- Preserve the existing stable tag format without a `v` prefix and the immutable-release policy.
- Confirm the npm publication prerequisite before triggering the unified tag workflow.
- Commit each completed Helm phase separately; do not push without explicit instruction.

## Phases

1. P1 — curated notes and synchronized metadata form a reviewable 0.4.0 release candidate — status: done
2. P2 — the complete release gate and product artifact checks establish a verified release commit — status: done
3. P3 — the verified commit reaches local main and stops at an explicit publication checkpoint — status: done
