# P3 — local publication checkpoint

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

The exact verified 0.4.0 commit is present on local `main`, the main worktree's prior edits are preserved, and all external publication inputs are reported before push or tagging.

## Assumptions

- Local `main` still points to the release branch's original base and can be fast-forwarded.
- The seven pre-existing main-worktree documentation edits do not overlap release metadata or Helm release files.
- Publishing remains an explicit separate authorization because pushes and immutable tags are outward-facing.

## Approach

Commit the P2 evidence, verify ancestry and file overlap, fast-forward local `main` without checking out or rewriting its existing files beyond Git's safe merge behavior, then perform read-only publication preflight and stop before push/tag.

## Tasks

- [ ] Commit the complete P2 verification evidence.
- [ ] Confirm release branch cleanliness, exact tip, and fast-forward ancestry from local `main`.
- [ ] Reconfirm the main worktree's pre-existing edits and absence of overlap with the release commit.
- [ ] Fast-forward local `main` to the verified release commit.
- [ ] Confirm main's existing uncommitted edits remain byte-for-byte intact.
- [ ] Report the commit, tag, assets, workflow prerequisites, and publication commands without executing them.

## Verification

- `git merge-base --is-ancestor main release/0.4.0` before merge.
- `git merge --ff-only release/0.4.0` in the main worktree.
- Main status retains exactly its seven prior modified documentation paths.
- Local `main` and `release/0.4.0` resolve to the same verified commit.
- No `0.4.0` local/remote tag, npm version, or GitHub Release exists before publication.

## Abort / reshape triggers

- If main has diverged, do not create an implicit merge commit; rebase or revalidate a new exact candidate.
- If any release file overlaps an uncommitted main-worktree path, stop and preserve the user's edits before merging.
- If a publication target already contains 0.4.0, do not overwrite or reuse it.
