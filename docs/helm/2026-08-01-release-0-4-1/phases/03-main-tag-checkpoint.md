# P3 — main and tag checkpoint

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

The verified 0.4.1 fix is on remote main, all local edits remain protected, and publication stops before creating the separately authorized immutable 0.4.1 tag.

## Assumptions

- Main can fast-forward from the published 0.4.0 tag commit to the verified candidate.
- Pushing main is a reversible code publication relative to creating the immutable product tag and was included in the fix-forward plan.
- A new tag requires explicit authorization because the original authorization named 0.4.0.

## Approach

Commit evidence, fast-forward local main, verify protected hashes, push main, reconfirm target availability, and present the exact tag action for authorization.

## Tasks

- [ ] Commit P2 evidence.
- [ ] Fast-forward local main without overlapping seven dirty documentation files.
- [ ] Confirm protected file hashes and push corrected main.
- [ ] Reconfirm 0.4.1 target availability and workflow prerequisite.
- [ ] Stop and request explicit authorization for annotated tag 0.4.1.

## Verification

- Local and remote main resolve to the verified candidate.
- Seven pre-existing files remain modified and byte-identical.
- 0.4.1 remains absent from tags, npm, and GitHub Releases.
- No 0.4.1 tag exists before authorization.

## Abort / reshape triggers

- If main diverges or dirty files overlap, stop before merge/push.
- If 0.4.1 is occupied, choose the next version and rerun metadata checks.
