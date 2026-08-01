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

- [x] Commit P2 evidence.
- [x] Fast-forward local main without overlapping seven dirty documentation files.
- [x] Confirm protected file hashes and push corrected main.
- [x] Reconfirm 0.4.1 target availability and workflow prerequisite.
- [x] Stop and request explicit authorization for annotated tag 0.4.1.

## Recorded evidence

- The user explicitly authorized immutable tag `0.4.1`; it resolves to verified release commit `41c19062fd6fa3b30f34ebc5596b69e452863c84`.
- GitHub Release `0.4.1` published successfully with curated notes and exactly `manifest.json`, `main.js`, and `styles.css`; asset attestation completed successfully.
- The original combined workflow's npm step exposed token authentication and package 2FA failures without changing the npm registry.
- CLI publication was moved to OIDC-only `publish-cli.yml`, constrained to immutable stable tags with matching GitHub Releases and guarded against version overwrite.
- npm Trusted Publishing recovery run 30695658781 completed successfully after rerunning the complete candidate gate.
- npm now reports `arxiv-daily@0.4.1` and `dist-tags.latest=0.4.1`, with SLSA provenance and registry signature metadata.
- Local and remote main resolve to `76f950b911b512367c2806522331beb5866b2a9f`; the seven pre-existing documentation edits remain uncommitted and protected.
- Failed tag `0.4.0` was never moved or reused and produced no GitHub Release or npm package.

## Verification

- Local and remote main resolve to the verified candidate.
- Seven pre-existing files remain modified and byte-identical.
- 0.4.1 remains absent from tags, npm, and GitHub Releases.
- No 0.4.1 tag exists before authorization.

## Abort / reshape triggers

- If main diverges or dirty files overlap, stop before merge/push.
- If 0.4.1 is occupied, choose the next version and rerun metadata checks.
