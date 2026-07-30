# P2 — safe manual summary

goal_ref: ../goal.md
status: done

## Outcome

Manual summary reports actionable arXiv failures and consistently distinguishes verified detail summaries, safely replaceable generated stubs, and protected user or mismatched notes.

## Assumptions

- The generated lightweight note scaffold has enough stable structure to identify only an empty Notes stub.
- Existing handwritten content must be preserved even when it prevents automatic detail generation.
- A new `note_conflict` result is compatible with current callers once formatting and UI handling are updated together.

## Approach

Build one identity-aware classifier in core, apply it to manual fetch, dashboard history, and deletion validation, and route typed arXiv errors through existing string result boundaries with actionable wording.

## Tasks

- [x] Add and test a shared identity-aware paper-note classifier.
- [x] Branch manual fetch safely for detail, replaceable, conflict, mismatch, and read-error cases.
- [x] Add actionable arXiv error/result formatting and save the fetched abstract in manual index updates.
- [x] Reuse the classifier in dashboard history and deletion validation.
- [x] Update Dashboard and command handling so conflicts are noticed but not opened as success.
- [x] Run focused core/plugin tests, typechecks, and boundary checks.

## Verification

- Existing user content and ID-mismatched notes cause no Atom, LLM, index, or note mutation calls.
- Empty/frontmatter-only notes and exact generated empty stubs are atomically replaced.
- Dashboard history, action state, and deletion validation agree on detail identity.
- 429/503 messages explain rate limiting or temporary unavailability without relying on a long query URL.
- Focused core and plugin suites pass together with typechecks and boundary checks.

## Abort / reshape triggers

- If generated stubs cannot be distinguished conservatively from user notes, only auto-replace empty/frontmatter-only notes and protect all non-empty content.
- If plugin callers cannot accept a new result variant without broad persistence changes, encode the conflict as an existing non-success variant while retaining explicit formatting.
