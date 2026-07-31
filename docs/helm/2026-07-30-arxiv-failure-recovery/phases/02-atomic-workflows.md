# P2 — atomic workflows

goal_ref: ../goal.md
status: done

## Outcome

Category discovery commits only complete inputs, verified detail notes can repair their index state safely, and content retrieval continues across viable arXiv endpoints after non-cancellation failures.

## Assumptions

- Refetching all configured categories is safer and smaller than persisting per-category progress.
- Existing verified detail notes contain enough frontmatter or index context for network-free reconciliation.
- Source and abs fallbacks are useful after rendered-HTML transport failures.

## Approach

Make multi-category discovery atomic, introduce verified detail reconciliation rather than existence-based repair, and treat non-cancellation HTML/cache failures as fallback signals instead of terminal errors.

## Tasks

- [x] Fail an entire category discovery attempt when any configured category fails, preserving transient/permanent classification.
- [x] Verify no partial report/index mutations occur and a later full retry succeeds with canonical merge semantics.
- [x] Reconcile verified existing manual detail notes and post-write index failures without Atom or LLM calls.
- [x] Require classifier verification for daily pipeline detail-path repair.
- [x] Continue from thrown HTML/cache errors to source and abs fallbacks while preserving cancellation.
- [x] Add focused source, pipeline, manual, writer, and content fallback tests.
- [x] Run focused suites, affected typechecks, and boundary checks.

## Verification

- One failed category writes no daily report and the next invocation refetches every category.
- A verified detail with missing index state is repaired on retry without network or LLM work.
- Handwritten, mismatched, empty, or unreadable notes never become detail solely by path existence.
- HTML 429/503/network/cache exceptions still reach source/abs fallbacks; cancellation does not.
- Focused core/plugin regression tests and typechecks pass.

## Abort / reshape triggers

- If existing detail frontmatter cannot support safe network-free reconciliation, return an explicit repair-needed result rather than fetching or fabricating metadata.
- If atomic category failure conflicts with required partial availability, stop and design persisted category provenance before writing partial reports.
