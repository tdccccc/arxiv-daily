# P20 — Automatic settings persistence

goal_ref: ../goal.md
created: 2026-10-05T13:21:53+08:00
updated: 2026-10-05T13:21:53+08:00
revision: 1

## Outcome

Settings persist after user edits without a separate Save click; opening settings never enables scheduling, and completed/manual-only setup does not repeatedly demand Enable.

## Assumptions

- User requested removal of manual Save; actual task buttons (send, build, generate, authorize) stay explicit.
- Clarification pending whether Enable means onboarding or schedule toggle. Existing code ties guide completion to enabled schedule; do not flip user's saved choice.

## Chunks

1. Automatic settings UI persistence (behavior change, strict Red/Green): tests demonstrate missing autosave, queued edits, secrets, appearance, failed saves, safe close and first setup. Then implement serialization, saved/error feedback and remove visible Save action. Regress existing settings/model/appearance/onboarding/navigation tests and typechecks.
2. Onboarding/Enable (bug fix, strict Red/Green): reproduce recurring guide or toggle issue against user clarification; preserve schedule state and completed setup. No real scheduling/model invocation.
3. Package/docs (nonbehavioral): build next local preview package and update install docs, validate actual isolated DSH Host and CLI suite. No main merge, push or production installation.

## Abort / reshape triggers

Never save incomplete text as an irreversible external operation; no automatic email/library/model tasks. Conflicts and failed saves retain inputs and remain visible. Do not allow concurrent writes or closing to silently lose a pending edit. Preserve first-time setup readiness when required root fields are missing.
