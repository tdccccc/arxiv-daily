# P20 — Automatic settings persistence

goal_ref: ../goal.md
created: 2026-10-05T13:21:53+08:00
updated: 2026-10-05T13:35:43+08:00
revision: 2

## Outcome

Settings persist after user edits without a separate Save click; opening settings never enables scheduling, and completed/manual-only setup does not repeatedly demand Enable.

## Assumptions

- User requested removal of manual Save; actual task buttons (send, build, generate, authorize) stay explicit.
- User clarified Enable means the daily-discovery switch itself. Previous change events only updated local controls, so closing without Save reloaded the previous enabled value. Autosave now persists false before closing; opening never posts or enables it.

## Chunks

1. Automatic settings UI persistence (behavior change, strict Red/Green): tests demonstrate missing autosave, queued edits, secrets, appearance, failed saves, safe close and first setup. Then implement serialization, saved/error feedback and remove visible Save action. Regress existing settings/model/appearance/onboarding/navigation tests and typechecks.
2. Onboarding/Enable (bug fix, strict Red/Green): reproduce recurring guide or toggle issue against user clarification; preserve schedule state and completed setup. No real scheduling/model invocation.
3. Package/docs (nonbehavioral): build next local preview package and update install docs, validate actual isolated DSH Host and CLI suite. No main merge, push or production installation.

## Abort / reshape triggers

Never save incomplete text as an irreversible external operation; no automatic email/library/model tasks. Conflicts and failed saves retain inputs and remain visible. Do not allow concurrent writes or closing to silently lose a pending edit. Preserve first-time setup readiness when required root fields are missing.

## Accepted evidence

- All chunks accepted. Guide regression observed Enable requirement Red then4 setup/onboarding Green; commit4673dd9.
- Autosave implementation observed missing persistence and close behavior Red, then serial revisions, action-gate edits, preferences races, persistent conflicts and discard confirmation Green. Exact user scenario has both X/Escape app regressions and a real temporary HTTP→TOML→reopen check: true→false remains false without manual Save. Implementation ea75859.
- Final full CLI443 passed across45 files, no skips. CLI typecheck, boundaries, product inventory and DSH build passed.20 actual isolated DSH Host checks passed.34 documentation links and package bytes verified. No real model/email/user-corpus or system scheduling operations.
- Local package0.1.18: extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.18.tgz, sha25646d7d07de8df1d72a115d260bb7cfc82bacf4a4db0da436fa41e23b1b55f53cb. No main merge, push, publication or production installation. No Electron visual walkthrough.
- Text edits debounce500ms; selects/toggles save on change. Save requests serialize and retain newer edits. Valid save root is still required to finish first setup. Failure keeps inputs; persistent conflicts offer explicit discard-and-close confirmation. Appearance updates settings in place and refreshes the reading shell on close without losing drafts.
