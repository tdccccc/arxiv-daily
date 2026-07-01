# Codex Task: log-audit

task_id: log-audit
target_project: /home/tiandc/Documents/code/arxiv-daily
task_kind: implementation
mode: semi-auto
sandbox: workspace-write
provider: bnu
artifact_policy: keep-report-only
source: claude-code

## Goal

Add comprehensive Logger.info/warn/error calls to every user-triggered operation in the arXiv Daily Obsidian plugin, ensuring all actions are visible in the in-memory log buffer.

## Context

The detailed task breakdown is at `docs/tasks/log-audit/task.md`. Read it first.

Several files have already been instrumented (logger.ts, state-store.ts, run-lock.ts, pipeline files, scheduler.ts, dashboard LogModal). Do NOT touch those.

The working directory is `/home/tiandc/Documents/code/arxiv-daily`. Source is under `plugin/src/`.

There are uncommitted user changes (`git status --short` shows modified files). Preserve them.

## Scope

Allowed:

- Edit files under `plugin/src/` to add logger calls
- Edit `plugin/src/dashboard/view.ts` for Sections A and C
- Edit `plugin/src/commands.ts` for Section B
- Edit `plugin/src/services/manual-fetch.ts` for Section D
- Edit `plugin/src/services/paper-note.ts` for Section E
- Edit `plugin/src/onboarding.ts` for Section F
- Search the entire `plugin/src/` for empty catch blocks and add logging (Section G)
- Run `cd /home/tiandc/Documents/code/arxiv-daily/plugin && npx tsc --noEmit` to verify compilation
- Write report to `docs/tasks/log-audit/codex-report.md`

Out of scope:

- Files listed in the "Already Done" section of the task file
- Adding console.log anywhere (use Logger class only)
- Git add/commit operations
- Changes outside `plugin/src/`

## Constraints

- Do not run `git add`.
- Do not run `git commit`.
- Do not write temporary files outside `.codex-runs/log-audit/`.
- Preserve unrelated user changes.
- Ask for approval before using network access, installing dependencies, writing outside the target project, running destructive commands, or changing persistent databases.
- Ensure TypeScript compiles with zero errors: `cd /home/tiandc/Documents/code/arxiv-daily/plugin && npx tsc --noEmit`
- Do NOT touch files listed in "Already Done" section of `docs/tasks/log-audit/task.md`.

## Verification

Commands:

1. `cd /home/tiandc/Documents/code/arxiv-daily/plugin && npx tsc --noEmit`

Expected result:

- TypeScript compiles with zero errors

## Report

Write report to:

```
docs/tasks/log-audit/codex-report.md
```

Write a report even on failure. Include:
- Status (success / failed / partial)
- Which sections (A-G) were completed
- Summary of all logger.info/warn/error lines added (file, line context, message)
- Changed files list
- Verification results
- Any risks or follow-ups
- Suggested conventional commit message
