# Workbench daily calendar

status: done
created: 2026-10-01T23:40:36+08:00
updated: 2026-10-02T00:36:17+08:00
revision: 4
owner: /root

## Intent

Add a compact daily-report calendar to the local reading workbench, following the Obsidian Dashboard's date navigation while preserving the existing Markdown reader and product generation workflow.

## Success criteria

- [x] The daily tab offers month navigation, today, selected-date status and readable markers; paper-note search retains its existing layout.
- [x] Selecting a report date opens its Markdown; selecting other dates exposes accurate state and an explicit generation/retry action when the existing workflow permits it.
- [x] Month state combines real files and full run history: no matches, ungenerated, running, failed, skipped, missing report and future remain distinct. Browsing never calls a model or changes stored data.
- [x] Desktop and collapsible mobile calendars pass browser review; focused tests, relevant regressions and the portable CLI build pass, with screenshots for the user.

## Non-goals

- Replacing Obsidian Dashboard or changing scheduler, recovery, discovery or persistence rules.
- Inferring arXiv publication availability from absent files or automatically generating on date selection.

## Constraints

- Work in the existing `.worktree/claude-code-research-plugin`; preserve production config and data.
- Use the configured product timezone and existing stores; reuse date-grid helpers where practical without changing Obsidian behavior.
- Keep the current Markdown reading area, safe local server and agent-as-auxiliary product boundary.

## Phases

1. P1 — Calendar navigation, authoritative day states and browser acceptance — status: done

2. P2 — Larger calendar cells, semantic colors and per-day paper counts — status: done
