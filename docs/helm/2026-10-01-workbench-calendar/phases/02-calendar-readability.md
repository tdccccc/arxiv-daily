# P2 — calendar-readability

goal_ref: ../goal.md
created: 2026-10-02T00:20:27+08:00
updated: 2026-10-02T00:20:27+08:00
revision: 1

## Outcome

The calendar is comfortably readable in a wider sidebar, with colored day cells and visible paper counts like the Obsidian Dashboard.

## Assumptions

- Existing numeric run counts remain authoritative; missing counts can be recovered from unique Paper Index references to a report.
- Unknown counts must not appear as zero. Index lookup is optional and must not break reading if unavailable.
- Desktop sidebar can grow to about 420px while the article remains readable; small screens retain the collapsible calendar.

## Approach

Increase sidebar and calendar typography/cell dimensions; use blue/green/amber/red/neutral status surfaces and retain independent selected/today/focus styling. Render counts in each report cell with accessible labels. Fall back to read-only index counts when a report lacks a run count.

Visual thesis: a more prominent, legible calendar within the existing restrained reading layout.
Content plan: month controls, larger date/count cells, status legend, selected-day summary, report list.
Interaction thesis: preserve the accepted navigation/selection/dialog behavior; improve visibility without adding interactions.

## Chunks

### Chunk 1 — Counts and visual readability

- change kind: behavior change plus visual styling
- strategy: strict Red-Green for count display and index fallback; direct browser inspection for CSS
- Red: existing UI fails visible known/zero/unknown count assertions; HTTP calendar fails to recover a count from persisted report references
- Green: targeted calendar HTTP/UI tests pass with read-only index fallback and accessible count labels
- regressions: existing calendar navigation tests, CLI typecheck, packaged workflow test
- visual verification: desktop 1440px and medium 1100px preserve article space; 390px mobile and dark theme have no horizontal overflow, legible counts and distinct selection/status colors
- [ ] implementation and tests accepted

## Abort / reshape triggers

- If widening the sidebar makes the reader unusably narrow, hide the optional ToC earlier and adapt the sidebar at intermediate widths.
- If no authoritative count exists, show an unknown count rather than infer from Markdown headings or fabricate zero.
