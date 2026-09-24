# Onboarding and main review fixes

status: active
created: 2026-09-24T20:50:00+08:00
updated: 2026-09-24T22:05:00+08:00
revision: 4
owner: claude-main-session-2026-09-24

## Intent

Fix the defects confirmed in `docs/reviews/2026-09-24-main-review.md`, starting
with the first-run guide on Obsidian 1.13+ that users reported as "buttons do
nothing", so a new user can go from install to a first report and a running
daily schedule without dead ends.

## Success criteria

- [x] On Obsidian 1.13+, every guide action button visibly does something (scrolls to and focuses its section, or explains why it cannot); confirmed in a real Obsidian 1.13.7 session
- [x] Editing a research topic never re-renders the page or drops focus, before or after setup completes; confirmed in real Obsidian
- [x] The guide never shows all earlier steps complete while the run is blocked without saying why; new topics never get a duplicate tag
- [x] "Generate first report" succeeds on weekends / before the day's announcement by using the latest announced date
- [x] The guide includes a step to turn on the daily schedule and is not "complete" until it is on (user decision 2026-09-24)
- [ ] Every other review finding marked for fixing in the phase list is fixed or explicitly waived in `journal.md`
- [ ] Each fix lands as its own commit with a failing-first test; `npm run lint`, `npm run typecheck`, `npm test`, `npm run build`, `npm run check:boundaries`, `npm run check:obsidian-submission` stay green

## Non-goals

- No quick-start topic templates on the 1.13+ settings page (user decision 2026-09-24)
- No settings page redesign beyond the guide card layout fix
- No push / PR without explicit user approval

## Constraints

- Branch `fix/onboarding-review` from `origin/main` (976c12b); local `main` untouched
- Commit messages: several `-m` flags — subject, Why, What, Validation; never heredoc
- At most one helper agent for the whole task (user instruction)
- Real-Obsidian checks use the desktop acceptance harness under a virtual display with an isolated config; probe scripts stay in `/tmp`

## Phases

<!-- Single source of truth for phase status. PN ↔ filename NN. Integer IDs are permanent and never reused; replacement phases take the next unused number. Outcomes only — no steps. The active line is the current focus. -->
1. P1 — 1.13+ guide buttons reach and focus their section; topic edits keep focus after setup (F1, F2) — status: done
2. P2 — Guide tells the truth: unique topic tags, visible blocking reasons, schedule step, busy state, copy, full-width card, last-category delete (F4, F10, F7, F6, F8, F9) — status: done
3. P3 — First report uses the latest announced date; runs refresh open dashboards (F3, F18) — status: done
4. P4 — Settings controls commit once and consistently (F11, F12, F13, F14, F16) — status: active
5. P5 — Confirmed core / CLI / relay defects from the review are fixed — status: pending
6. P6 — Remaining low-priority findings fixed or waived (C1, C4, F15, F17) — status: pending

## Decisions

- 2026-09-24 (user): guide buttons scroll and focus the first pending field; no 1.13+ topic templates; guide gets a fifth "turn on daily reports" step; full-width guide card; model field accepts free text and Get models never replaces it.
