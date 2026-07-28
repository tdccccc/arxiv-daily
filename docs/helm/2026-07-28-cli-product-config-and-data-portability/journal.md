## 2026-07-28 — note (grill-with-docs closed)

- evidence: Session grilled product path (two products vs auto-consistent settings), rejected settings sync, chose vault **data** portability, then reframed CLI as fixed XDG TOML + `init`, secrets in file, hard cut env/JSON, no override flags. Decisions captured in ADR 0003 and CONTEXT.md.
- change: New initiative `2026-07-28-cli-product-config-and-data-portability`; phase order config base → email → data export/import.
- disposition: Documentation only this turn; no application code.
- next: Implement P1 when user asks — TOML load path, strip env config, `init` wizard, tests/docs for hard cut.

## 2026-07-28 — note (TOML field draft)

- evidence: User asked for a docs-only TOML field draft before implementation.
- change: Added `cli-toml-schema.md` — path, snake_case tables, core mapping, init email branch, schedule ignored, JSON hand-migration cheat sheet, full sample.
- disposition: Draft only; not wired to code.
- next: Still P1 implement when requested; schema is the field checklist.

## 2026-07-28 — note (TOML UX: comments, no profile, no hosted_base_url)

- evidence: User: no profile surface; leave room for agent-edited topics; TOML needs comments; drop knobs users need not set (e.g. hosted_base_url).
- change: Rewrote `cli-toml-schema.md` — init writes annotated template; omit detail_selection/profile, schedule, hosted_base_url; deep-dive default = core balanced + per-topic `detail`; agent blurb in header; 中文注释 + English keys.
- disposition: Docs only.
- next: P1 implement when requested.

## 2026-07-28 — note (CLI schedule B: toml + cron install)

- evidence: User chose B — `[schedule]` in toml + install into cron; optional interval with defaults; init does not ask schedule questions, only writes defaults.
- change: Updated `cli-toml-schema.md` and ADR 0003 §5: fields `enabled`/`on`/`interval_hours` (0=once)/`until`/`weekdays_only`; commands `schedule show|install|uninstall`; no in-process tick.
- disposition: Docs only.
- next: Implement schedule commands with or right after P1 config base.

## 2026-07-28 — note (CLI command surface draft)

- evidence: User asked to lock CLI commands.
- change: Added `cli-commands.md` — full target surface, phase map, removed flags, email/schedule/data groups.
- disposition: Docs only; await user name tweaks.
- next: Confirm commands then P1 implement.

## 2026-07-28 — note (run surface simplified)

- evidence: User: drop `run-pending` (no multi-day batch); use `run --today` aligned with `run --date`; replace `summarize --id` with `run --id`.
- change: Rewrote `cli-commands.md`; cron/schedule install → `run --today`; updated schema, ADR 0003 consequence, P1 tasks.
- disposition: Docs only. Tradeoff: missed days need manual `run --date`.
- next: Confirm remaining names (email/schedule/data) then P1 implement.

## 2026-07-28 — note (implementation landed)

- evidence: User asked to implement per current docs (P1–P3).
- change: CLI rewritten — XDG TOML config (`smol-toml`), `init`, `run --today|--date|--id`, `email test|status|verify-start`, `schedule show|install|uninstall`, `data export|import` (jszip); removed env/JSON flags/run-pending/summarize public commands; tests updated; README + zh README CLI sections updated.
- disposition: Keep; run `npm run test -w @arxiv-daily/cli` and typecheck green before commit.
- next: Optional polish (interactive import confirm UX, Windows schedule install); user may want release notes.

## 2026-07-28 — note (branch + Windows schedule scope)

- evidence: User noted work was on main without branch/worktree; Windows schedule can be deferred with WSL/Git Bash guidance.
- change: Moved uncommitted work to branch `feat/cli-toml-product-surface` (same worktree). `schedule install|uninstall` refuse native win32 with WSL/Git Bash message + printed lines; docs/README updated. No Task Scheduler.
- disposition: Continue on feature branch; still not committed unless user asks.
- next: Commit/PR when user wants; optional plugin data UI later.

## 2026-07-28 — note (Windows messaging: WSL or plugin)

- evidence: User: Windows guidance should be WSL, or recommend the plugin—not Git Bash.
- change: schedule-cmd win32 errors + README/zh/commands/schema: WSL for CLI cron, Obsidian plugin for desktop schedule; drop Git Bash as primary tip.
- disposition: Keep.
- next: unchanged.

## 2026-07-28 — note (README user-facing rewrite)

- evidence: User grilled README: drop AI-ish / defensive openers, dedupe flow, less implementer detail; dual product sections; outputs = daily report + paper note (not “deep dive” as hero); EN primary + 中文 link; one-line optional email.
- change: Rewrote `README.md` and `docs/README.zh-CN.md`; glossary in `CONTEXT.md` for Daily report / Paper note / structured summary.
- disposition: Overview only; getting-started still has older deep-dive wording (follow-up if desired).
- next: Optional align getting-started terminology; commit when user asks.

## 2026-07-28 — note (getting-started user rewrite)

- evidence: User: commit first, then unify getting-started in user voice + Daily report / Paper note terms.
- change: Rewrote `docs/getting-started.md` and `getting-started.zh-CN.md` — shorter, user steps, drop schema/algorithm/requestUrl; paper notes not “deep dive” hero; CLI pointer only.
- disposition: Keep; commit as docs follow-up.
- next: User may want another commit.

## 2026-07-28 — note (init UX: guided lists + English toml)

- evidence: User: init too harsh for first-time users — need field explanations, category picker, interactive hosted verify; config.toml comments in English.
- change: Rewrote `apps/cli/src/init.ts` — section help, provider/model pickers, category group+list, email menu with optional verify-start + paste token, timezone list; `renderInitToml` English comments; tests in cli-init.test.ts.
- disposition: Keep on branch feat/cli-init-ux; bump npm version when publishing again.
- next: Commit / PR; local try `arxiv-daily init` after build.
