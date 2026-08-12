# P1 — cli-toml-init

goal_ref: ../goal.md
status: done

## Outcome

CLI uses only `$XDG_CONFIG_HOME/arxiv-daily/config.toml` (Windows APPDATA equivalent), created by interactive `init`; other commands refuse to run without it; env and cwd JSON config surfaces are gone.

## Assumptions

- Personal/server users accept one default vault per OS user account.
- Secrets in TOML are acceptable (same honesty class as plugin data.json).
- Breaking existing `ARXIV_DAILY_*` and JSON-file workflows is OK if release notes and getting-started are updated in this phase or immediately with the release that ships it.

## Approach

Replace CLI config loading with a single XDG TOML path. Map tables to existing `PluginSettings` + `vault_root` / `cache_dir`. Implement `init` as the only writer for first-time setup (plus user hand-edits thereafter). Delete or stop calling env/JSON merge paths. Keep command args for run identity (`--date`, `--id`), not for config location.

## Tasks

- [x] Implement against `../cli-toml-schema.md` (tables: llm / arxiv / email / output / advanced / detail_selection + vault_root + cache_dir).
- [x] Implement XDG/APPDATA path resolution and TOML parse/serialize (dependency choice documented in PR).
- [x] Implement `init`: overwrite/merge/cancel; step order vault → LLM → optional email → placeholder topic → arXiv; write defaults including `cache_dir = <vault>/.cache/arxiv-daily`.
- [x] Gate `run` (`--today` / `--date` / `--id`), `email-*`, `schedule-*` (if present) on config presence with “run init” error.
- [x] Remove public `run-pending` and `summarize`; map pipeline to `run --today`/`--date` and deep-dive to `run --id`.
- [x] Remove `ARXIV_DAILY_*` settings/secrets env application and cwd JSON load; remove `--config` / `--vault-root` / `--cache-dir`.
- [x] Update CLI tests and README / getting-started CLI sections for hard cut + init.
- [x] Smoke: init → run --date (or dry validation) against a temp vault.

## Verification

- Tests: missing config exits non-zero with init hint; init writes file; load round-trips secrets and vault_root; env vars do not change settings when set.
- Manual: `init` then `run --today` without any env.
- `npm run typecheck` / CLI package tests green.

## Abort / reshape triggers

- If TOML dependency or Obsidian boundary rules block the chosen parser, stop and pick an allowlisted approach.
- If hard-cutting env is too harsh for a minor release, L2: keep one minor deprecation only if user explicitly reopens ADR 0003 (default remains hard cut).
