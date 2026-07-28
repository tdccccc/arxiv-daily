# CLI command surface (target)

status: draft (docs only; not fully implemented)  
decision_ref: `docs/adr/0003-two-products-cli-config-and-data-portability.md`  
schema_ref: `./cli-toml-schema.md`  
goal_ref: `./goal.md`

Binary name: **`arxiv-daily`** (npm package name: **`arxiv-daily`**).

Install for users: `npm install -g arxiv-daily` then `arxiv-daily …`.  
Developers in this monorepo may use `npm run cli -- …` (runs the local dist).

Config: only `$XDG_CONFIG_HOME/arxiv-daily/config.toml`.  
**No** `--config` / `--vault-root` / `--cache-dir` / `ARXIV_DAILY_*` settings env.

---

## 0. Global rules

| Rule | Detail |
|---|---|
| Help | `arxiv-daily help` or `--help` / `-h` |
| Missing config | Any command except `help` / `init` → non-zero + “run `arxiv-daily init`” |
| Exit codes | `0` ok; `1` runtime failure; `2` usage/config error |

**Target USAGE:**

```text
Usage:
  arxiv-daily init
  arxiv-daily run --today
  arxiv-daily run --date YYYY-MM-DD
  arxiv-daily run --id ARXIV_ID [--date YYYY-MM-DD]
  arxiv-daily email test [--date YYYY-MM-DD]
  arxiv-daily email status
  arxiv-daily email verify-start
  arxiv-daily schedule show
  arxiv-daily schedule install
  arxiv-daily schedule uninstall
  arxiv-daily data export --out PATH.zip
  arxiv-daily data import PATH.zip [--yes]
  arxiv-daily help
```

---

## 1. Setup

### `init`

| | |
|---|---|
| Purpose | First-run wizard; write annotated `config.toml` |
| Needs config? | No. If exists → overwrite / merge / cancel |
| Args | none (v1) |
| Writes | Commented TOML per schema (default `[schedule]`, `enabled=false`) |
| Does not ask | schedule fields; detail profile; hosted_base_url |

---

## 2. `run` — one mode only (no multi-day batch)

**Product decision:** CLI does **not** expose lookback “run all pending days”.  
Each invocation runs **at most one** daily date, or **one** paper deep-dive.

Mutually exclusive modes (exactly one of `--today` | `--date` | `--id`):

| Mode | Command | Behavior |
|---|---|---|
| Today | `run --today` | Daily pipeline for **today** in `arxiv.timezone` |
| Date | `run --date YYYY-MM-DD` | Daily pipeline for that date only |
| Paper | `run --id ARXIV_ID [--date YYYY-MM-DD]` | Manual deep-dive for one arXiv id (former `summarize`). Optional `--date` defaults to today in timezone (note dating only) |

### Rules

- **Do not** accept `--today` together with `--date`, or either with `--id` → usage error (exit 2).
- Daily modes (`--today` / `--date`): full daily pipeline; may auto-email if `email.enabled` and completed.
- `--id` mode: **no** daily multi-paper run; writes/updates paper note path as today; **no** lookback.
- If the chosen daily date is already completed, behavior follows existing scheduler/pipeline skip semantics for that single date (idempotent), not “scan other days”.
- **Removed:** `run-pending`, `summarize` as public commands (no aliases required in v1 target; implementer may keep hidden aliases one release if needed—default **no**).

### Cron / schedule install

Generated crontab lines must call:

```text
arxiv-daily run --today
```

not a multi-day catch-up. Missed days: user re-runs with `run --date …` manually (or accepts gap). Document this tradeoff.

---

## 3. Email (`email …`)

### `email test [--date YYYY-MM-DD]`

Force sample/test digest; does not mark calendar day delivered.

### `email status`

Print mode, to, enabled, credentials ready (no send).

### `email verify-start`

Official delivery: magic-link to `email.to`; user pastes `hosted_token` into TOML.

**Not in v1:** `email verify-save`, `hosted_base_url` in config.

Transition: old `email-test` may map to `email test` for one release (optional).

---

## 4. Schedule (`schedule …`)

| Command | Purpose |
|---|---|
| `schedule show` | Print managed cron lines from `[schedule]` (fires = `run --today`) |
| `schedule install` | Install/replace `# arxiv-daily-managed` lines; requires `schedule.enabled=true` |
| `schedule uninstall` | Remove managed lines only |

No in-process daemon. Re-install after editing `[schedule]`.

**Windows:** `schedule install` / `uninstall` exit with a clear error and print the would-be cron lines. Prefer **WSL** for CLI + cron, or use the **Obsidian plugin** for desktop scheduling. `init` / `run` / `email` / `data` still work with `%APPDATA%\arxiv-daily\config.toml`.

---

## 5. Data (`data …`)

| Command | Purpose |
|---|---|
| `data export --out PATH.zip` | `daily` + `papers` + `.index` + manifest |
| `data import PATH.zip [--yes]` | mtime plan; TTY confirm; non-TTY needs `--yes` to write |

---

## 6. Removed / not added

| Item | Status |
|---|---|
| `run-pending` | **removed** (no multi-day CLI batch) |
| `summarize` | **removed** → `run --id` |
| `--config` / `--vault-root` / `--cache-dir` | **removed** |
| `ARXIV_DAILY_*` config env | **removed** |
| `serve` / daemon | **not added** |
| Plugin “run all pending lookback” | **plugin-only**; not CLI |

---

## 7. Phase mapping

| Phase | Commands |
|---|---|
| **P1** | `init`; `run --today` / `--date` / `--id`; TOML load; drop env/flags/old names |
| **P1b** | `schedule show\|install\|uninstall` → install `run --today` |
| **P2** | `email test\|status\|verify-start` |
| **P3** | `data export\|import` |

---

## 8. Today (code) vs target

| Today | Target |
|---|---|
| `run --date` | keep |
| `run-pending` (lookback batch) | **drop** → use `run --today` for cron |
| `summarize --id` | **`run --id`** |
| `email-test` | `email test` |
| flags + env config | fixed XDG TOML only |

---

## 9. Naming locked

1. Single verb **`run`** with flags `--today` | `--date` | `--id`.  
2. Cron entrypoint: **`run --today`**.  
3. Groups: `email`, `schedule`, `data`.  
4. No public `run-pending` / `summarize`.
