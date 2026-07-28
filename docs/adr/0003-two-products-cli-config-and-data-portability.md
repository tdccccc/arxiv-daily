# ADR 0003: Two products, CLI config home, and vault data portability

Status: Accepted (2026-07-28 grill-with-docs)

Related: ADR 0001 (one TypeScript core, two hosts); ADR 0002 (paper identity, dual-host email delivery).

## Context

The repository already shares one business core between the Obsidian plugin and a Node CLI (ADR 0001), and both hosts may send email digests with vault-backed delivery idempotency (ADR 0002). Users increasingly want the CLI on servers or always-on machines (including VPN egress) while still reading Markdown in Obsidian.

During design we considered:

1. **Dual-host config auto-consistency** (shared vault-side product settings, conflict protocols).
2. **Bidirectional settings export/import** between plugin and CLI.
3. **Two products** with independent configuration, optional **vault data** export/import only.

Auto-consistent settings and bidirectional settings sync add conflict rules, secret placement debates, and rare real-world dual-writer scenarios. Most users run one “driver” at a time (plugin only, CLI only, or “configure once then cron”). Convenience of a single CLI config file (including secrets) outweighs splitting secrets into environment variables for the intended personal/server audience.

## Decision

### 1. Two products, one engine

- Treat **plugin** and **CLI** as **two products** that share **core** behavior and, when pointed at the same vault, **vault data** semantics.
- **Product settings are not automatically synchronized.** Each product is configured in its own store. Users may mirror choices manually.
- Do **not** ship configuration import/export between products in the initiative that follows this ADR (can be revisited later).

### 2. CLI configuration model

| Rule | Detail |
|---|---|
| Format | **TOML** only for the supported CLI config file |
| Location | **XDG config home**: `$XDG_CONFIG_HOME/arxiv-daily/config.toml`, default `~/.config/arxiv-daily/config.toml`; on Windows, `%APPDATA%\arxiv-daily\config.toml` |
| Contents | Product settings **and secrets** (LLM key, Resend key, official-delivery token) plus deployment fields such as `vault_root` and `cache_dir` |
| Discovery | **Only** that fixed path. No cwd search for config. No `ARXIV_DAILY_*` environment variables for settings or secrets. No `--config`, `--vault-root`, or `--cache-dir` overrides |
| First run | User must run interactive **`init`** successfully before other CLI commands; missing config → clear error pointing at `init` |
| Legacy JSON | **Hard cut**: do not read `arxiv-daily.config.json` or other cwd JSON configs; document manual field mapping for anyone still on JSON |
| Path flags | Command **arguments** that select work (e.g. `--date`, arXiv id) remain; they are not configuration overrides |

Default **cache_dir** written by `init`: `<vault_root>/.cache/arxiv-daily` (outside the daily/papers output tree, still vault-local).

**`init` behavior (normative intent):**

- Interactive wizard; if config already exists → ask **overwrite / merge / cancel**.
- Step order: **vault path → LLM → optional email → placeholder topic → arXiv categories / timezone / summary language**.
- Email step: ask whether to configure mail; if yes, choose Send yourself vs Official delivery and collect to + credentials/token as applicable; if no, leave email disabled.
- Topics: write a **placeholder** topic; tell the user to edit the topics section in the TOML for real research text.
- Secret file hygiene: **no special chmod or gitignore automation** (personal convenience preferred).

### 3. Vault data portability (not settings)

Ship **data export/import** for vault data so users can back up or move between machines:

| Rule | Detail |
|---|---|
| Default contents | Logical **`daily/`**, **`papers/`**, **`.index/`** |
| Excluded | Product settings; **PDFs** (no include switch in v1) |
| Surfaces | **Both** plugin UI and CLI |
| Package | **Zip** with **logical** layout + thin **`arxiv-daily-export.json`** manifest (`formatVersion`, `exportedAt`, `contents`, …) |
| Path mapping | Import maps logical folders onto the **target product’s current output directories** |
| Conflicts | Compare **filesystem mtime**; keep newer; **preview then confirm or cancel** only (no per-file strategy edits) |
| Index files | Same whole-file conflict model as Markdown (including `run-state` / `delivery-state`) |
| CLI non-TTY | Preview by default; require **`--yes`** to apply |

Export implementations should preserve source mtimes when writing the zip so import comparisons remain meaningful.

### 4. Implementation order

1. **CLI config base**: TOML load/store at XDG path, remove env-based settings, `init`, hard cut from JSON.
2. **CLI email completeness**: status/verify helpers and docs on the new TOML fields (not env).
3. **Data export/import** as specified above.

### 5. CLI schedule intent → OS timer (not in-process)

- TOML may include a CLI **`[schedule]`** block: `enabled`, `on` (`HH:MM`), `interval_hours` (default **0** = once daily), `until`, `weekdays_only`.
- This is **not** the plugin’s in-process tick window. The CLI process does not sleep or poll.
- Applying the schedule means generating **managed user crontab** lines (or platform equivalent) via explicit commands, e.g. `schedule install` / `show` / `uninstall`.
- **`init` writes default `[schedule]` comments and values; it does not ask schedule questions.** Users edit the file, set `enabled = true`, then run `schedule install`.
- Changing TOML does not update the OS timer until install is run again.

### 6. What this does not change

- ADR 0001: one core, one-shot CLI, no daemon.
- ADR 0002: dual-host email channel, digest contract, delivery-state idempotency when the same vault is used.
- Plugin continues to use its Obsidian settings store; plugin schedule remains “while Obsidian is open.”

## Consequences

- CLI documentation becomes “`init` once → edit `~/.config/arxiv-daily/config.toml` → optional `schedule install` or hand cron `run --today`,” not env recipes. CLI does not batch lookback days (`run-pending` retired); missed days use `run --date`.
- Existing scripts that rely on `ARXIV_DAILY_*` or cwd JSON **break** by design; release notes must state the hard cut.
- Single global CLI config implies **one default vault per user account** until a future profile feature exists.
- Secrets on disk in TOML match the plugin’s plaintext-secret honesty model; users who need stronger isolation must protect the home config directory themselves.
- Data portability improves machine mobility without reopening settings-sync complexity.
- Plugin and CLI can diverge in topics or email mode; only vault data and delivery-state keep shared operational memory when the vault is shared.

## Non-decisions

- Exact TOML key names and table layout (implementer; keep close to existing `PluginSettings` shape).
- Whether a later release adds config profiles or settings export.
- Official delivery Worker operations (separate from CLI config home).
- Automatic rebuild of paper index from Markdown-only folders (import assumes `.index` is in the package when the user exported it).
