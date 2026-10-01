# Product command reference

The plugin bundles the exact arXiv Daily CLI product build, including native storage and its core dependencies:

```sh
node "${CLAUDE_PLUGIN_ROOT}/dist/arxiv-daily-cli.cjs" COMMAND [ARGUMENTS]
```

| Command | Behavior |
|---|---|
| `init` | Human-operated terminal wizard for CLI TOML; needs an interactive terminal |
| `status` | Read-only JSON: paths, topics, categories, model readiness, paper count, recent run state; no credentials or endpoints |
| `run --today` | Full daily workflow for today in the configured timezone |
| `run --date YYYY-MM-DD` | Full daily workflow for the requested announcement date |
| `run --id ARXIV_ID [--date YYYY-MM-DD]` | Full detailed-note workflow, including Markdown and Paper Index updates |
| `papers [--query TEXT] [--offset N] [--limit N]` | JSON from the existing Paper Index; query uses shared Dashboard lexical search; limit 1–100, default 30 |
| `schedule show` | Inspect the existing scheduling configuration |
| `schedule install` / `schedule uninstall` | Apply/remove the managed OS schedule, only when requested |
| `data export --out PATH.zip` / `data import PATH.zip [--yes]` | Existing Vault-data portability workflow, only when requested |
| `help` | Complete product command help |

Generation commands use progress messages and a final result line, with status 0 for completed/skipped/done/already-exists, 1 for pending/runtime failure, and 2 for configuration/argument failure. A background process ID is not a completed result. `status` and `papers` print JSON; they do not start generation or require a valid model key to inspect existing data.

Configuration is `$XDG_CONFIG_HOME/arxiv-daily/config.toml` (default `~/.config/arxiv-daily/config.toml`) on Linux/macOS, or `%APPDATA%\\arxiv-daily\\config.toml` on Windows. `vault_root` selects the output folder. Settings are not discovered from the current directory and are not automatically copied from Obsidian. The old prototype's `--workspace`, `save`, `read`, `library`, and `confirm-direction` interface has been removed.

Daily reports, detailed notes, Paper Index, run state and checkpoints follow the normal product output layout. Do not directly edit internal JSON, generate replacement reports in chat, or migrate old `arxiv-daily-agent/` records into it. Read the CLI exit status and preserve existing user-authored notes.
