# arXiv Daily CLI

Command-line tool for [arXiv Daily](https://github.com/tdccccc/arxiv-daily): fetch arXiv by category, filter with an LLM by your research topics, and write Markdown **daily reports** (and optional **paper notes**).

Works standalone on a server or always-on machine. The Obsidian plugin is separate; both can share the same vault folder layout.

## Requirements

- Node.js **20.11.0** or newer

## Install

```bash
npm install -g arxiv-daily
```

Or without a global install:

```bash
npx arxiv-daily --help
```

## Quick start

```bash
arxiv-daily init
# edit ~/.config/arxiv-daily/config.toml  (topics, API keys, vault_root)
arxiv-daily run --today
```

Config path:

- Linux/macOS: `$XDG_CONFIG_HOME/arxiv-daily/config.toml` (default `~/.config/arxiv-daily/config.toml`)
- Windows: `%APPDATA%\arxiv-daily\config.toml`

There are no settings environment variables and no `--config` / `--vault-root` flags.

## Commands

```text
arxiv-daily init
arxiv-daily run --today
arxiv-daily run --date YYYY-MM-DD
arxiv-daily run --id ARXIV_ID [--date YYYY-MM-DD]
arxiv-daily email test|status|verify-start
arxiv-daily schedule show|install|uninstall
arxiv-daily data export --out PATH.zip
arxiv-daily data import PATH.zip [--yes]
arxiv-daily help
```

- **`run --today`** — one day only (typical cron entry). Missed days: `run --date …`.
- **`schedule install`** — writes managed user crontab lines (Linux/macOS/WSL). Not supported on native Windows Task Scheduler; use WSL or the Obsidian plugin for desktop scheduling.

## Cron example

```cron
30 9 * * 1-5  arxiv-daily run --today
```

Or set `[schedule]` in the config file and run `arxiv-daily schedule install`.

## Develop from the monorepo

This package is built from the [arxiv-daily](https://github.com/tdccccc/arxiv-daily) workspace:

```bash
git clone https://github.com/tdccccc/arxiv-daily.git
cd arxiv-daily
npm ci
npm run build
npm run cli -- run --today    # from repo root
# or after build:
npx arxiv-daily run --today
```

The published tarball ships a single bundled binary (`dist/arxiv-daily-cli.cjs`).

## License

MIT — see [LICENSE](./LICENSE).
