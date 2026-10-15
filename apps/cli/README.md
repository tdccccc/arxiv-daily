English · [简体中文](https://github.com/tdccccc/arxiv-daily/blob/main/apps/cli/README.zh-CN.md)

# arXiv Daily CLI

Fetch arXiv by category, filter with an LLM against your research topics, and write Markdown **daily reports** and **paper notes** — from a terminal, a local browser, or a server/cron box.

Part of [arXiv Daily](https://github.com/tdccccc/arxiv-daily). The Obsidian plugin is separate and optional; both can share the same vault folder.

## Features

- **Daily filtering** — keeps only papers matching the directions in your research topics
- **Local browser workbench** — a reader at `127.0.0.1`, no Obsidian required (`arxiv-daily ui`)
- **Scheduling** — cron on Linux/macOS/WSL, or the in-browser scheduler while the workbench is open
- **Personal library** — index your own PDFs and turn them into reviewed, accepted research topics
- **Resumable runs** — an interrupted run doesn't repeat paid model calls on retry
- **Email digests** — optional, via your own Resend key or hosted delivery

## Requirements

- Node.js **20.19.0** or newer

## Install

```bash
npm install -g arxiv-daily
```

Or without a global install:

```bash
npx arxiv-daily@latest help
```

Package page: https://www.npmjs.com/package/arxiv-daily

## Quick start

```bash
arxiv-daily init        # guided setup: vault, LLM, categories, topics, …
arxiv-daily run --today
arxiv-daily ui           # open the local reading workbench
```

`init` is a TUI wizard — **↑/↓** move, **Space** multi-select, **Enter** confirm/accept default, **Ctrl+C**/**Esc** exit. It walks through: vault location → LLM provider/API key/model → optional email → arXiv categories → timezone → report language → research topics → optional paper-note and schedule flags.

## Configuration

| | |
|---|---|
| Linux/macOS | `$XDG_CONFIG_HOME/arxiv-daily/config.toml` (default `~/.config/arxiv-daily/config.toml`) |
| Windows | `%APPDATA%\arxiv-daily\config.toml` |

Default vault from `init`: `~/arxiv-daily`. There are no settings env vars and no `--config`/`--vault-root` flags. The file holds your API keys in plain text — lock it down:

```bash
chmod 600 ~/.config/arxiv-daily/config.toml
```

**Daily paper limit** — new daily reports contain at most 20 papers across all topics. Change it in the existing `[output]` table:

```toml
[output]
max_daily_papers = 20
```

Zero, negative numbers, fractions, and quoted strings are rejected; missing values fall back to 20. Already-written reports aren't rewritten when you change this.

## Commands

| Command | What it does |
|---|---|
| `init` | Guided setup wizard |
| `status` | JSON overview: output paths, topics, model readiness, paper count, last run states |
| `ui [--port PORT] [--no-open]` | Open the local reading [workbench](https://github.com/tdccccc/arxiv-daily/blob/main/docs/workbench.md) |
| `papers [--query TEXT] [--offset N] [--limit N]` | Paginated JSON from the local Paper Index (default limit 30, max 100) |
| `library …` | Connect and index a personal PDF folder — see [below](#optional-personal-library) |
| `update [--check] [--yes]` | Check npm for a newer version and optionally install it; `--check` only reports |
| `run --today` | Generate today's report (typical cron entry) |
| `run --date YYYY-MM-DD` | Generate (or retry) a report for a specific date |
| `run --id ARXIV_ID [--date YYYY-MM-DD]` | Generate a paper note for one arXiv ID |
| `email test\|status\|verify-start` | Send a test digest, check delivery config, or start hosted-email verification |
| `schedule show\|install\|uninstall` | Inspect or manage the cron entry (Linux/macOS/WSL only) |
| `data export --out PATH.zip` / `data import PATH.zip [--yes]` | Back up or restore your vault |
| `help` | Print full usage |

An interrupted run resumes without repeating paid model calls; to force a clean rerun, delete `<output-root>/.index/*-checkpoints/<date>*`.

## Local reading workbench

```bash
arxiv-daily ui                   # open in your default browser
arxiv-daily ui --port 8123       # fixed port instead of an auto-picked one
arxiv-daily ui --no-open         # print the URL instead of opening a browser
```

Serves only on `127.0.0.1`. Full feature guide: [docs/workbench.md](https://github.com/tdccccc/arxiv-daily/blob/main/docs/workbench.md).

## Scheduling

```cron
30 9 * * 1-5  arxiv-daily run --today
```

Or configure `[schedule]` in `config.toml` and let the CLI manage it:

```bash
arxiv-daily schedule install
```

`schedule install`/`uninstall` write a managed crontab entry on Linux/macOS/WSL. Native Windows Task Scheduler isn't supported — use WSL, or run the scheduler built into the Obsidian plugin or the workbench.

## Optional personal library

Index your own PDFs so matching papers join daily filtering alongside your manually entered topics. Access is read-only and limited to the folder you connect.

```bash
arxiv-daily library connect /path/to/pdfs
arxiv-daily library status                        # scope + authorization fingerprint
arxiv-daily library authorize --fingerprint HASH   # authorize that exact scope
arxiv-daily library prepare                        # install optional PDF/embedding runtime
arxiv-daily library scan
arxiv-daily library index
arxiv-daily library propose                        # generate topic/direction candidates
arxiv-daily library propose --new                  # after adding papers: send only the new ones, with your existing directions
arxiv-daily library directions                     # review candidates
arxiv-daily library confirm --candidate ID --proposal-revision N
# or review/accept several at once:
arxiv-daily library review --input REQUEST.json
```

Accepted directions become ordinary research-topic settings. `library search --query TEXT` searches the indexed library directly. The [workbench](https://github.com/tdccccc/arxiv-daily/blob/main/docs/workbench.md) also supports connecting, indexing, browsing and direction review from a browser.

## Update

```bash
arxiv-daily update
```

Checks npm for a newer `arxiv-daily` and optionally installs it; your config is untouched.

## Uninstall

```bash
npm uninstall -g arxiv-daily
```

This removes the global command only. It does **not** delete `~/.config/arxiv-daily/` (config, secrets) or your vault/output folder (for example `~/arxiv-daily`) — remove those yourself for a full cleanup.

## Develop from the monorepo

This package is built from the [arxiv-daily](https://github.com/tdccccc/arxiv-daily) workspace:

```bash
git clone https://github.com/tdccccc/arxiv-daily.git
cd arxiv-daily
npm ci
npm run build
npm run cli -- run --today    # or, after build: npx arxiv-daily run --today
```

The published tarball ships a single bundled binary (`dist/arxiv-daily-cli.cjs`).

## License

MIT — see [LICENSE](./LICENSE).
