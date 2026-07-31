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
npx arxiv-daily@latest help
```

Package page: https://www.npmjs.com/package/arxiv-daily

## Quick start

```bash
arxiv-daily init
# guided TUI: vault, LLM, optional email, categories, topic, …
arxiv-daily run --today
```

`init` is a TUI wizard:

- **↑/↓** move · **Space** multi-select · **Enter** confirm (keep default when shown)
- **← Go back** (last item in menus) → previous step  
- **Ctrl+C** (or **Esc**) → **exit** the wizard (not “back”)  
- Defaults appear in prompts — press **Enter** to accept without retyping  

Flow: vault → provider → URL → API key → optional model fetch → model → email →
categories → timezone → language → topic → optional paper-note / schedule flags.
`link_style` and `log_level` are not asked (fixed defaults: wikilink / info).
Config comments are English; **`schema_version` is at the bottom — leave it alone**.

Config path:

- Linux/macOS: `$XDG_CONFIG_HOME/arxiv-daily/config.toml` (default `~/.config/arxiv-daily/config.toml`)
- Windows: `%APPDATA%\arxiv-daily\config.toml`

Default vault from init: `~/arxiv-daily`. No settings env vars; no `--config` / `--vault-root` flags.

## Uninstall

```bash
npm uninstall -g arxiv-daily
```

This removes the global command only. It does **not** delete:

- `~/.config/arxiv-daily/` (config, secrets)
- your vault / output folder (for example `~/arxiv-daily`)

Remove those yourself if you want a full cleanup.

## Commands

```text
arxiv-daily init
arxiv-daily update [--check] [--yes]
arxiv-daily run --today
arxiv-daily run --date YYYY-MM-DD
arxiv-daily run --id ARXIV_ID [--date YYYY-MM-DD]
arxiv-daily email test|status|verify-start
arxiv-daily schedule show|install|uninstall
arxiv-daily data export --out PATH.zip
arxiv-daily data import PATH.zip [--yes]
arxiv-daily help
```

- **`update`** — check npm for a newer `arxiv-daily` and optionally `npm install -g` it. Config is not touched. `--check` only reports; `--yes` skips the confirm prompt.
- **`run --today`** — one day only (typical cron entry). Missed days: `run --date …`.
- **`schedule install`** — writes managed user crontab lines (Linux/macOS/WSL). Not supported on native Windows Task Scheduler; use WSL or the Obsidian plugin for desktop scheduling.

## Interrupted daily runs and checkpoints

During a daily run, arXiv Daily stores each completed per-paper **structured summary** as an internal checkpoint before starting the next paper. If the process is cancelled or crashes, rerunning the same date can reuse compatible work instead of paying for the same LLM calls again. These files are internal **Vault data**, not partial daily reports or paper notes.

Checkpoints live under the active output layout at `<output-root>/.index/daily-summary-checkpoints/YYYY-MM-DD.json` (with an internal `.bak` recovery file). With the default configuration this is `arxiv-daily/.index/daily-summary-checkpoints/`. The backup retains the last valid primary across successful replacements and is removed with the date checkpoint after report commit or explicit cleanup; it is not an unbounded history. The output root is derived from the configured `daily_dir` and `papers_dir`; do not manually construct, move, merge, or edit checkpoint JSON. `data export` includes the active `.index/**` tree, including nested checkpoints. `data import` restores `.index` only when the archive and active canonical output layouts match. Across different layouts it still imports daily and paper files but warns and skips `.index`; internal index/checkpoint data is never silently relocated.

A checkpoint is reused only when the paper source and all effective summary-generation inputs still match, including language, provider endpoint, model, reasoning settings, and prompt/result contract versions. A validated structured result can be reused. A validation-exhausted typed fallback is reused only on the same exact compatibility fingerprint; a transport-exhausted fallback is retried on resume. Corrupt or incompatible entries are ignored safely and generation falls back to fresh work.

The complete daily report remains authoritative and is written once through the normal atomic commit path. Existing report/index recovery takes precedence over stale checkpoints; checkpoints never need to be concatenated into Markdown. After a successful report commit they are cleaned up automatically. To force recomputation after an interrupted run, it is safe to delete that date's checkpoint JSON and backup, or the `daily-summary-checkpoints` directory while no arXiv Daily process is running. This does not delete a committed daily report.

Treat checkpoint files, their backups, and `data export` archives as sensitive Vault data: they may contain paper inputs (titles, authors, abstracts, extracted sections) and model-generated results. Endpoint identity is stored only as a digest and API keys are excluded, but that does not make the files safe to publish. Protect archives like the Vault, and delete copies according to your own retention policy. Export/import intentionally rejects symlinked Vault roots, data roots, intermediate components, and targets. These checks reduce accidental path escape but Node filesystem APIs cannot eliminate TOCTOU against a concurrent hostile local process; only import into a trusted, quiescent Vault.

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
