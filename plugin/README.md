# arXiv Daily — Obsidian plugin

English · [简体中文](README.zh-CN.md)

Follow research directions, filter new arXiv papers, and keep daily reports and paper notes as local Markdown — right inside your vault.

Part of the [arXiv Daily](../README.md) project. The same discovery and reading workflow is also available without Obsidian via the [CLI and browser workbench](../apps/cli/README.md).

## Features

- **Filters the flood** — keeps only papers matching the directions inside your research topics
- **Daily report** — up to 20 papers a day by default, grouped by topic, each with a short structured summary
- **Paper notes** — longer per-paper notes on demand, automatic for a configurable share of eligible papers, or by arXiv ID
- **Dashboard** — calendar of daily reports, Starred/All tabs, local relevance search, filters, and row actions (open note, daily report, arXiv page, PDF)
- **Similar Papers** — local, offline lexical matching against your indexed papers; no network or LLM call at query time
- **Personal library (optional)** — index your own PDFs; review and explicitly accept proposed topics/directions before they affect filtering
- **Scheduling** — an in-process tick loop runs while Obsidian is open; use the [CLI](../apps/cli/README.md) with cron for unattended runs
- **Optional email** — a short digest after a successful day, via your own Resend key or an official-delivery beta
- **Config validation & secret-safe storage** — scheduled and manual runs are gated on valid settings; a saved API key only ever displays as `Configured`

Reports land in `arxiv-daily/daily/YYYY-MM-DD.md`; paper notes in `arxiv-daily/papers/<arxiv_id>.md`.

## Install

Desktop only (Obsidian does not support community plugins on mobile).

1. **Community plugins** — Settings → Community plugins → Browse → search **arXiv Daily**
2. **BRAT** — install [BRAT](https://github.com/TfTHacker/obsidian42-brat), then add `tdccccc/arxiv-daily`
3. **Manual** — download `manifest.json`, `main.js`, `styles.css` from the [latest release](https://github.com/tdccccc/arxiv-daily/releases/latest) and place them in `<vault>/.obsidian/plugins/arxiv-daily/`

Enable the plugin, then open **Settings → arXiv Daily**.

## Quick start

A five-step in-app guide (Connect AI → choose paper sources → describe research interests → generate your first report → turn on daily reports) walks you through first setup. For the full walkthrough, including email and the personal library, see [Getting Started](../docs/getting-started.md) (中文：[新手教程](../docs/getting-started.zh-CN.md)).

## Usage

### Dashboard

Open it from the ribbon icon or the **Open reading dashboard** command. It's the usual home after setup:

- **Starred** / **All** tabs, plus search and filters (topic, date range, note presence, detail availability)
- **Calendar** — dates with daily reports; dates within the lookback window that are runnable are highlighted, click to generate
- Row actions — open/create a paper note, open the daily report, open on arXiv, open or download the PDF
- **Similar Papers** — deterministic local matches for a paper, with match reasons and resource links

### Commands

| Command | Action |
|---|---|
| `Run today` | Pulls today's papers, writes the daily report, and writes any selected paper notes |
| `Run for date…` | Runs a specific date within the last 5 days |
| `Run all pending in lookback window` | Runs every pending date |
| `Retry failed dates in lookback window` | Clears failed state for recent dates and reruns them |
| `Force run for date…` | Clears stored state for one date and reruns it, bypassing schedule guards |
| `Clear run state…` | Clears persisted run state without deleting notes |
| `Cancel active tasks` | Cancels active runs, manual paper-note summaries, and PDF downloads |
| `Summarize by paper ID…` | Writes a paper note for one arXiv ID |
| `Set paper mark…` | Sets a paper's mark to Unmarked / Watch / Highlight / Saved / Ignored |
| `Create paper note…` | Creates a lightweight note for an indexed paper |
| `Mark current paper as <mark>` | Sets the open paper note's mark |
| `Open today's daily report` | Opens today's report file |
| `Open reading dashboard` | Opens the Dashboard view |
| `Send test email` | Sends a test digest with the current email settings |
| `Show recent run state` | Lists the last 20 dates and their statuses |
| `Show run history` | Shows the full run history |
| `Show diagnostics` | Shows a copyable local diagnostic report (no API key) |

Personal library adds its own commands (review/accept proposed directions, preview/scan/reload the catalog, index and search full text, diagnose the full-text runtime) — see [Getting Started](../docs/getting-started.md).

Manual runs are gated on config validity: `Run today`, `Run all pending`, and `Run for date…` need a valid LLM config and at least one topic; `Summarize by paper ID…` needs a valid LLM config only.

### Daily selections

Each paper in a daily report has two checkboxes you can tick directly in the Markdown:

```markdown
- [ ] 关注 <!-- arxiv-daily:2606.12345:watch -->
- [ ] 重点 <!-- arxiv-daily:2606.12345:highlight -->
```

Checking them syncs a `Watch`/`Highlight`-equivalent priority back to the paper index; the Dashboard's star button reflects the same state. Daily report files stay the primary place to triage.

### Scheduling

A single tick loop (every 20 minutes by default) runs **only while Obsidian is open**: it walks the lookback window, skips completed/skipped/weekend/not-yet-due dates, and runs the rest. Enabling the schedule asks whether to run today, skip today, or cancel.

For unattended runs on a machine that stays online, use the [CLI](../apps/cli/README.md) with cron instead.

## Settings overview

| Section | Covers |
|---|---|
| **LLM** | Provider, base URL, API key, model, reasoning effort |
| **arXiv categories** | Which arXiv categories to fetch |
| **Research topics** | Topic cards (name, directions, detail-note eligibility), templates |
| **Output & schedule** | Daily/paper note paths, link style, run time, tick interval, lookback days (≤ 5) |
| **Personal library** | Optional local PDF folder, eligible file types, processing depth, proposed directions |
| **Email delivery** | Send-yourself (Resend key) or official-delivery beta, daily auto-send |
| **Advanced** | Request delay, cache retention, section/char limits, log level |
| **Help & feedback** | Links to docs and a prefilled GitHub issue |

## Email and personal library

Email is optional and never blocks the daily report on failure. Personal library access is read-only and scoped to the folder you pick; the library can propose topics and directions, but only ones you explicitly accept affect filtering. Both are covered in detail in [Getting Started](../docs/getting-started.md).

## Troubleshooting

See the [Getting Started troubleshooting table](../docs/getting-started.md#troubleshooting) for common issues (disabled run button, no papers in the Dashboard, failed runs, email errors).

A rerun after an interruption resumes from the last completed step instead of repeating paid model calls.

## Development

From the repository root:

```bash
npm ci
npm run build          # builds all workspaces, including this plugin
npm test
```

Building just the plugin: `cd plugin && npm run build` (esbuild; outputs `main.js` next to `manifest.json` and `styles.css`). Copy those three files into a vault's `.obsidian/plugins/arxiv-daily/` to test a local build.

## License

MIT — see [LICENSE](../LICENSE).
