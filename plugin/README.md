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
- **Config validation** — scheduled and manual runs are gated on valid settings; API keys are masked in the settings page (a show/hide button reveals them) and kept in the plugin's `data.json` in your vault's plugin folder

Reports land in `arxiv-daily/daily/YYYY-MM-DD.md`; paper notes in `arxiv-daily/papers/<arxiv_id>.md`.

## Install

Desktop only — the plugin does not run on Obsidian mobile. Requires Obsidian 1.5.7 or newer.

1. **Community plugins** — Settings → Community plugins → Browse → search **arXiv Daily**
2. **BRAT** — install [BRAT](https://github.com/TfTHacker/obsidian42-brat), then add `tdccccc/arxiv-daily`
3. **Manual** — download `manifest.json`, `main.js`, `styles.css` from the [latest release](https://github.com/tdccccc/arxiv-daily/releases/latest) and place them in `<vault>/.obsidian/plugins/arxiv-daily/`

Enable the plugin, then open **Settings → arXiv Daily**.

## Quick start

A five-step in-app guide (Connect AI → choose paper sources → describe research interests → generate your first report → turn on daily reports) walks you through first setup. For the full walkthrough, including email and the personal library, see [Getting Started](../docs/getting-started.md) (中文：[新手教程](../docs/getting-started.zh-CN.md)).

## Usage

### Dashboard

Open it from the ribbon icon or the **Open reading dashboard** command. It's the usual home after setup:

- **Starred** / **All** tabs, plus search and filters (topic, date range, papers with a detail summary)
- **Calendar** — dates with daily reports; dates within the lookback window that are runnable are highlighted, click to generate
- Row actions — star, open or generate a paper note, find similar papers, open the daily report, open on arXiv, open or download the PDF
- **Similar Papers** — deterministic local matches for a paper, with match reasons and resource links

An open Dashboard refreshes itself when a background run finishes.

### Commands

| Command | Action |
|---|---|
| `Run today` | Pulls today's papers, writes the daily report, and writes any selected paper notes |
| `Run for date…` | Runs a specific date within the last 5 days |
| `Run all pending in lookback window` | Runs every pending date |
| `Retry failed dates in lookback window` | Clears failed state for recent dates and reruns them |
| `Force run for date…` | Clears stored state for one date and reruns it, bypassing schedule guards |
| `Clear run state…` | Clears persisted run state without deleting notes |
| `Cancel active tasks` | Cancels active runs, paper-note summaries, PDF downloads, and personal library scans, indexing and direction generation |
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

Personal library adds its own commands (review/accept proposed directions, preview/scan/reload the catalog, index and search full text, clean up old search indexes, diagnose the full-text runtime) — see [Getting Started](../docs/getting-started.md).

Manual runs are gated on config validity: `Run today`, `Run all pending`, and `Run for date…` need a valid LLM config and at least one topic; `Summarize by paper ID…` needs a valid LLM config only.

### Marking papers

Star a paper with the star button in the Dashboard (or **Star** / **Unstar** on selected rows). For finer marks use `Set paper mark…` (Unmarked / Watch / Highlight / Saved / Ignored), or `Mark current paper as <mark>` on an open paper note. Marks are stored in the paper index, not in the Markdown, so daily reports have no checkboxes to tick; older reports may still carry tick boxes, and ticking them changes nothing.

### Scheduling

A single tick loop (every 20 minutes by default) runs **only while Obsidian is open**: it walks the lookback window, skips completed/skipped/weekend/not-yet-due dates, and runs the rest. If today's report is still open, enabling the schedule asks whether to run today, skip today, or cancel; if today already has a result, it simply turns on.

For unattended runs on a machine that stays online, use the [CLI](../apps/cli/README.md) with cron instead.

## Settings overview

| Section | Covers |
|---|---|
| **Enable** | Turns the in-app scheduler on or off |
| **LLM** | API base URL, API key, Model, Reasoning effort |
| **arXiv categories** | Which arXiv categories to fetch |
| **Research topics** | Topic cards (name, tag, description, directions, detail report) and templates |
| **Automatic detail notes** | How often a longer paper note is written automatically (Fewer / Recommended / More) |
| **Timezone** | Which timezone defines the current day for reports and schedules |
| **Output & schedule** | Daily paper limit, Daily reports folder, Paper notes folder, Link style, Summary language, Run window, Check every (minutes) |
| **Personal library** | Library (choose a folder, index it, revoke authorization), Topics from library (review proposed directions), Embedding (local or remote), Better PDF parser (optional local sidecar) |
| **Email delivery** | How to send, Your email, Verification code, Resend API key, From email, From name, Send test email, Daily auto-send |
| **Advanced** | Log level |
| **Help & feedback** | Prefilled bug-report and feature-request issues, documentation, repository |

The lookback window is fixed at 5 days and is not a setting.

## Email and personal library

Email is optional and never blocks the daily report on failure. Personal library access is read-only and scoped to the folder you pick; the library can propose topics and directions, but only ones you explicitly accept affect filtering. Both are covered in detail in [Getting Started](../docs/getting-started.md).

## Troubleshooting

See the [Getting Started troubleshooting table](../docs/getting-started.md#troubleshooting) for common issues (disabled run button, no papers in the Dashboard, failed runs, email errors, an unreadable `data.json`).

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
