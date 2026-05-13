# arXiv Daily — Obsidian plugin

Native TypeScript Obsidian plugin that replaces `arxiv_daily.py` with an
in-vault settings GUI, catch-up scheduling, and on-demand manual runs.

## Features (v0.1.2)

- **Disabled by default** — fresh installs don't auto-summarize; opt-in via settings or ribbon
- **Skip existing files** — pipeline checks daily/paper files before any network or LLM call
- **Status bar progress** — live display of current stage and paper count during runs
- **Provider presets** — LLM settings with DeepSeek/OpenAI/Anthropic/GLM dropdowns; all fields remain editable
- **Latest models** — OpenAI GPT-5.2–5.5, Anthropic Claude Opus 4.7, GLM-5.1
- **arXiv category dropdown** — grouped by field (Physics/CS/Math/Stats), custom input supported
- **Simplified config** — auto-generated tag/display maps from detail categories; comma-separated input
- **Timezone dropdown** — common presets + custom input
- **English UI** — all settings in English
- Daily fetch from `https://arxiv.org/list/<category>/recent?show=2000`
  with a 5-day rolling lookback
- Atom-API abstract enrichment so the filter LLM sees full summaries
- LLM-based paper filtering, daily summary, and per-paper detail reports
  via any OpenAI-compatible endpoint
- Catch-up scheduler that runs while Obsidian is open; weekend skip
- Manual commands plus a ribbon icon with Enable/Disable toggle
- Per-machine run state persistence
- Cross-platform (Windows / macOS / Linux)

## Installation

### Option A — via BRAT (recommended)

1. Install the [BRAT](https://github.com/TfTHacker/obsidian42-brat) plugin.
2. Open BRAT settings → Add Beta plugin → paste `tdccccc/arxiv-daily`.
3. Enable **arXiv Daily** in Community plugins.
4. Open Settings → arXiv Daily, pick a provider and fill in the API Key.

### Option B — manual install

1. Download `manifest.json`, `main.js`, and `styles.css` from the
   [latest release](https://github.com/tdccccc/arxiv-daily/releases).
2. Drop them into `<vault>/.obsidian/plugins/arxiv-daily/`.
3. Enable the plugin in Obsidian.

### Option C — build from source

```bash
git clone https://github.com/tdccccc/arxiv-daily.git
cd arxiv-daily/plugin
npm install
npm run build
# then copy manifest.json + main.js + styles.css into the vault
```

## Settings overview

| Section | Fields |
|---|---|
| **Enable** | Toggle, shows Running / Paused status |
| **LLM** | Provider dropdown, API Key, Base URL, Model, Temperature, Timeout, Thinking mode, Reasoning effort |
| **arXiv** | Category dropdown (grouped), Research interests, Detail criteria, Detail categories (comma-separated), Timezone |
| **Output & Schedule** | Daily / Papers paths, Run time, Tick interval, Lookback days (≤ 5) |
| **Advanced** | Request delay, cache TTL, char limits, skip / priority sections, log level |

## Commands

| Command | Action |
|---|---|
| `arXiv Daily: Run now` | Pulls today, writes daily + papers |
| `arXiv Daily: Run for date…` | Pulls a specific date within the last 5 days |
| `arXiv Daily: Run all pending in lookback window` | Runs every pending date |
| `arXiv Daily: Summarize by arXiv ID…` | Summarize a single paper by ID |
| `arXiv Daily: Open today's daily report` | Opens `<dailyDir>/<today>.md` |
| `arXiv Daily: Show recent run state` | Lists last 20 dates and their statuses |

Ribbon icon opens a menu with: Status + Enable/Disable toggle, Run for today, Run all pending, Run for specific date, Summarize by ID.

## Scheduling

The plugin uses a single in-process tick loop (default every 20 minutes
while Obsidian is open). On each tick:

1. Walk back over the lookback window (today, yesterday, …, day-4).
2. Skip days that are `completed` or `failed_permanent`.
3. Skip today if the local clock is still before `runAtLocal`.
4. Skip weekends (Sat/Sun in configured timezone).
5. Skip dates where the daily file already exists.
6. Skip `failed_transient` days whose last attempt is within one tick interval.
7. Otherwise acquire the per-date RunLock, mark running, and run the pipeline.

On Enable: starts the interval and triggers a today-only run (bypasses
runAtLocal). On Disable: stops the interval. Manual commands always work
regardless of enabled state.

## Development

```bash
cd plugin
npm run dev      # watch build
npm test         # run unit + integration tests (vitest)
npm test:watch   # vitest watcher
```
