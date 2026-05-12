# arXiv Daily — Obsidian plugin

Native TypeScript Obsidian plugin that replaces `arxiv_daily.py` with an
in-vault settings GUI, catch-up scheduling, and on-demand manual runs.

## Features (v1 MVP)

- Daily fetch from `https://arxiv.org/list/<category>/recent?show=2000`
  with a 5-day rolling lookback (the upper bound `/recent` exposes)
- Atom-API abstract enrichment so the filter LLM sees full summaries,
  not just titles
- LLM-based paper filtering, daily summary, and per-paper detail reports
  via any OpenAI-compatible endpoint (default: DeepSeek V4)
- Catch-up scheduler that runs while Obsidian is open; backfills missed
  days within the lookback window
- Manual `Run now` and `Run for date…` commands plus a ribbon icon
- `Show recent run state` command for a quick status table
- Per-machine run state persistence; does not sync via the vault
- Cross-platform (Windows / macOS / Linux)

## Installation

### Option A — via BRAT (recommended for early adopters)

1. Install the [BRAT](https://github.com/TfTHacker/obsidian42-brat) plugin
   from Obsidian's Community Plugins.
2. Open BRAT settings → **Add Beta plugin** → paste
   `tdccccc/arxiv-daily`.
3. BRAT installs the latest release and offers updates as new versions ship.
4. Enable **arXiv Daily** in Obsidian → Settings → Community plugins.
5. Open Settings → arXiv Daily, fill in the API Key. The default endpoint
   and model are already DeepSeek-friendly
   (`https://api.deepseek.com/v1`, `deepseek-v4-pro`).

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
# then copy manifest.json + main.js + styles.css into the vault as in Option B
```

## Settings overview

| Section | Fields |
|---|---|
| **LLM** | API key, base URL, model, temperature, timeout, thinking mode, reasoning effort |
| **arXiv** | category (`astro-ph` default), research interests, detail criteria, semantic-category map, timezone |
| **Output** | daily and papers directories (vault-relative) |
| **Schedule** | enabled, daily run time `HH:MM`, tick interval, lookback days (≤ 5) |
| **Advanced** | request delay, cache TTL, char limits, skip / priority sections, log level |

## Commands

| Command | Action |
|---|---|
| `arXiv Daily: Run now` | Pulls today, writes daily + papers |
| `arXiv Daily: Run for date…` | Pulls a specific date within the last 5 days |
| `arXiv Daily: Open today's daily report` | Opens `<dailyDir>/<today>.md` |
| `arXiv Daily: Show recent run state` | Lists last 20 dates and their statuses |

## How scheduling works

The plugin uses a single in-process tick loop (default every 20 minutes
while Obsidian is open). On each tick:

1. Walk back over the lookback window (today, yesterday, …, day-4).
2. Skip days that are `completed` or `failed_permanent`.
3. Skip today if the local clock is still before `runAtLocal`.
4. Skip `failed_transient` days whose last attempt is within one tick
   interval (avoids hammering on rate-limited LLMs).
5. Otherwise acquire the per-date `RunLock`, mark the day `running`,
   and run the pipeline (fetch /recent → enrich abstracts → filter →
   fetch content → summarize → write).

This means Obsidian must be open for at least one tick after `runAtLocal`
on each desired day (or any day in the lookback window). Manual
triggers bypass the time gate.

## Development

```bash
cd plugin
npm run dev      # watch build (writes main.js incrementally)
npm test         # run unit + integration tests (vitest)
npm test:watch   # vitest watcher
```

Tests cover the pure modules (parser, extractor, state, lock, retry,
time, scheduler, atom-parser, html-cache) plus an integration test
for the pipeline against a real arXiv fixture in `tests/fixtures/`.
Obsidian-bound surfaces (settings tab, commands, ribbon) are verified
by the manual smoke checklist in
`docs/superpowers/plans/2026-05-11-obsidian-plugin-mvp.md` (Task 24).

## v2 roadmap

- Multi-profile (multiple research directions in parallel)
- OS-level cron fallback (CLI entrypoint)
- Per-profile LLM overrides
- Optional vault-synced run state for multi-machine setups
