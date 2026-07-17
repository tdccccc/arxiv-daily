# arXiv Daily

> Turn daily arXiv feeds into a manageable, searchable reading list — right inside Obsidian.

[Getting Started](https://github.com/tdccccc/arxiv-daily/blob/main/docs/getting-started.md) · [中文说明](https://github.com/tdccccc/arxiv-daily/blob/main/docs/README.zh-CN.md)

Every day, hundreds of new papers appear on arXiv. arXiv Daily helps you stay on top of the ones that matter: it fetches papers from your chosen categories, uses an LLM to filter and summarize by your research topics, and writes structured Markdown reports into your vault — all on a configurable schedule.

The result is a compact daily reading list you can skim, star, and act on, without leaving Obsidian.

## Why arXiv Daily?

- **Save time**: let the LLM filter hundreds of papers down to the handful relevant to your topics, with structured summaries (core problem, method, result, relevance, limitations).
- **Stay organized**: daily reports, paper notes, and PDFs all live as plain Markdown in your vault — searchable, linkable, and future-proof.
- **Review across days**: the Dashboard gives you a calendar view, local relevance-ranked search, topic/date/status filters, and sorting to revisit papers across dates.
- **Works with your workflow**: star important papers, open arXiv or PDF links, create detailed notes for papers you want to dig into. When a paper is ready for your formal library, import it into Zotero as usual.
- **Set and forget**: configure once — categories, topics, LLM provider, schedule — and the plugin runs daily. Missed days are caught up automatically.

## Quick Start

1. Install and enable the plugin.
2. Open **Settings → arXiv Daily**.
3. Choose an LLM provider and enter an API key.
4. Select one or more arXiv categories (e.g. `cs.CL`, `astro-ph`, `stat.ML`).
5. Add research topics — natural-language descriptions of what you want to track.
6. Enable the scheduler, or open the Dashboard and click **Run Today**.

For a detailed walkthrough, see [Getting Started](https://github.com/tdccccc/arxiv-daily/blob/main/docs/getting-started.md).

## Dashboard

The Dashboard is the main entry point after setup.

- **Starred / All tabs**: star the papers that matter; unstarred papers stay neutral.
- **Calendar**: dates with reports are marked; today is highlighted; click any date to open its report.
- **Search & filters**: local relevance-ranked search covers arXiv ID, title, authors, topics, categories, and structured summary fields, with English technical-token and Chinese bigram tokenization. Exact modern arXiv IDs (including URL/version forms) are prioritized.
- **Sort**: a non-empty search defaults to relevance; choosing starred, published date, topic, or title keeps that explicit sort as the primary order.
- **Similar Papers**: local BM25-style lexical matches over non-ignored Paper Index entries, with deterministic match reasons and actions to open the detail, daily report, arXiv page, or PDF. It uses no network request, LLM, embedding, or database.
- **Paper actions**: from each row, open or create a paper note, open the daily report, open the arXiv page, open or download a PDF.
- **Batch operations**: run today, run pending lookback dates, or run a specific date. **Cancel active tasks** cooperatively cancels automatic or manual daily runs, manual detail summaries, and PDF downloads; **Get Models** is excluded. An Obsidian `requestUrl` call that was already issued may finish before cancellation takes effect, while later work is stopped.

## Daily Reports

Each daily report is a Markdown file. Selected papers include:

- Authors and arXiv link
- Source sections used for summarization
- Core problem, key method, main result
- Why it is relevant to your topics
- Limitations or boundaries
- Watch/highlight checkboxes

Highlighting a paper in the daily report maps to a Dashboard star.

Daily reports and generated detail notes end with a folded **Generation metrics** callout. It reports total pipeline elapsed time when available, LLM elapsed time, logical calls, HTTP attempts, and only token usage reported by the provider. Missing usage is shown as unavailable or incomplete rather than zero; retries make usage incomplete when failed-attempt usage is unavailable. No cost estimate is calculated.

Existing Markdown remains usable; adding this callout does not require rewriting older reports.

## Output Layout

Files are organized under `arxiv-daily/` in your vault:

```text
arxiv-daily/
  daily/
    2026-06-13.md
  papers/
    2606.12345.md
  pdfs/
    2606.12345.pdf
  .index/
    papers.json
    run-state.json
```

- `daily/YYYY-MM-DD.md` — daily discovery report grouped by topic
- `papers/<arxiv_id>.md` — detailed paper notes
- `pdfs/<arxiv_id>.pdf` — downloaded PDFs
- `.index/papers.json` — local paper index (read by the Dashboard); search and Similar Papers build a derived in-memory index without changing its schema
- `.index/run-state.json` — scheduler run state

Existing settings, Paper Index files, and Markdown reports remain usable; no Paper Index schema migration is required for these features.

## Installation

arXiv Daily is desktop-only.

### Community Plugins

1. Open **Settings → Community plugins → Browse**.
2. Search for **arXiv Daily**.
3. Install and enable it.

### BRAT (Beta)

1. Install [BRAT](https://github.com/TfTHacker/obsidian42-brat).
2. Open **BRAT settings → Add Beta plugin**.
3. Enter: `tdccccc/arxiv-daily`

### Manual Install

Download `manifest.json`, `main.js`, and `styles.css` from the [latest release](https://github.com/tdccccc/arxiv-daily/releases/latest). In your vault, create the hidden plugin directory if needed and place all three files directly in it:

```text
<your-vault>/.obsidian/plugins/arxiv-daily/
  manifest.json
  main.js
  styles.css
```

Do not leave the files inside a nested release or repository folder. Restart Obsidian, then enable **arXiv Daily** under **Settings → Community plugins**.

## Commands

| Action | Where |
|---|---|
| Open Dashboard | Ribbon icon or command palette |
| Run today | Dashboard toolbar |
| Run pending lookback dates | Dashboard toolbar |
| Run a specific date | Dashboard **More** menu or command palette |
| Summarize by arXiv ID | Dashboard **More** menu or command palette |
| Cancel active tasks | Dashboard **More** menu or command palette |
| Find similar papers | Paper-row **Find similar papers** action |
| Open a daily report | Dashboard calendar |
| Star a paper | Dashboard star button or daily report highlight checkbox |

## Network & Privacy

- Connects to `arxiv.org` and `export.arxiv.org` to fetch listings, abstracts, and PDFs.
- Connects to your configured LLM provider. Sent content includes paper titles, abstracts, and selected text snippets needed for filtering and summarization.
- A saved API key is displayed only as **Configured** in Settings; use explicit **Replace** or **Clear** actions to change it. The key remains plaintext in `<your-vault>/.obsidian/plugins/arxiv-daily/data.json` for compatibility—Obsidian Sync or another vault backup may copy that file. There is no keyring or encryption claim. Restrict access to the vault and its backups, and use **Clear** to remove the saved key. Logs, diagnostics, and presented errors are redacted.
- Fetched arXiv HTML/source content is cached in `<your-vault>/.obsidian/plugins/arxiv-daily/.cache/` for the configured retention period (seven days by default). Delete that directory while the plugin is disabled to clear it; it will be recreated as needed.
- The CLI reads its key from `ARXIV_DAILY_API_KEY` or a user-supplied config file and caches fetched content in `.arxiv-daily/cache/` relative to the working directory unless `--cache-dir` or `ARXIV_DAILY_CACHE_DIR` overrides it. Protect or delete those local files according to your environment.
- No client-side telemetry. Generated reports, paper notes, PDFs, indexes, and run state are written under `arxiv-daily/` in the vault by default; configured output paths may change that location.

## CLI Usage

The Node CLI is available for cron or server workflows and requires Node.js 20.11.0 or newer.

```bash
npm ci
npm run build

ARXIV_DAILY_API_KEY=sk-... npm run cli -- run-pending --vault-root /path/to/vault
```

The canonical executable is `apps/cli/dist/arxiv-daily-cli.cjs`. The build also
refreshes `plugin/arxiv-daily-cli.cjs` and keeps `arxiv_daily.py` as a deprecated
compatibility shim.

With a config file:

```bash
npm run cli -- run --date 2026-06-13 --config arxiv-daily.config.json --vault-root /path/to/vault
npm run cli -- summarize --id 2606.12345 --config arxiv-daily.config.json --vault-root /path/to/vault
```

## Development

```bash
npm ci
npm run check:boundaries
npm run typecheck
npm test
npm run build
```

For a release, synchronize every workspace package, internal dependency spec,
Obsidian manifest/version map, and lockfile before validating the release:

```bash
npm run sync:release-version -- 0.3.0
npm run check:release-version -- 0.3.0
```

Review the generated diff before committing. The sync command only updates
version metadata; it does not publish, commit, tag, or push.

This repository is one npm workspace. `packages/core` is the only business
core, `packages/node-runtime` contains Node adapters, `apps/cli` is the one-shot
CLI, and `plugin` contains only the Obsidian host and UI. There is intentionally
no protocol or daemon layer.

## License

[MIT](./LICENSE). Bundled dependency notices are in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
