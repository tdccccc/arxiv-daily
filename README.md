# arXiv Daily

> Turn daily arXiv feeds into a manageable, searchable reading list — right inside Obsidian.

[Getting Started](./docs/getting-started.md) · [中文说明](./docs/README.zh-CN.md)

Every day, hundreds of new papers appear on arXiv. arXiv Daily helps you stay on top of the ones that matter: it fetches papers from your chosen categories, uses an LLM to filter and summarize by your research topics, and writes structured Markdown reports into your vault — all on a configurable schedule.

The result is a compact daily reading list you can skim, star, and act on, without leaving Obsidian.

## Why arXiv Daily?

- **Save time**: let the LLM filter hundreds of papers down to the handful relevant to your topics, with structured summaries (core problem, method, result, relevance, limitations).
- **Stay organized**: daily reports, paper notes, and PDFs all live as plain Markdown in your vault — searchable, linkable, and future-proof.
- **Review across days**: the Dashboard gives you a calendar view, full-text search, topic/date/status filters, and sorting to revisit papers across dates.
- **Works with your workflow**: star important papers, open arXiv or PDF links, create detailed notes for papers you want to dig into. When a paper is ready for your formal library, import it into Zotero as usual.
- **Set and forget**: configure once — categories, topics, LLM provider, schedule — and the plugin runs daily. Missed days are caught up automatically.

## Quick Start

1. Install and enable the plugin.
2. Open **Settings → arXiv Daily**.
3. Choose an LLM provider and enter an API key.
4. Select one or more arXiv categories (e.g. `cs.CL`, `astro-ph`, `stat.ML`).
5. Add research topics — natural-language descriptions of what you want to track.
6. Enable the scheduler, or open the Dashboard and click **Run Today**.

For a detailed walkthrough, see [Getting Started](./docs/getting-started.md).

## Dashboard

The Dashboard is the main entry point after setup.

- **Starred / All tabs**: star the papers that matter; unstarred papers stay neutral.
- **Calendar**: dates with reports are marked; today is highlighted; click any date to open its report.
- **Search & filters**: filter by keyword, topic, date range, note presence, or detail availability.
- **Sort**: by starred, recently seen, published date, topic, or title.
- **Paper actions**: from each row, open or create a paper note, open the daily report, open the arXiv page, open or download a PDF.
- **Batch operations**: run today, run pending lookback dates, or run a specific date.

## Daily Reports

Each daily report is a Markdown file. Selected papers include:

- Authors and arXiv link
- Source sections used for summarization
- Core problem, key method, main result
- Why it is relevant to your topics
- Limitations or boundaries
- Watch/highlight checkboxes

Highlighting a paper in the daily report maps to a Dashboard star.

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
- `.index/papers.json` — local paper index (read by the Dashboard)
- `.index/run-state.json` — scheduler run state

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

Download `manifest.json`, `main.js`, and `styles.css` from the [latest release](https://github.com/tdccccc/arxiv-daily/releases/latest) and place them in:

```text
<vault>/.obsidian/plugins/arxiv-daily/
```

Then restart Obsidian and enable **arXiv Daily**.

## Commands

| Action | Where |
|---|---|
| Open Dashboard | Ribbon icon or command palette |
| Run today | Dashboard toolbar |
| Run pending lookback dates | Dashboard toolbar |
| Run a specific date | Dashboard **More** menu or command palette |
| Summarize by arXiv ID | Dashboard **More** menu or command palette |
| Open a daily report | Dashboard calendar |
| Star a paper | Dashboard star button or daily report highlight checkbox |

## Network & Privacy

- Connects to `arxiv.org` and `export.arxiv.org` to fetch listings, abstracts, and PDFs.
- Connects to your configured LLM provider. Sent content includes paper titles, abstracts, and selected text snippets needed for filtering and summarization.
- API keys are stored in Obsidian plugin settings. They are never included in logs.
- No client-side telemetry.
- Generated files are written only under `arxiv-daily/` in your vault.

## CLI Usage

The Node CLI is available for cron or server workflows.

```bash
cd plugin
npm install
npm run build

ARXIV_DAILY_API_KEY=sk-... npm run cli -- run-pending --vault-root /path/to/vault
```

With a config file:

```bash
npm run cli -- run --date 2026-06-13 --config arxiv-daily.config.json --vault-root /path/to/vault
npm run cli -- summarize --id 2606.12345 --config arxiv-daily.config.json --vault-root /path/to/vault
```

## Development

```bash
cd plugin
npm install
npm test
npm run build
```

## License

[MIT](./LICENSE)
