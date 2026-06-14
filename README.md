# arXiv Daily

> Daily arXiv discovery for Obsidian, with LLM filtering, structured paper summaries, and a Dashboard for reviewing papers across dates.

[Getting Started](./docs/getting-started.md) · [中文说明](./docs/README.zh-CN.md)

arXiv Daily helps you turn the daily arXiv feed into a small, reviewable reading list inside your vault. It fetches new papers, filters them against your research topics, writes Markdown daily reports, and keeps a local paper index that powers the **arXiv Daily Dashboard**.

## Core Workflow

1. Configure arXiv categories, research topics, and an LLM provider.
2. Let the plugin run on schedule, or run it manually from the Dashboard.
3. Read the daily Markdown report and star important papers.
4. Use the Dashboard to search, filter, revisit daily reports, open paper notes, and open arXiv/PDF links.
5. When a paper should enter your formal library, open its arXiv page and import it with Zotero.

arXiv Daily is for discovery and review. Zotero remains the source of truth for citation keys, BibTeX, and long-term bibliography management.

## Highlights

- **Dashboard-first reading flow**: the ribbon icon opens the arXiv Daily Dashboard directly.
- **Starred / All review model**: star only the papers that matter; unstarred papers stay neutral.
- **Daily report calendar**: dates with reports are marked, today is highlighted, and clicking a date opens that report.
- **Search and filters**: filter by keyword, topic, first-seen date range, note presence, and detail availability.
- **Focused paper actions**: open or create a paper note, open the source daily report, open arXiv, open a PDF, or download the PDF.
- **Markdown-native output**: daily reports and paper notes stay as ordinary Markdown files in your vault.
- **Catch-up scheduling**: missed weekdays in the lookback window are retried when Obsidian is open.
- **Shared core and CLI**: the Obsidian plugin and Node CLI use the same pipeline.

## Dashboard

The Dashboard is the main entry point after setup.

- **Starred** shows high-priority papers you marked for follow-up.
- **All** shows every non-ignored indexed paper.
- The left side contains search, topic/date/note/detail filters, and summary counts.
- The calendar opens historical daily reports without browsing folders.
- Paper rows keep the reading actions close to the title, authors, and summary.

The Dashboard reads `arxiv-daily/.index/papers.json`. It does not replace the Markdown editor; paper notes and daily reports still open in Obsidian's native editor.

## Output Layout

By default, generated files live under `arxiv-daily/` in your vault:

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

- `daily/YYYY-MM-DD.md`: daily discovery report grouped by topic.
- `papers/<arxiv_id>.md`: detailed notes for detail papers or manually created notes.
- `pdfs/<arxiv_id>.pdf`: manually downloaded arXiv PDFs.
- `.index/papers.json`: local paper index used by the Dashboard.
- `.index/run-state.json`: scheduler and CLI run state.

## Install

arXiv Daily is desktop-only.

### Community Plugins

After the plugin is approved in the Obsidian Community directory:

1. Open **Settings -> Community plugins -> Browse**.
2. Search for **arXiv Daily**.
3. Install and enable it.

### BRAT Beta

Before the community listing is fully available, install with [BRAT](https://github.com/TfTHacker/obsidian42-brat):

1. Install and enable BRAT.
2. Open **BRAT settings -> Add Beta plugin**.
3. Enter:

```text
tdccccc/arxiv-daily
```

### Manual Install

Download `manifest.json`, `main.js`, and `styles.css` from the latest release:

https://github.com/tdccccc/arxiv-daily/releases/latest

Place them in:

```text
<vault>/.obsidian/plugins/arxiv-daily/
```

Then restart Obsidian and enable **arXiv Daily**.

## Quick Start

For a first-run walkthrough, see [Getting Started](./docs/getting-started.md).

1. Open **Settings -> arXiv Daily**.
2. Choose an LLM provider and enter an API key.
3. Select one or more arXiv categories.
4. Add at least one research topic.
5. Enable the scheduler, or open the Dashboard and click **Run Today**.

Research topics are natural-language descriptions of what you want to track. The LLM uses them to decide which papers are relevant and how they should be grouped.

## Daily Reports

Daily reports are Markdown files. Each selected paper includes:

- authors and arXiv link
- source sections used for summarization
- core problem
- key method
- main result
- why it is relevant
- limitations or boundaries
- watch/highlight checkboxes for Markdown-first triage

Highlighting a paper in the daily report maps to a Dashboard star.

## Commands

Most day-to-day actions are available from the Dashboard toolbar.

| Action | Where |
|---|---|
| Open Dashboard | Ribbon icon or command palette |
| Run today | Dashboard toolbar |
| Run pending lookback dates | Dashboard toolbar |
| Run a specific date | Dashboard **More** menu or command palette |
| Summarize by arXiv ID | Dashboard **More** menu or command palette |
| Open a daily report | Dashboard calendar |
| Star a paper | Dashboard row star button or daily report highlight checkbox |

## Network And Privacy

arXiv Daily uses network access only for the services needed to fetch and summarize papers.

- It connects to `arxiv.org` and `export.arxiv.org` to fetch paper listings, abstracts, HTML pages, and manually downloaded PDFs.
- It connects to the LLM provider endpoint you configure in settings. Sent content can include paper titles, authors, abstracts, and selected paper text snippets needed for filtering and summarization.
- API keys are stored in Obsidian plugin settings. Diagnostic output does not include API keys.
- The plugin does not include client-side telemetry.
- The plugin does not send vault content to services other than arXiv and the configured LLM provider for the requested workflow.
- By default, generated files are written only under `arxiv-daily/` inside the vault.

## CLI Usage

The Node CLI is available for cron or server workflows. It is secondary to the Obsidian plugin and intentionally kept minimal in this README.

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

The legacy root `arxiv_daily.py` is only a compatibility shim that forwards to the Node CLI.

## Development

Implementation details and developer notes live in [plugin/README.md](./plugin/README.md).

```bash
cd plugin
npm install
npm test
npm run build
```

## License

[MIT](./LICENSE)
