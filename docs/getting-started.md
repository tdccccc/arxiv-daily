# Getting Started

This guide walks through the first successful arXiv Daily run in Obsidian.

## Before You Start

You need:

- Obsidian desktop.
- An LLM provider API key.
- One or more arXiv categories, such as `astro-ph`, `cs.LG`, or `hep-th`.
- A short description of the research topics you want to track.

arXiv Daily stores generated files in your vault. API keys are saved as plaintext in `<your-vault>/.obsidian/plugins/arxiv-daily/data.json` for compatibility; this is not keyring or encrypted storage, and vault sync/backup tools may copy it. After saving, Settings shows only **Configured**, with explicit **Replace** and **Clear** actions. Logs, diagnostics, and presented errors are redacted. Fetched source content is cached locally under the adjacent `.cache/` directory for the configured retention period (seven days by default); disable the plugin and delete that directory to clear it.

## 1. Open The Plugin Settings

After installing and enabling the plugin, open:

```text
Settings -> arXiv Daily
```

The top of the settings page has a **Getting Started** checklist. Use it as the first-run guide:

- **LLM API key, base URL, and model**
- **At least one arXiv category**
- **At least one complete research topic**
- **Ready to run**

Buttons in the checklist jump to the missing section.

## 2. Configure The LLM

Choose a provider, then enter and save the API key. A saved key is replaced by the **Configured** sentinel rather than rendered back into the page; use **Replace** or **Clear** when needed. The provider preset fills the base URL and model, but both remain editable.

For a first run, keep the default temperature, timeout, and reasoning settings unless your provider requires different values.

## 3. Choose arXiv Categories

Pick the arXiv categories you want to fetch. Multiple categories are allowed, and duplicate papers are merged by arXiv ID.

Examples:

- `astro-ph` for astrophysics.
- `astro-ph.CO` for cosmology.
- `cs.LG` for machine learning.

## 4. Add Research Topics

Each topic becomes one section in the daily report.

A topic needs:

- **Name**: readable section title.
- **Tag**: short Obsidian tag slug.
- **Description**: natural-language rule for what belongs in this topic.

Example:

```text
Name: Photometric Redshift
Tag: photo-z
Description: Methods, benchmarks, uncertainty calibration, catalog construction, and systematics for photometric redshift estimation.
```

Use a template if one matches your field, then edit the topics to match your actual work.

## 5. Run The First Daily Report

Open the **arXiv Daily Dashboard** from the ribbon icon or command palette.

Click **Run Today**. The plugin will:

1. Fetch recent arXiv papers for the configured categories.
2. Filter papers against your topics.
3. Summarize selected papers with the configured LLM.
4. Write a Markdown daily report.
5. Update the Dashboard index.

The generated report appears under:

```text
arxiv-daily/daily/YYYY-MM-DD.md
```

## 6. Use The Dashboard

The Dashboard is the normal entry point after setup.

- **Starred** shows papers you marked as important.
- **All** shows every indexed paper that is not ignored.
- Search is local and relevance-ranked across arXiv ID, title, authors, topics, categories, and structured summary fields. It recognizes exact modern arXiv IDs and tokenizes English technical terms and Chinese text; a non-empty search defaults to relevance, while an explicit starred/published/topic/title sort remains primary.
- **Similar Papers** (the **Find similar papers** row action) uses local BM25-style lexical retrieval over non-ignored Paper Index entries. It shows deterministic match reasons and uses no network, LLM, embedding, or database.
- The calendar opens daily reports by date.
- Row actions open or create a paper note, find similar papers, open the source daily report, open arXiv, open the PDF, or download the PDF. Similar-paper results can open the detail, daily report, arXiv page, or PDF.
- **Dashboard -> More -> Cancel active tasks** cooperatively cancels scheduled or manual daily runs, manual detail summaries, and PDF downloads. **Get Models** is excluded; an already-issued Obsidian `requestUrl` call may finish before later work stops.

For the intended Zotero workflow, open the arXiv page from the Dashboard and import the paper with the Zotero browser extension.

Generated daily reports and detail notes end with a folded **Generation metrics** callout. It shows total pipeline elapsed time when available, LLM elapsed time, logical calls, HTTP attempts, and provider-reported tokens. Missing or retry-incomplete usage is labeled unavailable/incomplete rather than zero, and no cost is estimated. Existing settings, Paper Index files, and Markdown remain usable; no Paper Index schema migration is required.

## 7. Enable Scheduling

If the first manual run works, return to **Settings -> arXiv Daily** and enable the scheduler.

The scheduler runs while Obsidian is open. Missed weekdays in the lookback window are retried later.

## Troubleshooting

If **Run Today** is disabled, finish the checklist in **Settings -> arXiv Daily**.

If the Dashboard says there are no indexed papers, run today or run pending dates first.

If a run fails, use **Dashboard -> More -> Show diagnostics** to inspect settings, date context, and recent run state.

If too many papers are selected, narrow your arXiv categories or make topic descriptions more specific.
