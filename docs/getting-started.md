# Getting Started

This guide walks through the first successful arXiv Daily run in Obsidian.

## Before You Start

You need:

- Obsidian desktop.
- An LLM provider API key.
- One or more arXiv categories, such as `astro-ph`, `cs.LG`, or `hep-th`.
- A short description of the research topics you want to track.

arXiv Daily stores generated files in your vault. API keys are saved in Obsidian plugin settings.

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

Choose a provider, then enter the API key. The provider preset fills the base URL and model, but both remain editable.

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
- Search and filters narrow the list by keyword, topic, date, note, or detail status.
- The calendar opens daily reports by date.
- Row actions open or create a paper note, open the source daily report, open arXiv, open the PDF, or download the PDF.

For the intended Zotero workflow, open the arXiv page from the Dashboard and import the paper with the Zotero browser extension.

## 7. Enable Scheduling

If the first manual run works, return to **Settings -> arXiv Daily** and enable the scheduler.

The scheduler runs while Obsidian is open. Missed weekdays in the lookback window are retried later.

## Troubleshooting

If **Run Today** is disabled, finish the checklist in **Settings -> arXiv Daily**.

If the Dashboard says there are no indexed papers, run today or run pending dates first.

If a run fails, use **Dashboard -> More -> Show diagnostics** to inspect settings, date context, and recent run state.

If too many papers are selected, narrow your arXiv categories or make topic descriptions more specific.
