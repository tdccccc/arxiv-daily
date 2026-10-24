English · [简体中文](workbench.zh-CN.md)

# Workbench

A local browser reader for [arXiv Daily](https://github.com/tdccccc/arxiv-daily) — no Obsidian required. The same reader is also embedded in the [DSH extension](../extensions/dsh-arxiv-daily/README.md) sidebar.

## Open it

```bash
arxiv-daily ui
arxiv-daily ui --port 8123   # fixed port
arxiv-daily ui --no-open     # print the URL instead of opening a browser
```

It works with or without an existing configuration, serves only on `127.0.0.1`, and keeps running until you press Ctrl+C.

## First-run setup

With no configuration yet, the workbench opens setup automatically: output folder, LLM provider/API key/model, arXiv categories, and research topics. Daily auto-generation stays optional until you turn it on.

## Browsing

- A resizable, collapsible sidebar (drag the separator or use its arrow keys) holds the **calendar** and **search/topic/reading filters**; width and collapse state persist.
- The paper list shows all indexed discoveries, including papers without a detailed note, with sorting and pagination. Picking a date filters to that day; an explicit action opens the full saved report.
- Opening a paper shows its saved overview and detail actions. Day cells are colored and show paper counts directly.
- Markdown renders tables, code, images, and math (`$…$` and `$$…$$`). Relative links and unambiguous Obsidian wikilinks navigate between existing reports; **View on arXiv** and **Read PDF** open the original paper.
- Back/forward moves through your reading history for the current workbench session (filters, anchors, and scroll position are restored); it doesn't persist across restarts. **Back to list** restores the list's filters, page, and scroll position. **Browse Markdown files** also exposes standalone notes.
- Dates use your configured timezone. Browsing never infers arXiv publication timing or starts generation on its own.
- On mobile-width screens, open the calendar/filters through the dedicated button.

## Reading status & favorites

Each paper has a reading status (unmarked / to-read / read) and an independent favorite flag, both saved in the shared Paper Index. Marking a paper never rewrites its Markdown file.

## Generating reports

An explicit **Generate** action runs the same daily/manual CLI workflow (including configured email delivery), with progress and cancellation. Other days show their saved state and, when available, a date-prefilled generate/retry form.

Four outcomes are shown distinctly, so you can tell them apart at a glance:

| Status | Meaning |
|---|---|
| Waiting for announcement | arXiv hasn't published that day's listing yet; doesn't count against the failure retry budget |
| No updates | Confirmed — no new papers in your categories that day |
| No match | Filtering ran, but no paper matched your topics |
| Failed | A real failure occurred |

When a run ends waiting for an announcement, the task panel and that day's calendar card and heading name the newest day arXiv has announced and offer **Generate the report for that day instead** in one click.

The reading footer shows input/output/total tokens, generation duration, and the generation timestamp (UTC). Older reports missing this data show "not recorded" rather than substituting a file's modification time.

## Settings

Settings mirrors the Obsidian plugin's section order: LLM, arXiv categories, research topics, automatic detail notes, output & schedule, personal library, email delivery, advanced, and help.

- Changes save automatically; closing Settings waits for a pending save (retry on failure, or discard).
- Secret fields (API keys) stay blank and keep the existing saved value when left empty. **Show** reveals a saved key; **Hide** clears the reveal.
- **Get models** populates the model dropdown for your provider; the field stays freely editable and filterable.
- Settings can't be saved while a generation task is running.
- If `config.toml` is edited outside the workbench while it is open, generating reports and marking papers are refused with a message saying so, until you restart the workbench or reopen Settings and save; browsing and reading keep working.
- The setup guide's **Generate first report** uses the newest day arXiv has announced (today when it is already out, otherwise the latest earlier day), so it works at weekends and before the announcement; its step shows the progress and result, because the Settings dialog covers the task panel.
- **Output & schedule** also has an in-process scheduler: **Enable** + **Check every (minutes)** runs while the workbench process stays open (stored separately as `workbench_schedule`, independent of any external cron job you've set up with `schedule install`).

## Appearance & UI language

**Settings → Appearance** sets the theme and interface language (English/Chinese), independent of the language your reports and summaries are generated in.

## Personal library & direction review

Connect and index a local PDF folder from **Settings → Personal library**, then use:

- **Personal library** — browse/search your indexed papers by title and abstract, and open local PDFs (up to 25 MiB) in a new tab.
- **Review directions** — inspect proposed topics/directions with representative evidence and previews, edit them, and accept some or all. Accepted directions become ordinary research topics and join daily filtering. When the library holds papers no earlier analysis accounted for, the page offers an update that sends only those new papers to the model. Opening either page doesn't call a model by itself.

Library search and indexing from a terminal are also available via `arxiv-daily library …` — see the [CLI README](https://github.com/tdccccc/arxiv-daily/blob/main/apps/cli/README.md#optional-personal-library).

## Privacy

No reading or browsing action calls a model or edits your Markdown. Only explicit actions — generating a report, indexing/proposing in the personal library, fetching the model list — make network/model calls.
