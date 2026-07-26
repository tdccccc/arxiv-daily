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

The top of the settings page has an accessible four-step **Getting started** guide:

1. **Connect AI**
2. **Choose paper sources**
3. **Describe your research interests**
4. **Generate your first report**

It shows how many of the four steps are complete. Action buttons appear only for pending steps and jump to the existing Settings forms; the guide does not duplicate provider, category, or topic inputs. The final action becomes available when configuration is valid and runs the existing run-now command. Detailed validation reasons remain available under **Configuration details**.

The full guide stays visible until the first report completes. With a completed report and valid configuration, it becomes a compact **Setup complete** summary with the latest completed report date and **Open dashboard**. If configuration later becomes invalid, the full guide returns.

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

Each topic also has a **Detail report** toggle. It does not change whether relevant papers receive structured summaries in the daily report; it only makes papers assigned to that topic eligible for automatic standalone deep dives under `papers/`.

Below the topics, **Automatic detail notes** controls how selective automatic standalone note creation should be: choose **Fewer**, **Recommended**, or **More**. **Recommended** is the default. Existing custom policies remain active until you select another option; advanced custom thresholds remain available through persisted or CLI configuration.

## 5. Run The First Daily Report

Use **Generate first report** in the Settings guide. You can also open the **arXiv Daily Dashboard** from the ribbon icon or command palette and click **Run Today**. The plugin will:

1. Fetch recent arXiv papers for the configured categories.
2. Filter papers against your topics.
3. Fetch available full text and, only when eligible deep-dive candidates exist, make one additional LLM call to score them.
4. Summarize every selected paper into the structured Markdown daily report.
5. Create standalone `papers/<arxiv_id>.md` deep dives for automatically selected papers.
6. Update the Dashboard index.

The generated report appears under:

```text
arxiv-daily/daily/YYYY-MM-DD.md
```

Treat these outputs separately:

- `daily/YYYY-MM-DD.md` contains the day's structured summaries and remains the complete daily reading list.
- `papers/<arxiv_id>.md` is an optional, standalone full-text deep dive for one paper.

Deep-dive scoring happens after full-text retrieval. Eligibility requires the paper's assigned topic to have **Detail report** enabled, usable full text, and no existing paper file. If there are no eligible candidates, there is no extra selector call. If the selector call fails or its response is invalid, no new automatic deep dives are created, but daily summarization continues and the daily run can still succeed. The manual **Summarize by arXiv ID** command is unaffected.

## 6. Use The Dashboard

The Dashboard is the normal entry point after setup.

- **Starred** shows papers you marked as important.
- **All** shows every indexed paper that is not ignored.
- Search is local and relevance-ranked across arXiv ID, title, authors, topics, categories, and structured summary fields. It recognizes exact modern arXiv IDs and tokenizes English technical terms and Chinese text; a non-empty search defaults to relevance, while an explicit starred/published/topic/title sort remains primary.
- **Similar Papers** (the **Find similar papers** row action) uses persisted abstracts and structured summaries recovered from daily reports, with weighted multi-concept and multi-field lexical matching. Weak and author-only matches are suppressed. Results show match reasons, metadata, resource availability, and relevant open actions. The query itself is entirely local: no network, LLM, embedding, or database.
- The calendar opens daily reports by date.
- Row actions open or create a paper note, find similar papers, open the source daily report, open arXiv, open the PDF, or download the PDF. Similar-paper results can open the detail, daily report, arXiv page, or PDF.
- **Dashboard -> More -> Cancel active tasks** cooperatively cancels scheduled or manual daily runs, manual detail summaries, and PDF downloads. **Get Models** is excluded; an already-issued Obsidian `requestUrl` call may finish before later work stops.

For the intended Zotero workflow, open the arXiv page from the Dashboard and import the paper with the Zotero browser extension.

Generated daily reports and detail notes end with a folded **Generation metrics** callout. It shows total pipeline elapsed time when available, LLM elapsed time, logical calls, HTTP attempts, and provider-reported tokens. Missing or retry-incomplete usage is labeled unavailable/incomplete rather than zero, and no cost is estimated.

Paper Index schema 3 stores abstracts and reads existing schema 1 and 2 files. Older entries gain abstracts lazily when encountered by later daily runs; there is no network migration or required bulk rewrite, and existing Markdown remains usable.

## 7. Enable Scheduling

If the first manual run works, return to **Settings -> arXiv Daily** and enable the scheduler.

The scheduler runs while Obsidian is open. Missed weekdays in the lookback window are retried later.

## 8. Optional: Email Digest (Resend, Bring Your Own Key)

Email is optional. The plugin does **not** host a mail server for you. You create a free [Resend](https://resend.com) account, paste your own API key, and the plugin sends **to your personal inbox** after a daily run **completes**. Delivery failures never fail the daily pipeline.

### Important limit (quick setup)

With the default test sender (`onboarding@resend.dev`, used when **From email** is left empty):

- You can usually send **only to the email address on your Resend account**.
- If you signed up with GitHub, that is typically your **GitHub primary email**, not every address linked to GitHub.
- Sending to any other address returns an error until you **verify your own domain** in Resend and set a custom From.

Treat email as a **personal** reminder channel, not a team broadcast tool, unless you complete domain verification.

### Quick setup (recommended)

1. Open [resend.com](https://resend.com) → sign up (GitHub is fine).
2. **API Keys → Create** → copy the `re_…` key (shown once).
3. In Obsidian: **Settings → arXiv Daily → Email delivery**:
   - **To**: the **same** address as your Resend account email (see limit above).
   - **Resend API key**: paste `re_…` and save.
   - **From email**: leave **empty** for quick setup.
4. Click **Send test**. Check inbox and spam for that account address.
5. Only after a successful test, turn on **Daily auto-send**.

API keys are stored in the same local `data.json` style as the LLM key (plaintext on disk; not rendered back into the settings page after save).

### After a real daily run

When **Daily auto-send** is on, a successful **completed** daily run may send one digest for that date. The same date is not resent by default (see vault `arxiv-daily/.index/delivery-state.json`). Repair-only completions do not send mail.

### Advanced: send to other addresses

1. In Resend, add and verify a domain you control (DNS SPF/DKIM).
2. Set **From email** to an address on that domain.
3. **To** may then be any inbox you choose (within Resend’s rules and quota).

## Troubleshooting

If **Run Today** is disabled, finish the checklist in **Settings -> arXiv Daily**.

If the Dashboard says there are no indexed papers, run today or run pending dates first.

If a run fails, use **Dashboard -> More -> Show diagnostics** to inspect settings, date context, and recent run state.

If too many papers are selected, narrow your arXiv categories or make topic descriptions more specific.

If **Send test email** fails with HTTP **403** and a message about “only send testing emails to your own email address”, set **To** exactly to the address named in that error (your Resend account email). Do not use a secondary GitHub email unless it is the account email. Leave **From** empty until you verify a domain.
