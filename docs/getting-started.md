# Getting Started

Get your first **daily report** in Obsidian, then optionally turn on scheduling and email.

For a short product overview (plugin + CLI), see the [README](../README.md). This guide is for the **Obsidian plugin**.

## What you need

- Obsidian **desktop**
- An LLM API key (and base URL / model if not using a preset)
- One or more arXiv categories (for example `astro-ph`, `cs.LG`, `hep-th`)
- A short description of each research topic you want to track

Generated files go in your vault under `arxiv-daily/` by default. API keys are stored in the plugin data file on this device (shown as **Configured** after save—use **Replace** or **Clear** to change them).

## 1. Open settings

Install and enable **arXiv Daily**, then open:

```text
Settings → arXiv Daily
```

At the top, a four-step guide walks you through setup:

1. **Connect AI**
2. **Choose paper sources**
3. **Describe your research interests**
4. **Generate your first report**

Buttons jump to the matching form. The full guide stays until a report completes; then you get a short “setup complete” summary (it returns if settings become invalid).

## 2. Connect AI

Pick a provider, enter your API key, and save. Adjust base URL and model if needed. Defaults are fine for a first run unless your provider requires otherwise.

## 3. Choose arXiv categories

Select the categories to fetch. You can pick several; the same paper is only kept once.

Examples: `astro-ph`, `astro-ph.CO`, `cs.LG`.

## 4. Add research topics

Each topic becomes a **section in the daily report**.

For each topic set:

- **Name** — section title  
- **Tag** — short slug  
- **Description** — in plain language, which papers belong here  

Example:

```text
Name: Photometric Redshift
Tag: photo-z
Description: Methods, benchmarks, uncertainty calibration, catalog construction,
and systematics for photometric redshift estimation.
```

You can start from a template, then edit.

**Paper notes (optional depth):**  
Each topic has a **Detail report** option: papers in that topic may get a longer **paper note** under `papers/` (not just the short entry in the daily report). Below the topic list, **Automatic detail notes** (Fewer / Recommended / More) controls how often those notes are created automatically. You can always create a paper note manually later (for example **Summarize by arXiv ID**).

## 5. Generate your first daily report

In the guide, use **Generate first report**, or open the **Dashboard** and click **Run Today**.

The plugin will:

1. Fetch recent papers for your categories  
2. Keep those that match your topics  
3. Write a **daily report** with a short structured summary for each selected paper  
4. Sometimes create **paper notes** for a few papers (if detail options allow)  
5. Update the Dashboard  

Your report is here:

```text
arxiv-daily/daily/YYYY-MM-DD.md
```

| Output | Path | Role |
|---|---|---|
| **Daily report** | `daily/YYYY-MM-DD.md` | That day’s reading list (main result) |
| **Paper note** | `papers/<arxiv_id>.md` | Longer note for one paper |

If automatic paper notes fail or are skipped, the daily report can still succeed.

## 6. Use the Dashboard

After setup, the Dashboard is the usual home:

- **Starred** / **All** — focus on papers you mark as important  
- **Calendar** — open a daily report by date  
- **Search and filters** — find papers in your local index  
- **Row actions** — open the daily report, paper note, arXiv page, or PDF; star a paper  

Open arXiv from a row if you want to import into a reference manager (for example Zotero).

## 7. Turn on the scheduler

When a manual run works, enable the scheduler in **Settings → arXiv Daily**.

It only runs while **Obsidian is open**, on weekdays in your configured window. Missed weekdays can be picked up later while the app is open.

## 8. Optional: email

Email is optional. A failed send **never** fails the daily report.

| Mode | What you do |
|---|---|
| **Send yourself** (default) | Your [Resend](https://resend.com) API key; no project quota |
| **Official delivery (Beta)** | Verify your address once; shared free capacity—light personal use only |

### Send yourself (quick)

1. Create a Resend account and API key (`re_…`).  
2. **Settings → arXiv Daily → Email delivery**  
   - How to send: **Send yourself**  
   - **Your email**: usually the **same** address as your Resend account  
   - Paste the API key; leave **From email** empty for the simplest setup  
3. **Send test**, check inbox/spam.  
4. Only then turn on **Daily auto-send**.

With From empty, Resend’s test sender often allows mail **only to your Resend account email** (GitHub login → often your GitHub **primary** email). To send elsewhere, verify a domain in Resend and set a custom From.

### Official delivery (Beta)

1. Choose **Official delivery (Beta)**.  
2. Enter your email → **Send verification email**.  
3. Open the link; paste the **long code from the web page** (not the short code in the link).  
4. **Send test**, then enable **Daily auto-send** if it works.  
5. If you hit the daily limit, wait for the next UTC day or switch to **Send yourself**.

Beta capacity is intentionally small (a few messages per verified inbox per UTC day; tests count). For heavy use, use **Send yourself**.

### After a real run

With auto-send on, a **completed** daily run may email one digest for that date. The same date is not resent by default. **Send test** does not block the real daily for that day.

## Troubleshooting

| Problem | What to try |
|---|---|
| **Run Today** disabled | Finish the checklist in Settings → arXiv Daily |
| No papers on the Dashboard | Run today (or another date) once so the index fills |
| Run failed | Dashboard → More → Show diagnostics |
| Too many papers selected | Fewer categories, or more specific topic descriptions |
| Test email HTTP **403** | Set **Your email** to the Resend account address named in the error; leave From empty until you verify a domain |

## CLI (optional)

If you want reports without keeping Obsidian open, see the [CLI section in the README](../README.md#cli): `init`, then `run --today`, config at `~/.config/arxiv-daily/config.toml`. On Windows, prefer **WSL** for CLI scheduling, or stay on the plugin.
