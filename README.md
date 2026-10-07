# arXiv Daily

Follow research directions, filter new arXiv papers, and keep daily reports and detailed paper notes in local Markdown.

[Getting Started](docs/getting-started.md) · [中文说明](docs/README.zh-CN.md) · [新手教程](docs/getting-started.zh-CN.md)

**arXiv Daily** is a personal research companion built around the **Obsidian plugin**. It fetches the categories you follow, selects papers matching the directions inside your research topics, and saves daily reports and paper notes you can search, link, and keep. A standalone reading workbench also works without Obsidian, in your browser or inside DeepSeek Harness (DSH).

## What it does

- **Filters the flood** — listings down to papers relevant to *your* topics
- **Writes a daily report** — up to 20 papers per day by default, ranked by relevance and grouped by topic, with a short structured summary per paper; adjust **Daily paper limit** in settings
- **Can add paper notes** — longer per-paper notes when you want more depth (automatic or by arXiv ID)
- **Helps you review** — Obsidian Dashboard or standalone workbench with calendar, paper list, search, topic filters, and favorites
- **Uses your literature context** — optionally index a personal library and review proposed directions before accepting them into daily discovery
- **Runs on a schedule** — in Obsidian while the app is open, or via CLI on a machine that stays online
- **Optional email** — a short digest after a successful day (your Resend key, or Official delivery Beta)

## What you get

| Output | Where | What it is |
|---|---|---|
| **Daily report** | `arxiv-daily/daily/YYYY-MM-DD.md` | That day’s reading list: topics, selected papers, structured short summaries |
| **Paper note** | `arxiv-daily/papers/<arxiv_id>.md` | A longer note for one paper (not the same as the daily entry) |
| **Reading views** | Obsidian Dashboard, local browser, or DSH sidebar | Browse the calendar and papers; open reports, notes, arXiv pages, and PDFs |

```text
arxiv-daily/
  daily/          # daily reports
  papers/         # paper notes
  pdfs/           # optional downloads
  .index/         # local index & run state
```

## Choose your entry

| Entry | Best for | Configuration and scheduling |
|---|---|---|
| **Obsidian plugin** | Reading and maintaining research in your vault | Native plugin settings; scheduling while Obsidian is open |
| **Standalone workbench** | Reading and generating reports without Obsidian | Graphical settings backed by CLI TOML; scheduling while the workbench runs |
| **DSH plugin** | The same workbench in a fixed DSH sidebar | Reuses the CLI/workbench configuration and local records |
| **CLI** | Terminal tasks or external scheduling | CLI TOML; optional system cron |

Start with **Obsidian** if you already use it. Choose the **workbench** if you want a standalone reader. Research does not depend on an agent conversation.

The hosts share core discovery rules, settings definitions, and record formats. **Obsidian and CLI configuration values and API keys do not automatically synchronize.** DSH and the browser workbench both use the CLI configuration. Pointing hosts at the same output directory shares records, not settings.

The standalone workbench and DSH **0.1.18** described here are available from source/local builds; these additions have not been published to npm. See the build instructions below rather than assuming the latest npm release includes them.

---

## Obsidian plugin

### Install

Desktop Obsidian only.

1. **Community plugins** — Settings → Community plugins → Browse → **arXiv Daily**
2. **BRAT** — add `tdccccc/arxiv-daily`
3. **Manual** — put `manifest.json`, `main.js`, and `styles.css` from the [latest release](https://github.com/tdccccc/arxiv-daily/releases/latest) in:

```text
<vault>/.obsidian/plugins/arxiv-daily/
```

Enable the plugin, then open **Settings → arXiv Daily**.

### Quick start

1. **Connect AI** — API key, base URL, model  
2. **Choose paper sources** — one or more arXiv categories  
3. **Describe your research interests** — at least one named topic containing specific research directions
4. **Generate your first report** — from the settings guide or Dashboard **Run Today**

The guide also offers scheduling; after all setup steps are completed, it stays dismissed. Details: [Getting Started](docs/getting-started.md).

### Day to day

- Open the **Dashboard** (ribbon or command palette)
- **Run Today** or let the scheduler run on weekdays while Obsidian is open
- Read the **daily report**; star papers that matter
- Open or create a **paper note** when you want more depth
- Optional: enable **email** after a successful test send

### Personal library access (desktop preview)

The plugin can connect one local paper-library folder, including a folder outside your Vault. Access is **read-only** and limited to the folder you explicitly select: arXiv Daily cannot write, rename, or delete its files, and symbolic links are not followed.

- **Inventory preview stays local** and shows which PDFs are eligible or ignored; it does not require model-processing authorization.
- **Model processing is separately authorized** after showing the selected folder, eligible file types, processing depth, and effective model endpoint.
- Changing the folder, endpoint, eligible file types, or processing depth invalidates authorization. You can also revoke it at any time.
- The library can propose a few broad topics and directions. Accept your selected topics and directions into settings to use them for daily filtering.

---

## Standalone reading workbench and DSH

From the repository, with Node.js 20.19+ and the [native build prerequisites](apps/cli/README.md):

```bash
npm ci
npm run build
node apps/cli/dist/arxiv-daily-cli.cjs ui
```

Keep the process running while using the browser page. With no configuration, the workbench opens first-use settings: choose a save directory, configure the model, select arXiv categories, and add topics and directions. Existing CLI settings are reused.

The left calendar shows report status and paper counts; the right side starts with a paper list and opens a report or paper when selected. Read, mark, and favorite existing papers without model calls. Generate a daily report or a detailed paper note when needed.

- Read rendered Markdown, tables, and LaTeX formulas; original Markdown remains available and unchanged by reading.
- Move backward and forward through the current reading session, retaining filters, anchors, and scroll position.
- Review sources separately from the body and see recorded token usage, generation duration, and timestamp. Missing statistics in older records are shown as unrecorded; report-wide usage is labeled accordingly.
- Edit settings in the workbench; changes save automatically, with visible progress and errors. **Appearance** controls theme and Chinese/English interface language separately from summary language.
- Distinguish awaiting announcements, no updates, no matching papers, and genuine failures. Waiting for an announcement does not consume ordinary failure retries.

To embed this workbench in DSH, follow the [DSH build and installation guide](extensions/dsh-arxiv-daily/README.md). Open **arxiv-daily** above Settings in the left sidebar or from the right sidebar; no message is required. The local package includes native modules for its build platform. Linux/x64 and DSH Host integration have been tested; cross-platform packages and Electron visual behavior are not fully verified.

The workbench reads Markdown; it does not edit it. **Personal library** opens catalog browsing, indexed title/abstract search, and local PDFs. **Review directions** shows proposals and a library overview: inspect representative papers, edit or move candidates, preview matches, and explicitly accept selected directions into research topics. Connection and indexing remain in settings. Proposed directions only affect discovery after acceptance. Richer run management remains planned; the [CLI guide](apps/cli/README.md) also documents terminal library commands.

---

## CLI

For cron or a machine that stays online. Requires Node.js 20.19.0+.

### Install (npm)

Requires Node.js 20.19+.

```bash
npm install -g arxiv-daily
arxiv-daily init          # guided TUI; Enter keeps defaults
arxiv-daily run --today
```

Or without a global install: `npx arxiv-daily@latest help`.

Config is only **`$XDG_CONFIG_HOME/arxiv-daily/config.toml`** (default `~/.config/arxiv-daily/config.toml`). No settings env vars; no `--config` / `--vault-root`. After init you can hand-edit topics in that file. The file holds your API keys in plain text — lock it down: `chmod 600 ~/.config/arxiv-daily/config.toml`.

```bash
arxiv-daily update              # upgrade global install when a newer npm release exists
arxiv-daily update --check      # only print current vs latest
arxiv-daily run --date 2026-06-13
arxiv-daily run --id 2606.12345
arxiv-daily email test
# set [schedule] enabled = true, then:
arxiv-daily schedule install
```

Uninstall the CLI package (does **not** delete config or vault data):

```bash
npm uninstall -g arxiv-daily
# optional: rm -rf ~/.config/arxiv-daily
```

On **Windows**, prefer **WSL** for CLI + cron, or the **Obsidian plugin** for desktop scheduling.

More: [CLI installation and command reference](apps/cli/README.md).

---

## License

MIT — see [LICENSE](LICENSE).
