# arXiv Daily

Filter new arXiv papers by your research topics and write Markdown daily reports—via the Obsidian plugin or the CLI.

[Getting Started](docs/getting-started.md) · [中文说明](docs/README.zh-CN.md) · [新手教程](docs/getting-started.zh-CN.md)

**arXiv Daily** fetches the categories you care about, uses an LLM to keep papers that match your topics, and writes **Markdown you can search and link**: a **daily report**, optional **paper notes**, and a **Dashboard** to revisit them.

## What it does

- **Filters the flood** — listings down to papers relevant to *your* topics
- **Writes a daily report** — one Markdown file per day, grouped by topic, with a short structured summary per paper
- **Can add paper notes** — longer per-paper notes when you want more depth (automatic or by arXiv ID)
- **Helps you review** — Dashboard with calendar, search, topics, and stars
- **Runs on a schedule** — in Obsidian while the app is open, or via CLI on a machine that stays online
- **Optional email** — a short digest after a successful day (your Resend key, or Official delivery Beta)

## What you get

| Output | Where | What it is |
|---|---|---|
| **Daily report** | `arxiv-daily/daily/YYYY-MM-DD.md` | That day’s reading list: topics, selected papers, structured short summaries |
| **Paper note** | `arxiv-daily/papers/<arxiv_id>.md` | A longer note for one paper (not the same as the daily entry) |
| **Dashboard** | In Obsidian | Calendar, search, filters, stars, open report / note / arXiv / PDF |

```text
arxiv-daily/
  daily/          # daily reports
  papers/         # paper notes
  pdfs/           # optional downloads
  .index/         # local index & run state
```

## Two ways to use it

| | **Obsidian plugin** | **CLI** |
|---|---|---|
| Best for | Daily use in your vault, UI, Dashboard | Servers, cron, always-on machines (e.g. VPN) |
| Config | Plugin settings in Obsidian | `~/.config/arxiv-daily/config.toml` (`init`) |
| Schedule | While Obsidian is open | System cron → `run --today` (or WSL on Windows) |
| Shared | Same core pipeline; same vault layout if you point at the same folder | |

Most people start with the **plugin**. Use the **CLI** when you want reports without keeping Obsidian open.

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
3. **Describe your research interests** — at least one topic (name, tag, description)  
4. **Generate your first report** — from the settings guide or Dashboard **Run Today**

The guide stays until a report completes. Details: [Getting Started](docs/getting-started.md).

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
- The current preview does not change daily filtering, reports, paper notes, or email delivery.

---

## CLI

For cron or a machine that stays online. Requires Node.js 20.11.0+.

### Install (npm)

Requires Node.js 20.11+.

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

More: [apps/cli/README.md](apps/cli/README.md) · [0.3.4 release notes](docs/releases/0.3.4.md) · [CLI design notes](docs/helm/2026-07-28-cli-product-config-and-data-portability/).

### From this repository (developers)

```bash
npm ci && npm run build
npm run cli -- run --today    # runs apps/cli/dist/arxiv-daily-cli.cjs
```

---

## Development

```bash
npm ci
npm run check:boundaries
npm run lint
npm run typecheck
npm test
npm run build
```

One npm workspace: `packages/core` (pipeline), `packages/node-runtime`, `apps/cli`, `plugin` (Obsidian UI). Release version sync: `npm run sync:release-version -- <ver>`.
