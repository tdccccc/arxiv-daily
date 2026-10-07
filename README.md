# arXiv Daily

Follow research directions, filter new arXiv papers, and keep daily reports and detailed paper notes in local Markdown.

[Getting Started](docs/getting-started.md) · [中文说明](docs/README.zh-CN.md) · [新手教程](docs/getting-started.zh-CN.md)

**arXiv Daily** is a personal research companion built around the **Obsidian plugin**, with a standalone browser workbench and a DeepSeek Harness (DSH) extension that work without Obsidian.

## Features

- **Filters the flood** — keeps only papers matching the directions inside your research topics
- **Daily report** — up to 20 papers a day by default, grouped by topic, each with a short structured summary
- **Paper notes** — longer per-paper notes on demand, automatic or by arXiv ID
- **Reading views** — Obsidian Dashboard, browser workbench, or DSH sidebar, with calendar, search, topic filters, and favorites
- **Personal library** — optionally index your own PDFs; review and explicitly accept proposed topics and directions before they affect discovery
- **Scheduling** — in Obsidian while it's open, or via CLI/cron on a machine that stays online
- **Optional email** — a short digest after a successful day (your Resend key, or Official delivery Beta)

Reports land in `arxiv-daily/daily/YYYY-MM-DD.md`; paper notes in `arxiv-daily/papers/<arxiv_id>.md`.

## Ways to use

| Entry | Best for |
|---|---|
| **Obsidian plugin** | Reading and maintaining research in your vault |
| **CLI + browser workbench** | Terminal use, external scheduling, or a standalone reader without Obsidian |
| **DSH extension** | The same workbench in a fixed DeepSeek Harness sidebar |

Obsidian and CLI configuration values and API keys do not automatically synchronize; the browser workbench and DSH extension both reuse the CLI configuration.

## Install / quick start

**Obsidian** (desktop only):

1. **Community plugins** — Settings → Community plugins → Browse → **arXiv Daily**
2. **BRAT** — add `tdccccc/arxiv-daily`
3. **Manual** — copy `manifest.json`, `main.js`, `styles.css` from the [latest release](https://github.com/tdccccc/arxiv-daily/releases/latest) into `<vault>/.obsidian/plugins/arxiv-daily/`

Enable the plugin, then open **Settings → arXiv Daily**. Details: [Getting Started](docs/getting-started.md).

**CLI & browser workbench** (Node.js 20.19+):

```bash
npm install -g arxiv-daily
arxiv-daily init          # guided setup; Enter keeps defaults
arxiv-daily run --today
arxiv-daily ui            # open the local reading workbench in your browser
```

More: [CLI installation and command reference](apps/cli/README.md).

**DSH** — build and install the extension from source: [extensions/dsh-arxiv-daily/README.md](extensions/dsh-arxiv-daily/README.md).

## License

MIT — see [LICENSE](LICENSE).
