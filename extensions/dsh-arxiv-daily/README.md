English · [简体中文](README.zh-CN.md)

# arXiv Daily for DeepSeek Harness

The [arXiv Daily](../../README.md) reading workbench inside DeepSeek Harness (DSH) — a fixed **arxiv-daily** entry above Settings in the left sidebar, and a matching tab in the right sidebar. Both open the same full workbench: calendar, paper list, daily reports, detailed summaries, and reading/favorite marks. Opening, searching, and marking never start an Agent conversation or call a model.

The workbench uses its own saved model settings, separate from the DSH chat model, and shares your existing CLI configuration — no separate setup if you already use the [CLI](../../apps/cli/README.md). The [Obsidian plugin](../../plugin/README.md) keeps its own settings and API keys, which are not shared, but it can use the same output folder.

## Requirements

- DSH with Node.js **22.19.0** or newer
- [pnpm](https://pnpm.io) on `PATH` (DSH's plugin installer uses it)

## Install

**DSH Web**:

```sh
dsh plugin --profile web add dsh-arxiv-daily
```

Then restart DSH so the host and client both load the new plugin.

**DSH Desktop**: open **Plugins → Add plugin**, enter the package name `dsh-arxiv-daily` in the package field, click **Install**, then **Enable now**. DSH Desktop's Add plugin dialog accepts a plain npm package name (the same thing that follows `dsh plugin add` on the Web profile) — no local file or path needed.

The plugin registers its own **arxiv-daily** sidebar tab; no manual configuration is required to see the entry point.

## Upgrade

**DSH Web**:

```sh
dsh plugin --profile web add dsh-arxiv-daily@latest
```

Then restart DSH.

**DSH Desktop**: installed plugins don't update automatically. Upgrade by uninstalling the plugin from **Plugins**, then adding it again the same way as a fresh install (Desktop has no version picker or in-place upgrade).

## Uninstall

Removing the plugin package (Desktop: **Plugins → arxiv-daily → Uninstall**; Web: `dsh plugin --profile web remove dsh-arxiv-daily`) only removes the plugin itself. Your arXiv Daily configuration, Markdown reports/notes, and paper index are untouched — they live outside the plugin, in your CLI config and output folder (see [CLI configuration](../../apps/cli/README.md#configuration)).

## First use

Click **arxiv-daily** above Settings in the left sidebar (or the matching right-sidebar tab) — no need to send a message first. If you already have a CLI configuration, the workbench reuses it directly; otherwise it opens setup automatically, the same first-run flow as the [standalone workbench](../../docs/workbench.md#first-run-setup).

## Usage

- The workbench inside DSH is the same one described in [docs/workbench.md](../../docs/workbench.md) — calendar/filters sidebar, paper list, Markdown reader with math, settings, personal library, and direction review all work the same way here.
- Use DSH's own sidebar widen/float controls to resize the panel. On narrow widths, switch to the calendar/filters through the dedicated toggle.
- Model requests use arXiv Daily's own saved API settings, separate from the DSH chat model.

See [docs/workbench.md](../../docs/workbench.md) for the full feature guide.

## Limitations

- **Local DSH only** — the workbench talks to a locally running `arxiv-daily` process; it does not support a remote DSH reaching back into your computer's paper library.
- **Restart after update** — replacing the plugin package on disk doesn't replace an already-running workbench process; quit and restart DSH, then reload the browser tab. The workbench process stops together with DSH, also when DSH is force-quit or crashes, so quitting DSH is enough.
- **One origin per workbench lifetime** — a workbench session binds to a single page origin. If you mix `localhost` and `127.0.0.1` across DSH Web sessions, go back to the original address, or re-enable the plugin after your session ends.

## Build from source

To try an unreleased change, or to use a local build:

```sh
npm ci
npm ci --prefix extensions/dsh-arxiv-daily --ignore-scripts
node extensions/dsh-arxiv-daily/build.mjs
npm pack ./extensions/dsh-arxiv-daily/dist/package --pack-destination ./extensions/dsh-arxiv-daily/dist
```

This requires CMake, a C++ compiler, and Node-API headers to build the native storage module for your current OS/architecture; the resulting `.tgz` only contains a build for that platform.

Install the local tarball:

```sh
dsh plugin --profile web add /absolute/path/dsh-arxiv-daily-<version>.tgz
```

**DSH Desktop** also accepts an absolute local path or a `.tgz` in the same **Add plugin** dialog.

## License

MIT — see [LICENSE](../../LICENSE).
