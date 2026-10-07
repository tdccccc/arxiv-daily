# arXiv Daily CLI

Command-line tool for [arXiv Daily](https://github.com/tdccccc/arxiv-daily): fetch arXiv by category, filter with an LLM by your research topics, and write Markdown **daily reports** (and optional **paper notes**).

Works standalone in a terminal, local browser, or on a server or always-on machine. The Obsidian plugin is separate; both can share the same vault folder layout.

## Requirements

- Node.js **20.19.0** or newer

## Install

```bash
npm install -g arxiv-daily
```

Or without a global install:

```bash
npx arxiv-daily@latest help
```

Package page: https://www.npmjs.com/package/arxiv-daily

## Quick start

```bash
arxiv-daily init
# guided TUI: vault, LLM, optional email, categories, topic, …
arxiv-daily run --today
```

`init` is a TUI wizard:

- **↑/↓** move · **Space** multi-select · **Enter** confirm (keep default when shown)
- **← Go back** (last item in menus) → previous step  
- **Ctrl+C** (or **Esc**) → **exit** the wizard (not “back”)  
- Defaults appear in prompts — press **Enter** to accept without retyping  

Flow: vault → provider → URL → API key → optional model fetch → model → email →
categories → timezone → language → topic → optional paper-note / schedule flags.
`link_style` and `log_level` are not asked (fixed defaults: wikilink / info).
Config comments are English; **`schema_version` is at the bottom — leave it alone**.

Config path:

- Linux/macOS: `$XDG_CONFIG_HOME/arxiv-daily/config.toml` (default `~/.config/arxiv-daily/config.toml`)
- Windows: `%APPDATA%\arxiv-daily\config.toml`

Default vault from init: `~/arxiv-daily`. No settings env vars; no `--config` / `--vault-root` flags. The config file holds your API keys in plain text — lock it down: `chmod 600 ~/.config/arxiv-daily/config.toml`.

## Daily paper limit

New daily reports contain at most 20 papers across all topics by default. Set a different positive integer in the existing `[output]` table:

```toml
[output]
max_daily_papers = 20
```

The most relevant matching papers are kept before full-text retrieval, detail-note selection, and summarization. Missing values use 20; zero, negative numbers, fractions, and quoted strings are rejected. Changing only this limit can reuse cached filtering results. Reports already written are not automatically rewritten.

## Uninstall

```bash
npm uninstall -g arxiv-daily
```

This removes the global command only. It does **not** delete:

- `~/.config/arxiv-daily/` (config, secrets)
- your vault / output folder (for example `~/arxiv-daily`)

Remove those yourself if you want a full cleanup.

## Commands

```text
arxiv-daily init
arxiv-daily status
arxiv-daily ui [--port PORT] [--no-open]
arxiv-daily papers [--query TEXT] [--offset N] [--limit N]
arxiv-daily library connect PATH
arxiv-daily library status|prepare|scan|index|propose|directions|revoke
arxiv-daily library authorize --fingerprint HASH
arxiv-daily library confirm --candidate ID --proposal-revision N
arxiv-daily library search --query TEXT [--mode hybrid|lexical|dense] [--limit N]
arxiv-daily library review --input REQUEST.json
arxiv-daily update [--check] [--yes]
arxiv-daily run --today
arxiv-daily run --date YYYY-MM-DD
arxiv-daily run --id ARXIV_ID [--date YYYY-MM-DD]
arxiv-daily email test|status|verify-start
arxiv-daily schedule show|install|uninstall
arxiv-daily data export --out PATH.zip
arxiv-daily data import PATH.zip [--yes]
arxiv-daily help
```

- **`update`** — check npm for a newer `arxiv-daily` and optionally `npm install -g` it. Config is not touched. `--check` only reports; `--yes` skips the confirm prompt.
- **`run --today`** — one day only (typical cron entry). Missed days: `run --date …`.
- **`status`** — JSON overview of the configured output paths, topics, model readiness, paper count, and latest run states. Does not start generation or return model endpoints/credentials.
- **`papers`** — paginated JSON from the existing Paper Index; `--query` reuses Dashboard lexical search. Default limit 30, maximum 100. Does not query arXiv or the model.
- **`schedule install`** — writes managed user crontab lines (Linux/macOS/WSL). Not supported on native Windows Task Scheduler; use WSL or the Obsidian plugin for desktop scheduling.

## Local reading workbench

Run `arxiv-daily ui` (with or without an existing configuration) to open the local browser reader. The command prints a full `Workbench:` URL and keeps the service running until Ctrl+C. Use `--no-open` to open the URL yourself or `--port 8123` to choose a fixed port; the default chooses an available loopback port. It serves only on `127.0.0.1`.

Browse daily reports and paper notes, search their titles/authors/IDs/dates, and read existing Markdown with tables, code, images and scientific math. Relative links and unambiguous Obsidian wikilinks navigate between existing reports. Source and PDF buttons open original papers. No reading action changes the original Markdown or calls a model; browser assets and math fonts are embedded in the CLI.

The left sidebar contains the calendar and search/topic/reading filters. The right side starts with all indexed discoveries, including papers without detailed notes, with sorting and pagination. Selecting a date filters its papers; an explicit action opens the complete saved report. Opening a paper shows its saved overview and detail actions. Back/forward stays within the current workbench reading history, restoring filters, anchors and scroll positions; unavailable directions are disabled. History does not persist across workbench restarts. Return to list restores list filters, page and scroll position. “Browse Markdown files” also exposes standalone notes. Other days show their persisted state and offer a date-prefilled generation/retry form when available. Completed zero-match runs remain distinct from ungenerated days, and completed runs whose files are missing do not offer a misleading rerun. Dates use the configured product timezone; browsing does not infer arXiv publication availability or trigger generation.

Waiting for an announcement, confirmed no updates, no filtering matches and genuine failures have distinct statuses. Waiting does not consume the ordinary failure retry budget.

Summary sources and following appendix material are separated from the body. The reading footer shows input/output/total tokens, generation duration and the generation timestamp in UTC. Older missing values are shown as not recorded; file modification time is never substituted. A paper overview that uses its source daily report’s metrics labels that report-wide scope.

Larger colored day cells show paper counts directly. Known run totals take precedence; otherwise existing Paper Index references can provide a count for older reports. Missing counts remain unknown (—), not zero. Drag the sidebar separator or use its arrow keys to resize; the sidebar can also collapse. Width and collapse preferences persist beside CLI configuration across service ports. Mobile navigation opens through the calendar/filter button, with matching light/dark status colors.

Reading status (unmarked/to-read/read) and independent favorites persist through the shared Paper Index. Existing legacy states remain until explicitly changed, stale conflicting edits are rejected, and marking never rewrites Markdown.

Explicit generation actions invoke the same daily/manual CLI workflow, including configured email delivery, and show progress, cancellation and final results. Settings follows the Obsidian 1.13+ section order and controls, including model suggestions, categories, topics, detail policy, output and schedule, personal library, email, advanced and help. With no configuration it opens first-run setup automatically. Save activates the settings immediately; secret inputs stay blank and preserve existing keys when left empty. Show explicitly reveals a saved key through a revision-checked local POST; Hide clears a revealed saved key. Get models populates the dropdown attached to the same editable Model field; it supports typing, filtering and keyboard selection. Changes from another editor are rejected instead of overwritten, and settings cannot be saved during an active generation task. Embedding, email and automatic detail policy are editable; legacy PDF sidecar values remain stored while their retired settings controls stay hidden. Library indexing and explicit email/model-list actions reuse the shared workflows. Enable and Check every (minutes) run shared scheduler checks while the workbench process is open, using a separate workbench_schedule TOML table; existing external cron intent is preserved. The initial workbench is a reader, not a Markdown editor or full Obsidian host. Settings already supports connecting, authorizing and indexing a library. Dedicated library-search and direction-review pages, and fuller run management, remain planned; use the commands below for library search and proposal acceptance.

For the workbench described here, use the current source build until a CLI release containing these changes is published:

```bash
npm ci
npm run build --workspace apps/cli
npm run cli -- ui
```

Run these from the repository root. Source builds require CMake, a C++ compiler and Node-API headers for native storage. The same workbench is available inside [DSH](../../extensions/dsh-arxiv-daily/README.md). Neither integration requires Obsidian.

Settings changes save automatically. Closing settings waits for pending saves; failures keep the editor open so you can retry. Automatic daily reports remain optional after completing your first report.

## Optional personal library

In the workbench, connect and index a folder from **Settings**, then open **Personal library** to browse/search titles and abstracts or open local PDFs (up to 25 MiB). **Review directions** provides proposal editing, representative evidence, previews, explicit partial acceptance, and a library overview. Accepted directions become ordinary research-topic settings; opening either page does not automatically call a model. The commands below remain available for terminal use.

After basic setup, `library connect` selects a read-only PDF source. `library status` displays the processing scope and endpoint-bound authorization fingerprint. Authorize only the displayed scope using `library authorize --fingerprint …`; `library revoke` revokes it.

Run `library prepare` to install pinned optional PDF/runtime components in the configured cache, then `library scan` and `library index`. Local embedding uses the same e5 q8 model and downloads weights on first use. Remote embedding skips the local CPU component and requires the displayed title-and-abstract processing authorization. Local indexing can run before model-processing authorization; direction generation requires authorization.

`library propose` generates topic-grouped direction candidates. Review `library directions`, then confirm a candidate using its displayed proposal revision, or accept selected topics with `library review`. Acceptance atomically writes normal topic settings and receipts. These directions participate in ordinary daily filtering alongside manually entered directions.

The catalog, title-and-abstract indexes and proposals use existing core formats under the active output layout. Accepted topics and their receipts are saved together in the CLI TOML. Library settings live in the CLI TOML and are not automatically synchronized with Obsidian settings. Connection/authorization updates preserve setting values but normalize TOML formatting. Avoid concurrent library rebuild/review writers from different hosts against the same output directory.

The Node CPU runtime has been exercised on Linux Node 20.19 and 22; Windows/macOS CPU smoke remains to be completed before general release.

## Interrupted daily runs and checkpoints

During a daily run, arXiv Daily checkpoints both the validated **paper-filter batch** and each completed per-paper **structured summary** before downstream work proceeds. If the process is cancelled or crashes, rerunning the same date can reuse exact-compatible work instead of repeating paid LLM calls. A valid filter result containing zero selected papers is still retained and reusable. These files are internal **Vault data**, not partial daily reports or paper notes.

The two date-scoped documents live under the active output layout:

- filter: `<output-root>/.index/filter-checkpoints/YYYY-MM-DD.json`
- summaries: `<output-root>/.index/daily-summary-checkpoints/YYYY-MM-DD.json`

Each may have an internal `.bak` recovery file. With the default configuration the roots are `arxiv-daily/.index/filter-checkpoints/` and `arxiv-daily/.index/daily-summary-checkpoints/`. A backup retains the last valid primary across successful replacements and is removed with its date checkpoint after report commit or explicit cleanup; it is not an unbounded history. The output root is derived from the configured `daily_dir` and `papers_dir`; do not manually construct, move, merge, or edit checkpoint JSON. `data export` includes the active `.index/**` tree and both checkpoint kinds without changing the archive format. `data import` restores `.index` only when the archive and active canonical output layouts match. Across different layouts it still imports daily and paper files but warns and skips `.index`; internal index/checkpoint data is never silently relocated.

Filter reuse requires an exact match to the complete rendered filter request (paper IDs, titles, abstracts, topic tags/descriptions, and request ordering), effective provider endpoint identity, model, generation mode, and prompt/result contract versions. Any change invalidates the whole batch; there is no partial filter reuse. Summary reuse likewise requires compatible paper source and effective summary-generation inputs, including language, endpoint identity, model, reasoning settings, and prompt/result contracts. A validated structured result can be reused. A validation-exhausted typed fallback is reused only on the same exact compatibility fingerprint; a transport-exhausted fallback is retried on resume. Corrupt or incompatible state is ignored safely and fresh generation takes over.

At `info` log level, `paper-filter: checkpoint hit|miss|persisted` reports filter recovery and `summarizeDaily: checkpoint hit|miss|persisted` reports each paper's summary recovery. A zero-result filter persistence logs `count=0`; it is not a missing checkpoint. Corruption, backup recovery, and cleanup failure appear as warnings.

The complete committed daily report remains authoritative and is written once through the normal atomic commit path. Existing report/index recovery takes precedence over stale checkpoints. After a successful report commit, both checkpoint documents and their backups are cleaned up best-effort; cleanup failure warns but does not revoke the committed report. To force recomputation after an interrupted run, while no plugin or CLI process is running, delete that date's filter and summary JSON plus `.bak` files. It is also safe to delete either whole checkpoint directory when no run is active. This does not delete a committed daily report. Never clean these files while either host is running against the Vault.

Treat both checkpoint kinds, their backups, and `data export` archives as sensitive Vault data. Filter files contain rendered requests (including paper metadata and research-topic descriptions) and validated model decisions; summary files may contain titles, authors, abstracts, extracted sections, and model-generated results. Endpoint identity is stored only as a digest, and credentials, plaintext endpoints, and raw provider responses are excluded, but the files are still unsafe to publish. On Node hosts, checkpoint primary, temporary, backup, and backup-temporary files are written with mode `0600`; Obsidian's storage API has no portable chmod capability.

A checkpoint fingerprint is only a deterministic **compatibility digest**. It is not a MAC, signature, authenticity proof, or defense against someone who can rewrite Vault files. The Vault and every import archive must therefore come from a trusted source. A malicious local process or user with write access to the Vault is outside this recovery feature's threat model. Protect archives like the Vault and delete copies according to your retention policy. Import validates the raw ZIP central directory before JSZip parsing, then incrementally enforces compressed-size, raw record-count, per-entry and cumulative emitted-byte, compression-ratio, and CRC32 checks while streaming each entry into a private same-directory temporary file.

For a normal runtime error during multi-file promotion, import moves existing targets to private same-directory rollback paths, promotes staged files, and best-effort reverses already completed changes if a later operation fails. A failed rollback reports an aggregate error and preserves recoverable rollback artifacts instead of deleting them. This is not a crash-safe filesystem transaction: process termination, power loss, filesystem failure, or a hostile concurrent local writer can still leave temporary or rollback artifacts requiring manual recovery. Import also rejects symlinked Vault roots, data roots, intermediate components, and targets. These checks reduce accidental path escape and resource exhaustion but cannot eliminate TOCTOU against a concurrent hostile local process; only import into a trusted, quiescent Vault.

## Cron example

```cron
30 9 * * 1-5  arxiv-daily run --today
```

Or set `[schedule]` in the config file and run `arxiv-daily schedule install`.

## Develop from the monorepo

This package is built from the [arxiv-daily](https://github.com/tdccccc/arxiv-daily) workspace:

```bash
git clone https://github.com/tdccccc/arxiv-daily.git
cd arxiv-daily
npm ci
npm run build
npm run cli -- run --today    # from repo root
# or after build:
npx arxiv-daily run --today
```

The published tarball ships a single bundled binary (`dist/arxiv-daily-cli.cjs`).

## License

MIT — see [LICENSE](./LICENSE).
