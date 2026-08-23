# arXiv Daily CLI

Command-line tool for [arXiv Daily](https://github.com/tdccccc/arxiv-daily): fetch arXiv by category, filter with an LLM by your research topics, and write Markdown **daily reports** (and optional **paper notes**).

Works standalone on a server or always-on machine. The Obsidian plugin is separate; both can share the same vault folder layout.

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

Default vault from init: `~/arxiv-daily`. No settings env vars; no `--config` / `--vault-root` flags.

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
- **`schedule install`** — writes managed user crontab lines (Linux/macOS/WSL). Not supported on native Windows Task Scheduler; use WSL or the Obsidian plugin for desktop scheduling.

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
