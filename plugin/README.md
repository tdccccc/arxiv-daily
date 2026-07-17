# arXiv Daily — Obsidian plugin

Native TypeScript Obsidian plugin. Replaces `arxiv_daily.py` with an
in-vault settings GUI, catch-up scheduling, and on-demand manual runs.

## Features

- **Shared core + Node CLI** — the fetch/filter/summarize/write pipeline is
  host-neutral and reused by Obsidian and the Node CLI (`run`, `run-pending`,
  `summarize`).
- **Submitted-date catch-up fallback** — manual date runs outside arXiv
  `/recent` use the export API `submittedDate` day window and annotate daily
  reports with the approximate date semantics.
- **Configurable markdown links** — daily reports can use Obsidian wikilinks
  or standard relative Markdown links for CLI / VS Code-friendly output.
- **arXiv Daily Dashboard as the primary entry** — the Obsidian ribbon opens
  the Dashboard directly. Starred / All tabs, run controls, and the More menu
  are available from the top toolbar.
- **Daily-report calendar** — the Dashboard shows monthly daily reports,
  highlights today, provides a Today shortcut, and opens a report by clicking
  its date.
- **Simplified reading workflow** — one-click stars mark high-priority papers;
  unstarred papers stay neutral. Dashboard filters cover search, topic,
  first-seen date range, note presence, and detail availability.
- **Local relevance search** — BM25F-style ranking over arXiv ID, title,
  authors, topics, categories, and structured summary fields, with exact modern
  arXiv-ID priority plus deterministic English technical-token and Chinese
  bigram tokenization. Non-empty searches default to relevance; an explicitly
  selected starred/published/topic/title sort remains primary.
- **Similar Papers** — deterministic local lexical retrieval over non-ignored
  Paper Index entries, with match reasons and actions for detail, daily report,
  arXiv, and PDF. It makes no network or LLM request and uses no embedding or
  database.
- **First-run onboarding** — Settings includes a Getting Started checklist, and
  Dashboard empty states point users to setup, Run Today, Run Pending, or All.
- **Safer arXiv date handling** — current or scheduled dates newer than the
  latest `/recent` announce bucket stay retryable instead of using the
  submitted-date fallback.
- **Focused row actions** — open/create the paper note, open the source daily
  report, open arXiv, open the PDF, or download the PDF. Paper titles,
  authors, and summaries are selectable for copying.
- **Research workflow handoff** — arXiv Daily handles discovery, review,
  local notes, and optional PDF downloads. Zotero remains the external source
  of truth for citation keys and BibTeX.
- **Paper index schema v2** — stores structured daily summary fields
  (`coreProblem`, `keyMethod`, `mainResult`, `whyRelevant`, `limitations`,
  `sourceSections`) for Dashboard search and review.
- **Workflow quick wins** — startup sync for daily checkbox selections,
  missed-paper fallback lists, and multiple arXiv categories.
- **arXiv date parsing fix** — accepts abbreviated month names in listing
  headers (for example `Wed, 10 Jun 2026`) so June and later daily runs
  match the correct `/recent` bucket.
- **Topic cards** — collapsible cards replace the five legacy arXiv fields
  (`researchInterests` / `detailCriteria` / `detailCategories` /
  `categoryTagMap` / `categoryDisplayMap`). Each topic owns `name`
  (daily-report heading), `tag` (slug, auto-derived on creation),
  `description` (LLM filter input), and a per-topic `detail` toggle.
- **Template presets** — Blank, Astrophysics + ML, NLP / LLMs,
  Computer Vision, Bioinformatics. Load Template dropdown with
  confirm-before-replace for non-empty lists.
- **Config validation** — `validateLlmConfig` / `validateFilterConfig`
  gate scheduled and manual runs. Settings panel renders a red banner
  listing missing required fields.
- **Secret-safe settings and output** — a saved API key renders only as
  `Configured`, with explicit Replace and Clear actions. The compatible
  `llm.apiKey` value remains plaintext in
  `<vault>/.obsidian/plugins/arxiv-daily/data.json`; vault sync/backup tools may
  copy it, and there is no keyring or encryption. Fetched source content is
  cached in the adjacent `.cache/` directory for the configured retention
  period. Logs, diagnostics, and presented errors are redacted.
- **Generation metrics** — generated daily and detail Markdown ends with a
  folded callout reporting pipeline elapsed time when available, LLM elapsed
  time, logical calls, HTTP attempts, and provider-reported tokens. Missing or
  retry-incomplete usage is not treated as zero, and no cost is estimated.
- **Unified cancellation** — `Cancel active tasks` covers scheduled and manual
  daily runs, manual detail summaries, and PDF downloads, but not Get Models.
  Obsidian `requestUrl` remains cooperative: already-issued requests may finish
  while later work unwinds.
- **First-enable modal** — `chooseModal` asks Run today / Skip today /
  Cancel on every OFF→ON transition. "Skip today" marks the current
  date as `skipped` in runState so interval ticks leave it alone.
- **Tiered help text** — inline `.setDesc()` on essential fields;
  Obsidian `setTooltip`-backed `(?)` badge on advanced fields; muted
  hint text below topic-card labels.
- One-shot, lossy migration from v0.1.x on first load (`migration.ts`).
- Full vitest suite and production build are expected to pass before release.

## Dashboard Features

### Settings Button
- Click the "Settings" button in the top-right corner of the Dashboard to quickly access plugin settings

### Calendar Runnable Dates
- Dates within the 5-day lookback window that don't have daily reports are highlighted in green
- Click on a green date to generate a report for that date
- Hover over the date to see a tooltip with instructions

## Installation

### Option A — via BRAT (recommended)

1. Install the [BRAT](https://github.com/TfTHacker/obsidian42-brat) plugin.
2. Open BRAT settings → Add Beta plugin → paste `tdccccc/arxiv-daily`.
3. Enable **arXiv Daily** in Community plugins.
4. Open Settings → arXiv Daily, pick a provider and fill in the API Key.

### Option B — manual install

1. Download `manifest.json`, `main.js`, and `styles.css` from the
   [latest release](https://github.com/tdccccc/arxiv-daily/releases/latest).
2. Create `<your-vault>/.obsidian/plugins/arxiv-daily/` if needed and put the
   three files directly inside it, with no nested release/repository folder.
3. Restart Obsidian and enable the plugin under **Settings → Community plugins**.

### Option C — build from source

```bash
git clone https://github.com/tdccccc/arxiv-daily.git
cd arxiv-daily
npm ci
npm run build
# then copy manifest.json + main.js + styles.css into the vault
```

## Settings overview (user-visible)

| Section | Fields |
|---|---|
| **Enable** | Toggle, shows Running / Paused status |
| **LLM** | Provider dropdown, API Key (`Configured` sentinel with Replace/Clear), Base URL, Model, Temperature, Timeout, Thinking mode, Reasoning effort |
| **arXiv** | Category dropdown (grouped), Research Topics (collapsible cards with Name/Tag/Description/Detail toggle), Load Template dropdown, + Add Topic button, Timezone |
| **Output & Schedule** | Daily / Papers paths, Link style, Run time, Tick interval, Lookback days (≤ 5) |
| **Advanced** | Request delay, cache TTL, char limits, skip / priority sections, log level |

## Key modules (developer reference)

### Core contracts

| File | Role |
|---|---|
| `packages/core/src/core/adapters.ts` | Host-neutral contracts for HTTP, storage, secrets, progress, and resource opening |

### Host adapters

| File | Role |
|---|---|
| `src/hosts/obsidian/http-client.ts` | Obsidian `requestUrl` implementation of the core `HttpClient` contract |
| `src/hosts/obsidian/storage-adapter.ts` | Obsidian `Vault` implementation of the core `StorageAdapter` contract |
| `src/hosts/obsidian/*.ts` | Obsidian host adapter assembly for HTTP, vault storage, settings secrets, progress, and workspace resource opening |
| `packages/node-runtime/src/*.ts` | Node implementations for HTTP, filesystem storage, env secrets, progress, and resource output |

### Settings layer

| File | Role |
|---|---|
| `packages/core/src/settings/types.ts` | `Topic`, `ArxivSettings`, `PluginSettings`, `RunStatus` (`"skipped"` added in v0.1.2) |
| `packages/core/src/settings/defaults.ts` | `DEFAULT_SETTINGS` — topics is `[]` by default |
| `packages/core/src/settings/migration.ts` | `migrateArxivSettings(raw)` — lossy upgrade from v0.1.x legacy fields |
| `packages/core/src/settings/validation.ts` | `validateLlmConfig`, `validateFilterConfig` — gatekeeper for all runs |
| `packages/core/src/settings/topic-templates.ts` | `TOPIC_TEMPLATES` — five static presets |
| `plugin/src/settings/tab.ts` | `ArxivDailySettingTab` — full Obsidian settings UI (topic cards, template loader, validation banner, `attachHelp`, `confirmReplace`) |
| `packages/core/src/settings/providers.ts` | LLM provider presets |
| `packages/core/src/settings/arxiv-categories.ts` | Grouped arXiv category data |

### Pipeline

| File | Role |
|---|---|
| `packages/core/src/pipeline/arxiv-fetcher.ts` | HTTP fetcher for `/recent`, export API submittedDate fallback, and `/abs` |
| `packages/core/src/pipeline/arxiv-parser.ts` | HTML listing → `PaperMeta[]` |
| `packages/core/src/pipeline/atom-parser.ts` | Atom API → abstract enrichment |
| `packages/core/src/pipeline/paper-filter.ts` | LLM-call: classifies papers into topics (or `"skip"`); short-circuits when topics is empty |
| `packages/core/src/pipeline/paper-content.ts` | Full-text HTML fetch + section extraction |
| `packages/core/src/pipeline/section-extractor.ts` | Splits HTML body into named sections |
| `packages/core/src/pipeline/summarizer.ts` | Daily summary + per-paper detail LLM prompts |
| `packages/core/src/pipeline/markdown-writer.ts` | Writes `.md` files into vault paths; reads tags from `topics` array |
| `packages/core/src/pipeline/pipeline.ts` | Orchestrator: fetch → filter → summarize → write |
| `packages/core/src/pipeline/html-cache.ts` | Disk cache for paper HTML |

### CLI

| File | Role |
|---|---|
| `apps/cli/src/config.ts` | Loads Node CLI runtime config from JSON and environment variables |
| `apps/cli/src/main.ts` | Minimal Node CLI entrypoint for run, run-pending, and summarize commands |
| `apps/cli/src/runtime.ts` | Builds Node CLI pipeline dependencies from config and host adapters |

### Dashboard

| File | Role |
|---|---|
| `packages/core/src/dashboard/model.ts` | Host-neutral query/filter/sort/stat/action model reused by the Obsidian view |
| `packages/core/src/dashboard/paper-search-index.ts` | Derived in-memory BM25-style index shared by Dashboard search and Similar Papers; no persistence/schema migration |
| `src/dashboard/view.ts` | Obsidian custom view, command/ribbon target, table rendering, filters, row actions, batch operations |
| `src/dashboard/similar-papers-modal.ts` | Local Similar Papers results, deterministic reasons, and existing paper actions |

### Services

| File | Role |
|---|---|
| `packages/core/src/services/paper-index.ts` | Hidden `.index/papers.json` store, schema migration, mark/status/priority/summary/project updates |
| `src/services/paper-note.ts` | Shared lightweight paper-note creation helper |
| `packages/core/src/services/daily-selection.ts` | Daily markdown checkbox parser and sync service |
| `packages/core/src/services/pdf.ts` | Cancellable manual arXiv PDF download into vault storage and `pdfPath` updates |
| `packages/core/src/services/operations.ts` | Shared registry for daily runs, manual detail summaries, and PDF downloads |
| `packages/core/src/services/project-notes.ts` | Project-note append workflow and `projects` field updates |
| `packages/core/src/services/scheduler.ts` | Tick-loop scheduler; `tickToday`, `runForDateNow`, `runAllPending` |
| `packages/core/src/services/state-store.ts` | Per-date `RunStatus` persistence in `.index/run-state.json`; `isDone` includes `"skipped"` |
| `packages/core/src/services/run-lock.ts` | Mutex per-date to prevent double-runs |
| `packages/core/src/services/logger.ts` | Logging with Obsidian `Notice` integration |
| `packages/core/src/services/progress.ts` | `ProgressReporter` interface + `NoopProgressReporter` |
| `src/services/status-bar.ts` | Obsidian status-bar live-progress display |
| `packages/core/src/services/manual-fetch.ts` | `Summarize by arXiv ID` pipeline |
| `src/services/modal.ts` | `chooseModal` — generic multi-button modal (used by enable confirm) |

### Top-level

| File | Role |
|---|---|
| `main.ts` | Plugin lifecycle (`onload`, `setScheduleEnabled`, `buildPipeline`) |
| `src/commands.ts` | Command palette + ribbon menu registrations; config gating on manual commands |
| `packages/core/src/utils/slugify.ts` | `slugify` — topic-name → kebab-case ASCII tag |
| `packages/core/src/utils/time.ts` | Timezone-aware date utilities |
| `packages/core/src/llm/client.ts` | OpenAI-compatible LLM caller with retry-aware provider usage collection |
| `packages/core/src/metrics/generation.ts` | Generation timing/call/token aggregation and folded Markdown callout |
| `packages/core/src/utils/redaction.ts` | Secret redaction for text, URLs, values, and errors |

## Data model (schema v2)

```ts
interface Topic {
  id: string;          // UUID, internal
  name: string;        // display name, e.g. "Photo-z", daily report heading
  tag: string;         // kebab-case slug, auto-derived from name, Obsidian YAML #tag
  description: string; // natural language, sent to the filter LLM
  detail: boolean;     // generate deep-dive report for primary contributions
}

interface ArxivSettings {
  category: string;     // primary category, kept for compatibility
  categories: string[]; // e.g. ["astro-ph", "cs.LG"]
  topics: Topic[];
  timezone: string;
}

type RunStatus = "pending" | "running" | "completed"
               | "failed_transient" | "failed_permanent"
               | "skipped";  // v0.1.2: user opted out at enable time

interface PaperSummary {
  sourceSections?: string;
  coreProblem?: string;
  keyMethod?: string;
  mainResult?: string;
  whyRelevant?: string;
  limitations?: string;
}
```

The `topic.detail` flag replaces v0.1.1's separate `detailCategories` list.
When `detail` is off, even if the LLM says a paper is a primary contribution,
the filter demotes `isDetail` to `false`.

## Migration (v0.1.1 → v0.1.2)

`migrateArxivSettings` runs in `main.ts:loadSettingsAndState`:

- If the persisted `arxiv.topics` is already populated → keep as-is.
- If `arxiv` has legacy `detailCategories` → build one `Topic` per entry:
  `name` from `categoryDisplayMap` (fallback: titleCase), `tag` = entry,
  `description = ""`, `detail = true`.
- If neither is present → empty topics (the new default).
- Free-form `researchInterests` / `detailCriteria` text is **discarded**.
- Legacy keys are dropped from the in-memory `ArxivSettings`; on next
  `saveSettings`, `data.json` is rewritten without them.

## Commands

| Command | Action |
|---|---|
| `arXiv Daily: Run now` | Pulls today, writes daily + papers |
| `arXiv Daily: Run for date…` | Pulls a specific date within the last 5 days |
| `arXiv Daily: Run all pending in lookback window` | Runs every pending date |
| `arXiv Daily: Retry failed dates in lookback window` | Clears failed state for recent failed dates and reruns them |
| `arXiv Daily: Force run for date…` | Clears stored state for one date and runs it without schedule guards |
| `arXiv Daily: Clear run state…` | Clears persisted completed/failed/skipped state without deleting notes |
| `arXiv Daily: Cancel active tasks` | Cooperatively cancels active daily runs, manual detail summaries, and PDF downloads; Get Models is excluded |
| `arXiv Daily: Summarize by arXiv ID…` | Summarize a single paper by ID |
| `arXiv Daily: Set paper mark…` | Updates one indexed paper to Unmarked / Watch / Highlight / Saved / Ignored |
| `arXiv Daily: Create paper note…` | Creates a lightweight note for an indexed paper |
| `arXiv Daily: Mark current paper as <mark>` | Updates the active paper note's indexed mark |
| `arXiv Daily: Open today's daily report` | Opens `<dailyDir>/<today>.md` |
| `arXiv Daily: Open reading dashboard` | Opens the arXiv Daily Dashboard custom view |
| `arXiv Daily: Show recent run state` | Lists last 20 dates and their statuses |
| `arXiv Daily: Show diagnostics` | Shows a copyable local diagnostic report without exposing the API key |

Manual commands gate on config validity:
- Run today / Run all pending / Run for date → requires LLM config + topics
- Summarize by arXiv ID → requires LLM config only

## Daily selections

Daily reports remain markdown files under `<dailyDir>`. Filtered papers are
also indexed in the hidden `<root>/.index/papers.json`, where `<root>` is
derived from the configured daily/papers output directories. Ordinary relevant
papers stay in JSON only; detail papers, saved papers, and manually promoted
papers get markdown notes under `<papersDir>`.

Each paper in a daily report includes two markdown checkboxes:

```markdown
- [ ] 关注 <!-- arxiv-daily:2606.12345:watch -->
- [ ] 重点 <!-- arxiv-daily:2606.12345:highlight -->
```

When the daily file changes, the plugin automatically syncs checked boxes back
to `papers.json`: `关注` becomes `status=to_read, priority=normal`; `重点`
becomes `status=to_read, priority=high`. `papers.json` is an internal state
file for de-duplication and later integrations; daily reports remain the
primary triage surface.

Dashboard treats `priority=high` as `Starred`. The star button only toggles
that priority, so unstarred papers stay neutral in the dashboard. The older
`status` values remain in `papers.json` for compatibility with daily checkbox
sync, saved notes, and existing indexes.

Search and Similar Papers derive one in-memory index from these existing
entries. They do not change the Paper Index schema or persistence, so old
settings, Paper Index files, and Markdown reports remain usable without a
migration.

Generated daily reports and detail notes append a folded `Generation metrics`
callout. Daily reports include total pipeline wall time; both report types show
LLM duration, logical calls, HTTP attempts, and only usage returned by the
provider. Missing usage and retry-incomplete usage are shown as unavailable or
incomplete, never as zero. There is no model-price or cost estimate.

## Scheduling

The plugin uses a single in-process tick loop (default every 20 minutes
while Obsidian is open). On each tick:

1. Walk back over the lookback window (today, yesterday, …, day-4).
2. Skip days that are `completed`, `failed_permanent`, or `skipped`.
3. Skip today if the local clock is still before `runAtLocal`.
4. Skip weekends (Sat/Sun in configured timezone).
5. Skip dates where the daily file already exists.
6. Skip `failed_transient` days whose last attempt is within one tick interval.
7. Otherwise acquire the per-date RunLock, mark running, and run the pipeline.

On Enable: validates config, then asks Run today / Skip today / Cancel.
On Disable: stops the interval. Manual commands always work regardless of
enabled state (subject to config validation).

## Pipeline Error Handling

### No Empty Files
The pipeline no longer writes empty daily files when:
- arXiv returns 0 papers (scheduler will retry later)
- LLM filtering results in 0 relevant papers (calendar shows "0")

### Calendar States
The calendar now shows different states:
- Purple border + number: successful report
- "0": no relevant papers (LLM filtering)
- Green + play icon: runnable (can generate report)
- No mark: arXiv not updated or weekend

### API Testing
- Test API connectivity from settings
- Fetch available models from API
- Select model from dropdown

## Development

Run all commands from the repository root:

```bash
npm ci
npm run check:boundaries
npm run typecheck
npm test
npm run build
npm run cli -- --help
```

`packages/core` is the sole business core; this directory contains only the
Obsidian host and UI. There is no daemon or protocol layer.
