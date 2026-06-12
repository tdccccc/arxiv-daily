# arXiv Daily — Obsidian plugin

Native TypeScript Obsidian plugin. Replaces `arxiv_daily.py` with an
in-vault settings GUI, catch-up scheduling, and on-demand manual runs.

## Features (v0.1.6)

- **Reading Dashboard** — Obsidian custom view for cross-date paper review.
  It reads `.index/papers.json` and supports tabs, search, topic/date/status/
  priority/note/detail/citation/Zotero filters, summary stats, row actions,
  and batch status/priority/note operations.
- **Paper index schema v2** — stores structured daily summary fields
  (`coreProblem`, `keyMethod`, `mainResult`, `whyRelevant`, `limitations`,
  `sourceSections`) for Dashboard search and review.
- **Workflow quick wins** — startup sync for daily checkbox selections,
  missed-paper fallback lists, BibTeX quick copy, and multiple arXiv categories.
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
- **First-enable modal** — `chooseModal` asks Run today / Skip today /
  Cancel on every OFF→ON transition. "Skip today" marks the current
  date as `skipped` in runState so interval ticks leave it alone.
- **Tiered help text** — inline `.setDesc()` on essential fields;
  Obsidian `setTooltip`-backed `(?)` badge on advanced fields; muted
  hint text below topic-card labels.
- One-shot, lossy migration from v0.1.x on first load (`migration.ts`).
- Full vitest suite and production build are expected to pass before release.

## Installation

### Option A — via BRAT (recommended)

1. Install the [BRAT](https://github.com/TfTHacker/obsidian42-brat) plugin.
2. Open BRAT settings → Add Beta plugin → paste `tdccccc/arxiv-daily`.
3. Enable **arXiv Daily** in Community plugins.
4. Open Settings → arXiv Daily, pick a provider and fill in the API Key.

### Option B — manual install

1. Download `manifest.json`, `main.js`, and `styles.css` from the
   [latest release](https://github.com/tdccccc/arxiv-daily/releases).
2. Drop them into `<vault>/.obsidian/plugins/arxiv-daily/`.
3. Enable the plugin in Obsidian.

### Option C — build from source

```bash
git clone https://github.com/tdccccc/arxiv-daily.git
cd arxiv-daily/plugin
npm install
npm run build
# then copy manifest.json + main.js + styles.css into the vault
```

## Settings overview (user-visible)

| Section | Fields |
|---|---|
| **Enable** | Toggle, shows Running / Paused status |
| **LLM** | Provider dropdown, API Key, Base URL, Model, Temperature, Timeout, Thinking mode, Reasoning effort |
| **arXiv** | Category dropdown (grouped), Research Topics (collapsible cards with Name/Tag/Description/Detail toggle), Load Template dropdown, + Add Topic button, Timezone |
| **Output & Schedule** | Daily / Papers paths, Run time, Tick interval, Lookback days (≤ 5) |
| **Advanced** | Request delay, cache TTL, char limits, skip / priority sections, log level |

## Key modules (developer reference)

### Core contracts

| File | Role |
|---|---|
| `src/core/adapters.ts` | Host-neutral contracts for HTTP, storage, secrets, progress, and resource opening |

### Host adapters

| File | Role |
|---|---|
| `src/hosts/obsidian/http-client.ts` | Obsidian `requestUrl` implementation of the core `HttpClient` contract |
| `src/hosts/obsidian/storage-adapter.ts` | Obsidian `Vault` implementation of the core `StorageAdapter` contract |
| `src/hosts/node/*.ts` | Node implementations for HTTP, filesystem storage, env secrets, progress, and resource output |

### Settings layer

| File | Role |
|---|---|
| `src/settings/types.ts` | `Topic`, `ArxivSettings`, `PluginSettings`, `RunStatus` (`"skipped"` added in v0.1.2) |
| `src/settings/defaults.ts` | `DEFAULT_SETTINGS` — topics is `[]` by default |
| `src/settings/migration.ts` | `migrateArxivSettings(raw)` — lossy upgrade from v0.1.x legacy fields |
| `src/settings/validation.ts` | `validateLlmConfig`, `validateFilterConfig` — gatekeeper for all runs |
| `src/settings/topic-templates.ts` | `TOPIC_TEMPLATES` — five static presets |
| `src/settings/tab.ts` | `ArxivDailySettingTab` — full settings UI (topic cards, template loader, validation banner, `attachHelp`, `confirmReplace`) |
| `src/settings/providers.ts` | LLM provider presets |
| `src/settings/arxiv-categories.ts` | Grouped arXiv category data |

### Pipeline

| File | Role |
|---|---|
| `src/pipeline/arxiv-fetcher.ts` | HTTP fetcher for `/recent` and `/abs` |
| `src/pipeline/arxiv-parser.ts` | HTML listing → `PaperMeta[]` |
| `src/pipeline/atom-parser.ts` | Atom API → abstract enrichment |
| `src/pipeline/paper-filter.ts` | LLM-call: classifies papers into topics (or `"skip"`); short-circuits when topics is empty |
| `src/pipeline/paper-content.ts` | Full-text HTML fetch + section extraction |
| `src/pipeline/section-extractor.ts` | Splits HTML body into named sections |
| `src/pipeline/summarizer.ts` | Daily summary + per-paper detail LLM prompts |
| `src/pipeline/markdown-writer.ts` | Writes `.md` files into vault paths; reads tags from `topics` array |
| `src/pipeline/pipeline.ts` | Orchestrator: fetch → filter → summarize → write |
| `src/pipeline/html-cache.ts` | Disk cache for paper HTML |

### Dashboard

| File | Role |
|---|---|
| `src/dashboard/model.ts` | Host-neutral query/filter/sort/stat/action model reused by the Obsidian view and future VS Code Webview |
| `src/dashboard/view.ts` | Obsidian custom view, command/ribbon target, table rendering, filters, row actions, batch operations |

### Services

| File | Role |
|---|---|
| `src/services/paper-index.ts` | Hidden `.index/papers.json` store, schema migration, status/priority/citation/summary updates |
| `src/services/paper-note.ts` | Shared lightweight paper-note creation helper |
| `src/services/daily-selection.ts` | Daily markdown checkbox parser and sync service |
| `src/services/bibtex.ts` | arXiv BibTeX fetch, citation key extraction, `citationKey` update |
| `src/services/scheduler.ts` | Tick-loop scheduler; `tickToday`, `runForDateNow`, `runAllPending` |
| `src/services/state-store.ts` | Per-date `RunStatus` persistence; `isDone` includes `"skipped"` |
| `src/services/run-lock.ts` | Mutex per-date to prevent double-runs |
| `src/services/logger.ts` | Logging with Obsidian `Notice` integration |
| `src/services/progress.ts` | `ProgressReporter` interface + `NoopProgressReporter` |
| `src/services/status-bar.ts` | Obsidian status-bar live-progress display |
| `src/services/manual-fetch.ts` | `Summarize by arXiv ID` pipeline |
| `src/services/modal.ts` | `chooseModal` — generic multi-button modal (used by enable confirm) |

### Top-level

| File | Role |
|---|---|
| `main.ts` | Plugin lifecycle (`onload`, `setScheduleEnabled`, `buildPipeline`) |
| `src/commands.ts` | Command palette + ribbon menu registrations; config gating on manual commands |
| `src/utils/slugify.ts` | `slugify` — topic-name → kebab-case ASCII tag |
| `src/utils/time.ts` | Timezone-aware date utilities |
| `src/llm/client.ts` | OpenAI-compatible LLM caller |

## Data model (v0.1.6)

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
| `arXiv Daily: Summarize by arXiv ID…` | Summarize a single paper by ID |
| `arXiv Daily: Set paper status…` | Updates one indexed paper to to_read/reading/read/saved/ignored |
| `arXiv Daily: Create paper note…` | Creates a lightweight note for an indexed paper |
| `arXiv Daily: Copy BibTeX for current paper` | Copies arXiv BibTeX for the active paper |
| `arXiv Daily: Copy BibTeX by arXiv ID…` | Copies arXiv BibTeX by ID and stores `citationKey` when indexed |
| `arXiv Daily: Mark current paper as <status>` | Updates the active paper note's indexed status |
| `arXiv Daily: Open today's daily report` | Opens `<dailyDir>/<today>.md` |
| `arXiv Daily: Open reading dashboard` | Opens the Reading Dashboard custom view |
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

Paper status values are `inbox`, `to_read`, `reading`, `read`, `saved`, and
`ignored`; `inbox` is the internal default for papers that have appeared in a
daily report but have not been selected. Marking a paper as `saved` creates a
lightweight paper note if one does not already exist.

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

## Development

```bash
cd plugin
npm run dev      # watch build
npm test         # run unit + integration tests (vitest)
npm test:watch   # vitest watcher
npm run build    # production build
```

Type checks: `npx tsc -noEmit -skipLibCheck -p tsconfig.json`
