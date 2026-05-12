# arXiv Settings Redesign — Topic-Cards Model

**Date:** 2026-05-13
**Scope:** `plugin/` (Obsidian plugin), settings UX overhaul of the arXiv section.

## Background

The current arXiv-related settings require the user to maintain four coupled fields:

| Field | Form | Where it's used |
|---|---|---|
| `researchInterests` | free-form text | filter LLM system prompt — "what to track" |
| `detailCriteria` | free-form text | filter LLM system prompt — "what merits a detail report" |
| `detailCategories` | comma-separated string | LLM bucketing constraint + detail eligibility |
| `categoryDisplayMap` / `categoryTagMap` | JSON, auto-generated, manually overridable | daily-report headings / markdown YAML tags |

Three pain points:

1. **Implicit consistency burden.** The user must keep the natural-language description and the tag list in sync. If `researchInterests` mentions ML but `detailCategories` doesn't include `ml`, those papers land in an unlabelled "other" bucket without any UI warning.
2. **`detail` overloading.** *Detail criteria* (a paper-level depth filter) and *detail categories* (the bucketing tag set) are different concepts that happen to share a word.
3. **High cold-start cost.** A new community user opening the settings panel has to think simultaneously about three layers — what they care about, how to bucket it, which buckets deserve deep reports — before any defaults make sense.

This redesign collapses all four fields into a single list of **research topics**. Each topic is the unit of organisation: it has one description (for the filter LLM), one display name (for the daily report heading), one tag (for the markdown `tags:` YAML), and one flag for "generate a detail report on primary contributions to this topic".

## Requirements (user-visible behaviour)

After this change:

1. The arXiv section of the settings panel shows a **Research Topics** block instead of the four old fields. Each topic appears as a card with: name, tag, description, detail toggle, delete button.
2. The user can **add** new topics with a `+ Add Topic` button and **load templates** from a dropdown of presets (Blank / Astrophysics + ML / NLP / Computer Vision / Bioinformatics).
3. Loading a template into a non-empty topic list shows a confirmation dialog; the template overwrites both `topics` and `arxiv.category`.
4. The `tag` field is auto-derived from `name` on topic creation (kebab-case ASCII slug). It is independently editable afterwards; renaming `name` does not change `tag` once set. A subtle `Auto` indicator next to the tag input disappears once `tag !== slugify(name)`.
5. There is no implicit "Other" bucket. Papers that don't match any topic are silently dropped by the filter (LLM returns `"skip"`). A user who wants a catch-all defines a topic with a broad description.
6. Each paper is assigned to **exactly one** topic. Multi-topic membership is out of scope for this revision.
7. A daily report contains one `## <display name>` section per topic, in topic-list order, with `今日无相关论文更新。` placeholders for topics that received no papers.
8. The filter LLM is told only the topic list (tag, description, detail flag); there is no separate global "detail criteria" text. A paper is marked `detail=true` iff (a) the LLM says it is a primary contribution to its assigned topic and (b) that topic's `detail` flag is on.

## Migration

Existing installations (the developer + early v0.1.x adopters) carry the old settings shape. The migration is **lossy by design** for free-form fields:

- On `loadSettings`, if `settings.arxiv.topics` is missing or empty AND any of the legacy fields are present:
  - Build one `Topic` per entry of `detailCategories`:
    - `id = uuid()`
    - `tag = entry`
    - `name = categoryDisplayMap[entry] ?? titleCase(entry)`
    - `description = ""`
    - `detail = true`
  - **Discard** `researchInterests`, `detailCriteria`, `categoryDisplayMap`, `categoryTagMap`, `detailCategories`.
  - Persist the migrated settings.
- The user will see topics with empty descriptions and is expected to fill them in. A one-line console log records the migration; no in-app banner or notice is shown.

If the user had no `detailCategories` (very unlikely), they end up with `topics: []` and must build their own or load a template.

## Non-goals

- Multi-topic-per-paper.
- An "Other" / catch-all bucket built into the system.
- An in-app banner or wizard for migration.
- LLM-assisted topic generation from a one-liner description (could be a future enhancement; not needed for v1).
- Drag-to-reorder topic cards (the array order is the daily-report section order; for v1 the user reorders by deleting and re-adding).
- Preserving the old field names in `settings.json` for backward roll-back — once migrated, there's no path back via the UI.
- Touching the LLM / output / schedule / advanced sections.

## Block 1 — Data model

### 1a. Type changes

**File:** `plugin/src/settings/types.ts`

```ts
export interface Topic {
  id: string;          // UUID, internal; React-style stable key, never displayed.
  name: string;        // Display name (heading text). Free-form, any language.
  tag: string;         // Kebab-case ASCII slug. Used in markdown YAML tags.
  description: string; // Natural language; injected into the filter LLM prompt.
  detail: boolean;     // If true, primary-contribution papers get a full detail report.
}

export interface ArxivSettings {
  category: string;
  topics: Topic[];
  timezone: string;
}
```

The fields `researchInterests`, `detailCriteria`, `detailCategories`, `categoryTagMap`, `categoryDisplayMap` are removed from `ArxivSettings`.

### 1b. Default settings

**File:** `plugin/src/settings/defaults.ts`

`DEFAULT_SETTINGS.arxiv` becomes:

```ts
arxiv: {
  category: "astro-ph",
  topics: [
    { id: <uuid>, name: "Photo-z",        tag: "photo-z",        description: "Photometric redshift methods, catalogs, comparisons.", detail: true },
    { id: <uuid>, name: "Galaxy Cluster", tag: "galaxy-cluster", description: "Cluster surveys, mass calibration, catalogs, SZ/X-ray/optical.", detail: true },
    { id: <uuid>, name: "ML in Astro",    tag: "ml-astro",       description: "Deep learning, simulation-based inference (SBI), and related ML/DL applications in astrophysics.", detail: false },
  ],
  timezone: "Asia/Shanghai",
},
```

UUIDs are generated at module load via a small helper (`crypto.randomUUID()` is available in the Obsidian Electron environment).

### 1c. Slugify helper

**File:** `plugin/src/utils/slugify.ts` (new)

```ts
export function slugify(input: string): string {
  return input
    .toLowerCase()
    .replace(/[\s_]+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}
```

If the result is empty (e.g. all-Chinese name), callers fall back to `topic-<n>` where `n` is `topics.length + 1` at creation time.

## Block 2 — Settings UI

### 2a. Layout

**File:** `plugin/src/settings/tab.ts`

Replace the existing arXiv block (lines 171–284 in the current source) with:

```
arXiv Category:   [astro-ph                                  ▾]

── Research Topics ─────────────────────────────────────────────
Each topic becomes one section in the daily report.

[Load Template ▾]    [+ Add Topic]

<topic-card>
<topic-card>
...

Timezone:         [Asia/Shanghai (UTC+8)                     ▾]
```

The block uses a custom `<div>` container (not Obsidian's `Setting` row) because each card has more than two controls. The cards are stacked vertically; the container is rendered fresh whenever any of its data mutates (a single `renderTopics()` method on the tab class).

### 2b. Topic card structure

Each card:

```
┌─ Topic ──────────────────────────────────────────────┐
│ Name:           [Photometric Redshift             ]  │
│ Tag:            [photo-z         ] Auto  ✎          │
│ Description:    [photo-z 方法、目录、比较          ]  │
│                 [                                 ]  │
│ Detail report:  [✓] enabled                  [Del]  │
└──────────────────────────────────────────────────────┘
```

- **Name** (single-line text). On blur, if `tag === ""` or `tag` was last set by auto-derive, the tag input updates to `slugify(name)`. Otherwise unchanged.
- **Tag** (single-line text). The small `Auto` chip is shown only when `tag === slugify(name)` and the user has not manually edited the tag for this topic. The `✎` is decorative; the input is always editable.
- **Description** (3-row textarea).
- **Detail report** (toggle).
- **Delete** (small button; immediate, no confirmation).

A small "manually edited" flag on the topic (in-memory only, not persisted) tracks whether the user has typed in the tag field — once set, `Auto` is no longer shown and name-rename does not propagate.

### 2c. Add / Load Template actions

- **`+ Add Topic`**: appends a new topic `{ id: uuid(), name: "", tag: "", description: "", detail: false }` and re-renders. The new card's name input gets focus.
- **`Load Template ▾`**: opens a dropdown of templates (see Block 3). On select:
  - If `topics.length === 0`: apply directly.
  - Else: show an Obsidian modal "Replace your N topics with template &lt;name&gt;? This cannot be undone." with [Replace] / [Cancel].
  - On apply: `topics` becomes the template's topics (each with a fresh UUID), and `arxiv.category` updates to the template's category. Saved and re-rendered.

## Block 3 — Templates

**File:** `plugin/src/settings/topic-templates.ts` (new)

```ts
export interface TopicTemplate {
  id: string;          // template identifier, e.g. "astro-ml"
  name: string;        // display name in the dropdown
  category: string;    // arXiv category to set
  topics: Omit<Topic, "id">[];
}

export const TOPIC_TEMPLATES: TopicTemplate[] = [
  { id: "blank", name: "Blank", category: "astro-ph", topics: [] },
  { id: "astro-ml", name: "Astrophysics + ML", category: "astro-ph", topics: [
    { name: "Photo-z",        tag: "photo-z",        description: "Photometric redshift methods, catalogs, comparisons.", detail: true },
    { name: "Galaxy Cluster", tag: "galaxy-cluster", description: "Cluster surveys, mass calibration, catalogs, SZ/X-ray/optical.", detail: true },
    { name: "ML in Astro",    tag: "ml-astro",       description: "Deep learning, simulation-based inference, and related ML/DL applications in astrophysics.", detail: false },
  ]},
  { id: "nlp", name: "NLP / LLMs", category: "cs.CL", topics: [
    { name: "LLM Training",   tag: "llm-training",   description: "Pre-training, fine-tuning, RLHF, scaling laws, mixture-of-experts.", detail: true },
    { name: "RAG",            tag: "rag",            description: "Retrieval-augmented generation, vector stores, hybrid retrieval.", detail: true },
    { name: "Alignment",      tag: "alignment",      description: "Safety, interpretability, jailbreaks, constitutional AI.", detail: true },
    { name: "Evaluation",     tag: "eval",           description: "Benchmarks, leaderboards, evaluation methodology, contamination.", detail: false },
  ]},
  { id: "cv", name: "Computer Vision", category: "cs.CV", topics: [
    { name: "Diffusion",      tag: "diffusion",      description: "Diffusion-based image / video / 3D generation models.", detail: true },
    { name: "3D Vision",      tag: "3d-vision",      description: "NeRF, Gaussian splatting, 3D reconstruction, depth estimation.", detail: true },
    { name: "Video",          tag: "video",          description: "Video understanding, generation, action recognition.", detail: false },
  ]},
  { id: "bio", name: "Bioinformatics", category: "q-bio", topics: [
    { name: "Protein Structure", tag: "protein-structure", description: "Structure prediction, AlphaFold-style models, protein design.", detail: true },
    { name: "Genomics ML",       tag: "genomics-ml",       description: "Foundation models for genomics, single-cell, sequence modeling.", detail: true },
    { name: "Drug Discovery",    tag: "drug-discovery",    description: "Molecular generation, docking, binding affinity prediction.", detail: false },
  ]},
];
```

Templates are read-only and shipped in source. Adding a new template means editing this file.

## Block 4 — LLM prompt construction

### 4a. Filter prompt

**File:** `plugin/src/pipeline/paper-filter.ts`

If `arxivSettings.topics.length === 0`, short-circuit: log a warning and return `[]` without calling the LLM. This avoids feeding the model an empty topic list.

Otherwise, replace the system prompt construction (current lines 24–57) with:

```ts
const topicLines = arxivSettings.topics.map(t =>
  `- ${t.tag}${t.detail ? " [DETAIL]" : ""}: ${t.description}`,
).join("\n");

const validTags = arxivSettings.topics.map(t => t.tag).join("|");

const systemPrompt = `你是一位研究者的助手。请根据下方主题列表，为每篇论文选择最匹配的主题。

## 主题列表
${topicLines}

## 输出格式
请只输出一个 JSON 对象，不要输出任何其他内容：
{"papers": [
  {"id": "YYMM.NNNNN", "category": "${validTags}|skip", "detail": true/false},
  ...
]}

规则：
- category 选择最匹配的主题 tag；若与所有主题都不相关，返回 "skip"
- detail 仅在 [DETAIL] 主题上有意义；当且仅当该论文是该主题的核心贡献时设为 true，其余设为 false
- detail 判定从严：宁可漏选也不要错选
- 如果没有任何相关论文，返回 {"papers": []}`;
```

The parser drops any paper with `category === "skip"` or with a category not in `validTags`. The `arxivSettings.detailCategories.includes(category)` demote-step is replaced by a lookup into the topic list — if the chosen topic has `detail: false`, force `isDetail = false`.

### 4b. Summarizer prompt

**File:** `plugin/src/pipeline/summarizer.ts`

Replace the `categoryDisplayMap` lookup (current lines 59–61) with:

```ts
const displayMap = Object.fromEntries(arxivSettings.topics.map(t => [t.tag, t.name]));
const categoryList = arxivSettings.topics.map(t => `- ${t.tag} → ${t.name}`).join("\n");
```

The rest of the summarizer is unchanged. The "必须输出所有 category 的二级标题" rule continues to apply, iterating over `arxivSettings.topics` in defined order. There is no longer an `other` heading.

### 4c. Markdown writer

**File:** `plugin/src/pipeline/markdown-writer.ts`

`tagsFor(paper)` currently looks up `categoryTagMap[paper.category]`. Replace with a direct lookup into `topics`:

```ts
private tagsFor(paper: DailyPaperWithContent): string[] {
  const tags = ["arxiv", "paper"];
  const topic = this.opts.arxiv.topics.find(t => t.tag === paper.category);
  if (topic) tags.push(topic.tag);
  return tags;
}
```

Since `paper.category` is already the tag (the filter returns the tag directly), this is equivalent to pushing `paper.category` when it matches a known topic. Kept explicit for safety.

## Block 5 — Migration

**File:** `plugin/main.ts` (or wherever `loadSettings` lives — verify during implementation)

Add a one-shot migration step inside `loadSettings`, after the deep-merge but before returning:

```ts
private migrateArxivSettings(raw: any): ArxivSettings {
  if (Array.isArray(raw.arxiv?.topics) && raw.arxiv.topics.length > 0) {
    return raw.arxiv as ArxivSettings;
  }
  // Build topics from legacy fields
  const legacy = raw.arxiv ?? {};
  const detailCategories: string[] = Array.isArray(legacy.detailCategories) ? legacy.detailCategories : [];
  const displayMap: Record<string, string> = legacy.categoryDisplayMap ?? {};
  const topics: Topic[] = detailCategories.map(tag => ({
    id: crypto.randomUUID(),
    name: displayMap[tag] ?? titleCase(tag),
    tag,
    description: "",
    detail: true,
  }));
  this.logger.info(`migrateArxivSettings: built ${topics.length} topics from legacy fields; researchInterests/detailCriteria discarded`);
  return {
    category: legacy.category ?? DEFAULT_SETTINGS.arxiv.category,
    topics: topics.length > 0 ? topics : DEFAULT_SETTINGS.arxiv.topics.map(t => ({ ...t, id: crypto.randomUUID() })),
    timezone: legacy.timezone ?? DEFAULT_SETTINGS.arxiv.timezone,
  };
}
```

The legacy keys (`researchInterests`, `detailCriteria`, `detailCategories`, `categoryDisplayMap`, `categoryTagMap`) are dropped from the in-memory object. On the next `saveSettings`, `data.json` is rewritten without those keys (a clean cut).

If both `topics` and the old fields exist in `data.json` (e.g. someone hand-edited), `topics` wins.

## Block 6 — Testing

### New tests

- `tests/utils/slugify.test.ts` — Chinese-only input, mixed input, spaces, underscores, dashes, empty result.
- `tests/settings/migration.test.ts` — Cases:
  - Fresh install (no `arxiv` field) → defaults applied.
  - Legacy v0.1.x shape with `detailCategories=["photo-z","galaxy-cluster"]` and `categoryDisplayMap={...}` → 2 topics with names from the map, empty descriptions, `detail: true`.
  - Legacy shape without `detailCategories` → defaults applied (3 topics from `DEFAULT_SETTINGS.arxiv.topics`).
  - New shape already present → unchanged.
- `tests/settings/topic-templates.test.ts` — every template's topics produce non-empty unique tags; categories are non-empty.

### Updated tests

- `tests/pipeline/paper-filter.test.ts` — replace `researchInterests`/`detailCriteria`/`detailCategories` fixtures with `topics`. Add cases for `"skip"` category, unknown-tag drop, and empty `topics` short-circuit.
- `tests/pipeline/summarizer.test.ts` — fixture switches to `topics`; assert no `other` heading is emitted.
- `tests/pipeline/markdown-writer.test.ts` (if present) — `tagsFor()` resolves against topics array.

### Manual smoke checks

After implementation:

1. Fresh-vault install: defaults load, three sample topics show in the panel, daily run produces the expected three sections.
2. Existing-vault upgrade (use the developer's own `plugin_test/` data.json): migration runs, two topics appear with empty descriptions, daily run still produces output (now with empty descriptions the filter LLM may underperform — expected; user is meant to fill them in).
3. Load each template; verify topics and category update.
4. Add a topic with a Chinese name, verify tag falls back to `topic-<n>`; manually fix tag; verify tag stays after renaming the name.

## Open questions for review

- The `Auto` indicator's exact rule (`tag === slugify(name)` versus an explicit "user has typed in tag" flag) — current spec uses the latter, persisted in-memory only. If we want it persistent across reloads, we add a `tagIsCustom` boolean on `Topic`. Default plan: in-memory only; on reload, recompute by comparing strings.
- Whether to log the migration to the Obsidian Notice toast in addition to the console — current spec is console-only. Easy to flip.
- Whether to delete `plugin_test/`'s old-format `data.json` before testing or to keep it as a migration test bed — implementer's call.
