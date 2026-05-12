# arXiv Topic-Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four coupled arXiv settings (`researchInterests`, `detailCriteria`, `detailCategories`, `categoryTagMap` / `categoryDisplayMap`) with a single `topics: Topic[]` list, surface a topic-card UI with template presets, and migrate existing installs lossily on first load.

**Architecture:** Each `Topic` owns its name, kebab-case tag, natural-language description, and a `detail` boolean. The filter LLM is fed a topic list and returns the chosen tag (or `"skip"`); the summariser derives the display map on the fly; the markdown writer reads tags directly from the topics array. Templates live as a static array in source. Migration runs once inside `loadSettingsAndState`, gated on the absence of `topics` in the loaded data.

**Tech Stack:** TypeScript (strict), Obsidian Plugin API, vitest, esbuild. No new runtime deps.

**Spec:** `docs/superpowers/specs/2026-05-13-arxiv-topics-redesign-design.md`

---

## File Structure

### Create

- `plugin/src/utils/slugify.ts` — pure helper `slugify(input: string): string`.
- `plugin/src/settings/topic-templates.ts` — exported `TopicTemplate[]` constant + `TopicTemplate` type.
- `plugin/src/settings/migration.ts` — pure `migrateArxivSettings(raw: any): ArxivSettings`, extracted so the test does not need to load `main.ts`.
- `plugin/tests/slugify.test.ts` — pure unit tests for slugify.
- `plugin/tests/topic-templates.test.ts` — assert each template's tags are unique, non-empty, slug-shaped.
- `plugin/tests/migration.test.ts` — assert the migration function maps legacy shapes correctly.
- `plugin/tests/paper-filter.test.ts` — new test file for the filter (currently no direct tests).

### Modify

- `plugin/src/settings/types.ts` — add `Topic`; add `topics` to `ArxivSettings`; mark legacy fields optional in Task 2, remove them entirely in Task 9.
- `plugin/src/settings/defaults.ts` — add `topics: Topic[]` next to the legacy default values in Task 2; trim to just the new shape in Task 9.
- `plugin/src/pipeline/paper-filter.ts` — new prompt builder from topics; `"skip"` handling; empty-topics short-circuit.
- `plugin/src/pipeline/summarizer.ts` — derive display map from topics; drop the `other` heading.
- `plugin/src/pipeline/markdown-writer.ts` — read tag straight from the topics array.
- `plugin/main.ts` — call `migrateArxivSettings(...)` inside `loadSettingsAndState`.
- `plugin/src/settings/tab.ts` — replace the four arXiv form rows with a topic-card stack, an `Add Topic` button, and a `Load Template` dropdown.

### Ordering rationale

The shape change is **additive then cleanup** so the codebase stays compilable and the test suite stays green at every commit: Task 2 adds the new fields without removing the old, Tasks 3–8 migrate one consumer at a time, and Task 9 removes the legacy fields once nothing reads them.

---

## Conventions used throughout

- All shell commands assume CWD is the repo root (`/home/tiandc/Documents/code/arxiv-daily`).
- Tests run via `cd plugin && npm test -- <pattern>`; full suite is `cd plugin && npm test`. `tsc` is `cd plugin && npx tsc -noEmit -skipLibCheck`.
- TDD discipline: write the failing test, run it red, implement, run it green, commit.
- Commit messages use the conventional-commit prefix matching recent history (`feat`, `refactor`, `test`, `docs`, `chore`).
- UUIDs come from the global `crypto.randomUUID()` (available in Node ≥ 19 and in Obsidian's Electron).

---

## Task 1: Slugify utility

**Files:**
- Create: `plugin/src/utils/slugify.ts`
- Test: `plugin/tests/slugify.test.ts`

- [ ] **Step 1.1: Write the failing test**

Create `plugin/tests/slugify.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { slugify } from "../src/utils/slugify";

describe("slugify", () => {
  it("lowercases ASCII letters", () => {
    expect(slugify("Photo-z")).toBe("photo-z");
  });

  it("converts spaces to dashes", () => {
    expect(slugify("Galaxy Cluster")).toBe("galaxy-cluster");
  });

  it("converts underscores to dashes", () => {
    expect(slugify("photo_z_methods")).toBe("photo-z-methods");
  });

  it("collapses repeated separators", () => {
    expect(slugify("a   b___c")).toBe("a-b-c");
  });

  it("trims leading and trailing dashes", () => {
    expect(slugify("--hello--")).toBe("hello");
  });

  it("drops non-ASCII characters", () => {
    expect(slugify("Photo-z 相关")).toBe("photo-z");
  });

  it("returns empty string when input has no ASCII alphanumerics", () => {
    expect(slugify("光度红移")).toBe("");
    expect(slugify("")).toBe("");
    expect(slugify("   ")).toBe("");
  });

  it("preserves digits and dots-to-dashes", () => {
    expect(slugify("v0.1.1 release")).toBe("v0-1-1-release");
  });
});
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `cd plugin && npm test -- slugify`
Expected: FAIL with `Cannot find module '../src/utils/slugify'`.

- [ ] **Step 1.3: Implement slugify**

Create `plugin/src/utils/slugify.ts`:

```ts
export function slugify(input: string): string {
  return input
    .toLowerCase()
    .replace(/[\s_.]+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}
```

(Note the `.` is included in the separator class so the test case `v0.1.1 release` produces `v0-1-1-release` rather than `v011-release`.)

- [ ] **Step 1.4: Run test to verify it passes**

Run: `cd plugin && npm test -- slugify`
Expected: PASS, 8 test cases.

- [ ] **Step 1.5: Commit**

```bash
git add plugin/src/utils/slugify.ts plugin/tests/slugify.test.ts
git commit -m "feat(plugin): add slugify utility for topic tag derivation"
```

---

## Task 2: Introduce Topic type and topics field (additive)

This task adds the new `Topic` interface and the `topics` field on `ArxivSettings` alongside the existing legacy fields, which become optional. Defaults gain the same `topics` array next to the still-present legacy values. The codebase continues to compile and all existing tests continue to pass.

**Files:**
- Modify: `plugin/src/settings/types.ts`
- Modify: `plugin/src/settings/defaults.ts`

- [ ] **Step 2.1: Update `types.ts`**

Open `plugin/src/settings/types.ts`. Replace the `ArxivSettings` interface and add the `Topic` interface immediately above it:

```ts
export interface Topic {
  id: string;
  name: string;
  tag: string;
  description: string;
  detail: boolean;
}

export interface ArxivSettings {
  category: string;
  topics: Topic[];
  timezone: string;
  // Legacy — removed in Task 9 once all consumers read from `topics`.
  researchInterests?: string;
  detailCriteria?: string;
  detailCategories?: string[];
  categoryTagMap?: Record<string, string>;
  categoryDisplayMap?: Record<string, string>;
}
```

The other interfaces stay unchanged.

- [ ] **Step 2.2: Update `defaults.ts`**

Open `plugin/src/settings/defaults.ts`. Replace the entire `arxiv:` block with one that holds **both** the legacy values and the new `topics` list:

```ts
  arxiv: {
    category: "astro-ph",
    topics: [
      {
        id: crypto.randomUUID(),
        name: "Photo-z",
        tag: "photo-z",
        description: "Photometric redshift methods, catalogs, comparisons.",
        detail: true,
      },
      {
        id: crypto.randomUUID(),
        name: "Galaxy Cluster",
        tag: "galaxy-cluster",
        description: "Cluster surveys, mass calibration, catalogs, SZ/X-ray/optical.",
        detail: true,
      },
      {
        id: crypto.randomUUID(),
        name: "ML in Astro",
        tag: "ml-astro",
        description: "Deep learning, simulation-based inference (SBI), and related ML/DL applications in astrophysics.",
        detail: false,
      },
    ],
    timezone: "Asia/Shanghai",
    // Legacy — removed in Task 9.
    researchInterests:
      "1. 星系光度红移估计 (photometric redshift / photo-z)：方法、目录、比较\n" +
      "2. 星系团 (galaxy clusters)：搜寻、质量标定、目录、SZ/X-ray/光学巡天\n" +
      "3. 天文中的 ML/DL 应用：深度学习、模拟推断 (SBI) 等",
    detailCriteria:
      "- Photo-z 方法论文（提出或比较 photo-z 方法/目录）\n" +
      "- 星系团巡天/目录/质量标定论文",
    detailCategories: ["photo-z", "galaxy-cluster"],
    categoryTagMap: {
      "photo-z": "photo-z",
      "galaxy-cluster": "galaxy-cluster",
      "ml": "ml",
    },
    categoryDisplayMap: {
      "galaxy-cluster": "Galaxy Cluster 相关",
      "photo-z": "Photo-z 相关",
      "ml": "ML 相关",
      "other": "其他",
    },
  },
```

The rest of `DEFAULT_SETTINGS` (`llm`, `output`, `schedule`, `advanced`) stays unchanged.

- [ ] **Step 2.3: Typecheck and run tests**

Run: `cd plugin && npx tsc -noEmit -skipLibCheck`
Expected: No errors. The legacy consumers (`paper-filter`, `summarizer`, `markdown-writer`, `tab`) still find every field they read.

Run: `cd plugin && npm test`
Expected: All existing tests pass.

- [ ] **Step 2.4: Commit**

```bash
git add plugin/src/settings/types.ts plugin/src/settings/defaults.ts
git commit -m "feat(plugin): add Topic type and topics field alongside legacy arXiv settings"
```

---

## Task 3: Migration helper

Extracted to its own file so the test does not need to load `main.ts` (which would force mocking the entire Obsidian plugin API).

**Files:**
- Create: `plugin/src/settings/migration.ts`
- Test: `plugin/tests/migration.test.ts`

- [ ] **Step 3.1: Write the failing test**

Create `plugin/tests/migration.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { migrateArxivSettings } from "../src/settings/migration";

describe("migrateArxivSettings", () => {
  it("returns the same topics when already in new shape", () => {
    const input = {
      category: "cs.CL",
      topics: [
        { id: "u1", name: "LLM", tag: "llm", description: "x", detail: true },
      ],
      timezone: "UTC",
    };
    const out = migrateArxivSettings(input);
    expect(out.topics).toEqual(input.topics);
    expect(out.category).toBe("cs.CL");
    expect(out.timezone).toBe("UTC");
  });

  it("builds topics from legacy detailCategories + displayMap", () => {
    const legacy = {
      category: "astro-ph",
      researchInterests: "ignored",
      detailCriteria: "ignored",
      detailCategories: ["photo-z", "galaxy-cluster"],
      categoryTagMap: { "photo-z": "photo-z", "galaxy-cluster": "galaxy-cluster" },
      categoryDisplayMap: {
        "photo-z": "Photo-z 相关",
        "galaxy-cluster": "Galaxy Cluster 相关",
        "other": "其他",
      },
      timezone: "Asia/Shanghai",
    };
    const out = migrateArxivSettings(legacy);
    expect(out.topics).toHaveLength(2);
    expect(out.topics[0]).toMatchObject({
      tag: "photo-z",
      name: "Photo-z 相关",
      description: "",
      detail: true,
    });
    expect(out.topics[0].id).toMatch(/^[0-9a-f-]{36}$/i);
    expect(out.topics[1]).toMatchObject({
      tag: "galaxy-cluster",
      name: "Galaxy Cluster 相关",
      description: "",
      detail: true,
    });
  });

  it("falls back to title-case when display map lacks an entry", () => {
    const legacy = {
      category: "astro-ph",
      detailCategories: ["photo-z"],
      categoryDisplayMap: {},
      timezone: "UTC",
    };
    const out = migrateArxivSettings(legacy);
    expect(out.topics[0].name).toBe("Photo Z");
  });

  it("uses defaults when neither topics nor legacy detailCategories are present", () => {
    const out = migrateArxivSettings({ category: "astro-ph", timezone: "UTC" });
    expect(out.topics.length).toBeGreaterThan(0);
    for (const t of out.topics) {
      expect(t.id).toMatch(/^[0-9a-f-]{36}$/i);
    }
  });

  it("never carries legacy fields through to the returned shape", () => {
    const legacy = {
      category: "astro-ph",
      researchInterests: "ABC",
      detailCriteria: "XYZ",
      detailCategories: ["photo-z"],
      categoryDisplayMap: {},
      timezone: "UTC",
    };
    const out = migrateArxivSettings(legacy) as Record<string, unknown>;
    expect(out.researchInterests).toBeUndefined();
    expect(out.detailCriteria).toBeUndefined();
    expect(out.detailCategories).toBeUndefined();
    expect(out.categoryTagMap).toBeUndefined();
    expect(out.categoryDisplayMap).toBeUndefined();
  });

  it("handles null / undefined raw input", () => {
    const out = migrateArxivSettings(undefined);
    expect(out.topics.length).toBeGreaterThan(0);
    expect(out.category.length).toBeGreaterThan(0);
    expect(out.timezone.length).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `cd plugin && npm test -- migration`
Expected: FAIL with `Cannot find module '../src/settings/migration'`.

- [ ] **Step 3.3: Implement migration**

Create `plugin/src/settings/migration.ts`:

```ts
import { DEFAULT_SETTINGS } from "./defaults";
import type { ArxivSettings, Topic } from "./types";

function titleCase(slug: string): string {
  return slug
    .replace(/-/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function freshDefaults(): Topic[] {
  return DEFAULT_SETTINGS.arxiv.topics.map((t) => ({ ...t, id: crypto.randomUUID() }));
}

export function migrateArxivSettings(raw: unknown): ArxivSettings {
  const arxiv = (raw ?? {}) as Record<string, unknown>;

  const category =
    typeof arxiv.category === "string" ? arxiv.category : DEFAULT_SETTINGS.arxiv.category;
  const timezone =
    typeof arxiv.timezone === "string" ? arxiv.timezone : DEFAULT_SETTINGS.arxiv.timezone;

  // New shape already present.
  if (Array.isArray(arxiv.topics) && arxiv.topics.length > 0) {
    return { category, topics: arxiv.topics as Topic[], timezone };
  }

  // Legacy: build from detailCategories + categoryDisplayMap.
  const detailCategories = Array.isArray(arxiv.detailCategories)
    ? (arxiv.detailCategories as string[])
    : [];
  const displayMap =
    (arxiv.categoryDisplayMap as Record<string, string> | undefined) ?? {};

  const topics: Topic[] =
    detailCategories.length > 0
      ? detailCategories.map((tag) => ({
          id: crypto.randomUUID(),
          name: displayMap[tag] ?? titleCase(tag),
          tag,
          description: "",
          detail: true,
        }))
      : freshDefaults();

  return { category, topics, timezone };
}
```

- [ ] **Step 3.4: Run migration test**

Run: `cd plugin && npm test -- migration`
Expected: PASS, 6 test cases.

- [ ] **Step 3.5: Wire migration into `main.ts`**

Edit `plugin/main.ts`. Add the import near the existing settings imports:

```ts
import { migrateArxivSettings } from "./src/settings/migration";
```

Replace the body of `loadSettingsAndState`:

```ts
private async loadSettingsAndState(): Promise<void> {
  const data = ((await this.loadData()) as PersistedData | null) ?? {
    settings: DEFAULT_SETTINGS,
    runState: {},
  };
  const merged = mergeSettings(DEFAULT_SETTINGS, data.settings ?? ({} as PluginSettings));
  merged.arxiv = migrateArxivSettings((data.settings as any)?.arxiv);
  this.settings = merged;
}
```

The migration is fed the **raw** persisted `arxiv` block (not the deep-merged one) so that a v0.1.x install — which has no `topics` field in storage — is correctly detected and rebuilt from its legacy fields. For a fresh install (`loadData()` returns `null`), `data.settings` is `DEFAULT_SETTINGS` which already contains the new shape; migration returns it as-is.

- [ ] **Step 3.6: Run full test suite**

Run: `cd plugin && npm test`
Expected: All tests pass.

- [ ] **Step 3.7: Commit**

```bash
git add plugin/src/settings/migration.ts plugin/tests/migration.test.ts plugin/main.ts
git commit -m "feat(plugin): migrate legacy arXiv settings to topics on load"
```

---

## Task 4: Switch `paper-filter` to topics

**Files:**
- Modify: `plugin/src/pipeline/paper-filter.ts`
- Test: `plugin/tests/paper-filter.test.ts`

- [ ] **Step 4.1: Write the failing test**

Create `plugin/tests/paper-filter.test.ts`:

```ts
import { describe, it, expect, vi } from "vitest";
import { filterPapers } from "../src/pipeline/paper-filter";
import { Logger } from "../src/services/logger";
import type { ArxivSettings, Topic } from "../src/settings/types";
import type { PaperMeta } from "../src/pipeline/arxiv-parser";

vi.mock("obsidian", () => ({
  Notice: class { constructor() {} },
  normalizePath: (p: string) => p,
  requestUrl: vi.fn(),
}));

function makeTopics(): Topic[] {
  return [
    { id: "t1", name: "Photo-z",     tag: "photo-z",        description: "photo-z methods", detail: true },
    { id: "t2", name: "Galaxy",      tag: "galaxy-cluster", description: "cluster surveys", detail: true },
    { id: "t3", name: "ML in Astro", tag: "ml-astro",       description: "ML/DL in astro", detail: false },
  ];
}

function makeArxiv(topics: Topic[]): ArxivSettings {
  return { category: "astro-ph", topics, timezone: "UTC" };
}

const samplePaper: PaperMeta = {
  id: "2601.12345",
  title: "A new photo-z method",
  authors: "X. Author et al.",
  abstract: "We propose ...",
};

describe("filterPapers", () => {
  it("returns [] without calling LLM when topics is empty", async () => {
    const llm = { call: vi.fn() };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv([]),
    });
    expect(out).toEqual([]);
    expect(llm.call).not.toHaveBeenCalled();
  });

  it("includes topic list with [DETAIL] markers in the system prompt", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "photo-z", detail: true }] }),
      ),
    };
    await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    const sys = llm.call.mock.calls[0][0][0].content as string;
    expect(sys).toContain("- photo-z [DETAIL]: photo-z methods");
    expect(sys).toContain("- galaxy-cluster [DETAIL]: cluster surveys");
    expect(sys).toContain("- ml-astro: ML/DL in astro");
    expect(sys).toContain("photo-z|galaxy-cluster|ml-astro|skip");
  });

  it("keeps papers with valid tag and respects detail flag", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "photo-z", detail: true }] }),
      ),
    };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    expect(out).toHaveLength(1);
    expect(out[0].category).toBe("photo-z");
    expect(out[0].isDetail).toBe(true);
  });

  it("demotes detail=true when the chosen topic has detail=false", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "ml-astro", detail: true }] }),
      ),
    };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    expect(out).toHaveLength(1);
    expect(out[0].category).toBe("ml-astro");
    expect(out[0].isDetail).toBe(false);
  });

  it("drops papers with category 'skip'", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "skip", detail: false }] }),
      ),
    };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    expect(out).toEqual([]);
  });

  it("drops papers with an unknown tag", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(
        JSON.stringify({ papers: [{ id: "2601.12345", category: "nope", detail: false }] }),
      ),
    };
    const out = await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    expect(out).toEqual([]);
  });
});
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `cd plugin && npm test -- paper-filter`
Expected: FAIL — the current filter references removed fields (`arxivSettings.categoryDisplayMap`) and crashes.

- [ ] **Step 4.3: Replace filter implementation**

Replace the entire contents of `plugin/src/pipeline/paper-filter.ts`:

```ts
import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import type { ArxivSettings, Topic } from "../settings/types";
import type { PaperMeta } from "./arxiv-parser";

export interface FilteredPaper extends PaperMeta {
  category: string;
  isDetail: boolean;
}

export interface PaperFilterDeps {
  llm: LlmClient;
  logger: Logger;
  arxivSettings: ArxivSettings;
}

export async function filterPapers(
  papers: PaperMeta[],
  deps: PaperFilterDeps,
): Promise<FilteredPaper[]> {
  const { llm, logger, arxivSettings } = deps;
  if (papers.length === 0) return [];

  const topics: Topic[] = arxivSettings.topics ?? [];
  if (topics.length === 0) {
    logger.warn("paper-filter: no topics configured, skipping LLM call");
    return [];
  }

  const topicLines = topics
    .map((t) => `- ${t.tag}${t.detail ? " [DETAIL]" : ""}: ${t.description}`)
    .join("\n");
  const tagOptions = topics.map((t) => t.tag).join("|") + "|skip";
  const validTags = new Set(topics.map((t) => t.tag));
  const topicByTag = new Map(topics.map((t) => [t.tag, t] as const));

  const papersText = papers
    .map((p) => `---\nID: ${p.id}\nTitle: ${p.title}\nAbstract: ${p.abstract}\n`)
    .join("");

  const systemPrompt = `你是一位研究者的助手。请根据下方主题列表，为每篇论文选择最匹配的主题。

## 主题列表
${topicLines}

## 输出格式
请只输出一个 JSON 对象，不要输出任何其他内容：
{"papers": [
  {"id": "YYMM.NNNNN", "category": "${tagOptions}", "detail": true/false},
  ...
]}

规则：
- category 选择最匹配的主题 tag；若与所有主题都不相关，返回 "skip"
- detail 仅在带 [DETAIL] 标记的主题上有意义；当且仅当该论文是该主题的核心贡献时设为 true，其余设为 false
- detail 判定从严：宁可漏选也不要错选——不确定时设为 false
- 如果没有任何相关论文，返回 {"papers": []}`;

  const userContent = `以下是今日 arXiv ${arxivSettings.category} 的所有新论文：\n\n${papersText}`;

  let raw: string;
  try {
    raw = await llm.call(
      [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent },
      ],
      { temperature: 0 },
    );
  } catch (e) {
    logger.error("paper-filter: LLM call failed", e);
    return [];
  }

  let parsed: { papers?: Array<{ id?: string; category?: string; detail?: boolean }> };
  try {
    parsed = JSON.parse(raw);
  } catch {
    const m = /\{[\s\S]*\}/.exec(raw);
    if (!m) {
      logger.error("paper-filter: no JSON in LLM response", raw.slice(0, 200));
      return [];
    }
    try {
      parsed = JSON.parse(m[0]);
    } catch (e) {
      logger.error("paper-filter: JSON parse failed", e);
      return [];
    }
  }

  const idMap = new Map(papers.map((p) => [p.id, p] as const));
  const out: FilteredPaper[] = [];
  for (const item of parsed.papers ?? []) {
    const id = item.id ?? "";
    const meta = idMap.get(id);
    if (!meta) {
      logger.warn(`paper-filter: unknown id ${id}, skipping`);
      continue;
    }
    const category = item.category ?? "";
    if (category === "skip") continue;
    if (!validTags.has(category)) {
      logger.info(`paper-filter: unknown tag '${category}' for ${id}, dropping`);
      continue;
    }
    const topic = topicByTag.get(category)!;
    let isDetail = Boolean(item.detail);
    if (isDetail && !topic.detail) {
      isDetail = false;
      logger.info(`paper-filter: demote detail for ${id} (topic ${category} has detail=false)`);
    }
    out.push({ ...meta, category, isDetail });
  }
  logger.info(`paper-filter: kept ${out.length}/${papers.length} papers`);
  return out;
}
```

- [ ] **Step 4.4: Run paper-filter test**

Run: `cd plugin && npm test -- paper-filter`
Expected: PASS, 6 test cases.

- [ ] **Step 4.5: Run full test suite**

Run: `cd plugin && npm test`
Expected: All tests pass (existing pipeline tests use DEFAULT_SETTINGS which now contains topics).

- [ ] **Step 4.6: Commit**

```bash
git add plugin/src/pipeline/paper-filter.ts plugin/tests/paper-filter.test.ts
git commit -m "refactor(plugin): paper-filter reads topics list and supports 'skip'"
```

---

## Task 5: Switch summariser to topics

**Files:**
- Modify: `plugin/src/pipeline/summarizer.ts`

There is no dedicated summariser unit test; the pipeline test exercises this path end-to-end via DEFAULT_SETTINGS, and is sufficient.

- [ ] **Step 5.1: Replace the display-map derivation**

Edit `plugin/src/pipeline/summarizer.ts`. Inside `callDailyLlm`, locate:

```ts
const categoryList = Object.entries(arxivSettings.categoryDisplayMap)
    .map(([k, v]) => `- ${k} → ${v}`)
    .join("\n");
```

and replace with:

```ts
const categoryList = arxivSettings.topics
  .map((t) => `- ${t.tag} → ${t.name}`)
  .join("\n");
```

- [ ] **Step 5.2: Run pipeline tests**

Run: `cd plugin && npm test -- pipeline`
Expected: PASS.

- [ ] **Step 5.3: Run full test suite**

Run: `cd plugin && npm test`
Expected: All tests pass.

- [ ] **Step 5.4: Commit**

```bash
git add plugin/src/pipeline/summarizer.ts
git commit -m "refactor(plugin): summarizer derives display names from topics array"
```

---

## Task 6: Switch markdown writer to topics

**Files:**
- Modify: `plugin/src/pipeline/markdown-writer.ts`

In the new model, `paper.category` is already the topic tag (the filter assigns it directly), so the writer looks it up by tag.

- [ ] **Step 6.1: Replace `tagsFor`**

Edit `plugin/src/pipeline/markdown-writer.ts`. Replace the `tagsFor` method:

```ts
private tagsFor(paper: DailyPaperWithContent): string[] {
  const tags = ["arxiv", "paper"];
  const t = this.opts.arxiv.categoryTagMap[paper.category];
  if (t) tags.push(t);
  return tags;
}
```

with:

```ts
private tagsFor(paper: DailyPaperWithContent): string[] {
  const tags = ["arxiv", "paper"];
  const topic = this.opts.arxiv.topics.find((t) => t.tag === paper.category);
  if (topic) tags.push(topic.tag);
  return tags;
}
```

- [ ] **Step 6.2: Run all tests**

Run: `cd plugin && npm test`
Expected: All tests pass (existing `markdown-writer.test.ts` uses `DEFAULT_SETTINGS.arxiv` which now has topics).

- [ ] **Step 6.3: Commit**

```bash
git add plugin/src/pipeline/markdown-writer.ts
git commit -m "refactor(plugin): markdown-writer resolves tags via topics array"
```

---

## Task 7: Topic templates

**Files:**
- Create: `plugin/src/settings/topic-templates.ts`
- Test: `plugin/tests/topic-templates.test.ts`

- [ ] **Step 7.1: Write the failing test**

Create `plugin/tests/topic-templates.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { TOPIC_TEMPLATES } from "../src/settings/topic-templates";
import { slugify } from "../src/utils/slugify";

describe("TOPIC_TEMPLATES", () => {
  it("includes a Blank template with no topics", () => {
    const blank = TOPIC_TEMPLATES.find((t) => t.id === "blank");
    expect(blank).toBeDefined();
    expect(blank!.topics).toEqual([]);
  });

  it("every template has a unique id", () => {
    const ids = TOPIC_TEMPLATES.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("every template has a non-empty arXiv category", () => {
    for (const t of TOPIC_TEMPLATES) {
      expect(t.category.length).toBeGreaterThan(0);
    }
  });

  it("every non-blank template has at least one topic", () => {
    for (const t of TOPIC_TEMPLATES) {
      if (t.id === "blank") continue;
      expect(t.topics.length).toBeGreaterThan(0);
    }
  });

  it("topic tags within a template are unique and slug-shaped", () => {
    for (const t of TOPIC_TEMPLATES) {
      const tags = t.topics.map((x) => x.tag);
      expect(new Set(tags).size).toBe(tags.length);
      for (const tag of tags) {
        expect(tag).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
        expect(slugify(tag)).toBe(tag);
      }
    }
  });

  it("every topic has a non-empty name and description", () => {
    for (const t of TOPIC_TEMPLATES) {
      for (const topic of t.topics) {
        expect(topic.name.length).toBeGreaterThan(0);
        expect(topic.description.length).toBeGreaterThan(0);
      }
    }
  });
});
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `cd plugin && npm test -- topic-templates`
Expected: FAIL with `Cannot find module '../src/settings/topic-templates'`.

- [ ] **Step 7.3: Implement templates**

Create `plugin/src/settings/topic-templates.ts`:

```ts
import type { Topic } from "./types";

export interface TopicTemplate {
  id: string;
  name: string;
  category: string;
  topics: Omit<Topic, "id">[];
}

export const TOPIC_TEMPLATES: TopicTemplate[] = [
  {
    id: "blank",
    name: "Blank",
    category: "astro-ph",
    topics: [],
  },
  {
    id: "astro-ml",
    name: "Astrophysics + ML",
    category: "astro-ph",
    topics: [
      { name: "Photo-z",        tag: "photo-z",        description: "Photometric redshift methods, catalogs, comparisons.", detail: true },
      { name: "Galaxy Cluster", tag: "galaxy-cluster", description: "Cluster surveys, mass calibration, catalogs, SZ/X-ray/optical.", detail: true },
      { name: "ML in Astro",    tag: "ml-astro",       description: "Deep learning, simulation-based inference, and related ML/DL applications in astrophysics.", detail: false },
    ],
  },
  {
    id: "nlp",
    name: "NLP / LLMs",
    category: "cs.CL",
    topics: [
      { name: "LLM Training", tag: "llm-training", description: "Pre-training, fine-tuning, RLHF, scaling laws, mixture-of-experts.", detail: true },
      { name: "RAG",          tag: "rag",          description: "Retrieval-augmented generation, vector stores, hybrid retrieval.", detail: true },
      { name: "Alignment",    tag: "alignment",    description: "Safety, interpretability, jailbreaks, constitutional AI.", detail: true },
      { name: "Evaluation",   tag: "eval",         description: "Benchmarks, leaderboards, evaluation methodology, contamination.", detail: false },
    ],
  },
  {
    id: "cv",
    name: "Computer Vision",
    category: "cs.CV",
    topics: [
      { name: "Diffusion", tag: "diffusion", description: "Diffusion-based image / video / 3D generation models.", detail: true },
      { name: "3D Vision", tag: "3d-vision", description: "NeRF, Gaussian splatting, 3D reconstruction, depth estimation.", detail: true },
      { name: "Video",     tag: "video",     description: "Video understanding, generation, action recognition.", detail: false },
    ],
  },
  {
    id: "bio",
    name: "Bioinformatics",
    category: "q-bio",
    topics: [
      { name: "Protein Structure", tag: "protein-structure", description: "Structure prediction, AlphaFold-style models, protein design.", detail: true },
      { name: "Genomics ML",       tag: "genomics-ml",       description: "Foundation models for genomics, single-cell, sequence modeling.", detail: true },
      { name: "Drug Discovery",    tag: "drug-discovery",    description: "Molecular generation, docking, binding affinity prediction.", detail: false },
    ],
  },
];
```

- [ ] **Step 7.4: Run topic-templates test**

Run: `cd plugin && npm test -- topic-templates`
Expected: PASS, 6 test cases.

- [ ] **Step 7.5: Commit**

```bash
git add plugin/src/settings/topic-templates.ts plugin/tests/topic-templates.test.ts
git commit -m "feat(plugin): add topic templates (astro+ML, NLP, CV, bio, blank)"
```

---

## Task 8: Settings tab — topic-card UI

This task makes `tab.ts` compile again by replacing the legacy-field UI with the topic-card UI. There are no unit tests for the tab; the existing `tsc -noEmit` run plus a build + manual smoke check stand in.

**Files:**
- Modify: `plugin/src/settings/tab.ts`

- [ ] **Step 8.1: Update imports**

Open `plugin/src/settings/tab.ts`. Replace the top-of-file import block:

```ts
import { App, PluginSettingTab, Setting } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { PROVIDER_PRESETS, type ProviderPreset } from "./providers";
import { ARXIV_CATEGORIES } from "./arxiv-categories";
```

with:

```ts
import { App, Modal, PluginSettingTab, Setting } from "obsidian";
import type ArxivDailyPlugin from "../../main";
import { PROVIDER_PRESETS, type ProviderPreset } from "./providers";
import { ARXIV_CATEGORIES } from "./arxiv-categories";
import { TOPIC_TEMPLATES } from "./topic-templates";
import type { Topic } from "./types";
import { slugify } from "../utils/slugify";
```

- [ ] **Step 8.2: Replace the arXiv section of `display()`**

In `display()`, locate the section beginning with `// ─── arXiv ────` (around the current line 171) and ending immediately before `// ─── Output & Schedule ────`. Delete everything between those two comment markers (inclusive of the `arXiv` opener, the four legacy form rows, the `Advanced maps` block, the `syncCategoryMaps` invocation, and the `Timezone` setting).

Insert in place:

```ts
    // ─── arXiv ────────────────────────────────────────
    containerEl.createEl("h2", { text: "arXiv" });

    // Category — grouped dropdown + custom text
    new Setting(containerEl)
      .setName("arXiv Category")
      .addDropdown((d) => {
        for (const group of ARXIV_CATEGORIES) {
          const optgroup = d.selectEl.createEl("optgroup");
          optgroup.label = group.label;
          for (const cat of group.categories) {
            const opt = optgroup.createEl("option");
            opt.value = cat.id;
            opt.textContent = `${cat.id} — ${cat.name}`;
          }
        }
        d.setValue(s.arxiv.category).onChange(async (v) => {
          s.arxiv.category = v;
          await this.plugin.saveSettings();
        });
      })
      .addText((t) => {
        t.setPlaceholder("or enter custom category")
          .setValue("")
          .onChange(async (v) => {
            if (v.trim()) {
              s.arxiv.category = v.trim();
              await this.plugin.saveSettings();
            }
          });
      });

    // ─── Research Topics ─────────────────────────────
    containerEl.createEl("h3", { text: "Research Topics" });
    const topicsDesc = containerEl.createEl("div", {
      text: "Each topic becomes one section in the daily report.",
    });
    topicsDesc.style.opacity = "0.7";
    topicsDesc.style.marginBottom = "0.5em";

    const controlsRow = containerEl.createDiv();
    controlsRow.style.display = "flex";
    controlsRow.style.gap = "0.5em";
    controlsRow.style.marginBottom = "0.75em";

    const templateSelect = controlsRow.createEl("select");
    const placeholderOpt = templateSelect.createEl("option");
    placeholderOpt.value = "";
    placeholderOpt.textContent = "Load Template…";
    for (const tpl of TOPIC_TEMPLATES) {
      const opt = templateSelect.createEl("option");
      opt.value = tpl.id;
      opt.textContent = tpl.name;
    }
    templateSelect.onchange = async () => {
      const id = templateSelect.value;
      if (!id) return;
      templateSelect.value = "";
      const tpl = TOPIC_TEMPLATES.find((t) => t.id === id);
      if (!tpl) return;
      const apply = async () => {
        s.arxiv.category = tpl.category;
        s.arxiv.topics = tpl.topics.map((t) => ({ ...t, id: crypto.randomUUID() }));
        await this.plugin.saveSettings();
        this.display();
      };
      if (s.arxiv.topics.length === 0) {
        await apply();
        return;
      }
      const confirm = await this.confirmReplace(
        `Replace your ${s.arxiv.topics.length} topic(s) with the "${tpl.name}" template?`,
      );
      if (confirm) await apply();
    };

    const addBtn = controlsRow.createEl("button", { text: "+ Add Topic" });
    addBtn.onclick = async () => {
      s.arxiv.topics.push({
        id: crypto.randomUUID(),
        name: "",
        tag: `topic-${s.arxiv.topics.length + 1}`,
        description: "",
        detail: false,
      });
      await this.plugin.saveSettings();
      this.display();
    };

    const topicsContainer = containerEl.createDiv();
    for (let i = 0; i < s.arxiv.topics.length; i++) {
      this.renderTopicCard(topicsContainer, s.arxiv.topics, i);
    }

    // Timezone
    new Setting(containerEl)
      .setName("Timezone")
      .addDropdown((d) => {
        const zones = [
          { v: "Asia/Shanghai", l: "Shanghai (UTC+8)" },
          { v: "Asia/Tokyo", l: "Tokyo (UTC+9)" },
          { v: "US/Eastern", l: "US East (UTC-5)" },
          { v: "US/Pacific", l: "US West (UTC-8)" },
          { v: "Europe/London", l: "London (UTC+0)" },
          { v: "Europe/Berlin", l: "Berlin (UTC+1)" },
          { v: "Europe/Moscow", l: "Moscow (UTC+3)" },
          { v: "Australia/Sydney", l: "Sydney (UTC+10)" },
          { v: "UTC", l: "UTC" },
        ];
        for (const z of zones) {
          d.addOption(z.v, z.l);
        }
        d.setValue(s.arxiv.timezone).onChange(async (v) => {
          s.arxiv.timezone = v;
          await this.plugin.saveSettings();
        });
      })
      .addText((t) => {
        t.setPlaceholder("or enter custom timezone")
          .setValue("")
          .onChange(async (v) => {
            if (v.trim()) {
              s.arxiv.timezone = v.trim();
              await this.plugin.saveSettings();
            }
          });
      });
```

- [ ] **Step 8.3: Delete the `syncCategoryMaps` method**

The `syncCategoryMaps` private method (just below `display()`) has no remaining callers. Delete the entire method, signature line through closing brace.

- [ ] **Step 8.4: Add `renderTopicCard` and `confirmReplace` helpers**

Add these two methods to the `ArxivDailySettingTab` class (location: just before the existing `textareaSetting` helper):

```ts
  private renderTopicCard(container: HTMLElement, topics: Topic[], index: number): void {
    const topic = topics[index];
    const card = container.createDiv();
    card.style.border = "1px solid var(--background-modifier-border)";
    card.style.borderRadius = "6px";
    card.style.padding = "0.75em";
    card.style.marginBottom = "0.75em";

    // Name row
    const nameRow = card.createDiv();
    nameRow.style.marginBottom = "0.5em";
    const nameLabel = nameRow.createEl("label", { text: "Name" });
    nameLabel.style.cssText = "display:block;font-weight:600;margin-bottom:0.25em;";
    const nameInput = nameRow.createEl("input", { type: "text" });
    nameInput.value = topic.name;
    nameInput.style.width = "100%";
    nameInput.placeholder = "e.g. Photometric Redshift";

    // Tag row
    const tagRow = card.createDiv();
    tagRow.style.marginBottom = "0.5em";
    const tagLabel = tagRow.createEl("label", { text: "Tag" });
    tagLabel.style.cssText = "display:block;font-weight:600;margin-bottom:0.25em;";
    const tagInput = tagRow.createEl("input", { type: "text" });
    tagInput.value = topic.tag;
    tagInput.style.width = "60%";
    tagInput.placeholder = "kebab-case-slug";
    const autoBadge = tagRow.createEl("span", { text: "  Auto" });
    autoBadge.style.cssText = "opacity:0.5;font-size:0.85em;margin-left:0.5em;";
    const refreshAutoBadge = () => {
      autoBadge.style.display = topic.tag === slugify(topic.name) ? "" : "none";
    };
    refreshAutoBadge();

    // Auto-update tag from name only while the user has not customised the tag.
    nameInput.oninput = async () => {
      const wasAuto = topic.tag === slugify(topic.name);
      topic.name = nameInput.value;
      if (wasAuto) {
        const derived = slugify(topic.name);
        topic.tag = derived || `topic-${index + 1}`;
        tagInput.value = topic.tag;
      }
      refreshAutoBadge();
      await this.plugin.saveSettings();
    };

    tagInput.oninput = async () => {
      topic.tag = tagInput.value;
      refreshAutoBadge();
      await this.plugin.saveSettings();
    };

    // Description row
    const descRow = card.createDiv();
    descRow.style.marginBottom = "0.5em";
    const descLabel = descRow.createEl("label", { text: "Description" });
    descLabel.style.cssText = "display:block;font-weight:600;margin-bottom:0.25em;";
    const descArea = descRow.createEl("textarea");
    descArea.value = topic.description;
    descArea.rows = 3;
    descArea.style.width = "100%";
    descArea.placeholder =
      "What kinds of papers should be grouped under this topic? (natural language)";
    descArea.oninput = async () => {
      topic.description = descArea.value;
      await this.plugin.saveSettings();
    };

    // Footer: detail toggle + delete
    const footer = card.createDiv();
    footer.style.display = "flex";
    footer.style.justifyContent = "space-between";
    footer.style.alignItems = "center";

    const detailLabel = footer.createEl("label");
    detailLabel.style.cursor = "pointer";
    const detailCheckbox = detailLabel.createEl("input", { type: "checkbox" });
    detailCheckbox.checked = topic.detail;
    detailCheckbox.style.marginRight = "0.4em";
    detailLabel.appendText("Detail report");
    detailCheckbox.onchange = async () => {
      topic.detail = detailCheckbox.checked;
      await this.plugin.saveSettings();
    };

    const delBtn = footer.createEl("button", { text: "Delete" });
    delBtn.onclick = async () => {
      topics.splice(index, 1);
      await this.plugin.saveSettings();
      this.display();
    };
  }

  private confirmReplace(message: string): Promise<boolean> {
    return new Promise((resolve) => {
      const modal = new Modal(this.app);
      modal.titleEl.setText("Confirm");
      modal.contentEl.createEl("p", { text: message });
      const btns = modal.contentEl.createDiv();
      btns.style.display = "flex";
      btns.style.justifyContent = "flex-end";
      btns.style.gap = "0.5em";
      btns.style.marginTop = "0.75em";
      const cancel = btns.createEl("button", { text: "Cancel" });
      const ok = btns.createEl("button", { text: "Replace" });
      ok.classList.add("mod-warning");
      let settled = false;
      const finish = (value: boolean) => {
        if (settled) return;
        settled = true;
        resolve(value);
        modal.close();
      };
      cancel.onclick = () => finish(false);
      ok.onclick = () => finish(true);
      modal.onClose = () => finish(false);
      modal.open();
    });
  }
```

- [ ] **Step 8.5: Typecheck**

Run: `cd plugin && npx tsc -noEmit -skipLibCheck`
Expected: No errors. (Legacy fields are still present-as-optional in `ArxivSettings`; Task 9 removes them.)

- [ ] **Step 8.6: Build**

Run: `cd plugin && npm run build`
Expected: build succeeds, `plugin/main.js` regenerated, no error logs.

- [ ] **Step 8.7: Run full test suite**

Run: `cd plugin && npm test`
Expected: All tests pass.

- [ ] **Step 8.8: Commit**

```bash
git add plugin/src/settings/tab.ts
git commit -m "feat(plugin): topic-card UI with templates and add/delete controls"
```

---

## Task 9: Remove legacy fields (cleanup)

With every consumer migrated, the optional legacy fields can leave `ArxivSettings` and `DEFAULT_SETTINGS`. The migration helper (Task 3) reads the raw saved data as `Record<string, unknown>`, so removing the typed legacy fields does not affect it.

**Files:**
- Modify: `plugin/src/settings/types.ts`
- Modify: `plugin/src/settings/defaults.ts`

- [ ] **Step 9.1: Trim `ArxivSettings`**

In `plugin/src/settings/types.ts`, replace the `ArxivSettings` interface with:

```ts
export interface ArxivSettings {
  category: string;
  topics: Topic[];
  timezone: string;
}
```

The `Topic` interface stays unchanged.

- [ ] **Step 9.2: Trim `defaults.ts`**

In `plugin/src/settings/defaults.ts`, replace the entire `arxiv:` block with:

```ts
  arxiv: {
    category: "astro-ph",
    topics: [
      {
        id: crypto.randomUUID(),
        name: "Photo-z",
        tag: "photo-z",
        description: "Photometric redshift methods, catalogs, comparisons.",
        detail: true,
      },
      {
        id: crypto.randomUUID(),
        name: "Galaxy Cluster",
        tag: "galaxy-cluster",
        description: "Cluster surveys, mass calibration, catalogs, SZ/X-ray/optical.",
        detail: true,
      },
      {
        id: crypto.randomUUID(),
        name: "ML in Astro",
        tag: "ml-astro",
        description: "Deep learning, simulation-based inference (SBI), and related ML/DL applications in astrophysics.",
        detail: false,
      },
    ],
    timezone: "Asia/Shanghai",
  },
```

- [ ] **Step 9.3: Typecheck**

Run: `cd plugin && npx tsc -noEmit -skipLibCheck`
Expected: No errors.

- [ ] **Step 9.4: Run full test suite**

Run: `cd plugin && npm test`
Expected: All tests pass.

- [ ] **Step 9.5: Build**

Run: `cd plugin && npm run build`
Expected: build succeeds.

- [ ] **Step 9.6: Commit**

```bash
git add plugin/src/settings/types.ts plugin/src/settings/defaults.ts
git commit -m "refactor(plugin): drop legacy arXiv settings fields"
```

---

## Final verification

After Task 9:

- [ ] **All tests pass:** `cd plugin && npm test`
- [ ] **Typecheck passes:** `cd plugin && npx tsc -noEmit -skipLibCheck`
- [ ] **Build succeeds:** `cd plugin && npm run build`
- [ ] **Git log shows nine focused commits** matching the task list.
- [ ] **Manual smoke (developer):**
  - Launch Obsidian with the dev build.
  - Open the plugin settings panel; the arXiv section shows three default topic cards (Photo-z, Galaxy Cluster, ML in Astro) plus the `+ Add Topic` button and `Load Template` dropdown.
  - Add a topic, type a Chinese-only name; tag falls back to `topic-N`. Edit tag manually; `Auto` badge disappears. Rename name; tag stays.
  - Load the "NLP / LLMs" template (confirm dialog appears, topics replaced, category set to `cs.CL`).
  - Reload Obsidian; settings persist.
  - For an older install whose `data.json` still has `detailCategories`, first load shows topic cards with names from the old `categoryDisplayMap`, empty descriptions, `detail: true`. The legacy keys are gone from `data.json` after the first save triggered by any setting interaction.

---

## Open follow-ups (out of scope for this plan)

- Promoting empty-description topics to a UI warning state.
- Drag-to-reorder topic cards.
- Migration banner / one-time notice (intentionally not built; see spec).
- LLM-assisted "describe your research → generate topics" button.
