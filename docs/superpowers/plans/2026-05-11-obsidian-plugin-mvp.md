# arxiv-daily Obsidian Plugin MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native TypeScript Obsidian plugin that replaces `arxiv_daily.py`, exposing all configuration through a settings GUI and running on a catch-up loop with a 5-day rolling lookback window via arXiv's `/list/<cat>/recent` endpoint.

**Architecture:** Layered: Settings → Scheduler → Pipeline (Fetch → Parse → Filter → Extract → Summarize → Write) → State. Each layer is one module with one responsibility. Pure logic is unit-tested; Obsidian-specific bits (Vault adapter, settings tab) are integration-tested manually.

**Tech Stack:**
- **Language:** TypeScript (strict)
- **Build:** esbuild (Obsidian plugin template default)
- **Test:** Vitest + happy-dom (provides DOMParser in tests)
- **LLM:** `openai` npm SDK (OpenAI-compatible, supports DeepSeek)
- **HTML parsing:** native `DOMParser` (renderer + happy-dom)
- **HTTP:** Obsidian's `requestUrl` in production; native `fetch` in tests
- **Obsidian API:** `Plugin`, `PluginSettingTab`, `Setting`, `Vault.adapter`, `Notice`

**Spec reference:** `docs/superpowers/specs/2026-05-11-obsidian-plugin-design.md`

**Working directory:** All paths below are relative to the plugin subdirectory `plugin/` inside the repo root, unless prefixed with `<repo>/`.

---

## File Structure

```
arxiv-daily/                              # repo root
├── arxiv_daily.py                        # legacy, untouched
├── docs/superpowers/{specs,plans}/
└── plugin/                               # ← all new code here
    ├── manifest.json                     # Obsidian plugin manifest
    ├── versions.json                     # version → minAppVersion map
    ├── package.json
    ├── tsconfig.json
    ├── esbuild.config.mjs
    ├── vitest.config.ts
    ├── .gitignore
    ├── main.ts                           # plugin entrypoint (lifecycle)
    ├── styles.css                        # optional, empty for v1
    ├── src/
    │   ├── settings/
    │   │   ├── types.ts                  # PluginSettings interface
    │   │   ├── defaults.ts               # DEFAULT_SETTINGS
    │   │   └── tab.ts                    # ArxivDailySettingTab class
    │   ├── services/
    │   │   ├── scheduler.ts              # SchedulerService
    │   │   ├── state-store.ts            # StateStore (state machine)
    │   │   ├── run-lock.ts               # RunLock
    │   │   └── logger.ts                 # Logger (Notice + console)
    │   ├── pipeline/
    │   │   ├── pipeline.ts               # ArxivPipeline orchestrator
    │   │   ├── arxiv-parser.ts           # parse /recent HTML by date
    │   │   ├── arxiv-fetcher.ts          # GET arxiv pages with retry
    │   │   ├── html-cache.ts             # TTL file cache (Electron userData)
    │   │   ├── paper-content.ts          # fetch + extract abstract/sections
    │   │   ├── section-extractor.ts      # extract sections from paper HTML
    │   │   ├── paper-filter.ts           # LLM filter call
    │   │   ├── summarizer.ts             # LLM daily + detail summaries
    │   │   └── markdown-writer.ts        # write daily.md + papers/*.md
    │   ├── llm/
    │   │   └── client.ts                 # OpenAI SDK wrapper
    │   ├── commands.ts                   # command + ribbon registration
    │   └── utils/
    │       ├── time.ts                   # TZ-aware now/today
    │       └── retry.ts                  # async retry with backoff
    └── tests/
        ├── fixtures/
        │   └── arxiv-recent-astroph.html # real snapshot for parser tests
        ├── arxiv-parser.test.ts
        ├── section-extractor.test.ts
        ├── state-store.test.ts
        ├── run-lock.test.ts
        ├── retry.test.ts
        ├── time.test.ts
        ├── pipeline.test.ts
        └── scheduler.test.ts
```

---

## Task Index

| # | Task | Files touched |
|---|---|---|
| 1 | Scaffold plugin project | manifest, package, configs |
| 2 | Settings types & defaults | settings/{types,defaults}.ts |
| 3 | Time utility (TZ-aware) | utils/time.ts |
| 4 | Async retry utility | utils/retry.ts |
| 5 | RunLock | services/run-lock.ts |
| 6 | StateStore | services/state-store.ts |
| 7 | Logger | services/logger.ts |
| 8 | arXiv parser (capture fixture, parse by date) | pipeline/arxiv-parser.ts |
| 9 | arXiv fetcher | pipeline/arxiv-fetcher.ts |
| 10 | HTML cache | pipeline/html-cache.ts |
| 11 | Section extractor | pipeline/section-extractor.ts |
| 12 | Paper content fetcher | pipeline/paper-content.ts |
| 13 | LLM client wrapper | llm/client.ts |
| 14 | Paper filter | pipeline/paper-filter.ts |
| 15 | Daily summarizer (with batching) | pipeline/summarizer.ts |
| 16 | Paper detail summarizer | pipeline/summarizer.ts |
| 17 | Markdown writer | pipeline/markdown-writer.ts |
| 18 | ArxivPipeline orchestrator | pipeline/pipeline.ts |
| 19 | SchedulerService | services/scheduler.ts |
| 20 | Settings UI tab | settings/tab.ts |
| 21 | Commands + ribbon | commands.ts |
| 22 | main.ts lifecycle | main.ts |
| 23 | README + dev docs | plugin/README.md |
| 24 | Manual smoke test | (checklist only) |

---

## Task 1: Scaffold plugin project

**Files:**
- Create: `plugin/manifest.json`
- Create: `plugin/versions.json`
- Create: `plugin/package.json`
- Create: `plugin/tsconfig.json`
- Create: `plugin/esbuild.config.mjs`
- Create: `plugin/vitest.config.ts`
- Create: `plugin/.gitignore`
- Create: `plugin/styles.css` (empty)

- [ ] **Step 1: Create `plugin/manifest.json`**

```json
{
  "id": "arxiv-daily",
  "name": "arXiv Daily",
  "version": "0.1.0",
  "minAppVersion": "1.4.0",
  "description": "Daily arXiv tracker that filters and summarizes papers via LLM into your vault.",
  "author": "Da-Chuan Tian",
  "isDesktopOnly": true
}
```

- [ ] **Step 2: Create `plugin/versions.json`**

```json
{
  "0.1.0": "1.4.0"
}
```

- [ ] **Step 3: Create `plugin/package.json`**

```json
{
  "name": "obsidian-arxiv-daily",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "node esbuild.config.mjs",
    "build": "tsc -noEmit -skipLibCheck && node esbuild.config.mjs production",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "openai": "^4.65.0"
  },
  "devDependencies": {
    "@types/node": "^20.11.30",
    "builtin-modules": "^4.0.0",
    "esbuild": "^0.20.2",
    "happy-dom": "^14.7.1",
    "obsidian": "latest",
    "tslib": "^2.6.2",
    "typescript": "^5.4.3",
    "vitest": "^1.4.0"
  }
}
```

- [ ] **Step 4: Create `plugin/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "resolveJsonModule": true,
    "allowSyntheticDefaultImports": true,
    "forceConsistentCasingInFileNames": true,
    "lib": ["DOM", "ES2020"],
    "types": ["node"],
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["main.ts", "src/**/*.ts", "tests/**/*.ts"]
}
```

- [ ] **Step 5: Create `plugin/esbuild.config.mjs`**

```js
import esbuild from "esbuild";
import process from "process";
import builtins from "builtin-modules";

const prod = process.argv[2] === "production";

const ctx = await esbuild.context({
  entryPoints: ["main.ts"],
  bundle: true,
  external: ["obsidian", "electron", ...builtins],
  format: "cjs",
  target: "es2020",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  outfile: "main.js",
  minify: prod,
});

if (prod) {
  await ctx.rebuild();
  await ctx.dispose();
} else {
  await ctx.watch();
}
```

- [ ] **Step 6: Create `plugin/vitest.config.ts`**

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "happy-dom",
    globals: false,
    include: ["tests/**/*.test.ts"],
  },
  resolve: {
    alias: { "@": "/src" },
  },
});
```

- [ ] **Step 7: Create `plugin/.gitignore`**

```
node_modules/
main.js
*.log
.vitest-cache/
```

- [ ] **Step 8: Create empty `plugin/styles.css`**

```css
/* arXiv Daily plugin styles */
```

- [ ] **Step 9: Create stub `plugin/main.ts` so build can run**

```ts
import { Plugin } from "obsidian";

export default class ArxivDailyPlugin extends Plugin {
  async onload() {
    console.log("arxiv-daily plugin loaded");
  }
  onunload() {
    console.log("arxiv-daily plugin unloaded");
  }
}
```

- [ ] **Step 10: Install deps and verify build**

Run:
```bash
cd plugin && npm install && npm run build
```
Expected: `main.js` is generated; `tsc -noEmit` passes.

- [ ] **Step 11: Commit**

```bash
git add plugin/
git commit -m "feat(plugin): scaffold Obsidian plugin project"
```

---

## Task 2: Settings types & defaults

**Files:**
- Create: `plugin/src/settings/types.ts`
- Create: `plugin/src/settings/defaults.ts`

- [ ] **Step 1: Write `src/settings/types.ts`**

```ts
export interface LlmSettings {
  apiKey: string;
  baseUrl: string;
  model: string;
  temperature: number;
  timeoutMs: number;
  thinkingMode: boolean;
  reasoningEffort: "low" | "medium" | "high";
}

export interface ArxivSettings {
  category: string;
  researchInterests: string;
  detailCriteria: string;
  detailCategories: string[];
  categoryTagMap: Record<string, string>;
  categoryDisplayMap: Record<string, string>;
  timezone: string;
}

export interface OutputSettings {
  dailyDir: string;
  papersDir: string;
}

export interface ScheduleSettings {
  enabled: boolean;
  runAtLocal: string; // "HH:MM"
  tickIntervalMin: number;
  lookbackDays: number;
}

export interface AdvancedSettings {
  requestDelayMs: number;
  cacheExpiryDays: number;
  sectionCharLimit: number;
  paperCharLimit: number;
  dailyCharLimit: number;
  skipSections: string[];
  prioritySections: string[];
  logLevel: "debug" | "info" | "warn" | "error";
}

export interface PluginSettings {
  llm: LlmSettings;
  arxiv: ArxivSettings;
  output: OutputSettings;
  schedule: ScheduleSettings;
  advanced: AdvancedSettings;
}

export type RunStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed_transient"
  | "failed_permanent";

export interface RunStateEntry {
  status: RunStatus;
  lastAttempt: number; // epoch ms
  attempts: number;
  error?: string;
  papersWritten?: number;
}

export type RunState = Record<string, RunStateEntry>; // key = YYYY-MM-DD
```

- [ ] **Step 2: Write `src/settings/defaults.ts`**

```ts
import type { PluginSettings } from "./types";

export const DEFAULT_SETTINGS: PluginSettings = {
  llm: {
    apiKey: "",
    baseUrl: "https://api.deepseek.com/v1",
    model: "deepseek-v4-pro",
    temperature: 0.3,
    timeoutMs: 300_000,
    thinkingMode: true,
    reasoningEffort: "high",
  },
  arxiv: {
    category: "astro-ph",
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
    timezone: "Asia/Shanghai",
  },
  output: {
    dailyDir: "arxiv-daily/daily",
    papersDir: "arxiv-daily/papers",
  },
  schedule: {
    enabled: true,
    runAtLocal: "09:30",
    tickIntervalMin: 20,
    lookbackDays: 5,
  },
  advanced: {
    requestDelayMs: 3000,
    cacheExpiryDays: 7,
    sectionCharLimit: 8000,
    paperCharLimit: 50_000,
    dailyCharLimit: 400_000,
    skipSections: [
      "reference",
      "bibliography",
      "appendix",
      "acknowledgement",
      "acknowledgment",
      "author contribution",
      "data availability",
      "conflict of interest",
      "orcid",
    ],
    prioritySections: ["abstract", "conclusion", "summary"],
    logLevel: "info",
  },
};
```

- [ ] **Step 3: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add plugin/src/settings/
git commit -m "feat(plugin): add settings types and defaults"
```

---

## Task 3: Time utility (TZ-aware)

**Files:**
- Create: `plugin/src/utils/time.ts`
- Test: `plugin/tests/time.test.ts`

- [ ] **Step 1: Write failing test `tests/time.test.ts`**

```ts
import { describe, it, expect } from "vitest";
import { todayInTz, formatDate, parseHHMM, minutesSinceMidnight } from "../src/utils/time";

describe("time utils", () => {
  it("todayInTz returns Asia/Shanghai date for given UTC instant", () => {
    // 2026-05-11 18:00 UTC = 2026-05-12 02:00 Shanghai
    const d = todayInTz(new Date("2026-05-11T18:00:00Z"), "Asia/Shanghai");
    expect(formatDate(d)).toBe("2026-05-12");
  });

  it("todayInTz returns UTC date for UTC tz", () => {
    const d = todayInTz(new Date("2026-05-11T18:00:00Z"), "UTC");
    expect(formatDate(d)).toBe("2026-05-11");
  });

  it("parseHHMM parses HH:MM correctly", () => {
    expect(parseHHMM("09:30")).toEqual({ hour: 9, minute: 30 });
    expect(parseHHMM("23:59")).toEqual({ hour: 23, minute: 59 });
  });

  it("parseHHMM throws on invalid input", () => {
    expect(() => parseHHMM("9:30")).toThrow();
    expect(() => parseHHMM("25:00")).toThrow();
  });

  it("minutesSinceMidnight computes minutes for given tz", () => {
    const d = new Date("2026-05-11T01:30:00Z"); // 09:30 Shanghai
    expect(minutesSinceMidnight(d, "Asia/Shanghai")).toBe(9 * 60 + 30);
  });
});
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd plugin && npx vitest run tests/time.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/utils/time.ts`**

```ts
export function todayInTz(now: Date, tz: string): { y: number; m: number; d: number } {
  const fmt = new Intl.DateTimeFormat("en-CA", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = fmt.formatToParts(now);
  const get = (t: string) => Number(parts.find((p) => p.type === t)!.value);
  return { y: get("year"), m: get("month"), d: get("day") };
}

export function formatDate(d: { y: number; m: number; d: number }): string {
  const mm = String(d.m).padStart(2, "0");
  const dd = String(d.d).padStart(2, "0");
  return `${d.y}-${mm}-${dd}`;
}

export function parseHHMM(s: string): { hour: number; minute: number } {
  const m = /^(\d{2}):(\d{2})$/.exec(s);
  if (!m) throw new Error(`Invalid HH:MM: ${s}`);
  const hour = Number(m[1]);
  const minute = Number(m[2]);
  if (hour > 23 || minute > 59) throw new Error(`Invalid HH:MM: ${s}`);
  return { hour, minute };
}

export function minutesSinceMidnight(now: Date, tz: string): number {
  const fmt = new Intl.DateTimeFormat("en-GB", {
    timeZone: tz,
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
  const parts = fmt.formatToParts(now);
  const hour = Number(parts.find((p) => p.type === "hour")!.value);
  const minute = Number(parts.find((p) => p.type === "minute")!.value);
  return hour * 60 + minute;
}

export function daysBefore(
  date: { y: number; m: number; d: number },
  n: number,
): { y: number; m: number; d: number } {
  const utc = Date.UTC(date.y, date.m - 1, date.d) - n * 86_400_000;
  const dt = new Date(utc);
  return {
    y: dt.getUTCFullYear(),
    m: dt.getUTCMonth() + 1,
    d: dt.getUTCDate(),
  };
}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd plugin && npx vitest run tests/time.test.ts`
Expected: 5 passing.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/utils/time.ts plugin/tests/time.test.ts
git commit -m "feat(plugin): add TZ-aware time utilities with tests"
```

---

## Task 4: Async retry utility

**Files:**
- Create: `plugin/src/utils/retry.ts`
- Test: `plugin/tests/retry.test.ts`

- [ ] **Step 1: Write failing test `tests/retry.test.ts`**

```ts
import { describe, it, expect, vi } from "vitest";
import { retry } from "../src/utils/retry";

describe("retry", () => {
  it("returns value on first success", async () => {
    const fn = vi.fn().mockResolvedValue("ok");
    const result = await retry(fn, { maxAttempts: 3, baseDelayMs: 1 });
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries on failure then succeeds", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("boom"))
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue("ok");
    const result = await retry(fn, { maxAttempts: 3, baseDelayMs: 1 });
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("throws after max attempts", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("permanent"));
    await expect(retry(fn, { maxAttempts: 2, baseDelayMs: 1 })).rejects.toThrow("permanent");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("respects shouldRetry predicate", async () => {
    const err = new Error("4xx");
    const fn = vi.fn().mockRejectedValue(err);
    await expect(
      retry(fn, {
        maxAttempts: 5,
        baseDelayMs: 1,
        shouldRetry: () => false,
      }),
    ).rejects.toThrow("4xx");
    expect(fn).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd plugin && npx vitest run tests/retry.test.ts`
Expected: FAIL (not found).

- [ ] **Step 3: Implement `src/utils/retry.ts`**

```ts
export interface RetryOptions {
  maxAttempts: number;
  baseDelayMs: number;
  backoff?: number; // multiplier, default 2
  shouldRetry?: (err: unknown, attempt: number) => boolean;
  onRetry?: (err: unknown, attempt: number, waitMs: number) => void;
}

export async function retry<T>(fn: () => Promise<T>, opts: RetryOptions): Promise<T> {
  const backoff = opts.backoff ?? 2;
  let lastError: unknown;
  for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt >= opts.maxAttempts) break;
      if (opts.shouldRetry && !opts.shouldRetry(err, attempt)) break;
      const wait = opts.baseDelayMs * Math.pow(backoff, attempt - 1);
      opts.onRetry?.(err, attempt, wait);
      await new Promise((r) => setTimeout(r, wait));
    }
  }
  throw lastError;
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd plugin && npx vitest run tests/retry.test.ts`
Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/utils/retry.ts plugin/tests/retry.test.ts
git commit -m "feat(plugin): add retry utility with tests"
```

---

## Task 5: RunLock

**Files:**
- Create: `plugin/src/services/run-lock.ts`
- Test: `plugin/tests/run-lock.test.ts`

- [ ] **Step 1: Write failing test**

```ts
import { describe, it, expect } from "vitest";
import { RunLock } from "../src/services/run-lock";

describe("RunLock", () => {
  it("first acquire succeeds", () => {
    const lock = new RunLock();
    expect(lock.tryAcquire("2026-05-11")).toBe(true);
  });

  it("second acquire on same key fails", () => {
    const lock = new RunLock();
    expect(lock.tryAcquire("2026-05-11")).toBe(true);
    expect(lock.tryAcquire("2026-05-11")).toBe(false);
  });

  it("release allows re-acquire", () => {
    const lock = new RunLock();
    lock.tryAcquire("k");
    lock.release("k");
    expect(lock.tryAcquire("k")).toBe(true);
  });

  it("different keys are independent", () => {
    const lock = new RunLock();
    expect(lock.tryAcquire("a")).toBe(true);
    expect(lock.tryAcquire("b")).toBe(true);
  });

  it("withLock executes fn and releases on success", async () => {
    const lock = new RunLock();
    const result = await lock.withLock("k", async () => 42);
    expect(result).toBe(42);
    expect(lock.tryAcquire("k")).toBe(true);
  });

  it("withLock releases on error", async () => {
    const lock = new RunLock();
    await expect(lock.withLock("k", async () => { throw new Error("x"); })).rejects.toThrow();
    expect(lock.tryAcquire("k")).toBe(true);
  });

  it("withLock returns undefined if locked", async () => {
    const lock = new RunLock();
    lock.tryAcquire("k");
    const r = await lock.withLock("k", async () => 1);
    expect(r).toBe(undefined);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd plugin && npx vitest run tests/run-lock.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement `src/services/run-lock.ts`**

```ts
export class RunLock {
  private held = new Set<string>();

  tryAcquire(key: string): boolean {
    if (this.held.has(key)) return false;
    this.held.add(key);
    return true;
  }

  release(key: string): void {
    this.held.delete(key);
  }

  isHeld(key: string): boolean {
    return this.held.has(key);
  }

  /**
   * Runs `fn` under the lock for `key`. Returns the fn result, or undefined if the lock is held.
   * Releases the lock whether fn resolves or rejects.
   */
  async withLock<T>(key: string, fn: () => Promise<T>): Promise<T | undefined> {
    if (!this.tryAcquire(key)) return undefined;
    try {
      return await fn();
    } finally {
      this.release(key);
    }
  }
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd plugin && npx vitest run tests/run-lock.test.ts`
Expected: 7 passing.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/services/run-lock.ts plugin/tests/run-lock.test.ts
git commit -m "feat(plugin): add RunLock with tests"
```

---

## Task 6: StateStore

**Files:**
- Create: `plugin/src/services/state-store.ts`
- Test: `plugin/tests/state-store.test.ts`

- [ ] **Step 1: Write failing test**

```ts
import { describe, it, expect, vi } from "vitest";
import { StateStore } from "../src/services/state-store";
import type { RunState } from "../src/settings/types";

function makeStore(initial: RunState = {}) {
  const data: { runState: RunState } = { runState: { ...initial } };
  const load = vi.fn(async () => ({ runState: { ...data.runState } }));
  const save = vi.fn(async (d: { runState: RunState }) => {
    data.runState = { ...d.runState };
  });
  return { store: new StateStore(load, save), data, save };
}

describe("StateStore", () => {
  it("get returns pending when no entry", async () => {
    const { store } = makeStore();
    await store.load();
    expect(store.get("2026-05-11").status).toBe("pending");
    expect(store.get("2026-05-11").attempts).toBe(0);
  });

  it("setRunning marks running with bumped attempts", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    const e = store.get("2026-05-11");
    expect(e.status).toBe("running");
    expect(e.attempts).toBe(1);
  });

  it("setCompleted marks completed and records papers", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 7);
    const e = store.get("2026-05-11");
    expect(e.status).toBe("completed");
    expect(e.papersWritten).toBe(7);
  });

  it("setFailed transient keeps attempts low; permanent after threshold", async () => {
    const { store } = makeStore();
    await store.load();
    for (let i = 0; i < 9; i++) {
      await store.setRunning("d");
      await store.setFailed("d", "transient", "boom");
      expect(store.get("d").status).toBe("failed_transient");
    }
    await store.setRunning("d");
    await store.setFailed("d", "transient", "boom");
    expect(store.get("d").status).toBe("failed_permanent");
  });

  it("setFailed permanent applies immediately", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("d");
    await store.setFailed("d", "permanent", "bad config");
    expect(store.get("d").status).toBe("failed_permanent");
  });

  it("isDone returns true for completed and failed_permanent", async () => {
    const { store } = makeStore();
    await store.load();
    await store.setRunning("a");
    await store.setCompleted("a", 1);
    expect(store.isDone("a")).toBe(true);
    await store.setFailed("b", "permanent", "x");
    expect(store.isDone("b")).toBe(true);
    expect(store.isDone("c")).toBe(false);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd plugin && npx vitest run tests/state-store.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement `src/services/state-store.ts`**

```ts
import type { RunState, RunStateEntry, RunStatus } from "../settings/types";

const MAX_TRANSIENT_ATTEMPTS = 10;

export type StateLoadFn = () => Promise<{ runState: RunState }>;
export type StateSaveFn = (data: { runState: RunState }) => Promise<void>;

export class StateStore {
  private state: RunState = {};

  constructor(
    private readonly loadFn: StateLoadFn,
    private readonly saveFn: StateSaveFn,
  ) {}

  async load(): Promise<void> {
    const data = await this.loadFn();
    this.state = data?.runState ?? {};
  }

  get(date: string): RunStateEntry {
    return (
      this.state[date] ?? {
        status: "pending" as RunStatus,
        lastAttempt: 0,
        attempts: 0,
      }
    );
  }

  isDone(date: string): boolean {
    const s = this.get(date).status;
    return s === "completed" || s === "failed_permanent";
  }

  async setRunning(date: string): Promise<void> {
    const prev = this.get(date);
    this.state[date] = {
      ...prev,
      status: "running",
      lastAttempt: Date.now(),
      attempts: prev.attempts + 1,
    };
    await this.saveFn({ runState: this.state });
  }

  async setCompleted(date: string, papersWritten: number): Promise<void> {
    const prev = this.get(date);
    this.state[date] = {
      ...prev,
      status: "completed",
      lastAttempt: Date.now(),
      papersWritten,
      error: undefined,
    };
    await this.saveFn({ runState: this.state });
  }

  async setFailed(
    date: string,
    kind: "transient" | "permanent",
    message: string,
  ): Promise<void> {
    const prev = this.get(date);
    let status: RunStatus = kind === "permanent" ? "failed_permanent" : "failed_transient";
    if (status === "failed_transient" && prev.attempts >= MAX_TRANSIENT_ATTEMPTS) {
      status = "failed_permanent";
    }
    this.state[date] = {
      ...prev,
      status,
      lastAttempt: Date.now(),
      error: message,
    };
    await this.saveFn({ runState: this.state });
  }

  snapshot(): RunState {
    return { ...this.state };
  }
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd plugin && npx vitest run tests/state-store.test.ts`
Expected: 6 passing.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/services/state-store.ts plugin/tests/state-store.test.ts
git commit -m "feat(plugin): add StateStore with state machine and tests"
```

---

## Task 7: Logger

**Files:**
- Create: `plugin/src/services/logger.ts`

This module wraps Obsidian's `Notice` plus console with leveled filtering. No unit test needed (trivial wrapper); we verify by usage.

- [ ] **Step 1: Implement `src/services/logger.ts`**

```ts
import { Notice } from "obsidian";

export type LogLevel = "debug" | "info" | "warn" | "error";
const ORDER: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

export class Logger {
  constructor(private level: LogLevel = "info") {}

  setLevel(level: LogLevel) {
    this.level = level;
  }

  private allowed(l: LogLevel) {
    return ORDER[l] >= ORDER[this.level];
  }

  debug(msg: string, ...rest: unknown[]) { if (this.allowed("debug")) console.debug("[arxiv-daily]", msg, ...rest); }
  info(msg: string, ...rest: unknown[])  { if (this.allowed("info"))  console.log("[arxiv-daily]", msg, ...rest); }
  warn(msg: string, ...rest: unknown[])  { if (this.allowed("warn"))  console.warn("[arxiv-daily]", msg, ...rest); }
  error(msg: string, ...rest: unknown[]) { if (this.allowed("error")) console.error("[arxiv-daily]", msg, ...rest); }

  /** Show a transient Obsidian toast. timeoutMs=0 means sticky. */
  notice(msg: string, timeoutMs = 5000) {
    new Notice(msg, timeoutMs);
  }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/services/logger.ts
git commit -m "feat(plugin): add Logger wrapping Notice and console"
```

---

## Task 8: arXiv parser

**Files:**
- Create: `plugin/tests/fixtures/arxiv-recent-astroph.html` (manually capture from arXiv)
- Create: `plugin/src/pipeline/arxiv-parser.ts`
- Test: `plugin/tests/arxiv-parser.test.ts`

- [ ] **Step 1: Capture a real fixture**

Run:
```bash
mkdir -p plugin/tests/fixtures
curl -A "Mozilla/5.0" -L "https://arxiv.org/list/astro-ph/recent" -o plugin/tests/fixtures/arxiv-recent-astroph.html
```
Expected: file ~500KB+ with HTML containing multiple `<h3>` date headers.

Verify it has multiple dates: `grep -c '<h3>' plugin/tests/fixtures/arxiv-recent-astroph.html`
Expected: ≥3.

- [ ] **Step 2: Write failing test `tests/arxiv-parser.test.ts`**

```ts
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { parseRecent } from "../src/pipeline/arxiv-parser";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = readFileSync(
  resolve(here, "fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

describe("parseRecent", () => {
  it("returns at least one date bucket", () => {
    const buckets = parseRecent(fixture);
    expect(buckets.length).toBeGreaterThan(0);
  });

  it("each bucket has YYYY-MM-DD date and paper list", () => {
    const buckets = parseRecent(fixture);
    for (const b of buckets) {
      expect(b.announceDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(Array.isArray(b.papers)).toBe(true);
    }
  });

  it("first paper has id/title/authors/abstract", () => {
    const buckets = parseRecent(fixture);
    const p = buckets.find((b) => b.papers.length > 0)!.papers[0];
    expect(p.id).toMatch(/^\d{4}\.\d{4,5}/);
    expect(p.title.length).toBeGreaterThan(0);
    expect(p.authors.length).toBeGreaterThan(0);
  });

  it("buckets are returned in document order (newest first)", () => {
    const buckets = parseRecent(fixture);
    for (let i = 1; i < buckets.length; i++) {
      expect(buckets[i - 1].announceDate >= buckets[i].announceDate).toBe(true);
    }
  });
});
```

- [ ] **Step 3: Run, verify fail**

Run: `cd plugin && npx vitest run tests/arxiv-parser.test.ts`
Expected: FAIL (module missing).

- [ ] **Step 4: Implement `src/pipeline/arxiv-parser.ts`**

```ts
export interface PaperMeta {
  id: string;       // YYMM.NNNNN
  title: string;
  authors: string;  // "First Author et al." or single name
  abstract: string;
}

export interface DateBucket {
  announceDate: string; // YYYY-MM-DD
  papers: PaperMeta[];
}

const MONTHS: Record<string, number> = {
  january: 1, february: 2, march: 3, april: 4, may: 5, june: 6,
  july: 7, august: 8, september: 9, october: 10, november: 11, december: 12,
};

function parseHeaderDate(headerText: string): string | null {
  // "Mon, 11 May 2026 (showing 87 of 87 entries)"
  const m = /(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})/.exec(headerText);
  if (!m) return null;
  const day = Number(m[1]);
  const month = MONTHS[m[2].toLowerCase()];
  const year = Number(m[3]);
  if (!month) return null;
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

/**
 * Parses arXiv /list/<cat>/recent HTML into date-grouped paper buckets.
 *
 * Structure (post-2024 arXiv):
 *   <h3>Mon, 11 May 2026 ...</h3>
 *   <dl>
 *     <dt> <a name="..."> <a title="Abstract" href="/abs/2605.12345">arXiv:2605.12345</a> ... </dt>
 *     <dd> <div class="meta">
 *            <div class="list-title"> Title: ... </div>
 *            <div class="list-authors"> <a>Foo</a> <a>Bar</a> </div>
 *            <p class="mathjax"> abstract text </p>
 *          </div>
 *     </dd>
 *     ...
 *   </dl>
 *   <h3>Fri, 8 May 2026 ...</h3>
 *   ...
 */
export function parseRecent(html: string): DateBucket[] {
  const doc = new DOMParser().parseFromString(html, "text/html");

  // The page may use a single <dl> spanning multiple dates with <h3> interleaved,
  // OR separate <dl> blocks per date. We scan children of the main content in document order.
  // Approach: walk all <h3> and following <dl> until next <h3>.
  const buckets: DateBucket[] = [];
  const h3s = Array.from(doc.querySelectorAll("h3"));
  for (const h3 of h3s) {
    const date = parseHeaderDate(h3.textContent ?? "");
    if (!date) continue;
    const papers: PaperMeta[] = [];

    // Collect <dl> blocks until next <h3>.
    let n: Element | null = h3.nextElementSibling;
    while (n && n.tagName.toLowerCase() !== "h3") {
      if (n.tagName.toLowerCase() === "dl") {
        const dts = Array.from(n.querySelectorAll(":scope > dt"));
        const dds = Array.from(n.querySelectorAll(":scope > dd"));
        const pairs = Math.min(dts.length, dds.length);
        for (let i = 0; i < pairs; i++) {
          const p = parsePaper(dts[i], dds[i]);
          if (p) papers.push(p);
        }
      }
      n = n.nextElementSibling;
    }

    buckets.push({ announceDate: date, papers });
  }

  // Sort descending by date (newest first) just in case
  buckets.sort((a, b) => (a.announceDate > b.announceDate ? -1 : 1));
  return buckets;
}

function parsePaper(dt: Element, dd: Element): PaperMeta | null {
  // Find the "Abstract" link inside dt → arXiv:YYMM.NNNNN
  const absLink = dt.querySelector('a[title="Abstract"]');
  if (!absLink) return null;
  const idText = (absLink.textContent ?? "").replace("arXiv:", "").trim();
  if (!idText) return null;

  const titleDiv = dd.querySelector(".list-title");
  const title = (titleDiv?.textContent ?? "").replace(/^\s*Title:\s*/, "").trim();

  const authorsDiv = dd.querySelector(".list-authors");
  let authors = "Unknown";
  if (authorsDiv) {
    const links = Array.from(authorsDiv.querySelectorAll("a"));
    if (links.length > 0) {
      const first = (links[0].textContent ?? "").trim();
      authors = links.length > 1 ? `${first} et al.` : first;
    } else {
      authors = (authorsDiv.textContent ?? "").replace(/^\s*Authors:\s*/, "").trim();
    }
  }

  const abstractP = dd.querySelector("p.mathjax");
  const abstract = (abstractP?.textContent ?? "").trim();

  return { id: idText, title, authors, abstract };
}
```

- [ ] **Step 5: Run, verify pass**

Run: `cd plugin && npx vitest run tests/arxiv-parser.test.ts`
Expected: 4 passing.

If the fixture's structure differs from the assumption above (arXiv tweaks layouts), inspect with:
```bash
head -200 plugin/tests/fixtures/arxiv-recent-astroph.html | less
```
Adjust selectors in `arxiv-parser.ts` accordingly.

- [ ] **Step 6: Commit**

```bash
git add plugin/src/pipeline/arxiv-parser.ts plugin/tests/arxiv-parser.test.ts plugin/tests/fixtures/
git commit -m "feat(plugin): parse arXiv /recent HTML into date buckets"
```

---

## Task 9: arXiv fetcher

**Files:**
- Create: `plugin/src/pipeline/arxiv-fetcher.ts`

No unit test (thin HTTP wrapper); covered by Pipeline integration test later.

- [ ] **Step 1: Implement `src/pipeline/arxiv-fetcher.ts`**

```ts
import { requestUrl } from "obsidian";
import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";

export interface ArxivFetcherOptions {
  category: string;
  logger: Logger;
  requestDelayMs: number;
}

export class ArxivFetcher {
  private lastRequestAt = 0;

  constructor(private opts: ArxivFetcherOptions) {}

  async fetchRecent(): Promise<string> {
    const url = `https://arxiv.org/list/${this.opts.category}/recent`;
    return this.fetchHtml(url, { allow404: false });
  }

  async fetchPaperHtml(arxivId: string): Promise<{ ok: true; body: string } | { ok: false; status: number }> {
    const url = `https://arxiv.org/html/${arxivId}`;
    try {
      const body = await this.fetchHtml(url, { allow404: true });
      return { ok: true, body };
    } catch (err: any) {
      if (err?.status === 404) return { ok: false, status: 404 };
      throw err;
    }
  }

  async fetchPaperAbsPage(arxivId: string): Promise<string> {
    const url = `https://arxiv.org/abs/${arxivId}`;
    return this.fetchHtml(url, { allow404: false });
  }

  private async fetchHtml(url: string, opts: { allow404: boolean }): Promise<string> {
    await this.respectDelay();
    return retry(
      async () => {
        const res = await requestUrl({
          url,
          method: "GET",
          headers: { "User-Agent": "obsidian-arxiv-daily/0.1" },
          throw: false,
        });
        if (res.status >= 200 && res.status < 300) return res.text;
        if (opts.allow404 && res.status === 404) {
          const e: any = new Error(`HTTP 404: ${url}`);
          e.status = 404;
          throw e;
        }
        throw new Error(`HTTP ${res.status}: ${url}`);
      },
      {
        maxAttempts: 3,
        baseDelayMs: 2000,
        shouldRetry: (err: any) => {
          if (err?.status === 404) return false; // 404 is terminal
          return true;
        },
        onRetry: (err, attempt, wait) =>
          this.opts.logger.warn(`fetch retry #${attempt} after ${wait}ms: ${url}: ${(err as Error).message}`),
      },
    );
  }

  private async respectDelay() {
    const elapsed = Date.now() - this.lastRequestAt;
    const wait = this.opts.requestDelayMs - elapsed;
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    this.lastRequestAt = Date.now();
  }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/pipeline/arxiv-fetcher.ts
git commit -m "feat(plugin): add ArxivFetcher with throttling and retry"
```

---

## Task 10: HTML cache

**Files:**
- Create: `plugin/src/pipeline/html-cache.ts`

Uses Node's `fs/promises` via Electron `app.getPath('userData')`. Lives outside the vault. No unit test (filesystem wrapper); manual smoke test covers it.

- [ ] **Step 1: Implement `src/pipeline/html-cache.ts`**

```ts
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { createHash } from "node:crypto";

export interface HtmlCacheOptions {
  rootDir: string;          // resolved cache root (subdir of userData)
  expiryDays: number;
}

export class HtmlCache {
  constructor(private opts: HtmlCacheOptions) {}

  async get(key: string, kind: "html" | "abs"): Promise<string | null> {
    const p = this.pathFor(key, kind);
    try {
      const stat = await fs.stat(p);
      const ageDays = (Date.now() - stat.mtimeMs) / 86_400_000;
      if (ageDays > this.opts.expiryDays) {
        await fs.unlink(p).catch(() => {});
        return null;
      }
      return await fs.readFile(p, "utf8");
    } catch {
      return null;
    }
  }

  async set(key: string, kind: "html" | "abs", content: string): Promise<void> {
    const p = this.pathFor(key, kind);
    await fs.mkdir(path.dirname(p), { recursive: true });
    await fs.writeFile(p, content, "utf8");
  }

  private pathFor(key: string, kind: "html" | "abs"): string {
    const safe = createHash("sha1").update(key).digest("hex").slice(0, 24);
    return path.join(this.opts.rootDir, kind, `${safe}.html`);
  }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/pipeline/html-cache.ts
git commit -m "feat(plugin): add file-based HTML cache with TTL"
```

---

## Task 11: Section extractor

**Files:**
- Create: `plugin/src/pipeline/section-extractor.ts`
- Test: `plugin/tests/section-extractor.test.ts`

This ports Python's `_extract_sections` and `_extract_abstract_conclusion`.

- [ ] **Step 1: Write failing test**

```ts
import { describe, it, expect } from "vitest";
import { extractAbstractConclusion, extractSections } from "../src/pipeline/section-extractor";

const sample = `
<html><body>
<div class="ltx_abstract">This is the abstract content with key findings.</div>
<h2>Introduction</h2><p>intro text body here</p>
<h2>Methods</h2><p>methods body</p>
<h2>Conclusions</h2><p>final remarks summary</p>
<h2>References</h2><p>[1] paper</p>
<h2>Appendix A</h2><p>extra</p>
</body></html>
`;

describe("section-extractor", () => {
  it("extractAbstractConclusion finds abstract and conclusion sections", () => {
    const out = extractAbstractConclusion(sample, { sectionCharLimit: 8000 });
    expect(out).toContain("## Abstract");
    expect(out).toContain("abstract content");
    expect(out).toContain("## Conclusions");
    expect(out).toContain("final remarks");
  });

  it("extractSections includes priority + body, skips refs/appendix", () => {
    const out = extractSections(sample, {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
      skipSections: ["reference", "appendix", "bibliography"],
      prioritySections: ["abstract", "conclusion", "summary"],
    });
    expect(out).toContain("## Introduction");
    expect(out).toContain("## Methods");
    expect(out).toContain("## Conclusions");
    expect(out).not.toContain("## References");
    expect(out).not.toContain("## Appendix");
  });

  it("extractSections returns null when no useful sections", () => {
    const out = extractSections("<html><body><p>no headings here</p></body></html>", {
      sectionCharLimit: 8000,
      paperCharLimit: 50000,
      skipSections: [],
      prioritySections: [],
    });
    expect(out).toBeNull();
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd plugin && npx vitest run tests/section-extractor.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement `src/pipeline/section-extractor.ts`**

```ts
export interface AbstractConclusionOpts { sectionCharLimit: number; }

export interface ExtractSectionsOpts {
  sectionCharLimit: number;
  paperCharLimit: number;
  skipSections: string[];
  prioritySections: string[];
}

function parse(html: string): Document {
  return new DOMParser().parseFromString(html, "text/html");
}

function stripNoise(doc: Document) {
  for (const tag of ["script", "style", "nav", "footer", "figure", "table"]) {
    for (const el of Array.from(doc.querySelectorAll(tag))) {
      el.parentNode?.removeChild(el);
    }
  }
}

function textBetween(start: Element): string {
  const parts: string[] = [];
  let n: Element | null = start.nextElementSibling;
  while (n && !/^h[2-4]$/i.test(n.tagName)) {
    const t = (n.textContent ?? "").replace(/\s+/g, " ").trim();
    if (t) parts.push(t);
    n = n.nextElementSibling;
  }
  return parts.join("\n");
}

export function extractAbstractConclusion(
  html: string,
  opts: AbstractConclusionOpts,
): string | null {
  const doc = parse(html);
  stripNoise(doc);
  const sections: string[] = [];

  const abstractDiv = doc.querySelector("div.ltx_abstract");
  if (abstractDiv) {
    const txt = (abstractDiv.textContent ?? "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, opts.sectionCharLimit);
    if (txt) sections.push(`## Abstract\n${txt}`);
  }

  const headers = Array.from(doc.querySelectorAll("h2, h3, h4"));
  for (const h of headers) {
    const title = (h.textContent ?? "").trim();
    const lower = title.toLowerCase();
    if (!/conclusion|summary/.test(lower)) continue;
    const body = textBetween(h).slice(0, opts.sectionCharLimit);
    if (body) sections.push(`## ${title}\n${body}`);
  }

  return sections.length ? sections.join("\n\n") : null;
}

export function extractSections(html: string, opts: ExtractSectionsOpts): string | null {
  const doc = parse(html);
  stripNoise(doc);
  const headers = Array.from(doc.querySelectorAll("h2, h3, h4"));
  if (headers.length === 0) return null;

  // First pass: collect non-skipped sections
  type S = { title: string; body: string; priority: boolean };
  const all: S[] = [];
  for (const h of headers) {
    const title = (h.textContent ?? "").trim();
    const lower = title.toLowerCase();
    if (opts.skipSections.some((s) => lower.includes(s.toLowerCase()))) continue;
    const body = textBetween(h).slice(0, opts.sectionCharLimit);
    if (!body) continue;
    const priority = opts.prioritySections.some((s) => lower.includes(s.toLowerCase()));
    all.push({ title, body, priority });
  }
  if (all.length === 0) return null;

  // Second pass: reserve budget for priority sections; fill normal until budget; preserve original order
  const reserved = all.filter((s) => s.priority).reduce((sum, s) => sum + s.body.length, 0);
  let budget = opts.paperCharLimit - reserved;
  const order = new Map(all.map((s, i) => [s.title, i]));
  const selected: S[] = [];
  let used = 0;
  for (const s of all) {
    if (s.priority) continue;
    if (used + s.body.length > budget) {
      const remaining = budget - used;
      if (remaining > 500) selected.push({ ...s, body: s.body.slice(0, remaining) });
      break;
    }
    selected.push(s);
    used += s.body.length;
  }
  const merged = [...selected, ...all.filter((s) => s.priority)];
  merged.sort((a, b) => (order.get(a.title) ?? 999) - (order.get(b.title) ?? 999));
  return merged.length ? merged.map((s) => `## ${s.title}\n${s.body}`).join("\n\n") : null;
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd plugin && npx vitest run tests/section-extractor.test.ts`
Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/pipeline/section-extractor.ts plugin/tests/section-extractor.test.ts
git commit -m "feat(plugin): port section extractor with priority budget"
```

---

## Task 12: Paper content fetcher

**Files:**
- Create: `plugin/src/pipeline/paper-content.ts`

Combines fetcher + cache + extractor. No standalone test (covered by Pipeline integration test).

- [ ] **Step 1: Implement `src/pipeline/paper-content.ts`**

```ts
import type { ArxivFetcher } from "./arxiv-fetcher";
import type { HtmlCache } from "./html-cache";
import type { Logger } from "../services/logger";
import {
  extractAbstractConclusion,
  extractSections,
  type ExtractSectionsOpts,
} from "./section-extractor";

export interface PaperContent {
  abstractConclusion: string;
  fullSections: string | null;
}

export interface PaperContentOpts {
  isDetail: boolean;
  sectionCharLimit: number;
  paperCharLimit: number;
  skipSections: string[];
  prioritySections: string[];
}

export class PaperContentFetcher {
  constructor(
    private fetcher: ArxivFetcher,
    private cache: HtmlCache,
    private logger: Logger,
  ) {}

  async fetch(arxivId: string, opts: PaperContentOpts): Promise<PaperContent> {
    // 1. HTML full-text path (cached)
    const htmlKey = `html/${arxivId}`;
    let html = await this.cache.get(htmlKey, "html");
    if (!html) {
      const res = await this.fetcher.fetchPaperHtml(arxivId);
      if (res.ok) {
        html = res.body;
        await this.cache.set(htmlKey, "html", html);
      }
    }

    if (html) {
      const ac = extractAbstractConclusion(html, {
        sectionCharLimit: opts.sectionCharLimit,
      });
      const sectionsOpts: ExtractSectionsOpts = {
        sectionCharLimit: opts.sectionCharLimit,
        paperCharLimit: opts.paperCharLimit,
        skipSections: opts.skipSections,
        prioritySections: opts.prioritySections,
      };
      const fs = opts.isDetail ? extractSections(html, sectionsOpts) : null;
      if (ac) return { abstractConclusion: ac, fullSections: fs };
      // fallback: strip tags
      const plain = html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").slice(0, opts.paperCharLimit);
      return { abstractConclusion: plain, fullSections: fs };
    }

    // 2. Fallback to /abs page
    const absKey = `abs/${arxivId}`;
    let abs = await this.cache.get(absKey, "abs");
    if (!abs) {
      try {
        abs = await this.fetcher.fetchPaperAbsPage(arxivId);
        await this.cache.set(absKey, "abs", abs);
      } catch (e) {
        this.logger.error(`paper-content: abs fetch failed ${arxivId}`, e);
        return {
          abstractConclusion: `[获取失败] arXiv ID: ${arxivId}`,
          fullSections: null,
        };
      }
    }
    const doc = new DOMParser().parseFromString(abs, "text/html");
    const bq = doc.querySelector("blockquote.abstract");
    const text = (bq?.textContent ?? "").replace(/^\s*Abstract:?\s*/, "").trim() || "N/A";
    return { abstractConclusion: `## Abstract\n${text}`, fullSections: null };
  }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/pipeline/paper-content.ts
git commit -m "feat(plugin): add paper content fetcher (HTML + abs fallback, cached)"
```

---

## Task 13: LLM client wrapper

**Files:**
- Create: `plugin/src/llm/client.ts`

Wraps `openai` SDK with streaming, retry, and optional thinking-mode (DeepSeek). No standalone test (mock at Pipeline level).

- [ ] **Step 1: Implement `src/llm/client.ts`**

```ts
import OpenAI from "openai";
import { retry } from "../utils/retry";
import type { Logger } from "../services/logger";
import type { LlmSettings } from "../settings/types";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CallOptions {
  /** Overrides settings.temperature. Ignored if thinkingMode = true. */
  temperature?: number;
}

export class LlmClient {
  private client: OpenAI;

  constructor(private settings: LlmSettings, private logger: Logger) {
    this.client = new OpenAI({
      apiKey: settings.apiKey,
      baseURL: settings.baseUrl,
      timeout: settings.timeoutMs,
      maxRetries: 0,
      dangerouslyAllowBrowser: true, // Obsidian runs in Electron renderer
    });
  }

  async call(messages: ChatMessage[], opts: CallOptions = {}): Promise<string> {
    return retry(
      async () => {
        const params: Record<string, unknown> = {
          model: this.settings.model,
          messages,
          stream: true,
        };
        if (this.settings.thinkingMode) {
          params.reasoning_effort = this.settings.reasoningEffort;
          (params as any).extra_body = { thinking: { type: "enabled" } };
        } else {
          params.temperature = opts.temperature ?? this.settings.temperature;
        }
        const stream = await this.client.chat.completions.create(params as any);
        const chunks: string[] = [];
        for await (const chunk of stream as any) {
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) chunks.push(delta);
        }
        return chunks.join("");
      },
      {
        maxAttempts: 3,
        baseDelayMs: 5000,
        onRetry: (err, attempt, wait) =>
          this.logger.warn(`LLM retry #${attempt} after ${wait}ms: ${(err as Error).message}`),
      },
    );
  }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/llm/client.ts
git commit -m "feat(plugin): add LLM client wrapping openai SDK with streaming + retry"
```

---

## Task 14: Paper filter

**Files:**
- Create: `plugin/src/pipeline/paper-filter.ts`

Ports Python's `llm_filter_papers`. No standalone test (covered by Pipeline test with mocked LLM).

- [ ] **Step 1: Implement `src/pipeline/paper-filter.ts`**

```ts
import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import type { ArxivSettings } from "../settings/types";
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

  const categories = Object.keys(arxivSettings.categoryDisplayMap);
  const categoryOptions = categories.length
    ? categories.join("|")
    : "photo-z|galaxy-cluster|ml|other";

  const papersText = papers
    .map(
      (p) =>
        `---\nID: ${p.id}\nTitle: ${p.title}\nAbstract: ${p.abstract}\n`,
    )
    .join("");

  const systemPrompt = `你是一位研究者的助手。请根据研究兴趣，从下方论文列表中筛选出相关论文。

## 研究兴趣
${arxivSettings.researchInterests}

## 详细收录标准
以下类型的论文应标记 detail: true（会生成详细报告）：
${arxivSettings.detailCriteria}

## 输出格式
请只输出一个 JSON 对象，不要输出任何其他内容：
{"papers": [
  {"id": "YYMM.NNNNN", "category": "${categoryOptions}", "detail": true/false},
  ...
]}

规则：
- 只收录与研究兴趣相关的论文，不相关的直接忽略
- category 从 ${categoryOptions} 中选择最匹配的一个
- detail 判定要从严：只有核心主题直接匹配详细收录标准时才设为 true
- 宁可漏选 detail 也不要错选——不确定时设为 false，日报已包含所有相关论文的总结
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
    const category = item.category ?? "other";
    let isDetail = Boolean(item.detail);
    if (isDetail && !arxivSettings.detailCategories.includes(category)) {
      isDetail = false;
      logger.info(`paper-filter: demote detail for ${id} (category=${category})`);
    }
    out.push({ ...meta, category, isDetail });
  }
  logger.info(`paper-filter: kept ${out.length}/${papers.length} papers`);
  return out;
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/pipeline/paper-filter.ts
git commit -m "feat(plugin): port LLM paper filter to TypeScript"
```

---

## Task 15: Daily summarizer (with batching)

**Files:**
- Create: `plugin/src/pipeline/summarizer.ts` (this task adds the daily summarizer; Task 16 adds detail).

- [ ] **Step 1: Implement daily portion of `src/pipeline/summarizer.ts`**

```ts
import type { LlmClient } from "../llm/client";
import type { Logger } from "../services/logger";
import type { ArxivSettings, AdvancedSettings } from "../settings/types";
import type { FilteredPaper } from "./paper-filter";

export interface DailyPaperWithContent extends FilteredPaper {
  abstractConclusion: string;
  fullSections: string | null;
}

export interface SummarizerDeps {
  llm: LlmClient;
  logger: Logger;
  arxivSettings: ArxivSettings;
  advanced: AdvancedSettings;
  llmTemperature: number;
}

function buildPaperBlock(p: DailyPaperWithContent): string {
  const detailMark = p.isDetail ? ` → [[${p.id}]]` : "";
  return (
    `=== Paper: ${p.id} [category: ${p.category}]${detailMark} ===\n` +
    `Title: ${p.title}\n` +
    `Authors: ${p.authors}\n` +
    `${p.abstractConclusion}\n\n`
  );
}

function splitBatches(papers: DailyPaperWithContent[], charLimit: number): DailyPaperWithContent[][] {
  const batches: DailyPaperWithContent[][] = [];
  let cur: DailyPaperWithContent[] = [];
  let size = 0;
  for (const p of papers) {
    const bs = buildPaperBlock(p).length;
    if (cur.length && size + bs > charLimit) {
      batches.push(cur);
      cur = [];
      size = 0;
    }
    cur.push(p);
    size += bs;
  }
  if (cur.length) batches.push(cur);
  return batches;
}

async function callDailyLlm(
  papers: DailyPaperWithContent[],
  dateStr: string,
  nTotal: number,
  nDetail: number,
  isPartial: boolean,
  deps: SummarizerDeps,
): Promise<string> {
  const { llm, arxivSettings, llmTemperature } = deps;
  const categoryList = Object.entries(arxivSettings.categoryDisplayMap)
    .map(([k, v]) => `- ${k} → ${v}`)
    .join("\n");
  const papersInfo = papers.map(buildPaperBlock).join("");
  const partialNote = isPartial
    ? `\n注意：这是分批处理的一部分（本批 ${papers.length} 篇），请只为本批论文生成总结，不要输出标题头和统计行。\n`
    : "";
  const headerFmt = isPartial
    ? ""
    : `# arXiv ${arxivSettings.category} 每日追踪 ${dateStr}\n` +
      `共 ${nTotal} 篇相关论文，其中 ${nDetail} 篇详细收录。\n\n`;

  const systemPrompt = `你是一个专业的研究助手。请根据提供的论文摘要与结论，生成 arXiv 每日论文追踪日报。

## Category 与显示名称对应关系
${categoryList}
${partialNote}
请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，直接输出内容）：

${headerFmt}## [显示名称]
### <实际论文标题> → [[YYMM.NNNNN]]
- **作者**: First Author et al.
- **arXiv**: [ID](https://arxiv.org/abs/ID)
- **一句话总结**: 用一句话概括本文做了什么
- **数据**: 使用了什么数据集/样本/巡天（2-4句）
- **方法**: 采用了什么方法或模型，关键技术细节是什么（2-4句）
- **主要结果**: 核心发现是什么，给出关键定量数值（精度、误差、提升幅度等），与已有工作的对比（2-4句）
- **意义**: 对领域的贡献或启示，局限性，未来展望（1-2句）

注意：
- 所有论文（无论是否详细收录）都必须按上述完整格式输出，包含五个字段，不得省略或只列标题
- 使用中文撰写，保留关键英文术语
- 数学公式必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$
- 必须输出所有 category 的二级标题（使用上面的显示名称），如果某个 category 今日无论文，在标题下写"今日无相关论文更新。"
- 标题后带 → [[YYMM.NNNNN]] 的论文为详细收录论文，请保留此标记
- 未标记的论文不要加 [[]] 链接
- 重点提取定量结果，避免泛泛而谈`;

  return llm.call(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: `以下是今日筛选出的论文：\n\n${papersInfo}` },
    ],
    { temperature: llmTemperature },
  );
}

export async function summarizeDaily(
  papers: DailyPaperWithContent[],
  dateStr: string,
  deps: SummarizerDeps,
): Promise<string> {
  const nTotal = papers.length;
  const nDetail = papers.filter((p) => p.isDetail).length;
  const totalChars = papers.reduce((s, p) => s + buildPaperBlock(p).length, 0);
  deps.logger.info(`summarizeDaily: ${totalChars} chars (limit ${deps.advanced.dailyCharLimit})`);

  if (totalChars <= deps.advanced.dailyCharLimit) {
    return callDailyLlm(papers, dateStr, nTotal, nDetail, false, deps);
  }

  const batches = splitBatches(papers, deps.advanced.dailyCharLimit);
  deps.logger.info(`summarizeDaily: batching into ${batches.length} (${batches.map((b) => b.length).join(",")})`);
  const header =
    `# arXiv ${deps.arxivSettings.category} 每日追踪 ${dateStr}\n` +
    `共 ${nTotal} 篇相关论文，其中 ${nDetail} 篇详细收录。\n`;
  const parts: string[] = [header];
  for (let i = 0; i < batches.length; i++) {
    deps.logger.info(`summarizeDaily: batch ${i + 1}/${batches.length}`);
    parts.push(await callDailyLlm(batches[i], dateStr, nTotal, nDetail, true, deps));
  }
  return parts.join("\n\n");
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/pipeline/summarizer.ts
git commit -m "feat(plugin): port daily summarizer with batching"
```

---

## Task 16: Paper detail summarizer

**Files:**
- Modify: `plugin/src/pipeline/summarizer.ts` (add detail function)

- [ ] **Step 1: Append `summarizePaperDetail` to `src/pipeline/summarizer.ts`**

Add to the file:

```ts
export async function summarizePaperDetail(
  paper: DailyPaperWithContent,
  deps: SummarizerDeps,
): Promise<string> {
  if (!paper.fullSections) {
    throw new Error(`summarizePaperDetail: paper ${paper.id} has no full sections`);
  }

  const systemPrompt = `你是一个专业的研究助手。请根据提供的论文各章节内容，生成一篇详细的中文论文总结。

请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，不要输出 YAML frontmatter，直接从 # 标题开始）：

# ${paper.title}

- **arXiv**: [${paper.id}](https://arxiv.org/abs/${paper.id})

## 背景与动机
（研究背景、前人工作、本文动机）

## 数据
（使用了什么数据集、样本大小、数据处理方法）

## 方法
（核心方法/模型/算法的详细描述）

## 结果
（主要发现、定量结果、与前人工作的比较）

## 讨论
（结果的意义、局限性、与其他工作的对比）

## 结论
（核心结论、未来展望）

注意：
- 使用中文撰写
- 保留关键英文术语（如专有名词、物理量）
- 数学公式、物理量和符号必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$
- 尽可能包含定量结果（数值、误差）
- 如果某个章节的信息不足，可以简要说明`;

  const userContent =
    `论文 ID: ${paper.id}\n` +
    `标题: ${paper.title}\n` +
    `作者: ${paper.authors}\n\n` +
    `以下是论文各章节内容：\n\n${paper.fullSections}`;

  return deps.llm.call(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userContent },
    ],
    { temperature: deps.llmTemperature },
  );
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/pipeline/summarizer.ts
git commit -m "feat(plugin): add detail paper summarizer"
```

---

## Task 17: Markdown writer

**Files:**
- Create: `plugin/src/pipeline/markdown-writer.ts`

Writes via `Vault.adapter` (cross-platform). No unit test (covered manually).

- [ ] **Step 1: Implement `src/pipeline/markdown-writer.ts`**

```ts
import { type Vault, normalizePath } from "obsidian";
import type { Logger } from "../services/logger";
import type { ArxivSettings, OutputSettings } from "../settings/types";
import type { DailyPaperWithContent } from "./summarizer";

export interface MarkdownWriterOpts {
  vault: Vault;
  logger: Logger;
  arxiv: ArxivSettings;
  output: OutputSettings;
}

export class MarkdownWriter {
  constructor(private opts: MarkdownWriterOpts) {}

  async writeDaily(dateStr: string, summary: string): Promise<string> {
    const path = normalizePath(`${this.opts.output.dailyDir}/${dateStr}.md`);
    await this.ensureDir(this.opts.output.dailyDir);
    await this.backupIfExists(path);
    const frontmatter = `---\ndate: ${dateStr}\ntags: [arxiv, daily]\n---\n\n`;
    await this.opts.vault.adapter.write(path, frontmatter + summary);
    this.opts.logger.info(`wrote daily: ${path}`);
    return path;
  }

  async writePaperDetail(
    paper: DailyPaperWithContent,
    dateStr: string,
    summary: string,
  ): Promise<string> {
    const path = normalizePath(`${this.opts.output.papersDir}/${paper.id}.md`);
    await this.ensureDir(this.opts.output.papersDir);
    await this.backupIfExists(path);
    const tags = this.tagsFor(paper);
    const fm =
      `---\n` +
      `title: "${escapeYaml(paper.title)}"\n` +
      `authors: "${escapeYaml(paper.authors)}"\n` +
      `arxiv: "${paper.id}"\n` +
      `date: ${dateStr}\n` +
      `tags: [${tags.join(", ")}]\n` +
      `---\n\n`;
    await this.opts.vault.adapter.write(path, fm + summary);
    this.opts.logger.info(`wrote paper: ${path}`);
    return path;
  }

  async writeEmptyDaily(dateStr: string): Promise<string> {
    const summary =
      `# arXiv ${this.opts.arxiv.category} 每日追踪 ${dateStr}\n\n今日未发现相关论文。\n`;
    return this.writeDaily(dateStr, summary);
  }

  private tagsFor(paper: DailyPaperWithContent): string[] {
    const tags = ["arxiv", "paper"];
    const t = this.opts.arxiv.categoryTagMap[paper.category];
    if (t) tags.push(t);
    return tags;
  }

  private async ensureDir(rel: string): Promise<void> {
    const norm = normalizePath(rel);
    if (!(await this.opts.vault.adapter.exists(norm))) {
      await this.opts.vault.adapter.mkdir(norm);
    }
  }

  private async backupIfExists(path: string): Promise<void> {
    if (await this.opts.vault.adapter.exists(path)) {
      const bak = path.replace(/\.md$/, ".bak.md");
      if (await this.opts.vault.adapter.exists(bak)) {
        await this.opts.vault.adapter.remove(bak);
      }
      await this.opts.vault.adapter.rename(path, bak);
      this.opts.logger.info(`backed up existing file → ${bak}`);
    }
  }
}

function escapeYaml(s: string): string {
  return s.replace(/"/g, '\\"');
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/pipeline/markdown-writer.ts
git commit -m "feat(plugin): add MarkdownWriter using Vault adapter"
```

---

## Task 18: ArxivPipeline orchestrator

**Files:**
- Create: `plugin/src/pipeline/pipeline.ts`
- Test: `plugin/tests/pipeline.test.ts`

- [ ] **Step 1: Implement `src/pipeline/pipeline.ts`**

```ts
import type { Logger } from "../services/logger";
import type { ArxivSettings, AdvancedSettings, OutputSettings, LlmSettings } from "../settings/types";
import type { ArxivFetcher } from "./arxiv-fetcher";
import type { PaperContentFetcher } from "./paper-content";
import type { MarkdownWriter } from "./markdown-writer";
import type { LlmClient } from "../llm/client";
import { parseRecent, type DateBucket } from "./arxiv-parser";
import { filterPapers } from "./paper-filter";
import { summarizeDaily, summarizePaperDetail, type DailyPaperWithContent } from "./summarizer";

export type PipelineResult =
  | { kind: "completed"; papersWritten: number }
  | { kind: "failed_transient"; reason: string }
  | { kind: "failed_permanent"; reason: string };

export interface PipelineDeps {
  fetcher: ArxivFetcher;
  paperFetcher: PaperContentFetcher;
  writer: MarkdownWriter;
  llm: LlmClient;
  logger: Logger;
  arxiv: ArxivSettings;
  advanced: AdvancedSettings;
  output: OutputSettings;
  llmSettings: LlmSettings;
}

export class ArxivPipeline {
  constructor(private deps: PipelineDeps) {}

  async runForDate(dateStr: string): Promise<PipelineResult> {
    const { fetcher, logger } = this.deps;
    logger.info(`pipeline: start for ${dateStr}`);

    // 1. Fetch /recent
    let recentHtml: string;
    try {
      recentHtml = await fetcher.fetchRecent();
    } catch (e) {
      return { kind: "failed_transient", reason: `fetch /recent failed: ${(e as Error).message}` };
    }

    // 2. Parse and find bucket
    let buckets: DateBucket[];
    try {
      buckets = parseRecent(recentHtml);
    } catch (e) {
      return { kind: "failed_permanent", reason: `parse failed: ${(e as Error).message}` };
    }
    const bucket = buckets.find((b) => b.announceDate === dateStr);
    if (!bucket) {
      return {
        kind: "failed_transient",
        reason: `date ${dateStr} not in /recent (have: ${buckets.map((b) => b.announceDate).join(",")})`,
      };
    }
    logger.info(`pipeline: ${bucket.papers.length} papers for ${dateStr}`);

    // 3. Empty day
    if (bucket.papers.length === 0) {
      await this.deps.writer.writeEmptyDaily(dateStr);
      return { kind: "completed", papersWritten: 0 };
    }

    // 4. LLM filter
    const filtered = await filterPapers(bucket.papers, {
      llm: this.deps.llm,
      logger,
      arxivSettings: this.deps.arxiv,
    });
    if (filtered.length === 0) {
      await this.deps.writer.writeEmptyDaily(dateStr);
      return { kind: "completed", papersWritten: 0 };
    }

    // 5. Fetch content for each filtered paper
    const enriched: DailyPaperWithContent[] = [];
    for (const p of filtered) {
      try {
        const c = await this.deps.paperFetcher.fetch(p.id, {
          isDetail: p.isDetail,
          sectionCharLimit: this.deps.advanced.sectionCharLimit,
          paperCharLimit: this.deps.advanced.paperCharLimit,
          skipSections: this.deps.advanced.skipSections,
          prioritySections: this.deps.advanced.prioritySections,
        });
        enriched.push({ ...p, abstractConclusion: c.abstractConclusion, fullSections: c.fullSections });
      } catch (e) {
        logger.error(`pipeline: content fetch failed for ${p.id}`, e);
        enriched.push({
          ...p,
          abstractConclusion: `[获取失败] arXiv ID: ${p.id}`,
          fullSections: null,
        });
      }
    }

    // 6. Daily summary
    let dailySummary: string;
    try {
      dailySummary = await summarizeDaily(enriched, dateStr, {
        llm: this.deps.llm,
        logger,
        arxivSettings: this.deps.arxiv,
        advanced: this.deps.advanced,
        llmTemperature: this.deps.llmSettings.temperature,
      });
    } catch (e) {
      return { kind: "failed_transient", reason: `daily summary LLM failed: ${(e as Error).message}` };
    }
    await this.deps.writer.writeDaily(dateStr, dailySummary);

    // 7. Detail papers
    const detailPapers = enriched.filter((p) => p.isDetail && p.fullSections);
    for (const p of detailPapers) {
      logger.info(`pipeline: detail report for ${p.id}`);
      try {
        const detail = await summarizePaperDetail(p, {
          llm: this.deps.llm,
          logger,
          arxivSettings: this.deps.arxiv,
          advanced: this.deps.advanced,
          llmTemperature: this.deps.llmSettings.temperature,
        });
        await this.deps.writer.writePaperDetail(p, dateStr, detail);
      } catch (e) {
        logger.error(`pipeline: detail failed for ${p.id}`, e);
      }
    }

    return { kind: "completed", papersWritten: enriched.length };
  }
}
```

- [ ] **Step 2: Write integration test `tests/pipeline.test.ts`**

```ts
import { describe, it, expect, vi } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { ArxivPipeline } from "../src/pipeline/pipeline";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

// Mock obsidian.Notice for happy-dom env
vi.mock("obsidian", () => ({ Notice: class { constructor() {} }, normalizePath: (p: string) => p }));

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(resolve(here, "fixtures/arxiv-recent-astroph.html"), "utf8");

function makeDeps(opts: { firstDate: string }) {
  const writes: Record<string, string> = {};
  const fetcher = {
    fetchRecent: vi.fn().mockResolvedValue(recentHtml),
    fetchPaperHtml: vi.fn().mockResolvedValue({ ok: false, status: 404 }),
    fetchPaperAbsPage: vi.fn().mockResolvedValue(
      `<html><body><blockquote class="abstract">Abstract: stub abstract</blockquote></body></html>`,
    ),
  };
  const paperFetcher = {
    fetch: vi.fn().mockResolvedValue({ abstractConclusion: "## Abstract\nstub", fullSections: null }),
  };
  const writer = {
    writeDaily: vi.fn(async (date: string, content: string) => {
      writes[`daily/${date}.md`] = content;
      return `daily/${date}.md`;
    }),
    writePaperDetail: vi.fn(async (p: any, date: string, content: string) => {
      writes[`papers/${p.id}.md`] = content;
      return `papers/${p.id}.md`;
    }),
    writeEmptyDaily: vi.fn(async (date: string) => {
      writes[`daily/${date}.md`] = "empty";
      return `daily/${date}.md`;
    }),
  };
  const llm = {
    call: vi
      .fn()
      // first call = filter; return empty list to keep the test fast
      .mockResolvedValueOnce(JSON.stringify({ papers: [] })),
  };
  const logger = new Logger("error");
  return { writes, fetcher, paperFetcher, writer, llm, logger };
}

describe("ArxivPipeline", () => {
  it("returns failed_transient when date not in /recent", async () => {
    const d = makeDeps({ firstDate: "2026-05-11" });
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const result = await pipeline.runForDate("1999-01-01");
    expect(result.kind).toBe("failed_transient");
  });

  it("writes empty daily when LLM returns no relevant papers", async () => {
    const d = makeDeps({ firstDate: "" });
    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    // Find a real date present in the fixture so the bucket lookup succeeds:
    const m = /(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})/.exec(recentHtml);
    expect(m).toBeTruthy();
    const months: Record<string, number> = { January:1, February:2, March:3, April:4, May:5, June:6, July:7, August:8, September:9, October:10, November:11, December:12 };
    const date = `${m![3]}-${String(months[m![2]]).padStart(2,"0")}-${String(Number(m![1])).padStart(2,"0")}`;
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    expect((result as any).papersWritten).toBe(0);
    expect(d.writer.writeEmptyDaily).toHaveBeenCalled();
  });
});
```

- [ ] **Step 3: Run, verify pass**

Run: `cd plugin && npx vitest run tests/pipeline.test.ts`
Expected: 2 passing.

- [ ] **Step 4: Commit**

```bash
git add plugin/src/pipeline/pipeline.ts plugin/tests/pipeline.test.ts
git commit -m "feat(plugin): orchestrate fetch→filter→content→summarize→write"
```

---

## Task 19: SchedulerService

**Files:**
- Create: `plugin/src/services/scheduler.ts`
- Test: `plugin/tests/scheduler.test.ts`

- [ ] **Step 1: Write failing test**

```ts
import { describe, it, expect, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { Logger } from "../src/services/logger";
import { StateStore } from "../src/services/state-store";
import { RunLock } from "../src/services/run-lock";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

vi.mock("obsidian", () => ({ Notice: class {} }));

function makeStore() {
  const data = { runState: {} as Record<string, any> };
  return new StateStore(
    async () => ({ runState: { ...data.runState } }),
    async (d) => {
      data.runState = { ...d.runState };
    },
  );
}

describe("SchedulerService", () => {
  it("does not run before runAtLocal time", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 0 });
    const settings = { ...DEFAULT_SETTINGS, schedule: { ...DEFAULT_SETTINGS.schedule, runAtLocal: "23:59" } };
    const svc = new SchedulerService({
      getSettings: () => settings,
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T00:00:00Z"), // 08:00 Shanghai
    });
    await svc.tick();
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("runs today after runAtLocal", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 3 });
    const settings = { ...DEFAULT_SETTINGS, schedule: { ...DEFAULT_SETTINGS.schedule, runAtLocal: "00:01" } };
    const svc = new SchedulerService({
      getSettings: () => settings,
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"), // 13:00 Shanghai
    });
    await svc.tick();
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(store.get("2026-05-11").status).toBe("completed");
  });

  it("skips dates already completed", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 5);
    const lock = new RunLock();
    const runForDate = vi.fn();
    const settings = { ...DEFAULT_SETTINGS };
    const svc = new SchedulerService({
      getSettings: () => settings,
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    await svc.tick();
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("respects failed_transient backoff", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setFailed("2026-05-11", "transient", "x");
    const lock = new RunLock();
    const runForDate = vi.fn();
    const settings = { ...DEFAULT_SETTINGS };
    const svc = new SchedulerService({
      getSettings: () => settings,
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date(Date.now()), // immediately after
    });
    await svc.tick();
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("runForDateNow ignores scheduled-time gate", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 2 });
    const settings = { ...DEFAULT_SETTINGS, schedule: { ...DEFAULT_SETTINGS.schedule, runAtLocal: "23:59" } };
    const svc = new SchedulerService({
      getSettings: () => settings,
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T00:00:00Z"),
    });
    await svc.runForDateNow("2026-05-11");
    expect(runForDate).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement `src/services/scheduler.ts`**

```ts
import type { Logger } from "./logger";
import type { StateStore } from "./state-store";
import type { RunLock } from "./run-lock";
import type { PluginSettings } from "../settings/types";
import { todayInTz, formatDate, parseHHMM, minutesSinceMidnight, daysBefore } from "../utils/time";
import type { PipelineResult } from "../pipeline/pipeline";

export interface SchedulerDeps {
  getSettings: () => PluginSettings;
  store: StateStore;
  lock: RunLock;
  runForDate: (date: string) => Promise<PipelineResult>;
  logger: Logger;
  now?: () => Date;
}

export class SchedulerService {
  private intervalHandle: number | null = null;

  constructor(private deps: SchedulerDeps) {}

  start(): void {
    const min = this.deps.getSettings().schedule.tickIntervalMin;
    this.stop();
    this.intervalHandle = window.setInterval(() => {
      this.tick().catch((e) => this.deps.logger.error("scheduler tick failed", e));
    }, Math.max(1, min) * 60_000);
    // also run once on start
    this.tick().catch((e) => this.deps.logger.error("scheduler initial tick failed", e));
  }

  stop(): void {
    if (this.intervalHandle != null) {
      window.clearInterval(this.intervalHandle);
      this.intervalHandle = null;
    }
  }

  async tick(): Promise<void> {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) return;
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();

    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    const minutesNow = minutesSinceMidnight(now, tz);
    const t = parseHHMM(s.schedule.runAtLocal);
    const scheduledMin = t.hour * 60 + t.minute;

    for (let i = 0; i < s.schedule.lookbackDays; i++) {
      const date = formatDate(daysBefore(todayObj, i));
      const entry = this.deps.store.get(date);
      if (this.deps.store.isDone(date)) continue;
      if (entry.status === "running") continue;

      if (date === today && minutesNow < scheduledMin) continue;

      if (entry.status === "failed_transient") {
        const tickMs = s.schedule.tickIntervalMin * 60_000;
        if (now.getTime() - entry.lastAttempt < tickMs) continue;
      }

      await this.tryRun(date);
    }
  }

  /** Manual trigger: ignore scheduled time gate, still respect lock and isDone. */
  async runForDateNow(date: string): Promise<PipelineResult | { kind: "skipped"; reason: string }> {
    const entry = this.deps.store.get(date);
    if (entry.status === "running") {
      return { kind: "skipped", reason: "already running" };
    }
    return (await this.tryRun(date)) ?? { kind: "skipped", reason: "lock held" };
  }

  private async tryRun(date: string): Promise<PipelineResult | undefined> {
    return this.deps.lock.withLock(date, async () => {
      await this.deps.store.setRunning(date);
      let result: PipelineResult;
      try {
        result = await this.deps.runForDate(date);
      } catch (e) {
        result = { kind: "failed_transient", reason: (e as Error).message };
      }
      if (result.kind === "completed") {
        await this.deps.store.setCompleted(date, result.papersWritten);
        this.deps.logger.notice(`arXiv ${date}: ${result.papersWritten} papers written`);
      } else if (result.kind === "failed_transient") {
        await this.deps.store.setFailed(date, "transient", result.reason);
        this.deps.logger.warn(`arXiv ${date} transient: ${result.reason}`);
      } else {
        await this.deps.store.setFailed(date, "permanent", result.reason);
        this.deps.logger.error(`arXiv ${date} permanent: ${result.reason}`);
        this.deps.logger.notice(`arXiv ${date}: failed (${result.reason})`, 10_000);
      }
      return result;
    });
  }
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts`
Expected: 5 passing.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/services/scheduler.ts plugin/tests/scheduler.test.ts
git commit -m "feat(plugin): add SchedulerService with catch-up loop"
```

---

## Task 20: Settings UI tab

**Files:**
- Create: `plugin/src/settings/tab.ts`

No unit test (Obsidian-bound UI; verified manually).

- [ ] **Step 1: Implement `src/settings/tab.ts`**

```ts
import { App, PluginSettingTab, Setting } from "obsidian";
import type ArxivDailyPlugin from "../../main";

export class ArxivDailySettingTab extends PluginSettingTab {
  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app, plugin);
  }

  display(): void {
    const { containerEl } = this;
    const s = this.plugin.settings;
    containerEl.empty();

    // ─── LLM ──────────────────────────────────────────
    containerEl.createEl("h2", { text: "LLM 配置" });

    new Setting(containerEl)
      .setName("API Key")
      .setDesc("OpenAI 兼容 API Key（DeepSeek、OpenAI、其他）")
      .addText((t) =>
        t.inputEl.type = "password",
      )
      .addText((t) =>
        t
          .setPlaceholder("sk-...")
          .setValue(s.llm.apiKey)
          .onChange(async (v) => { s.llm.apiKey = v; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Base URL")
      .setDesc("API 端点")
      .addText((t) =>
        t
          .setValue(s.llm.baseUrl)
          .onChange(async (v) => { s.llm.baseUrl = v; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Model")
      .setDesc("推荐 deepseek-v4-pro；可选 deepseek-v4-flash / deepseek-chat (将弃用) / deepseek-reasoner (将弃用)")
      .addText((t) =>
        t
          .setValue(s.llm.model)
          .onChange(async (v) => { s.llm.model = v; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Temperature")
      .addText((t) =>
        t
          .setValue(String(s.llm.temperature))
          .onChange(async (v) => { s.llm.temperature = Number(v) || 0; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Timeout (秒)")
      .addText((t) =>
        t
          .setValue(String(s.llm.timeoutMs / 1000))
          .onChange(async (v) => { s.llm.timeoutMs = (Number(v) || 300) * 1000; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Thinking mode")
      .setDesc("启用推理模式（DeepSeek V4 系列支持）")
      .addToggle((t) =>
        t
          .setValue(s.llm.thinkingMode)
          .onChange(async (v) => { s.llm.thinkingMode = v; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Reasoning effort")
      .addDropdown((d) =>
        d
          .addOption("low", "low")
          .addOption("medium", "medium")
          .addOption("high", "high")
          .setValue(s.llm.reasoningEffort)
          .onChange(async (v) => { s.llm.reasoningEffort = v as any; await this.plugin.saveSettings(); }),
      );

    // ─── arXiv ────────────────────────────────────────
    containerEl.createEl("h2", { text: "arXiv 配置" });

    new Setting(containerEl)
      .setName("分类")
      .setDesc("arXiv 分类，如 astro-ph、cs.LG、hep-ph")
      .addText((t) =>
        t
          .setValue(s.arxiv.category)
          .onChange(async (v) => { s.arxiv.category = v.trim(); await this.plugin.saveSettings(); }),
      );

    this.textareaSetting(containerEl, "研究兴趣", "用自然语言描述",
      s.arxiv.researchInterests,
      async (v) => { s.arxiv.researchInterests = v; await this.plugin.saveSettings(); });

    this.textareaSetting(containerEl, "详细收录标准", "符合此标准的论文会生成详细报告",
      s.arxiv.detailCriteria,
      async (v) => { s.arxiv.detailCriteria = v; await this.plugin.saveSettings(); });

    this.textareaSetting(containerEl, "允许 detail 的语义分类", "一行一个，LLM 输出的语义分类（非 arXiv 官方分类）",
      s.arxiv.detailCategories.join("\n"),
      async (v) => { s.arxiv.detailCategories = v.split("\n").map((x) => x.trim()).filter(Boolean); await this.plugin.saveSettings(); });

    this.textareaSetting(containerEl, "Category → Tag map (JSON)", "",
      JSON.stringify(s.arxiv.categoryTagMap, null, 2),
      async (v) => { try { s.arxiv.categoryTagMap = JSON.parse(v); await this.plugin.saveSettings(); } catch {} });

    this.textareaSetting(containerEl, "Category → Display name (JSON)", "",
      JSON.stringify(s.arxiv.categoryDisplayMap, null, 2),
      async (v) => { try { s.arxiv.categoryDisplayMap = JSON.parse(v); await this.plugin.saveSettings(); } catch {} });

    new Setting(containerEl)
      .setName("时区")
      .addText((t) =>
        t
          .setValue(s.arxiv.timezone)
          .onChange(async (v) => { s.arxiv.timezone = v.trim(); await this.plugin.saveSettings(); }),
      );

    // ─── Output & Schedule ────────────────────────────
    containerEl.createEl("h2", { text: "输出 & 调度" });

    new Setting(containerEl)
      .setName("Daily 路径")
      .setDesc("vault 内相对路径")
      .addText((t) =>
        t
          .setValue(s.output.dailyDir)
          .onChange(async (v) => { s.output.dailyDir = v.trim(); await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Papers 路径")
      .setDesc("vault 内相对路径")
      .addText((t) =>
        t
          .setValue(s.output.papersDir)
          .onChange(async (v) => { s.output.papersDir = v.trim(); await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("启用自动调度")
      .addToggle((t) =>
        t
          .setValue(s.schedule.enabled)
          .onChange(async (v) => {
            s.schedule.enabled = v;
            await this.plugin.saveSettings();
            this.plugin.restartScheduler();
          }),
      );

    new Setting(containerEl)
      .setName("调度时间 (HH:MM)")
      .addText((t) =>
        t
          .setValue(s.schedule.runAtLocal)
          .onChange(async (v) => { s.schedule.runAtLocal = v.trim(); await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Tick interval (分钟)")
      .addText((t) =>
        t
          .setValue(String(s.schedule.tickIntervalMin))
          .onChange(async (v) => {
            s.schedule.tickIntervalMin = Number(v) || 20;
            await this.plugin.saveSettings();
            this.plugin.restartScheduler();
          }),
      );

    new Setting(containerEl)
      .setName("Lookback 天数")
      .setDesc("最大 5（受 arXiv /recent 限制）")
      .addText((t) =>
        t
          .setValue(String(s.schedule.lookbackDays))
          .onChange(async (v) => {
            s.schedule.lookbackDays = Math.min(5, Math.max(1, Number(v) || 5));
            await this.plugin.saveSettings();
          }),
      );

    // ─── Advanced ─────────────────────────────────────
    containerEl.createEl("h2", { text: "高级" });

    new Setting(containerEl)
      .setName("Request delay (ms)")
      .addText((t) =>
        t
          .setValue(String(s.advanced.requestDelayMs))
          .onChange(async (v) => { s.advanced.requestDelayMs = Number(v) || 3000; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Cache expiry (days)")
      .addText((t) =>
        t
          .setValue(String(s.advanced.cacheExpiryDays))
          .onChange(async (v) => { s.advanced.cacheExpiryDays = Number(v) || 7; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Section char limit")
      .addText((t) =>
        t
          .setValue(String(s.advanced.sectionCharLimit))
          .onChange(async (v) => { s.advanced.sectionCharLimit = Number(v) || 8000; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Paper char limit")
      .addText((t) =>
        t
          .setValue(String(s.advanced.paperCharLimit))
          .onChange(async (v) => { s.advanced.paperCharLimit = Number(v) || 50000; await this.plugin.saveSettings(); }),
      );

    new Setting(containerEl)
      .setName("Daily char limit")
      .addText((t) =>
        t
          .setValue(String(s.advanced.dailyCharLimit))
          .onChange(async (v) => { s.advanced.dailyCharLimit = Number(v) || 400000; await this.plugin.saveSettings(); }),
      );

    this.textareaSetting(containerEl, "Skip sections (一行一个)", "",
      s.advanced.skipSections.join("\n"),
      async (v) => { s.advanced.skipSections = v.split("\n").map((x) => x.trim()).filter(Boolean); await this.plugin.saveSettings(); });

    this.textareaSetting(containerEl, "Priority sections (一行一个)", "",
      s.advanced.prioritySections.join("\n"),
      async (v) => { s.advanced.prioritySections = v.split("\n").map((x) => x.trim()).filter(Boolean); await this.plugin.saveSettings(); });

    new Setting(containerEl)
      .setName("Log level")
      .addDropdown((d) =>
        d
          .addOption("debug", "debug")
          .addOption("info", "info")
          .addOption("warn", "warn")
          .addOption("error", "error")
          .setValue(s.advanced.logLevel)
          .onChange(async (v) => {
            s.advanced.logLevel = v as any;
            await this.plugin.saveSettings();
            this.plugin.logger.setLevel(v as any);
          }),
      );
  }

  private textareaSetting(
    container: HTMLElement,
    name: string,
    desc: string,
    value: string,
    onChange: (v: string) => Promise<void>,
  ) {
    new Setting(container)
      .setName(name)
      .setDesc(desc)
      .addTextArea((t) => {
        t.setValue(value).onChange((v) => onChange(v));
        t.inputEl.rows = 6;
        t.inputEl.style.width = "100%";
      });
  }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/settings/tab.ts
git commit -m "feat(plugin): add settings tab UI"
```

---

## Task 21: Commands + ribbon

**Files:**
- Create: `plugin/src/commands.ts`

- [ ] **Step 1: Implement `src/commands.ts`**

```ts
import { App, Modal, Notice, Setting } from "obsidian";
import type ArxivDailyPlugin from "../main";
import { todayInTz, formatDate } from "./utils/time";

export function registerCommands(plugin: ArxivDailyPlugin): void {
  const s = () => plugin.settings;

  plugin.addCommand({
    id: "arxiv-daily-run-now",
    name: "Run now (today)",
    callback: async () => {
      const today = formatDate(todayInTz(new Date(), s().arxiv.timezone));
      new Notice(`arXiv Daily: running for ${today}…`);
      const result = await plugin.scheduler.runForDateNow(today);
      new Notice(`arXiv Daily ${today}: ${describeResult(result)}`);
    },
  });

  plugin.addCommand({
    id: "arxiv-daily-run-for-date",
    name: "Run for date…",
    callback: () => {
      new DatePickerModal(plugin.app, async (date) => {
        if (!date) return;
        new Notice(`arXiv Daily: running for ${date}…`);
        const result = await plugin.scheduler.runForDateNow(date);
        new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
      }).open();
    },
  });

  plugin.addCommand({
    id: "arxiv-daily-open-today",
    name: "Open today's daily report",
    callback: async () => {
      const today = formatDate(todayInTz(new Date(), s().arxiv.timezone));
      const path = `${s().output.dailyDir}/${today}.md`;
      const file = plugin.app.vault.getAbstractFileByPath(path);
      if (file) {
        await plugin.app.workspace.openLinkText(path, "", false);
      } else {
        new Notice(`No daily report at ${path}`);
      }
    },
  });

  plugin.addCommand({
    id: "arxiv-daily-show-state",
    name: "Show recent run state",
    callback: () => new StateModal(plugin.app, plugin).open(),
  });

  plugin.addRibbonIcon("calendar-clock", "arXiv Daily: Run now", async () => {
    const today = formatDate(todayInTz(new Date(), s().arxiv.timezone));
    new Notice(`arXiv Daily: running for ${today}…`);
    const result = await plugin.scheduler.runForDateNow(today);
    new Notice(`arXiv Daily ${today}: ${describeResult(result)}`);
  });
}

function describeResult(r: any): string {
  if (!r) return "no result";
  if (r.kind === "completed") return `done (${r.papersWritten} papers)`;
  if (r.kind === "failed_transient") return `transient: ${r.reason}`;
  if (r.kind === "failed_permanent") return `permanent: ${r.reason}`;
  if (r.kind === "skipped") return `skipped: ${r.reason}`;
  return JSON.stringify(r);
}

class DatePickerModal extends Modal {
  private value = "";
  constructor(app: App, private onSubmit: (date: string | null) => void) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "Run arXiv Daily for date" });
    new Setting(contentEl)
      .setName("Date")
      .setDesc("YYYY-MM-DD (must be within the past 5 days for arXiv /recent)")
      .addText((t) =>
        t.setPlaceholder("2026-05-10").onChange((v) => {
          this.value = v.trim();
        }),
      );
    new Setting(contentEl).addButton((b) =>
      b
        .setButtonText("Run")
        .setCta()
        .onClick(() => {
          if (!/^\d{4}-\d{2}-\d{2}$/.test(this.value)) {
            new Notice("Invalid date format");
            return;
          }
          this.close();
          this.onSubmit(this.value);
        }),
    );
  }
  onClose() {
    this.contentEl.empty();
  }
}

class StateModal extends Modal {
  constructor(app: App, private plugin: ArxivDailyPlugin) {
    super(app);
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: "arXiv Daily — Recent state" });
    const snap = this.plugin.stateStore.snapshot();
    const entries = Object.entries(snap).sort((a, b) => (a[0] < b[0] ? 1 : -1));
    if (entries.length === 0) {
      contentEl.createEl("p", { text: "No runs yet." });
      return;
    }
    const ul = contentEl.createEl("ul");
    for (const [date, e] of entries.slice(0, 20)) {
      const li = ul.createEl("li");
      li.setText(
        `${date}: ${e.status} (attempts=${e.attempts}` +
          (e.papersWritten != null ? `, papers=${e.papersWritten}` : "") +
          (e.error ? `, err=${e.error.slice(0, 80)}` : "") +
          `)`,
      );
    }
  }
  onClose() {
    this.contentEl.empty();
  }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `cd plugin && npx tsc -noEmit`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/commands.ts
git commit -m "feat(plugin): add commands, ribbon, date-picker modal, state modal"
```

---

## Task 22: main.ts plugin lifecycle

**Files:**
- Modify: `plugin/main.ts`

- [ ] **Step 1: Replace stub `main.ts` with full implementation**

```ts
import { Plugin } from "obsidian";
import * as path from "node:path";
import { DEFAULT_SETTINGS } from "./src/settings/defaults";
import type { PluginSettings, RunState } from "./src/settings/types";
import { ArxivDailySettingTab } from "./src/settings/tab";
import { Logger } from "./src/services/logger";
import { StateStore } from "./src/services/state-store";
import { RunLock } from "./src/services/run-lock";
import { SchedulerService } from "./src/services/scheduler";
import { LlmClient } from "./src/llm/client";
import { ArxivFetcher } from "./src/pipeline/arxiv-fetcher";
import { HtmlCache } from "./src/pipeline/html-cache";
import { PaperContentFetcher } from "./src/pipeline/paper-content";
import { MarkdownWriter } from "./src/pipeline/markdown-writer";
import { ArxivPipeline } from "./src/pipeline/pipeline";
import { registerCommands } from "./src/commands";

interface PersistedData {
  settings: PluginSettings;
  runState: RunState;
}

export default class ArxivDailyPlugin extends Plugin {
  settings!: PluginSettings;
  logger!: Logger;
  stateStore!: StateStore;
  scheduler!: SchedulerService;
  private runLock = new RunLock();

  async onload() {
    await this.loadSettingsAndState();
    this.logger = new Logger(this.settings.advanced.logLevel);

    this.stateStore = new StateStore(
      async () => {
        const data = (await this.loadData()) as PersistedData | null;
        return { runState: data?.runState ?? {} };
      },
      async ({ runState }) => {
        await this.persistAll(runState);
      },
    );
    await this.stateStore.load();

    this.scheduler = new SchedulerService({
      getSettings: () => this.settings,
      store: this.stateStore,
      lock: this.runLock,
      logger: this.logger,
      runForDate: (date) => this.buildPipeline().runForDate(date),
    });

    this.addSettingTab(new ArxivDailySettingTab(this.app, this));
    registerCommands(this);

    if (this.settings.schedule.enabled) this.scheduler.start();
  }

  onunload() {
    this.scheduler?.stop();
  }

  async saveSettings(): Promise<void> {
    await this.persistAll(this.stateStore?.snapshot() ?? {});
  }

  restartScheduler(): void {
    this.scheduler.stop();
    if (this.settings.schedule.enabled) this.scheduler.start();
  }

  private async loadSettingsAndState(): Promise<void> {
    const data = ((await this.loadData()) as PersistedData | null) ?? {
      settings: DEFAULT_SETTINGS,
      runState: {},
    };
    this.settings = mergeSettings(DEFAULT_SETTINGS, data.settings ?? {});
  }

  private async persistAll(runState: RunState): Promise<void> {
    const data: PersistedData = { settings: this.settings, runState };
    await this.saveData(data);
  }

  private buildPipeline(): ArxivPipeline {
    const llm = new LlmClient(this.settings.llm, this.logger);
    const fetcher = new ArxivFetcher({
      category: this.settings.arxiv.category,
      logger: this.logger,
      requestDelayMs: this.settings.advanced.requestDelayMs,
    });
    const cache = new HtmlCache({
      rootDir: this.resolveCacheDir(),
      expiryDays: this.settings.advanced.cacheExpiryDays,
    });
    const paperFetcher = new PaperContentFetcher(fetcher, cache, this.logger);
    const writer = new MarkdownWriter({
      vault: this.app.vault,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      output: this.settings.output,
    });
    return new ArxivPipeline({
      fetcher,
      paperFetcher,
      writer,
      llm,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      advanced: this.settings.advanced,
      output: this.settings.output,
      llmSettings: this.settings.llm,
    });
  }

  private resolveCacheDir(): string {
    // Use Electron's userData; falls back to home if unavailable.
    let base: string;
    try {
      const electron = require("electron");
      base = electron.remote
        ? electron.remote.app.getPath("userData")
        : electron.app?.getPath("userData") ?? require("os").homedir();
    } catch {
      base = require("os").homedir();
    }
    return path.join(base, "arxiv-daily-cache");
  }
}

function mergeSettings(defaults: PluginSettings, partial: Partial<PluginSettings>): PluginSettings {
  return {
    llm: { ...defaults.llm, ...(partial.llm ?? {}) },
    arxiv: { ...defaults.arxiv, ...(partial.arxiv ?? {}) },
    output: { ...defaults.output, ...(partial.output ?? {}) },
    schedule: { ...defaults.schedule, ...(partial.schedule ?? {}) },
    advanced: { ...defaults.advanced, ...(partial.advanced ?? {}) },
  };
}
```

- [ ] **Step 2: Build the plugin (production)**

Run: `cd plugin && npm run build`
Expected: `main.js` is generated; `tsc -noEmit` passes; no esbuild errors.

- [ ] **Step 3: Run full test suite**

Run: `cd plugin && npm test`
Expected: All tests pass (parser, section-extractor, retry, time, run-lock, state-store, pipeline, scheduler).

- [ ] **Step 4: Commit**

```bash
git add plugin/main.ts
git commit -m "feat(plugin): wire up plugin lifecycle, settings, scheduler"
```

---

## Task 23: README + dev docs

**Files:**
- Create: `plugin/README.md`
- Modify: repo root `README.md` (mention plugin)

- [ ] **Step 1: Create `plugin/README.md`**

```markdown
# arXiv Daily — Obsidian plugin

Native TypeScript rewrite of the `arxiv_daily.py` script as an Obsidian plugin.

## Features (v1 MVP)

- Daily fetch from `https://arxiv.org/list/<category>/recent` with 5-day rolling lookback
- LLM-based paper filtering (OpenAI-compatible: DeepSeek, OpenAI, …)
- Daily summary and per-paper detail summaries written as Markdown into your vault
- Catch-up scheduler that runs while Obsidian is open; backfills missed days within the lookback window
- Manual "Run now" / "Run for date…" commands and ribbon icon
- Cross-platform (Windows / macOS / Linux)

## Installation (dev/test)

1. Build the plugin:
   ```bash
   cd plugin
   npm install
   npm run build
   ```
2. Copy `manifest.json`, `main.js`, `styles.css` into `<vault>/.obsidian/plugins/arxiv-daily/`.
3. Enable the plugin in Obsidian (Settings → Community plugins).
4. Open Settings → arXiv Daily, fill in API Key (default endpoint is DeepSeek's).

## Settings overview

- **LLM**: API key, base URL, model (default `deepseek-v4-pro`), temperature, timeout, thinking mode + reasoning effort
- **arXiv**: category (`astro-ph` by default), research interests, detail criteria, semantic-category configuration, timezone
- **Output**: daily and papers directories (vault-relative)
- **Schedule**: enable, daily run time `HH:MM`, tick interval, lookback days (≤5)
- **Advanced**: request delay, cache TTL, char limits, skip/priority sections, log level

## Commands

| Command | Action |
|---|---|
| `arXiv Daily: Run now` | Pulls today, writes daily + papers |
| `arXiv Daily: Run for date…` | Pulls a specific date within last 5 days |
| `arXiv Daily: Open today's daily report` | Opens `dailyDir/<today>.md` |
| `arXiv Daily: Show recent run state` | Lists last 20 dates and their statuses |

## Development

```bash
cd plugin
npm run dev   # watch build
npm test      # run unit tests (vitest)
```

## v2 roadmap

- Multi-profile (multiple research directions in parallel)
- OS-level cron fallback (CLI entrypoint)
- Per-profile LLM overrides
```

- [ ] **Step 2: Append a section to repo root `README.md`**

Edit `README.md` and append after the existing content:

```markdown

## Obsidian plugin

A native TypeScript Obsidian plugin in `plugin/` provides the same functionality with a settings GUI and catch-up scheduling. See `plugin/README.md` for installation and development.
```

- [ ] **Step 3: Commit**

```bash
git add plugin/README.md README.md
git commit -m "docs: add plugin README and link from root README"
```

---

## Task 24: Manual smoke test

This task has no code; it's a checklist to be performed by a human (or whoever is running the implementation) before marking v1 done.

- [ ] **Step 1: Install plugin into a real vault**

Run:
```bash
cd plugin && npm run build
# Create a test vault dir, then copy:
mkdir -p ~/obsidian-test-vault/.obsidian/plugins/arxiv-daily
cp manifest.json main.js styles.css ~/obsidian-test-vault/.obsidian/plugins/arxiv-daily/
```
Open the vault in Obsidian and enable the plugin.

- [ ] **Step 2: Configure**

In Settings → arXiv Daily:
- Set API Key to a real DeepSeek API key.
- Leave category `astro-ph`, default research interests.
- Set Daily/Papers paths to `arxiv-daily/daily` / `arxiv-daily/papers`.

- [ ] **Step 3: Run now**

Click the ribbon icon. Expected:
- Notice "running for YYYY-MM-DD…"
- After 1-3 minutes, Notice "done (N papers)"
- A new file at `arxiv-daily/daily/<today>.md` opens correctly in Obsidian
- A handful of files appear in `arxiv-daily/papers/` for detail-marked papers
- Internal links `[[YYMM.NNNNN]]` resolve to those papers

- [ ] **Step 4: Run for date…**

Use the command and enter yesterday's date. Expected:
- Either succeeds (if date is in `/recent`) and writes a daily for yesterday
- Or returns `failed_transient` if the date is outside `/recent` (uncommon for yesterday)

- [ ] **Step 5: Show state**

Use the "Show recent run state" command. Expected: at least one row showing today/yesterday status.

- [ ] **Step 6: Restart Obsidian, verify state persists**

Close Obsidian fully and reopen. Run "Show state" again — same rows visible.

- [ ] **Step 7: Verify scheduler tick fires**

In settings, set Tick interval to 1 minute and runAtLocal to a time 1 minute in the future. Wait. Expected: a tick fires (visible via Notice if today wasn't already completed; otherwise silent).

- [ ] **Step 8: Final commit**

If any tweaks were needed during smoke testing, commit them. Otherwise, no commit needed.

```bash
git status   # confirm clean
```

- [ ] **Step 9: Merge prep**

If smoke test passes, the branch `obsidian-plugin` is ready to merge. Coordinate with user before merging or opening a PR.

---

## Self-Review

Spec coverage check (against `docs/superpowers/specs/2026-05-11-obsidian-plugin-design.md`):

| Spec section | Implemented by tasks |
|---|---|
| §3 Architecture | Tasks 5, 6, 18, 19, 22 |
| §4.1 Settings model | Tasks 2, 22 |
| §4.2 Run state model | Tasks 2, 6 |
| §5.1 Catch-up tick | Task 19 |
| §5.2 Pipeline stages | Tasks 8-18 |
| §5.3 Manual triggers | Tasks 19, 21 |
| §6 UI settings tab | Task 20 |
| §6 Commands + ribbon | Task 21 |
| §7 Cross-platform | Tasks 9 (requestUrl), 10 (Electron userData), 17 (normalizePath) |
| §8 Error handling | Tasks 4 (retry), 6 (state machine), 18 (result classification) |
| §9 Testing | Tasks 3, 4, 5, 6, 8, 11, 18, 19 (vitest unit tests); Task 24 (manual smoke) |
| §10 Build & layout | Task 1 |
| §11 v2 forward-compat | Settings shape leaves room (Task 2); migration not implemented in v1 (acceptable) |
| §12 Risks | Mitigated: parser is loose (Task 8), atomic state writes (Task 6), char limits enforced (Task 2/11), model field free-text (Task 20) |

Placeholder scan: no TBD / TODO / "fill in later" — all steps contain concrete code or commands.

Type consistency: `PipelineResult` shape used identically in `pipeline.ts` (Task 18), `scheduler.ts` (Task 19), and `commands.ts` (Task 21). `RunStatus` values match `state-store.test.ts` (Task 6) and `scheduler.ts` (Task 19). `DailyPaperWithContent` flows from `paper-filter.ts` → `pipeline.ts` → `summarizer.ts` consistently.
