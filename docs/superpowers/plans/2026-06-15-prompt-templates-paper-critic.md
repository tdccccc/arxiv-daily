# Prompt 模板化 + 详细总结 paper-critic 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **本项目约束**：禁止派发 subagent，全部 inline 执行。Git commit 用多个 `-m`（subject + body 说明 why/what + validation），不用 heredoc。

**Goal:** 把三个内联 LLM prompt 抽离成 Markdown 模板（零行为变更），统一两条章节抽取路径的排序并放宽字数预算，再把详细总结升级为带评价维度的 paper-critic。

**Architecture:** 新增 `src/prompts/` 存放 `.md` 模板，由 esbuild `text` loader 构建时内联、`renderPrompt` 做 `{{var}}` 替换；vitest 用一个 `load` 插件让 `.md` 在测试里也按文本解析。抽取层改两个 `sectionRank` 函数 + 放宽 `defaults.ts`。critic 升级只动详细总结的 `.md` 模板与 `DETAIL_SUMMARY_HEADINGS`，日报格式 / parser / Dashboard 数据模型不动。

**Tech Stack:** TypeScript、esbuild（`platform: node`、CJS 双产物）、vitest（happy-dom，`globals: false`）。

**Spec:** `docs/superpowers/specs/2026-06-15-prompt-templates-paper-critic-design.md`

**实施顺序:** Part A（Task 1–4）→ Part C（Task 5–7）→ Part B（Task 8）。

---

## File Structure

**新建**
- `plugin/src/prompts/render.ts` —— `renderPrompt(template, vars)`，唯一职责：`{{var}}` 替换 + 残留占位符报错。
- `plugin/src/prompts/md.d.ts` —— `declare module "*.md"`，让 `tsc` 认识 `.md` 文本导入。
- `plugin/src/prompts/paper-filter.system.md` —— 筛选 prompt 模板。
- `plugin/src/prompts/daily-summary.system.md` —— 日报 prompt 模板。
- `plugin/src/prompts/paper-detail.system.md` —— 详细总结 prompt 模板。
- `plugin/tests/render-prompt.test.ts` —— `renderPrompt` 单测。

**修改**
- `plugin/esbuild.config.mjs` —— `common` 加 `loader: { ".md": "text" }`。
- `plugin/vitest.config.mts` —— 加 `markdown-as-text` 插件。
- `plugin/src/pipeline/paper-filter.ts:45` —— 用模板 + `renderPrompt` 替换内联 prompt。
- `plugin/src/pipeline/summarizer.ts:126,366` —— 同上（日报、详细总结）。
- `plugin/src/pipeline/section-extractor.ts:277` —— HTML 路径 `sectionRank` 提权 intro/related。
- `plugin/src/pipeline/source-extractor.ts:454` —— LaTeX 路径 `sectionRank` 统一语义。
- `plugin/src/settings/defaults.ts:34` —— `paperCharLimit` 50k→100k、`sectionCharLimit` 8k→16k。
- `plugin/src/dashboard/detail-summary.ts:2` —— `DETAIL_SUMMARY_HEADINGS` 跟随新结构。
- `plugin/tests/section-extractor.test.ts`、`plugin/tests/source-extractor.test.ts`、`plugin/tests/dashboard-detail-summary.test.ts` —— 对应新行为的断言。

---

## Task 1: 模板渲染基建（renderPrompt + 构建/测试 loader）

**Files:**
- Create: `plugin/src/prompts/render.ts`
- Create: `plugin/src/prompts/md.d.ts`
- Modify: `plugin/esbuild.config.mjs:9-18`
- Modify: `plugin/vitest.config.mts`
- Test: `plugin/tests/render-prompt.test.ts`

- [ ] **Step 1: 写失败测试**

`plugin/tests/render-prompt.test.ts`：

```ts
import { describe, it, expect } from "vitest";
import { renderPrompt } from "../src/prompts/render";

describe("renderPrompt", () => {
  it("substitutes {{var}} placeholders", () => {
    expect(renderPrompt("a {{x}} b {{y}}", { x: "1", y: "2" })).toBe("a 1 b 2");
  });

  it("leaves single braces (JSON examples) untouched", () => {
    expect(renderPrompt('{"papers": [{{tag}}]}', { tag: "T" })).toBe(
      '{"papers": [T]}',
    );
  });

  it("throws on an unfilled placeholder", () => {
    expect(() => renderPrompt("a {{missing}}", { x: "1" })).toThrow(
      /missing/,
    );
  });
});
```

- [ ] **Step 2: 运行确认失败**

Run: `cd plugin && npx vitest run tests/render-prompt.test.ts`
Expected: FAIL（`renderPrompt` 不存在 / 模块找不到）。

- [ ] **Step 3: 实现 render.ts**

`plugin/src/prompts/render.ts`：

```ts
/**
 * Fill {{name}} placeholders in a prompt template. Single braces (e.g. JSON
 * examples) are left alone. Throws if any {{...}} placeholder remains unfilled,
 * which catches template typos and missing variables at call time.
 */
export function renderPrompt(
  template: string,
  vars: Record<string, string>,
): string {
  const rendered = template.replace(/\{\{(\w+)\}\}/g, (match, key: string) =>
    key in vars ? vars[key] : match,
  );
  const leftover = /\{\{\w+\}\}/.exec(rendered);
  if (leftover) {
    throw new Error(`renderPrompt: unfilled placeholder ${leftover[0]}`);
  }
  return rendered;
}
```

- [ ] **Step 4: 运行确认通过**

Run: `cd plugin && npx vitest run tests/render-prompt.test.ts`
Expected: PASS（3 passed）。

- [ ] **Step 5: 让 tsc 认识 `.md` 文本导入**

`plugin/src/prompts/md.d.ts`：

```ts
declare module "*.md" {
  const content: string;
  export default content;
}
```

（`tsconfig.json` 的 `include` 已含 `src/**/*.ts`，会自动覆盖此声明文件。）

- [ ] **Step 6: esbuild 把 `.md` 当文本内联**

`plugin/esbuild.config.mjs` 的 `common` 对象加一行 `loader`：

```js
const common = {
  bundle: true,
  format: "cjs",
  target: "es2020",
  platform: "node",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  minify: prod,
  loader: { ".md": "text" },
};
```

- [ ] **Step 7: 让 vitest 也把 `.md` 当文本**

`plugin/vitest.config.mts`，在文件顶部 import 后加插件，并挂到配置的 `plugins`：

```ts
import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { readFileSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));

const markdownAsText = {
  name: "markdown-as-text",
  enforce: "pre" as const,
  load(id: string) {
    const path = id.split("?")[0];
    if (path.endsWith(".md")) {
      return `export default ${JSON.stringify(readFileSync(path, "utf-8"))};`;
    }
    return null;
  },
};

export default defineConfig({
  plugins: [markdownAsText],
  test: {
    environment: "happy-dom",
    globals: false,
    include: ["tests/**/*.test.ts"],
    environmentOptions: {
      happyDOM: {
        settings: {
          disableJavaScriptEvaluation: true,
          disableJavaScriptFileLoading: true,
          disableCSSFileLoading: true,
        },
      },
    },
  },
  resolve: {
    alias: {
      "@": resolve(here, "src"),
      obsidian: resolve(here, "tests/__mocks__/obsidian.ts"),
    },
  },
});
```

- [ ] **Step 8: 提交**

```bash
cd plugin && git add src/prompts/render.ts src/prompts/md.d.ts esbuild.config.mjs vitest.config.mts tests/render-prompt.test.ts && git commit -m "feat(prompts): add renderPrompt helper and .md text loaders" -m "Why: prepare for moving inline LLM prompts into standalone .md templates that are inlined at build time (esbuild) and resolvable in tests (vitest)." -m "What: renderPrompt does {{var}} substitution and throws on leftover placeholders; add *.md type decl, esbuild text loader, and a vitest load plugin so the same import works in both build and tests." -m "Validation: npx vitest run tests/render-prompt.test.ts (3 passed)."
```

---

## Task 2: 抽离筛选 prompt（零行为变更）

**Files:**
- Create: `plugin/src/prompts/paper-filter.system.md`
- Modify: `plugin/src/pipeline/paper-filter.ts:45-61`
- Test: `plugin/tests/paper-filter.test.ts`（新增快照用例）

- [ ] **Step 1: 写表征快照测试（对当前代码记录基线）**

在 `plugin/tests/paper-filter.test.ts` 的 `describe("filterPapers", ...)` 内追加：

```ts
  it("system prompt matches the golden snapshot", async () => {
    const llm = {
      call: vi.fn().mockResolvedValue(JSON.stringify({ papers: [] })),
    };
    await filterPapers([samplePaper], {
      llm: llm as any,
      logger: new Logger("error"),
      arxivSettings: makeArxiv(makeTopics()),
    });
    expect(llm.call.mock.calls[0][0][0].content as string).toMatchSnapshot();
  });
```

- [ ] **Step 2: 运行记录基线快照（当前内联代码）**

Run: `cd plugin && npx vitest run tests/paper-filter.test.ts`
Expected: PASS，并生成 `tests/__snapshots__/paper-filter.test.ts.snap`（这是抽离前的“黄金基线”）。

- [ ] **Step 3: 新建模板文件（逐字复制当前 prompt，变量改占位符）**

`plugin/src/prompts/paper-filter.system.md`，内容与 `paper-filter.ts:45-61` 的模板字符串**逐字一致**，仅把 `${topicLines}`→`{{topicLines}}`、`${tagOptions}`→`{{tagOptions}}`：

```md
你是一位研究者的助手。请根据下方主题列表，为每篇论文选择最匹配的主题。

## 主题列表
{{topicLines}}

## 输出格式
请只输出一个 JSON 对象，不要输出任何其他内容：
{"papers": [
  {"id": "YYMM.NNNNN", "category": "{{tagOptions}}", "detail": true/false},
  ...
]}

规则：
- category 选择最匹配的主题 tag；若与所有主题都不相关，返回 "skip"
- detail 仅在带 [DETAIL] 标记的主题上有意义；当且仅当该论文是该主题的核心贡献时设为 true，其余设为 false
- detail 判定从严：宁可漏选也不要错选——不确定时设为 false
- 如果没有任何相关论文，返回 {"papers": []}
```

> 注意：文件**末尾不要有多余换行**——模板字符串以 `[]}` 结尾、无尾随换行。若快照在 Step 5 因末尾换行不一致而失败，删掉文件末尾的换行符。

- [ ] **Step 4: 接线 paper-filter.ts**

在 `plugin/src/pipeline/paper-filter.ts` 顶部 import：

```ts
import { renderPrompt } from "../prompts/render";
import filterSystemTemplate from "../prompts/paper-filter.system.md";
```

把 `:45` 的 `const systemPrompt = \`...\`;`（整个模板字面量）替换为：

```ts
  const systemPrompt = renderPrompt(filterSystemTemplate, {
    topicLines,
    tagOptions,
  });
```

- [ ] **Step 5: 运行确认快照不变**

Run: `cd plugin && npx vitest run tests/paper-filter.test.ts`
Expected: PASS，快照**未更新**（逐字节一致即证明零行为变更）。若 FAIL，对照 diff 修模板空白，**不要**用 `-u` 强写。

- [ ] **Step 6: 提交**

```bash
cd plugin && git add src/prompts/paper-filter.system.md src/pipeline/paper-filter.ts tests/paper-filter.test.ts tests/__snapshots__/paper-filter.test.ts.snap && git commit -m "refactor(paper-filter): extract system prompt to a Markdown template" -m "Why: keep prompt text in a standalone file that can be viewed, diffed and iterated independently of code." -m "What: move the filter system prompt into src/prompts/paper-filter.system.md rendered via renderPrompt; a golden snapshot test pins byte-for-byte equality with the previous inline string." -m "Validation: npx vitest run tests/paper-filter.test.ts (snapshot unchanged)."
```

---

## Task 3: 抽离日报 prompt（零行为变更）

**Files:**
- Create: `plugin/src/prompts/daily-summary.system.md`
- Modify: `plugin/src/pipeline/summarizer.ts:126-164`
- Test: `plugin/tests/summarizer.test.ts`（新增快照用例）

- [ ] **Step 1: 写表征快照测试**

在 `plugin/tests/summarizer.test.ts` 顶部确保 `import { summarizeDaily } from "../src/pipeline/summarizer";` 已存在，然后在 `describe` 内追加：

```ts
  it("daily system prompt matches the golden snapshot", async () => {
    const calls: any[] = [];
    const llm = {
      call: vi.fn(async (messages: any[]) => {
        calls.push(messages);
        return "## Topic\n今日无相关论文更新。";
      }),
    };
    await summarizeDaily(
      [
        {
          id: "2606.12345",
          title: "Snapshot Paper",
          authors: "A. Author",
          abstract: "abstract",
          category: "topic",
          isDetail: true,
          abstractConclusion: "## Abstract\nabstract",
          fullSections: null,
          detailLink: "[[2606.12345]]",
        },
      ],
      "2026-06-13",
      {
        llm: llm as any,
        logger: new Logger("error"),
        arxivSettings: {
          ...DEFAULT_SETTINGS.arxiv,
          topics: [
            { id: "topic", name: "Topic", tag: "topic", description: "topic", detail: true },
          ],
        },
        advanced: DEFAULT_SETTINGS.advanced,
        llmTemperature: DEFAULT_SETTINGS.llm.temperature,
      },
    );
    expect(calls[0][0].content as string).toMatchSnapshot();
  });
```

- [ ] **Step 2: 运行记录基线快照**

Run: `cd plugin && npx vitest run tests/summarizer.test.ts`
Expected: PASS，生成 / 追加到 `tests/__snapshots__/summarizer.test.ts.snap`。

- [ ] **Step 3: 新建日报模板**

`plugin/src/prompts/daily-summary.system.md`，与 `summarizer.ts:126-164` 模板**逐字一致**，仅替换：`${categoryList}`→`{{categoryList}}`、`${partialNote}`→`{{partialNote}}`、`${headerFmt}`→`{{headerFmt}}`、`${detailLinkTemplate}`→`{{detailLinkTemplate}}`。开头如下，其余规则段照抄：

```md
你是一个专业的研究助手。请根据提供的论文摘要、结论与可用正文摘录，生成 arXiv 每日论文追踪日报。

你的任务不是复述摘要，而是帮助研究者快速判断这篇论文的核心价值：它解决了什么具体问题、用了什么关键方法、得到什么证据、结论边界在哪里。

## Category 与显示名称对应关系
{{categoryList}}
{{partialNote}}
请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，直接输出内容）：

{{headerFmt}}## [显示名称]
### <实际论文标题>
> 信息来源：<按输入的 Source sections 填写，例如 Abstract, Conclusion；不要编造>
- **作者**: First Author et al.
- **arXiv**: [ID](https://arxiv.org/abs/ID)
- **核心问题**: 论文试图解决的具体问题是什么，为什么值得研究（1句）
- **关键方法**: 作者用了什么方法、数据、模型、观测、模拟或理论工具（1-2句）
- **主要结果**: 优先写数值、误差、显著性、提升幅度、样本规模、参数范围、与前人/基线的对比；没有数值则写清作者声称的定性结果（1-2句）
- **为什么值得看**: 这篇论文具体改变了什么判断、解决了什么问题、约束了什么范围，或适用于什么场景（1句）
- **局限或边界**: 适用条件、不确定性、未覆盖的问题；若原文未说明，写"原文未说明"（1句）

详细收录论文的唯一格式差异：如果输入的 Paper 标题行已经带有本地 detail 链接，则对应标题写成：
### <实际论文标题> → {{detailLinkTemplate}}

注意：
- 所有论文（无论是否详细收录）都必须按上述完整格式输出，包含信息来源与五个核心字段，不得省略或只列标题
- 使用中文撰写，保留关键英文术语
- 数学公式必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$
- 必须输出所有 category 的二级标题（使用上面的显示名称），如果某个 category 今日无论文，在标题下写"今日无相关论文更新。"
- 同一个 category 只能输出一次；属于同一 category 的论文必须放在同一个二级标题下
- 只有输入 Paper 标题行已经带 → 本地链接的论文才保留该链接标记
- 未标记的论文不要自行新增 →、wikilink 或本地 Markdown 链接
- 不要输出未入选论文、候选论文、漏报列表或补充列表
- 输入中的 Inbox 行说明论文是 new 还是 seen_before；可在总结中自然保留该状态，不要把 ignored 论文补回来
- 先在内部判断论文属于方法、观测、理论、模拟、数据发布、综述等哪类，但不要输出类型；根据论文类型提取最核心的信息
- 只基于输入内容回答，不要引入外部知识，不要补全输入中没有说明的数据、实验、指标或结论
- 如果输入没有说明某项信息，请写"原文未说明"，不要猜测
- 如果输入只有摘要或摘要+结论，请按摘要级信息生成快速筛选摘要；如果输入包含正文结果、实验、方法或讨论章节，请优先使用这些高密度证据
- 区分作者已经用数据/实验/理论推导支持的结果和仅由作者声称的结果；证据细节不足时写"作者声称"
- 不要写"具有重要意义""提高了理解"这类空泛句子；每个价值判断必须说明具体改变了什么判断、约束了什么问题、或适用于什么场景
```

> 末尾同样不要多余换行（原字面量以 `...适用于什么场景` 结束）。`{{partialNote}}` / `{{headerFmt}}` 是带前后换行的条件串，原样占位即可；快照负责锁住空白。

- [ ] **Step 4: 接线 summarizer.ts（日报）**

在 `plugin/src/pipeline/summarizer.ts` 顶部 import：

```ts
import { renderPrompt } from "../prompts/render";
import dailySystemTemplate from "../prompts/daily-summary.system.md";
```

把 `callDailyLlm` 内 `:126` 的 `const systemPrompt = \`...\`;` 整块替换为：

```ts
  const systemPrompt = renderPrompt(dailySystemTemplate, {
    categoryList,
    partialNote,
    headerFmt,
    detailLinkTemplate,
  });
```

- [ ] **Step 5: 运行确认快照不变**

Run: `cd plugin && npx vitest run tests/summarizer.test.ts`
Expected: PASS，快照未更新。FAIL 则修模板空白，不用 `-u`。

- [ ] **Step 6: 提交**

```bash
cd plugin && git add src/prompts/daily-summary.system.md src/pipeline/summarizer.ts tests/summarizer.test.ts tests/__snapshots__/summarizer.test.ts.snap && git commit -m "refactor(summarizer): extract daily report prompt to a template" -m "Why: keep the daily prompt editable and diffable on its own." -m "What: move the daily system prompt into src/prompts/daily-summary.system.md rendered via renderPrompt; golden snapshot pins byte-for-byte equality." -m "Validation: npx vitest run tests/summarizer.test.ts (snapshot unchanged)."
```

---

## Task 4: 抽离详细总结 prompt（零行为变更）

**Files:**
- Create: `plugin/src/prompts/paper-detail.system.md`
- Modify: `plugin/src/pipeline/summarizer.ts:366-403`
- Test: `plugin/tests/summarizer.test.ts`（新增详细总结快照用例）

- [ ] **Step 1: 写表征快照测试**

在 `plugin/tests/summarizer.test.ts` 顶部加 `import { summarizePaperDetail } from "../src/pipeline/summarizer";`，并追加：

```ts
  it("detail system prompt matches the golden snapshot", async () => {
    const calls: any[] = [];
    const llm = {
      call: vi.fn(async (messages: any[]) => {
        calls.push(messages);
        return "## 研究问题\nx";
      }),
    };
    await summarizePaperDetail(
      {
        id: "2606.12345",
        title: "Detail Snapshot Paper",
        authors: "A. Author",
        abstract: "abstract",
        category: "topic",
        isDetail: true,
        abstractConclusion: "## Abstract\nabstract",
        fullSections: "## Method\nWe model the likelihood.",
      },
      {
        llm: llm as any,
        logger: new Logger("error"),
        arxivSettings: DEFAULT_SETTINGS.arxiv,
        advanced: DEFAULT_SETTINGS.advanced,
        llmTemperature: DEFAULT_SETTINGS.llm.temperature,
      },
    );
    expect(calls[0][0].content as string).toMatchSnapshot();
  });
```

- [ ] **Step 2: 运行记录基线快照**

Run: `cd plugin && npx vitest run tests/summarizer.test.ts`
Expected: PASS，追加详细总结快照。

- [ ] **Step 3: 新建详细总结模板（当前结构，零变更）**

`plugin/src/prompts/paper-detail.system.md`，与 `summarizer.ts:366-403` **逐字一致**，仅把 `${paper.title}`→`{{title}}`、`${paper.id}`→`{{id}}`（出现两次）：

```md
你是一个专业的研究助手。请根据提供的论文各章节内容，生成一篇详细的中文论文总结。

你的任务不是复述摘要，而是还原论文的贡献链条：研究问题 -> 方法设计 -> 关键证据 -> 主要结论 -> 适用边界。

请严格按照以下 Markdown 格式输出（不要输出 Markdown 代码块标记，不要输出 YAML frontmatter，直接从 # 标题开始）：

# {{title}}

- **arXiv**: [{{id}}](https://arxiv.org/abs/{{id}})

## 研究问题
论文要解决的具体问题是什么？为什么这个问题值得研究？

## 方法设计
作者采用了什么核心方法、模型、数据、实验、观测、模拟或理论框架？

## 关键证据
作者用什么证据支持结论？优先保留数值、样本规模、误差、显著性、参数范围、基线对比或实验设置。

## 主要结论
论文最核心的发现或贡献是什么？区分作者已经证明的结果和作者提出的解释。

## 适用边界
结论在哪些条件下成立？有哪些限制、不确定性或未覆盖的问题？

## 一句话价值判断
用一句话说明这篇论文最值得关注的点，避免空泛评价。

注意：
- 使用中文撰写
- 保留关键英文术语（如专有名词、物理量）
- 数学公式、物理量和符号必须使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$
- 只基于输入内容回答，不要引入外部知识，不要补全输入中没有说明的数据、实验、指标或结论
- 如果某项信息在输入中没有说明，请写"原文未说明"
- 先在内部判断论文属于方法、观测、理论、模拟、数据发布、综述等哪类，但不要输出类型；根据论文类型组织重点
- 优先提取数值、误差、显著性、提升幅度、样本规模、参数范围、与前人/基线的对比
- 区分作者已经用数据/实验/理论推导支持的结果和作者提出的解释；证据细节不足时写"作者声称"
- 不要写"具有重要意义""提高了理解"这类空泛句子；每个价值判断必须说明具体改变了什么判断、约束了什么问题、或适用于什么场景
```

> 末尾不要多余换行。

- [ ] **Step 4: 接线 summarizer.ts（详细总结）**

顶部 import：

```ts
import detailSystemTemplate from "../prompts/paper-detail.system.md";
```

把 `summarizePaperDetail` 内 `:366` 的 `const systemPrompt = \`...\`;` 整块替换为：

```ts
  const systemPrompt = renderPrompt(detailSystemTemplate, {
    title: paper.title,
    id: paper.id,
  });
```

- [ ] **Step 5: 运行确认快照不变**

Run: `cd plugin && npx vitest run tests/summarizer.test.ts`
Expected: PASS，快照未更新。

- [ ] **Step 6: 全量构建自检（确认 esbuild loader + tsc 通畅）**

Run: `cd plugin && npm run build`
Expected: tsc 无报错、两个 bundle 产物生成（`.md` 已内联进 `main.js`）。

- [ ] **Step 7: 提交**

```bash
cd plugin && git add src/prompts/paper-detail.system.md src/pipeline/summarizer.ts tests/summarizer.test.ts tests/__snapshots__/summarizer.test.ts.snap && git commit -m "refactor(summarizer): extract paper-detail prompt to a template" -m "Why: finish moving all three prompts out of code; detail prompt is the next one to be enhanced and is easier to iterate as a file." -m "What: move the detail system prompt into src/prompts/paper-detail.system.md rendered via renderPrompt; golden snapshot pins equality; npm run build confirms the esbuild text loader inlines templates." -m "Validation: npx vitest run tests/summarizer.test.ts (snapshot unchanged); npm run build green."
```

---

## Task 5: 统一 HTML 路径章节排序（Option A）

**Files:**
- Modify: `plugin/src/pipeline/section-extractor.ts:277-294`
- Test: `plugin/tests/section-extractor.test.ts`

- [ ] **Step 1: 写失败测试（intro 应排在杂项之上）**

在 `plugin/tests/section-extractor.test.ts` 的 `describe` 内追加：

```ts
  it("ranks introduction above generic sections when budget is tight", () => {
    const intro = "Background and motivation text. ".repeat(40);
    const generic = "Notation conventions used throughout. ".repeat(40);
    const html = `<html><body>
      <h2>Introduction</h2><p>${intro}</p>
      <h2>Notation</h2><p>${generic}</p>
    </body></html>`;
    const out = extractSections(html, {
      sectionCharLimit: 2000,
      paperCharLimit: 900,
      skipSections: [],
      prioritySections: [],
    });
    expect(out).toContain("## Introduction");
    expect(out).not.toContain("## Notation");
  });
```

- [ ] **Step 2: 运行确认失败**

Run: `cd plugin && npx vitest run tests/section-extractor.test.ts`
Expected: FAIL（当前 `other`=rank2 高于 `introduction`=rank3，因此被保留的是 Notation，不是 Introduction）。

- [ ] **Step 3: 改 sectionRank**

把 `plugin/src/pipeline/section-extractor.ts:277-294` 的 `sectionRank` 整个函数替换为：

```ts
function sectionRank(kinds: SectionKind[], configuredPriority: boolean): number {
  if (
    configuredPriority ||
    kinds.some((k) => k === "abstract" || k === "conclusion")
  ) {
    return 0;
  }
  if (
    kinds.some((k) =>
      ["result", "experiment", "method", "data", "limitation", "discussion"].includes(k),
    )
  ) {
    return 1;
  }
  if (kinds.some((k) => k === "introduction" || k === "related")) return 2;
  return 3;
}
```

- [ ] **Step 4: 运行确认通过（含原有“核心证据优先”用例不回归）**

Run: `cd plugin && npx vitest run tests/section-extractor.test.ts`
Expected: PASS。原有 `prioritizes high-value classified sections when budget is tight`（intro 仍低于 method/result，预算 700 时仍被排除）应继续 PASS。

- [ ] **Step 5: 提交**

```bash
cd plugin && git add src/pipeline/section-extractor.ts tests/section-extractor.test.ts && git commit -m "feat(section-extractor): promote introduction/related above misc sections" -m "Why: introduction/related is where a paper states its novelty vs prior work; it was ranked last and dropped first on long papers, starving the contribution analysis." -m "What: re-tier the HTML-path sectionRank to abstract/conclusion > core evidence > introduction/related > misc; core numerical evidence still wins budget first." -m "Validation: npx vitest run tests/section-extractor.test.ts (new + existing pass)."
```

---

## Task 6: 统一 LaTeX 路径章节排序（与 HTML 一致语义）

**Files:**
- Modify: `plugin/src/pipeline/source-extractor.ts:454-472`
- Test: `plugin/tests/source-extractor.test.ts`

- [ ] **Step 1: 写失败测试（limitation 应作为核心证据高于 introduction）**

在 `plugin/tests/source-extractor.test.ts` 的 `describe("extractLatexSource", ...)` 内追加：

```ts
  it("treats limitations as core evidence over introduction when budget is tight", () => {
    const intro = "Motivation and background context here. ".repeat(25);
    const limitation = "Caveats, systematics, and uncertainty budget. ".repeat(20);
    const source = String.raw`
\documentclass{article}
\begin{document}
\section{Introduction}
${intro}
\section{Limitations}
${limitation}
\end{document}
`;
    const result = extractLatexSource(texBuffer(source), {
      sectionCharLimit: 2000,
      paperCharLimit: 800,
      skipSections: ["references", "appendix"],
      prioritySections: [],
    });
    const full = result.fullSections ?? "";
    expect(full).toContain("## Limitations");
    expect(full).not.toContain("## Introduction");
  });
```

- [ ] **Step 2: 运行确认失败**

Run: `cd plugin && npx vitest run tests/source-extractor.test.ts`
Expected: FAIL（当前 `intro/background`=rank5 高于 `limitation`=默认 rank10，保留的是 Introduction）。

- [ ] **Step 3: 改 LaTeX 路径 sectionRank**

把 `plugin/src/pipeline/source-extractor.ts:454-472` 的 `sectionRank` 整个函数替换为：

```ts
function sectionRank(section: Section, opts: SourceExtractOpts): number {
  const title = section.title.toLowerCase();
  if (
    opts.prioritySections.some((term) => {
      const lower = term.trim().toLowerCase();
      return lower && title.includes(lower);
    })
  ) {
    return 0;
  }
  const classified = classifySection(section.title, section.body);
  if (classified.includes("abstract") || classified.includes("conclusion")) {
    return 0;
  }
  if (
    classified.some((k) =>
      ["result", "experiment", "method", "data", "limitation", "discussion"].includes(k),
    )
  ) {
    return 1;
  }
  if (
    classified.some((k) => k === "introduction" || k === "related") ||
    /intro|background/i.test(title)
  ) {
    return 2;
  }
  return 3;
}
```

- [ ] **Step 4: 运行确认通过（含原有用例不回归）**

Run: `cd plugin && npx vitest run tests/source-extractor.test.ts`
Expected: PASS（三个原用例输入都在预算内、全部章节保留，排序变化不改变其断言）。

- [ ] **Step 5: 提交**

```bash
cd plugin && git add src/pipeline/source-extractor.ts tests/source-extractor.test.ts && git commit -m "feat(source-extractor): harmonize LaTeX section ranking with HTML path" -m "Why: the two extraction paths ranked sections differently (LaTeX dumped related/experiment/discussion/limitation into misc), so a paper's excerpt drifted depending on whether HTML or source was used." -m "What: re-tier LaTeX-path sectionRank to the same contract — abstract/conclusion > core evidence > introduction/related > misc." -m "Validation: npx vitest run tests/source-extractor.test.ts (new + existing pass)."
```

---

## Task 7: 放宽默认字数限制

**Files:**
- Modify: `plugin/src/settings/defaults.ts:34-35`
- Test: `plugin/tests/validation.test.ts`（或新增最小断言）

- [ ] **Step 1: 写断言测试**

在 `plugin/tests/validation.test.ts` 末尾追加一个 describe（确认默认值）：

```ts
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

describe("advanced default char limits", () => {
  it("uses the relaxed extraction budgets", () => {
    expect(DEFAULT_SETTINGS.advanced.paperCharLimit).toBe(100_000);
    expect(DEFAULT_SETTINGS.advanced.sectionCharLimit).toBe(16_000);
  });
});
```

> 若 `validation.test.ts` 顶部已 import 了 `describe/it/expect`，复用即可；`DEFAULT_SETTINGS` 若未导入则加上面那行 import。

- [ ] **Step 2: 运行确认失败**

Run: `cd plugin && npx vitest run tests/validation.test.ts`
Expected: FAIL（当前是 50_000 / 8_000）。

- [ ] **Step 3: 改默认值**

`plugin/src/settings/defaults.ts:34-35`：

```ts
    sectionCharLimit: 16000,
    paperCharLimit: 100_000,
```

- [ ] **Step 4: 运行确认通过 + 全量回归**

Run: `cd plugin && npx vitest run tests/validation.test.ts && npx vitest run`
Expected: 目标用例 PASS；全量 PASS。`diagnostics.test.ts` 不断言具体数值（只断言键存在），不受影响；若有任何用例硬编码了旧默认值，按其语义更新为新值。

- [ ] **Step 5: 提交**

```bash
cd plugin && git add src/settings/defaults.ts tests/validation.test.ts && git commit -m "feat(settings): relax default extraction budgets to 100k/16k" -m "Why: with introduction/related now promoted into the extract, the detail summary needs more room so novelty context does not crowd out numerical evidence." -m "What: bump default paperCharLimit 50k->100k and sectionCharLimit 8k->16k (both remain user-tunable in settings). paperCharLimit is detail-scoped; sectionCharLimit's daily impact is light since abstracts/conclusions rarely exceed 8k." -m "Validation: npx vitest run (full suite green)."
```

---

## Task 8: 详细总结升级为 paper-critic

**Files:**
- Modify: `plugin/src/prompts/paper-detail.system.md`
- Modify: `plugin/src/dashboard/detail-summary.ts:2-9`
- Test: `plugin/tests/summarizer.test.ts`、`plugin/tests/dashboard-detail-summary.test.ts`

- [ ] **Step 1: 改详细总结快照测试为显式断言新结构**

把 Task 4 加的 `detail system prompt matches the golden snapshot` 用例**替换**为显式内容断言（结构变更后快照会变，这里改成断言关键段落，更可读）：

```ts
  it("detail prompt is a structured paper-critic", async () => {
    const calls: any[] = [];
    const llm = {
      call: vi.fn(async (messages: any[]) => {
        calls.push(messages);
        return "## 研究问题\nx";
      }),
    };
    await summarizePaperDetail(
      {
        id: "2606.12345",
        title: "Critic Paper",
        authors: "A. Author",
        abstract: "abstract",
        category: "topic",
        isDetail: true,
        abstractConclusion: "## Abstract\nabstract",
        fullSections: "## Method\nWe model the likelihood.",
      },
      {
        llm: llm as any,
        logger: new Logger("error"),
        arxivSettings: DEFAULT_SETTINGS.arxiv,
        advanced: DEFAULT_SETTINGS.advanced,
        llmTemperature: DEFAULT_SETTINGS.llm.temperature,
      },
    );
    const sys = calls[0][0].content as string;
    expect(sys).toContain("## 贡献与新意");
    expect(sys).toContain("## 阅读价值");
    expect(sys).toContain("精读");
    expect(sys).toContain("略读");
    expect(sys).toContain("记一个点");
    expect(sys).not.toContain("## 一句话价值判断");
    // 反幻觉护栏保留
    expect(sys).toContain("不要引入外部知识");
    expect(sys).toContain("原文未说明");
  });
```

删除旧的 `detail system prompt matches the golden snapshot` 用例，并删除 `tests/__snapshots__/summarizer.test.ts.snap` 中对应的详细总结快照条目（日报快照条目保留）。

- [ ] **Step 2: 运行确认失败**

Run: `cd plugin && npx vitest run tests/summarizer.test.ts`
Expected: FAIL（当前模板无「贡献与新意」「阅读价值」）。

- [ ] **Step 3: 改详细总结模板为 critic 结构**

编辑 `plugin/src/prompts/paper-detail.system.md`：在「## 主要结论」与「## 适用边界」之间插入「## 贡献与新意」，并把末尾「## 一句话价值判断」整段替换为「## 阅读价值」。改后的章节区与注意区如下（其余开头、护栏行保持不变）：

```md
## 主要结论
论文最核心的发现或贡献是什么？区分作者已经证明的结果和作者提出的解释。

## 贡献与新意
相对已有工作，这篇论文新在哪里？是关键技术点，还是已有方法的工程组合？
只依据原文自身（引言、相关工作、摘要）给出的定位与对比来写；原文没有给出与前人的对比，就写"原文未说明"，不要用你自己的知识编造前人工作或差异。

## 适用边界
结论在哪些条件下成立？有哪些限制、不确定性或未覆盖的问题？

## 阅读价值
先给出三选一的分诊结论：**精读** / **略读** / **记一个点**，再用一句话说明理由。避免空泛评价。
```

注意区追加一条（与现有护栏并列，放在「不要写空泛句子」那条之前或之后均可）：

```md
- 「贡献与新意」只能基于原文对自身的定位；原文未与前人对比时写"原文未说明"，禁止编造前人工作
```

- [ ] **Step 4: 运行确认通过**

Run: `cd plugin && npx vitest run tests/summarizer.test.ts`
Expected: PASS。

- [ ] **Step 5: 让检测器标题集跟随新结构（失败测试先行）**

把 `plugin/tests/dashboard-detail-summary.test.ts:14-43` 的 `detects generated detail summaries` 用例里的标题块更新为新结构（用真实新标题），确保检测仍为真：

```ts
    const markdown = [
      "---",
      "type: paper",
      "---",
      "",
      "# A Real Paper Title",
      "",
      "- **arXiv**: [2606.12345](https://arxiv.org/abs/2606.12345)",
      "",
      "## 研究问题",
      repeatedText(2),
      "",
      "## 方法设计",
      repeatedText(2),
      "",
      "## 关键证据",
      repeatedText(2),
      "",
      "## 主要结论",
      repeatedText(2),
      "",
      "## 贡献与新意",
      repeatedText(2),
      "",
      "## 适用边界",
      repeatedText(2),
      "",
      "## 阅读价值",
      repeatedText(1),
    ].join("\n");

    expect(looksLikeDetailSummary(markdown)).toBe(true);
```

- [ ] **Step 6: 运行（此时应仍 PASS，因保留了 ≥3 个原标题）**

Run: `cd plugin && npx vitest run tests/dashboard-detail-summary.test.ts`
Expected: PASS（`looksLikeDetailSummary` 靠 5 个保留原标题命中 ≥3）。

- [ ] **Step 7: 更新 DETAIL_SUMMARY_HEADINGS 跟随真实标题**

`plugin/src/dashboard/detail-summary.ts:2-9` 改为：

```ts
const DETAIL_SUMMARY_HEADINGS = [
  "研究问题",
  "方法设计",
  "关键证据",
  "主要结论",
  "贡献与新意",
  "适用边界",
  "阅读价值",
];
```

- [ ] **Step 8: 全量回归 + 构建**

Run: `cd plugin && npx vitest run && npm run build`
Expected: 全量 PASS；构建 green。

- [ ] **Step 9: 提交**

```bash
cd plugin && git add src/prompts/paper-detail.system.md src/dashboard/detail-summary.ts tests/summarizer.test.ts tests/dashboard-detail-summary.test.ts tests/__snapshots__/summarizer.test.ts.snap && git commit -m "feat(summarizer): upgrade detail summary into a paper-critic" -m "Why: the detail summary lacked explicit evaluation dimensions; add contribution/novelty positioning and a concrete read-priority verdict while keeping all anti-hallucination guardrails." -m "What: add 贡献与新意 (grounded only in the paper's own intro/related/abstract; 原文未说明 when absent) and replace 一句话价值判断 with 阅读价值 (精读/略读/记一个点 + reason); track DETAIL_SUMMARY_HEADINGS to the new structure. Daily report, parser and dashboard data model untouched." -m "Validation: npx vitest run (full suite green); npm run build green."
```

---

## Self-Review

**Spec coverage**
- Part A（抽离）→ Task 1（基建）+ Task 2/3/4（三个 prompt，快照锁零变更）。✓
- Part B（critic 升级 + DETAIL_SUMMARY_HEADINGS）→ Task 8。✓
- Part C（两个 sectionRank 统一 + 放宽限制）→ Task 5（HTML）+ Task 6（LaTeX）+ Task 7（限制）。✓
- esbuild text loader + 运行时无需读盘 → Task 1 Step 6。✓
- vitest 能解析 `.md`（生产模块被测试 import 时需要）→ Task 1 Step 7。✓
- 日报格式 / parser / Dashboard 数据模型不动 → 三处均未在任务中修改；critic 仅动详细总结模板与检测标题集。✓

**Placeholder scan**：无 TBD / TODO；每个 code step 给了完整代码与确切命令、预期输出。✓

**Type/命名一致性**
- `renderPrompt(template, vars)` 签名在 Task 1 定义，Task 2/3/4 调用一致。✓
- import 名 `dailySystemTemplate` / `detailSystemTemplate` / `filterSystemTemplate` 在各自任务内自洽。✓
- 两个 `sectionRank` 的核心证据集合（result/experiment/method/data/limitation/discussion）在 Task 5、6 完全一致。✓
- `DETAIL_SUMMARY_HEADINGS`（Task 8 Step 7）与模板新标题（Task 8 Step 3）、检测测试（Step 5）三者标题集一致。✓

**已知顺序依赖**：Task 4 先把详细总结按原样抽离并打快照，Task 8 再有意改结构（删旧快照条目、改为显式断言）——这是设计内的行为变更，已在 Task 8 Step 1 显式处理。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-15-prompt-templates-paper-critic.md`. 两种执行方式：

1. **Subagent-Driven（skill 默认推荐）** —— 每个 task 派新 subagent、任务间复审。**但本项目 memory 明确禁止派 subagent**，故不适用。
2. **Inline Execution（本项目适用）** —— 在当前会话用 executing-plans 逐任务执行、带检查点复审。

按你项目约束，建议走 **Inline Execution**。要我现在开始执行 Task 1 吗？
