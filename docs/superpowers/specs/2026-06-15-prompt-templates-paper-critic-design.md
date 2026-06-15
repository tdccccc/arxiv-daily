# Prompt 抽离为 Markdown 模板 + 详细总结升级为 paper-critic

- 日期：2026-06-15
- 状态：设计已确认，待写实施计划
- 涉及面：`plugin/` 流水线的 prompt 与章节抽取；不动每日日报格式与 Dashboard 数据模型

## 背景

当前三个 LLM prompt 内联在代码里：

- `plugin/src/pipeline/paper-filter.ts:45` —— 论文筛选 / 分类 / detail 判定
- `plugin/src/pipeline/summarizer.ts:126` —— 每日日报总结
- `plugin/src/pipeline/summarizer.ts:366` —— 单篇详细总结

内联导致 prompt 难以独立查看、diff、迭代。同时详细总结现有结构（研究问题 / 方法设计 / 关键证据 / 主要结论 / 适用边界 / 一句话价值判断）虽有不少好护栏（不编造、缺失写"原文未说明"、优先数值、区分证明 vs 声称、禁空话），但缺少明确的"评价"维度——贡献新意、阅读价值分诊。

## 目标

1. 把三个 prompt 抽离成独立 Markdown 模板，代码只保留薄薄一层模板渲染。**零行为变更**，可逐字节验证。
2. 在抽离后的基线上，把详细总结升级为带评价维度的 paper-critic：problem → method → evidence → results → **contribution/novelty** → limitations → **reading value**。
3. 让"与已有工作的差异（novelty）"有可靠的输入来源——把章节抽取里被压到最末的 introduction / related 提权，并放宽抽取字数预算。

## 非目标（明确排除）

- **不动每日日报**的 5 字段格式（核心问题 / 关键方法 / 主要结果 / 为什么值得看 / 局限或边界）。
- **不动** `daily-summary-parser.ts` 的解析逻辑与 `PaperSummary` 类型。
- **不动** Dashboard 的数据模型与渲染。
- 不统一两个抽取器（HTML / LaTeX 源码）的整体机制，仅统一其章节排序语义（见 Part C）。
- 不为详细总结 / 日报引入各自独立的字数限制（沿用同一组设置项，已确认影响面可接受）。

## Part A：Prompt 抽离（零行为变更）

### 文件

新增 `plugin/src/prompts/`：

- `paper-filter.system.md` —— 占位符 `{{topicLines}}`、`{{tagOptions}}`
- `daily-summary.system.md` —— 占位符 `{{categoryList}}`、`{{partialNote}}`、`{{headerFmt}}`、`{{detailLinkTemplate}}`
- `paper-detail.system.md` —— 占位符 `{{title}}`、`{{id}}`

模板内容 = 各 system prompt 的**逐字原文**，仅把代码里算好的变量替换成占位符。`partialNote` / `headerFmt` 这类条件块仍在代码里算好后作为字符串传入（沿用现有接口）。

### 构建

`plugin/esbuild.config.mjs:9` 的 `common` 配置加：

```js
loader: { ".md": "text" }
```

两个构建上下文（插件 `main.js`、CLI `arxiv-daily-cli.cjs`）都继承 `common`，因此 `.md` 在**构建时内联成字符串**，运行时无需读盘——这是 Obsidian 插件单文件分发下唯一可行的方式。

### 渲染层

新增极薄的 `renderPrompt(template: string, vars: Record<string, string>): string`：

- 纯 `{{key}}` 替换。
- 替换后若仍残留 `{{...}}`，抛错（捕捉占位符拼写错误 / 漏传变量）。
- 位置建议：`plugin/src/prompts/render.ts`。

调用点（`paper-filter.ts`、`summarizer.ts` 三处）改为 `import` 模板文本 + 调 `renderPrompt`。

### 验证闸门（关键）

为三个模板各写快照 / 等值测试：**给定相同输入，`renderPrompt(...)` 的输出与当前内联字符串逐字节相等**。这条通过即证明抽离没有任何行为变化（空行 / 换行细节一并锁住）。Part A 必须先于 Part B/C 合入。

## Part B：详细总结升级为 paper-critic（唯一的输出行为变更，仅详细总结）

改写 `paper-detail.system.md`（原 `summarizer.ts:366`）的章节结构，**保留全部现有护栏**：

```
# {{title}}

- **arXiv**: [{{id}}](https://arxiv.org/abs/{{id}})

## 研究问题        (problem，保留)
## 方法设计        (method，保留)
## 关键证据        (evidence，保留)
## 主要结论        (results，保留，含"证明 vs 声称"区分)
## 贡献与新意      (NEW)
## 适用边界        (limitations，保留)
## 阅读价值        (由"一句话价值判断"升级)
```

### 新增「贡献与新意」（contribution + novelty 合并一段）

- 内容：相对已有工作"新在哪"；是关键技术点还是工程组合。
- **反幻觉约束**：只依据原文自身在引言 / 相关工作 / 摘要里给出的定位与对比来写。原文没有给出与前人的对比，就写"原文未说明"；**禁止用模型自身知识编造前人工作或对比**。（这是允许的，因为从原文引言提取差异属于"读输入"，不属于引入外部知识。）

### 升级「阅读价值」（替换「一句话价值判断」）

- 必须先给出三选一的分诊标签：**精读 / 略读 / 记一个点**，再跟一句理由。
- 目的：从模糊的价值句变成一眼可分诊的结论。

### 保留的护栏（不变）

中文撰写、保留英文术语、LaTeX 公式、只基于输入、缺失写"原文未说明"、优先数值 / 误差 / 样本规模 / 基线对比、区分证明 vs 声称、禁"具有重要意义"类空话。

### 兼容性

- 更新 `plugin/src/dashboard/detail-summary.ts:2` 的 `DETAIL_SUMMARY_HEADINGS`：加入「贡献与新意」「阅读价值」，移除「一句话价值判断」，让检测标题集跟随真实结构。
- 保留的 5 个原标题（研究问题 / 方法设计 / 关键证据 / 主要结论 / 适用边界）使 `looksLikeDetailSummary`（"命中 ≥3 标题" 或 "≥4 段落" 两条任一）稳定为真——即使不更新标题集也不会破，更新只是为了不靠巧合。

## Part C：章节抽取提权（Option A）+ 放宽字数限制

### 两个 sectionRank 要统一语义

抽取有两条路径、两套排序，需改成一致语义：

- **HTML 路径** `plugin/src/pipeline/section-extractor.ts:277` —— 现状 introduction/related 在最末档（rank 3，低于 "other" 的 rank 2）。
- **LaTeX 源码路径** `plugin/src/pipeline/source-extractor.ts:454` —— 现状 `abstract=0, conclusion=1, result=2, method=3, data=4, intro/background=5, 其余(含 related/experiment/discussion/limitation)=10`，方案完全不同。

统一为以下**分层契约**（两路径都改写 `sectionRank` 至此语义；同层内按文档顺序 `index` 决定先后）：

1. **必留**：abstract、conclusion（最高优先；HTML 路径维持现有"预留预算全保留"机制）
2. **核心证据**：result、experiment、method、data、limitation、discussion
3. **背景 / 新意**：introduction、background、related
4. **杂项**：其余

效果：introduction / related 不再最先被砍、且排在杂项之上，为「贡献与新意」提供可靠输入；长论文预算紧张时核心数值证据仍先拿到预算。HTML 与 LaTeX 两条路径取材一致，不因走哪条而漂移。

### 放宽默认限制

`plugin/src/settings/defaults.ts:34`：

- `paperCharLimit`: 50_000 → **100_000**
- `sectionCharLimit`: 8_000 → **16_000**

两者均为 `设置` 面板可调项（`plugin/src/settings/tab.ts:499`、`:509`），此处改的是默认值。

### 影响面（已确认可接受）

- `paperCharLimit` 主要作用于 `extractSections` / `extractLatexSource` → `fullSections`，**仅在 `isDetail` 时构建**（`paper-content.ts:69`、`:92`）。唯一例外是 HTML 结构抽取失败时的兜底 `.slice(0, paperCharLimit)`（`paper-content.ts:81`）。≈ 详细总结专属。
- `sectionCharLimit` 作用于 `extractAbstractConclusion` → `abstractConclusion`，**每篇论文都跑**（`paper-content.ts:60`），是日报对非 detail 论文的主要输入。属全局，但摘要 / 结论很少超 8k，放宽后日报侧实际影响轻微。
- detail 论文的 `fullSections` 也会进入日报中该篇的 block（`summarizer.ts:57`），故放宽 `paperCharLimit` 也会增大这些块；日报有自身 `dailyCharLimit`（400k）+ 分批兜底，不会失控。
- 成本：每篇 detail 论文 LLM 输入可能近翻倍（detail 论文数量少，可接受）。前提是所用模型 context 足够容纳 ~100k 字符输入。

## 实施顺序

1. **Part A**（纯重构）—— 抽离 + esbuild loader + renderPrompt + 快照等值测试。合入后行为零变化。
2. **Part C**（输入管线）—— 两个 sectionRank 统一 + 放宽默认值。先于 B，使 B 看到的输入已含 intro/related。
3. **Part B**（详细总结 prompt）—— 重组维度 + 更新 `DETAIL_SUMMARY_HEADINGS`。

## 验证计划

- **A**：三个模板的快照等值测试（渲染输出 == 当前内联字符串）。
- **C**：两个 `sectionRank` 的单测（核心证据 > 背景/新意 > 杂项）；构造超预算的长论文用例，断言 results/method 仍被保留、intro/related 排在杂项之前被收入。
- **B**：样例详细总结仍被 `looksLikeDetailSummary` 判为真；在 1–2 篇真实论文上肉眼检查输出含「贡献与新意」「阅读价值」，且「阅读价值」带 精读/略读/记一个点 标签。
- 全量：`npm run build`（tsc + esbuild）通过；`npm test` 通过。
```
