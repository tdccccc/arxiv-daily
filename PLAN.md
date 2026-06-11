# arXiv Daily Plan

## Product Direction

定位：

> arXiv Daily = 每日发现 + 论文收件箱 + 后续阅读管理

当前插件已经基本完成“每日发现”：抓取 arXiv、按研究主题筛选、生成日报、为重点论文生成详情页。后续不建议立刻做成单独软件，而是先继续做 Obsidian-native 插件，把论文详情页升级成可长期维护的论文对象。

核心目标不是替代 Zotero、Dataview 或 Obsidian，而是把每天新出现的论文稳定地接入现有科研笔记工作流。

## Glossary

- **Daily discovery**：每天抓取 arXiv，按 topic 筛选和总结，生成日报。
- **Paper inbox**：新发现但还没有被用户处理过的论文集合。这里的 inbox 是一种状态，不一定是一个文件夹。
- **Paper object**：以 arXiv ID 为稳定主键的一篇论文，对应一个 markdown 详情页和一组标准 frontmatter。
- **Reading management**：对论文做后续处理，例如待读、正在读、已读、收藏、忽略、关联 Zotero 或项目笔记。

## Current State

已经具备：

- 按 arXiv category 抓取 `/recent`。
- 按用户配置的 topics 进行 LLM 分类和筛选。
- 生成每日 markdown 日报。
- 为 detail topic 生成单篇论文详情页。
- 支持手动按日期运行、补跑 lookback、按 arXiv ID 生成详情。
- 支持日期级 run state：completed、failed、skipped、running。
- 支持失败日期重试、强制清除日期状态重跑、清空 run state。
- 支持 diagnostics 报告，用于定位配置、日期窗口和运行状态问题。
- release 已自动化，tag 触发 GitHub release asset 构建。

仍然缺少：

- 论文级状态：inbox、to_read、reading、read、saved、ignored。
- 以 arXiv ID 为中心的跨日期去重和更新。
- 论文级 inbox 页面或命令。
- 当前论文状态的快捷修改命令。
- 已收藏但未进 Zotero、待读但未阅读等科研工作流视图。
- PDF、BibTeX、Zotero、项目笔记的结构化接入。

## Design Principles

1. **Markdown/frontmatter 是主数据源**

   论文详情页应该在没有插件时也能读、能迁移、能被 Dataview 查询。`data.json` 只保存运行状态、缓存和必要索引，不作为唯一真相来源。

2. **Obsidian 优先，不先做单独软件**

   现阶段用户价值来自和 Obsidian vault、双链、搜索、Dataview、同步工具的自然结合。单独软件会引入数据库、同步、UI 和发布维护成本，应等需求明显超出 Obsidian 后再考虑。

3. **论文为中心，日报只是视图**

   当前以日期为中心：某天是否跑过、某天生成了什么。下一阶段要把论文变成稳定对象：这篇论文何时发现、属于什么 topic、当前处理状态是什么、是否值得收藏。

4. **轻量状态机，不做复杂项目管理**

   状态字段应该足够支持科研阅读管理，但避免一开始做成复杂任务系统。

5. **先结构化，再做集成**

   Zotero、PDF、引用和项目笔记都依赖稳定的论文 metadata。先把 frontmatter 和去重做好，再做外部工具接入。

## Proposed Paper Frontmatter

论文详情页建议使用如下字段：

```yaml
---
type: paper
source: arxiv
arxiv_id: "2606.12345"
title: "Example Paper Title"
authors:
  - "A. Author"
published: "2026-06-11"
updated: "2026-06-11"
category: "astro-ph"
topics:
  - "photo-z"
primary_topic: "photo-z"
detail: true
status: inbox
priority: normal
seen_dates:
  - "2026-06-11"
daily_reports:
  - "arxiv-daily/daily/2026-06-11.md"
arxiv_url: "https://arxiv.org/abs/2606.12345"
pdf_url: "https://arxiv.org/pdf/2606.12345"
pdf_path: ""
zotero_key: ""
citation_key: ""
projects: []
tags:
  - arxiv
  - photo-z
---
```

### Status Values

| Status | Meaning |
|---|---|
| `inbox` | 新发现，还没决定是否要读 |
| `to_read` | 已决定之后要读 |
| `reading` | 正在读 |
| `read` | 已读完或已处理 |
| `saved` | 值得长期收藏，可能进 Zotero 或项目笔记 |
| `ignored` | 不感兴趣，之后日报中可以弱化或隐藏 |

### Priority Values

| Priority | Meaning |
|---|---|
| `low` | 有一定相关性，但不急 |
| `normal` | 默认优先级 |
| `high` | 高价值或近期需要阅读 |

## v0.2.0: Paper Inbox Layer

目标：在不大改现有工作流的前提下，让每篇论文成为可管理的稳定对象。

### Scope

1. 标准化论文详情页 frontmatter。
2. 以 `arxiv_id` 去重：
   - 已存在论文页时不重复创建。
   - 追加 `seen_dates`。
   - 追加 `daily_reports`。
   - 保留用户手动修改过的 `status`、`priority`、`projects`、`zotero_key` 等字段。
3. 新论文默认写入 `status: inbox` 和 `priority: normal`。
4. 日报中区分 new / seen before。
5. 增加论文状态命令：
   - `Mark current paper as to read`
   - `Mark current paper as reading`
   - `Mark current paper as read`
   - `Mark current paper as saved`
   - `Mark current paper as ignored`
6. 增加 inbox 视图：
   - 命令生成或打开 `arxiv-daily/inbox.md`。
   - 列出 `status: inbox` 的论文。
   - 按 topic、priority、published date 排序。

### Acceptance Criteria

- 同一 arXiv ID 多次出现在 lookback 或不同日期时，只保留一个论文详情页。
- 用户把论文标记为 `ignored` 后，后续再次出现不会被当作新论文提醒。
- 用户修改过的状态字段不会被重新生成详情页时覆盖。
- Dataview 可以直接查询未处理论文：

```dataview
TABLE published, primary_topic, priority
FROM "arxiv-daily/papers"
WHERE type = "paper" AND status = "inbox"
SORT published DESC
```

## v0.3.0: Reading Workflow

目标：把 inbox 变成日常科研阅读入口。

可做功能：

- 生成 `to-read.md`、`saved.md`、`recent-high-priority.md` 等工作流页面。
- 支持批量状态修改，例如把一组低相关论文标记为 ignored。
- 支持设置默认策略：
  - detail 论文默认 `to_read`，非 detail 默认 `inbox`。
  - 某些 topic 默认 high priority。
- 支持重新分类某篇论文的 topic。
- diagnostics 增加 paper frontmatter 检查：
  - 缺少 `arxiv_id`。
  - 重复 arXiv ID。
  - 非法 status。
  - `seen_dates` 格式错误。

## v0.4.0: Research Tool Integrations

目标：连接 Zotero、PDF 和引用管理，但不替代它们。

优先级建议：

1. **BibTeX / citation helper**
   - 从 arXiv 导出 BibTeX。
   - 写入 `citation_key`。
   - 支持复制引用片段。

2. **Zotero bridge**
   - 先支持手动字段：`zotero_key`、`zotero_uri`。
   - 再考虑 Better BibTeX citekey 或 Zotero local API。
   - 视图中列出 `saved` 但没有 `zotero_key` 的论文。

3. **PDF management**
   - 可选下载 arXiv PDF。
   - 写入 `pdf_path`。
   - 避免默认大量下载，先做手动命令。

4. **Project notes**
   - 支持 `projects` frontmatter。
   - 允许把论文链接追加到某个项目笔记。

## Possible Obsidian Workflows

### Daily Triage

1. 每天自动生成日报。
2. 打开日报，快速扫 new papers。
3. 对感兴趣论文执行 `Mark as to read` 或 `Mark as saved`。
4. 对无关论文执行 `Mark as ignored`。
5. 关闭日报，后续从 inbox/to-read 页面继续处理。

### Weekly Review

1. 打开 `arxiv-daily/inbox.md`。
2. 处理过去一周未决论文。
3. 把高价值论文转为 `saved`。
4. 把近期要读的论文转为 `to_read`。
5. 把无关论文转为 `ignored`。

### Zotero Follow-up

1. 查询 `status: saved` 且 `zotero_key` 为空的论文。
2. 逐篇导入 Zotero。
3. 把 Zotero citekey 写回论文 frontmatter。
4. 项目笔记中引用 Obsidian 论文页和 Zotero citekey。

## Technical Notes

### Frontmatter Updates

需要避免简单字符串拼接。建议使用 Obsidian metadata/frontmatter API 或成熟 YAML parser 来更新 frontmatter，确保：

- 保留用户手写内容。
- 不破坏正文。
- 数组字段去重。
- 缺失字段补齐。
- 用户字段不被覆盖。

### Paper Index

可以增加派生缓存，但不要让缓存成为唯一数据源。

可选结构：

```ts
interface PaperIndexEntry {
  arxivId: string;
  path: string;
  status: string;
  priority: string;
  primaryTopic: string;
  seenDates: string[];
  updatedAt: number;
}
```

用途：

- 加速 inbox modal 或命令。
- diagnostics 快速发现重复和缺字段。
- 不用于取代 markdown frontmatter。

### Writer Behavior

生成或更新论文页时应区分两类字段：

- 插件可更新字段：`seen_dates`、`daily_reports`、`updated`、`topics`。
- 用户控制字段：`status`、`priority`、`projects`、`zotero_key`、`citation_key`、正文笔记。

默认策略：只补缺，不覆盖用户控制字段。

## When to Consider a Standalone App

短期不建议做单独软件。只有出现以下需求时再评估：

- 需要后台常驻自动运行，不依赖 Obsidian 打开。
- 需要复杂多列 UI、批量拖拽、跨库聚合。
- 需要多设备实时同步且不依赖 vault 文件同步。
- 需要数据库级查询和大规模历史分析。
- 需要同时接 arXiv、ADS、Semantic Scholar、RSS、Zotero 等多个来源。

即使未来做单独软件，也应优先把当前插件里的抓取、分类、去重、frontmatter 逻辑抽成可复用核心，而不是重写。

## Recommended Next Step

下一步建议开 `v0.2.0`，只做 Paper Inbox Layer 的最小闭环：

1. 论文详情页 frontmatter 标准化。
2. `arxiv_id` 去重和已有论文页更新。
3. `status` / `priority` / `seen_dates` / `daily_reports` 字段。
4. 当前论文状态修改命令。
5. 一个可生成的 inbox 页面。

这个范围足够小，和当前架构连续，但能明显改变工作流：从“每天读一篇日报”变成“长期维护一个可筛选、可追踪、可接入 Zotero 的论文收件箱”。
