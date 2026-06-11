# arXiv Daily Plan

## Product Direction

定位：

> arXiv Daily = 每日发现 + 论文收件箱 + 后续阅读管理

当前插件已经基本完成“每日发现”：抓取 arXiv、按研究主题筛选、生成日报、为重点论文生成详情页。后续不建议立刻做成单独软件，而是先继续做 Obsidian-native 插件，用 JSON 维护完整 paper inbox，只为重要论文生成长期 markdown 笔记。

核心目标不是替代 Zotero、Dataview 或 Obsidian，而是把每天新出现的论文稳定地接入现有科研笔记工作流。

## Glossary

- **Daily discovery**：每天抓取 arXiv，按 topic 筛选和总结，生成日报。
- **Paper inbox**：新发现但还没有被用户处理过的论文集合。这里的 inbox 是一种状态，不一定是一个文件夹。
- **Paper object**：以 arXiv ID 为稳定主键的一篇论文记录。默认存放在 `papers.json` 中；只有 detail、high priority、saved 或用户主动创建笔记的论文才对应 markdown 详情页。
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

1. **JSON 索引是 paper inbox 主数据源**

   `arxiv-daily/index/papers.json` 保存所有被筛选为相关的论文记录。不要把长期 paper inbox 放进 `.obsidian/plugins/arxiv-daily/data.json`，避免插件安装、更新或手动覆盖时误伤长期数据。

2. **Obsidian 优先，不先做单独软件**

   现阶段用户价值来自和 Obsidian vault、双链、搜索、Dataview、同步工具的自然结合。单独软件会引入数据库、同步、UI 和发布维护成本，应等需求明显超出 Obsidian 后再考虑。

3. **论文为中心，日报只是视图**

   当前以日期为中心：某天是否跑过、某天生成了什么。下一阶段要把论文变成稳定对象：这篇论文何时发现、属于什么 topic、当前处理状态是什么、是否值得收藏。markdown 详情页是重要论文的长期笔记，不是所有相关论文的必要载体。

4. **轻量状态机，不做复杂项目管理**

   状态字段应该足够支持科研阅读管理，但避免一开始做成复杂任务系统。

5. **先结构化，再做集成**

   Zotero、PDF、引用和项目笔记都依赖稳定的论文 metadata。先把 JSON schema、去重和状态流转做好，再做外部工具接入。

## Proposed Paper Index

完整 paper inbox 建议保存在 vault 可见目录：

```text
arxiv-daily/index/papers.json
```

建议结构：

```json
{
  "schemaVersion": 1,
  "updatedAt": "2026-06-11T01:30:00.000Z",
  "papers": {
    "2606.12345": {
      "arxivId": "2606.12345",
      "source": "arxiv",
      "title": "Example Paper Title",
      "authors": ["A. Author"],
      "published": "2026-06-11",
      "updated": "2026-06-11",
      "category": "astro-ph",
      "topics": ["photo-z"],
      "primaryTopic": "photo-z",
      "detail": true,
      "status": "inbox",
      "priority": "normal",
      "seenDates": ["2026-06-11"],
      "dailyReports": ["arxiv-daily/daily/2026-06-11.md"],
      "paperPath": "arxiv-daily/papers/2606.12345.md",
      "arxivUrl": "https://arxiv.org/abs/2606.12345",
      "pdfUrl": "https://arxiv.org/pdf/2606.12345",
      "pdfPath": "",
      "zoteroKey": "",
      "citationKey": "",
      "projects": []
    }
  }
}
```

### Paper Note Creation

不是每篇相关论文都需要创建 markdown。默认策略建议为：

| Case | Create markdown note |
|---|---|
| arXiv 当天全部论文 | No |
| LLM 判断相关但非重点论文 | No, only index in JSON |
| `detail: true` | Yes |
| `priority: high` | Yes |
| `status: saved` | Yes |
| 用户执行 `Create paper note` | Yes |

轻量相关论文留在 `papers.json`，避免 vault 每天增加大量低价值文件。重要论文进入 `arxiv-daily/papers/<arxiv_id>.md`，用于长期阅读笔记、项目链接和 Zotero 跟踪。

### Paper Note Frontmatter

生成 markdown note 时，frontmatter 应该从 `papers.json` 派生，并只写必要字段：

JSON 内部字段使用 camelCase；markdown frontmatter 可以继续使用更适合 Obsidian 查询和人工阅读的 snake_case。

```yaml
---
type: paper
source: arxiv
arxiv_id: "2606.12345"
status: saved
priority: high
primary_topic: photo-z
seen_dates:
  - "2026-06-11"
zotero_key: ""
citation_key: ""
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

1. 新增 `arxiv-daily/index/papers.json`，作为完整 paper inbox 主索引。
2. 以 `arxiv_id` 去重和合并：
   - 已存在记录时不重复创建。
   - 追加 `seenDates`。
   - 追加 `dailyReports`。
   - 更新插件可维护字段，例如 `topics`、`detail`、`updated`。
   - 保留用户控制字段，例如 `status`、`priority`、`projects`、`zoteroKey`、`citationKey`。
3. 新相关论文默认写入 `status: inbox` 和 `priority: normal`。
4. 只在必要时创建 markdown note：
   - `detail: true`。
   - `priority: high`。
   - `status: saved`。
   - 用户手动执行 `Create paper note`。
5. 日报中区分 new / seen before，并可以弱化已 `ignored` 的论文。
6. 增加论文状态命令：
   - `Mark current paper as to read`
   - `Mark current paper as reading`
   - `Mark current paper as read`
   - `Mark current paper as saved`
   - `Mark current paper as ignored`
   - `Create paper note`
7. 增加 inbox 视图：
   - 命令生成或打开 `arxiv-daily/inbox.md`。
   - 从 `papers.json` 列出 `status: inbox` 的论文。
   - 按 topic、priority、published date 排序。

### Acceptance Criteria

- 同一 arXiv ID 多次出现在 lookback 或不同日期时，`papers.json` 中只保留一条记录。
- 重复出现的论文会追加 `seenDates` 和 `dailyReports`，不会丢失用户状态。
- 非 detail / 非 high priority / 未 saved 的普通相关论文默认不创建 markdown note。
- detail、high priority、saved 或用户主动创建的论文会拥有 `paperPath` 和 markdown note。
- 用户把论文标记为 `ignored` 后，后续再次出现不会被当作新论文提醒。
- 用户修改过的状态字段不会被后续日报生成覆盖。
- inbox 页面可以从 JSON 生成，列出未处理论文：

```markdown
# arXiv Daily Inbox

## Photo-z

- [ ] 2606.12345 Example Paper Title
  - status: inbox
  - priority: normal
  - arXiv: https://arxiv.org/abs/2606.12345
```

## v0.3.0: Reading Workflow

目标：把 inbox 变成日常科研阅读入口。

可做功能：

- 生成 `to-read.md`、`saved.md`、`recent-high-priority.md` 等工作流页面。
- 支持批量状态修改，例如把一组低相关论文标记为 ignored。
- 支持设置默认策略：
  - detail 论文默认 `to_read` 并创建 markdown note。
  - 非 detail 论文默认 `inbox` 且只进入 JSON。
  - 某些 topic 默认 high priority。
- 支持重新分类某篇论文的 topic。
- diagnostics 增加 paper index / note consistency 检查：
  - `papers.json` schema 版本不支持。
  - 重复 arXiv ID。
  - 非法 status。
  - `seenDates` 格式错误。
  - `paperPath` 指向的 markdown note 不存在。
  - markdown note 中的 `arxiv_id` 和 JSON 不一致。

## v0.4.0: Research Tool Integrations

目标：连接 Zotero、PDF 和引用管理，但不替代它们。

优先级建议：

1. **BibTeX / citation helper**
   - 从 arXiv 导出 BibTeX。
   - 写入 `citationKey`。
   - 支持复制引用片段。

2. **Zotero bridge**
   - 先支持手动字段：`zoteroKey`、`zoteroUri`。
   - 再考虑 Better BibTeX citekey 或 Zotero local API。
   - 视图中列出 `saved` 但没有 `zoteroKey` 的论文。

3. **PDF management**
   - 可选下载 arXiv PDF。
   - 写入 `pdfPath`。
   - 避免默认大量下载，先做手动命令。

4. **Project notes**
   - 支持 `projects` 字段。
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

1. 从 `papers.json` 查询 `status: saved` 且 `zoteroKey` 为空的论文。
2. 逐篇导入 Zotero。
3. 把 Zotero citekey 写回 `papers.json`。
4. 项目笔记中引用 Obsidian 论文页和 Zotero citekey。

## Technical Notes

### JSON Index Updates

`papers.json` 是长期 paper inbox 主索引。更新时需要做到：

- 原子写入，避免写一半损坏 JSON。
- 读入后按 `schemaVersion` 迁移。
- `papers` 以 arXiv ID 为 key。
- `seenDates`、`dailyReports`、`topics` 数组去重。
- 插件自动字段可更新，用户控制字段只补缺不覆盖。
- JSON parse 失败时保留原文件并给出 diagnostics。

### Markdown Note Updates

只有重要论文才创建 markdown note。note 更新需要避免简单字符串拼接。建议使用 Obsidian metadata/frontmatter API 或成熟 YAML parser，确保：

- 保留用户手写正文。
- 不破坏已有 frontmatter。
- 缺失字段补齐。
- 用户控制字段不被覆盖。
- JSON 和 markdown 的关键字段保持一致。

### Paper Index Schema

建议 TypeScript 结构：

```ts
interface PaperInbox {
  schemaVersion: 1;
  updatedAt: string;
  papers: Record<string, PaperIndexEntry>;
}

interface PaperIndexEntry {
  arxivId: string;
  source: "arxiv";
  title: string;
  authors: string[];
  published: string;
  updated: string;
  category: string;
  topics: string[];
  primaryTopic: string;
  detail: boolean;
  status: "inbox" | "to_read" | "reading" | "read" | "saved" | "ignored";
  priority: "low" | "normal" | "high";
  seenDates: string[];
  dailyReports: string[];
  paperPath: string | null;
  arxivUrl: string;
  pdfUrl: string;
  pdfPath: string;
  zoteroKey: string;
  citationKey: string;
  projects: string[];
}
```

用途：

- 加速 inbox modal 或命令。
- diagnostics 快速发现重复和缺字段。
- 支持后续 Zotero / PDF / project note 接入。

### Writer Behavior

生成或更新 JSON / markdown note 时应区分两类字段：

- 插件可更新字段：`seenDates`、`dailyReports`、`updated`、`topics`、`detail`。
- 用户控制字段：`status`、`priority`、`projects`、`zoteroKey`、`citationKey`、正文笔记。

默认策略：只补缺，不覆盖用户控制字段。

## When to Consider a Standalone App

短期不建议做单独软件。只有出现以下需求时再评估：

- 需要后台常驻自动运行，不依赖 Obsidian 打开。
- 需要复杂多列 UI、批量拖拽、跨库聚合。
- 需要多设备实时同步且不依赖 vault 文件同步。
- 需要数据库级查询和大规模历史分析。
- 需要同时接 arXiv、ADS、Semantic Scholar、RSS、Zotero 等多个来源。

即使未来做单独软件，也应优先把当前插件里的抓取、分类、去重、JSON schema 和 note 生成逻辑抽成可复用核心，而不是重写。

## Recommended Next Step

下一步建议开 `v0.2.0`，只做 Paper Inbox Layer 的最小闭环：

1. 新增 `arxiv-daily/index/papers.json`。
2. `arxiv_id` 去重和已有 JSON 记录更新。
3. `status` / `priority` / `seenDates` / `dailyReports` 字段。
4. 当前论文状态修改命令。
5. detail / high priority / saved 论文的 markdown note 创建策略。
6. 一个从 JSON 生成的 inbox 页面。

这个范围足够小，和当前架构连续，也不会让 vault 每天增加大量低价值 markdown 文件。工作流会从“每天读一篇日报”变成“长期维护一个可筛选、可追踪、可按需生成笔记、可接入 Zotero 的论文收件箱”。
