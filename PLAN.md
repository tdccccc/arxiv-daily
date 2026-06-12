# arXiv Daily Plan

## Product Direction

定位：

> arXiv Daily = 每日发现 + 日报内筛选 + Reading Dashboard + 后续阅读管理

当前插件已经基本完成“每日发现”：抓取 arXiv、按研究主题筛选、生成日报、为重点论文生成详情页。下一阶段的重点不是增加一个单独 inbox 页面，而是把日报变成自然筛选入口：用户读日报时直接勾选“关注”或“重点”，插件把选择同步到内部 JSON 索引；只有重要论文才生成长期 markdown 笔记。

用户对 GUI 的需求是成立的：当关注、重点、saved、read 等状态积累起来后，不能要求用户逐篇日报回翻。后续应先做 Obsidian 内的 Reading Dashboard，用表格、筛选、搜索和汇总视图消费 `papers.json`；独立软件只有在需求明显超出 Obsidian 后再评估。

核心目标不是替代 Zotero、Dataview 或 Obsidian，而是把每天新出现的论文稳定地接入现有科研笔记工作流，并提供一个能回看、检索和批量处理论文状态的工作台。

## Glossary

- **Daily discovery**：每天抓取 arXiv，按 topic 筛选和总结，生成日报。
- **Paper index**：以 JSON 保存的论文级状态和去重索引，不是日常阅读入口。
- **Paper object**：以 arXiv ID 为稳定主键的一篇论文记录。默认存放在 `papers.json` 中；只有 detail、high priority、saved 或用户主动创建笔记的论文才对应 markdown 详情页。
- **Reading Dashboard**：Obsidian 内的插件视图，用于检索、筛选、回看、汇总和批量更新论文状态；它读取 `papers.json`，但不替代 Markdown 编辑器。
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

仍然可以继续加强：

- 更好的周回顾视图或搜索方式，用于查看已勾选“关注/重点”的论文。
- Obsidian 内 GUI Dashboard：按状态、priority、topic、日期范围、是否有 note / Zotero 字段筛选论文。
- 对关注/重点论文的批量状态修改、快速打开日报/论文笔记、周度汇总。
- 已收藏但未进 Zotero、待读但未阅读等科研工作流视图。
- PDF、BibTeX、Zotero、项目笔记的结构化接入。

## Design Principles

1. **JSON 索引是内部状态主数据源**

   `arxiv-daily/.index/papers.json` 保存所有被筛选为相关的论文记录。不要把长期论文状态放进 `.obsidian/plugins/arxiv-daily/data.json`，避免插件安装、更新或手动覆盖时误伤长期数据。

2. **Obsidian 优先，不先做单独软件**

   现阶段用户价值来自和 Obsidian vault、双链、搜索、Dataview、同步工具的自然结合。单独软件会引入数据库、同步、UI 和发布维护成本，应等需求明显超出 Obsidian 后再考虑。

3. **GUI 先做成 Obsidian 内工作台**

   需要 GUI 来解决回看、检索、汇总和批量处理，但第一版 GUI 应该是 Obsidian custom view。Markdown 阅读、编辑、双链、文件同步继续交给 Obsidian，Dashboard 只负责把 `papers.json` 变成可操作的论文列表和回顾视图。

4. **论文为中心，日报只是视图**

   当前以日期为中心：某天是否跑过、某天生成了什么。下一阶段要把论文变成稳定对象：这篇论文何时发现、属于什么 topic、当前处理状态是什么、是否值得收藏。markdown 详情页是重要论文的长期笔记，不是所有相关论文的必要载体。

5. **轻量状态机，不做复杂项目管理**

   状态字段应该足够支持科研阅读管理，但避免一开始做成复杂任务系统。

6. **先结构化，再做集成**

   Zotero、PDF、引用和项目笔记都依赖稳定的论文 metadata。先把 JSON schema、去重和状态流转做好，再做外部工具接入。

## Storage Layout

JSON-first 只针对论文级状态索引，不改变日报作为 markdown 入口的定位。

建议目录结构：

```text
arxiv-daily/
  daily/
    2026-06-11.md
  .index/
    papers.json
  papers/
    2606.12345.md
```

职责划分：

| Path | Role | Created by default |
|---|---|---|
| `arxiv-daily/daily/YYYY-MM-DD.md` | 每日发现入口，按 topic 展示当天相关论文 | Yes |
| `arxiv-daily/.index/papers.json` | 插件内部状态：所有相关论文的状态、去重、seen dates 和外部工具字段；默认隐藏 | Yes |
| `arxiv-daily/papers/<arxiv_id>.md` | 重要论文的长期阅读笔记和深度分析 | Only for detail / high priority / saved / manual |

日报仍然是每天最主要的阅读入口。它应该从当天 pipeline 结果和 `papers.json` 共同生成：

- 按 topic 分组展示当天相关论文。
- 标明论文是 `new` 还是 `seen before`。
- 对 `ignored` 论文弱化或默认隐藏，具体策略可以后续配置。
- 如果论文已有 markdown note，链接到 `paperPath`。
- 如果论文没有 markdown note，展示 arXiv 链接、摘要和一句话结论。
- 将日报路径写回对应论文记录的 `dailyReports`。

## Proposed Paper Index

完整 paper index 建议保存在 vault 可见目录：

```text
arxiv-daily/.index/papers.json
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

## v0.2.0: Daily Selection Layer

目标：把日报变成论文筛选的主操作界面。用户打开当天日报，快速扫过大多数论文，只对少数感兴趣或重点关注的论文打勾；插件自动把勾选结果同步到 `papers.json`。日常操作闭环直接发生在日报里。

核心原则：

- **日报是主入口**：用户不应该读完日报后再去 inbox 里找同一批论文处理。
- **只做正向筛选**：大部分论文扫过不动；少数勾“关注”；极少数勾“重点”。
- **勾选即同步**：checkbox 修改后自动更新 `papers.json`，不要求用户手动运行同步命令。
- **index 是内部状态**：`papers.json` 用于去重、状态和后续集成，不作为日常阅读文件。

### Scope

1. 新增 `arxiv-daily/.index/papers.json`，作为完整 paper index 主索引；旧的 `arxiv-daily/index/papers.json` 会兼容读取，并在下次保存后迁移到隐藏路径。
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
5. 日报继续创建 markdown，并从 `papers.json` 标注 new / seen before，弱化已 `ignored` 的论文。
6. 日报中每篇论文加入两个轻量 checkbox：

```markdown
- [ ] 关注 <!-- arxiv-daily:2606.12345:watch -->
- [ ] 重点 <!-- arxiv-daily:2606.12345:highlight -->
```

   - 未勾选：保持普通 `inbox`。
   - 勾选关注：`status: to_read`, `priority: normal`。
   - 勾选重点：`status: to_read`, `priority: high`。
   - 重点不自动生成长笔记，避免 checkbox 带来昂贵副作用；需要笔记时仍用 detail 或手动创建。
7. 插件监听 daily markdown 文件修改，解析 checkbox 标记并自动同步到 `papers.json`。
8. 保留论文状态命令作为高级操作，但它不是主要筛选路径；默认入口只保留日报和单篇论文笔记。

### Acceptance Criteria

- 同一 arXiv ID 多次出现在 lookback 或不同日期时，`papers.json` 中只保留一条记录。
- 重复出现的论文会追加 `seenDates` 和 `dailyReports`，不会丢失用户状态。
- 非 detail / 非 high priority / 未 saved 的普通相关论文默认不创建 markdown note。
- detail、high priority、saved 或用户主动创建的论文会拥有 `paperPath` 和 markdown note。
- 用户把论文标记为 `ignored` 后，后续再次出现不会被当作新论文提醒。
- 用户修改过的状态字段不会被后续日报生成覆盖。
- 每个成功运行的日期仍会创建或保留 `arxiv-daily/daily/YYYY-MM-DD.md`。
- 日报中没有 markdown note 的普通论文仍可通过 arXiv 链接访问。
- 日报中的 `关注` / `重点` checkbox 被勾选后，无需手动同步命令，插件会自动更新 `papers.json`。
- checkbox 取消勾选时，如果论文仍是插件默认的 `to_read` 状态，则回到 `status: inbox`；如果用户已手动设置为 `saved` / `read` / `ignored`，不自动降级。
- `arxiv-daily/.index/papers.json` 是内部状态文件，默认隐藏，日常不需要在文件树中打开。

## v0.3.0: Obsidian Reading Dashboard

目标：做一个 Obsidian 内 GUI 工作台，让用户不用逐篇日报回翻，就能查看、搜索、汇总和处理已关注、重点、saved、read、ignored 等论文。

核心原则：

- **Dashboard 是主回顾入口**：日报负责当天 triage，Dashboard 负责跨日期回看。
- **不重做 Markdown 编辑器**：论文笔记仍在 Obsidian editor 中打开，Dashboard 只提供列表、筛选和状态操作。
- **先本地、后集成**：第一版只读写 `papers.json` 和 vault markdown，不引入数据库或外部同步。
- **可批量处理**：关注/重点论文多起来后，必须支持多选和批量状态修改。

### Scope

1. 新增 `arXiv Daily: Open reading dashboard` 命令和 ribbon 菜单入口。
2. 新增 Obsidian custom view，例如 `arxiv-daily-dashboard`。
3. Dashboard 从 `PaperIndexStore` 读取数据，提供这些 tabs：
   - `关注`：`status: to_read` 且 `priority !== high`。
   - `重点`：`priority: high`。
   - `正在读`：`status: reading`。
   - `已收藏`：`status: saved`。
   - `已读`：`status: read`。
   - `全部`：除 ignored 以外的所有论文。
   - `忽略`：`status: ignored`。
4. 提供筛选：
   - topic。
   - date range（按 `published` / `seenDates`）。
   - status。
   - priority。
   - 是否有 `paperPath`。
   - 是否 `detail`。
   - 是否缺 `zoteroKey` / `citationKey`。
5. 提供搜索：
   - arXiv ID。
   - title。
   - authors。
   - topic/tag。
   - 后续可加入日报摘要片段或结构化 summary 字段。
6. 表格列建议：
   - checkbox。
   - priority。
   - status。
   - title。
   - topic。
   - published / first seen。
   - note。
   - arXiv / PDF。
7. 行内操作：
   - `to_read` / `reading` / `read` / `saved` / `ignored`。
   - 创建或打开 paper note。
   - 打开首次出现的 daily report。
   - 复制 arXiv / PDF / citation placeholder。
8. 批量操作：
   - 标记为 ignored。
   - 标记为 read。
   - 标记为 saved。
   - 设置 priority。
   - 为 selected 创建 lightweight notes（必须确认）。
9. 汇总区：
   - 当前筛选结果数量。
   - 按 topic 统计。
   - 按 status / priority 统计。
   - 本周新增、已关注、重点、saved 数量。
   - saved 但缺 Zotero / citation key 的数量。
10. diagnostics 增加 paper index / note consistency 检查：
    - `papers.json` schema 版本不支持。
    - 非法 status / priority。
    - `seenDates` 格式错误。
    - `paperPath` 指向的 markdown note 不存在。
    - markdown note 中的 `arxiv_id` 和 JSON 不一致。

### Acceptance Criteria

- 用户可以从命令面板或 ribbon 打开 Dashboard。
- Dashboard 不需要打开任何日报，也能列出所有 `关注` 和 `重点` 论文。
- 搜索 `arxivId` / 标题 / 作者任意关键词可以过滤结果。
- 用户可以按 topic、状态、priority、日期范围筛选。
- 用户可以单篇或批量修改 status / priority，保存到 `papers.json`。
- 用户可以从 Dashboard 打开 paper note、创建 lightweight note、打开 daily report。
- Dashboard 的汇总数字与当前筛选结果一致。
- Dashboard 不创建新的 markdown 回顾页面作为主入口；如后续需要，可以提供“导出当前视图为 Markdown”命令。
- 所有状态修改都保留用户手写的 paper note 内容。

## v0.4.0: Core Extraction + CLI Fallback

目标：把抓取、分类、去重、总结、索引这些核心逻辑抽成可复用 core，为 cron/headless 和未来独立客户端打基础。

可做功能：

- 将 Obsidian 相关依赖隔离在 adapter 层：
  - `requestUrl` -> `HttpClient`。
  - `Vault` -> `StorageAdapter`。
  - `Notice` / status bar -> `ProgressReporter`。
- 让 Node CLI 复用同一套 pipeline：
  - `arxiv-daily run --date YYYY-MM-DD`。
  - `arxiv-daily run-pending`。
  - `arxiv-daily summarize --id 2606.12345`。
- 可选把 run state 写进 vault 输出目录，避免 CLI 和 Obsidian scheduler 重复跑。
- 保持 Obsidian 插件仍是主用户界面。

## v0.5.0: Research Tool Integrations

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
3. 大部分论文不动。
4. 对感兴趣论文勾选 `关注`。
5. 对特别重要论文勾选 `重点`。
6. 插件自动把勾选结果同步到 `papers.json`。
7. 关闭日报；后续需要时从单篇笔记、搜索或未来的回顾视图继续处理少数被挑出的论文。

### Weekly Review

1. 打开 Reading Dashboard。
2. 切到 `关注` 或 `重点` tab，按本周日期范围过滤。
3. 回顾已勾选关注/重点的论文。
4. 把高价值论文转为 `saved`。
5. 把近期正在读的论文转为 `reading` 或 `read`。
6. 批量把无关论文转为 `ignored`。

### Zotero Follow-up

1. 从 `papers.json` 查询 `status: saved` 且 `zoteroKey` 为空的论文。
2. 逐篇导入 Zotero。
3. 把 Zotero citekey 写回 `papers.json`。
4. 项目笔记中引用 Obsidian 论文页和 Zotero citekey。

## Technical Notes

### JSON Index Updates

`papers.json` 是长期 paper index 主索引。更新时需要做到：

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
interface PaperIndex {
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

- 支撑日报勾选状态、跨日期去重和后续周回顾。
- diagnostics 快速发现重复和缺字段。
- 支持后续 Zotero / PDF / project note 接入。

### Writer Behavior

生成或更新 JSON / markdown note 时应区分两类字段：

- 插件可更新字段：`seenDates`、`dailyReports`、`updated`、`topics`、`detail`。
- 用户控制字段：`status`、`priority`、`projects`、`zoteroKey`、`citationKey`、正文笔记。

默认策略：只补缺，不覆盖用户控制字段。

### Dashboard Architecture

Dashboard 应该被实现为 Obsidian custom view，而不是生成一个长期维护的 `inbox.md` 文件：

- `PaperIndexStore` 继续是唯一状态源。
- Dashboard 启动时 load index，状态更新后 save index 并局部刷新。
- 表格过滤、搜索、排序先在内存中完成；数据量到几千篇前不需要数据库。
- Paper note 的阅读和编辑通过 `workspace.openLinkText` 打开 Obsidian markdown editor。
- Dashboard 不直接编辑 note 正文；只更新 `papers.json` 和必要 frontmatter。
- 如果未来做独立软件，优先复用 query/filter/action 这些 dashboard model 层，而不是复用 Obsidian DOM 视图。

## When to Consider a Standalone App

GUI 需求成立，但短期不建议做单独软件。先把 GUI 做成 Obsidian 内 Dashboard。只有出现以下需求时再评估独立客户端：

- 需要后台常驻自动运行，不依赖 Obsidian 打开。
- 需要复杂多列 UI、批量拖拽、跨库聚合。
- 需要面向不使用 Obsidian 的用户。
- 需要内置完整 Markdown 文件阅读/编辑、同步和冲突处理，而不是复用 Obsidian。
- 需要多设备实时同步且不依赖 vault 文件同步。
- 需要数据库级查询和大规模历史分析。
- 需要同时接 arXiv、ADS、Semantic Scholar、RSS、Zotero 等多个来源。

即使未来做单独软件，也应优先把当前插件里的抓取、分类、去重、JSON schema 和 note 生成逻辑抽成可复用核心，而不是重写。

## Recommended Next Step

下一步建议按两段推进：

**阶段 1：收尾 Daily Selection Layer**

1. 日报每篇论文加入 `关注` / `重点` checkbox，并带稳定 HTML 注释标记。
2. 新增 daily selection parser，能从日报 markdown 中解析每个 arXiv ID 的勾选状态。
3. 监听 daily 文件修改，防抖后自动同步到 `papers.json`。
4. 勾选关注映射为 `to_read/normal`，勾选重点映射为 `to_read/high`。
5. 确定 `papers.json` 最终路径：继续用 `arxiv-daily/index/papers.json`，或实现迁移到 `arxiv-daily/.index/papers.json`，避免文档和代码长期分叉。
6. 保留手动状态命令作为高级操作。

**阶段 2：实现 Obsidian Reading Dashboard**

1. 先做只读 Dashboard：tabs、搜索、筛选、汇总、打开 note/daily/arXiv。
2. 再做单篇状态修改。
3. 最后做多选和批量操作。
4. Dashboard 稳定后，再考虑 core extraction / CLI fallback。

这个顺序最贴近日常使用：日报负责“今天挑出来”，Dashboard 负责“之后找得到、看得清、处理得动”。
