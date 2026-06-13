# arXiv Daily Plan

## Product Direction

定位：

> arXiv Daily = 每日发现 + 日报内筛选 + Reading Dashboard + 后续阅读管理

主线只有一条：把 arXiv pipeline 这套核心资产（抓取、筛选、总结、paper index、状态流转）持续做深，并逐步接入真实科研工作流——当天 triage、跨日期回看、写作引用、Zotero / PDF / 项目笔记。当前已完成“每日发现”和“日报内筛选”（v0.1.4）、Workflow Quick Wins（v0.1.5）、Obsidian Reading Dashboard（v0.1.6）、Core Extraction + CLI Fallback（v0.1.7）以及 Research Tool Integrations（v0.1.8）；下一步推进轻量 VS Code Companion Extension。

宿主策略已定：**以 Obsidian 为主**，长期作为第一宿主和移动端方案；v0.1.7 把 core 抽成 host 无关之后，**顺带**产出一个轻量 VS Code 扩展（复用 core 和 Dashboard model，见 VS Code Companion Extension 一节）；不做独立 app。

核心目标不是替代 Zotero、Dataview 或 Obsidian，而是把每天新出现的论文稳定地接入现有科研笔记工作流，并提供一个能回看、检索和批量处理论文状态的工作台。

## Glossary

- **Daily discovery**：每天抓取 arXiv，按 topic 筛选和总结，生成日报。
- **Paper index**：以 JSON 保存的论文级状态和去重索引，不是日常阅读入口。
- **Paper object**：以 arXiv ID 为稳定主键的一篇论文记录。默认存放在 `papers.json` 中；只有 detail、saved 或用户主动创建笔记的论文才对应 markdown 详情页；high priority 默认只保留在 JSON 状态中。
- **Reading Dashboard**：Obsidian 内的插件视图，用于检索、筛选、回看、汇总和批量更新论文状态；它读取 `papers.json`，但不替代 Markdown 编辑器。
- **Reading management**：对论文做后续处理，例如待读、正在读、已读、收藏、忽略、关联 Zotero 或项目笔记。
- **Host**：承载 arXiv Daily 的外壳，例如 Obsidian 或 VS Code 扩展。Host 负责编辑器、文件系统入口、命令面板、状态提示和 UI 容器。
- **Host adapter**：把 core 需要的能力抽象出来的适配层，例如 HTTP、storage、secret、progress、open note、open URL。v0.1.7 之后不应该让 core 直接依赖 Obsidian API。

## Current State

已经具备：

- 按一个或多个 arXiv category 抓取 `/recent`，按 arXiv ID 去重后进入同一轮 LLM 筛选；Atom API 补全摘要，章节抽取优先保留高价值正文内容。
- 按用户配置的 topics 进行 LLM 分类和筛选。
- 生成每日 markdown 日报：每篇论文按核心问题 / 关键方法 / 主要结果 / 为什么值得看 / 局限或边界五字段总结，并标注信息来源章节。
- 为 detail topic 生成单篇论文详情页（研究问题 / 方法设计 / 关键证据 / 主要结论 / 适用边界 / 一句话价值判断）。
- 隐藏主索引 `arxiv-daily/.index/papers.json`：按 arXiv ID 去重，合并 seenDates / dailyReports，用户控制字段不被覆盖，旧 `index/` 路径自动迁移。
- 日报内“关注 / 重点” checkbox，修改后防抖自动同步到 papers.json；取消勾选只回退插件默认状态，不降级用户手动设置的 saved / read / ignored。
- 插件启动后补扫 lookback 窗口内日报 checkbox，手机 / 其他设备同步回来的勾选会在桌面端重启后补写到 `papers.json`。
- 日报标注 new / seen_before；ignored 论文不再进入日报。
- 日报末尾折叠列出未入选论文，作为 LLM 漏报兜底；ignored 论文不进入兜底列表。
- 支持复制 arXiv BibTeX，解析 entry key 并写回 `citationKey`。
- Dashboard 当前筛选结果可以批量导出 `.bib`，重复 citation key 会在导出时改写为唯一 key 并回写索引。
- 支持复制 LaTeX、pandoc Markdown、Typst citation snippet；缺少 `citationKey` 时会先抓取 BibTeX 补齐。
- 支持在 Dashboard 手动维护 Zotero key / URI、通过 `zoteroUri` 打开 Zotero item，并用缺失筛选和汇总提示待导入 Zotero 的 saved 论文。
- 支持从 Dashboard 手动下载单篇 arXiv PDF 到 vault，写回 `pdfPath`，之后优先打开本地 PDF。
- 支持从 Dashboard 把论文追加到项目笔记，并维护 `projects` 字段；重复追加会自动去重。
- 支持手动按日期运行、补跑 lookback、按 arXiv ID 生成详情、手动创建论文笔记、论文状态命令。
- 超出 arXiv `/recent` 5 天窗口的手动日期补跑会 fallback 到 arXiv export API 的 submittedDate 单日窗口，并在日报中标注近似窗口语义。
- 支持日期级 run state：completed、failed、skipped、running；失败重试、强制重跑、清空状态、取消当前运行。
- 支持 diagnostics 报告，覆盖配置、日期窗口、运行状态和 paper index 一致性。
- Obsidian Reading Dashboard 已可从命令面板或 ribbon 打开，支持跨日期 tabs、搜索、筛选、汇总、打开 note / daily / arXiv / PDF、单篇和批量状态 / priority 修改。
- release 已自动化，tag 触发 GitHub release asset 构建（v0.1.4 起生效）；当前插件版本已准备到 v0.1.8。

仍然可以继续加强：

- 轻量 VS Code Companion Extension，复用 core 和 Dashboard model。

## Design Principles

1. **JSON 索引是内部状态主数据源**

   `arxiv-daily/.index/papers.json` 保存所有被筛选为相关的论文记录。不要把长期论文状态放进 `.obsidian/plugins/arxiv-daily/data.json`，避免插件安装、更新或手动覆盖时误伤长期数据。

2. **Obsidian 为主宿主**

   Obsidian 长期是第一宿主和移动端方案（vault、双链、同步、笔记生态）。VS Code 扩展是 core 抽取后的顺带产物，不承担主线功能首发；不做独立 app。

3. **先抽 core，再加宿主**

   新宿主只在 v0.1.7 core 抽取完成后接入。v0.1.6 先在 Obsidian 内验证 Dashboard 的数据模型、筛选模型和操作模型，避免把产品问题和宿主问题混在一起；core 不直接依赖 Obsidian API（见 Host Adapter Requirements）。

4. **GUI 先做成 Obsidian 内工作台**

   需要 GUI 来解决回看、检索、汇总和批量处理，但第一版 GUI 是 Obsidian custom view。Markdown 阅读、编辑、双链、文件同步继续交给 Obsidian，Dashboard 只负责把 `papers.json` 变成可操作的论文列表和回顾视图。

5. **论文为中心，日报只是视图**

   把论文作为稳定对象：何时发现、属于什么 topic、当前处理状态、是否值得收藏。markdown 详情页是重要论文的长期笔记，不是所有相关论文的必要载体。

6. **轻量状态机，不做复杂项目管理**

   状态字段应该足够支持科研阅读管理，但避免做成复杂任务系统。

7. **先结构化，再做集成**

   Zotero、PDF、引用和项目笔记都依赖稳定的论文 metadata。先把 JSON schema、去重和状态流转做好，再做外部工具接入。

## Storage Layout

JSON-first 只针对论文级状态索引，不改变日报作为 markdown 入口的定位。

目录结构：

```text
arxiv-daily/
  daily/
    2026-06-11.md
  .index/
    papers.json
    run-state.json
  papers/
    2606.12345.md
  pdfs/
    2606.12345.pdf
```

职责划分：

| Path | Role | Created by default |
|---|---|---|
| `arxiv-daily/daily/YYYY-MM-DD.md` | 每日发现入口，按 topic 展示当天相关论文 | Yes |
| `arxiv-daily/.index/papers.json` | 插件内部状态：所有相关论文的状态、去重、seen dates 和外部工具字段；默认隐藏 | Yes |
| `arxiv-daily/.index/run-state.json` | Obsidian scheduler 与 Node CLI 共享的日期级运行状态 | Yes |
| `arxiv-daily/papers/<arxiv_id>.md` | 重要论文的长期阅读笔记和深度分析 | Only for detail / saved / manual |
| `arxiv-daily/pdfs/<arxiv_id>.pdf` | 用户手动下载的 arXiv PDF，本地打开优先于远程 PDF URL | Only manual |

日报是每天最主要的阅读入口，从当天 pipeline 结果和 `papers.json` 共同生成：

- 按 topic 分组展示当天相关论文。
- 标明论文是 `new` 还是 `seen before`。
- `ignored` 论文不进入日报。
- 如果论文已有 markdown note，链接到 `paperPath`；没有则展示 arXiv 链接和摘要总结。
- 将日报路径写回对应论文记录的 `dailyReports`。
- 末尾折叠列出当日全部未入选论文的标题与链接，作为 LLM 漏报的兜底（v0.1.5）。

## Paper Index

主索引 `arxiv-daily/.index/papers.json`，schema v2 已实现：

```ts
interface PaperIndex {
  schemaVersion: 2;
  updatedAt: string;
  papers: Record<string, PaperIndexEntry>;
}

interface PaperSummary {
  sourceSections?: string;
  coreProblem?: string;
  keyMethod?: string;
  mainResult?: string;
  whyRelevant?: string;
  limitations?: string;
}

interface PaperIndexEntry {
  arxivId: string;
  source: "arxiv";
  title: string;
  authors: string[];
  published: string;
  updated: string;
  category: string; // primary/source-compatible category
  categories?: string[]; // actual source categories when fetched from multiple lists
  summary?: PaperSummary;
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
  zoteroUri: string;
  citationKey: string;
  projects: string[];
}
```

schema v2 在 v0.1.6 前置中定稿：新增 `summary` 字段，保存日报中已经生成的核心问题 / 关键方法 / 主要结果 / 为什么值得看 / 局限或边界，以及信息来源章节。旧 schema v1 文件会在读取后迁移，后续保存统一写出 schemaVersion 2。

### Paper Note Creation

不是每篇相关论文都需要创建 markdown：

| Case | Create markdown note |
|---|---|
| arXiv 当天全部论文 | No |
| LLM 判断相关但非重点论文 | No, only index in JSON |
| `detail: true` | Yes |
| `priority: high`（勾选“重点”） | No, only index in JSON（避免 checkbox 触发昂贵副作用） |
| `status: saved` | Yes |
| 用户执行 `Create paper note` | Yes |

轻量相关论文留在 `papers.json`，避免 vault 每天增加大量低价值文件。重要论文进入 `arxiv-daily/papers/<arxiv_id>.md`，用于长期阅读笔记、项目链接和 Zotero 跟踪。

### Paper Note Frontmatter

frontmatter 从 `papers.json` 派生，只写必要字段。JSON 内部字段使用 camelCase；markdown frontmatter 使用更适合 Obsidian 查询的 snake_case。

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
zotero_uri: ""
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
| `ignored` | 不感兴趣，之后日报中不再出现 |

### Priority Values

| Priority | Meaning |
|---|---|
| `low` | 有一定相关性，但不急 |
| `normal` | 默认优先级 |
| `high` | 高价值或近期需要阅读 |

## Release Versioning

发布 tag 与里程碑编号一致，只递增最后一位（v0.1.4、v0.1.5、…）。如果某个里程碑分多次 patch 发布、或中间插入 hotfix，后续里程碑编号顺延并更新本文档。

## v0.1.4: Daily Selection Layer（已发布）

> 2026-06-12 以 tag `v0.1.4` 发布（`c8ee23d` / `094a9e8`），180 条测试通过，release 流水线自此生效。

已交付：日报成为筛选主入口——每篇论文带“关注 / 重点” checkbox（稳定 HTML 注释标记），修改后防抖自动同步 `papers.json`；勾关注映射 `to_read/normal`，勾重点映射 `to_read/high`，取消勾选只回退插件默认状态、不降级用户手动设置；隐藏索引 `.index/papers.json` 含旧路径自动迁移；按 arXiv ID 去重合并 seenDates / dailyReports，用户控制字段不被覆盖；日报标注 new / seen_before，ignored 论文不再出现。

与最初设计的一处差异：`priority: high` 不自动创建 markdown note——勾“重点”保持零副作用，笔记只通过 detail / saved / 手动创建产生（Paper Note Creation 表已同步）。

## v0.1.5: Workflow Quick Wins（已完成）

> 2026-06-13 完成，插件版本准备到 `0.1.5`。验证：`npm test` 193 条通过，`npm run build` 通过。发布时创建 tag `v0.1.5`。

目标：在 Dashboard 之前，用小成本补齐科研日常里最痛的几个断点。各项彼此独立，可逐项发布。

### 1. 勾选状态启动补扫

勾选同步目前只依赖 Obsidian 运行中的 `modify` 事件。在手机或另一台机器上勾选“关注 / 重点”，文件经 Obsidian Sync / iCloud 同步回来后，桌面端重启不会补处理这些勾选。

- 插件启动（layout ready）后，重新解析最近 lookback 窗口内的日报文件并同步勾选状态。
- 复用现有 parser、映射和防抖逻辑，不引入第二套实现。
- 补扫和实时同步都不得把 saved / read / ignored 降级回 to_read：勾选映射只在当前状态属于 inbox / to_read 时生效（当前 `stateForSelection` 对已勾选论文会无条件返回 to_read，补扫前需要先收紧这一点，否则每次启动都会把手动标记的 saved 拉回去）。
- 验收：手机勾选 → 文件同步 → 桌面重启 Obsidian，`papers.json` 状态正确，全程无需手动命令；对已 saved 论文反复补扫不改变其状态。

### 2. 日报漏报兜底

LLM 筛选存在漏报，而漏掉一篇关键论文的代价远大于多扫几十行标题。补上这一块后，日报可以完全替代刷 arXiv 列表页。

- 日报末尾追加折叠区（callout 或 `<details>`），列出当日该分类全部未入选论文的标题 + arXiv 链接。
- 纯标题列表，不经过 LLM，token 成本为零。
- `ignored` 论文不进此列表。
- 验收：相关论文数 + 未入选数 = 当日该分类论文总数；折叠区默认收起，不干扰正常阅读。

### 3. BibTeX 快捷获取

读论文的终点是写作引用。arXiv 直接提供 `https://arxiv.org/bibtex/<arxiv_id>` 端点，实现成本很低，从 v0.1.8 提前。

- 新增命令：按 arXiv ID / 当前论文笔记复制 BibTeX。
- 解析 BibTeX entry key 写回 `citationKey` 字段。
- 完整的 Zotero bridge 和批量导出仍留在 v0.1.8。
- 验收：从日报或论文笔记一步拿到可直接粘进 `.bib` 的条目，`papers.json` 同步记录 `citationKey`。

### 4. 多 arXiv 分类

跨方向研究是常态（astro-ph + cs.LG、cs.CL + cs.AI）。当前设置只支持单分类，是真实覆盖缺口。

- `arxiv.category: string` 兼容保留为 primary category，新增 `categories: string[]`，含设置迁移和 UI 适配。
- 多分类抓取后合并，按 arXiv ID 去重，交叉挂载论文只处理一次。
- 日报标题、frontmatter 和 index 的 `category` 字段适配多分类；index 中保存论文实际所属分类，而不是配置值。
- 验收：配置两个分类时，交叉挂载论文只出现一次、只消耗一次 LLM 调用。

## v0.1.6: Obsidian Reading Dashboard（已完成）

> 2026-06-13 完成，插件版本准备到 `0.1.6`。验证：`npm test` 203 条通过，`npm run build` 通过。发布时创建 tag `v0.1.6`。

目标：做一个 Obsidian 内 GUI 工作台，让用户不用逐篇日报回翻，就能查看、搜索、汇总和处理已关注、重点、saved、read、ignored 等论文。

核心原则：

- **Dashboard 是主回顾入口**：日报负责当天 triage，Dashboard 负责跨日期回看。
- **不重做 Markdown 编辑器**：论文笔记仍在 Obsidian editor 中打开，Dashboard 只提供列表、筛选和状态操作。
- **先本地、后集成**：第一版只读写 `papers.json` 和 vault markdown，不引入数据库或外部同步。
- **可批量处理**：关注/重点论文多起来后，必须支持多选和批量状态修改。
- **model 层 host 无关**：query/filter/action model 与 Obsidian DOM 视图分离，为 v0.1.7 后复用到 VS Code Webview 做准备。

### Scope

前置（schema v2，已完成）：paper index 已新增 `summary` 字段，pipeline 会从日报 markdown 解析核心问题 / 关键方法 / 主要结果 / 为什么值得看 / 局限或边界并写入每条论文记录。Dashboard 卡片、搜索和视图导出都以该字段为结构化数据源；旧 schema v1 会迁移到 v2。

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
   - schema v2 的结构化 summary 字段（核心问题 / 关键方法 / 主要结果）。
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
- Dashboard 不创建新的 markdown 回顾页面作为主入口；视图导出 / 组会分享需求较小，整条路线最后再做（见 Backlog）。
- 所有状态修改都保留用户手写的 paper note 内容。

## v0.1.7: Core Extraction + CLI Fallback

> 2026-06-13 完成，插件版本准备到 `0.1.7`。验证：`npm test` 241 条通过，`npm run build` 通过。发布时创建 tag `v0.1.7`。

目标：把抓取、分类、去重、总结、索引这些核心逻辑抽成可复用 core，为 cron/headless 和 VS Code 扩展打基础。

可做功能：

- 将 Obsidian 相关依赖隔离在 adapter 层（见 Host Adapter Requirements）。
- 让 Node CLI 复用同一套 pipeline：
  - `arxiv-daily run --date YYYY-MM-DD`。
  - `arxiv-daily run-pending`。
  - `arxiv-daily summarize --id 2606.12345`。
- CLI 的配置方案（API key、topics、输出路径从 env / 配置文件读取）；这也是 VS Code 扩展的配置基础。
- 可选把 run state 写进 vault 输出目录，避免 CLI 和 Obsidian scheduler 重复跑。
- 用 arXiv export API 按日期范围补跑超出 `/recent` 5 天窗口的缺失日期（出差、休假场景）；注意 announce date 与 submittedDate 的分桶差异，补跑产物需标注为近似窗口。
- 生成 markdown 的链接风格可配置：`[[wikilink]]`（Obsidian）或标准相对链接（通用编辑器可导航，Obsidian 也原生支持）；数学公式坚持标准 LaTeX 语法。
- Node CLI 稳定后退役根目录 `arxiv_daily.py`：它和插件已是两份并行维护的 pipeline，统一到共享 core 后停止双份维护。
- 保持 Obsidian 插件仍是主用户界面。

### Host Adapter Requirements

adapter 设计以 Obsidian 继续可用、VS Code 可低成本接入为目标。core 不应直接依赖这些宿主能力：

- HTTP 请求：Obsidian `requestUrl`、Node `fetch`、VS Code `fetch` 都通过 `HttpClient` 注入。
- 文件读写：Obsidian `Vault`、Node fs、VS Code workspace fs 都通过 `StorageAdapter` 注入。
- 密钥存储：Obsidian settings、环境变量、VS Code SecretStorage 都通过 `SecretProvider` 注入。
- 状态反馈：Obsidian Notice/status bar、CLI stdout、VS Code notification 都通过 `ProgressReporter` 注入。
- 打开资源：open note、open daily report、open arXiv/PDF URL 由 host adapter 实现。

Dashboard 的 query/filter/action model 同样从 Obsidian DOM 视图中剥离，复用到 VS Code Webview。

## VS Code Companion Extension

定位：core 抽取后的**顺带产物**，不是主线。Obsidian 仍是第一宿主和移动端方案；VS Code 扩展服务于“代码 / 论文同工作区 + 终端 agent（Claude Code / Codex）”的使用方式。

时机：v0.1.7 完成后择机启动，可与 v0.1.8 并行，优先级始终低于 pipeline 主线。以独立 VSIX 发布，版本序列与 Obsidian 插件 tag 解耦。

最小 scope：

1. 把包含 `arxiv-daily/` 的 workspace folder 当作 vault，复用 core 读写 `.index/papers.json`。
2. Webview 承载 Reading Dashboard（复用 v0.1.6 的 query/filter/action model）：tabs、搜索、筛选、打开 note / daily / arXiv / PDF、单篇状态修改。
3. 命令面板调用 pipeline：run、run-pending、summarize by ID（经 core 或 CLI）。
4. API key 存 VS Code SecretStorage；生成链接用标准相对链接。

不做：自研 markdown 编辑（直接用 VS Code 原生 editor / preview）、独立 app、要求用户迁移笔记。存量笔记里的 `[[wikilink]]` 在 VS Code 预览中不可点击是预期内的，评估体验时以新生成内容和 Dashboard 导航为准。

## v0.1.8: Research Tool Integrations

> 2026-06-13 完成，插件版本准备到 `0.1.8`。验证：`npm test` 254 条通过，`npm run build` 通过。发布时创建 tag `v0.1.8`。

目标：连接 Zotero、PDF 和引用管理，但不替代它们。

优先级建议：

1. **BibTeX / citation helper**（单篇复制 BibTeX、写入 `citationKey` 已提前到 v0.1.5）
   - 批量导出当前筛选结果为 `.bib` 文件。
   - 引用片段模板（`\cite{key}`、pandoc、Typst）。

2. **Zotero bridge**
   - 先支持手动字段：`zoteroKey`、`zoteroUri`。
   - 通过手动维护的 `zoteroUri` 打开 Zotero item。
   - Better BibTeX citekey 或 Zotero local API 自动同步留作后续增强。
   - 视图中列出 `saved` 但没有 `zoteroKey` 的论文。

3. **PDF management**
   - 可选下载 arXiv PDF。
   - 写入 `pdfPath`。
   - 避免默认大量下载，先做手动命令。

4. **Project notes**
   - 支持 `projects` 字段。
   - 允许把论文链接追加到某个项目笔记。

## Backlog

优先级未定、但值得保留的方向：

- 作者关注：维护 watched authors 列表，抓取阶段字符串匹配，命中论文在日报中加标记；零 LLM 成本，适合跟踪导师、合作者和竞争组。
- 版本更新提醒：已关注 / saved 的论文出现新版本（v2+）时在日报提示；v2 往往意味着被接收或大修。
- 多来源接入：bioRxiv、ADS、OpenReview、RSS；依赖 v0.1.7 core 抽取后的 fetcher 抽象，避免在 Obsidian 插件里直接堆来源。
- Better BibTeX / Zotero local API 自动同步：从 Zotero 自动回填 citekey 或 URI；需要评估用户本机 Zotero 运行状态、Better BibTeX 安装情况和失败提示，当前不作为 v0.1.8 硬依赖。
- 视图导出与组会分享：把 Dashboard 当前筛选结果导出为一页 markdown 清单（journal club / 周报），与周度自动汇总合并考虑；需求较小，放在整条路线最后。
- vault 内 CLAUDE.md：教终端 agent `papers.json` 的 schema、状态语义和笔记约定，支持对话式文献整理；与 VS Code 扩展互补，零开发成本即可先行试用。

## Research Workflows

### Daily Triage

1. 每天自动生成日报。
2. 打开日报，快速扫 new papers。
3. 大部分论文不动。
4. 对感兴趣论文勾选 `关注`。
5. 对特别重要论文勾选 `重点`。
6. 插件自动把勾选结果同步到 `papers.json`。
7. 关闭日报；后续从单篇笔记、搜索或 Dashboard 继续处理少数被挑出的论文。

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

只有重要论文才创建 markdown note。note 更新需要避免简单字符串拼接，使用 Obsidian metadata/frontmatter API 或成熟 YAML parser，确保：

- 保留用户手写正文。
- 不破坏已有 frontmatter。
- 缺失字段补齐。
- 用户控制字段不被覆盖。
- JSON 和 markdown 的关键字段保持一致。

### Writer Behavior

生成或更新 JSON / markdown note 时区分两类字段：

- 插件可更新字段：`seenDates`、`dailyReports`、`updated`、`topics`、`detail`。
- 用户控制字段：`status`、`priority`、`projects`、`zoteroKey`、`citationKey`、正文笔记。

默认策略：只补缺，不覆盖用户控制字段。

### Dashboard Architecture

Dashboard 实现为 Obsidian custom view，而不是生成一个长期维护的 `inbox.md` 文件：

- `PaperIndexStore` 继续是唯一状态源。
- Dashboard 启动时 load index，状态更新后 save index 并局部刷新。
- 表格过滤、搜索、排序先在内存中完成；数据量到几千篇前不需要数据库。
- Paper note 的阅读和编辑通过 `workspace.openLinkText` 打开 Obsidian markdown editor。
- Dashboard 不直接编辑 note 正文；只更新 `papers.json` 和必要 frontmatter。
- query/filter/action model 与 DOM 视图分离，VS Code 扩展复用 model 层而不是 Obsidian 视图。

## Detailed Roadmap Todo

每个 checklist item 都应该尽量形成一个可独立验证、可独立提交的完成点。提交前先跑相关测试；涉及插件行为的改动至少跑 `npm test`，里程碑收尾再跑 `npm run build`。

### v0.1.5 Workflow Quick Wins

- [x] 启动补扫：收紧 checkbox selection 到 paper state 的映射，确保 saved / read / ignored 不会被已勾选 checkbox 降级。
- [x] 启动补扫：把现有 daily selection parser 暴露为可复用同步入口，避免启动补扫和实时 `modify` 事件维护两套逻辑。
- [x] 启动补扫：在 `layoutReady` 后扫描最近 lookback 窗口内的 `daily/*.md`，按日期排序同步 checkbox 到 `papers.json`。
- [x] 启动补扫：补测试覆盖移动端同步场景、已 saved 论文反复补扫、无日报文件和无 index 文件的空操作。
- [x] 漏报兜底：让 pipeline 保留当日抓取全集与入选论文集合，按 arXiv ID 计算未入选列表。
- [x] 漏报兜底：日报末尾渲染默认收起的未入选论文列表，只包含标题、arXiv 链接和基础 metadata，不消耗 LLM。
- [x] 漏报兜底：过滤 ignored 论文，并补测试保证入选数 + 未入选数等于抓取总数。
- [x] BibTeX：新增按 arXiv ID 获取 BibTeX 的服务，解析 entry key 并处理网络失败、空响应和非法 ID。
- [x] BibTeX：新增命令从当前论文笔记或用户输入 arXiv ID 获取 BibTeX，复制到剪贴板并写回 `citationKey`。
- [x] BibTeX：补测试覆盖 BibTeX key 解析、index 更新、当前文件 frontmatter / 正文识别 arXiv ID。
- [x] 多分类：把 settings 从 `arxiv.category` 迁移到 `arxiv.categories: string[]`，保留旧配置兼容读取。
- [x] 多分类：设置页支持增删多个 arXiv category，并校验重复、空值和非法分类。
- [x] 多分类：pipeline 多分类抓取后按 arXiv ID 去重，交叉挂载论文只进入一次 LLM 分类和 index 更新。
- [x] 多分类：明确 index schema 中 `category` / `categories` 的兼容策略，并补迁移和测试。
- [x] v0.1.5 收尾：更新 README / PLAN 状态，跑完整测试和 build，发布前确认版本号与 tag 计划。

### v0.1.6 Obsidian Reading Dashboard

- [x] schema v2：确定 `summary` 字段结构，新增 migration，并让 pipeline 写入结构化摘要。
- [x] Dashboard model：抽出 host 无关 query/filter/sort/stat/action model，覆盖 tabs、搜索、筛选和汇总。
- [x] Dashboard view shell：注册 `arxiv-daily-dashboard` custom view、命令和 ribbon 入口。
- [x] Dashboard list：实现表格 / 列表渲染、空状态、加载失败状态和基础样式。
- [x] Dashboard filters：实现 topic、date range、status、priority、has note、detail、missing citation / Zotero 筛选。
- [x] Dashboard actions：实现打开 note / daily / arXiv / PDF、创建 note、单篇状态和 priority 修改。
- [x] Dashboard batch：实现多选和批量 ignored / read / saved / priority 修改；批量创建 note 必须二次确认。
- [x] Dashboard diagnostics：扩展 paper index / note consistency 检查，并把结果接入现有 diagnostics 报告。
- [x] v0.1.6 收尾：补单元测试和 UI model 测试，更新文档，跑完整测试和 build。

### v0.1.7 Core Extraction + CLI Fallback

- [x] Adapter contracts：定义 `HttpClient`、`StorageAdapter`、`SecretProvider`、`ProgressReporter`、resource opener 等接口。
- [x] Core extraction：把 fetch、filter、summarize、write daily、paper index 更新从 Obsidian API 中剥离。
- [x] Obsidian adapter：用 adapter 重新接回现有插件功能，保持用户行为不变。
- [x] Node CLI：新增 `run --date`、`run-pending`、`summarize --id` 命令，复用 core。
- [x] CLI config：支持 env / 配置文件读取 API key、topics、输出路径和 link style。
- [x] Run state：把 CLI 与 Obsidian scheduler 的 run state 放到同一 vault 输出目录，避免重复运行。
- [x] 超窗补跑：用 arXiv export API 支持日期范围 fallback，并在日报中标注近似窗口语义。
- [x] Link style：支持 wikilink 和标准相对链接，保证 Obsidian 与 VS Code / 通用编辑器都能导航。
- [x] Python 退役：CLI 稳定后冻结或移除根目录 `arxiv_daily.py` 的主流程，文档引导到 Node CLI。
- [x] v0.1.7 收尾：跑插件测试、CLI 测试、build，更新 README / PLAN。

### v0.1.8 Research Tool Integrations

- [x] BibTeX 批量导出：从 Dashboard 当前筛选结果导出 `.bib`，处理重复 citation key。
- [x] 引用片段模板：支持 LaTeX、pandoc markdown、Typst 等 citation snippet。
- [x] Zotero 手动字段：支持 `zoteroKey` / `zoteroUri` 的读取、编辑、校验和 Dashboard 缺失提示。
- [x] Zotero bridge：通过手动维护的 `zoteroUri` 打开 Zotero item；Better BibTeX / Zotero local API 自动同步留作后续增强。
- [x] PDF 管理：手动下载 arXiv PDF、写入 `pdfPath`、打开本地 PDF，不默认批量下载。
- [x] Project notes：支持 `projects` 字段维护，并把论文链接追加到指定项目笔记。
- [x] v0.1.8 收尾：更新 docs、测试、build，并确认不替代 Zotero / PDF 阅读器的边界。

### VS Code Companion Extension

- [x] Extension scaffold：独立 VS Code extension 目录、manifest、build/test 脚本和 VSIX 发布策略。
- [x] Workspace adapter：把包含 `arxiv-daily/` 的 workspace folder 识别为 vault，接入 core storage。
- [x] Secret adapter：用 VS Code SecretStorage 保存 API key。
- [x] Webview Dashboard：用对齐 v0.1.6 dashboard model 语义的本地 model，实现 tabs、搜索、筛选、打开资源和单篇状态修改。
- [ ] Commands：命令面板接入 run、run-pending、summarize by ID。
- [ ] Link compatibility：新生成 markdown 使用标准相对链接；存量 wikilink 只保证 Dashboard 导航可用。
- [ ] Extension 收尾：最小手动验收 VS Code 打开 vault、浏览 Dashboard、改状态、运行 pipeline。

## Recommended Next Step

**阶段 1：v0.1.5 Workflow Quick Wins**（已完成）

按收益 / 成本顺序推进，每项独立可发布：

1. 勾选状态启动补扫（手机 / 多设备 triage 立刻可用，注意先收紧降级保护）。
2. 日报漏报兜底（未入选论文折叠列表，零 token）。
3. BibTeX 快捷获取（接通写作引用环节）。
4. 多 arXiv 分类支持。

**阶段 2：v0.1.6 Reading Dashboard**（已完成）

1. ~~先定 paper index schema v2（结构化摘要字段），让 pipeline 从现在开始积累数据。~~ 已完成。
2. ~~只读 Dashboard：tabs、搜索、筛选、汇总、打开 note/daily/arXiv。~~ 已完成。
3. ~~单篇状态修改，再做多选和批量操作。~~ 已完成。

**阶段 3：v0.1.7 Core Extraction**（已完成）

adapter 层、Node CLI（退役 Python 脚本）、CLI 配置方案、超窗补跑 fallback、链接风格可配置。

**阶段 4：v0.1.8 Research Tool Integrations（已完成）+ VS Code Companion Extension（顺带，下一步）**

这个顺序最贴近日常使用：日报负责“今天挑出来”，快赢层补齐“手机上也能挑、漏不掉、引用拿得到”，Dashboard 负责“之后找得到、看得清、处理得动”，core 抽取让这套能力既能 cron 跑、也能顺带长进 VS Code。
