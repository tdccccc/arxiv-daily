# arXiv Daily Plan

## Product Direction

定位：

> arXiv Daily = 每日发现 + 日报内筛选 + Reading Dashboard + 后续阅读管理

当前插件已经完成“每日发现”和“日报内筛选”（v0.1.4）：抓取 arXiv、按研究主题筛选、生成日报、为重点论文生成详情页；用户读日报时直接勾选“关注”或“重点”，插件把选择自动同步到内部 JSON 索引，只有重要论文才生成长期 markdown 笔记。下一阶段分两步走：先用小成本补齐科研工作流断点（多设备勾选、漏报兜底、BibTeX、多分类），再做 Obsidian 内 Reading Dashboard 解决跨日期回看。

用户对 GUI 的需求是成立的：当关注、重点、saved、read 等状态积累起来后，不能要求用户逐篇日报回翻。后续应先做 Obsidian 内的 Reading Dashboard，用表格、筛选、搜索和汇总视图消费 `papers.json`；独立工作台是长期愿景，按 Standalone Research Workbench 一节的验证路径推进。

核心目标不是替代 Zotero、Dataview 或 Obsidian，而是把每天新出现的论文稳定地接入现有科研笔记工作流，并提供一个能回看、检索和批量处理论文状态的工作台。

## Glossary

- **Daily discovery**：每天抓取 arXiv，按 topic 筛选和总结，生成日报。
- **Paper index**：以 JSON 保存的论文级状态和去重索引，不是日常阅读入口。
- **Paper object**：以 arXiv ID 为稳定主键的一篇论文记录。默认存放在 `papers.json` 中；只有 detail、high priority、saved 或用户主动创建笔记的论文才对应 markdown 详情页。
- **Reading Dashboard**：Obsidian 内的插件视图，用于检索、筛选、回看、汇总和批量更新论文状态；它读取 `papers.json`，但不替代 Markdown 编辑器。
- **Reading management**：对论文做后续处理，例如待读、正在读、已读、收藏、忽略、关联 Zotero 或项目笔记。

## Current State

已经具备：

- 按 arXiv category 抓取 `/recent`，Atom API 补全摘要，章节抽取优先保留高价值正文内容。
- 按用户配置的 topics 进行 LLM 分类和筛选。
- 生成每日 markdown 日报：每篇论文按核心问题 / 关键方法 / 主要结果 / 为什么值得看 / 局限或边界五字段总结，并标注信息来源章节。
- 为 detail topic 生成单篇论文详情页（研究问题 / 方法设计 / 关键证据 / 主要结论 / 适用边界 / 一句话价值判断）。
- 隐藏主索引 `arxiv-daily/.index/papers.json`：按 arXiv ID 去重，合并 seenDates / dailyReports，用户控制字段不被覆盖，旧 `index/` 路径自动迁移。
- 日报内“关注 / 重点” checkbox，修改后防抖自动同步到 papers.json；取消勾选只回退插件默认状态，不降级用户手动设置的 saved / read / ignored。
- 日报标注 new / seen_before；ignored 论文不再进入日报。
- 支持手动按日期运行、补跑 lookback、按 arXiv ID 生成详情、手动创建论文笔记、论文状态命令。
- 支持日期级 run state：completed、failed、skipped、running；失败重试、强制重跑、清空状态、取消当前运行。
- 支持 diagnostics 报告，覆盖配置、日期窗口、运行状态和 paper index 一致性。
- release 已自动化，tag 触发 GitHub release asset 构建。

仍然可以继续加强：

- 勾选同步只监听 Obsidian 运行中的文件修改事件：手机 / 其他设备上勾选、经文件同步回来的改动，桌面端重启后不会被补处理。
- 只支持单个 arXiv 分类，跨方向研究（如 astro-ph + cs.LG）覆盖不了。
- LLM 筛选存在漏报风险，日报里看不到当天未入选的论文，无法快速兜底确认。
- 五字段结构化摘要只存在于日报 markdown 中，papers.json 里没有，Dashboard 搜索、周报复用都缺数据。
- 没有 BibTeX / 引用获取能力，从“读到论文”到“写作引用”之间断链。
- 缺少跨日期回看、筛选、批量处理的 GUI Dashboard（v0.1.6）。
- 超出 arXiv `/recent` 5 天窗口的日期无法补跑（出差、休假场景论文直接丢失）。
- PDF、Zotero、项目笔记的结构化接入（v0.1.8）。

## Design Principles

1. **JSON 索引是内部状态主数据源**

   `arxiv-daily/.index/papers.json` 保存所有被筛选为相关的论文记录。不要把长期论文状态放进 `.obsidian/plugins/arxiv-daily/data.json`，避免插件安装、更新或手动覆盖时误伤长期数据。

2. **Obsidian 优先，独立工作台走验证路径**

   现阶段用户价值来自和 Obsidian vault、双链、搜索、Dataview、同步工具的自然结合。独立工作台是长期愿景，但要按 Standalone Research Workbench 一节的方式先零成本验证（VS Code + 终端 agent 打开 vault），并在 core 抽取完成后再选宿主，避免过早承担数据库、同步、UI 和发布维护成本。

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
- 末尾折叠列出当日全部未入选论文的标题与链接，作为 LLM 漏报的兜底（v0.1.5）。

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
| `priority: high`（勾选“重点”） | No, only index in JSON（避免 checkbox 触发昂贵副作用） |
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

## Release Versioning

发布 tag 与里程碑编号一致，只递增最后一位（v0.1.4、v0.1.5、…）。如果某个里程碑分多次 patch 发布、或中间插入 hotfix，后续里程碑编号顺延并更新本文档。

## v0.1.4: Daily Selection Layer

> 状态：已完成（2026-06-12，`c8ee23d` / `094a9e8`），180 条测试全部通过，以 tag `v0.1.4` 发布。与最初 scope 的一处差异：`priority: high` 不自动创建 markdown note——勾选“重点”保持零副作用，笔记只通过 detail / saved / 手动创建产生。

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

## v0.1.5: Workflow Quick Wins

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

- `arxiv.category: string` 改为 `categories: string[]`，含设置迁移和 UI 适配。
- 多分类抓取后合并，按 arXiv ID 去重，交叉挂载论文只处理一次。
- 日报标题、frontmatter 和 index 的 `category` 字段适配多分类。
- 验收：配置两个分类时，交叉挂载论文只出现一次、只消耗一次 LLM 调用。

## v0.1.6: Obsidian Reading Dashboard

目标：做一个 Obsidian 内 GUI 工作台，让用户不用逐篇日报回翻，就能查看、搜索、汇总和处理已关注、重点、saved、read、ignored 等论文。

核心原则：

- **Dashboard 是主回顾入口**：日报负责当天 triage，Dashboard 负责跨日期回看。
- **不重做 Markdown 编辑器**：论文笔记仍在 Obsidian editor 中打开，Dashboard 只提供列表、筛选和状态操作。
- **先本地、后集成**：第一版只读写 `papers.json` 和 vault markdown，不引入数据库或外部同步。
- **可批量处理**：关注/重点论文多起来后，必须支持多选和批量状态修改。

### Scope

前置（schema v2）：在做 Dashboard UI 之前，先扩展 paper index schema，把 pipeline 已经生成的结构化摘要写进每条论文记录（如 `summary: { coreProblem, keyMethod, mainResult, whyRelevant }`，或先只收一个一句话字段）。Dashboard 卡片、搜索和视图导出都依赖它；字段来源可以是解析日报 markdown，也可以让 daily LLM 改为结构化输出后由插件渲染 markdown（倾向后者，但需保持现有日报格式稳定）。先定 schema 让数据从现在开始积累，避免 Dashboard 上线后再改数据结构和回填。

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
- 用 arXiv export API 按日期范围补跑超出 `/recent` 5 天窗口的缺失日期（出差、休假场景）；注意 announce date 与 submittedDate 的分桶差异，补跑产物需标注为近似窗口。
- 生成 markdown 的链接风格可配置：`[[wikilink]]`（Obsidian）或标准相对链接（VS Code 等通用编辑器可导航），为宿主迁移做准备；数学公式坚持标准 LaTeX 语法。
- Node CLI 稳定后退役根目录 `arxiv_daily.py`：它和插件已是两份并行维护的 pipeline（章节抽取优化就同时改了两处），统一到共享 core 后停止双份维护。
- 保持 Obsidian 插件仍是主用户界面。

## v0.1.8: Research Tool Integrations

目标：连接 Zotero、PDF 和引用管理，但不替代它们。

优先级建议：

1. **BibTeX / citation helper**（单篇复制 BibTeX、写入 `citationKey` 已提前到 v0.1.5）
   - 批量导出当前筛选结果为 `.bib` 文件。
   - 引用片段模板（`\cite{key}`、pandoc、Typst）。

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

## Backlog

优先级未定、但值得保留的方向：

- 作者关注：维护 watched authors 列表，抓取阶段字符串匹配，命中论文在日报中加标记；零 LLM 成本，适合跟踪导师、合作者和竞争组。
- 版本更新提醒：已关注 / saved 的论文出现新版本（v2+）时在日报提示；v2 往往意味着被接收或大修。
- 多来源接入：bioRxiv、ADS、OpenReview、RSS；依赖 v0.1.7 core 抽取后的 fetcher 抽象，避免在 Obsidian 插件里直接堆来源。
- 视图导出与组会分享：把 Dashboard 当前筛选结果导出为一页 markdown 清单（journal club / 周报），与周度自动汇总合并考虑；需求较小，放在整条路线最后。

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

## Standalone Research Workbench

长期愿景：一个 VS Code / Claude Desktop 式的独立工作台——打开目标文件夹，浏览和编辑 markdown 论文笔记，内嵌终端 agent（Claude Code / Codex 等），arXiv Daily pipeline 和 Reading Dashboard 内置其中。

这个愿景拆成三层看：

1. 领域层：抓取、筛选、总结、paper index、状态流转——已有，是真正的差异化资产。
2. 工作台层：文件树、markdown 阅读 / 编辑（数学公式、双链、搜索）、Dashboard 视图。
3. agent 层：在工作区内运行终端 agent，让它读写论文笔记和 `papers.json`（对话式文献整理、批量改状态、写综述草稿）。

降低风险的推进方式：

- 零成本先验证工作方式：vault 本身就是一个文件夹，现在就可以用 VS Code / Cursor 打开 vault、在集成终端里跑 Claude Code 来模拟这个工作台；再给 vault 写一份 CLAUDE.md（教 agent papers.json 的 schema、状态语义和笔记约定），agent 层的核心体验立刻可用。如果这种用法成为日常、Obsidian 越开越少，就是独立化的真实证据。
- 中期路线不变：v0.1.5 快赢 → v0.1.6 Dashboard（model 层保持 host 无关）→ v0.1.7 core 抽取。这三步对所有结局（留在 Obsidian / VS Code 扩展 / 独立 app）都是必经路径，没有浪费。
- 到决策点再选宿主，按成本排序（当前倾向：短期留在 Obsidian 把 v0.1.5–v0.1.7 做完，VS Code 扩展是最可能的迁移目标）：
  1. VS Code 扩展：编辑器、终端、agent 集成全部现成，只需写 sidebar Dashboard 和命令；最贴近“像 VS Code”的形态，成本最低。
  2. Electron 壳组装：CodeMirror（编辑）+ xterm.js / node-pty（终端 agent）+ 自有 Dashboard UI，复用 core；UI 完全自主，但要自担三平台打包、自动更新、安全的长尾维护。
  3. 留在 Obsidian：如果验证发现双链生态和移动端同步仍离不开它。
- 不自研 markdown 编辑器内核：数学渲染、双链、全文搜索、文件同步、移动端是无底洞，永远组装现成组件。

立即独立化的触发信号：

- VS Code + Claude Code 工作流试用数周后明显优于 Obsidian 工作流。
- 需要后台常驻自动运行，不依赖任何宿主打开。
- 需要面向不使用 Obsidian 的用户。
- agent 工作流需要宿主深度配合（agent 驱动的批量整理、对话式检索），Obsidian 插件 API 无法支撑。
- 需要数据库级查询、多来源聚合（arXiv、ADS、Semantic Scholar、RSS、Zotero）。

无论最终选哪个宿主，都先把抓取、分类、去重、JSON schema 和 note 生成逻辑抽成可复用核心（v0.1.7），不重写。

## Recommended Next Step

**阶段 0：发布 v0.1.4（已完成 2026-06-12）**

Daily Selection Layer 以 tag `v0.1.4` 发布；后续 tag 只递增最后一位。

**阶段 1：v0.1.5 Workflow Quick Wins**

按收益 / 成本顺序推进，每项独立可发布：

1. 勾选状态启动补扫（手机 / 多设备 triage 立刻可用，注意先收紧降级保护）。
2. 日报漏报兜底（未入选论文折叠列表，零 token）。
3. BibTeX 快捷获取（接通写作引用环节）。
4. 多 arXiv 分类支持。

**阶段 2：v0.1.6 Reading Dashboard**

1. 先定 paper index schema v2（结构化摘要字段），让 pipeline 从现在开始积累数据。
2. 只读 Dashboard：tabs、搜索、筛选、汇总、打开 note/daily/arXiv。
3. 单篇状态修改，再做多选和批量操作。

**阶段 3：v0.1.7 及之后**

core 抽取 + Node CLI（退役 Python 脚本）、超窗补跑 fallback；之后按 v0.1.8 推进 Zotero / PDF / 项目笔记接入。

这个顺序最贴近日常使用：日报负责“今天挑出来”，快赢层补齐“手机上也能挑、漏不掉、引用拿得到”，Dashboard 负责“之后找得到、看得清、处理得动”。
