# arXiv Daily Plan

## Product Direction

定位：

> arXiv Daily = 每日发现 + 日报 triage + Dashboard 回看检索 + Zotero 外部写作流程

主线只有一条：把 arXiv pipeline 这套核心资产（抓取、筛选、总结、paper index、状态流转）持续做深，并让它贴合真实科研流程：

1. 今天打开 Obsidian 看日报。
2. 对少数感兴趣或重要的论文做重点标记。
3. 之后用 Dashboard 按日期、topic、关键词和星标回看。
4. 真正要进入文献库和写作时，打开 arXiv 页面，用 Zotero 浏览器插件导入；citation key、BibTeX 和文献库仍由 Zotero 管。

当前产品判断：

- **不做独立 GUI app**。独立 app 需要重做 Markdown 阅读/编辑、文件同步、设置、插件更新和跨平台打包，收益不够。
- **Obsidian 是第一宿主**。它负责 vault、Markdown 编辑、双链、移动端和同步；arXiv Daily 负责发现、筛选、Dashboard 和状态索引。
- **VS Code 是 companion host**。它适合代码/论文同工作区、终端 agent 和原生 Markdown preview，但不替代 Obsidian 主线。
- **Zotero 是 citation 的 source of truth**。插件不再把 citation key、BibTeX 导出、引用片段和 Zotero 字段维护作为主路径。

v0.1.10 的重心是把 Dashboard 明确为 Obsidian 内主入口，并把阅读工作流简化为：看日报或 Dashboard，觉得重要就 Star，之后从 Dashboard 找回并打开 note / daily / arXiv / PDF。

## Glossary

- **Daily discovery**：每天抓取 arXiv，按 topic 筛选和总结，生成日报。
- **Daily triage**：当天扫日报，只对感兴趣或重要论文做少量操作。
- **Paper index**：以 JSON 保存的论文级状态和去重索引，不是日常手动编辑入口。
- **Paper object**：以 arXiv ID 为稳定主键的一篇论文记录。默认存放在 `papers.json` 中；只有 detail、saved 或用户主动创建笔记的论文才对应 Markdown 详情页。
- **Starred**：Dashboard 中的重点论文；当前实现映射为 `priority: high`。
- **Reading Dashboard**：Obsidian custom view，用于跨日期检索、筛选、回看、汇总和打开资源；它读取 `papers.json`，但不替代 Markdown 编辑器。
- **Host**：承载 arXiv Daily 的外壳，例如 Obsidian 或 VS Code 扩展。
- **Host adapter**：把 core 需要的能力抽象出来的适配层，例如 HTTP、storage、secret、progress、open note、open URL。

## Current State

截至 v0.1.10，已经具备：

- 按一个或多个 arXiv category 抓取 `/recent`，按 arXiv ID 去重后进入同一轮 LLM 筛选。
- Atom API 补全摘要，章节抽取优先保留高价值正文内容。
- 超出 arXiv `/recent` 5 天窗口的显式旧日期补跑，可 fallback 到 export API 的 `submittedDate` 单日窗口，并在日报中标注近似语义。
- 按用户配置的 topics 进行 LLM 分类和筛选。
- 生成每日 Markdown 日报：每篇论文按核心问题 / 关键方法 / 主要结果 / 为什么值得看 / 局限或边界五字段总结，并标注信息来源章节。
- 为 detail topic 生成单篇论文详情页。
- 隐藏主索引 `arxiv-daily/.index/papers.json`：按 arXiv ID 去重，合并 `seenDates` / `dailyReports`，用户控制字段不被 pipeline 覆盖，旧 `index/` 路径自动迁移。
- 日报内保留 `关注 / 重点` checkbox，修改后防抖自动同步到 `papers.json`；插件启动后会补扫 lookback 窗口内日报，处理移动端或其他设备同步回来的勾选。
- 日报标注 `new / seen_before`；ignored 论文不再进入日报。
- 日报末尾折叠列出未入选论文，作为 LLM 漏报兜底。
- 支持手动按日期运行、补跑 lookback、按 arXiv ID 生成详情、手动创建论文笔记、论文 mark 命令。
- 支持日期级 run state：`completed`、`failed_transient`、`failed_permanent`、`skipped`、`running`；支持失败重试、强制重跑、清空状态、取消当前运行。
- 支持 diagnostics 报告，覆盖配置、日期窗口、运行状态和 paper index 一致性。
- Shared core + Node CLI 已可复用同一套 pipeline：`run`、`run-pending`、`summarize`。
- Markdown 链接风格可配置为 Obsidian wikilink 或标准相对链接，方便 CLI / VS Code 输出。
- release 已自动化：push `v*.*.*` tag 会触发 GitHub Actions 构建 Obsidian release assets。

Dashboard 当前状态：

- Ribbon 图标单击直接打开 `arXiv Daily Dashboard`，并默认靠近 ribbon 下方。
- Dashboard 顶部是 `Starred / All` tabs；Starred 只表示 `priority=high`，未 star 的论文保持中性。
- `Refresh / Run Today / Run Pending / More` 与 tabs 在同一工具栏，Dashboard 成为主要操作入口。
- More 菜单承载低频命令：调度开关、Run for date、Force run、Retry failed、Cancel current run、Summarize by arXiv ID、Create paper note、Set paper mark、Show diagnostics、Show recent run state、Clear run state。
- Dashboard 有紧凑月历：有日报的日期会标出，今天会突出显示，并提供 Today 按钮回到当前月份；点击日期直接打开对应日报。
- Dashboard 过滤区包括 Search、Topic、From、To、Note、Detail；From / To 固定按 `seenDates` 过滤，不再提供 Date field 选择。
- Summary stats 包括 Shown、This week、Starred、Notes。
- 论文列表列为 Select、Star、Title、Topic、Published、Actions；不再显示 Mark、First seen / Last seen、citation 或 Zotero 字段。
- 论文标题字体加大，标题、作者和摘要文本可以选中复制。
- 行操作保留 5 个：Open/Create note、Open daily report、Open arXiv、Open PDF、Download PDF；`Add to project` 已从 Dashboard 移除。
- Actions 使用紧凑图标网格，宽度足够时呈现第一行 3 个、第二行 2 个。
- 批量操作保留 Star、Unstar、Create notes 和 Clear selection。

VS Code companion 当前状态：

- 已有独立 VS Code extension scaffold。
- 可识别包含 `arxiv-daily/` 的 workspace folder。
- API key 使用 VS Code SecretStorage。
- Webview Dashboard 可浏览、搜索、筛选、打开资源并修改单篇状态。
- pipeline 命令当前通过 CLI 终端桥接运行。
- 新生成 Markdown 强制使用标准相对链接。

不再作为主线的内容：

- 插件不负责维护 citation key、BibTeX 和写作引用片段。
- Dashboard 不显示 citation / Zotero 字段，也不把“缺 citation”当成待办。
- Zotero 导入流程交给用户在 arXiv 网页用 Zotero 浏览器插件完成。
- 项目笔记服务和旧索引字段可作为兼容遗留存在，但 Dashboard 不把 `Add to project` 作为主路径。

## Design Principles

1. **Dashboard 是主入口，Markdown 仍是正文编辑器**

   Dashboard 负责检索、筛选、回看和打开资源；日报和论文详情仍是普通 Markdown 文件，由 Obsidian / VS Code 原生编辑器处理。

2. **JSON 索引是内部状态主数据源**

   `arxiv-daily/.index/papers.json` 保存论文记录、去重、seen dates、daily reports、star 和 note 路径。不要把长期论文状态放进 `.obsidian/plugins/arxiv-daily/data.json`。

3. **状态保持轻量**

   日常使用只需要 Starred / All。底层 `status` 继续保留兼容性和命令能力，但 Dashboard 不把多状态管理作为默认体验。

4. **Zotero 管 citation**

   arXiv Daily 负责发现和回看；Zotero 负责文献库、citekey、BibTeX 和写作引用。未来最多做轻量跳转或导入状态提示，不在插件里重建 citation manager。

5. **Obsidian 为主，VS Code 为辅**

   Obsidian 长期是第一宿主和移动端方案。VS Code extension 适合代码工作区和终端 agent，但新功能先在 Obsidian Dashboard 验证。

6. **先共享 core，再扩展宿主**

   新宿主通过 host adapter 接入 HTTP、storage、secret、progress 和 open resource。core 不直接依赖 Obsidian API。

## Storage Layout

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

| Path | Role | Created by default |
|---|---|---|
| `arxiv-daily/daily/YYYY-MM-DD.md` | 每日发现入口，按 topic 展示当天相关论文 | Yes |
| `arxiv-daily/.index/papers.json` | 插件内部状态：论文去重、seen dates、daily reports、star、note/PDF 路径 | Yes |
| `arxiv-daily/.index/run-state.json` | Obsidian scheduler 与 Node CLI 共享的日期级运行状态 | Yes |
| `arxiv-daily/papers/<arxiv_id>.md` | 重要论文的长期阅读笔记和深度分析 | Only for detail / saved / manual |
| `arxiv-daily/pdfs/<arxiv_id>.pdf` | 用户手动下载的 arXiv PDF，本地打开优先于远程 PDF URL | Only manual |

日报是当天阅读入口；Dashboard 是跨日期入口。两者都从 `papers.json` 获得去重和状态信息。

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
  category: string;
  categories?: string[];
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

  // Legacy compatibility fields. They should not drive the main Dashboard UX.
  zoteroKey: string;
  zoteroUri: string;
  citationKey: string;
  projects: string[];
}
```

### Paper Note Creation

不是每篇相关论文都需要创建 Markdown：

| Case | Create markdown note |
|---|---|
| arXiv 当天全部论文 | No |
| LLM 判断相关但非重点论文 | No, only index in JSON |
| `detail: true` | Yes |
| Starred / `priority: high` | No, only index in JSON |
| `status: saved` | Yes |
| 用户执行 `Create paper note` | Yes |

轻量相关论文留在 `papers.json`，避免 vault 每天增加大量低价值文件。真正要深入读的论文再进入 `arxiv-daily/papers/<arxiv_id>.md`。

### Status And Priority

Dashboard 默认只强调 Star：

| UI concept | Stored as |
|---|---|
| Starred | `priority: high` |
| Unstarred | `priority: normal` 或 `priority: low` |
| Hidden from normal lists | `status: ignored` |

底层 status 仍保留：

| Status | Meaning |
|---|---|
| `inbox` | 新发现，还没决定是否要读 |
| `to_read` | 已决定之后要读 |
| `reading` | 正在读 |
| `read` | 已读完或已处理 |
| `saved` | 值得长期保留，会创建 Markdown note |
| `ignored` | 不感兴趣，之后日报和 Dashboard 默认列表中不再出现 |

## Release Versioning

发布 tag 与当前插件版本一致，只递增最后一位（例如 `v0.1.9` -> `v0.1.10`）。如果某个里程碑分多次 patch 发布、或中间插入 hotfix，后续计划编号顺延并更新本文档。

发布前最低要求：

- `plugin/manifest.json`、`plugin/package.json`、`plugin/package-lock.json`、`plugin/versions.json` 版本一致。
- `cd plugin && npm run build` 通过。
- `cd plugin && npm test` 通过。
- 提交 release prep commit 后创建 tag。
- tag push 后确认 GitHub release workflow 成功，并把 release notes 改为手写说明。

## Completed Milestones

### v0.1.4: Daily Selection Layer（已发布）

日报成为筛选主入口。每篇论文带 `关注 / 重点` checkbox，修改后自动同步 `papers.json`；按 arXiv ID 去重合并 `seenDates` / `dailyReports`；日报标注 `new / seen_before`，ignored 论文不再出现。

### v0.1.5: Workflow Quick Wins（已完成）

完成启动补扫、多设备 checkbox 同步、日报漏报兜底、多 arXiv 分类、基础手动命令增强。

### v0.1.6: Obsidian Reading Dashboard（已完成）

完成 schema v2、host-neutral Dashboard model、Obsidian custom view、搜索筛选、汇总、打开资源、单篇和批量状态操作。

### v0.1.7: Core Extraction + CLI Fallback（已完成）

完成 host adapter 抽取、Node CLI、共享 run state、submittedDate 旧日期 fallback、link style 配置，并冻结旧 Python 主流程。

### v0.1.8: Research Tool Integrations（已完成，后续降级）

完成过 PDF 下载、本地 PDF 打开、项目笔记服务以及 citation/Zotero 相关试验。基于实际工作流判断，citation/Zotero 不再作为插件主路径；保留 Zotero 外部导入流程。

### v0.1.9: Dashboard UX Iteration（已发布）

围绕 Dashboard 主入口做连续体验修正：精简 row actions、恢复 PDF actions、调整 ribbon 位置、移除 Date field 选择、简化星标流程。

### v0.1.10: Dashboard Primary Entry Release（当前发布）

- Dashboard 标题统一为 `arXiv Daily Dashboard`。
- Ribbon 单击直接打开 Dashboard。
- Starred / All 与 Refresh / Run Today / Run Pending / More 处于同一主工具栏。
- More 菜单收纳低频命令，减少命令入口分散。
- 月历支持日报日期标记、今日突出显示、Today 回到当月、点击打开日报。
- 左侧保留 Search / Topic / From / To / Note / Detail 和 Shown / This week / Starred / Notes。
- 论文标题字体增大，标题/作者/摘要可选中复制。
- Dashboard 不显示 citation / Zotero 字段。
- Row actions 删除 `Add to project`，保留 note / daily / arXiv / open PDF / download PDF。
- Actions 使用 3+2 的紧凑图标布局。

## Next Directions

优先级从高到低：

1. **Dashboard 检索增强**

   - 保存常用筛选视图，例如最近 7 天 Starred、某个 topic、has note。
   - 增加更好的排序：first seen、published、starred first、topic。
   - 增加选中论文的详情侧栏，减少表格横向信息压力。

2. **VS Code companion 收敛**

   - 把 VS Code extension 的本地 Dashboard model mirror 替换成可直接 import 的共享 package。
   - 把终端 CLI bridge 替换成直接 core 调用。
   - 给 VS Code dashboard 对齐 Obsidian 的 Starred / All、月历和 5 个 row actions。

3. **Pipeline 可靠性**

   - 明确区分“arXiv 今日公告还没发布”和“用户显式补跑旧日期”，避免今天尚未 announce 时写入近似日报并停止重试。
   - 加强 run state diagnostics，给出更清晰的 retry / force run 建议。
   - 对 API 临时失败、LLM 限流和 arXiv 页面格式变化补更细的错误分类。

4. **Zotero 外部流程辅助**

   - 不内建 citation 管理。
   - 可选增加轻量检测或跳转：从 Dashboard 打开 arXiv 页面后由 Zotero 浏览器插件导入。
   - 如未来确实需要集成，只做状态提示，不在插件内维护 citekey 作为主数据。

5. **数据清理与兼容迁移**

   - 评估是否迁移或隐藏 legacy `citationKey`、`zoteroKey`、`zoteroUri`、`projects` 字段。
   - Project note 服务保留底层能力，但不作为 Dashboard 主入口；如果后续要恢复，需要重新设计入口和使用场景。

6. **Agent / 文献整理辅助**

   - 在 vault 内生成轻量说明文档，教终端 agent 读取 `papers.json`、理解 Starred 和 seen dates。
   - 支持对话式周报整理，但输出仍应落到普通 Markdown。

## Research Workflows

### Daily Triage

1. 每天自动生成日报。
2. 打开日报或 Dashboard。
3. 大部分论文不操作。
4. 对特别相关或重要的论文点 Star，或在日报中勾 `重点`。
5. 插件把结果同步到 `papers.json`。
6. 后续从 Dashboard 的 Starred、日期月历或搜索找回论文。

### Dashboard Review

1. 打开 `arXiv Daily Dashboard`。
2. 用 Starred 看重点论文，用 All 看完整历史。
3. 用 Search / Topic / From / To / Note / Detail 缩小范围。
4. 通过行操作打开 paper note、source daily report、arXiv、PDF。
5. 对真正要深入读的论文创建 note 或下载 PDF。

### Zotero Follow-up

1. 在 Dashboard 或日报中找到要进文献库的论文。
2. 打开 arXiv 页面。
3. 用 Zotero 浏览器插件导入 Zotero。
4. 写论文时从 Zotero / Better BibTeX 复制 citation key 和 BibTeX。

## Technical Notes

### JSON Index Updates

`papers.json` 是长期 paper index 主索引。更新时需要做到：

- 原子写入，避免写一半损坏 JSON。
- 读入后按 `schemaVersion` 迁移。
- `papers` 以 arXiv ID 为 key。
- `seenDates`、`dailyReports`、`topics` 数组去重。
- pipeline 自动字段可更新，用户控制字段只补缺不覆盖。
- JSON parse 失败时保留原文件并给出 diagnostics。

### Markdown Note Updates

只有重要论文才创建 Markdown note。note 更新需要确保：

- 保留用户手写正文。
- 不破坏已有 frontmatter。
- 缺失字段补齐。
- 用户控制字段不被覆盖。
- JSON 和 Markdown 的关键字段保持一致。

### Dashboard Architecture

Dashboard 实现为 Obsidian custom view，而不是生成长期维护的 `inbox.md`：

- `PaperIndexStore` 继续是唯一状态源。
- Dashboard 启动时 load index，状态更新后 save index 并局部刷新。
- 表格过滤、搜索、排序先在内存中完成；几千篇论文前不需要数据库。
- Paper note 的阅读和编辑通过 `workspace.openLinkText` 打开 Obsidian Markdown editor。
- Dashboard 不直接编辑 note 正文。
- query/filter/action model 与 DOM 视图分离，方便 VS Code extension 继续收敛到共享实现。

## Detailed Roadmap Todo

### v0.1.10 Dashboard Primary Entry

- [x] Ribbon 单击直接打开 Dashboard。
- [x] Ribbon 图标默认靠近下方。
- [x] Dashboard 标题改为 `arXiv Daily Dashboard`。
- [x] Starred / All 与 Refresh / Run Today / Run Pending / More 放在同一行。
- [x] More 菜单收纳低频插件命令。
- [x] 移除 Date field selector，From / To 固定按 `seenDates` 过滤。
- [x] 月历缩小为 Dashboard 的一部分，并支持今日突出显示和 Today 按钮。
- [x] 左侧保留搜索、topic、日期、note、detail 和 summary stats。
- [x] Star 单按钮替代多状态 Mark 下拉。
- [x] 论文标题字体增大。
- [x] 标题、作者和摘要可选中复制。
- [x] 移除 Dashboard 中的 citation / Zotero 显示。
- [x] 移除 `Add to project` row action。
- [x] Row actions 保留 5 个：note、daily、arXiv、open PDF、download PDF。
- [x] Actions 布局改为紧凑 3+2。
- [x] v0.1.10 release prep：版本号、README、PLAN、build 和测试就绪。

### Next Candidate Work

- [ ] Dashboard 保存筛选视图。
- [ ] Dashboard 详情侧栏。
- [ ] Dashboard 排序增强。
- [ ] VS Code extension 复用共享 Dashboard package。
- [ ] VS Code pipeline 命令改成直接 core 调用。
- [ ] 清理 legacy citation / Zotero / projects 字段的迁移策略。
- [ ] 写一份 vault 内 agent 使用说明，帮助终端 agent 整理 Starred 论文和周报。
