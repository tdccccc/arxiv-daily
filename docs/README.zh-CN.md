# arXiv Daily

> 每日 arXiv 论文跟踪器：LLM 筛选摘要、按主题组织日报、Dashboard 含日历搜索筛选、PDF 笔记，支持 CLI。

[新手教程](https://github.com/tdccccc/arxiv-daily/blob/main/docs/getting-started.zh-CN.md) · [English README](https://github.com/tdccccc/arxiv-daily/blob/main/README.md)

arXiv Daily 的目标不是替代 Zotero 或 PDF 阅读器，而是把每天新出现的 arXiv 论文稳定地接入 Obsidian：自动抓取、按研究主题筛选、生成 Markdown 日报，并用 **arXiv Daily Dashboard** 回看重点论文。

## 核心流程

1. 配置 arXiv 分类、研究主题和 LLM provider。
2. 插件按计划自动运行，或从 Dashboard 手动运行。
3. 阅读当天 Markdown 日报，对重要论文点星标或勾选“重点”。
4. 之后通过 Dashboard 搜索、筛选、打开日报、论文笔记、arXiv 页面或 PDF。
5. 真正要进入文献库时，打开 arXiv 页面，用 Zotero 浏览器插件导入。

Zotero 仍然负责 citation key、BibTeX 和正式文献库管理。

## 主要功能

- **Dashboard 作为主入口**：左侧 ribbon 图标单击直接打开 Dashboard。
- **Starred / All 工作流**：只给重要论文点星标，未星标论文保持中性。
- **日报日历**：有日报的日期会标出，今天会高亮，点击日期即可打开对应日报。
- **本地相关度搜索**：覆盖 arXiv ID、标题、作者、topic、分类和结构化摘要字段；支持英文技术词及中文双字切词，并优先精确匹配现代 arXiv ID（含 URL/version 形式）。有搜索词时默认按相关度排序；显式选择星标、发表日期、topic 或标题后，该排序保持为主排序。
- **Similar Papers**：在未忽略的 Paper Index 条目上进行本地 BM25 风格词法检索，显示确定性的匹配原因，并可打开 detail、来源日报、arXiv 或 PDF；不使用网络、LLM、embedding 或数据库。
- **聚焦阅读动作**：打开/创建论文笔记、打开来源日报、打开 arXiv、打开 PDF、下载 PDF。
- **统一取消**：**Cancel active tasks** 协作式取消自动/手动日报运行、手动 detail 总结和 PDF 下载；**Get Models** 不在取消范围内。Obsidian 已经发出的 `requestUrl` 请求可能先完成，但后续工作会停止。
- **Markdown 原生输出**：日报和论文笔记都是普通 Markdown 文件。
- **自动补跑**：Obsidian 打开时会补跑 lookback 窗口内漏掉的工作日。
- **共享 core + CLI**：Obsidian 插件和 Node CLI 复用同一套 pipeline。

## Dashboard

Dashboard 是设置完成后的主要入口。

- **Starred**：显示你标记为重点的论文。
- **All**：显示所有未忽略的历史论文。
- 左侧提供 Search、Topic、From、To、Note、Detail 和汇总数字；Search 使用上述本地相关度索引，并在相关度排序时显示匹配字段原因。
- Sort 控件可以按相关度、星标优先、发表日期、topic 或标题切换列表顺序；用户显式选择的排序不会被搜索覆盖。
- 右侧日历可以直接打开某一天的日报。
- 每行论文保留标题、作者、摘要和常用操作。

Dashboard 读取 `arxiv-daily/.index/papers.json`。它不替代 Markdown 编辑器；日报和论文笔记仍然用 Obsidian 原生编辑器打开。

## 输出目录

默认写入 vault 内的 `arxiv-daily/`：

```text
arxiv-daily/
  daily/
    2026-06-13.md
  papers/
    2606.12345.md
  pdfs/
    2606.12345.pdf
  .index/
    papers.json
    run-state.json
```

- `daily/YYYY-MM-DD.md`：按 topic 分组的每日发现日报。
- `papers/<arxiv_id>.md`：detail 论文或手动创建的论文笔记。
- `pdfs/<arxiv_id>.pdf`：手动下载的 arXiv PDF。
- `.index/papers.json`：Dashboard 使用的本地论文索引；搜索和 Similar Papers 只构建派生的内存索引，不修改其 schema。
- `.index/run-state.json`：调度器和 CLI 共用的运行状态。

这些功能不需要 Paper Index schema migration；已有设置、Paper Index 和 Markdown 文件仍可继续使用。

## 安装

arXiv Daily 仅支持 Obsidian 桌面端。

### Community Plugins

插件通过 Obsidian 社区审核后：

1. 打开 **Settings -> Community plugins -> Browse**。
2. 搜索 **arXiv Daily**。
3. 安装并启用。

### BRAT Beta

社区列表完全可用前，可以用 [BRAT](https://github.com/TfTHacker/obsidian42-brat) 安装：

1. 安装并启用 BRAT。
2. 打开 **BRAT settings -> Add Beta plugin**。
3. 输入：

```text
tdccccc/arxiv-daily
```

### 手动安装

从最新 release 下载 `manifest.json`、`main.js`、`styles.css`：

https://github.com/tdccccc/arxiv-daily/releases/latest

放到：

```text
<vault>/.obsidian/plugins/arxiv-daily/
```

然后重启 Obsidian 并启用 **arXiv Daily**。

## 快速开始

第一次使用建议先看 [新手教程](https://github.com/tdccccc/arxiv-daily/blob/main/docs/getting-started.zh-CN.md)。

1. 打开 **Settings -> arXiv Daily**。
2. 选择 LLM provider 并填写 API key。
3. 选择一个或多个 arXiv 分类。
4. 添加至少一个研究主题。
5. 启用调度器，或打开 Dashboard 点击 **Run Today**。

研究主题是自然语言描述，例如“photo-z 方法、目录构建、系统误差校正”。LLM 会根据这些主题判断论文是否相关以及应该归到哪个 topic。

## 日报

日报是普通 Markdown 文件。每篇入选论文包含：

- 作者和 arXiv 链接
- 用于总结的信息来源章节
- 核心问题
- 关键方法
- 主要结果
- 为什么值得看
- 局限或边界
- 用于 Markdown triage 的“关注 / 重点”checkbox

在日报里勾选“重点”会映射为 Dashboard 里的星标。

日报和生成的 detail 笔记末尾会附加折叠的 **Generation metrics** callout：在可用时显示 pipeline 总耗时，并显示 LLM 耗时、逻辑调用数、HTTP attempts 和 provider 实际报告的 token usage。缺失 usage 会显示 unavailable/incomplete，而不是记为 0；重试时若失败 attempt 的 usage 不可得，也会标为 incomplete。插件不估算费用。旧 Markdown 无需重写，仍可使用。

## 常用操作

| 操作 | 入口 |
|---|---|
| 打开 Dashboard | 左侧 ribbon 图标或命令面板 |
| 运行今天 | Dashboard 顶部 |
| 补跑 lookback 日期 | Dashboard 顶部 |
| 指定日期运行 | Dashboard **More** 菜单或命令面板 |
| 按 arXiv ID 生成单篇总结 | Dashboard **More** 菜单或命令面板 |
| 取消 active tasks | Dashboard **More** 菜单或命令面板 |
| 查找相似论文 | 论文行的 **Find similar papers** 操作 |
| 打开某天日报 | Dashboard 日历 |
| 标记重点论文 | Dashboard 星标按钮或日报“重点”checkbox |

## 网络与隐私

arXiv Daily 只为抓取和总结论文访问必要服务。

- 访问 `arxiv.org` 和 `export.arxiv.org`，用于获取论文列表、摘要、HTML 页面和用户手动下载的 PDF。
- 访问你在设置中配置的 LLM provider endpoint。发送内容可能包括论文标题、作者、摘要和用于筛选/总结的正文片段。
- 已保存的 API key 在设置页只显示 **Configured**；修改或删除必须显式使用 **Replace** / **Clear**。为保持兼容，key 仍以明文保存在插件本地 `data.json`，不宣称使用 keyring 或加密；日志、诊断和展示给用户的错误会做脱敏。
- 插件不包含客户端 telemetry。
- 插件不会把 vault 内容发送到 arXiv 和你配置的 LLM provider 之外的服务。
- 默认只在 vault 内的 `arxiv-daily/` 路径写入生成内容。

## CLI 简要说明

Node CLI 可用于 cron 或服务器工作流，但它不是主入口。

```bash
# 在仓库根目录执行
npm ci
npm run build

ARXIV_DAILY_API_KEY=sk-... npm run cli -- run-pending --vault-root /path/to/vault
```

使用配置文件：

```bash
npm run cli -- run --date 2026-06-13 --config arxiv-daily.config.json --vault-root /path/to/vault
npm run cli -- summarize --id 2606.12345 --config arxiv-daily.config.json --vault-root /path/to/vault
```

根目录的 `arxiv_daily.py` 只是兼容 shim，会转发到 Node CLI，不再维护独立 Python pipeline。

## 开发

实现细节和开发文档见 [plugin/README.md](../plugin/README.md)。

```bash
cd plugin
npm install
npm test
npm run build
```

## License

[MIT](../LICENSE)
