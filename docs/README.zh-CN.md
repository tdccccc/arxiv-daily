# arXiv Daily

> 面向 Obsidian 的每日 arXiv 发现工具：用 LLM 筛选和总结论文，并通过 Dashboard 跨日期回看。

[English README](../README.md)

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
- **搜索和筛选**：按关键词、topic、首次出现日期、是否有 note、是否有 detail 过滤。
- **聚焦阅读动作**：打开/创建论文笔记、打开来源日报、打开 arXiv、打开 PDF、下载 PDF。
- **Markdown 原生输出**：日报和论文笔记都是普通 Markdown 文件。
- **自动补跑**：Obsidian 打开时会补跑 lookback 窗口内漏掉的工作日。
- **共享 core + CLI**：Obsidian 插件和 Node CLI 复用同一套 pipeline。

## Dashboard

Dashboard 是设置完成后的主要入口。

- **Starred**：显示你标记为重点的论文。
- **All**：显示所有未忽略的历史论文。
- 左侧提供 Search、Topic、From、To、Note、Detail 和汇总数字。
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
- `.index/papers.json`：Dashboard 使用的本地论文索引。
- `.index/run-state.json`：调度器和 CLI 共用的运行状态。

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

## 常用操作

| 操作 | 入口 |
|---|---|
| 打开 Dashboard | 左侧 ribbon 图标或命令面板 |
| 运行今天 | Dashboard 顶部 |
| 补跑 lookback 日期 | Dashboard 顶部 |
| 指定日期运行 | Dashboard **More** 菜单或命令面板 |
| 按 arXiv ID 生成单篇总结 | Dashboard **More** 菜单或命令面板 |
| 打开某天日报 | Dashboard 日历 |
| 标记重点论文 | Dashboard 星标按钮或日报“重点”checkbox |

## 网络与隐私

arXiv Daily 只为抓取和总结论文访问必要服务。

- 访问 `arxiv.org` 和 `export.arxiv.org`，用于获取论文列表、摘要、HTML 页面和用户手动下载的 PDF。
- 访问你在设置中配置的 LLM provider endpoint。发送内容可能包括论文标题、作者、摘要和用于筛选/总结的正文片段。
- API key 保存在 Obsidian 插件设置中；诊断信息不会输出 API key。
- 插件不包含客户端 telemetry。
- 插件不会把 vault 内容发送到 arXiv 和你配置的 LLM provider 之外的服务。
- 默认只在 vault 内的 `arxiv-daily/` 路径写入生成内容。

## CLI 简要说明

Node CLI 可用于 cron 或服务器工作流，但它不是主入口。

```bash
cd plugin
npm install
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
