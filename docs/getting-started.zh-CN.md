# 新手教程

这份教程用于完成 arXiv Daily 在 Obsidian 里的第一次成功运行。

## 开始前

你需要准备：

- Obsidian 桌面端。
- 一个 LLM provider 的 API key。
- 一个或多个 arXiv 分类，例如 `astro-ph`、`cs.LG`、`hep-th`。
- 你希望跟踪的研究主题描述。

arXiv Daily 会把生成内容写入你的 vault。为保持兼容，API key 以明文保存在插件本地 `data.json`，不是 keyring 或加密存储。保存后设置页只显示 **Configured**，修改或删除需显式使用 **Replace** / **Clear**；日志、诊断和展示给用户的错误会做脱敏。

## 1. 打开插件设置

安装并启用插件后，打开：

```text
Settings -> arXiv Daily
```

设置页顶部有 **Getting Started** checklist。第一次配置时按这里走：

- **LLM API key, base URL, and model**
- **At least one arXiv category**
- **At least one complete research topic**
- **Ready to run**

Checklist 里的按钮会跳转到还没配置好的部分。

## 2. 配置 LLM

先选择 provider，然后填写并保存 API key。保存后的 key 不会重新渲染到页面，而是显示 **Configured** sentinel；需要修改或删除时使用 **Replace** 或 **Clear**。Provider preset 会自动填入 base URL 和 model，但这两个字段仍然可以手动修改。

第一次运行时，temperature、timeout、reasoning 等高级设置可以先保持默认，除非你的 provider 明确要求修改。

## 3. 选择 arXiv 分类

选择你要抓取的 arXiv 分类。可以选多个分类，重复论文会按 arXiv ID 合并。

例子：

- `astro-ph`：天体物理。
- `astro-ph.CO`：宇宙学。
- `cs.LG`：机器学习。

## 4. 添加研究主题

每个 topic 会变成日报里的一个章节。

一个 topic 需要：

- **Name**：日报里的章节标题。
- **Tag**：短的 Obsidian tag slug。
- **Description**：自然语言描述，说明什么论文应该归到这个 topic。

例子：

```text
Name: Photometric Redshift
Tag: photo-z
Description: Methods, benchmarks, uncertainty calibration, catalog construction, and systematics for photometric redshift estimation.
```

如果模板里有接近你方向的配置，可以先加载模板，再按自己的研究方向修改。

## 5. 第一次运行

从左侧 ribbon 图标或命令面板打开 **arXiv Daily Dashboard**。

点击 **Run Today**。插件会依次做这些事：

1. 按配置的分类抓取 arXiv 近期论文。
2. 根据你的 topic 筛选相关论文。
3. 用配置的 LLM 总结入选论文。
4. 写入 Markdown 日报。
5. 更新 Dashboard 索引。

生成的日报默认在：

```text
arxiv-daily/daily/YYYY-MM-DD.md
```

## 6. 使用 Dashboard

设置完成后，Dashboard 就是主要入口。

- **Starred**：显示你标记为重点的论文。
- **All**：显示所有未忽略的历史论文。
- Search 完全在本地进行，按相关度检索 arXiv ID、标题、作者、topic、分类和结构化摘要字段；支持精确现代 arXiv ID、英文技术词和中文切词。有搜索词时默认按相关度排序，显式选择星标/发表日期/topic/标题排序后则保持该主排序。
- **Similar Papers**（论文行的 **Find similar papers** 操作）在未忽略的 Paper Index 条目上做本地 BM25 风格词法检索，显示确定性的匹配原因，不使用网络、LLM、embedding 或数据库。
- 右侧日历可以按日期打开日报。
- 每行操作可以打开/创建论文笔记、查找相似论文、打开来源日报、打开 arXiv、打开 PDF、下载 PDF；相似论文结果可打开 detail、日报、arXiv 页面或 PDF。
- **Dashboard -> More -> Cancel active tasks** 会协作式取消自动/手动日报运行、手动 detail 总结和 PDF 下载。**Get Models** 不在范围内；已经发出的 Obsidian `requestUrl` 请求可能先完成，后续工作才停止。

如果某篇论文要进入正式文献库，建议从 Dashboard 打开 arXiv 页面，然后用 Zotero 浏览器插件导入。

生成的日报和 detail 笔记末尾会有折叠的 **Generation metrics** callout，显示可用的 pipeline 总耗时、LLM 耗时、逻辑调用数、HTTP attempts 和 provider 报告的 tokens。缺失或因重试而不完整的 usage 会显示 unavailable/incomplete，不会记为 0；插件不估算费用。已有设置、Paper Index 和 Markdown 仍可使用，不需要 Paper Index schema migration。

## 7. 启用自动运行

确认第一次手动运行成功后，回到 **Settings -> arXiv Daily**，启用 scheduler。

Scheduler 只会在 Obsidian 打开时运行。lookback 窗口内漏掉的工作日会在之后补跑。

## 常见问题

如果 **Run Today** 是 disabled，先完成 **Settings -> arXiv Daily** 顶部 checklist。

如果 Dashboard 显示还没有 indexed papers，先运行今天或运行 pending dates。

如果运行失败，用 **Dashboard -> More -> Show diagnostics** 查看设置、日期上下文和最近运行状态。

如果入选论文太多，缩小 arXiv 分类，或者把 topic description 写得更具体。
