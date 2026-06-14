# 新手教程

这份教程用于完成 arXiv Daily 在 Obsidian 里的第一次成功运行。

## 开始前

你需要准备：

- Obsidian 桌面端。
- 一个 LLM provider 的 API key。
- 一个或多个 arXiv 分类，例如 `astro-ph`、`cs.LG`、`hep-th`。
- 你希望跟踪的研究主题描述。

arXiv Daily 会把生成内容写入你的 vault。API key 保存在 Obsidian 插件设置中。

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

先选择 provider，然后填写 API key。Provider preset 会自动填入 base URL 和 model，但这两个字段仍然可以手动修改。

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
- Search 和筛选项可以按关键词、topic、日期、是否有 note、是否有 detail 过滤。
- 右侧日历可以按日期打开日报。
- 每行操作可以打开/创建论文笔记、打开来源日报、打开 arXiv、打开 PDF、下载 PDF。

如果某篇论文要进入正式文献库，建议从 Dashboard 打开 arXiv 页面，然后用 Zotero 浏览器插件导入。

## 7. 启用自动运行

确认第一次手动运行成功后，回到 **Settings -> arXiv Daily**，启用 scheduler。

Scheduler 只会在 Obsidian 打开时运行。lookback 窗口内漏掉的工作日会在之后补跑。

## 常见问题

如果 **Run Today** 是 disabled，先完成 **Settings -> arXiv Daily** 顶部 checklist。

如果 Dashboard 显示还没有 indexed papers，先运行今天或运行 pending dates。

如果运行失败，用 **Dashboard -> More -> Show diagnostics** 查看设置、日期上下文和最近运行状态。

如果入选论文太多，缩小 arXiv 分类，或者把 topic description 写得更具体。
