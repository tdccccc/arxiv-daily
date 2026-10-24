# arXiv Daily — Obsidian 插件

[English](README.md) · 简体中文

跟踪研究方向，筛选 arXiv 新论文，在 Vault 内把日报和论文总结沉淀为本地 Markdown。

本插件是 [arXiv Daily](../README.md) 项目的一部分；不使用 Obsidian 时，同样的发现与阅读流程可通过 [CLI 与浏览器工作台](../apps/cli/README.md) 使用。

## 功能

- **过滤信息过载** — 只保留匹配研究主题中具体方向的论文
- **生成日报** — 默认每天最多 20 篇，按主题分组，每篇附结构化短摘要
- **论文总结** — 按需生成更长的单篇总结，可按比例自动生成，也可按 arXiv ID 手动生成
- **Dashboard** — 日报日历、Starred/All 标签、本地相关性搜索、筛选项，以及行内操作（打开论文笔记、日报、arXiv 页面、PDF）
- **相似论文** — 对已索引论文进行本地离线词法匹配，查询时不发网络或模型请求
- **个人文献库（可选）** — 接入自己的 PDF；推荐的主题/方向需审核并明确接受后才会影响筛选
- **定时运行** — Obsidian 打开时由内置轮询调度；长期在线的机器可用 [CLI](../apps/cli/README.md) 配合 cron
- **可选邮件** — 日报成功后发一封简短摘要，可用自己的 Resend 密钥，或使用官方代发 Beta
- **配置校验** — 定时和手动运行都会校验配置；API 密钥在设置页中默认遮蔽（可点按钮显示），保存在 Vault 插件目录里的插件数据文件 `data.json` 中

日报存放在 `arxiv-daily/daily/YYYY-MM-DD.md`；论文总结存放在 `arxiv-daily/papers/<arxiv_id>.md`。

## 安装

仅支持桌面版——插件无法在 Obsidian 移动端运行。需要 Obsidian 1.5.7 及以上。

1. **社区插件** — 设置 → 第三方插件 → 浏览 → 搜索 **arXiv Daily**
2. **BRAT** — 安装 [BRAT](https://github.com/TfTHacker/obsidian42-brat)，然后添加 `tdccccc/arxiv-daily`
3. **手动安装** — 从[最新 Release](https://github.com/tdccccc/arxiv-daily/releases/latest) 下载 `manifest.json`、`main.js`、`styles.css`，放入 `<vault>/.obsidian/plugins/arxiv-daily/`

启用插件后，打开 **设置 → arXiv Daily**。

## 快速开始

插件内置的五步向导（连接 AI → 选择论文来源 → 描述研究方向 → 生成第一份日报 → 开启每日自动运行）会引导你完成初始配置。完整教程（包括邮件与个人文献库）见[新手教程](../docs/getting-started.zh-CN.md)（English: [Getting Started](../docs/getting-started.md)）。

## 使用

### Dashboard

从左侧栏图标或 **Open reading dashboard** 命令打开；配置完成后它就是日常入口：

- **Starred** / **All** 标签，以及搜索和筛选（主题、日期范围、是否有详细总结）
- **日历** — 显示已有日报的日期；回溯窗口内可运行的日期会高亮，点击即可生成
- 行内操作 — 加星、打开或生成论文总结、查找相似论文、打开日报、打开 arXiv 页面、打开或下载 PDF
- **相似论文** — 针对某篇论文给出确定性的本地匹配结果，附匹配理由和资源链接

已打开的 Dashboard 会在后台运行结束时自动刷新。

### 命令

| 命令 | 作用 |
|---|---|
| `Run today` | 拉取当天论文，写入日报，并写入已选中的论文总结 |
| `Run for date…` | 对最近 5 天内的某个日期运行 |
| `Run all pending in lookback window` | 运行回溯窗口内所有待处理日期 |
| `Retry failed dates in lookback window` | 清除最近失败日期的状态并重新运行 |
| `Force run for date…` | 清除某个日期的已存状态，跳过调度保护直接重新运行 |
| `Clear run state…` | 清除已持久化的运行状态，不删除笔记 |
| `Cancel active tasks` | 取消正在进行的运行、论文总结、PDF 下载，以及个人文献库的扫描、索引和方向生成 |
| `Summarize by paper ID…` | 为某个 arXiv ID 写入一份论文总结 |
| `Set paper mark…` | 将某篇论文标记为 Unmarked / Watch / Highlight / Saved / Ignored |
| `Create paper note…` | 为已索引论文创建一份简易笔记 |
| `Mark current paper as <mark>` | 设置当前打开论文笔记的标记 |
| `Open today's daily report` | 打开当天日报 |
| `Open reading dashboard` | 打开 Dashboard 视图 |
| `Send test email` | 用当前邮件设置发送一封测试邮件 |
| `Show recent run state` | 列出最近 20 个日期及其状态 |
| `Show run history` | 显示完整运行历史 |
| `Show diagnostics` | 显示可复制的本地诊断报告（不含 API 密钥） |

个人文献库还有一组专用命令（审核/接受推荐方向、预览/扫描/重载目录、索引与搜索全文、清理旧的搜索索引、诊断全文运行环境）——详见[新手教程](../docs/getting-started.zh-CN.md)。

手动运行会先校验配置：`Run today`、`Run all pending`、`Run for date…` 需要有效的 LLM 配置和至少一个主题；`Summarize by paper ID…` 只需要有效的 LLM 配置。

### 标记论文

在 Dashboard 里用行上的星标按钮加星（也可对选中的行使用 **Star** / **Unstar**）。需要更细的标记时，用 `Set paper mark…`（Unmarked / Watch / Highlight / Saved / Ignored），或在已打开的论文总结上用 `Mark current paper as <mark>`。标记保存在论文索引里，不写进 Markdown，所以日报里没有可勾选的复选框；旧日报可能还留着复选框，勾选它们不会有任何效果。

### 定时运行

内置的单一轮询循环（默认每 20 分钟一次）**仅在 Obsidian 打开时**运行：它会遍历回溯窗口，跳过已完成/已跳过/周末/尚未到时间的日期，运行其余日期。如果今天的日报还没有结果，开启定时时会询问是运行今天、跳过今天，还是取消；如果今天已经有结果，则直接开启。

需要在无人值守、长期在线的机器上运行时，请改用 [CLI](../apps/cli/README.md) 配合 cron。

## 设置概览

| 分区 | 内容 |
|---|---|
| **Enable** | 开启或关闭内置定时 |
| **LLM** | API base URL、API 密钥、模型、推理强度 |
| **arXiv categories** | 要拉取的 arXiv 分类 |
| **Research topics** | 主题卡片（名称、标签、描述、方向、是否生成详细总结）和模板 |
| **Automatic detail notes** | 自动生成较长论文总结的频率（Fewer / Recommended / More） |
| **Timezone** | 日报和定时以哪个时区判断“今天” |
| **Output & schedule** | 每日论文上限、日报文件夹、论文总结文件夹、链接样式、总结语言、运行时间窗口、检查间隔（分钟） |
| **Personal library** | 文献库（选择文件夹、建立索引、撤销授权）、来自文献库的主题（审核推荐方向）、Embedding（本地或远程）、更好的 PDF 解析器（可选的本地 sidecar） |
| **Email delivery** | 发送方式、你的邮箱、验证码、Resend API 密钥、发件邮箱、发件人名称、发送测试邮件、每日自动发送 |
| **Advanced** | 日志级别 |
| **Help & feedback** | 预填内容的缺陷报告和功能请求 Issue、文档、代码仓库 |

回溯窗口固定为 5 天，不是设置项。

## 邮件与个人文献库

邮件功能可选，发送失败不会影响日报本身。个人文献库的访问是只读的，且仅限于你选定的文件夹；库可以推荐主题和方向，但只有你明确接受的才会参与筛选。两者的详细说明见[新手教程](../docs/getting-started.zh-CN.md)。

## 故障排查

常见问题（运行按钮不可点、Dashboard 没有论文、运行失败、邮件报错、`data.json` 无法读取等）见[新手教程的常见问题表](../docs/getting-started.zh-CN.md#常见问题)。

中断后重新运行会从最后完成的步骤继续，不会重复已付费的模型调用。

## 开发

在仓库根目录执行：

```bash
npm ci
npm run build          # 构建所有工作区，包括本插件
npm test
```

只构建插件本体：`cd plugin && npm run build`（基于 esbuild，产物 `main.js` 与 `manifest.json`、`styles.css` 在同一目录）。将这三个文件复制到某个 Vault 的 `.obsidian/plugins/arxiv-daily/` 即可测试本地构建。

## 许可证

MIT — 见 [LICENSE](../LICENSE)。
