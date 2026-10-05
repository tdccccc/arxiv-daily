# arXiv Daily

跟踪研究方向，筛选 arXiv 新论文，将日报与详细论文总结沉淀为本地 Markdown。

[新手教程](getting-started.zh-CN.md) · [English README](../README.md) · [Getting Started](getting-started.md)

**arXiv Daily** 是以 **Obsidian 插件**为主要入口的个人研究助手：抓取关注分类的论文，按研究主题中的具体方向筛选，保存可搜索、可链接、可长期保留的日报与论文总结。独立阅读工作台也可在浏览器或 DeepSeek Harness（DSH）中使用，无需 Obsidian；Claude Code CLI 是可选的 Agent 辅助入口。

## 它能帮你做什么

- **过滤信息过载** — 从大量列表里留下与你主题相关的论文  
- **生成日报** — 默认每天最多 20 篇，按相关性选取后按主题分组，每篇有结构化短摘要；可在设置的 **Daily paper limit** 调整上限
- **论文总结** — 需要对单篇写更深时，生成更长的总结（可自动或按 arXiv ID）  
- **方便回看** — Obsidian Dashboard 或独立工作台：日历、论文列表、搜索、主题筛选与收藏
- **结合已有文献** — 可选接入个人文献库，索引并审核推荐方向，确认后用于每日发现
- **定时运行** — Obsidian 打开时用插件调度，或在长期开机的机器上用 CLI  
- **可选邮件** — 日报成功后发一封简短摘要（自备 Resend，或官方代发 Beta）

## 你会得到什么

| 产出 | 位置 | 说明 |
|---|---|---|
| **日报** | `arxiv-daily/daily/YYYY-MM-DD.md` | 当天的阅读列表：主题、入选论文、结构化短摘要 |
| **论文总结** | `arxiv-daily/papers/<arxiv_id>.md` | 单篇更长的总结（与日报里的条目不是同一份文件） |
| **阅读界面** | Obsidian Dashboard、本地浏览器或 DSH 侧栏 | 浏览日历和论文，打开日报 / 论文总结 / arXiv / PDF |

```text
arxiv-daily/
  daily/          # 日报
  papers/         # 论文总结
  pdfs/           # 可选 PDF
  .index/         # 本地索引与运行状态
```

## 选择使用入口

| 入口 | 适合 | 配置与定时 |
|---|---|---|
| **Obsidian 插件** | 在 Vault 内阅读和整理研究资料 | 原生插件设置；Obsidian 打开时调度 |
| **独立阅读工作台** | 不依赖 Obsidian 阅读与生成报告 | 图形设置保存到 CLI TOML；工作台运行时调度 |
| **DSH 插件** | 在 DSH 固定侧栏打开同一工作台 | 复用 CLI / 工作台配置与本地研究记录 |
| **CLI / Claude Code CLI** | 终端操作、外部定时或 Agent 辅助 | CLI TOML；可选系统 cron；Claude 调用相同产品命令 |

已有 Obsidian 用户可以从**插件**开始；希望独立使用则选择**工作台**。核心文献流程不依赖 Agent 对话。

各入口共用 core 的发现规则、设置定义和研究记录格式。**Obsidian 与 CLI 的配置值、API 密钥不会自动同步**；DSH 与浏览器工作台都使用 CLI 配置。指向同一输出目录可以共享记录，但不等于同步设置。

本文介绍的独立工作台与 DSH **0.1.17** 目前通过源码和本地构建使用，新增功能尚未发布到 npm。请使用下文构建方式，不要假定 npm 最新版已包含这些功能。

---

## Obsidian 插件

### 安装

仅桌面版 Obsidian。

1. **社区插件** — 设置 → 第三方插件 → 浏览 → **arXiv Daily**  
2. **BRAT** — 添加 `tdccccc/arxiv-daily`  
3. **手动** — 从 [最新 Release](https://github.com/tdccccc/arxiv-daily/releases/latest) 将 `manifest.json`、`main.js`、`styles.css` 放入：

```text
<vault>/.obsidian/plugins/arxiv-daily/
```

启用插件后打开 **设置 → arXiv Daily**。

### 快速开始

1. **连接 AI** — API key、Base URL、模型  
2. **选择论文来源** — 一个或多个 arXiv 分类  
3. **描述研究兴趣** — 至少一个有名称的主题，在主题中填写具体研究方向
4. **生成第一份日报** — 设置引导或 Dashboard 的 **Run Today**

引导还提供定时运行设置；所有步骤完成后会保持收起。细节见 [新手教程](getting-started.zh-CN.md)。

### 日常使用

- 打开 **Dashboard**（侧栏图标或命令面板）  
- **Run Today**，或在 Obsidian 打开时让调度自动跑工作日  
- 读 **日报**，给重要论文加星  
- 需要更深时打开或创建 **论文总结**  
- 可选：测试邮件成功后打开邮件自动发送  

### 个人文献库（桌面预览）

可连接一个本地论文目录，包括 Vault 外的目录。访问限定在明确选定的目录中，只读，不跟随符号链接，不改写、重命名或删除原文件。

- 文件清单预览在本地完成，无需模型处理授权。
- 模型处理前展示目录、文件类型、处理深度和实际模型端点，单独取得授权；相关范围变化后需重新授权，也可随时撤销。
- 文献库可以提出主题与方向，只有你确认接受的方向才会进入每日筛选。

---

## 独立阅读工作台与 DSH

在仓库目录中运行，需要 Node.js 20.19+ 及[原生模块构建依赖](../apps/cli/README.md)：

```bash
npm ci
npm run build
node apps/cli/dist/arxiv-daily-cli.cjs ui
```

使用浏览器页面期间保持该进程运行。没有配置时，工作台自动打开首次设置：选择保存目录、配置模型、选择 arXiv 分类，再填写主题与方向。已有 CLI 配置会直接沿用。

左侧日历显示日报状态和论文数，右侧默认展示论文列表，选择后打开日报或单篇论文。阅读、标记和收藏已有论文不调用模型，需要时再生成日报或详细总结。

- 正常阅读 Markdown、表格与 LaTeX 公式，也可查看原始 Markdown；阅读不改动原文件。
- 使用前进、后退浏览当前阅读会话，保留筛选、章节锚点与滚动位置。
- 正文与来源资料分开展示，页尾显示已记录的 Token 用量、生成耗时和具体时间。旧记录缺失值标为“未记录”，整份日报的统计会注明范围。
- 直接修改并保存设置；**外观**中可切换主题和中文 / English 界面语言，与总结语言独立。
- 区分等待公告、当日无更新、筛选无匹配和真实失败；等待公告不消耗普通失败重试次数。

在 DSH 内使用，请参考 [DSH 构建与安装指南](../extensions/dsh-arxiv-daily/README.md)。安装后点击左侧“设置”上方的 **arxiv-daily**，或从右侧栏入口打开，无需先发消息。本地包包含构建平台对应的原生模块；已验证 Linux/x64 与实际 DSH Host 集成，跨平台分发和 Electron 视觉表现尚未完整验收。

需要对话辅助时，可选用 [Claude Code CLI 集成](../extensions/claude-code-arxiv-daily/README.md)。它调用相同的筛选和总结流程；arXiv Daily 使用自己配置的模型端点，与 Claude 的对话模型独立。

当前工作台支持阅读 Markdown，尚不提供 Markdown 编辑。左侧 **个人文献库** 可浏览目录、检索已索引的标题与摘要、打开本地 PDF；**方向审核** 提供候选与文献库概览，可检查代表论文、修改或移动候选、预览匹配，并明确选择接受到研究主题。连接和索引仍在设置中。候选只有接受后才参与每日发现；更完整的运行管理仍待补充。[CLI 指南](../apps/cli/README.md) 保留对应终端命令。

---

## CLI

适合 cron 或长期在线的机器。需要 Node.js 20.19.0+。

### 安装（npm）

需要 Node.js 20.19+。

```bash
npm install -g arxiv-daily
arxiv-daily init          # 交互向导；直接 Enter 可保留默认值
arxiv-daily run --today
```

不装全局也可用：`npx arxiv-daily@latest help`。

配置**只**在 **`$XDG_CONFIG_HOME/arxiv-daily/config.toml`**（默认 `~/.config/arxiv-daily/config.toml`）。没有配置类环境变量，也没有 `--config` / `--vault-root`。init 之后可再手改 topics。

```bash
arxiv-daily update              # 有新版本时升级全局安装
arxiv-daily update --check      # 只查看当前版本与 npm 最新版
arxiv-daily run --date 2026-06-13
arxiv-daily run --id 2606.12345
arxiv-daily email test
# [schedule] 里 enabled = true 后：
arxiv-daily schedule install
```

卸载 CLI 包（**不会**删除配置和 vault 数据）：

```bash
npm uninstall -g arxiv-daily
# 可选：rm -rf ~/.config/arxiv-daily
```

**Windows** 上请用 **WSL** 跑 CLI + cron，或桌面用 **Obsidian 插件** 做定时。

更多：[CLI 安装与命令参考](../apps/cli/README.md)。

### 从本仓库开发

```bash
npm ci
npm run build
npm run cli -- run --today    # 运行 apps/cli/dist/arxiv-daily-cli.cjs
```

---

## 开发

```bash
npm ci
npm run check:boundaries
npm run lint
npm run typecheck
npm test
npm run build
```

单一 npm workspace：`packages/core`、`packages/node-runtime`、`apps/cli`、`plugin`。发版版本同步：`npm run sync:release-version -- <ver>`。
