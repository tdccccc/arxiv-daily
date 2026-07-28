# arXiv Daily

按主题筛选 arXiv 新论文，生成 Markdown 日报与论文总结；支持 Obsidian 插件与 CLI。

[新手教程](getting-started.zh-CN.md) · [English README](../README.md) · [Getting Started](getting-started.md)

**arXiv Daily** 按你关心的分类抓取 arXiv，用 LLM 按研究主题筛选，并写成可搜索、可双向链接的 **Markdown**：**日报（Daily report）**、可选的 **论文总结（Paper note）**，以及用于回看的 **Dashboard**。

## 它能帮你做什么

- **过滤信息过载** — 从大量列表里留下与你主题相关的论文  
- **生成日报** — 每天一个 Markdown 文件，按主题分组，每篇有结构化短摘要  
- **论文总结** — 需要对单篇写更深时，生成更长的总结（可自动或按 arXiv ID）  
- **方便回看** — Dashboard：日历、搜索、主题、星标  
- **定时运行** — Obsidian 打开时用插件调度，或在长期开机的机器上用 CLI  
- **可选邮件** — 日报成功后发一封简短摘要（自备 Resend，或官方代发 Beta）

## 你会得到什么

| 产出 | 位置 | 说明 |
|---|---|---|
| **日报** | `arxiv-daily/daily/YYYY-MM-DD.md` | 当天的阅读列表：主题、入选论文、结构化短摘要 |
| **论文总结** | `arxiv-daily/papers/<arxiv_id>.md` | 单篇更长的总结（与日报里的条目不是同一份文件） |
| **Dashboard** | Obsidian 内 | 日历、搜索、筛选、星标，打开日报 / 论文总结 / arXiv / PDF |

```text
arxiv-daily/
  daily/          # 日报
  papers/         # 论文总结
  pdfs/           # 可选 PDF
  .index/         # 本地索引与运行状态
```

## 两种使用方式

| | **Obsidian 插件** | **CLI** |
|---|---|---|
| 适合 | 日常在库里读、用界面和 Dashboard | 服务器、cron、长期开机（例如要 VPN） |
| 配置 | Obsidian 插件设置 | `~/.config/arxiv-daily/config.toml`（先 `init`） |
| 定时 | Obsidian 打开时 | 系统 cron → `run --today`（Windows 建议 WSL） |
| 共通 | 同一套核心流程；指向同一 vault 时目录结构一致 | |

多数人从 **插件** 开始；需要不打开 Obsidian 也能出日报时用 **CLI**。

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
3. **描述研究兴趣** — 至少一个主题（名称、标签、描述）  
4. **生成第一份日报** — 设置引导或 Dashboard 的 **Run Today**

第一份报告完成前引导会保留。细节见 [新手教程](getting-started.zh-CN.md)。

### 日常使用

- 打开 **Dashboard**（侧栏图标或命令面板）  
- **Run Today**，或在 Obsidian 打开时让调度自动跑工作日  
- 读 **日报**，给重要论文加星  
- 需要更深时打开或创建 **论文总结**  
- 可选：测试邮件成功后打开邮件自动发送  

---

## CLI

适合 cron 或长期在线的机器。需要 Node.js 20.11.0+。

配置**只**在 **`$XDG_CONFIG_HOME/arxiv-daily/config.toml`**（默认 `~/.config/arxiv-daily/config.toml`）。不再使用配置类环境变量，也没有 `--config` / `--vault-root`。

```bash
npm ci
npm run build

npm run cli -- init
# 编辑 ~/.config/arxiv-daily/config.toml 中的 topics 与密钥
npm run cli -- run --today
npm run cli -- run --date 2026-06-13
npm run cli -- run --id 2606.12345
npm run cli -- email test
# [schedule] 里 enabled = true 后：
npm run cli -- schedule install
```

**Windows** 上请用 **WSL** 跑 CLI + cron，或桌面直接用 **Obsidian 插件** 做定时。

可执行文件：`apps/cli/dist/arxiv-daily-cli.cjs`。设计说明：[CLI 文档](helm/2026-07-28-cli-product-config-and-data-portability/)。

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
