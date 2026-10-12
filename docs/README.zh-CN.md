[English](../README.md) · 简体中文

# arXiv Daily

跟踪研究方向，筛选 arXiv 新论文，将日报与详细论文总结沉淀为本地 Markdown。

![arxiv daily：跟踪研究方向，全部保存在本地 Markdown。在 Obsidian 社区插件中搜索 arXiv Daily 即可安装。](assets/readme-hero.png)

**arXiv Daily** 是以 **Obsidian 插件**为主要入口的个人研究助手；独立浏览器工作台和 DeepSeek Harness（DSH）插件同样无需 Obsidian 即可使用。

## 功能

- **过滤信息过载** — 只保留匹配研究主题中具体方向的论文
- **生成日报** — 默认每天最多 20 篇，按主题分组，每篇附结构化短摘要
- **论文总结** — 需要更深内容时生成更长总结，可自动或按 arXiv ID
- **阅读界面** — Obsidian Dashboard、浏览器工作台或 DSH 侧栏，提供日历、搜索、主题筛选与收藏
- **个人文献库** — 可选接入自己的 PDF；审核并明确接受推荐的主题与方向后才参与筛选
- **定时运行** — Obsidian 打开时用插件调度，或在常开机器上用 CLI/cron
- **可选邮件** — 日报成功后发一封简短摘要（自备 Resend，或官方代发 Beta）

日报存放在 `arxiv-daily/daily/YYYY-MM-DD.md`；论文总结存放在 `arxiv-daily/papers/<arxiv_id>.md`。

## 选择使用方式

| 入口 | 适合 |
|---|---|
| **Obsidian 插件** | 在 Vault 内阅读和整理研究资料 |
| **CLI + 浏览器工作台** | 终端操作、外部定时，或不依赖 Obsidian 的独立阅读 |
| **DSH 插件** | 在 DeepSeek Harness 固定侧栏打开同一工作台 |

Obsidian 与 CLI 的配置值、API 密钥不会自动同步；浏览器工作台和 DSH 插件都复用 CLI 配置。

## 安装

### Obsidian 插件（仅桌面版）

1. **社区插件** — 设置 → 第三方插件 → 浏览 → **arXiv Daily**
2. **BRAT** — 添加 `tdccccc/arxiv-daily`
3. **手动** — 将 [最新 Release](https://github.com/tdccccc/arxiv-daily/releases/latest) 中的 `manifest.json`、`main.js`、`styles.css` 放入 `<vault>/.obsidian/plugins/arxiv-daily/`

启用插件后打开 **设置 → arXiv Daily**。

### CLI + 浏览器工作台（需要 Node.js 20.19+）

```bash
npm install -g arxiv-daily
arxiv-daily init          # 交互向导；直接 Enter 可保留默认值
arxiv-daily run --today
arxiv-daily ui            # 在浏览器中打开本地阅读工作台
```

### DSH 插件（需要 Node.js 22.19+）

```bash
dsh plugin --profile web add dsh-arxiv-daily
```

重启 DSH，然后点击左侧栏“设置”上方的 **arxiv-daily**（或右侧栏对应标签）。DSH Desktop：**Plugins → Add plugin**，填写包名 `dsh-arxiv-daily`。

## 快速开始

第一次使用？跟着[新手教程](getting-started.zh-CN.md)（English: [Getting Started](getting-started.md)）走一遍完整的首次设置：连接模型、选择 arXiv 分类、描述研究主题，并生成第一份日报。

## 文档

| 文档 | English |
|---|---|
| [新手教程](getting-started.zh-CN.md) | [Getting Started](getting-started.md) |
| [CLI 安装与命令参考](../apps/cli/README.zh-CN.md) | [English](../apps/cli/README.md) |
| [浏览器工作台指南](workbench.zh-CN.md) | [English](workbench.md) |
| [Obsidian 插件](../plugin/README.zh-CN.md) | [English](../plugin/README.md) |
| [DSH 插件](../extensions/dsh-arxiv-daily/README.zh-CN.md) | [English](../extensions/dsh-arxiv-daily/README.md) |

## 贡献

见[功能与交互验收](../scripts/acceptance/README.md)，覆盖真实浏览器/桌面检查和可选的模型 API 探索。

## 许可证

MIT — 见 [LICENSE](../LICENSE)。
