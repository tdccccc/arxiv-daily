# arXiv Daily

跟踪研究方向，筛选 arXiv 新论文，将日报与详细论文总结沉淀为本地 Markdown。

[新手教程](getting-started.zh-CN.md) · [English README](../README.md) · [Getting Started](getting-started.md)

**arXiv Daily** 是以 **Obsidian 插件**为主要入口的个人研究助手；独立浏览器工作台和 DeepSeek Harness（DSH）扩展同样无需 Obsidian 即可使用。

## 功能

- **过滤信息过载** — 只保留匹配研究主题中具体方向的论文
- **生成日报** — 默认每天最多 20 篇，按主题分组，每篇附结构化短摘要
- **论文总结** — 需要更深内容时生成更长总结，可自动或按 arXiv ID
- **阅读界面** — Obsidian Dashboard、浏览器工作台或 DSH 侧栏，提供日历、搜索、主题筛选与收藏
- **个人文献库** — 可选接入自己的 PDF；审核并明确接受推荐的主题与方向后才参与筛选
- **定时运行** — Obsidian 打开时用插件调度，或在常开机器上用 CLI/cron
- **可选邮件** — 日报成功后发一封简短摘要（自备 Resend，或官方代发 Beta）

日报存放在 `arxiv-daily/daily/YYYY-MM-DD.md`；论文总结存放在 `arxiv-daily/papers/<arxiv_id>.md`。

## 使用方式

| 入口 | 适合 |
|---|---|
| **Obsidian 插件** | 在 Vault 内阅读和整理研究资料 |
| **CLI + 浏览器工作台** | 终端操作、外部定时，或不依赖 Obsidian 的独立阅读 |
| **DSH 扩展** | 在 DeepSeek Harness 固定侧栏打开同一工作台 |

Obsidian 与 CLI 的配置值、API 密钥不会自动同步；浏览器工作台和 DSH 扩展都复用 CLI 配置。

## 安装与快速开始

**Obsidian**（仅桌面版）：

1. **社区插件** — 设置 → 第三方插件 → 浏览 → **arXiv Daily**
2. **BRAT** — 添加 `tdccccc/arxiv-daily`
3. **手动** — 将 [最新 Release](https://github.com/tdccccc/arxiv-daily/releases/latest) 中的 `manifest.json`、`main.js`、`styles.css` 放入 `<vault>/.obsidian/plugins/arxiv-daily/`

启用插件后打开 **设置 → arXiv Daily**。细节见 [新手教程](getting-started.zh-CN.md)。

**CLI 与浏览器工作台**（需要 Node.js 20.19+）：

```bash
npm install -g arxiv-daily
arxiv-daily init          # 交互向导；直接 Enter 可保留默认值
arxiv-daily run --today
arxiv-daily ui            # 在浏览器中打开本地阅读工作台
```

更多：[CLI 安装与命令参考](../apps/cli/README.md)。

**DSH** — 从源码构建并安装扩展：[extensions/dsh-arxiv-daily/README.md](../extensions/dsh-arxiv-daily/README.md)。

## 许可证

MIT — 见 [LICENSE](../LICENSE)。
