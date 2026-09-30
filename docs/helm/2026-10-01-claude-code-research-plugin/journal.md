# Journal

## 2026-10-01 — note

- evidence: 用户要求隔离 worktree，确认不依赖 Obsidian；随后重新讨论单应用体验，并最终选择先做 Claude Code CLI 跑通。
- change: 从当前已提交版本建立 `feat/claude-code-research-plugin`。CLI 是首轮验证入口，最终阅读界面尚未决定。
- disposition: 不带入原目录设置页改动或其他文献库 worktree 的在途修改；保留共享 core，实现独立实验宿主。
- next: P1，先写本地研究记录与共享 arXiv 抓取的契约测试，再构建插件并通过真实 Claude 会话验收。

## Research references

- Claude 插件安装与 CLI/桌面范围：https://code.claude.com/docs/en/plugins/install
- 插件组件与持久化目录：https://code.claude.com/docs/en/plugins/components
- 桌面 Code 可用文件和 PDF 预览，但本次未选桌面入口：https://code.claude.com/docs/en/desktop
- MCP Apps 是另一个可调研的 UI 承载方式，本阶段不实现：https://claude.com/docs/connectors/building/mcp-apps/getting-started

## 2026-10-01 — note

- evidence: Chunk 1：5 个真实文件系统契约测试先 Red（init 未实现），后 Green；独立 TypeScript 检查、workspace boundaries、git diff --check 均通过。
- change: 提供只读目录清单、独立 Markdown 记录、方向草稿/确认、摘要版本冲突检查及共享进程锁。目录检索明确仅匹配文件名。
- disposition: 接受该独立变更；不修改现有 CLI/Obsidian 配置或索引；保留原文献目录只读。
- next: 共享 arXiv 抓取与真正的 CLI 插件入口。
