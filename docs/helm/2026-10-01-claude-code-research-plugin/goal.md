# Claude Code CLI research workflow experiment

status: active
created: 2026-10-01T00:48:41+08:00
updated: 2026-10-01T00:48:41+08:00
revision: 1
owner: codex-claude-code-plugin-2026-10-01

## Intent

先在 Claude Code CLI 中跑通个人文献研究流程：连接本地论文目录，按需查阅本地与 arXiv 资料，由当前 Claude 会话分析，保存研究方向、论文总结和阅读判断，并在新会话继续使用。

## Success criteria

- [ ] 插件通过 Claude CLI 官方校验，并在隔离测试目录完成真实加载验证。
- [ ] 无 Obsidian、无 arXiv Daily 模型 API 配置也能连接论文目录，分页列出可读 PDF，原始文献保持只读。
- [ ] arXiv 候选和单篇资料通过共享 core 获取；模型调用由 Claude 会话负责，输出明确资料范围与证据深度。
- [ ] 研究方向草稿、确认状态、论文总结、阅读记录和按需推荐以可读 Markdown 保存；新进程可查询，更新不静默覆盖已有内容。
- [ ] README 写清首次设置、首次发现、阅读、保存、恢复、当前局限与下一阶段；自动化测试和真实 CLI 验证留有证据。

## Non-goals

- 本阶段不实现桌面阅读页、MCP Apps、DeepSeek Harness、无人值守调度或邮件。
- 不承诺已接入 Obsidian 的全文向量索引、自动聚类或 JSON interest profile；实验记录独立存放，后续迁移另行设计。
- 不宣称阅读反馈已自动改善推荐，不以假数据演示代替真实宿主加载。

## Constraints

- worktree: `.worktree/claude-code-research-plugin`；branch: `feat/claude-code-research-plugin`；base: `84a85e4a1a76e62772e080effa79077e2470c6b3`。
- 保留原目录和其他 worktree 的未提交改动；不推送、不发布、不修改用户 Claude 全局插件配置。
- 用户确认 Claude Code 独立上手、Obsidian 可选；2026-10-01 最终选择 CLI 先跑通，阅读界面形态仍待实际使用后研究。
- 仅在隔离样例目录验收；原始论文只读；研究记录位于用户指定工作目录，不写插件安装缓存。
- 新行为遵循 Red/Green；文档、打包校验和真实宿主分别验证，不能互相代替。

## Phases

1. P1 — CLI 插件完成可恢复的按需文献研究流程 — status: active
2. P2 — 基于真实使用反馈决定完整文献库索引、方向推断和自动推荐的接入范围 — status: pending

## Open questions

- 在 CLI 实际使用后再确定完整阅读界面；此前 Markdown 选择是试跑载体，不是长期产品定案。
