# Independent arXiv Daily core with auxiliary Claude Code access

status: active
created: 2026-10-01T00:48:41+08:00
updated: 2026-10-01T19:42:26+08:00
revision: 3
owner: codex-claude-code-plugin-2026-10-01

## Intent

保留并复用 Obsidian 插件的完整业务核心，在独立运行环境中提供 arXiv 论文筛选、日报与详细总结沉淀，再以个人文献库、方向推断和个性化推荐增强该主流程。Claude Code CLI 是辅助配置、操作和解释结果的入口；对话不是当前产品主体。

## Success criteria

- [ ] 基础路径无需已有文献库：配置 arXiv 分类和手动研究主题即可执行完整筛选与日报流程。
- [ ] 日报复用现有 ArxivPipeline 的候选获取、筛选、结构化总结、写入与状态恢复；遵循既有零命中、未发布、失败重试和重复运行语义。
- [ ] 支持按 arXiv ID 生成详细总结并持久化，复用既有正文获取、总结、文件与 Paper Index 关联；保留原有自动详报选择能力。
- [ ] 日报、论文总结、阅读状态和索引沿用既有产品的数据语义与存储规则；不另建一套 agent 专属权威研究档案。
- [ ] 个人文献库作为可选增强：复用识别、PDF 解析、全文索引、方向草稿/确认、增量建议与个性化筛选；手动主题路径始终可用。
- [ ] 无活跃 Claude 会话时，独立程序仍可按配置运行上述核心任务。模型参与筛选和总结，但调用、校验、状态与持久化由产品流程管理。
- [ ] Claude Code 调用同一套业务任务，辅助配置、启动/取消/查询、解释推荐与按需问答；其加入不改变核心决策和数据契约。
- [ ] 通过无文献库的基础日报/详报、带文献库的个性化发现、命令与 agent 两种入口的验收；原 Obsidian 行为回归通过。

## Non-goals

- 不构建以聊天为中心的通用 Research Agent，不让 agent 每次临场重组既有筛选/总结业务流程。
- 不再扩展 P1 的文件名抽样 + 独立 Markdown 方向档案作为正式产品方案。
- 本轮不实现新的桌面阅读页、MCP Apps 或 DeepSeek Harness；长期阅读界面仍待产品研究。
- 不宣称现有阅读反馈已自动影响推荐，不以原型测试通过代替核心工作流一致性验收。

## Constraints

- worktree: `.worktree/claude-code-research-plugin`；branch: `feat/claude-code-research-plugin`；base: `84a85e4a1a76e62772e080effa79077e2470c6b3`。
- 保留原目录和其他 worktree 的未提交改动；不推送、不发布、不修改用户 Claude 全局插件配置。
- 2026-10-01 用户再次确认独立运行，复用 Obsidian 完整业务核心，agent 为辅助入口；随后明确基础 arXiv 筛选、日报和详细总结必须优先于文献库方向增强。
- 先沿用产品已有 LlmClient 和模型配置能力，不假设 Claude 订阅自动提供无人值守业务调用；模型凭据/费用边界需在接入说明中明确。
- 复用已有 core；把仍在 Obsidian 宿主中的通用编排抽出并补齐 Node 适配。共享业务数据语义不意味着自动同步各产品设置。
- 保留既有调度与断点恢复机制；先验收手动发起的完整业务流程，再验证独立调度，不把运行生命周期绑定到 Claude 会话。
- 仅在隔离样例目录验收；原始论文只读；研究记录位于用户指定工作目录，不写插件安装缓存。
- 新行为遵循 Red/Green；文档、打包校验和真实宿主分别验证，不能互相代替。

## Phases

1. P1 — agent 主导的按需研究原型（已验证可运行，但不符合修订后的产品目标） — status: superseded
2. P2 — 围绕原型继续扩展文献库功能 — status: superseded
3. P3 — 独立入口保留完整 arXiv 筛选、日报与详细总结沉淀 — status: active
4. P4 — 可选文献库与方向增强复用既有核心并接入同一发现主流程 — status: pending
5. P5 — Claude Code 辅助入口与直接运行使用同一套业务能力和研究记录 — status: pending

## Open questions

- 长期查看入口如何集中呈现日报、论文总结和文献库；CLI 先跑通不等于最终接受多应用切换。
- P1 实验记录仅保留为试验资料，若未来需要迁移必须显式设计，不自动导入现有索引或 confirmed interest profile。
