# P1 — CLI research flow

goal_ref: ../goal.md
created: 2026-10-01T00:48:41+08:00
updated: 2026-10-01T00:51:53+08:00
revision: 2

## Outcome

用户在 Claude Code CLI 中连接文献目录，获取资料、分析并保存结果，新会话可恢复同一研究记录。

## Assumptions

- 先用 Skill + 一个有结构化输出的本地命令验证工作流；暂不需要 MCP 传输层。
- Claude 已具备 PDF 阅读和分析能力；首版复用其 Read 工具，目录检索明确仅匹配文件名。
- 实验输出独立于已有 Obsidian 的权威索引，避免未经设计的双写与迁移。

## Approach

在 `extensions/claude-code-arxiv-daily` 放置独立实验入口。通过 esbuild 打包工作树内共享 core/Node 适配器；保存内容为 Markdown，原文件读取沿用 scoped library source，写入使用版本摘要与共享锁。用真实 `claude --plugin-dir` 会话验证技能发现和结构化命令。

## Chunks

### Chunk 1 — Local workspace and durable records

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: Node 测试调用初始命令表面，连接、目录分页、记录保存和重开契约均返回 not implemented。
- Green check: `node --test extensions/claude-code-arxiv-daily/tests/agent.test.cjs`；覆盖只读目录、路径边界、初始方向为草稿、更新冲突、重开与并发。
- regression checks: `npm run check:boundaries`、独立 tsconfig 类型检查。
- [x] implementation and tests accepted — 5 个契约测试先因 `Not implemented: init` 失败，再全部通过；独立类型检查、boundaries 与 diff check 通过。

### Chunk 2 — Shared arXiv retrieval

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 注入 fixture HTTP transport，经实际 core 解析器验证日期/分页/证据和正文上限，初始命令缺失。
- Green check: 同一 Node 契约测试通过；不执行付费模型请求，不把 listing 当成包含摘要的结果。
- regression checks: core arXiv parser 与相关抓取测试；构建检查所有内部依赖来自当前 worktree。
- [ ] implementation and tests accepted

### Chunk 3 — Claude plugin workflow and actual host smoke

- change kind: behavior change
- strategy: package contract + proportionate host verification
- Red / baseline signal: 缺少 manifest/skills 时插件结构检查失败。
- Green check: `claude plugin validate --strict --json <plugin-dir>`；临时复制插件后桥接命令仍可运行；真实 Claude 会话调用技能，读取隔离样例并保存记录。
- exception: 模型交互不做脆弱的逐字断言；自动化验证持久化结果，真实会话报告工具调用与产物。真实网络失败单独报告，不替代离线契约验证。
- regression checks: `npm run check:product-units`，diff 检查原目录未改动。
- [ ] implementation and tests accepted

## Phase verification

- 跨进程完成 init → library → save direction draft → confirm → save paper/reading → status/read。
- 官方插件校验和真实 CLI 会话证明插件可被发现、使用，而非只有测试桩通过。
- 报告未做的全文索引、自动聚类、后台调度与图形界面。

## Abort / reshape triggers

- 如果 Claude CLI 不能加载插件或恢复研究目录，先修复该路径，不扩大工具范围。
- 如果要修改现有 Obsidian 的权威索引或研究方向格式，停止并重新设计数据兼容；不隐式迁移。
