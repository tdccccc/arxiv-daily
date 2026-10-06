# P1 — CLI research flow

goal_ref: ../goal.md
created: 2026-10-01T00:48:41+08:00
updated: 2026-10-01T01:23:00+08:00
revision: 4

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
- [x] implementation and tests accepted — recent/paper 命令缺失时 2 个契约 Red，接入后共 8 个测试通过；全文提取为已有 core 的 Green 特性验证；core 四个相关文件 31 个回归测试通过。

### Chunk 3 — Claude plugin workflow and actual host smoke

- change kind: behavior change
- strategy: package contract + proportionate host verification
- Red / baseline signal: 缺少 manifest/skills 时插件结构检查失败。
- Green check: `claude plugin validate --strict --json <plugin-dir>`；临时复制插件后桥接命令仍可运行；真实 Claude 会话调用技能，读取隔离样例并保存记录。
- exception: 模型交互不做脆弱的逐字断言；自动化验证持久化结果，真实会话报告工具调用与产物。真实网络失败单独报告，不替代离线契约验证。
- regression checks: `npm run check:product-units`，diff 检查原目录未改动。
- [x] implementation and tests accepted — manifest、命令行、可移动包 3 个契约先 Red 后 Green；最终 11 个实验测试通过。Claude Code 2.1.285 实际加载 Skill、读取测试 PDF、保存三个记录，独立冷进程回读通过。

## Phase verification

- 跨进程完成 init → library → save direction draft → confirm → save paper/reading → status/read。
- 官方插件校验和真实 CLI 会话证明插件可被发现、使用，而非只有测试桩通过。
- 报告未做的全文索引、自动聚类、后台调度与图形界面。

### Observed evidence

- `node --test extensions/claude-code-arxiv-daily/tests/*.test.cjs`：11 passed。
- `node scripts/run-core-tests.mjs arxiv-parser.test.ts atom-parser.test.ts section-extractor.test.ts paper-content.test.ts`：4 files / 31 passed。
- 独立 TypeScript、官方 `claude plugin validate --strict --json`、boundaries、product-units、diff check 全部通过。
- 真实 arXiv 查询 `1706.03762` 返回 `Attention Is All You Need` 和 1136 字符摘要；测试设 20 秒中止上限并成功完成。
- `.artifacts/claude-smoke/` 为本地忽略的测试输出：真实 CLI 会话加载 `arxiv-daily:research`，调用 helper 的 init/library/save/status/read，Read 工具读到合成 PDF 第 1 页。冷进程确认 direction/paper/reading 各 1 条，confirmedDirections 为 0，摘要保留测试证据限制。
- 默认模型网关返回 400（模型不可用）；实际会话测试临时使用该网关公开返回的兼容模型 `gpt-6-astra-cc-format[1m]`，没有修改用户全局模型配置。隔离测试先关闭所有设置导致认证缺失，随后保留 user 认证、关闭已有扩展/hooks，使用临时 Node 命令权限完成验证。
- 全仓库 lint/typecheck/test/build、跨平台桌面、真实大文献库、完整个性化日报端到端未运行；本次只验证新的独立 CLI 宿主及其所用 core 回归。

## Abort / reshape triggers

- 如果 Claude CLI 不能加载插件或恢复研究目录，先修复该路径，不扩大工具范围。
- 如果要修改现有 Obsidian 的权威索引或研究方向格式，停止并重新设计数据兼容；不隐式迁移。
