# P3 — Core daily and detail workflow

goal_ref: ../goal.md
created: 2026-10-01T19:42:26+08:00
updated: 2026-10-01T20:08:00+08:00
revision: 2

## Outcome

无需已有文献库或活跃 Claude 会话，用户通过独立入口即可按分类/主题完成 arXiv 筛选、日报与详细总结，并按现有产品规则沉淀文件和索引。

## Assumptions

- 基础论文发现先于文献库个性化；没有文献库不是阻塞条件。
- 独立 CLI 已有 `run --today/--date/--id` 和共享 runtime，可作为最先复用的执行入口；不复制 pipeline 或另造权威档案。
- 产品流程仍可调用 LLM；“agent 辅助”指流程控制与产品状态独立，不表示取消已有 AI 筛选和总结。
- 当前 worktree 的基线与主目录并行开发可能不同；接入前按已提交版本核对差异，不带入其他会话未提交内容。

## Verified boundary

| 业务 | 已核实实现 | 新接入应遵守的边界 |
|---|---|---|
| 基础筛选与日报 | `apps/cli/src/runtime.ts` 的 buildCliRuntime 构建 ArxivPipeline、PaperIndexStore、检查点和 SchedulerService；`apps/cli/src/main.ts` 提供 run 命令 | 调用完整任务；不是让 agent 只取若干候选后自行组装日报 |
| 详细总结 | runtime 的 ManualFetchService，及 core pipeline 的自动详报选择 | 保留正文获取、结构化生成、文件/索引关联和已有文件处理规则 |
| 个人文献库方向 | `plugin/main.ts` 的 generatePersonalLibraryDirections 调用 core 的 proposeClusteredPersonalLibraryDirections | 下一阶段复用聚类证据与现有 proposal/profile，而不是读取少量 PDF 后保存独立方向 Markdown |
| 全文索引 | core 的 indexPersonalLibraryFullText；PDF parser、embedding 初始化和生命周期仍有 Obsidian 依赖 | 下一阶段补 Node 适配和共享编排，不把此项设为基础日报前置条件 |
| 个性化发现 | plugin 的 buildPersonalizedDailyDiscoverySnapshot / buildPipeline | 与手动主题组合，沿用证据深度、方向有效性、新意解释和取消约束 |

## Approach

先对已有 CLI 的基础任务建立 Green 基线并明确配置/产物契约；让试验入口复用这些任务，替换 P1 中临场筛选和另建输出的主路径。个人文献库接入与 agent 操作包装保持后续独立阶段。

## Chunks

### Chunk 1 — Establish the existing daily/detail contract

- change kind: behavior-preserving characterization
- strategy: Green characterization baseline
- baseline: 运行现有 CLI run/date/id 与 core pipeline/manual-fetch 相关测试；记录无文献库、零命中、未发布、已有文件和恢复的已有行为。
- Green check: 同样配置和固定输入生成既有日报/详报格式、索引与状态；不依赖 Claude 模型会话。
- regression checks: CLI/runtime/core 相关测试与类型检查；只读调查本身不修改生产代码。
- [x] baseline and contract accepted — 既有 CLI 46 tests、core pipeline/manual/checkpoint 256 tests Green。

### Chunk 2 — Connect the independent experiment to complete product tasks

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red: 针对试验入口调用真实产品任务的契约测试先失败，覆盖未连接文献库仍可运行、完整结果/错误返回和既有产物路径；测试不能只验证拼接命令或转发了几项参数。
- Green: 输入有效配置与 fixture 数据后，经实际任务完成筛选、日报/详报写入及 Paper Index/state 更新；不写 P1 的独立权威档案。
- regression checks: 原 CLI/core 测试、类型检查、boundaries、product-units；涉及 Obsidian 的共享抽取补前后 Green 回归。
- [x] implementation and tests accepted — 新 status/papers 契约先因命令不存在 Red；替换 bundle 测试先因目标产物不存在 Red；之后 113 个 CLI 测试与 6 个独立进程/打包测试 Green。

### Chunk 3 — Validate the product workflow without agent orchestration

- change kind: behavioral integration verification
- strategy: end-to-end fixture run plus proportionate real-provider smoke
- baseline: 隔离配置/输出目录，运行普通命令；不得读写生产 Vault 或已有全局模型配置。
- Green check: 手动主题 → 日期日报 → 单篇详细总结 → 重新读取与重复运行；核对模型调用、结果校验、持久化和恢复由产品负责。
- regression checks: 记录真实外部依赖检查与未运行项；图形界面和库增强不冒充本阶段完成条件。
- [x] end-to-end acceptance complete — 只替换 HTTP、实际使用原 CLI/scheduler/pipeline/native writer/index，完成日报、自动详报、另一 ID 手动详报、跨进程离线重跑、零命中、未发布与用户笔记保护。公网模型质量/供应商可用性不由 fixture 证明，本阶段没有借用生产配置作生成验收。

## Phase verification

- Observed: 所有 workspace typecheck、boundaries、product-units、官方 strict plugin manifest 校验、git diff --check 通过。
- 输出包直接复制官方 CLI 构建，字节一致测试通过；保留原生存储与第三方 notices。源码构建和本机 native loader 使用各自正常构建/缓存目录，未重设 HOME 或修改模型配置。
- 6de6d15：产品状态与 Paper Index 查询；e28bd12：插件复用原 CLI，删除 P1 的 agent.ts/main.ts、独立档案和旧测试。旧用户试用文件保留，测试证据仍在 Git 历史中。

- 用户可直接运行核心任务；不需要描述“先读哪些文件、再逐篇筛选、最后写报告”的长提示词。
- 日报和详细总结均有独立可检查的持久化产物，并受原有索引和状态管理。
- 用户配置手动研究主题即可开始，不要求先完成文献库或方向推断。

## Abort / reshape triggers

- 若为了接入重新实现筛选、总结、索引或方向格式，停止并回到共享核心边界。
- 若验收只证明 agent 能调用工具，却不能证明原有基础业务流程完成，不接受该实现。
- 若未来希望把核心推理改为宿主模型能力，单独设计推理端口与离线执行约束，不隐式改写本阶段目标。
