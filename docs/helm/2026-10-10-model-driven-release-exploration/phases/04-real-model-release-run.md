# P4 — real-model-release-run

goal_ref: ../goal.md
created: 2026-10-12T22:00:00+08:00
updated: 2026-10-12T23:30:00+08:00
revision: 4

## Outcome

真实模型以发版模式完整跑完两端任务库并留下证据；每个失败与每条疑点都有处置结论（产品修复、任务/提示修正、测试工具修复或确认误报）；发版默认预算按实测校准；发版执行说明可照文档复现。

## Assumptions

- 用户已接受 token 消耗；使用现有 CLI `[llm]` 配置，密钥不出现在任何输出中。
- 受控模型已证明全部目标任务可完成（P3）；真实模型失败优先归因于观察信息、任务描述或预算，而非直接放宽判定。
- Obsidian 14 个目标任务超过成功标准中的 10–12，属于有意的扩展，不再删减。

## Approach

先各跑一次真实模型发版运行并原样保存证据与用量；逐条分析失败与疑点，按类型分别修复（每个修复独立验收）；修复后再完整跑一次确认结果稳定；最后据两次实测数据设定发版预算默认值并改写 README 的探索与发版部分。

## Chunks

### Chunk 1 — first real-model release runs

- change kind: non-behavioral (evidence collection)
- strategy: proportionate check
- Red / baseline signal: 不适用；记录两端真实运行结果作为基线。
- Green check: `--suite exploration --host obsidian --release` 与 `--host workbench --release` 各跑一次，保存报告；整理逐任务状态、调用、token、耗时与全部疑点。
- regression checks: 无代码改动
- exception: 证据收集，不改代码
- [x] implementation and tests accepted — 结果与处置见 journal.md 2026-10-12。

### Chunk 2 — triage and fixes

- change kind: bug fix / test infrastructure（按发现分别定）
- strategy: 产品缺陷与测试工具缺陷均 strict Red-Green；任务描述修正以受控模型回归为证
- Red / baseline signal: 每个确认的问题先有复现（测试或受控脚本）再修。
- Green check: 每条疑点与失败都有结论记录；修复后受控模型全库回归通过。
- regression checks: npm run test:acceptance:tools；相关 workspace 测试；固定 workbench 与 Obsidian 验收
- [x] implementation and tests accepted (ce8fab9, a5adb9f, 28713a2) — 产品：Add category 先选再存；工作台配置变更提示写明受影响操作。测试工具：任务间重同步配置版本、新增 home 动作、下拉选项去重。受控全库回归两端通过。

### Chunk 3 — confirmation run, budget calibration, release documentation

- change kind: behavior change (defaults) + documentation
- strategy: Red-Green for changed defaults; documentation reviewed against an actual run
- Red / baseline signal: 新默认值的参数解析测试先失败。
- Green check: 修复后真实模型再完整跑一次；按两次实测设定 `--release` 的整次与逐任务预算；README 写明发版命令、看报告、退出码含义（含纯角色运行为 0 但不代表验证通过）、预计成本与耗时、如何把确认的疑点转为固定回归。
- regression checks: npm run test:acceptance:tools；--help/--list 与文档一致
- [x] implementation and tests accepted — 确认运行：workbench 6/6、Obsidian 14/14 通过，0 条疑点；--release 默认 220 步 / 440 调用 / 20 分钟；README 改写为发版流程。

## Phase verification

- 两端各有至少两次真实模型发版运行证据；第二次运行中目标任务的失败均已解释（产品问题已修或已记录）。
- 所有疑点有处置记录；README 命令照做可得到同样形式的报告。
- npm run test:acceptance:tools 与固定验收全部通过。

## Abort / reshape triggers

- 若真实模型在多数目标任务上失败且原因是观察或动作能力不足，停下与用户讨论是否换模型或调整任务难度，而非放宽判定。
- 若单次发版运行成本明显超出估算（> 5M 输入 token 或 > 60 分钟），先报告数据再决定预算。
