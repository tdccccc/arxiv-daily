# P1 — exploration-task-suite

goal_ref: ../goal.md
created: 2026-10-10T22:27:32+08:00
updated: 2026-10-10T22:27:32+08:00
revision: 1

## Outcome

`runExplorationLoop` 能连续运行一组目标任务与自由探索角色：单任务失败不中止后续任务，每个任务从确定起点开始，可按 id 选择任务/角色，每任务与整次运行各自受预算约束，报告分开展示目标结果与待复核疑点。

## Assumptions

- 循环已与宿主解耦，只依赖 `driver.observe/execute/capture/dispose` 与 `task.verify`（exploration.mjs:117）；改动集中在 exploration.mjs、run.mjs、report.mjs 与 workbench driver。
- workbench 任务间可在同一 session 内通过导航回首页、关闭对话框、确认界面就绪来复位，无需重启服务端；若复位不足以隔离，则在 driver 层提供可选的重启式复位。
- 自由探索需要一种既非 passed 也非 failed 的终态；报告与退出码需能表达“已探索、仅疑点”而不被误读为通过。
- 假模型（普通 async 函数）足以覆盖循环层行为；真实浏览器只用于阶段验收。

## Approach

在现有循环上做最小扩展：先改连续运行与任务复位（改变一个既有契约：首个非通过任务后停止），再加自由探索任务类型与终态，最后加任务选择、每任务预算和报告展示。每块都先写失败的循环层测试。阶段末用受控模型在真实 workbench 上跑默认任务加一个角色，并人为让中间任务失败，确认后续任务照常执行。

## Chunks

### Chunk 1 — continue past failed tasks with per-task reset

- change kind: behavior change (test infrastructure)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 在 scripts/tests/acceptance-exploration.test.mjs 新增“失败/blocked 任务之后其余任务仍执行”“每个任务开始前调用复位且复位失败使该任务 blocked、不连累其他任务”测试，当前因 exploration.mjs:227 的 `break` 失败；原“stops remaining tasks”测试按新契约改写并记录原因。
- Green check: node --test scripts/tests/acceptance-exploration.test.mjs
- regression checks: npm run test:acceptance:tools；npx eslint 改动文件
- [ ] implementation and tests accepted

### Chunk 2 — free-exploration persona tasks

- change kind: behavior change (test infrastructure)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新测试：无 verify 的角色任务被接受；在 done 或预算耗尽时以独立终态结束，仅携带 findings；永远不计为 passed；目标任务缺 verify 仍被拒绝；报告/退出码把该终态与 passed、failed、blocked、not-run 区分。当前因 exploration.mjs:122 拒绝无 verify 任务而失败。
- Green check: node --test scripts/tests/acceptance-exploration.test.mjs scripts/tests/acceptance-report.test.mjs
- regression checks: npm run test:acceptance:tools
- [ ] implementation and tests accepted

### Chunk 3 — task selection, per-task budgets, usage reporting

- change kind: behavior change (test infrastructure)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新测试：`--task`/`--persona` 只运行所选 id、未知 id 报错；单任务超出步数/调用预算时该任务 blocked 而后续任务继续，整次预算耗尽时其余任务记 not-run；报告逐任务记录模型调用、token 与耗时。当前无这些选项与计数而失败。
- Green check: node --test scripts/tests/acceptance-exploration.test.mjs scripts/tests/acceptance-runner.test.mjs scripts/tests/acceptance-report.test.mjs
- regression checks: npm run test:acceptance:tools；npm run test:acceptance -- --help 显示新选项
- [ ] implementation and tests accepted

## Phase verification

- 受控（脚本化）模型驱动真实 workbench：3 个默认任务 + 1 个角色，其中第 2 个任务被故意判失败；报告显示后续任务照常执行并通过、角色以独立终态结束且疑点可见、逐任务用量存在。
- npm run test:acceptance:tools 全部通过；npm run test:acceptance -- --suite workbench 仍 12/12 通过（固定验收未受影响）。

## Abort / reshape triggers

- 若同 session 复位无法让任务互不影响（复位后仍残留对话框、滚动、草稿或服务端状态），先在 driver 层加入重启式复位；若成本过高需重新划分 P1/P2 边界，L2。
- 若新的探索终态需要改动固定验收的报告语义或退出码约定，停下与用户确认后再继续。
