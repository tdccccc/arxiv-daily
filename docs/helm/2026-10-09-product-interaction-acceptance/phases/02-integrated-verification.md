# P2 — integrated-verification

goal_ref: ../goal.md
created: 2026-10-09T19:02:12Z
updated: 2026-10-09T19:02:12Z
revision: 1

## Outcome

默认命令在当前构建上完成两端验收，浏览器回归可进入 CI，使用与覆盖边界有可复现说明。

## Assumptions

- P1 的三个产品缺陷修复已有定向 Red/Green、真实场景及完整工作区回归证据。
- Obsidian 插件就绪可能早于工作区挂载，需要等待既有正向能力，而非放宽可用性标准。
- 公共 CI 只运行固定工作台场景；本机 Obsidian 与用户模型 API 保持显式本地验收。

## Approach

补启动就绪等待、接入固定浏览器 CI，然后运行最终统一命令并审查报告和文档。

## Chunks

### Chunk 1 — wait for a usable workspace before acceptance

- change kind: bug fix in test infrastructure
- strategy: Red-Green-Refactor
- Red / baseline signal: plugin 已就绪但工作区仍为 Loading vault，立即检查错误地产生 blocked；新增延迟 leaf 的契约先失败。
- Green check: 等待延迟的既有正向能力；一直缺失时仍 blocked，零 walk；原有 after-walk guard 继续生效。
- regression checks: desktop app-state tests 与真实默认两端验收。
- [ ] implementation and tests accepted

### Chunk 2 — CI runs fixed browser acceptance and retains evidence

- change kind: behavior change (CI configuration)
- strategy: focused CI contract Red-Green plus local execution of the same command
- Red / baseline signal: 当前仓库没有自动运行真实浏览器验收并上传证据的工作流。
- Green check: YAML 语义检查确认 PR/main/manual 入口、固定 suite、失败传播、始终保存 artifact 与固定 action SHA。
- regression checks: release-tools suites；默认工作台验收实际通过。
- [ ] implementation and tests accepted

### Chunk 3 — verify the complete command and publish usage evidence

- change kind: non-behavioral documentation and integrated verification
- strategy: observed command results and proportionate documentation checks
- baseline signal: P1 已观察工作台 12/12、Obsidian 8/8、真实 API 3/3，以及 4005 工作区测试通过（2 个可选语料测试跳过）。
- Green check: 不使用 skip-build 的默认两端命令通过；报告可打开并链接真实证据；README 命令与 --help 一致。
- regression checks: 无新增生产改动时复用 P1 全量回归；新增测试基础设施和 CI 跑相应 Node suites、边界与 diff 检查。
- [ ] implementation and tests accepted

## Phase verification

- 记录实际命令、输出数量与报告位置；明确未运行远程 CI。
- 文献库完整索引、邮件等未新增的 UI 路径在覆盖清单中明示，不声称穷尽功能组合。
- 新 worktree 的代码与文档提交完整，主分支未合并、未推送。

## Abort / reshape triggers

- 等待逻辑若掩盖永久不可用或丢失 after-walk guard，停止并修复契约。
- 若最终检查暴露新的产品缺陷，保留失败证据并针对稳定用户行为补回归，不通过改预期绕过。
