# P1 — fail-closed

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

论文过滤的非法模型响应会以可重试失败终止当前流水线，严格合法的空分类仍保持现有完成语义。

## Assumptions

- `decodePaperFilterRecords` 的严格契约已经正确，无需修改合法结果形状或版本。
- 现有 `failed_transient`、Scheduler 退避、CLI 退出码和 Plugin 通知足以承载响应验证失败。
- 专用错误比改变 `filterPapers` 成功返回类型更适合本次窄修复。

## Approach

复用结构化摘要的稳定错误码与结构 guard 模式，在过滤边界抛出专用响应验证错误；Pipeline 显式将其映射为 transient failure，并用单元和轻量 Pipeline 测试锁定无下游副作用与合法零结果兼容性。

## Tasks

- [ ] 新增过滤响应验证错误类型、稳定错误码和结构 guard。
- [ ] 将非法 JSON 与非法过滤契约改为抛出专用错误。
- [ ] 在 Pipeline 中显式映射为可诊断的 `failed_transient`。
- [ ] 更新 filter 单元测试并保留合法空/skip 行为。
- [ ] 增加 Pipeline 回归测试，确认非法响应无下游 mutation。
- [ ] 运行定向、类型、边界、全量测试与构建验证。
- [ ] 完成技术报告交接、提交与阶段收口。

## Verification

- `npm test --workspace @arxiv-daily/core -- --maxWorkers=1 tests/paper-filter.test.ts`
- `npm test --workspace @arxiv-daily/core -- --maxWorkers=1 tests/pipeline/pipeline-error-handling.test.ts`
- `npm test --workspace @arxiv-daily/core -- --maxWorkers=1 tests/daily-filter-checkpoint-store.test.ts`
- `npm run typecheck --workspace @arxiv-daily/core`
- `npm run check:boundaries`
- `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1`
- `npm run typecheck && npm run build`

## Abort / reshape triggers

- 如果合法空数组无法与非法响应在现有 decoder 边界稳定区分，停止并重审过滤 result contract。
- 如果实现需要新增 `PipelineResult` kind 或修改 checkpoint schema，停止并执行 L2 reshape，而不扩大本阶段。
- 如果验证显示 Scheduler 或 Host 不能正确消费 `failed_transient`，先报告并决定是否拆出依赖阶段。
