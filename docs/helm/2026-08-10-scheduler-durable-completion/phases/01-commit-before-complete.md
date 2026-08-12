# P1 — commit-before-complete

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

Scheduler 的完成结果、用户反馈、history 和 post-completion hook 都位于已确认 run-state commit 之后。

## Assumptions

- 现有 `failed_transient` 足以表达 Scheduler commit failure。
- StateStore 的 `loadFn` 可以在保存异常后判定 durable candidate 是否实际落盘。
- Run history 的 `safeAppend` 继续吞掉错误，不属于硬完成事务。

## Approach

先反转当前 integrity 测试，再将 StateStore mutation 改为候选快照提交；Scheduler 对 completed commit 单独分类，失败时返回安全 transient 结果并保留 batch 可继续性。

## Test strategy

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 修改 integrity 测试要求 disk-full completion 返回 transient，并新增“保存前不可见候选状态”等测试，确认当前实现仍返回 completed或泄露内存状态。
- Green / regression checks: Scheduler、StateStore、history 和 CLI 定向测试通过，所有既有 PipelineResult 消费方无需新增 kind，完整 release-equivalent gate 通过。

## Tasks

- [x] 以 Red 测试锁定 completion commit failure、callback/progress/history 边界和 batch 继续行为。
- [x] 将 StateStore mutation 改为保存后发布，并实现保存异常后的 durable readback 判定。
- [x] 让 Scheduler completion commit failure 映射为明确 `failed_transient`，禁止完成副作用。
- [x] 补充重试、history best-effort 和 CLI 结果回归测试。
- [x] 运行定向、完整回归和独立代码复核。
- [x] 完成独立技术报告交接、提交和 Helm 收口。

## Verification

- `npm test --workspace @arxiv-daily/core -- --maxWorkers=1 tests/scheduler-driver-integrity.test.ts tests/scheduler.test.ts tests/state-store.test.ts tests/run-history.test.ts tests/scheduling/history-recorder.test.ts tests/run-format.test.ts`
- `npm test --workspace arxiv-daily -- --maxWorkers=1 tests/cli-main.test.ts tests/cli-runtime.test.ts`
- `npm run check:boundaries && npm run lint && npm run typecheck`
- `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1`
- `npm run build && npm run smoke:build && git diff --check`

## Abort / reshape triggers

- 如果必须新增公开结果 kind 才能保持正确性，停止并执行 L2 reshape。
- 如果 history 被证明必须成为硬完成边界，停止并设计 journal/outbox 事务，不以调用顺序假装原子性。
- 如果 StateStore 无法通过 readback 区分已提交与未提交，采用保守失败，不放宽完成语义。
