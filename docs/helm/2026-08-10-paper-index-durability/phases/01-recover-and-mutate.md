# P1 — recover-and-mutate

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

Paper Index 始终从最新有效副本读取，保存失败不破坏恢复点，交错领域修改不会被陈旧完整快照覆盖。

## Assumptions

- 现有 checkpoint store 的 classified read 和 recovery-content 模式可以在 host-neutral Core 中复用。
- 同 realm 的 path-keyed queue 足以覆盖本事项声明的 mutation 保证。
- 保留一个最近有效 backup 不会破坏 export/import 或 legacy 兼容。

## Approach

先以失败注入测试锁定 backup 和陈旧快照缺口，再将读取、内部保存与 queued mutation 组合为完整事务；领域调用方通过专用 mutation 更新，不再公开提交旧快照。

## Test strategy

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: backup-only、corrupt-primary/valid-backup、双坏副本、提升失败和 selection/history 交错测试在当前实现上失败或丢失修改。
- Green / regression checks: Paper Index、daily selection、Dashboard history/commands 测试通过，既有 schema/legacy 测试不回归，并通过完整 release-equivalent gate。

## Tasks

- [x] 以 Red 测试锁定分类读取、backup-only 恢复和不可读 fail-closed。
- [x] 实现 primary/backup/legacy 的验证读取和可恢复替换流程。
- [x] 收紧裸 `save`，让所有领域读改写在队列内重新读取并保存。
- [x] 迁移 daily selection、history sync 与 diagnostics 并增加交错测试。
- [x] 运行定向、故障注入和完整回归。
- [x] 完成独立技术报告交接、提交和 Helm 收口。

## Verification

- `npm test --workspace @arxiv-daily/core -- --maxWorkers=1 tests/paper-index.test.ts tests/daily-selection.test.ts`
- `npm test --workspace obsidian-arxiv-daily -- --maxWorkers=1 tests/dashboard-history-sync.test.ts tests/commands.test.ts`
- `npm run check:boundaries && npm run lint && npm run typecheck`
- `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1`
- `npm run build && npm run smoke:build && git diff --check`

## Abort / reshape triggers

- 如果修复要求 Paper Index schema 迁移，停止并执行 L2 reshape。
- 如果需要修改通用 Storage contract 才能满足本事项声明的同 realm 保证，先证明现有 queue 不足并重新规划。
- 如果公共 `save` 有无法迁移的外部 API 消费者，停止并选择兼容的 queued replacement contract。
