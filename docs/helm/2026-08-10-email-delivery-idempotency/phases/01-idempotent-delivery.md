# P1 — idempotent-delivery

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

同一自动日报的并发与模糊失败只能留下一个可追踪投递决策，后续自动调用不会重复触发 provider。

## Assumptions

- Storage adapters 能提供“目标已存在即失败”的 exclusive create，而无需引入仓库级数据库。
- Resend 的 `Idempotency-Key` 可承载有界、稳定、无 PII 的逻辑投递 key。
- 结果不明时优先阻断自动重试比可能漏发更符合本事项的安全目标。

## Approach

以持久 delivery claim 串行化 Vault 内的逻辑投递，在 provider 边界复用稳定 key；严格区分 provider outcome 与本地 bookkeeping outcome，并让 Core 与 hosted relay 使用一致的阻断语义。

## Test strategy

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增 Core 双并发、损坏状态、provider-success/state-save-failure、模糊传输和 relay provider-key 测试，确认当前实现出现重复请求、fail-open 或错误 `failed` 结果。
- Green / regression checks: Core delivery、Node/Obsidian adapter、Plugin/CLI 调用方、relay tests/typecheck/Wrangler dry-run及完整 release-equivalent gate 全部通过。

## Tasks

- [ ] 以 Red 测试锁定客户端并发、严格状态读取和 provider/bookkeeping 结果边界。
- [ ] 为 Storage adapter 与 delivery state 实现 exclusive claim、严格读取和 v1 兼容阻断记录。
- [ ] 为 BYOK Resend 和 hosted 客户端实现稳定正式 key 与独立测试 key。
- [ ] 以 Red/Green 测试收紧 relay 的 Durable Object、Resend key、模糊失败和成功后记账顺序。
- [ ] 更新所有结果 union 消费方并运行定向与完整回归。
- [ ] 完成独立技术报告交接、提交和 Helm 收口。

## Verification

- `npm test --workspace @arxiv-daily/core -- --maxWorkers=1 tests/delivery/delivery-state.test.ts tests/delivery/resend.test.ts tests/delivery/hosted-deliver.test.ts`
- `npm test --workspace @arxiv-daily/node-runtime -- --maxWorkers=1 tests/node-adapters.test.ts`
- `npm test --workspace obsidian-arxiv-daily -- --maxWorkers=1 tests/obsidian-adapters.test.ts`
- `npm --prefix services/email-relay run typecheck && npm --prefix services/email-relay test -- --maxWorkers=1`
- `npm --prefix services/email-relay exec -- wrangler deploy --dry-run`
- `npm run check:boundaries && npm run lint && npm run typecheck && NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1 && npm run build && npm run smoke:build`

## Abort / reshape triggers

- 如果需要持久化完整 digest 并建立自动重试 outbox，停止并执行 L2 reshape。
- 如果 Storage host 无法提供真实 exclusive create，停止并重新界定跨进程保证，不以进程内锁替代。
- 如果新状态必须让旧客户端 fail open，停止并设计安全迁移后再继续。
