# P1b — idempotent-delivery

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

同一自动日报在具备真实 OS exclusive-create 的客户端与按认证设备串行化的 relay 中只能形成一个 provider 投递决策；能力不足、状态不明或恢复不安全时均 fail closed。

## Assumptions

- Obsidian 桌面端 `FileSystemAdapter.getBasePath()` 可将受校验的 Vault 相对路径交给 Node `fs.open(..., "wx")`；非桌面 DataAdapter 不具备可证明的原子创建能力。
- Resend 稳定 `Idempotency-Key` 能在本地进程崩溃和内部重试窗口中提供最终 provider 去重边界。
- Relay 可按认证设备而非客户端 key 选择 Durable Object，使配额和所有该设备投递决策在一个串行边界内执行。
- 结果不明时阻断自动重试优先于可能漏发。

## Approach

保留已验证的严格状态读取、稳定 provider key、结果 union 和 fail-closed binding；替换通用 `DataAdapter.copy` claim、按裸 key 分片的 relay DO、非原子 KV quota 和过宽 HTTP 重试分类。桌面 Plugin 与 CLI 使用操作系统级 exclusive create，能力不足的 Obsidian 宿主拒绝自动投递。Relay 以认证设备作用域串行配额和幂等 ledger，并用请求指纹校验 replay；Provider 明确拒绝、模糊结果和未尝试取消分别处理。对锁与 claim 增加有界恢复协议，恢复操作不得重新开放已开始的模糊投递。

## Test strategy

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增桌面 OS claim/非桌面 fail-closed、孤儿锁与 claim 恢复、HTTP 400/408/409/429/5xx 分类、pre-attempt cancellation、跨租户 key、同设备不同 key 配额并发、请求指纹与 ledger retention 测试，确认当前候选会重复、冲突、永久阻断或错误释放 claim。
- Green / regression checks: Core delivery、Node/Obsidian adapter、Plugin/CLI、relay tests/typecheck/Wrangler dry-run及完整 release-equivalent gate 全部通过；不执行真实邮件投递。

## Tasks

- [ ] 用 Red 测试替换对通用 DataAdapter copy 原子性的错误假设，并锁定桌面 OS claim、非桌面 fail-closed 与根路径/符号链接边界。
- [ ] 为本地状态锁与 claim 实现可验证的有界恢复；只有可证明未开始 provider 调用的孤儿 claim 可释放，模糊或已开始记录继续阻断。
- [ ] 收紧 Core 与 relay 的 provider 结果分类、内部重试和 pre-attempt cancellation，保持稳定 key 且不将 408/409/5xx 当明确未发送。
- [ ] 将 relay DO 改为认证设备作用域，在同一串行边界内执行幂等、请求指纹和配额预占/结算；阻止跨租户抢占和不同 key 并发绕过。
- [ ] 定义并测试 relay ledger 的 pending/done 保留与清理策略；发布采用单版本 cutover，不宣称 eventual-consistent KV 可提供安全并存/回滚。
- [ ] 更新所有结果消费者并运行定向、独立复核和完整回归。
- [ ] 完成独立技术报告交接、提交和 Helm 收口。

## Verification

- `npm test --workspace @arxiv-daily/core -- --maxWorkers=1 tests/delivery/delivery-state.test.ts tests/delivery/resend.test.ts tests/delivery/hosted-deliver.test.ts`
- `npm test --workspace @arxiv-daily/node-runtime -- --maxWorkers=1 tests/node-adapters.test.ts`
- `npm test --workspace obsidian-arxiv-daily -- --maxWorkers=1 tests/obsidian-adapters.test.ts`
- `npm --prefix services/email-relay run typecheck && npm --prefix services/email-relay test -- --maxWorkers=1`
- `npm --prefix services/email-relay exec -- wrangler deploy --dry-run`
- `npm run check:boundaries && npm run lint && npm run typecheck && NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1 && npm run build && npm run smoke:build`

## Abort / reshape triggers

- 如果桌面 Vault 真实路径无法安全限定或使用内核级 exclusive create，停止并禁用该宿主自动投递，不退回 DataAdapter check-then-write/copy。
- 如果 relay 无法在一个认证设备 DO 内串行配额与幂等 ledger，停止并重新界定 hosted delivery 服务边界。
- 如果需要持久化完整 digest、建立自动 outbox 或把邮件结果纳入 Scheduler 硬事务，再执行 L2 reshape。
- 如果安全迁移需要旧 Worker 与新 Worker 双写并存，停止发布；本 phase 只接受可控单版本 cutover，不以 Workers KV 最终一致性冒充 CAS。
