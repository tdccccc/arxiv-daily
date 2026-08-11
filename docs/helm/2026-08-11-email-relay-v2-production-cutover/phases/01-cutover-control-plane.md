# P1 — cutover-control-plane

goal_ref: ../goal.md
updated: 2026-08-11

## Outcome

Relay 以单一 Durable Object 状态机管理可恢复 cutover，能够阻断新 automatic attempt、等待在途请求、保守扫描真实 legacy evidence，并通过幂等 operator API 查询和推进状态。

## Assumptions

- 单用户场景允许 cutover 期间暂停 automatic delivery，安全优先于可用性。
- Cloudflare KV 只作为 legacy evidence/marker 传播层，absence 不作为未投递证明。
- Durable Object 可作为新 Worker 的 cutover phase 权威状态，但不能约束不读取该状态的旧 Worker。
- 撤销旧 Resend 凭据是覆盖所有旧 Worker 与旧在途请求的真实 provider fence；新凭据只在新 Worker 仍保持 automatic locked 时配置。

## Approach

新 Worker 在没有已验证 v2 control state 时默认拒绝 automatic。生产 runbook 先撤销旧 Resend 凭据，再部署使用新凭据但仍锁死 automatic 的新 build；服务端状态机随后扫描真实 legacy evidence、等待 pending TTL 与 KV propagation、封存 marker，再显式激活。所有 action 由稳定 operation ID 去重并可安全重放；等待只设最短 barrier，不设置错过即永久阻塞的上限。扫描在 DO 长事务外执行，以真实开始/完成时钟记录，marker 通过多次、间隔开的只读 observation 确认，而不是一次同位置读取即宣称全局可见。

## Test strategy

- change kind: safety protocol fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 当前 marker 无 writer、旧 key 漏扫、proof observation 可错过且没有 status 恢复。
- Green / regression checks: 重复 action、响应丢失、分页扫描、未知 key、quiesce 竞态、KV visibility 和 pending evidence 都有确定且 fail-closed 的结果。

## Tasks

- [ ] 以 Red 测试锁定 cutover phase、operation idempotency 与 status contract。
- [ ] 实现默认 automatic lock、旧 provider 凭据撤销 attestation 与新 build 激活边界。
- [ ] 扫描两代真实 legacy key，服务端构建、写入并封存 marker。
- [ ] 实现 authenticated GET status 与 action POST，保留安全错误分类。
- [ ] 运行 relay 定向回归并接受 P1 chunk。

## Verification

- `npm --prefix services/email-relay test -- --maxWorkers=1`
- `npm --prefix services/email-relay run typecheck`
- `npm --prefix services/email-relay exec -- wrangler deploy src/index.ts --dry-run --config wrangler.toml`
- `git diff --check`

## Abort / reshape triggers

- 若无法通过撤销旧 provider 凭据覆盖所有旧 Worker 与旧在途 automatic invocation，停止并不提供 activation action。
- 若 KV absence 必须被当作无历史证据，停止并保持 automatic fail closed。
- 若需要在 operator 请求中携带 marker/邮箱/token，停止并改为服务端构建。
