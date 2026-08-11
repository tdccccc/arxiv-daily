# P2 — readiness-token-and-recipient-ledger

goal_ref: ../goal.md
updated: 2026-08-11

## Outcome

Relay 的公开 readiness、v2 token 签发和 automatic delivery 使用同一个 authoritative ready generation；同一 verified recipient 的多个 ready 后 token 共享 automatic ledger、quota 与 provider identity。

## Assumptions

- P1 的 v3 control Durable Object 是 automatic readiness 的唯一权威状态，KV audit marker 只用于恢复与审计。
- Verification magic link 可以在未 ready 时保留 pending，不应先消费后失败。
- Test delivery 仍需按 device 与每次随机 logical key 隔离，不与 automatic identity 合并。

## Approach

先用跨 token 并发、ready 前验证和 readiness contract 测试锁定行为。公开 readiness 只暴露有界 protocol/build/phase 信息；verify complete 在读取但未消费 pending 前确认 ready generation，再签发带该 generation 的 token。Automatic 请求按 recipient identity 路由，ledger/provider identity 与 fingerprint 都移除 device identity；token 仍用于认证、收件人绑定和审计字段，test 请求继续路由到 device object。

## Test strategy

- change kind: safety protocol completion
- strategy: strict Red-Green-Refactor
- Red / baseline signal: ready 前 verification 会消费 pending 并签发不可用 token；同一邮箱两个 token 进入不同 DO/provider key；`/health` 无 readiness generation。
- Green / regression checks: ready 前 token 数为零且 pending 保留；ready 后 token 绑定 ready generation；两个 token 同日并发只有一次 automatic provider 请求，test 仍彼此独立；readiness/build mismatch fail closed。

## Tasks

- [ ] 增加公开、无 secret/PII 的 readiness contract 与 build/protocol identity。
- [ ] 让 verification completion 在消费 pending 前通过 authoritative ready generation gate。
- [ ] 将 automatic DO、logical key、fingerprint、quota/ledger 改为 recipient scope，保留 test device scope。
- [ ] 验证 legacy token、pre-ready token、generation mismatch 和多 token 竞态全部 fail closed。
- [ ] 运行 relay 定向/完整回归、typecheck 与 Wrangler dry-run，接受 P2 chunk。

## Verification

- `npm --prefix services/email-relay test -- --maxWorkers=1`
- `npm --prefix services/email-relay run typecheck`
- `npm --prefix services/email-relay exec -- wrangler deploy src/index.ts --dry-run --config wrangler.toml`
- `git diff --check`

## Abort / reshape triggers

- 若 automatic 仍可从 device identity 派生 provider key 或进入 device-scoped ledger，停止并不声明 recipient 级幂等。
- 若 verify complete 必须消费 pending 后才能读取 readiness，先拆分 pending peek/consume，不接受不可重试 token 丢失。
- 若 build identity 只能由客户端声明，停止并改为构建/部署配置注入的服务端值。
