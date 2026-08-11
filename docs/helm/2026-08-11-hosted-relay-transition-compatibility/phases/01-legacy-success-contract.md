# P1 — legacy-success-contract

goal_ref: ../goal.md
updated: 2026-08-11

## Outcome

Hosted 客户端可在过渡期安全接受当前生产 Worker 的有限成功体，同时保持严格响应验证和敏感 provider ID 丢弃。

## Assumptions

- 当前生产旧 Worker 的成功体只存在 `{ok:true,id}` 与 `{ok:true,id,deduped:true}` 两种形状。
- Provider ID 不参与客户端幂等状态，只需验证类型和长度后丢弃。

## Approach

先以测试锁定允许和拒绝矩阵，再把成功体判定收敛到一个纯验证函数；只接受精确字段集合、非空有界 ID 和字面量 `deduped: true`。

## Test strategy

- change kind: compatibility bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 当前客户端把旧 Worker 成功体判为 ambiguous。
- Green / regression checks: 允许的旧响应返回 delivered；所有其他变体保持 ambiguous，第二次 automatic 调用不再请求 relay，provider ID 不进入任何持久或可见结果。

## Tasks

- [ ] 以 Red 测试锁定旧响应白名单及拒绝矩阵。
- [ ] 实现有界、严格且不返回 provider ID 的成功体验证。
- [ ] 运行 Core 定向测试与完整回归。
- [ ] 完成技术报告交接、实现提交与 Helm 收口。

## Verification

- `npm test --workspace @arxiv-daily/core -- --maxWorkers=1 tests/delivery/hosted-deliver.test.ts`
- `npm run check:boundaries && npm run lint && npm run typecheck`
- `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1`
- `npm run build && npm run smoke:build && git diff --check`

## Abort / reshape triggers

- 若生产旧 Worker 还存在其他无法严格界定的成功体，停止并先核对部署版本。
- 若兼容需要记录或返回 provider ID，停止并保持 fail closed。
