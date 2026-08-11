# Email Relay V2 Production Cutover

status: active
updated: 2026-08-11
owner: zcode-main-session

## Intent

补齐 relay v2 的可恢复 cutover 控制面、真实 legacy evidence、recipient 级 automatic 幂等和安全 token/readiness gate，并提供带人工确认点的半自动生产迁移工具。

## Success criteria

- [ ] Durable cutover 状态机支持幂等 mutation、只读 status、响应丢失恢复和无上限 observation 等待。
- [ ] Automatic delivery 具备真实 quiesce/drain 边界，并保守导入两代 legacy KV evidence。
- [ ] Proof ready 前不签发 v2 token；readiness 明确报告 build/protocol/automatic 状态。
- [ ] 同一收件人、日期在多个 v2 token 下共享 automatic ledger 和 provider key。
- [ ] Operator 工具完成安全 preflight、dry-run、显式确认、私有 journal 和断点恢复，CI 保持 no-deploy。
- [ ] Relay/Core 定向测试、独立产品 workflow invariants、完整门禁与技术报告同步通过。

## Non-goals

- 不在实现阶段部署生产 Worker、修改生产 KV、读取生产 secret、发布客户端或发送真实邮件。
- 不建立邮件 outbox、自动重试 daemon、mixed-version rollout、自动 rollback 或 strict distributed exactly-once 声明。

## Constraints

- 首次部署的新 Worker 必须从 automatic-locked 状态启动；旧 Worker 的真实 provider fence 由 operator 先撤销旧 Resend 凭据建立，新凭据只交给仍锁死 automatic 的新 Worker。
- 不以新 Worker 内的 lease、KV marker 或部署完成时间声称旧 Worker 已 quiesce；撤销旧 provider 凭据和确认新 build 均是后续生产 runbook 的独立人工检查点。
- 模糊 provider/operator 结果必须 fail closed，并通过 status/fix-forward 恢复，不能盲目重试。
- Secret 不进入 argv、日志、journal、PR 或人工确认短语；不持久化邮箱、token、KV 原值或邮件正文。
- 真实部署、provider 凭据撤销/更新、生产 mutation、activation 与真实邮件分别需要后续明确授权。

## Phases

1. P1 — 可恢复 cutover 状态机与 legacy/provider-fence proof — status: done
2. P2 — Token/readiness gate 与 recipient 级幂等 — status: active
3. P3 — 半自动 operator 工具、runbook 与治理门禁 — status: pending
