# Email Delivery Idempotency

status: active
updated: 2026-08-10
owner: zcode-main-session

## Intent

让同一日期和收件人的自动日报邮件在并发调用、provider 响应不明及本地记账失败时保持 fail closed，避免 Plugin、CLI 或 hosted relay 自动重复投递。

## Success criteria

- [ ] 同一 `date + normalized recipient` 的并发自动投递最多产生一次 provider 请求。
- [ ] BYOK 与 hosted relay 都使用稳定 provider idempotency key，测试发送不复用正式日报 key。
- [ ] delivery state 损坏或不可读时不发信，provider 成功但状态未记录与传输结果不明均有阻断重复投递的明确结果。
- [ ] 既有 v1 delivered/failed 记录保持兼容，旧客户端看到新阻断记录时不会自动重发。
- [ ] Core、Host、relay 定向测试和仓库 release-equivalent 门禁通过，技术报告同步。

## Non-goals

- 不建立邮件 outbox、后台重试守护进程或严格分布式 exactly-once 声明。
- 不把邮件结果改写为 Pipeline 或 Scheduler 运行失败。
- 不执行真实邮件投递。

## Constraints

- 支持自动投递的宿主必须提供操作系统级跨进程 exclusive claim；无法提供该能力的 Obsidian 宿主必须 fail closed。
- 不记录 API key、token、原始邮件正文或未哈希收件人到 provider idempotency key。
- 不恢复已经删除的 CLI 配置方式，也不扩大到 Scheduler durable completion。

## Phases

1. P1 — 以通用 DataAdapter copy 实现跨进程 claim — status: superseded
2. P1b — 自动邮件在具备真实 OS claim 的客户端与租户隔离 relay 边界获得可验证防护 — status: active
