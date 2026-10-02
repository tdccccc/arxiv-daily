# P2 — relay-json-validation

goal_ref: ../goal.md
created: 2026-09-28T22:33:38+08:00
updated: 2026-09-28T22:44:52+08:00
revision: 2

## Outcome

Relay 的公开 JSON 请求入口对 null、数组和标量返回 400，不因类型断言抛出 500，也不把这些请求传入投递或控制流程。

## Assumptions

- 验证发起、投递、已认证的 cutover 操作都要求 JSON 对象；原有认证顺序保持不变。
- 请求级对象形状校验与字段级验证分开；合法对象继续由既有逻辑验证。

## Approach

增加统一的 JSON 对象解析函数，复用现有 invalid JSON/body 错误分支；覆盖三个公开请求入口，并在直接接收请求的 DeliverGate 边界使用同一解析规则。不改内部已构造的控制消息协议。

## Chunks

### Chunk 1 — 非对象 JSON 在请求边界被拒绝（F32）

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: relay tests 的 verify/start、已认证 deliver、已认证 cutover 收到 null/数组/字符串/数字/布尔时须返回 400 且无 provider 调用/状态写入；原路径 null 抛出 500，部分其它形状进入较深层字段校验。
- Green check: `npm test --prefix services/email-relay`
- regression checks: relay typecheck 与全部 relay 测试；合法请求的认证、配额、幂等性和控制状态机现有测试保持通过。
- [x] implementation and tests accepted (12 observed Reds; 20 focused Greens; full relay 161 tests and typecheck passed)

## Phase verification

- 请求入口测试使用内存 KV/DO 与 provider spy，禁止真实邮件调用。
- 观察完整 relay suite 和独立 typecheck，通过后提交并进入 F19。

## Abort / reshape triggers

- 若改动影响认证优先级或合法 payload 语义，缩小至对象形状校验。
