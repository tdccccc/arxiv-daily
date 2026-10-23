# 官方代发的验证码（device token）签发满一年就失效，活跃用户也一样；失效后日报邮件悄悄停发，界面没有任何提示

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 MAIL-04
> 范围：`services/email-relay` device token 生命周期；官方代发（Official delivery）自动邮件的失败可见性 · 严重程度：中 · 状态：未修复 · 置信度：代码推断

## 问题

用户完成邮箱验证后，中继把 device token 的记录写进 Workers KV，并设置 `expirationTtl` 为 365 天。这个过期时间从**签发那一刻**算起，使用时从不续期：`authenticateDevice` 只读取，不会重写记录；`putDevice` 只在签发时被调用一次。所以不管用户每天都在正常收邮件，签发满一年的那一天，这个 token 就会在 KV 里消失。代码里的注释写的是“Approximately one year; expiry revokes inactive devices”（过期只会回收不活跃的设备），与实际行为不符。

过期之后发生的事：

1. 客户端带着旧 token 调用 `/v1/deliver`，中继返回 `401 {"error":"invalid or revoked token"}`。
2. 客户端把 401 记为明确拒绝（`provider_definitive_rejection`），这一天的邮件被标记为已失败并阻断；之后每一天都是同样的结果。
3. 日报文件继续生成，只有日志里有一行 `email: provider_definitive_rejection date=…`。插件界面不弹任何提示（只有“存储不支持自动邮件”会弹，`plugin/main.ts:2890`），CLI 的 `run`/定时任务也丢弃结果（`apps/cli/src/runtime.ts:194`）。设置页里的“发送测试邮件”会得到 `failed: provider_definitive_rejection`，看不出是 token 过期（见 MAIL-05）。
4. 客户端里写好的提示“Verification code is invalid or expired. Send a new verification email…”（`hosted.ts:159-161`）永远不会显示给用户。

用户的实际体验：某一天起，邮件再也不来了，没有错误提示；每年所有官方代发用户都会在各自签发的周年日遇到这件事。现有的 Beta 用户在 2026-07/08 完成验证，最早的 token 将在 2027-07 前后失效（旧版中继签发的 `device:` 记录同样是 365 天）。

README 和用户文档都没有提到 token 有有效期。

## 原因

- `services/email-relay/src/kv.ts:194-197` `putDevice`：写入时设置固定的 `expirationTtl: 60 * 60 * 24 * 365`，注释与行为不符。
- `services/email-relay/src/kv.ts:200-225` `authenticateDevice`：认证成功后没有续期。
- `services/email-relay/src/deliver-gate.ts`：成功投递后没有任何刷新设备记录的动作。
- `packages/core/src/delivery/deliver-email.ts:299-363` 与 `plugin/main.ts:2882-2890`、`apps/cli/src/runtime.ts:194`：失败只写日志，不向用户暴露。

## 可能的解决方案

推荐（二选一或组合）：

1. 认证成功后滑动续期：在 `authenticateDevice` 或成功投递后，如果记录的剩余寿命低于一半（KV 的 `getWithMetadata` 或在记录里自存 `issuedAt`/`renewedAt`），重新 `put` 同一条记录并刷新 TTL。KV 单键写入频率限制是每秒一次，这里每天最多一次，没有问题。这样只有真正闲置超过一年的设备才会被回收，与注释一致。
2. 在中继的 `401` 之外，让客户端能分辨“token 过期”和“token 从未存在”（例如在 token 里或记录里保留一个墓碑/到期时间，到期后返回稳定的错误码 `token_expired`），客户端据此提示“验证已过期，请重新验证”。
3. 失败要让用户看见（与 MAIL-02 的解决方案一致）：插件在自动邮件失败时弹一次通知并在设置页显示最近一次投递结果；CLI 在 `email status` 里显示最近一次结果；`hosted` 401 时给出明确的“重新验证”提示。
4. 在 README/用户文档中写明 token 有效期和续期方式。

## 修复后应补的测试

- 中继：一个签发 200 天前的 token 在成功认证后，KV 记录的 TTL 被刷新；签发不到一半寿命的 token 不重复写入。
- 中继：过期（记录不存在）与格式错误的 token 返回可区分的错误码，但不泄露内部细节，也不允许枚举。
- 客户端：401 对应的结果带有稳定的可展示原因（例如 `hosted_token_rejected`），插件通知和 CLI `email status` 能显示它。
- 回归：KV 写入失败不影响本次投递结果。
