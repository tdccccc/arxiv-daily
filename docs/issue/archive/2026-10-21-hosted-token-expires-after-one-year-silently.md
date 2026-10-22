# 官方代发的验证码（device token）签发满一年就失效，活跃用户也一样；失效后日报邮件悄悄停发，界面没有任何提示

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 MAIL-04
> 范围：`services/email-relay` device token 生命周期；官方代发（Official delivery）自动邮件的失败可见性 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch3`，提交 aef2a15、808fa82） · 置信度：代码推断

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

## 修复情况

官方代发的验证码（device token）现在随使用续期：已认证的 `/v1/deliver` 请求发现 KV 记录距写入或上次续期已满半个寿命（约 182 天）时，用同样内容加 `renewedAt` 重写并刷新一年 TTL（每个活跃设备一年约两次 KV 写入；写入失败不影响这次投递；`createdAt` 与绑定字段原样保留，因为 issuance claim 要比对它们；旧版 `device:` 记录按各自的 key 续期）。所以只有整整一年没被使用的设备才会过期，与代码注释一致。**这只在用户重新部署中继之后生效**：部署后第一次认证成功的请求会把已经超过半个寿命的既有 token 一起续上，因此部署要赶在最早的 token 满一年（2027-07 前后）之前。

没有按写作建议加 `token_expired` 墓碑错误码：KV 过期是静默删除，中继分不出“过期”和“从未存在”，墓碑要额外写入并带来枚举风险；客户端把 401 统一报告为 `hosted_token_rejected`（MAIL-05），文字是“验证码无效或已过期，请重新发送验证邮件并粘贴新的验证码”，已经覆盖用户要做的事。失败可见性：插件的自动邮件通知和测试邮件提示带这句话，日志写 `status=401`，CLI 的 `email test` 显示它；用户文档补了有效期与重新验证的说明。没有做“设置页显示最近一次投递结果”和“`email status` 显示最近结果”（需要读投递状态文件，是另一处设计），留作后续。

测试：`services/email-relay/tests/kv-idemp.test.ts` 新增 `device token lifetime`（200 天前的设备被续期且字段保留、不到半寿命不写 KV、续期后又过半寿命再续、写失败不影响投递、旧 `device:` 记录、未知 token 不写 KV、损坏的 `renewedAt`、续期后的设备仍被 issuance claim 认可，因此重放验证链接得到同一个 token）。整套中继测试在时钟拨快 400 天后也全部通过：夹具带真实时间的 `renewedAt`，不会随日历推进而失败。续期的 7 个用例在旧代码下失败。
