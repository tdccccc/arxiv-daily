# 测试邮件和自动邮件失败时，令牌错误、收件人不符、额度用完、域名未验证都只显示同一句 `provider_definitive_rejection`

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 MAIL-05
> 范围：邮件投递结果的用户提示（插件“发送测试邮件”、CLI `email test`、自动邮件日志；`self` 与 `hosted` 两种模式） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-21-batch3`，提交 808fa82） · 置信度：已复现

## 问题

“发送测试邮件”是用户确认邮件配置是否可用的唯一入口。失败时用户看到的是：

- 插件：Notice 文字 `failed: provider_definitive_rejection`（`plugin/src/services/email-delivery.ts:16` 抛出 `${result.kind}: ${result.reason}`）；
- CLI：`email test: failed (provider_definitive_rejection)`（`apps/cli/src/email-cmd.ts:78`）。

无论真正原因是什么，都是这同一句：

- 官方代发：令牌错误或已失效（401）、收件人与已验证邮箱不符（403）、当天额度用完（429）、中继拒绝内容（422）、请求格式被拒（400）；
- 自己发送：Resend API key 错误（401）、域名未验证或 `onboarding@resend.dev` 只能发给账户自己的邮箱（403）、收件地址无效（422）、被限流或额度用完（429）。

已复现（一次性测试，已删除）：`hosted` 模式下依次让中继返回 401、403、429、422、400，`sendSettingsTestEmail` 的结果全部是 `{"kind":"failed","reason":"provider_definitive_rejection","attempts":1}`。

这不是漏掉了文案：客户端其实写好了针对性的提示——`packages/core/src/delivery/hosted.ts:157-166` 为 401（验证码无效或过期，请重新发送验证邮件）、403（收件人与已验证邮箱不符）、429（当天发送上限，明天再试）、422 分别拼好了说明文字。但 `deliverDailyEmailIfEnabled` 的 catch 分支（`deliver-email.ts:334-363`）只取状态码做分类，只返回/记录枚举原因，`HostedDeliveryError.message` 里的提示从未被读取，用户永远看不到。设计上要求结果“无 PII、固定枚举”这一点本身没有问题，问题是枚举太粗：这些提示本来就是固定的静态文字，完全可以用更细的固定枚举表达。

后果：新用户配置官方代发时，测试邮件失败几乎无法自助排查；自动邮件失败时日志里也只有这一句（日志中看不到 HTTP 状态）。

## 原因

- `packages/core/src/delivery/deliver-email.ts:334-363`：所有明确拒绝统一映射为 `provider_definitive_rejection`；`:388-403` `providerOutcomeAmbiguous` 只做“明确/不明”二分。
- `packages/core/src/delivery/hosted.ts:154-174`：提示文字只存在于异常消息里，调用方不消费。
- `packages/core/src/delivery/resend.ts:106-123`：Resend 的 400/401/403/404/422/429 同样只留下 `ResendSendError.status`，没有传给结果。
- `packages/core/src/delivery/types.ts:93-111`：`EmailDeliveryReason` 里没有区分这些情况的原因码。

## 可能的解决方案

推荐：在 `EmailDeliveryReason` 里增加少量固定、无 PII 的细分原因，只用于返回值和日志，不改变已持久化的 `DeliveryResultErrorCode`（claim 里仍然写 `provider_definitive_rejection`，保持与旧状态文件兼容）：

- `hosted_token_rejected`（relay 401）、`hosted_recipient_mismatch`（relay 403）、`provider_quota_exceeded`（429）、`provider_request_rejected`（400/422）、`resend_api_key_rejected`（Resend 401）、`resend_sender_not_allowed`（Resend 403）；
- 插件与 CLI 把这些原因码映射成一句可执行的文字（例如“验证码无效或已过期，请在设置里重新验证”），日志里同时记下原因码。

备选：不新增枚举，只让测试邮件路径（`sendSettingsTestEmail`）把 HTTP 状态码放进返回值（例如 `reasonDetail: "http_401"`），因为状态码本身不含 PII。

## 修复后应补的测试

- `hosted`：401/403/429/422/400 各自得到不同的返回原因；自动投递的状态文件里仍然只有既有的 `provider_definitive_rejection`。
- `self`：Resend 401/403/422/429 各自得到不同的返回原因。
- 插件 `testEmailResultMessage` 与 CLI `emailTest` 对每种原因输出可执行的提示，不包含令牌、邮箱、响应正文（保持现有“不泄露敏感内容”的测试）。

## 修复情况

测试邮件和自动邮件失败时，原因不再都是 `provider_definitive_rejection`：返回值和日志现在区分 `hosted_token_rejected`（中继 401）、`hosted_recipient_mismatch`（403）、`provider_quota_exceeded`（429）、`provider_request_rejected`（400/422）、`resend_api_key_rejected`（Resend 401）、`resend_access_denied`（Resend 403），其余状态仍是 `provider_definitive_rejection`；日志同时写 `status=<HTTP 状态>`。claim 文件和 `delivery-state.json` 仍然只记录 `provider_definitive_rejection`，旧客户端读到的文件不变，之后同一天的运行仍得到 `skipped / provider_definitive_rejection`。core 新增 `emailFailureHint`，把这些原因翻成一句固定文字（不含令牌、地址、响应正文）；插件的测试邮件提示和自动邮件通知、CLI 的 `email test` 都会显示它（CLI 只改了 `apps/cli/src/email-cmd.ts` 里的一行）。

偏离写作的一点：Resend 的 403 命名为 `resend_access_denied` 而不是 `resend_sender_not_allowed`，因为 Resend 对无效 API key 也返回 403，提示文字同时覆盖 key、发件域名和 onboarding 发件人只能发给账户邮箱三种原因。不需要部署中继。

测试：`packages/core/tests/delivery/hosted-deliver.test.ts` 与 `resend.test.ts`（各状态码对应的原因、claim 文件里只有粗粒度代码、日志不含响应正文；这 12 个用例在旧代码下失败）、`plugin/tests/email-delivery.test.ts`（提示文字与错误码、令牌被拒时的自动邮件通知）、`apps/cli/tests/cli-email.test.ts`（`email test` 输出提示且不含令牌）。
