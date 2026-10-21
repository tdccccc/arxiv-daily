# 邮件在“确定没发出去”时（断网、DNS 失败、中继维护中）也被记成“结果不明”，那一天的邮件永远不会再发

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 MAIL-02
> 范围：自动日报邮件的结果分类（`self` 与 `hosted` 两种模式，CLI / DSH 工作台 / 插件） · 严重程度：中 · 状态：未修复 · 置信度：已复现

## 问题

自动邮件按“日期 + 收件人”只允许一次 provider 调用。只要结果被记成 `ambiguous`（结果不明），这一天就永久阻断，之后的运行只会得到 `skipped / provider_outcome_ambiguous`，没有重试，也没有任何界面或命令能解除。这个取舍对“请求可能已经到达服务商”的情况是合理的。但下面两类情况，客户端**已经掌握了请求根本没有送达**的证据，却仍然按“结果不明”处理，用户因此无故丢掉当天的邮件：

1. **请求发出之前就失败的网络错误。** Node 宿主的 HTTP 客户端把“连接阶段失败”（`ENOTFOUND`、`EAI_AGAIN`、`ECONNREFUSED`、`EHOSTUNREACH`、`ENETUNREACH`、连接超时、TLS 握手/证书错误）标成 `retryableAttempt: true`，语义就是“请求字节从未发出”。`self` 模式会对它重试，但只重试 3 次、总共约 1.2 秒（400 ms + 800 ms 退避），全部失败后最终错误被标成 `ambiguous`。`hosted` 模式更直接：`sendViaHosted` 不重试，任何异常都映射成 `ambiguous: true`。
   - 场景：用户在笔记本上唤醒后、网络或 VPN 还没就绪的几秒内跑日报（或服务商域名在其网络下偶发 DNS 失败）。日报本身已生成，邮件这一步遇到 DNS 失败：`self` 模式重试 3 次都失败，`hosted` 模式一次就失败，结果都是 `ambiguous`。网络一秒后恢复，再运行同一天，得到 `skipped / provider_outcome_ambiguous`，邮件没有请求发出过却再也发不出了。
2. **中继明确说“没有尝试”的 503。** 中继在尚未碰任何 provider 之前返回的 503 都带 `ambiguous: false`：自动投递未就绪（`delivery cutover is not ready`，包括部署后被锁定，见 MAIL-03）、账本不可用、`delivery attempt could not be started`、缺少 DO 绑定等（`services/email-relay/src/index.ts:190,213,243`、`deliver-gate.ts:162,232,245,263`）。客户端对所有 5xx 一律当作结果不明，不看这个字段。
   - 场景：中继维护或被锁定的那段时间里，所有 hosted 用户当天的邮件各被记成 `ambiguous`，即使中继恢复也不会补发，也没有任何人知道这些邮件“其实一封都没发”。

已用内存存储复现（一次性测试，已删除）：
- `self` 模式，三次尝试都抛 `HttpTransportError(network, retryableAttempt: true, cause.code = ENOTFOUND)` → `{"kind":"ambiguous","reason":"provider_outcome_ambiguous","attempts":3}`；网络恢复后再次调用 → `{"kind":"skipped","reason":"provider_outcome_ambiguous"}`，请求总数仍是 3。
- `hosted` 模式，一次 `ECONNREFUSED`（`retryableAttempt: true`）→ `ambiguous`；恢复后再次调用 → `skipped`，请求总数仍是 1。
- `hosted` 模式，中继返回 `503 {"error":"delivery cutover is not ready","ambiguous":false}` → `ambiguous`；中继恢复后再次调用 → `skipped`，请求总数仍是 1。

用户可见的后果：邮件没有到，但插件界面完全没有提示（`plugin/main.ts:2882-2890` 只处理 `delivery_storage_unsupported`），CLI 的 `onDailyCompleted` 直接丢弃返回值（`apps/cli/src/runtime.ts:194`），只有日志里有一行 `email: provider_outcome_ambiguous date=…`。

这是有意的“失败即关闭”设计里的一个缺口，而不是对设计的否定：`hosted-deliver.test.ts:418` 明确锁定了“不信任 5xx 里的 `ambiguous:false`”，`resend.test.ts` 也没有覆盖“所有尝试都是发送前失败”的情况。但 `retryableAttempt` 本来就是为“可证明未发送”引入的，重试用尽后却又把它当作“可能已发送”，两者自相矛盾。

## 原因

- `packages/core/src/delivery/resend.ts:147-163`：非 HTTP 响应的传输错误，`!canRetryPhysicalAttempt || attempt >= maxAttempts` 时一律抛 `ResendSendError(..., ambiguous = true)`，没有区分“每一次尝试都是发送前失败”。
- `packages/core/src/delivery/hosted.ts:145-152`：`catch {}` 吞掉所有异常（包括 `retryableAttempt: true` 的发送前失败），统一抛 `ambiguous = true`；没有重试。
- `packages/core/src/delivery/deliver-email.ts:388-403` `providerOutcomeAmbiguous`：对 `HostedDeliveryError` 只看 `status`（`undefined` 或 `>= 500` 都当结果不明），忽略错误本身和中继响应里的 `ambiguous` 字段。
- `packages/node-runtime/src/http-client.ts:124,179-203`：`retryableAttempt` 的来源与语义（“仅当失败发生在发送任何请求字节之前”），Obsidian 宿主恒为 `false`（`plugin/src/hosts/obsidian/http-client.ts:55,89`），所以插件宿主区分不出，不在本问题的修复范围内。

## 可能的解决方案

推荐：增加一个“可证明未发送”的结果，并按“未调用 provider”处理——释放这一代 claim，让下一次运行可以重试。

1. `sendViaResend` 记录每次尝试是否可能到达服务商：任何一次拿到 HTTP 响应或遇到 `retryableAttempt !== true` 的传输错误，就标记“可能已到达”。只有所有尝试都是 `retryableAttempt: true` 时，抛出带 `notSent: true` 的错误。
2. `sendViaHosted` 同样：`HttpTransportError` 且 `retryableAttempt === true` 时抛 `notSent`（可以顺带补上有界重试）。
3. `deliverDailyEmailIfEnabled` 对 `notSent` 走“未调用”分支：`releaseAutomaticDeliveryBeforeAttempt`（该函数已经支持“decision 已写入、本 owner 持有”的情形），返回 `failed / provider_not_invoked`，下一次运行重新 claim。
4. 对中继返回的 `503` 且响应体恰为 `{error, ambiguous:false}` 的情形，同样视为 `notSent`；其他 5xx 保持不明。因为这是明文 JSON 并经 TLS 传输，且只有 Worker 自己产生这种形状。如果不想放宽这条，至少在 503 后退避重试一两次再下结论。
5. 无论怎样，自动发送的失败要让用户看见：插件对 `ambiguous`/`failed` 给一次通知，CLI 在 `run` 的输出里打印邮件结果。

备选：提供 `arxiv-daily email retry --date YYYY-MM-DD`，对 `ambiguous` 的那一代 claim 在用户确认后创建新的 generation，并复用同一个 provider 幂等键（Resend 24 小时内去重、中继账本也有幂等记录），让用户可以手动补发。

## 修复后应补的测试

- `self`：所有尝试都是 `retryableAttempt: true` 的 `ENOTFOUND` → 结果为 `failed / provider_not_invoked`，claim 被释放；网络恢复后再次调用发送成功，只发一次。
- `self`：第 1 次 `ENOTFOUND`、第 2 次收到 HTTP 500 → 仍是 `ambiguous` 且保持阻断。
- `hosted`：`ECONNREFUSED`（`retryableAttempt: true`）→ 可重试；`retryableAttempt: false` 的超时 → 仍是 `ambiguous`。
- `hosted`：`503 {"error":"delivery cutover is not ready","ambiguous":false}` → 未调用，可重试；`503` 无该响应体、`500`、`502` → 仍是 `ambiguous`（保留现有的 408/418/500 测试）。
- 插件/CLI 在 `ambiguous`、`failed` 时给出可见提示的测试。
