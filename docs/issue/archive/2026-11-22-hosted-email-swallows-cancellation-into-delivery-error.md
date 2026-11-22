# Hosted 邮件投递把取消异常和前置校验错误吞成了 HostedDeliveryError（标记为不确定状态），且 Resend 发件人名称未转义引号

状态：已修复（分支 `fix/2026-11-22-review`）

> 来源：2026-11-22 main `2557b90a` 代码审查（邮件投递）
> 严重程度：中 · 置信度：已复现（离线实测：`RunCancelledError` 被包装成 `HostedDeliveryError(..., ambiguous: true, notSent: false)`）

## 问题与依据

### 1. `sendViaHosted` 把 `RunCancelledError` 和 `beforeProviderAttempt` 异常误包装为模糊网络错误
- `packages/core/src/delivery/hosted.ts:127-160`：
  ```ts
  try {
    await opts.beforeProviderAttempt?.();
    opts.onProviderInvocation?.();
    res = await opts.http.request({ ... });
  } catch (error) {
    const notSent = isHttpTransportError(error) && error.requestNotSent;
    throw new HostedDeliveryError(
      "Cannot reach Official delivery.",
      undefined,
      undefined,
      !notSent,
      notSent,
    );
  }
  ```
- 对比同目录的 `packages/core/src/delivery/resend.ts`：`resend.ts` 先在 `try/catch` 外执行 `beforeProviderAttempt`，并且显式透传 `isCancellationError(error)`。
- 在 `hosted.ts` 中，当用户取消任务（`beforeProviderAttempt()` 或 `http.request()` 抛出 `RunCancelledError`）或命名空间守卫失败时，第 149-159 行全部将其吞掉并伪装成 `HostedDeliveryError("Cannot reach Official delivery.", ..., ambiguous = true, notSent = false)`：
  - 上层 `deliverDailyEmailIfEnabled` 不再将其识别为取消（`cancelled_before_provider_attempt`），而是视为“可能已经发给中继的模糊失败”；
  - 即使请求根本还没发出去（在 `beforeProviderAttempt()` 阶段就被取消），也因为 `notSent = false` 而阻止自动重试、并向用户弹出误导性的无法连接官方邮件服务警告。

### 2. `formatResendFrom` 未对包含 RFC 5322 特殊字符的发件人显示名加引号
- `packages/core/src/delivery/resend.ts:192-199`：若用户配置的 `fromName` 含逗号、括号、句点、分号等 RFC 5322 特殊字符（如 `Dr. Smith, Lab`），直接拼接 `${fromName} <${fromEmail}>` 而未按 RFC 5322 进行双引号转义，可能被 Resend API 以 422 拒绝。

## 原因

`sendViaHosted` 把 `beforeProviderAttempt()` 和 `http.request()` 一起放进了无条件包装为 `HostedDeliveryError` 的 `try/catch` 中，遗漏了 `isCancellationError` 透传以及与 `resend.ts` 一致的阶段分离。

## 建议和验收

- 将 `await opts.beforeProviderAttempt?.()` 移到 `http.request` 的 `try/catch` 之前（或在 `catch` 开头判断若 `isCancellationError(error)` 直接 `throw error`，同时在调用 `onProviderInvocation()` 之前的失败不要标记为 `ambiguous: true`）。
- `formatResendFrom` 对含 RFC 5322 `specials`（如 `.,;:<>@[]()\"`）的 `fromName` 转义 `\` 与 `"` 并用双引号包裹。
- 验收：`beforeProviderAttempt` 抛出 `RunCancelledError` 时 `sendViaHosted` 原样抛出 `RunCancelledError` 而不是 `HostedDeliveryError`；`fromName: "Dr. Smith, Lab"` 格式化为 `"\"Dr. Smith, Lab\" <...>"`。

## 修复后应补的测试

- `packages/core/tests/delivery/`：
  - `sendViaHosted` 在 `beforeProviderAttempt` 或 `http.request` 抛出 `RunCancelledError` 时，断言原样 reject 为取消错误且不会把投递标记为 ambiguous；
  - `formatResendFrom` 针对含逗号、句点、双引号的 `fromName` 生成合法 RFC 5322 `From` 头。

## 修复情况

`sendViaHosted` 将 `beforeProviderAttempt` 移出 provider 的 try 块，取消异常原样透传，前置校验失败不再伪装为 `HostedDeliveryError(ambiguous)`；`formatResendFrom` 对含 RFC 5322 特殊字符的显示名加双引号并转义 `\` 与 `"`。测试：`hosted-deliver` + `resend` 118 项通过。
