# LLM "流式"请求其实整体缓冲：120 秒空闲超时从不生效，超时后还会盲目重发 3 次

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CORE-03
> 范围：所有 LLM 调用（日报过滤、摘要、文献库） · 严重程度：中 · 状态：部分修复（分支 `fix/2026-10-18-review`，提交 4e073ff、5394452；余项见文末） · 置信度：代码推断

## 问题

`LlmClient.call()` 发送 `stream: true`，但 `HttpClient.request()` 要等整个响应体收完才返回（Node 端是 `await res.text()`），流解析只是在一个已完整的字符串上迭代。后果：

1. `LLM_STREAM_IDLE_TIMEOUT_MS`（120 秒）永远不会对网络生效；每次尝试创建的 `AbortController` 也没接到 HTTP 请求上。
2. 唯一真实的上限是整次生成 300 秒。超时错误既不算永久错误也不算输出超限，于是同一请求被重发 3 次（每次最多等 5 分钟，外加退避），约 15 分钟、3 倍费用后才以暂时失败结束，之后各调度轮次还会继续重试。
3. 输出长度上限只在整个响应下载完后才检查，起不到限制内存和费用的作用。
4. arXiv 抓取器会尊重 `HttpTransportError.retryableAttempt=false`（超时的请求可能还在服务端执行，不应盲目重发），LLM 客户端却忽略它，可能同时产生多份计费生成。

## 原因

- `packages/core/src/llm/client.ts:361-368`：先 `await requestChat(...)` 拿到全文，再对 `parseSseText(raw)` 做空闲超时迭代。
- `client.ts:388-395`：只传入用户的 `signal`，没传每次尝试的 `controller.signal`。
- `client.ts:318-320` `shouldRetry` 不看 `retryableAttempt`。
- `packages/node-runtime/src/http-client.ts:54-58` 用 `res.text()` 缓冲。

## 可能的解决方案

方案 A（完整）：给 `HttpClient` 增加可选的流式接口，Node 端用 fetch 实现，Obsidian 的 `requestUrl` 退回缓冲；空闲超时和输出上限才能真实生效。

方案 B（本次推荐落地）：
- `shouldRetry` 尊重 `retryableAttempt=false`，超时类传输错误不再盲目重发（或最多重发一次）。
- 把每次尝试的中止信号与用户信号合并后传给 HTTP 请求。
- 修正注释/命名，不再声称有 120 秒空闲保护；方案 A 作为后续工作。

## 修复后应补的测试

- 返回 `retryableAttempt=false` 的超时错误时，LLM 调用只发一次请求。
- 普通 5xx 仍按原策略重试。

## 修复情况

方案 B 已落地：超时后不再重发；连接阶段的错误（DNS、拒绝连接、不可达、连接超时、TLS 失败）照常重试；请求中途断开（ECONNRESET、socket hang up）只重发一次；每次尝试的中止信号已接到请求上；注释不再声称有网络空闲超时。

**仍未解决**：真实流式（方案 A）没有做，空闲超时和输出上限仍要等整个响应下载完才生效。本文件保留在 `docs/issue/` 作为后续项。
