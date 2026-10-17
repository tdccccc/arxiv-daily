# 邮件中继验证码限流是非原子的 KV 读改写，并发请求可绕过按邮箱和按 IP 的上限

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CLI-9
> 范围：`services/email-relay` `/v1/verify/start` · 严重程度：低 · 状态：部分修复（分支 `fix/2026-10-18-review`，提交 84046ae；余项见文末） · 置信度：代码推断

## 问题

`/v1/verify/start` 无需认证，会从项目的 Resend 域名向任意地址发邮件。上限（每地址每小时 3 次、每 IP 每小时 10 次）通过 Workers KV 的"读—比较—写"实现。KV 最终一致（读可能落后约 60 秒），没有比较并交换，N 个并发请求都读到 0 并全部通过，攻击者可以对受害者地址邮件轰炸、消耗 Resend 发送信誉和额度，远超宣称的上限。

另外地址计数先于 IP 计数递增：已被 IP 限流的攻击者仍会消耗受害者地址每小时的 3 个名额，让真正的地址主人收不到验证邮件。

`kv.ts` 已注明这些限制是尽力而为，属于已知局限，不是回归。

## 原因

- `services/email-relay/src/kv.ts:222` `checkAndIncrRateLimit` 先读再写（带 `expirationTtl`），无原子计数。
- `services/email-relay/src/index.ts:106-127` `verifyStart` 先检查邮箱再检查 IP。

## 可能的解决方案

推荐（分两步）：
1. 本次：先检查 IP 上限，再检查邮箱上限，避免已被 IP 限流的请求消耗受害者地址的名额。
2. 后续：把计数器移入 Durable Object（已有 DeliverGate 命名空间），按 hash(email) / hash(IP) 事务性递增；或在路由前加 Cloudflare 速率限制规则。这涉及部署与 Cloudflare 配置变更，需单独评估。

## 修复后应补的测试

- IP 已超限时请求被拒绝，且邮箱计数不增加。

## 修复情况

已改为先检查 IP 上限再检查邮箱上限，被 IP 限流的请求不再消耗受害者地址的名额（新测试在旧顺序下失败）。

**仍未解决**：KV 读改写仍非原子，并发请求仍可超出上限。迁移到 Durable Object 或使用 Cloudflare 速率限制规则涉及部署与配置变更，保留为后续项。
