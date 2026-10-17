# 创建 LlmClient 会覆盖日志脱敏列表，邮件、托管令牌、嵌入密钥不再被脱敏

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CORE-04
> 范围：CLI 运行时、`library` 命令；插件只注册了 LLM 密钥 · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 199181d） · 置信度：已复现

## 问题

`Logger.setSensitiveValues()` 是"替换"而不是"追加"。CLI 运行时先注册 [LLM 密钥、Resend 密钥、email.apiKey、托管令牌]，随后 `new LlmClient(settings.llm, logger, http)` 的构造函数调用 `logger.setSensitiveValues([settings.apiKey])`，其余密钥全部被清掉。`library-cmd.ts` 同样如此（先注册 LLM 与嵌入密钥，再构造 LlmClient）。

之后邮件和嵌入凭据只能靠通用正则兜底（`Bearer`、`sk-/rk-/pk-/api-` 前缀、`token=` 等）；Resend 密钥（`re_...`）和托管验证码都匹配不上。目前没发现实际泄露路径，但这层纵深防御已经失效，以后任何把服务端回显写进日志的代码都可能把凭据写进诊断导出。

## 原因

- `packages/core/src/services/logger.ts:35-38`：赋值而非合并。
- `packages/core/src/llm/client.ts:193`：构造函数对共享 logger 有副作用。
- 调用方：`apps/cli/src/runtime.ts:66`、`apps/cli/src/library-cmd.ts:171`。

## 可能的解决方案

推荐：新增 `Logger.addSensitiveValues()`（只增不减），`LlmClient` 改用它；`setSensitiveValues` 保留给宿主在重新加载设置时整体替换。插件宿主也注册邮件与嵌入密钥。

## 修复后应补的测试

- 先注册多个密钥再构造 `LlmClient`，所有密钥仍被脱敏。

## 修复情况

新增 `Logger.addSensitiveValues`（只增不减），`LlmClient` 改用它。插件宿主本来就注册了全部密钥，之前同样被构造函数冲掉，修复后一并恢复。
