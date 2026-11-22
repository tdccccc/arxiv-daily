# 非原生 Anthropic 代理/网关配置下仍发送 Anthropic 原生思考字段且跳过参数回退

状态：已修复（分支 `fix/2026-11-22-review`）

> 来源：2026-11-22 main `2557b90a` 代码审查（LLM 客户端）
> 严重程度：中 · 置信度：已复现（离线请求体实测：向 `/chat/completions` 发送 `thinking` 与 `output_config`，且 `rejectedOptionalParameter` 返回 `undefined`）

## 问题与依据

`packages/core/src/llm/client.ts:241-245` 定义了 `usesNativeAnthropic(settings)`，只有当 `provider === "anthropic"` **且** `baseUrl` 指向 `api.anthropic.com` 时才走 Anthropic 原生 `/messages` 协议；当用户配置自定义反代、聚合网关或本地代理（`provider: "anthropic"` + 非 `api.anthropic.com` 的 `baseUrl`）时，请求路由到 OpenAI 兼容的 `/chat/completions`。

但两处关键逻辑没有使用 `usesNativeAnthropic(settings)`，而是直接判断了 `settings.provider === "anthropic"`：

1. `packages/core/src/llm/client.ts:723-748`（`generationParameters`）：
   - 开启 `thinkingMode` 且模型命中 `usesAdaptiveAnthropicThinking(model)`（如 `claude-opus-4-7`）时，向 OpenAI 兼容的 `/chat/completions` 请求体发送了 Anthropic 原生专有字段 `thinking: { type: "adaptive" }` 和 `output_config: { effort: ... }`；
   - 其他 Claude 模型发送 `thinking: { type: "enabled", budget_tokens: ... }`，关闭思考时发送 `thinking: { type: "disabled" }`。
2. `packages/core/src/llm/client.ts:770,790`（`rejectedOptionalParameter`）：
   - 第 790 行写的是 `if (settings.provider === "anthropic") return undefined;`（意图注释第 766 行明确写着只针对 `Native Anthropic`），而非 `if (usesNativeAnthropic(settings)) return undefined;`；
   - 且第 770 行 `DROPPABLE_GENERATION_PARAMETERS = ["thinking", "reasoning_effort", "temperature"]` 不包含 `output_config`。

结果：通过 OpenAI 兼容中转/网关调用 Claude 模型时，一旦网关按 OpenAI schema 校验并针对未知参数 `thinking` 或 `output_config` 返回 HTTP 400/422，客户端**不会**触发可选参数降级重试，而是直接将其判为永久失败（`isPermanentLlmError`），整个日报任务直接报错终止。

## 原因

`generationParameters()` 和 `rejectedOptionalParameter()`（后者来自近期提交 `363d8129`）仅按 `settings.provider === "anthropic"` 区分协议，未复用 `usesNativeAnthropic(settings)`（`rejectedOptionalParameter` 的参数签名甚至只收了 `Pick<LlmSettings, "provider">` 而未传 `baseUrl`）。

## 建议和验收

- `generationParameters()` 中生成 Anthropic 原生参数格式（`max_tokens` 默认值、`thinking: { type: "adaptive" }`、`output_config`、`budget_tokens`）应以 `usesNativeAnthropic(settings)` 为条件；当 `provider === "anthropic"` 但走 OpenAI 兼容网关时，按 OpenAI 兼容参数格式组装（如 `reasoning_effort`）。
- `rejectedOptionalParameter()` 接收 `Pick<LlmSettings, "provider" | "baseUrl">`，第 790 行改为 `if (usesNativeAnthropic(settings)) return undefined;`，并在需要时将 `output_config` 纳入可丢弃列表或保证非原生路径不再发出 `output_config`。
- 验收：配置 `provider: "anthropic"`、`baseUrl: "https://proxy.example.com/v1"`、`model: "claude-opus-4-7"`、`thinkingMode: true`：请求 `/chat/completions` 时不再发送 `output_config` / 原生 `thinking` 格式，且代理返回 400 拒绝可选推理参数时能自动丢弃该参数并重试成功。

## 修复后应补的测试

- `packages/core/tests/llm/`：
  - `provider: "anthropic"` + 非 `api.anthropic.com` `baseUrl` 时，请求体不带原生 `output_config` / `budget_tokens`，走 OpenAI 兼容参数；
  - 同配置下若端点返回 400 `"Unknown parameter: 'reasoning_effort'"`（或 `'thinking'`），`LlmClient` 自动降级丢弃参数并重试成功；
  - 原生 `api.anthropic.com` 仍保留原有行为（带原生 `thinking`/`output_config`，且不自动静默丢弃 `thinking`）。

## 修复情况

原生 Anthropic 参数生成改为以 `usesNativeAnthropic`（provider + `api.anthropic.com` 主机）为条件；Anthropic 代理/网关走 OpenAI 兼容参数（`reasoning_effort`），不再发送原生 `thinking`/`output_config`。`rejectedOptionalParameter` 接收 `provider`/`baseUrl` 并在非原生路径允许丢弃 `reasoning_effort`/`output_config`。测试：`tests/llm` 9 个文件 164 项通过，覆盖代理请求体与 400 降级重试。
