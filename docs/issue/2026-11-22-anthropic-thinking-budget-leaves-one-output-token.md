# 非自适应思考的 Claude 模型传入 maxCompletionTokens=4096 时，思考预算占掉 4095 token 仅剩 1 个输出 token

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（LLM 客户端与个人文献库方向生成/预览）
> 严重程度：中 · 置信度：已复现（离线请求体实测：`max_tokens: 4096`、`thinking.budget_tokens: 4095`，剩余输出 1 token）

## 问题与依据

在 `packages/core/src/llm/client.ts:740-748` 中，原生 Anthropic 非自适应思考模型（如 `claude-sonnet-4-5`、`claude-haiku-4-5`、`claude-3-7-sonnet`）开启 `thinkingMode: true` 时计算 `max_tokens` 与 `thinking.budget_tokens` 的逻辑如下：

```ts
const budgets: Record<string, number> = { low: 2048, medium: 8192, high: 16384 };
const requested = budgets[settings.reasoningEffort] ?? 8192;
const cap = maxCompletionTokens
  ?? Math.min(requested + 16_000, ANTHROPIC_THINKING_MAX_TOKENS);
if (cap <= 1024) {
  throw new TypeError("Anthropic thinking requires maxCompletionTokens greater than 1024");
}
params.max_tokens = cap;
params.thinking = { type: "enabled", budget_tokens: Math.min(requested, cap - 1) };
```

核心层有多处结构化 JSON 调用显式传入了 `maxCompletionTokens: 4096`：
- `packages/core/src/library/personal-library-direction-proposer.ts:63,285`（文献库研究方向提议与聚类综合）
- `packages/core/src/library/personal-library-direction-preview.ts:50`（研究方向样本预览筛选）

当用户使用 `claude-sonnet-4-5` / `claude-haiku-4-5` / `claude-3-7-sonnet` 且开启 `thinkingMode: true`、默认 `reasoningEffort: "medium"`（`requested = 8192`）或 `"high"`（`requested = 16384`）时：
- `cap = 4096`
- `budget_tokens = Math.min(8192, 4095) = 4095`
- Anthropic 的 `max_tokens` 是 **thinking + visible output 的总上限**，因此留给最终 JSON 回复的可见输出空间**只有 `4096 - 4095 = 1` 个 token**！

一旦模型思考接近预算上限，输出必然被截断（`stop_reason: "max_tokens"`），触发 `LlmOutputTruncatedError` 或损坏 JSON，导致文献库方向生成与方向预览在默认思考设置下高频失败。

## 原因

第 742-743 行在调用方未指定 `maxCompletionTokens` 时特意给思考加了 `+ 16_000` 的输出余量；但调用方传入 `maxCompletionTokens`（原意是约束**可见输出**上限，如 4096）时，直接把 `cap` 设成了 `maxCompletionTokens`，然后用 `Math.min(requested, cap - 1)` 把几乎全部额度都分配给了 `budget_tokens`。

## 建议和验收

- 区分“可见输出上限”与“思考预算”：在 Anthropic 原生非自适应思考路径下，当调用方传入 `maxCompletionTokens` 时，要么将 `params.max_tokens` 扩展为 `Math.min(requested + maxCompletionTokens, ANTHROPIC_THINKING_MAX_TOKENS)`（保证至少为可见输出保留完整的 `maxCompletionTokens`），要么在必须严格以 `maxCompletionTokens` 为总上限时将 `budget_tokens` 限制为至少预留合理输出空间（例如 `Math.min(requested, Math.max(1024, Math.floor(cap / 2)))`）。推荐前者，与未传 `maxCompletionTokens` 时 `requested + 16_000` 的思路一致。
- 验收：`provider: "anthropic"`、`baseUrl: "https://api.anthropic.com/v1"`、`model: "claude-sonnet-4-5"`、`thinkingMode: true`、`reasoningEffort: "medium"` 且 `maxCompletionTokens: 4096` 时，`params.max_tokens - params.thinking.budget_tokens >= 4096`（或至少保留充裕可见输出预算而非 1 token）。

## 修复后应补的测试

- `packages/core/tests/llm/`：针对 `claude-sonnet-4-5` + `thinkingMode: true` + `reasoningEffort: "medium"/"high"` + `maxCompletionTokens: 4096`，断言发送给 Anthropic 的 `max_tokens - thinking.budget_tokens` 至少覆盖可见输出预算（例如 `>= 4096`），且 `thinking.budget_tokens >= 1024`。
