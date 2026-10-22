# 模型服务商的内容安全拦截（HTTP 400）只要命中一篇论文，当天的日报就永久失败，不会再自动重试

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 PIPE-07
> 范围：日报流水线的论文过滤和逐篇摘要（所有宿主） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch2`，提交 100ccfc） · 置信度：已复现

## 问题

部分模型服务商会对输入做内容安全审核，命中时直接返回 HTTP 400，而不是正常回答，例如 DeepSeek 的 `Content Exists Risk`、智谱 GLM 的 1301（“输入或生成内容可能包含不安全或敏感内容”）。这是同一份输入每次都会得到的确定性结果。

客户端把除 408/409/425/429 以外的 4xx 都判为永久错误，而日报流水线对永久错误没有任何降级：

1. **过滤请求**：一天的所有候选论文（标题加完整摘要，通常几百篇）放在同一个请求里。只要其中一篇摘要触发审核，整个请求被拒，流水线返回 `failed_permanent`（“paper filter LLM failed: Content Exists Risk”）。永久失败的日期在调度器里属于“已处理”，不会自动重试；手动重试得到同样的结果。cs.CL、cs.CR、cs.CY 这类分类里，提到仇恨言论、越狱、极端内容、政治文本的论文并不少见，用上述两家服务商时，这一天就整天没有日报。
2. **逐篇摘要**：某一篇论文的摘要请求被拒，`summarizeDailyPaperWithValidationRetry` 把这个错误原样抛出（回退块只处理校验失败、被截断、暂时耗尽三种情况），整个 `summarizeDaily` 失败，当天所有论文（其余几十篇已经生成好的摘要）也一起作废，同样是 `failed_permanent`。

复现（一次性测试，用真实的 `LlmClient` 加假的 HTTP 层，被拦截的论文摘要里带 `FLAGGED-CONTENT`，服务返回 400 `Content Exists Risk`）：

- 摘要被拦：`{"kind":"failed_permanent","reason":"daily summary LLM failed: Content Exists Risk"}`，没有写日报，请求序列是 `filter, summary, summary`，另一篇正常的摘要已经白做；
- 过滤被拦：`{"kind":"failed_permanent","reason":"paper filter LLM failed: Content Exists Risk"}`，只发了一次请求，没有任何重新分批或排除可疑论文的尝试。

2026-10-18 的 CORE-06 把“上下文超长 400”靠分批避开了，但没有处理“某一份输入被服务商拒绝”这一类确定性 400。

## 原因

- `packages/core/src/llm/client.ts:616-627` `isPermanentLlmError`：除 408/409/425/429 以外的 4xx 一律永久。
- `packages/core/src/pipeline/pipeline.ts:293-296`：过滤阶段的任何永久错误直接返回 `failed_permanent`，没有隔离出问题论文的路径。
- `packages/core/src/pipeline/daily-paper-summary.ts:218-264`：永久错误走最后的 `throw error`（第 263 行），不产生回退结果；`packages/core/src/pipeline/pipeline.ts:547-553` 把它变成整天的 `failed_permanent`。
- `packages/core/src/services/state-store.ts:74-77`：`failed_permanent` 视为已完成，自动调度不再重试。

## 可能的解决方案

推荐：

1. 逐篇摘要：把“内容被服务商拒绝”的永久错误当作这一篇的回退（`reasonCode` 增加一类，显示原始摘要和“服务商拒绝处理”提示），其余论文照常出报告。认证、配额、模型不存在这类影响全局的 401/403/404 仍然整天失败。
2. 过滤：整批被 400 拒绝时，对批次做二分重试，找出被拦的论文并排除（记录在筛选决策里），其余论文继续分类；全部拆到单篇仍被拒才算这篇被跳过。
3. 区分错误类型：只对明确的内容审核错误（错误码、错误文本）走上述降级，避免把真正的请求格式错误吞掉。

备选：把这类错误降为暂时失败并提示用户换模型；缺点是确定性错误每天重试十次都会失败。

## 修复后应补的测试

- 单篇摘要请求返回内容审核 400：该论文显示回退块，其余论文正常，日报正常写出。
- 过滤批次中有一篇触发审核：其余论文仍被分类，被拦的论文被记录为跳过。
- 401/403/404 仍然让整天失败，并给出清晰的原因。

## 修复情况

服务商的内容安全拦截（HTTP 400，且错误码或文字明确指向内容审核：DeepSeek 的 "Content Exists Risk"、智谱 1301、Azure 和 Moonshot 的 `content_filter`、DashScope 的 `data_inspection_failed`、OpenAI 的 `invalid_prompt` 加政策措辞）不再让整天永久失败。逐篇摘要被拒时，该论文在日报里显示为回退块（原始摘要，并写明"模型服务商拒绝为这篇论文生成摘要"），其余论文照常出报告；这个结果存进检查点，续跑时不再重问。过滤请求被拒时，把批次对半拆开，找出被拒的论文并排除（日志里按编号列出，决策记录里记为"被模型跳过"），其余论文照常分类。一批里被拒超过 8 篇，或者全部被拒，仍按原样失败，避免把"提示词本身被拒"当成"今天没有相关论文"。401、402、403、404 和请求格式错误仍让整天失败，原因不变。

取舍：只识别上面有文档可依的错误码和文字，其他服务商的审核错误仍是整天永久失败，需要时再逐个加入。被拒的论文在过滤阶段不会出现在日报里（被当作不相关，无法单独提示）。

测试：`llm/provider-content-block.test.ts`（各家的真实错误体，以及不能误判的请求格式错误和 401/402/403/404）、`pipeline/provider-content-block.test.ts`（真实 `LlmClient` 加假服务，复现“复现”一节里的两个场景）、`paper-filter-refusal.test.ts`、`summarizer.test.ts`、`daily-paper-summary.test.ts`、`daily-summary-assembler.test.ts`、`daily-summary-checkpoint-store.test.ts`。
