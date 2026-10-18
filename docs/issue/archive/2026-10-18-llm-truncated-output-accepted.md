# 被截断的 LLM 输出被当成完整结果；Anthropic 默认 max_tokens=4096（含思考），论文笔记会被截断写盘

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CORE-05
> 范围：LLM 客户端、论文详情笔记、日报摘要与过滤 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 08cb46c） · 置信度：已复现

## 问题

- provider 为 anthropic（默认预设开启思考模式）且调用方没给 `maxCompletionTokens` 时，`generationParameters()` 把 `max_tokens` 设为 4096；自适应思考分支提前返回，不会调高。思考 token 计入 `max_tokens`，模型可能把大半预算用在思考上，可见回答被截断。Anthropic 兼容网关同样受影响（按 provider 判断，不按主机）。
- OpenAI 风格的 `finish_reason` 和 Anthropic 的 `stop_reason: max_tokens` 从不读取，截断的生成被当作正常字符串返回。
- 每日摘要和过滤 JSON 有结构校验，会明确失败（但温度 0 下每次都同样失败，见 CORE-06）；`summarizePaperDetail` 只检查非空，一篇写到一半的笔记（模板 9 节，中文加 LaTeX 很容易超过 4096 token）会写入 `papers/<id>.md`。流水线把已存在的笔记视为完成，以后不会再生成。

## 原因

- `packages/core/src/llm/client.ts:427-431`：anthropic 默认 `max_tokens = 4096`。
- `client.ts:438-443`：自适应思考分支提前返回。
- `client.ts:496-527`、`641-646`：流解析丢弃结束原因。
- `packages/core/src/pipeline/summarizer.ts:284-301` 及日报/过滤调用方未传 `maxCompletionTokens`。

## 可能的解决方案

1. 流收集器记录 `finish_reason` / `stop_reason`，为 length / max_tokens 时 `LlmClient.call()` 抛出类型化的 `LlmOutputTruncatedError`，调用方把它当作失败而不是成功。
2. 提高默认预算：Anthropic（尤其自适应思考）默认改为更宽裕的值（如 32k），日报过滤、摘要和详情笔记显式传入预算。
3. 详情笔记写盘前校验结构（必需标题都在），拒绝保存残缺笔记。

## 修复后应补的测试

- 模拟 `finish_reason: "length"` 与 `stop_reason: "max_tokens"` 时调用抛出截断错误。
- anthropic 默认参数不再是 4096。
- 截断的详情笔记不会写入磁盘。

## 修复情况

读到 `finish_reason: length` 或 `stop_reason: max_tokens` 时抛出 `LlmOutputTruncatedError`：被截断的每日摘要走降级，被截断的详情笔记不再写盘。Anthropic 默认 `max_tokens` 改为 8192（不开思考，所有 Claude 模型都接受的上限）/ 32000（开思考）。

注意：其他 provider 仍用服务端默认上限，上限较小的服务现在会让详情笔记明确失败，而不是保存残缺笔记。
