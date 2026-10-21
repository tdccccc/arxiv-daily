# 模型回复被安全拦截、被服务端打断或中途断流时，残缺文本会被当成完整回答；残缺的论文笔记会永久落盘

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 PIPE-01
> 范围：所有 LLM 调用（日报过滤、逐篇摘要、论文详细笔记） · 严重程度：中 · 状态：未修复 · 置信度：已复现

## 问题

`LlmClient.call()` 只把 `finish_reason: length`（以及 Anthropic 的 `stop_reason: max_tokens`）当成“被截断”（2026-10-18 的 CORE-05 已修）。下面这些同样意味着“回答没写完”的情况，现在都会被当成成功，把已收到的半截文本原样返回：

- 结束原因不是 `stop` 的其他取值：DeepSeek（默认推荐预设）文档里的 `insufficient_system_resource`（推理资源不足、生成被打断）和 `content_filter`，智谱 GLM（预设之一）的 `sensitive`（被安全审核拦截）和 `network_error`（推理异常），Azure/OpenAI 兼容网关的 `content_filter`，OpenRouter 类网关的 `finish_reason: "error"`。
- 流里出现 `data: {"error": {...}}` 事件（网关把上游失败转成 SSE 错误事件），随后正常给出 `data: [DONE]`。
- 连接被中间代理干净地关闭，既没有 `finish_reason` 也没有 `[DONE]`；Anthropic 原生接口则是没读到 `message_stop` 就结束。
- 服务端忽略了 `stream: true`，直接返回一个普通 JSON 对象：客户端找不到任何 `data:` 行，返回空字符串，不报错。设置页的“测试连接”走的是非流式请求，所以测试会显示成功，真正运行时才出问题。

后果：

1. **论文详细笔记**：`summarizePaperDetail` 只检查“非空”。一篇写到一半的笔记会写进 `papers/<id>.md`，日报里还会带上它的链接。实测，一篇在第 4 个小节中途被截断的笔记会被 `classifyPaperNote` 判为 `verified_detail`（只要正文不少于 400 字、至少 3 个已知小节标题或至少 4 个二级标题就算），以后永远被当成完整笔记：不会重新生成，选择器也不再考虑这篇论文。
2. **每日过滤**：半截或空的 JSON 只会得到“paper filter response validation failed: response is not strict JSON”，被记为暂时失败；同一份请求在温度 0 下一遍遍得到同样的结果，十次后整天变成永久失败，而日志里完全看不出真实原因是“模型被安全拦截”或“服务端忽略了流式”。
3. **逐篇摘要**：半截 JSON 会走重试和回退，最终显示为“自动摘要不可用”，同样看不出原因。

## 原因

- `packages/core/src/llm/client.ts:745-780` `collectParsedStream`：只认 `finish_reason` 是 `length` / `max_tokens`（第 775 行）；不读取 `error` 对象，也不要求看到任何结束标志，流结束就返回 `chunks.join("")`（第 779 行）。
- `client.ts:648-683` `parseAnthropicEvents`：`error` 事件被转成 HTTP 状态错误，但流在 `message_stop` 之前结束时没有任何检查；`stop_reason` 只区分 `max_tokens` 与其他（第 677-680 行）。
- `client.ts:711-719` `parseSseEvent` 与 `client.ts:690-709`：非 SSE 的响应体被当成“没有事件”，得到空内容而不是错误。
- `client.ts:242-251` `testConnection` 用 `stream: false`，没有覆盖真实调用所用的流式路径。
- `packages/core/src/pipeline/summarizer.ts:322` 只拒绝空字符串；`packages/core/src/pipeline/pipeline.ts:470-491` 写盘后直接把笔记当成功；`packages/core/src/dashboard/detail-summary.ts:3,23,25` 的“像不像详细笔记”判断无法识别半截笔记。

## 复现

用一个假的流式 `HttpClient` 调 `LlmClient.call()`（一次性测试，已删除），收到的都是 `{ok: "# Title\n\n## 研究问题\n部分内容，被截断于此"}`，没有抛出错误：

- 先给内容，再给 `finish_reason: "content_filter"`，再 `[DONE]`；
- 只给内容，流直接结束；
- 先给内容，再给 `{"error": {...}}` 事件，再 `[DONE]`；
- 先给内容，再给 `finish_reason: "error"`；
- Anthropic 原生事件流在 `message_stop` 之前结束；
- 响应体是普通 JSON（服务端忽略 `stream`）：返回空字符串。

## 可能的解决方案

推荐：

1. 流收集器记录“是否见到正常结束标志”：OpenAI 风格要求最后出现 `finish_reason` 为 `stop`（或 `tool_calls`），Anthropic 要求 `message_stop` 且 `stop_reason` 为 `end_turn` / `stop_sequence`。其余情况（缺失、`content_filter`、`sensitive`、`network_error`、`insufficient_system_resource`、`error`）抛出类型化错误。
2. 对 `data:` 里的 `error` 对象直接抛错；响应体不是 SSE 时，如果能解析成 `{choices:[{message:{content}}]}` 就按非流式结果使用，否则抛出带“服务端未按流式返回”说明的错误。
3. 结束原因类的错误按“确定性输出失败”处理（不盲目重发），给出能让用户看懂的原因；`insufficient_system_resource` 这类服务端资源问题可按暂时错误重试。
4. 详细笔记写盘前再检查一次结构（必需标题齐全），不通过就不落盘。

备选：只处理“流结束时没有结束标志”这一条，其余按提供商逐个加白名单。

## 修复后应补的测试

- 上述六种流形态下 `call()` 都抛出错误，且不会把半截文本返回给调用方。
- 正常 `stop` / `end_turn` 的流不受影响，`[DONE]` 之前的保活注释行仍然不算事件。
- 一篇被截断的详细笔记不会写入 `papers/`，日报里不会出现它的链接。
- 非 SSE 的 JSON 回应要么被正确解析，要么得到明确的错误提示。
