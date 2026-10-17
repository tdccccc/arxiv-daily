# LLM 短暂不可用时的逐篇降级摘要被直接写进日报，之后再也不会重试

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CORE-07
> 范围：日报逐篇摘要、论文详情笔记 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 2de28ed、8fd9339） · 置信度：代码推断

## 问题

服务端短暂故障、限流或超时（5xx、429、网络、超时在内部 3 次尝试后都变成 `LlmTransientExhaustedError`）时，`summarizeDailyPaperWithValidationRetry` 对该论文返回 `transport-exhausted` 降级结果（原始摘要），而不是失败。`summarizeDaily` 只计数并打一条警告，照常组装日报；`pipeline.runForDate` 写入日报并清理检查点。

从此日报文件就是最终结果：同一天再运行会在 `pipeline.ts:200` 直接走"日报已存在，修复索引"，不再调用 LLM。于是摘要循环中哪怕只有一分钟的故障，剩下的论文都永久变成原始摘要，运行记为 `completed`，用户也不知道可以重试。

这与文档设计相反（`docs/helm/2026-08-01-daily-summary-checkpoint-recovery`：`transport-exhausted` 条目在 `lookupReusable` 中视为未命中，"恢复的运行只重试那篇论文"）。但只有运行失败或取消、日报未写出时才会恢复，而代码在这种情况下从不让运行失败。详情笔记同理（`pipeline.ts:478-483` 只记录日志）。

## 原因

- `packages/core/src/pipeline/summarizer.ts:183-192`：把降级当成功。
- 没有策略把 `transport-exhausted` 转为 `failed_transient`；`pipeline.ts:555` 的 `cleanupCommittedCheckpoints` 删除了本可让下一轮只重跑这些论文的检查点。

## 可能的解决方案

推荐：任意论文以 `transport-exhausted` 结束时不写日报，返回 `failed_transient`（原因如"k/N 篇论文 LLM 不可用"）并保留检查点，下一轮调度只重试这些论文（现有检查点语义已支持）。为避免持续失败的论文导致无限循环，计入现有的暂时失败次数上限（`MAX_TRANSIENT_ATTEMPTS`），达到上限前的最后一次尝试再带降级结果提交日报。

备选：提交降级日报但在运行状态里记录"降级"标记，面板提供"重新生成降级论文"（需要 ADR 层面的决定）。

## 修复后应补的测试

- 摘要阶段出现 `transport-exhausted` 时运行返回 `failed_transient`，日报未写出，检查点保留；下一次运行只对该论文调用 LLM。
- 达到尝试上限的最后一次时，带降级结果提交日报。

## 修复情况

新增 `PipelineRunOptions.retryTransportFallbacks`。除最后一次允许的暂时失败尝试外，调度器都要求重试：流水线在第一篇不可达论文处停下，返回 `failed_transient` 并保留检查点，下一次只重做剩余论文；最后一次尝试仍带降级结果提交日报。CLI 与插件宿主都已透传该选项。
