# 日报摘要和详细笔记一篇做完再做下一篇，日报耗时随论文数线性增长

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：中 · 置信度：代码推断

## 问题与依据

`packages/core/src/pipeline/summarizer.ts:143-226` 的 `summarizeDaily` 用一个 `for` 循环逐篇 `await` 结构化摘要；`packages/core/src/pipeline/pipeline.ts:480-534` 的详细笔记同样逐篇串行。与之对照，正文抓取在 `pipeline.ts:106,385` 以 6 路并发进行。日志也写着 `assembled N sequential paper summaries`。

ADR 0015 第 71 行记录过"每篇一次串行模型调用"让 20 篇日报多花约 1 分钟——那是短输出的新颖性一行。结构化摘要带最多 3 次校验重试、详细笔记输出很长、思考模式下单次延迟更高，所以用户在 Obsidian 状态栏或工作台里看到的生成时间约等于 篇数 × 单次模型延迟：20 篇在推理模型下常达十几分钟。首次引导的"生成第一份日报"也要等同样久。

## 原因

- 早期为了限流和实现简单采用串行；后来没有加并发配置。
- 并行所需的保障其实已经具备：摘要 checkpoint 存储按日期文件排队写入（`packages/core/src/services/daily-summary-checkpoint-store.ts:443-458`），LLM 客户端对 429/5xx 已有带退避的重试（`packages/core/src/llm/client.ts:367-424`）。

## 建议和验收

- 为每篇摘要和每篇详细笔记加有界并发（默认 3，`advanced` 里可调，1 即为现状），结果按输入顺序回填 slots，日报内容与串行完全一致。
- 进度按"完成数"上报（见 PIPE-06）。
- `failOnTransportExhausted` 触发时取消其余在途调用，已完成的 checkpoint 保留。
- 详细笔记并行时每篇仍独立落盘；一篇失败不影响其他。

## 修复后应补的测试

- 用延迟 mock 的 LLM：N 篇时在途调用数不超过并发上限；slots 顺序与输入顺序一致；日报 Markdown 与串行版本逐字节相同。
- 一篇以 transport-exhausted 结束且 `failOnTransportExhausted` 为真：其余在途调用被取消，已完成篇目的 checkpoint 仍在。
- 并发设为 1 时所有现有 summarizer/pipeline 测试不变。
