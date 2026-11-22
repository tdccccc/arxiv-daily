# 损坏的日报摘要检查点文件会让该日期陷入重试死循环、消耗 10 次 LLM 调用且插件内无法自愈

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（日报流水线与摘要检查点存储）
> 严重程度：高 · 置信度：已复现

## 问题与依据

如果某天的日报摘要检查点文件 `.index/daily-summary-checkpoints/<date>.json` 因断电、同步冲突或手工误改而损坏，且 `.bak` 不存在或同样损坏：

1. 读取路径静默忽略损坏：`DailySummaryCheckpointStore.lookupReusable()` / `load()` 返回空结果，流水线认为没有缓存，于是正常调用 LLM 为论文生成摘要。
2. 写入路径拒绝覆盖损坏文件：`packages/core/src/services/daily-summary-checkpoint-store.ts:353-380`（`loadForMutation`）在第 369 行**仅允许** `primary.kind === "missing" && backup.kind === "missing"` 创建空文档，遇到 `primary.kind === "corrupt"`（JSON 语法坏或 schema 错）时直接抛出 `DailySummaryCheckpointStoreError("cannot mutate unreadable daily summary checkpoint: ...")`。
3. 流水线将其判为临时失败并持续重跑：`packages/core/src/pipeline/pipeline.ts:626-633` 捕获该异常后，因不是永久 LLM 错误而返回 `failed_transient`。后续定时调度器会反复重试最多 10 次，**每次重试都重新发起 LLM 请求，然后在 `upsert` 写检查点时以同一个错误失败**，耗尽额度后把当天打成 `failed_permanent`。
4. 用户在插件内没有任何恢复入口：经全仓检索，`DailySummaryCheckpointStore.removeAll(date)` 只在 `pipeline.ts:238`（日报文件已存在）和 `pipeline.ts:649`（`writeDaily` 写完之后）被调用；**Clear run state**（`plugin/src/commands.ts:161-175`）和 **Force run** 都不会删除损坏的摘要检查点，用户只能手动到 Vault 隐藏目录 `.index/daily-summary-checkpoints/` 下删文件。

对比同仓的筛选检查点 `packages/core/src/services/daily-filter-checkpoint-store.ts:319-322`（`assertMutable`）：它明确区分了 `"unreadable"`（I/O 故障如 `EIO`，拒绝覆写以免覆盖真实文件）和 `"corrupt"`（文件内容已确认损坏，允许用新生成的结果覆盖恢复）。

## 原因

`DailySummaryCheckpointStore.loadForMutation()` 把 `corrupt`（已读出内容但解析/解码失败，与 `.bak` 均无效）和 `unreadable`（磁盘 I/O 临时故障）合并为同一种拒绝写入策略；同时 `pipeline` 的强制重跑和插件的清除状态命令都没有提供清理/重建损坏检查点的通道。

## 建议和验收

- 方案 A（推荐，与 `DailyFilterCheckpointStore` 一致）：保留对 `unreadable`（I/O 错如 `EIO`）的拒绝写入保护，但对确定无法解析/解码且无有效 `.bak` 的 `corrupt` 摘要检查点允许按空文档重建并覆写（写入 warning 日志，必要时先将坏文件改名备份或直接替换）。
- 方案 B（如必须维持 `DailySummaryCheckpointStore` 自身对双损坏拒绝写入的既有单测契约 `daily-summary-checkpoint-store.test.ts:682`）：则在 `summarizeDaily` / `pipeline` 捕获 `DailySummaryCheckpointStoreError` 时不要误记成会烧 LLM 的无限重试，至少在 `Force run` 入口或 `clearRunState` 中清理对应日期的检查点，并在首次失败时给出可操作的修复提示或自动清理损坏检查点后重试写入（不重调 LLM）。
- 验收：向 `.index/daily-summary-checkpoints/<date>.json` 写入非法 JSON（无 `.bak`），运行当天日报或 Force run 后能够正常生成日报，不陷入 10 次重复调用 LLM 的失败循环。

## 修复后应补的测试

- `packages/core/tests/pipeline/pipeline-error-handling.test.ts`（及 `daily-summary-checkpoint-store.test.ts`）：当 `daily-summary-checkpoints/<date>.json` 为损坏 JSON 且无有效 `.bak` 时，运行 `DailyPipeline` 能完成日报生成并清理检查点，不会重复消耗 LLM 调用或永久卡死该日期；同时保留 `EIO` 读取故障不覆写原文件的保护测试。
