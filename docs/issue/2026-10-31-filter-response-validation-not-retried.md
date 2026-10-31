# 筛选模型一次回复格式不对，当天日报直接记为临时失败且不立即重问

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：中 · 置信度：代码推断（现有测试锁定了当前行为）

## 问题与依据

`packages/core/src/pipeline/paper-filter.ts:452-476` 的 `decodePaperFilterResponse` 在回复不是严格 JSON、或违反筛选合同（多一个字段、`relevanceScore` 写成字符串、未知 id、重复 id……）时抛 `PaperFilterResponseValidationError`；`classifyRefusingProvider`（第 419 行）不捕获，`filterPapers` 向上抛，`packages/core/src/pipeline/pipeline.ts:307-312` 把它转成 `failed_transient: paper filter response validation failed`。

调度侧的后果：`packages/core/src/services/scheduling/run-gate.ts:42-52` 让这一天等满一个 tick 周期才重试；`packages/core/src/services/state-store.ts:6,209-212` 把它计入 10 次临时失败配额之一；筛选 checkpoint 只在解码成功后才保存（`paper-filter.ts:334-348`），所以重试时整批（最多 40 万字符）重发。LLM 客户端没有请求 JSON 模式（`packages/core/src/llm/client.ts` 没有 `response_format`）。

对比：每篇摘要有 3 次带纠错提示的重试（`packages/core/src/pipeline/daily-paper-summary.ts:222-298`）；筛选一次都没有。`packages/core/tests/paper-filter.test.ts:102-172,330-331` 锁定了"一次失败即抛"。

用户看到的是：Obsidian 日历/状态栏当天显示"临时失败：paper filter response validation failed"，一个 tick 后自动再来；模型越爱出格式小错，越容易逼近永久失败，而且每次都是整批重发。

## 原因

筛选路径没有像摘要路径那样的"失败 → 受信纠错提示 → 重问"环节。

## 建议和验收

- 解码失败后立即用同一请求加一条简短纠错提示重问一次（必要时两次）：提示只说明违反了哪条合同（复用 `trustedCorrectionReason` 的做法，不带模型原文），仍失败才抛。
- metrics 记录全部调用；被服务商拒绝（content blocked）的路径保持不变。
- 分批时每批独立重问。

## 修复后应补的测试

- 第一次 `invalid-json` / `invalid-contract`、第二次合格：采用第二次结果，`llm.call` 共 2 次，checkpoint 保存。
- 两次都不合格：仍抛 `PaperFilterResponseValidationError`，pipeline 仍返回 `failed_transient`。
- 纠错提示不包含第一次回复的原文。
- 多批场景下只有失败的那一批被重问。
