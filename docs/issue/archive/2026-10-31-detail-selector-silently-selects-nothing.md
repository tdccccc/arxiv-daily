# 详细笔记选择器遇到不合格回复或一次网络失败就静默选零篇

状态：已修复（分支 `fix/2026-10-31-review`）

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：中 · 置信度：代码推断（现有测试锁定了当前行为）

## 问题与依据

`packages/core/src/pipeline/detail-selector.ts`：

- 107-120 行：模型调用失败（含传输重试耗尽）→ 只记 warn，返回空选择；
- 123-129 行：回复解析失败 → 只记 warn，返回空选择；
- 176-184 行：`parseEvaluations` 直接 `JSON.parse(raw)`，不容忍 ```json 围栏——筛选路径有 `unwrapSingleCodeFence`（`paper-filter.ts:483-487`），摘要路径靠带提示的重试，选择器两者都没有；
- 没有任何重试。

pipeline 随后照常写日报，`detailCount` 为 0，运行结果仍是 completed。`packages/core/tests/detail-selector.test.ts:242` 断言的就是这个"selecting no papers"日志。

用户看到的是：开了"自动生成详细笔记"的主题，某天一篇笔记也没有；日报、状态栏、运行历史都没有提示，只有开发者日志里一行 warn。对习惯把 JSON 包进代码围栏的模型来说，是永远没有笔记。

## 原因

选择器被设计成"失败不拖累日报"，但失败对用户不可见，也不尝试修复。

## 建议和验收

- 解析前容忍单层代码围栏（复用筛选的 unwrap）。
- 解析失败时带纠错提示重问一次；仍失败或传输失败时，把"笔记选择已跳过：原因"带进 `PipelineResult`（例如 completed 结果上的一个字段），由状态栏通知和运行历史显示；日报头部可选加一行说明。
- 日报仍然正常写出，不因选择器失败而失败。

## 修复后应补的测试

- 围栏包裹的合格回复被接受。
- 第一次无效、第二次有效：按第二次结果选择，`llm.call` 共 2 次。
- 两次都无效 / 传输失败：零篇，且结果携带可见的跳过原因。
- 现有 "selecting no papers" 测试改为断言跳过原因对调用方可见。

## 修复情况

详细笔记选择器现在接受单层 ```json 围栏；回复不合格时带规则说明重问一次；两次都不合格、或模型不可达时不再静默：运行结果带上"paper notes skipped: 原因"，Obsidian 的运行通知（`describeResult`）、调度器的完成提示和 CLI 的 stderr 都会显示它（工作台任务托盘显示 CLI 输出，因此也能看到）。日报本身照常写出。

测试：`packages/core/tests/detail-selector.test.ts`（围栏回复被接受；先无效后有效时重问一次并保留原请求；两次无效和传输失败都带可见原因）；新增 `pipeline-detail-selection-skipped.test.ts` 断言完成结果携带原因、日报仍写出，且选择器正常时没有该字段。
