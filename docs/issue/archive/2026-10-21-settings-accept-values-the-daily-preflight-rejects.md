# 设置页允许保存重名主题、超过 1000 字的研究方向、超过 120 字的主题标签，但日报要到模型过滤和详情笔记都做完之后才拒绝，整天反复失败直到永久失败

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 PIPE-06
> 范围：日报流水线（插件、CLI、工作台共用）与设置校验 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch2`，提交 d6afc4a、45d4d6e） · 置信度：已复现

## 问题

日报组装前有一道预检（`preflightDailySummaryPapers`），对主题和论文做严格检查。设置页和 CLI 配置读取并没有同样的限制，导致下面这些“合法配置”会让整天的日报失败：

1. **两个主题显示名相同**（标签不同）。设置页专门有 `uniqueTopicTag`，重名主题会自动得到 `photo-z`、`photo-z-2` 这样的不同标签，说明重名是被预期的；`validateFilterConfig` 只查标签重复，不查名称重复。运行时预检抛出 `duplicate topic name: Photo-z`。
2. **某条研究方向超过 1000 个字符。** 设置页和工作台的方向输入框都没有长度限制，保存时最大允许 20000 字（`text()` 校验）。命中这条方向的论文在预检里被判为 `invalid topic directions`，因为命中记录最多允许 1000 字。库提议审阅界面却限制了 1000 字，所以只有手写方向会碰到。
3. **主题标签超过 120 字符**（例如把一整句话当主题名，标签由名称自动推导，没有长度上限）：同样被预检判为 `invalid topic directions`。

失败发生的位置太靠后：这些检查都在 `summarizeDaily` 开头，也就是过滤模型调用、详情选择、详情笔记生成都已经花完钱之后。复现（一次性测试，已删除）：

- 方向 1000 字：正常完成；1001 字和 3000 字：`{"kind":"failed_transient","reason":"daily summary failed: preflightDailySummaryAssembly: paper 2605.08080 has invalid topic directions"}`，没有写日报；
- 标签 120 字符：正常；121 字符：同样失败；
- 两个主题都叫 `Photo-z`：`reason: "daily summary failed: preflightDailySummaryAssembly: duplicate topic name: Photo-z"`。

用户看到的是“暂时失败”，调度器每个 tick 都会重试，十次后升级为永久失败（之后不再自动运行）。方向超长的报错没有说是哪条方向、为什么，用户很难自己找到原因。

## 原因

- `packages/core/src/pipeline/daily-summary-assembler.ts:150-161`：预检要求主题标签、名称都唯一。
- `packages/core/src/pipeline/topic-direction-hits.ts:10-13,28-49`：命中记录的标签 ≤120、编号 ≤128、方向文本 ≤1000、条数 ≤256。
- `packages/core/src/settings/validation.ts:107-124` `validateFilterConfig`：只检查名称非空、标签非空且不重复、方向非空；`packages/core/src/settings/editing.ts:40-47,108-111`：方向文本和标签最长 20000 字；命令行配置读取（`apps/cli/src/config.ts:333-361`）没有这些检查。
- `packages/core/src/pipeline/summarizer.ts:136`：预检在 `summarizeDaily` 内才执行；`packages/core/src/pipeline/pipeline.ts:266`（过滤）、`:457-514`（详情笔记）都在它前面。

## 可能的解决方案

推荐：

1. 把预检会拒绝的条件前移到 `validateFilterConfig`（所有宿主运行前都会调用）：名称重复、方向超过 1000 字、标签超过 120 字，并在原因里带上具体的主题名和方向序号。
2. 设置页输入框和保存路径同步限制长度（方向 1000 字、标签 120 字），或者保存时给出明确提示；标签自动推导时截断到 120 字。
3. 流水线在调用模型之前就对主题和方向跑一遍同样的检查，失败时返回带原因的永久配置错误，而不是“暂时失败”。

备选：放宽预检（方向超长时在报告里截断显示、重名主题用标签区分），两边限制保持一致。

## 修复后应补的测试

- `validateFilterConfig` 对重名主题、1001 字方向、121 字符标签返回明确原因。
- 这三种配置下流水线在任何模型调用之前就失败，并且不计入暂时失败次数。
- 合法的 1000 字方向、120 字符标签继续正常出报告。

## 修复情况

设置页和命令行现在在保存的地方就拒绝日报装不下的值：研究方向超过 1000 字、主题标签超过 120 字，会给出带主题名和方向序号的原因；插件和工作台的方向输入框限制为 1000 字，命令行 `init` 向导对过长的标签和描述重新提问；由很长的主题名推导标签时截到 100 字，仍留出重名后缀的位置。已经保存了这类值的旧配置不会被悄悄改动：运行前的检查（`validateFilterConfig`）会列出具体问题，例如"主题 1 和 3 都叫 "Photo-z""、"主题 2 "X" 的第 3 条方向有 1001 字，上限 1000"，命令行因此退出码为 2，插件和工作台显示为不可运行。即使仍有配置到了流水线，它也在列表读完之后、第一次调用模型之前返回永久失败并带同样的原因，不再花完过滤和详情的模型调用才失败，也不再被当作暂时失败重试十次。

取舍：重名的主题仍可以保存（设置页本来就会给重名的主题自动分配不同的标签），改为在运行前报告，而不是在保存时拒绝，避免打断正在编辑的草稿。没有方向的主题仍照旧：过滤阶段跳过它，不算错误。

测试：`validation.test.ts`、`settings-editing.test.ts`、`settings-topics.test.ts`、`pipeline/topic-config-preflight.test.ts`（三种配置都在任何模型调用之前失败）、插件 `settings-declarative-tab.test.ts`、工作台 `workbench-settings-ui.test.ts`、命令行 `cli-main.test.ts`（旧配置运行前报错）和 `cli-init.test.ts`。
