# 被每日上限截掉的论文也进了论文索引，在 Dashboard 和工作台里当普通条目出现

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

`packages/core/src/pipeline/pipeline.ts:340` 把全部通过筛选的论文 upsert 进论文索引，之后 352-355 行才按 `maxDailyPapers` 截取可见集合；594-597 行只给可见论文挂日报路径。`packages/core/src/services/paper-index.ts:885-887` 给新条目 `status: inbox`、`seenDates` 含当天。于是被截掉的论文在索引里是：无摘要、`dailyReports` 为空、看起来和日报里的论文一样。

- Dashboard 的 "All" 页列出所有非 ignored 条目（`packages/core/src/dashboard/model.ts:197-199`）；"本周新增" 按 `seenDates` 统计（221、537 行）；
- 工作台 inbox 计数（`apps/cli/src/workbench/papers.ts:44`）同样包含它们；
- `packages/core/tests/pipeline-daily-cap.test.ts:139-140` 断言了这一存储行为，说明存储层是有意为之，但界面没有相应处理。

用户看到的是：日报只写 "N 篇因每日上限省略"，而 Dashboard 里能翻到这些论文，没有摘要、没有日报链接、没有任何"被省略"的标记，像是坏掉的条目。

## 原因

索引 upsert 发生在截取之前；条目上没有"因上限省略"的信息。

## 建议和验收

二选一：

- 只索引可见论文（筛选决策记录已保留 `cut_by_limit`，评估不受影响）；
- 保留但打标记（如记录被省略的日期），Dashboard / 工作台显示"于某日因上限省略"，不计入本周新增。

## 修复后应补的测试

- 更新 `pipeline-daily-cap` 测试以断言所选方案。
- 若保留方案：Dashboard model 测试中被截论文不计入 `weekAdded`，行上有标记。
