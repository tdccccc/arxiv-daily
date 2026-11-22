# 零命中日报完成后遗漏清理检查点文件，且流水线在摘要成功前就写入 papers.json 残留孤立条目

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（日报流水线状态清理与索引一致性）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

`packages/core/src/pipeline/pipeline.ts` 中存在两处状态清理/写入顺序遗漏：

1. **零命中工作日完成时不清理该日期残留的检查点文件**：
   - 正常有命中论文的分支在写完日报后（`pipeline.ts:649`）以及日报已存在跳过分支（`pipeline.ts:238`）都会调用 `await this.cleanupCommittedCheckpoints(dateStr)` 清理 `.index/daily-filter-checkpoints/<date>.json` 和 `.index/daily-summary-checkpoints/<date>.json`。
   - 但在零命中完成分支（`pipeline.ts:332-349` 无论文返回、`pipeline.ts:374-397` 筛选后无命中写空日报）调用 `writeEmptyDaily(dateStr)` 并返回 `{ kind: "completed", papersWritten: 0 }` 之前，**遗漏了 `await this.cleanupCommittedCheckpoints(dateStr)`**。
   - 若某天第一次按较宽主题跑筛选中途失败留下检查点、随后用户调整主题重跑变为 0 篇并完成，该日期的旧检查点文件会永久残留在 Vault `.index/` 下。
2. **命中论文在摘要阶段之前就被写入 `papers.json`，失败或取消会残留无摘要无日报链接的孤立条目**：
   - `packages/core/src/pipeline/pipeline.ts:407-413` 在步骤 5（筛选后、正文抓取和摘要生成之前）就调用 `paperIndex.recordDailyMatch(...)` 把当天选中的论文写入 `papers.json`。
   - 若后续步骤（正文抓取、精读生成、日报摘要 LLM）失败或被用户中途取消，日报文件并未写出，但这些论文已经留在 `papers.json` 中，且既没有 `summary` 也没有 `dailyReports` 链接；如果用户随后调紧主题或放弃该天重跑，Dashboard "All" 列表会长期展示这些无摘要、无日报来源的孤立条目（与已修复的 `PIPE-04` 现象类似）。

## 原因

- `cleanupCommittedCheckpoints` 补加时只覆盖了主完成路径和“日报已存在”早退路径，漏掉了两条零命中完成路径。
- `paperIndex.recordDailyMatch` 放在筛选后立刻持久化，而关联的 `addDailyReports` 和 `setSummaries` 都在 `writeDaily` 之后（第 653-670 行）才写入，两阶段未对齐到同一个提交点。

## 建议和验收

- 在两条零命中完成分支（`writeEmptyDaily` 写完之后、返回 `completed` 之前）补上 `await this.cleanupCommittedCheckpoints(dateStr)`。
- 评估将 `paperIndex.recordDailyMatch` 延后到 `writeDaily` 成功之后与 `addDailyReports` / `setSummaries` 合并提交（注意精读笔记步骤 `writePaperDetail` 仍可直接从内存候选对象构造所需 `indexEntry` 视图，仅在 `confirmedPath` 落盘后再持久化已生成的精读笔记条目）。
- 验收：
  - 先给某日期写入残留的筛选/摘要检查点，随后触发 0 命中完成，断言 `.index/daily-*-checkpoints/<date>.json` 均已清除；
  - 模拟摘要阶段报错/取消，断言未落盘精读且未出日报的论文不会以无摘要孤立条目污染 `papers.json`。

## 修复后应补的测试

- `packages/core/tests/pipeline.test.ts`：
  - 预置某日期的筛选与摘要检查点后运行 0 命中流程，断言检查点文件被 `cleanupCommittedCheckpoints` 清除；
  - 模拟 `summarizeDaily` 抛错或取消，断言 `papers.json` 不新增无 `dailyReports` 且无 `summary` 的孤立条目。
