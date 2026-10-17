# 重跑报告的精确率和 F1 只按样本内计算，会严重高估新提示词或新模型的精确率

> 发现日期：2026-10-17 · 来源：main `e6eaed5`（#93–#95 合并后）代码审查，质量评估脚本部分由子代理审查、主会话核对
>
> 范围：开发者工具 `scripts/quality-eval/`，不影响产品 · 严重程度：中 · 状态：未修复 · 置信度：已确认（下文例子用 `metrics.ts` 的实际代码核算；报告渲染代码已逐行核对）

## 问题

导出标注表时，被模型判为不相关（`skip`）的论文只随机抽一部分进表，每个抽中的行代表 `skippedWeight` 篇（被判不相关的总数 ÷ 已标注的跳过层行数）；被选中的论文全部进表，权重为 1。

- 对**记录的判定**，被选中的行全部来自选中层，权重都是 1，所以按行直接计数的"样本内精确率"没有偏差。
- 对**重跑**，新模型可能选中跳过层的抽样行。这些行在样本内各算 1 次，实际各代表几十篇论文，样本内精确率因此严重偏高。

`metricSet` 已经算出了加权精确率和加权 F1，但 `report.md` 里看不到，只写进了 `report.json`（`runs[].sets.all.weighted`）：

- "记录的判定与重跑对比"表（`report.ts:210-216`）的四个指标列是：样本内精确率、样本内召回率、样本内 F1、加权召回率。
- 每次重跑小节的"主要结果"表（`report.ts:55-58`，经 `report.ts:252-255` 调用）精确率列是样本内的，最后一列是"加权召回率（F1 加权 F1）"。同一行里样本内精确率和加权 F1 放在一起，数字对不上。
- "不同分数线下的结果"表（`report.ts:89-91`）的精确率也是样本内的。

这正好会误导 README 推荐的用法：改动前后各重跑一次，再"看精确率、召回率的变化"（`scripts/quality-eval/README.md:140-142`）。一个让模型多选的改动，例如为了提高召回，在报告里会表现为"召回大涨、精确率小降"，实际精确率可能已经大幅下降。

## 原因

- `scripts/quality-eval/metrics.ts:114-118` 的 `weighted()` 计算加权的 `precision`、`recall`、`f1`；`metrics.ts:135-147` 的 `metricSet` 同时返回样本内和加权两组结果。
- `scripts/quality-eval/rerun.ts:316-318` 的 `comparisonOf` 只把 `weighted.recall` 放进 `Comparison.weightedRecall`，丢掉了加权精确率和加权 F1。
- `scripts/quality-eval/report.ts:21-23` 的 `weightedRecall()` 只渲染加权召回率和加权 F1；`report.ts:57-58`、`report.ts:91`、`report.ts:216` 的精确率列都取样本内的 `precision`。
- README 只说明了召回率需要加权（`README.md:126`），没有提到重跑时精确率也需要。

## 复现

例子，数字由 `metricSet` 实际算出：

- 选中层 40 行，其中 30 行标 `y`。
- 跳过层共 400 篇（`labels.meta.json` 的 `counts.skippedAvailable`），抽 10 行标注，其中 1 行标 `y`，所以 `skippedWeight = 400 / 10 = 40`。
- 新模型保留全部 40 行，并从 10 个跳过层样本里选中 5 行，包括那 1 行 `y`。

| 来源 | 样本内精确率 | 样本内召回率 | 样本内 F1 | 加权精确率 | 加权召回率 | 加权 F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 记录的判定 | 75.0%（30/40） | 96.8%（30/31） | 84.5% | 75.0% | 42.9% | 54.5% |
| 重跑 | 68.9%（31/45） | 100.0%（31/31） | 81.6% | **29.2%** | 100.0% | 45.2% |

`report.md` 的对比表里，重跑一行显示精确率 68.9%、F1 81.6%、加权召回 100.0%；重跑小节的主要结果表显示精确率 68.9%，最后一列是"100.0%（F1 45.2%）"。29.2% 只在 `report.json` 里。读者会以为"召回从 43% 升到 100%，精确率只从 75% 降到 69%"，实际精确率从 75% 降到了 29%。

复算片段（文件放在仓库根目录，用 esbuild 打包后以 node 运行，输出 `0.6888… 0.2916…`）：

```ts
import { metricSet, type LabelledRow } from "./scripts/quality-eval/metrics";
const base = { date: "2026-10-14", title: "", abstract: "", hadAbstractAtFilter: true, modelDirections: [], modelScore: 0, modelOutcome: "", labelDirections: [], note: "" };
const rows = [
  ...Array.from({ length: 40 }, (_, i) => ({ ...base, row: i + 2, id: `s${i}`, modelCategory: "t", label: i < 30 ? "y" : "n", sampledAs: "selected" })),
  ...Array.from({ length: 10 }, (_, i) => ({ ...base, row: i + 42, id: `k${i}`, modelCategory: i < 5 ? "t" : "skip", label: i === 0 ? "y" : "n", sampledAs: "skipped" })),
] as LabelledRow[];
const set = metricSet(rows, 400);
console.log(set.precision.value, set.weighted?.precision.value);
```

## 可能的解决方案

1. **推荐：在报告里直接给出加权精确率和加权 F1。** `rerun.ts` 的 `Comparison` 改为携带完整的加权结果（或新增 `weightedPrecision`、`weightedF1`），`comparisonOf` 一并传入；`report.ts` 的对比表、主要结果表和分数线表都增加"精确率估计（加权）"列，或把最后一列改成"加权估计：精确率 / 召回率 / F1"。记录的判定里加权精确率等于样本内精确率，照常显示即可。同步更新 `README.md:123-126` 和 `README.md:142`，说明比较重跑结果时应看加权数字。
2. 保留现有列，把表头改成"精确率（样本内）""F1（样本内）"；某次重跑选中了跳过层行时，在报告开头加醒目提示，写出选中了多少跳过层行、加权精确率是多少。改动较小，但读者仍需自己判断该看哪一列。
3. 对比表全部改用加权口径（精确率、召回率、F1 都加权），样本内数字只留在每次重跑的小节里。对比表更直观，但需要在表头写清口径，避免与主要结果表混淆。

## 修复后应补的测试

- `scripts/tests/quality-eval-rerun.test.mjs`：现有用例（第 171-185 行）已经断言了 `weighted.precision` 的数值，但没有检查 `report.md`。补一条：新模型选中跳过层行时，`report.md` 的对比表和重跑小节都显示加权精确率，并与 `report.json` 一致。
- `scripts/tests/quality-eval-evaluate.test.mjs`：记录的判定报告里，加权精确率与样本内精确率相等。
