# 2026-10-30 摘要质量评估的后续问题：索引

> 来源：摘要质量评估（[helm](../helm/2026-10-28-summary-quality-evaluation/goal.md)，PR #144，合并提交 848c5a9）执行中发现、当时没有处理的问题
> 结果：4 条，未修复 3 条、待办 1 条

每条问题一个文件，结构与 `archive/` 相同。修复后按惯例移入 `archive/`，并在文件末尾补"修复情况"。

编号前缀 SQE 表示源自这次摘要质量评估；“范围”列说明问题所在：评测工具、日报流水线（core）或筛选评估。

| 编号 | 范围 | 状态 | 严重程度 | 问题 | 文件 |
|---|---|---|---|---|---|
| SQE-01 | 评测工具 | 未修复 | 中 | 评审提示词的"小问题"定义与"完整"定义自相矛盾，评审把次要遗漏也报成问题（复核不同意 33/74，估计评审问题约 61% 成立） | [judge-prompt-minor-severity-contradiction](2026-10-30-judge-prompt-minor-severity-contradiction.md) |
| SQE-02 | 日报流水线 | 未修复 | 低到中 | 公式校验把比例 `$6:1$` 和半开区间 `$(0, 0.15]$` 判为格式错误，日报摘要因此重试甚至回退 | [math-validator-rejects-ratios-and-half-open-intervals](2026-10-30-math-validator-rejects-ratios-and-half-open-intervals.md) |
| SQE-03 | 日报流水线 | 未修复 | 低到中 | 详细笔记生成后不做公式校验，`\(..\)`、拆开的公式等直接写进 Vault（旧笔记 46 篇中 7 篇有问题） | [detail-notes-skip-math-validation](2026-10-30-detail-notes-skip-math-validation.md) |
| SQE-04 | 筛选评估 | 待办 | 中 | 筛选质量基线只有 3 篇"该选"的标注，精确率数字不可靠；等 2–3 周筛选记录积累后补标注 | [filter-eval-needs-more-positive-labels](2026-10-30-filter-eval-needs-more-positive-labels.md) |

## 相关但不在本索引中

另一个会话 2026-10-29 的 [测试与评测审阅](2026-10-29-tests-and-llm-evaluation-review.md) 对同一批评测工具提出了 T01–T10。其中两条已复现，会影响评测结论，应在下次用这些工具做对比实验之前处理：

- **T01**：生成和评审续跑时不比较推理设置，只改 `reasoningEffort` 或 `thinkingMode` 时会直接复用旧结果。
- **T02**：评审报告读取输出目录下的全部记录，同一目录先大样本、后小样本时会混入上次的结果。

T01–T10 以该文档为准，本索引不重复记录。
