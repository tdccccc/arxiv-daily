# 被服务商内容安全拒绝的论文在筛选决策记录里被记成"模型跳过"

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

`packages/core/src/pipeline/paper-filter.ts:398-405`：服务商拒绝一篇单独的论文后返回空结果；321-333 行只记日志。这些 id 不出现在分类记录里，`packages/core/src/pipeline/pipeline.ts:108-109,713` 用 `omittedByModel` 补齐，outcome 记为 `skipped_by_model`。

筛选质量评估（`scripts/quality-eval`，SQE-04 正在收集正例标注）读这份记录时会把它们当成"模型判为不相关"，但模型根本没见过这些论文。

## 原因

拒绝信息没有进入 `FilterClassification`，决策记录也没有对应的 outcome。

## 建议和验收

- `FilterClassification` 增加被拒 id 列表；决策记录 outcome 增加 `refused_by_provider`。
- 评估脚本把该 outcome 排除在精确率/召回率之外，另行计数。

## 修复后应补的测试

- `paper-filter-refusal` 测试：`onClassified` 收到被拒 id。
- `pipeline-filter-decision-record` 测试：被拒论文的 outcome 为 `refused_by_provider`。
- quality-eval 评分测试：该 outcome 不计入正负例。
