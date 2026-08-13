# P1 — P0 修复：选题偏置、搜索模型校验、索引异常隔离

goal_ref: ../goal.md
updated: 2026-08-13

## Outcome

三项 P0 缺陷修复落地：方向生成选题按论文发布日新到旧参与（确定性），搜索入口拒绝跨模型静默错排，单篇损坏文档不再中止整轮索引或检索且旧文档不残留孤儿——每项带测试，全量回归绿。

## Assumptions

- arXiv paperKey 的 `YYMM` 前缀与 catalog `published` 字段同为时间信号；选题与聚类输入排序用 `published`（缺失时回退 paperKey 解析），保证确定性。
- KB manifest 顶层已存 `modelId`（索引入口已校验过），搜索入口只需在加载 manifest 后比较 `embedding.modelId`。
- "重索引失败覆盖 ready 记录"维持现语义（宁可让论文暂时退出检索，不保留陈旧向量），本次固化文档与测试，并在覆盖时清理旧 paper 文档（孤儿）。
- 异常隔离是局部 try/catch 级别修复，不需要动 store 接口。

## Approach

三个独立小改动，各自 Red → Green → 回归：
1. 选题排序：`selectPersonalLibraryDirectionPapers` 与 `buildClusteringInput` 按发布日新到旧（确定性决胜键），替代 paperKey 升序。
2. 搜索校验：在 `searchFullTextKnowledgeBase` 入口比对 manifest modelId 与 `embedding.modelId`，不一致抛明确错误（含重建指引），维度检查保留。
3. 异常隔离：`indexPersonalLibraryFullText` 复用/迁移路径的 loadPaper/savePaper 纳入 per-paper 失败处理；检索循环跳过损坏文档；`recordFailed` 覆盖 ready 时删除旧文档文件。

## Test strategy

- change kind: bug fix
- strategy: strict Red-Green-Refactor（每个任务一个行为块）
- Red / baseline signal:
  - T1：`npx vitest run tests/clustered-direction-proposer.test.ts` 期望 Red：新用例断言"超上限时保留最新发布论文"失败（现为最旧优先）。
  - T2：`npx vitest run tests/fulltext-retrieval.test.ts` 期望 Red：新用例断言"manifest modelId 与当前模型不一致抛错"失败（现静默计算）。
  - T3：`npx vitest run tests/fulltext-index-orchestration.test.ts` 期望 Red：新用例断言"单篇损坏文档不中止整轮"失败（现整轮抛错）。
- Green / regression checks: 各任务聚焦文件全绿后跑 core 全量（8 GiB 单 fork）、plugin 全量、CLI 全量、双 typecheck、boundaries。
- exception: 无。

## Tasks

- [x] T1 选题排序（change kind: bug fix）—— `selectPersonalLibraryDirectionPapers`（personal-library-direction-proposer.ts:180-188）与 `buildClusteringInput`（clustering/paper-vector.ts:35）改按发布日新到旧；测试覆盖超上限裁剪与聚类输入顺序。
- [x] T2 搜索 modelId 校验（change kind: bug fix）—— 搜索入口比对 manifest modelId 与当前嵌入模型，不一致抛错并附重建指引；测试覆盖同维不同模型与远程/本地切换场景。
- [x] T3 索引异常隔离与孤儿清理（change kind: bug fix）—— 复用/迁移路径 per-paper 隔离、检索循环跳过损坏文档、failed 覆盖 ready 时删除旧文档；测试固化覆盖语义；technical-report 记录边界更新。

## Verification

- core 聚焦测试全绿；core 全量 96 文件全绿；plugin 34 文件全绿；CLI 全绿。
- core/plugin/CLI/node-runtime typecheck、`npm run check:boundaries` 通过。
- lint 0 errors、warnings 不高于基线 64。

## Abort / reshape triggers

- 若 `published` 字段在真实 catalog 中大量缺失且 paperKey 解析不可靠，L1 改回退策略（如按 updated 或随机确定性抽样）。
- 若搜索入口拿不到 manifest modelId（入口设计变化），L1 调整校验位置到检索函数内。
- 若异常隔离需要重构 store 接口或编排签名，L2 reshape（停止局部修补，重新设计 per-paper 错误路径）。
