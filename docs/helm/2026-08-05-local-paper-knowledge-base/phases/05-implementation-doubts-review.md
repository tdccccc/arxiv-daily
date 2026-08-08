# P5 — 实现存疑项复核

goal_ref: ../goal.md
updated: 2026-08-07

## Outcome

P3 T5b 报告遗留的实现存疑项全部有代码核对的定论；可修复项已修复并通过全量验收；结论与 ADR 0007 的授权门拆分设计对齐。存疑项 1（建议文档整体替换 vs 合并）已由 ADR 0007 关闭（保持替换 + 覆盖未审阅提示），不在此阶段重复。

## Assumptions

- 镜像实现（plugin `centerChunksInPlace`）与 core 私有 centering 变换同构（P3 时按"同构"注释）；需实测核对，若漂移则以 core 为准并修正镜像路径。
- 增量更新复用 `personal-library-direction-generation` operation kind 是封闭 union 下的有意复用（共享授权门与撤销范围），不是疏漏。
- decoder 对 discoveryCues 的 ≥1 约束是生成契约的一部分，"new" 候选以 reason 截断单条填充是当时的最小满足方案。

## Approach

逐项做"代码核对 → 定论（修复 or 保持+记录理由）→ 修复项走测试/验收/technical-report handoff/提交"。T1 先核对两处 centering 实现是否数值同构，同构则 core 导出变换、plugin 切换共用（行为不变的纯重构）；T2 核对 new 候选构建路径，选择填充源改进或 decoder 放宽；T3 核对 operation kind 复用的授权门/撤销/取消语义，给出与 ADR 0007（placement 免许可、LLM 部分要许可）的衔接结论。

## Tasks

- [x] T1 存疑项 4（centering 镜像）：对比 plugin `centerChunksInPlace`/`normalizedChunkInPlace` 与 core 聚类 centering 实现（数值同构性）→ core 导出共用变换 → plugin 切换 → 行为不变验证（e2e 数值对比或等价测试）
  - 结论：两实现逐行同构（Float64 均值、同迭代序、减均值重归一、零范数拷贝）；core 的"先归一化"在 KB 单位向量上近似恒等。core 新增非可变 `centerCorpusChunks` 并接入内部流水线，plugin `loadCenteredClusteringInput` 切换共用，镜像函数删除；+4 core 测试；scratch 逐位等价验证（真实 KB 5 篇 187,776 floats，max diff = 0，BIT-IDENTICAL）；technical-report handoff no-impact。
- [ ] T2 存疑项 3（discoveryCues 截断）：定位 "new" 候选 `discoveryCues` 填充路径（reason 截断单条）→ 定修法（填充源 vs decoder 放宽，评估其它路径影响）→ 修复 + 测试
- [ ] T3 存疑项 2（operation kind 复用）：核对增量更新与全量生成共用 `personal-library-direction-generation` 的授权门/撤销范围/取消语义 → 定论（保持复用 or 新 kind）→ 记录与 ADR 0007 拆分授权门的衔接
- [ ] 收尾：每个被接受 chunk 的 technical-report handoff；core 全量（`NODE_OPTIONS=--max-old-space-size=8192 npx vitest run`，1528+）+ plugin 全量（420+）+ lint + boundaries；每阶段提交；goal.md P5 done + status done

## Verification

- T1：core 导出后 plugin 编译通过；切换后既有增量 e2e 场景结果不变（或新增数值等价测试）；core/plugin 全量测试通过。
- T2：修复后 "new" 建议的 discoveryCues 内容合理；decoder 若放宽不影响方向生成其它路径（相关测试全过）。
- T3：结论记录（保持/新增 kind），含对 ADR 0007 授权门拆分的实现提示。
- 每个修复 chunk：对应测试 + tsc + lint + boundaries；technical-report handoff 到 `updated` 或 `no-impact`。

## Abort / reshape triggers

- 若镜像与 core centering 数值不一致（T1 核对失败）：L2——以 core 为准修正，切换前先修 core 或明确差异来源，不强行合并。
- 若 decoder 放宽 discoveryCues 约束破坏生成契约（T2）：回退为"改进填充源"方案。
- 若 operation kind 结论需要新增 kind（T3）：与 ADR 0007 冲突时先向用户简报再定。
