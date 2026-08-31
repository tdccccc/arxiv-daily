# P1 — cross-cluster-synthesis

goal_ref: ../goal.md
created: 2026-08-31T22:41:07+08:00
updated: 2026-08-31T22:52:00+08:00
revision: 2

## Outcome

聚类式方向提案在落盘前跑一次跨簇综合：讲同一件事的候选被合并成一条，研究者看到的是综合后的集合，而不是各簇结果的拼接。

## Assumptions

- **综合机器可直接复用**。渲染消息、system prompt、带重试的校验、大小上限、`synthesis-too-large` 都已存在于不带聚类的那条路径，聚类路径只是没调用它们。若复用时发现校验假设与聚类路径不兼容（例如允许的 paperKey 集合口径不同），这条假设即失效。
- **综合失败时退回未综合的候选，而不是让整个提案失败**。此时已经为每个簇各花了一次抽取调用，因为一次综合调用连续校验失败就丢掉全部产出，会让研究者什么都拿不到；碎片化的提案仍然可用。**这与不带聚类的老路径不同**（那里 `synthesis-too-large` 直接抛错），差异是有意的，需在实现时写进注释与测试。若用户认为宁可失败也不要碎片提案，这条假设即失效。
- **候选 ID 在综合之后分配**，`lineage.candidateIds` 仍为自指，与老路径一致。综合前的模型候选没有 ID，不构成可追溯的前身。
- **`clusterMembers` 在合并时取并集**。schema 对成员数有上限，超限时按聚类置信度截断保留最高的若干条。若并集频繁超限，说明这个字段的语义在综合后站不住，需要重新考虑而不是继续截断。
- 证据单薄的标注与「一键全接默认不勾」属于 P2 的复审页，不在本阶段。本阶段不新增 schema 字段——「单薄」可由代表论文数在展示时推导。

## Approach

在 `proposeClusteredPersonalLibraryDirections` 的簇循环之后、构造 proposal 之前插入综合阶段，复用老路径的 `canonicalizeSynthesisInput` → `renderPersonalLibrarySynthesisUserMessage` → `callValidatedStage("synthesis", …)`。允许的最终 paperKey 集合取「综合输入里出现过的代表论文」与「聚类输入论文」的交集，与老路径同法。综合返回后再分配候选 ID 与证据指纹，`clusterMembers` 按来源候选取并集。

## Chunks

### Chunk 1 — 聚类提案跑综合阶段，合并后的候选携带来源簇成员的并集

> **L1 就地调整（2026-08-31）**：原计划把「跑综合」与「簇成员归谁」拆成两个 chunk。动手时发现拆不开——综合一旦合并候选，`clusterMembers` 必须同时有定义，否则提案通不过严格解码，第一个 chunk 无法独立验收。两者合为一个 chunk，阶段目标与后续 chunk 不变。

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增一条测试——两个簇各产出讲同一件事的候选，脚本化 LLM 在第三次调用（综合）返回合并后的单条；断言最终 proposal 只有 1 个候选。当前实现不发起第三次调用，红在「候选数为 2，期望 1」。同时现有断言 `expect(llm.calls).toHaveLength(2)` 会红在调用数上——那条断言编码的是「只有抽取、没有综合」的旧契约，属预期变红。
- Green check: `npm run test --workspace @arxiv-daily/core -- clustered-direction-proposer`，新测试转绿；「每簇恰好一次抽取且不串簇」这条契约仍绿（它仍然成立，只是总调用数从 2 变 3）。
- regression checks: `npm run test --workspace @arxiv-daily/core -- personal-library-direction-proposer`（老路径综合行为不得改变）；`npm run typecheck`。
- **observed**: 红的原文 `expected [ … ] to have a length of 2 but got 3`（新测试先红在调用数为 2）。绿之后 clustered-direction-proposer 14/14、老路径 23/23 逐字未变、core 各分片全绿、plugin 694/694、typecheck 四包、lint 0 error、check:boundaries OK。合并的簇成员规则：取代表论文所属各簇成员的并集（簇是划分，不重复计数），超 schema 上限时按置信度保留最高的若干条。
- **顺带更新的既有断言，均非契约变更**：`clustered-direction-proposer` 里「每簇一次抽取」那条的总调用数 2 → 3，判据限定到抽取调用；`plugin/tests/personal-library-profile-lifecycle` 两条把「生成方向 = 恰好一次 LLM 调用」当成脚手架常量，改为 2 次（抽取 + 综合）。授权闸、CAS、证据陈旧检测的判据一字未动。
- exception: 无
- [x] implementation and tests accepted

### Chunk 2 — 综合失败或过大时退回未综合的候选集，不丢弃整个提案

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增测试——脚本化 LLM 在综合调用上连续返回不合法输出直到重试耗尽，断言仍返回一个严格 decode 通过、含全部未综合候选的 proposal；另一条用例让综合输入超过 `PERSONAL_LIBRARY_DIRECTION_MAX_SYNTHESIS_CODE_UNITS`，断言同样退回而不是抛 `synthesis-too-large`。红在抛错。
- Green check: 同上测试文件转绿。
- regression checks: 取消语义不变——综合阶段前后各有一次 `throwIfCancelled`，已有取消测试仍绿；`npm run test --workspace @arxiv-daily/core -- personal-library-direction-proposer` 确认老路径仍然抛错（差异是有意的）。
- exception: 无
- [ ] implementation and tests accepted

## Phase verification

- `npm run test --workspace @arxiv-daily/core`（core 全绿；本机需 8 GiB 堆 + 单 fork）。
- `npm run typecheck` 四包通过；`npm run lint` 0 error；`npm run check:boundaries` OK。
- 授权面未变：综合仍在 `personal-library-direction-generation` 的授权与取消范围内，未新增 consent 分支——由既有授权测试仍绿佐证。
- **不构成本阶段验收的**：22 → 6–8 的真实收敛。那需要对测试库重跑一次方向生成，依赖 LLM 端点可达，属 P5 的端到端验证。

## Abort / reshape triggers

- 若综合的校验器要求「覆盖全部输入论文」这类聚类路径无法满足的不变式，停下重塑（L2）：可能要为聚类路径写单独的综合校验，而不是硬套老路径的。
- 若 `clusterMembers` 并集在真实库上普遍超限，停下——说明该字段在综合后语义不成立，应重新设计而不是继续截断。
- 若用户在看到 Chunk 3 的行为差异后认为不该退回（宁可整体失败），按 L1 就地改回抛错并同步 ADR 0009 的措辞。
