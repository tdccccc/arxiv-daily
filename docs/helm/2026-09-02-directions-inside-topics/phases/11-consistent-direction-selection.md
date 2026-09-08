# P11 — consistent-direction-selection

goal_ref: ../goal.md
created: 2026-09-07T03:13:17+08:00
updated: 2026-09-07T03:46:27+08:00
revision: 2

## Outcome

一篇论文的主题归属与详细总结资格可解释，详情评分依据实际命中方向，而不是主题第一条方向的影子字段。

## Assumptions

- 单篇只归一个主题，主题名只作标签，已接受库方向与手写方向同权；均保留既有决定。
- 同时命中不同主题时先选与核心研究问题最直接、范围最具体的方向；同样具体时按设置顺序裁决。获胜主题决定 detail 开关，不引入按来源或开关加权。
- 短摘要仍按论文内容生成，personal novelty 仍暂停。

## Approach

筛选 prompt 明写跨主题裁决，prompt contract 3→4（结果形状仍为3），旧分类缓存不复用。详情选取优先读取 paper.topicDirections 中所选主题的命中快照；无命中快照的旧调用读取主题全部非空 directions。topic.description 只保留回滚用途。评分提示词指明按最相关命中方向评估，不要求论文同时满足主题所有方向。

## Chunks

### Chunk 1 — 详细总结评分看到真正命中的方向

- files: packages/core/src/pipeline/detail-selector.ts、prompts/detail-selector.system.md、tests/detail-selector.test.ts。
- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 已有主题首条A/后条B，B命中的论文请求应包含B且不被A限定；现代码只含A。无snapshot时读取A和B；不同主题detail开关的资格边界保持。
- Green check: npm run test --workspace packages/core -- tests/detail-selector.test.ts tests/pipeline-daily-cap.test.ts --maxWorkers=1
- regression checks: core typecheck；对应pipeline tests；旧fixture用normalizeTopic得到实际生产输入。
- [x] implementation and tests accepted — 详情10 Red→45 Green；分类缓存1 Red→129 Green；联合174项及core全量通过。

### Chunk 2 — 重叠方向裁决与分类缓存升级

- files: paper-filter.system.md、paper-filter-contract.ts、daily-filter-checkpoint-store.test.ts、paper-filter.test.ts及snapshot。
- change kind: behavior change
- strategy: cache-boundary Red-Green + real-model verification
- Red / baseline signal: 保存promptContractVersion=3的分类后，当前默认请求应不能复用；改前仍复用。变更prompt并升到4后不复用；相同新合同仍复用。
- exception: 提示词的语义效果由外部LLM决定，单测桩不能证明模型会按具体度选择，不写以关键词分支模拟模型的测试。补偿验证为真实端点的明确宽/窄方向对照，记录输出；模型验证在P6汇总。
- Green check: npm run test --workspace packages/core -- tests/paper-filter.test.ts tests/daily-filter-checkpoint-store.test.ts --maxWorkers=1
- regression checks: core全量、四包typecheck、boundaries；人工核对变更后的prompt snapshot。
- [x] implementation and tests accepted — 详情10 Red→45 Green；分类缓存1 Red→129 Green；联合174项及core全量通过。

## Phase verification

- 最终选择保留单主题与该主题detail开关；第二条库方向有资格参与详细总结评分。
- 缓存边界Red/Green和回归通过。真实模型按具体度选择的结果在P6记录，未跑前不宣称语义实测通过。

## Abort / reshape triggers

- 模型把“具体度”误用为优先选择与核心问题无关的窄方向：修正判据并真实重测，不把宽方向一律降级。
- 详情上下文引入库证据或个人novelty推断：超出本phase，去掉额外推断。
