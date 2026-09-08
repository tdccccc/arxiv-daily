# P3 — bounded-library-organization

goal_ref: ../goal.md
created: 2026-09-08T08:09:00+08:00
updated: 2026-09-08T08:15:55+08:00
revision: 2

## Outcome

500/1000 篇文献库可通过有界请求组织成可复审主题，不把多数摘要压成前几个词，也不丢论文归属。

## Assumptions

- 小库且完整证据在请求上限内时继续单次组织。
- 大库的证据分组可以分成传输批次；这是预算边界，不是声称新增研究方向。
- 模型各批输出是有出处的中间线索，最终主题数量和含义仍经过综合。
- 单条方向允许覆盖当前支持库规模的全部 1000 篇；512 成员限制不是用户需求。

## Approach

叶批次保留每篇标题与完整摘要，按实际序列化预算分批；组织输出完整校验后成为下一层证据线索。线索只携带代表论文与研究文字，完整论文集合在本地映射中保留；逐层组织至最终主题数达标。已有覆盖在各层保留稳定方向依据，最终再构造统一提案。

## Chunks

### Chunk 1 — 有界分批与可追溯综合

- change kind: behavior change + optimization
- strategy: strict Red-Green-Refactor with capacity baseline
- files: core library organization/proposer and prompts; clustered-direction-proposer tests。
- baseline: 200 篇 1500 字符摘要只剩约 101 字符；500/1000 grouped papers 组织输入抛 evidence-too-large。
- Red signal: 500/1000 有效论文均进入叶请求，完整摘要末尾证据仍在；每次请求 <=60,000 字符，最终成员不重不漏、主题数量合法。
- Green check: clustered-direction-proposer、organization、schema tests。
- [x] implementation and tests accepted — 500/1000 两个容量 Red 后 Green；核心联合 172 Green，每篇完整摘要曾进入请求，最终成员完整，每条请求有界。

### Chunk 2 — 失败、取消与多领域完整性

- change kind: behavior change verification
- strategy: strict Red-Green-Refactor for missing guards; existing Green regression otherwise
- Red signal: 中间层缺失/重复归属拒绝；取消终止后续请求；单一大组和混合来源均不撞旧成员上限；现有设置预算超限明确拒绝。
- Green check: core organization/proposer/store 与 plugin generation integration tests。
- [x] implementation and tests accepted — 已有覆盖分批、取消、缺失归属有限重试、单篇过大与单一600篇组测试通过；plugin 生成10项、core typecheck、boundaries通过。

## Phase verification

- 正确性固定夹具和容量规模检查；模型语义优劣另留 P7 真实库验收。
- core/plugin typecheck、boundaries 与 prompt 契约版本记录。

## Abort / reshape triggers

- 中间线索体积不收敛：明确失败并调整批次契约，不能无限循环。
- 只保留代表论文导致成员证据丢失：拒绝该实现。
