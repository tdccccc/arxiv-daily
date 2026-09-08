# P2 — traceable-coverage

goal_ref: ../goal.md
created: 2026-09-08T08:05:16+08:00
updated: 2026-09-08T08:09:00+08:00
revision: 2

## Outcome

复审可以检查已覆盖论文与对应方向；方向修改、删除或旧提案缺失覆盖依据时，不再宣称当前覆盖成立。

## Assumptions

- 只让提案保留覆盖证据；已接受方向依然是普通设置文字，不恢复退休画像。
- 主题改名不改变方向覆盖；方向文字修改或方向跨主题移动需重新核实。
- 旧 schema 6 提案仍可复审新增候选，但覆盖依据缺失时诚实提示重新生成。

## Approach

提案增加可选 coverageEvidence，每项含 topicId、directionId、directionText 和 paperKeys；严格校验其与 coveredPaperKeys 的完整分区。生成从已验证的 coveredGroups 构造记录。复审按稳定身份及文字与当前设置对照，并折叠展示论文标题和来源方向；有效、已变化、未核实分开计数。

## Chunks

### Chunk 1 — 生成和持久化保留覆盖依据

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- files: core library interest-profile、direction-proposer；contract/proposer/store tests。
- Red signal: 输出应保留方向 ID、文字与所属论文；错误 key、重复归属、覆盖数不一致不可解码。
- Green check: core proposal-contract、clustered-direction-proposer、proposal-store tests。
- [x] implementation and tests accepted — 2 Red；contract/proposer/store 78 Green，存储重载保留完整覆盖依据。

### Chunk 2 — 当前覆盖核对与复审明细

- change kind: bug fix + behavior change
- strategy: strict Red-Green-Refactor
- files: core 覆盖查询及 plugin interest-profile-modal；UI tests。
- Red signal: 删除/改写方向后不显示全部覆盖；改主题名仍有效；旧记录标未核实；明细可见方向及论文。
- Green check: plugin proposal-acceptance-ui 与 interest-profile-modal tests；core覆盖查询测试。
- [x] implementation and tests accepted — 4 UI Red；plugin 82 Green；四包 typecheck 与 diff check 通过。

## Phase verification

- schema/store/proposer/DOM 联合回归；core/plugin typecheck；保留分次接受与草稿回归。

## Abort / reshape triggers

- 覆盖依据写入普通方向导致恢复画像：停止并保持提案边界。
- 为兼容旧结果而默认全部覆盖有效：拒绝，旧结果必须显示未核实。
