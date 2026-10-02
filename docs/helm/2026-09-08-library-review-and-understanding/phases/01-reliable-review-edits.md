# P1 — reliable-review-edits

goal_ref: ../goal.md
created: 2026-09-08T08:00:00+08:00
updated: 2026-09-08T08:05:16+08:00
revision: 2

## Outcome

用户编辑方向和代表证据后可以可靠保存并接受；选择、改归属和失败不会静默恢复旧文字，非 arXiv 论文不会阻断合法复审。

## Assumptions

- 接受记录及设置的原子写入继续由现有 settings transaction 负责。
- 提案保存可以先完成；随后接受失败仍保留已保存的提案，不能假称设置已更新。
- 草稿仅属于当前 proposalId/scopeFingerprint；新生成提案不继承旧草稿。
- 已收集的 file:sha256 证据可由提案的 catalogInputPapers 验证，arXiv 证据沿用当前目录校验。

## Approach

复审 modal 保留按 candidateId 索引的草稿，在重绘后恢复；已选方向接受前验证并保存全部修改，再调用现有接受入口。保存失败保留可重试文字并停止接受。代表证据解析支持当前目录与提案中已知的本地文件证据，同时保留 scope 与 fingerprint 校验。

## Chunks

### Chunk 1 — 草稿可靠保存与接受

- files: plugin/src/library/interest-profile-modal.ts; plugin/tests/proposal-acceptance-ui.test.ts
- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red signal: 编辑 textarea 后切换 topic checkbox 不应还原；编辑后接受应写新文字；保存失败后草稿保留且不接受；代表选择和另一方向编辑不因重绘丢失。
- Green check: npm run test --workspace plugin -- tests/proposal-acceptance-ui.test.ts tests/personal-library-interest-profile-modal.test.ts --maxWorkers=1
- regression checks: plugin typecheck；分次接受、改归属、已接受不可修改与提案替换原有回归。
- [x] implementation and tests accepted — 3 DOM Red；跨库守卫另 1 Red；78 plugin Green，包含失败重试、跨库切换与实际 file:sha256 持久化。

### Chunk 2 — 跨来源代表证据复审

- files: packages/core/src/library/personal-library-proposal-review.ts; packages/core/tests/personal-library-proposal-review.test.ts; plugin integration tests
- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red signal: 有效 file:sha256 提案和空 arXiv 目录保存文字时不应 evidence-mismatch；换为同提案其他本地证据可保存；未知 key 与不兼容库仍拒绝。
- Green check: npm run test --workspace packages/core -- tests/personal-library-proposal-review.test.ts --maxWorkers=1
- regression checks: core schema/store 与 plugin 编辑接线；core/plugin typecheck。
- [x] implementation and tests accepted — 本地证据 1 Red；core review/contract/store 65 Green，4 包 typecheck 通过，后续跨库修改 plugin typecheck 通过。

## Phase verification

- 自动化 DOM 复审接真实 core 更新和接受运算，验证保存失败重试及混合来源。
- 记录 Red 与 Green 数量，完成后更新 goal 阶段并计划 P2。

## Abort / reshape triggers

- 草稿需要改变设置接受事务的原子边界：先重新设计，不引入 settings/proposal 双写伪事务。
- 已知证据检查容许任意新论文或越过库范围：停止并收紧证据契约。
