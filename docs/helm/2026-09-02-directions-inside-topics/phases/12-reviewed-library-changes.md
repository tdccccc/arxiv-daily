# P12 — reviewed-library-changes

goal_ref: ../goal.md
created: 2026-09-07T01:00:10+08:00
updated: 2026-09-07T03:13:17+08:00
revision: 3

## Outcome

库建议尊重已有研究方向：完全覆盖时没有新增；未覆盖的建议归属可改；逐方向接受可以继续，旧提案不恢复用户已编辑或删除的内容。

## Assumptions

- 用户 2026-09-07 的“按照你说的开始完成当前 helm”授权落实本次 review 的六项修正；不新增个人 novelty，不改变单主题归属或每日默认 20 篇。
- 模型可以根据已有主题的方向文字区分已覆盖、追加方向、新建主题；不依赖未实测的相似度门槛。
- 已接受方向仍是普通方向。提案的接受记录只服务当前提案的幂等处理，不保存证据、画像、合并历史。
- 设置主题的 id 在改名时稳定；删除后目标缺失必须重新选择，不能回退为同名新建。

## Approach

组织输入增加 `existingTopics`（id/name/directions）。输出在 `topics` 之外允许 `coveredGroups`，每项以 groupId/topicId/directionId 指明已有覆盖；每个输入组恰好出现于已覆盖或新增方向的一处。新主题可带 `targetTopicId` 引用已有目标；完全覆盖可返回空 topics。首扫继续使用已验收的 2–4 主题、每主题 1–2 方向；后续生成不以数量下限强迫新增。提案 schema 升版，保存 `coveredPaperKeys` 以正确显示已覆盖与未覆盖数量。

接受逻辑计算完整的新 topics 与当前提案接受记录（proposalId、已处理 candidateIds、proposal-topic 到 settings-topic 的目标映射），两个结果同一次设置文件写入成功后才提交到内存。每个文献库 scope 保留一份当前提案记录，换库后回来不会忘记接受；记录放插件持久化 envelope，与 ProposalStore 无双写事务。Direction.id 保留候选身份；重复、用户编辑、用户删除均不重新应用已处理候选。纯文本规范化去重只是补充；不会把整主题同名当成所有方向已接受。

复审显示已有目标当前名称，可明确改到其他主题或新主题。缺失目标禁止接受并要求重新选择；选择结果持久化在提案。方向可逐条勾选，已处理行只读；只有全部方向均已处理才显示整主题 Added。Discovery cues 改为只读的证据线索，说明筛选使用上面的方向文字。

## Chunks

### Chunk 1 — 组织与持久化契约允许零新增和已有目标

- files: `packages/core/src/library/personal-library-{topic-organization,direction-proposer,interest-profile,proposal-store}.ts`、组织 prompt；相应 core tests。
- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 先断言全覆盖得到 topics=[]，混合覆盖与新增保持完整分区，已有主题进入真实 LLM request；旧代码分别拒空结果、缺输入或丢目标。
- Green check: `npm run test --workspace packages/core -- tests/personal-library-topic-organization.test.ts tests/clustered-direction-proposer.test.ts tests/personal-library-proposal-contract.test.ts tests/personal-library-proposal-store.test.ts --maxWorkers=1`
- regression checks: core typecheck；旧 schema 提案可重生成，取消/消息预算/引用范围检查继续通过。
- [x] implementation and tests accepted — 组织/生成/schema/store 159 项、core 全量与 typecheck 通过；观察到首轮 35 Red 及历史 schema/名称边界 Red。

### Chunk 2 — 逐方向接受是一次可恢复的设置变更

- files: `packages/core/src/settings/accept-proposed-topics.ts`、对应 tests；插件 settings transaction、持久化 envelope、接受入口与测试。
- change kind: bug fix + behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 同主题先接受 A 再接受 B；改名仍追加到同一 id；已接受 A 编辑/删除后不恢复；目标删除不新建；保存失败后内存与重载均未接受、重试成功；并发接受不会覆盖彼此。
- Green check: core `accept-proposed-topics.test.ts`；plugin `personal-library-interest-profile-modal.test.ts` 与 `settings-change-service.test.ts`。
- regression checks: `npm run test --workspace plugin`、`npm run typecheck`；接受计算在设置事务队列内读取最新值，`normalizeTopic` 继续唯一维护 description 影子。
- review follow-up: 普通主题/方向编辑也经相同事务队列按稳定 id 更新；不能让保存等待期间的 live 原地修改被接受的整数组快照覆盖。DOM 并发测试保留编辑/删除及同时接受的新方向。
- [x] implementation and tests accepted — 接受/设置/真实 DOM/生成接线 Red→Green；全插件 772 项、全工作区及构建通过。

### Chunk 3 — 生成和复审交互使用完整的已有设置

- files: `plugin/main.ts`、`plugin/src/library/interest-profile-modal.ts`、proposal-review 和相关 tests。
- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 真实 controller 接线测试看到已有方向进入生成；DOM 流程可接受剩余方向、改归属、识别失效目标；仅改证据线索不会伪装成筛选设置修改；全覆盖不是空提案错误。
- Green check: plugin modal/lifecycle tests；core proposal-review tests。
- regression checks: `npm run test:workspaces`、`npm run typecheck`、`npm run lint`、`npm run check:boundaries`、插件 build。
- [x] implementation and tests accepted — 接受/设置/真实 DOM/生成接线 Red→Green；全插件 772 项、全工作区及构建通过。

## Phase verification

- 自动化全链路：已有手写主题 → 混合覆盖/新增提案 → 分次接受 → 编辑/删除 → 重开提案；保存失败/重载/目标改名均有行为断言。
- 在真实测试库验证零新增、已有主题追加、另建主题及手动改归属；真实桌面验收仍按 goal Constraints，由用户看过后接受。
- 尚未进行的真实验收留在 P6，不以单测替代；实现可在集成检查完成后推进独立 P11/P13。

## Abort / reshape triggers

- 模型持续把已有方向重新表述为新方向，或全部归入同一主题：保留错误响应，调整判据后重测，不能静默丢证据。
- 接受需要先写一处再写另一处才能成立：停止该实现，恢复同一 settings 持久化边界。
- 接受处理改变普通方向的筛选权重或复活画像：超出本阶段，撤掉该路径。
