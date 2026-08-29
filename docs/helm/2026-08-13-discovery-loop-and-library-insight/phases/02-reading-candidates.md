# P2 — Reading candidates：保存待读与按方向回顾

goal_ref: ../goal.md
updated: 2026-08-13

## Outcome

用户可在 Dashboard 把一篇已发现论文保存为待读候选（快照其来源方向/主题、发现理由与 novelty 证据），并在独立的回顾界面按方向分组浏览、做出精读/略读/弃的决策；候选与决策持久化在文献库作用域的旁路 CAS 文档中，删除重建不影响权威记录。

## Assumptions

- 保存入口只在 Dashboard 行（日报文件内不做标记），回顾入口是命令 `review-reading-candidates` 加 Dashboard 头部按钮。
- 存储采用新旁路 store（沿用 IncrementalSuggestionsStore 的 CAS 模式），按 scope/identification 指纹分片，不进入 papers.json、catalog 或 profile。
- 候选以 arXiv paperKey 为身份键；来源快照保存触发方向 id/名称与手动 topic tag、日报路径与日期、发现理由、novelty 差异类型/比较基准/解释。
- 决策（read-closely / skim / dismiss）写入候选记录，P3 再消费回流；P2 不做过滤权重影响。
- 文档上限 500 条：超出时淘汰最旧的无决策候选；已决策候选保留为历史。

## Approach

core 新增 `packages/core/src/library/reading-candidates/`（domain + store），插件接线 store、Dashboard 行按钮与回顾 modal。四个行为块，各自 Red → Green。

## Test strategy

- change kind: behavior change
- strategy: strict Red-Green-Refactor（每任务一个行为块）
- Red / baseline signal:
  - T1：`npx vitest run tests/reading-candidates.test.ts`（新文件）期望 Red：upsert/决策/淘汰行为尚不存在。
  - T2：`npx vitest run tests/reading-candidates-store.test.ts`（新文件）期望 Red：CAS/恢复/解码尚不存在。
  - T3：`npx vitest run tests/dashboard-view.test.ts` 期望 Red：保存按钮与 Notice 行为尚不存在。
  - T4：`npx vitest run tests/reading-candidates-modal.test.ts`（新文件）期望 Red：分组与决策渲染尚不存在。
- Green / regression checks: 各任务聚焦全绿后跑 core 全量（8 GiB 单 fork）、plugin 全量、CLI、双 typecheck、boundaries。
- exception: 无。

## Tasks

- [ ] T1 core domain 与纯操作（change kind: behavior change）—— reading-candidates.ts：文档模型、严格解码、upsert（去重 + 500 上限淘汰）、决策、移除；测试覆盖边界（重复保存、淘汰顺序、无效决策值）。
- [ ] T2 core store（change kind: behavior change）—— reading-candidates-store.ts：路径分片、primary/backup、expectedRevision CAS、语义重放、损坏恢复；测试镜像 suggestions-store 的关键边界。
- [ ] T3 插件接线与保存入口（change kind: behavior change）—— main.ts 构建/加载/重建 store（随连接变化），Dashboard 行"保存待读"按钮（无连接时禁用+提示，保存后 Notice）；测试覆盖连接状态与保存动作。
- [ ] T4 回顾界面（change kind: behavior change）—— reading-candidates-modal.ts：按方向/主题分组、决策按钮（精读/略读/弃）与移除、待决策默认视图；命令注册与 Dashboard 头部入口；测试覆盖分组与决策。
- [ ] T5 真实 Obsidian 复验（non-behavioral）—— 测试 vault 安装构建产物：保存一篇 Dashboard 论文、打开回顾、决策、重启后持久；用户确认后关 P2。

## Verification

- core 聚焦与全量、plugin 全量、CLI 全绿；typecheck ×4；boundaries；lint warnings ≤64。
- 真实 Obsidian：保存/回顾/决策/重启持久四步走通。

## Abort / reshape triggers

- 若 Dashboard 行模型缺少保存快照所需的 provenance/novelty 字段（当前设计依赖 DashboardOccurrenceProvenance），L1 改为从 paper index 的 occurrence 数据构建快照。
- 若 500 上限淘汰与用户预期冲突（如丢弃了重要候选），L2 重新设计上限策略（如分区上限或仅提示不淘汰）。
- 若回顾界面与现有 direction 审核 modal 重复度过高且用户倾向合并入口，L2 合并进 Review directions modal。
