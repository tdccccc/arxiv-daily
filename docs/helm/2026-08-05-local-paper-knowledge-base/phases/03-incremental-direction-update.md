# P3 — 方向增量更新（incremental-direction-update）

<!-- Filename 03-incremental-direction-update.md ↔ P3 -->
goal_ref: ../goal.md
updated: 2026-08-06

## Outcome

新论文入库后方向可增量更新：未归入任何方向的已索引论文经**方向锚定**（与已确认方向代表论文的 maxchunk 相似度，相对阈值）就近归入或进缓冲池；缓冲池达到阈值触发**局部重聚类**与 **LLM diff 建议**（归入/新建/分裂/合并），全部进**审核队列**由用户应用；**方向锁定**（lockedAt + timeline 事件）使锁定方向不参与自动合并/分裂/改名但新论文仍可归入；机器建议永不覆盖用户决定。全量重建兜底 = 现有 `generatePersonalLibraryDirections`（P2 已验证），本阶段仅确认。

## Assumptions

- 方向锚 = 已确认方向的**代表论文**（≤5 篇，现成字段）的 chunk 集合；增量归入相似度 = 新论文与锚的 maxchunk（与聚类/检索同语义）；阈值用相对策略（最强方向 vs 次强方向的差距 + 绝对下限），对 e5 饱和分布鲁棒。
- 缓冲池 = 已索引但不在任何方向 `clusterMembers` 中的论文（增量新论文 + 上次聚类离群），从 KB + profile 即时派生，不新增持久化。
- 增量建议是**修改操作集合**（attach/new/split/merge），与全量 proposal（候选方向集）语义不同——独立文档 `incremental-suggestions.json`（CAS 模式同 profile store），审核 UI 独立区块。
- 方向锁定 = confirmed direction 可选 `lockedAt` 字段 + timeline `locked`/`unlocked`/`split` 事件（T4 事件类型扩展，decoder 允许键更新，旧文档兼容，schema 版本保持 3）。
- 归入操作更新方向 `clusterMembers`（追加论文 + 置信度）并写 `members-updated` 事件；clusterMembers 保持"方向覆盖论文集合"语义。
- 全量重建兜底已有实现（generatePersonalLibraryDirections 从 KB 全量聚类生成新 proposal），P3 不新增代码，仅端到端确认。

## Approach

core 新增 `packages/core/src/library/incremental/`：`suggestIncrementalPlacement`（新论文 vs 方向锚 → 归入建议/缓冲池）、`reclusterBufferPool`（缓冲池 + 方向锚 → 新簇/漂移信号）、`suggestDirectionDiff`（LLM：簇 + 现有方向 → attach/new/split/merge 建议，复用 LlmClient 契约与 validation 模式）；`personal-library-interest-profile.ts` 扩展 locked/split timeline 事件与 `lockedAt`；增量建议 store（CAS 文档）；plugin 审核 UI 应用建议 + 锁定按钮。所有建议仅入审核队列，应用才改 profile。

## Tasks

- [x] 方向锚定与增量归入（core 纯函数）：`suggestIncrementalPlacement`——未归入论文 vs 各方向代表论文 maxchunk，相对阈值（最强/次强差距 + 绝对下限）输出归入建议（方向 + 置信度）或缓冲池；单测（合成向量）。— `packages/core/src/library/incremental/placement.ts` + `tests/incremental-placement.test.ts`（9 测试）
- [x] 缓冲池触发与局部重聚类（core）：`reclusterPool`（边界词禁 `Buffer`，实现名从计划名 `reclusterBufferPool` 调整）——缓冲池 + 方向锚聚类 → 新簇候选（含 nearestDirection 漂移信号）与 stillPooled；触发阈值参数化；单测。— `recluster.ts` + `tests/incremental-placement.test.ts` 覆盖
- [x] LLM diff 建议（core + prompt）：`suggestDirectionDiff` 对每个新簇 + 现有方向上下文生成建议（attach/new/split/merge，含目标方向与理由），严格校验（kind-invalid/direction-unknown/direction-locked/paper-keys-invalid/reason-invalid/conflict）+ 最多 3 次重试；输出稳定排序 attach < merge < new < split；单测（fake LLM，15 测试）。— `diff-suggestions.ts` + `prompts/personal-library-direction-diff.system.md` + `tests/direction-diff-suggestions.test.ts`
- [x] 方向锁定（core）：timeline 事件扩展 `locked`/`unlocked`/`split`；confirmed direction 可选 `lockedAt`；lock/unlock mutation（写事件）；锁定方向在 diff 建议中仅可 attach（split/merge 拒绝、attach 允许）；单测（迁移兼容 + 排除逻辑，11 测试）。— `personal-library-interest-profile.ts` / `-review.ts` + `tests/personal-library-direction-lock.test.ts`
- [x] 增量建议存储与审核（core + plugin）：`IncrementalSuggestionsStore`（CAS、primary/backup、严格 decoder；core 35 测试）；plugin `runIncrementalDirectionUpdate`（placement attach → 建议文档；buffer ≥3 触发 recluster + LLM diff，整体 replace CAS）+ `applyIncrementalSuggestion`/`dismissIncrementalSuggestion`（内容键 `${kind}:${directionId}:${firstPaperKey}`；new 走候选确认流程入 proposal store）+ 命令 `check-incremental-direction-updates`/`review-incremental-suggestions` + 审核 UI（建议区块 + lock/unlock 按钮）；plugin 测试 16 新增。— `packages/core/src/library/incremental/{suggestions-store,apply}.ts`、`plugin/main.ts`、`plugin/src/commands.ts`、`plugin/src/library/interest-profile-modal.ts`、`plugin/tests/incremental-direction-update.test.ts`
- [x] 端到端验证：真实库初始聚类 → 确认方向 → 新增论文索引 → 增量归入/缓冲池 → 建议 → 审核应用 → 锁定方向不参与自动操作；全量测试 + boundaries + journal 记录实测阈值。— `tmp/incremental-e2e/`（scratch）：12 篇异构基线（P2 语料副本）+ 5 篇真实 DL PDF 增量索引，全链路 31 项 PASS；实测：默认阈值（0.25/0.05）下 attach 8/buffer 1（confidence min 0.190/median 0.339/max 0.552；margin min 0.046/median 0.150/max 0.392）；严格变体 0.35 → attach 3/buffer 6 驱动 recluster+LLM diff；locked 方向 split/merge 被 `direction-locked` 拒绝而 attach 放行；建议/画像/提案 store CAS 落盘一致。全量：boundaries OK、lint 0 error（60 warnings 达上限）、core 1528/1528（8192 堆）、plugin 405/405、typecheck 干净。

## Verification

- `npm run typecheck && npm test && npm run check:boundaries`（core 套件用 `NODE_OPTIONS=--max-old-space-size=8192`）通过。
- 归入判定单测：同主题新论文建议归入对应方向、跨主题进缓冲池、锁定方向仍可归入。
- 重聚类单测：缓冲池内部强关联组形成新簇建议、与现有方向高相似的簇标记漂移。
- 建议校验单测：LLM 输出越界（引用不存在论文、未知类型、超长）拒绝并重试。
- 锁定单测：locked 方向排除于自动合并/分裂/改名，lock/unlock 写事件，v3 旧文档兼容加载。
- 审核端到端：应用归入后 clusterMembers 更新 + members-updated 事件；应用新建后走确认流程；应用合并/分裂后 timeline 记录。
- 缓冲池触发参数在真实语料上实测后 journal 记录默认值。

## Abort / reshape triggers

- 增量归入的相对阈值在真实数据上仍不可用（误归入率高）→ L1/L2：归入改为"全部候选 + 用户筛选"，或引入方向级向量锚（代表论文聚合）替代逐篇 maxchunk。
- LLM diff 建议质量差（类型误判、理由不可信）→ L1：调整 prompt/示例；仍差 → L2：建议降级为纯规则（相似度驱动）不调 LLM。
- 审核队列与现有 proposal 确认流程冲突（双文档状态纠缠）→ L2：建议文档并入 proposal（candidates 加建议类型字段，schema v4）。
