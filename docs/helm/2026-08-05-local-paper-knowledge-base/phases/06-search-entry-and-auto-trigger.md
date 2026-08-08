# P6 — 检索入口与自动触发

goal_ref: ../goal.md
updated: 2026-08-07

## Outcome

ADR 0006（检索入口）与 ADR 0007（自动触发）落地：Dashboard 行内"Find similar papers"按钮打开双页模态框（库内全文相似 + 日报词法相似）；Dashboard 搜索框单框双结果（词法行过滤 + 文献库 KB 相似区块）；`index-personal-library-fulltext` 完成且 `indexed > 0` 自动触发增量更新；授权门拆分（placement 免许可、LLM diff 阶段检查 + 待授权状态）；建议文档整体替换覆盖未审阅时提示。

## Assumptions

- KB 存在性可经 `FullTextKnowledgeBaseFileStore.loadManifest()` 判定（null/错误 = 无 KB）；无 KB/未授权/模型失败时检索入口静默降级（只显示词法部分），不弹错误。
- 从论文出发的查询 = 标题+摘要（空摘要回退标题）；查询构建与嵌入复用现有 `searchPersonalLibraryFullText` 路径。
- 自动触发与手动命令共用 `runIncrementalDirectionUpdate`；placement 阶段无 LLM、无授权要求；LLM diff 阶段保留 `assertIncrementalUpdateCurrent` 的 authorizationFingerprint 检查（P5-T3 结论）。
- 建议文档整体替换语义不变；"覆盖未审阅提示"只提示不改变替换行为。

## Approach

T1 先扩展现有 `SimilarPapersModal` 为双页（库内相似页异步调用 KB 检索，日报相似页为现有词法结果；无 KB 只渲染日报页），行内按钮传入两个来源的构造参数。T2 在 Dashboard 搜索框查询路径上加 KB 相似区块（异步，结果复用 T1 的呈现组件）。T3 重构 `runIncrementalDirectionUpdate` 授权检查位置（run-entry → LLM diff 阶段），并在 `indexPersonalLibraryFullText` 成功后 `indexed > 0` 时自动调用；"待授权"状态写入审核 UI 可读的形态。T4 在建议文档整体替换前检测 pending 建议存在性，覆盖时提示。T5 全量验收 + handoff + 提交。

## Tasks

- [x] T1 双页模态框：`SimilarPapersModal` 扩展（库内相似页：异步加载标题+摘要查询的 KB 结果，相似度+命中段落+打开动作；日报相似页：现有词法结果）；行内按钮接线；无 KB 只显示日报页；插件测试
  - 实现：`library?` 可选选项（query + load）；有则 Library/Daily 双页（Library 默认选中，加载/错误/空三态）；无则原样渲染（既有测试不变）。Dashboard `buildLibrarySimilarOption` 以标题+摘要（空回退标题）经 `searchPersonalLibraryFullText` 异步加载；无可查询文本则不传 library。+4 模态框测试；plugin 424/424、tsc/lint/boundaries 全绿；technical-report Dashboard 段已更新（updated）。
- [x] T2 单框双结果：Dashboard 搜索框查询 → 现有行过滤（不变）+ 异步 KB 相似区块（复用 T1 呈现）；无 KB/失败静默降级；插件测试
  - 实现：`refreshLibrarySearch`（防抖回调 + 主渲染挂点，≥2 字符才触发，staleness token 丢弃过期响应）；纯渲染抽到 `library-search-block.ts`（loading/matches/empty/error 四态）。+5 区块测试；plugin 429/429、lint 0、boundaries OK；technical-report Dashboard 段更新（updated）。
- [ ] T2 单框双结果：Dashboard 搜索框查询 → 现有行过滤（不变）+ 异步 KB 相似区块（复用 T1 呈现）；无 KB/失败静默降级；插件测试
- [x] T3 自动触发 + 授权拆分：`runIncrementalDirectionUpdate` 授权检查下移到 LLM diff 阶段（placement 无条件跑）；索引完成后 `indexed > 0` 自动触发（复用同一方法，通知 summary）；"待授权"状态在审核 UI 呈现；插件测试
  - 实现：run-entry 授权 throw 移除；`assertIncrementalUpdateCurrent` 参数化 `requireAuthorization`（placement 路径 false，LLM 路径 true）；LLM 阶段检查授权 + authorizationFingerprint，无许可跳过并写 `pendingAuthorization`（缓冲数+时间戳）到建议文档（core decoder 兼容两种形状）；审核 UI 待授权横幅；`runIncrementalDirectionUpdateAfterIndex(summary)` 索引后自动触发（indexed≤0 跳过、失败仅记录）。core +2 测试、plugin 测试重写无授权场景 +3 自动触发；core 1534/1534、plugin 433/433；technical-report 授权门段更新（updated）。
- [x] T4 覆盖未审阅提示：整体替换建议文档前检测 pending 建议 → 覆盖时 Notice/状态栏提示；测试
  - 实现：mutation 内比较 current 与 nextDocument 建议集（JSON 相等即重放，不提示），不同则 summary 返回 `superseded` 计数；手动命令与自动触发通知追加 "N un-reviewed suggestion(s) superseded by new evidence"。+1 双轮测试（run 1 空 LLM → run 2 new 建议，superseded=1）；plugin 434/434；technical-report 命令段更新（updated）。
- [x] T5 收尾：core/plugin 全量测试、tsc、lint、boundaries；technical-report handoff（每个被接受 chunk）；提交；goal.md P6 done + status done

## Verification

- T1：模态框双页渲染正确；库内相似页展示分数+命中段落+打开动作；无 KB fixture 只显示日报页；命令/按钮触发正常。
- T2：搜索框输入后行过滤照常 + KB 区块出现；无 KB/模型失败时区块隐藏或降级文案；不阻塞行过滤。
- T3：无授权 fixture 下 placement 照跑、LLM diff 跳过并呈现"待授权"；有授权时全流程；索引后自动触发（indexed>0），复用型重跑不触发；手动命令仍可用。
- T4：替换前有 pending 建议 → 提示；无 pending → 不提示。
- T5：core 1532+ / plugin 420+ 全绿、lint 0 error、boundaries OK；每个被接受 chunk 的 technical-report handoff 到 `updated` 或 `no-impact`；每阶段提交。

## Abort / reshape triggers

- 若 KB 检索的异步加载与 Dashboard 渲染生命周期冲突（视图重建/卸载）：L2——区块改为独立小视图或延后加载，先向用户简报。
- 若自动触发与手动运行并发冲突（同 kind 互斥 throw）：自动触发在 already-active 时静默跳过（非错误），下次索引再补。
- 若"待授权"状态改动审核 UI 结构过大：降级为命令面板/诊断可见的文案提示，不侵入 review modal。
