# P3 — library-change-nudge

goal_ref: ../goal.md
created: 2026-10-15T16:30:00+08:00
updated: 2026-10-15T16:30:00+08:00
revision: 1

## Outcome

建好索引后，本地（不调模型、无需授权）数出「没有任何一次扫描计入过」的库论文；数量 >=3 时，插件提示库有变化并提供「更新方向建议」。研究者点更新后，只把这些未计入论文 + 现有主题/方向文字发给模型，产出 ADR 0014 的 已覆盖 / 新方向 / 新主题 提案，走现有复审与接受流程。新提案继承「已计入集合」（前任并本次输入，含未分组论文），库的其余部分不再显得新。替换含未审候选/草稿的提案沿用现有确认。休眠的 ADR 0007 增量代码删除，过时文档修正。CLI 同样能看到这个数并能触发只扫新增。

## 读码结论（file:line，均已核实）

- 提案文档已记录完整输入集：`catalogInputPapers`（personal-library-interest-profile.ts:105）= 聚类输入的全部论文，含离群/未分组者（direction-proposer.ts:545-549,583）。`coveredPaperKeys`(:110)、`clusterMembers`(:62) 只是其子集；plugin 的 `unclassifiedBufferPoolPapers`(interest-profile-modal.ts:1366)=输入集减去覆盖与候选成员，即「未分组」。所以旧提案（无新字段）的已计入集合 = 它的 `catalogInputPapers`。
- 提案 schemaVersion 6 已进入发布 tag（0.6.0/0.6.1/0.6.2/dsh-v0.1.18 均为 6）。decoder 里「proposal 从未发布」的注释（interest-profile.ts:327-330）已过期。旧版本提案走 `regeneration-required`（store.ts:262-281），升版会让所有人的提案作废并要整库重跑一次模型。结论：不升 schemaVersion，加可选字段。decoder 是 exact-key 白名单（:321-326），必须登记新键；`outputProposal`/`proposalDocument`（proposal-review.ts:221-229,326）每次编辑都重解码，新字段若漏返回（:381-390 的 return）会被编辑悄悄丢掉。
- 提案者用向量聚类：`buildClusteringInput`（clustering/paper-vector.ts:91，按近期排序、上限 1000、读 chunk 向量）→ `clusterPaperVectors`（clusterer.ts:132）。分位阈值是「语料自身 pair 分数的分位」（:170-173）：对 3–10 篇新论文，0.95 分位必然会并起最相近一对，凭空造簇；`centerCorpus` 对小集合也无意义；`minClusterSize=2` 下孤篇全成离群，全离群则抛 "no-evidence"（proposer:497）。**结论：子集不能走聚类。** 重扫直接让每篇新论文自成一个证据组（不读向量），语义归并交给模型（一个方向可含多个组，topic-organization.ts:174-186；组数 >=1 即合法，:84 topics<=组数）。这也符合 ADR 0015 §3「不做放置/相似度计算」。
- 初跑/后续由 `existingTopics.length` 决定（topic-organization.ts:81-84：空→2–4 个新主题；非空→可 0 个、可只 covered）。无现有主题时子集重扫无意义 → 回退整库扫。
- 论文标识：catalog 只含 arXiv（`arxiv:`），非 arXiv 兜底文件是 `file:sha256:…`，标题/摘要只在 manifest 记录里（knowledge-base.ts:92-95）；arXiv 记录的 title 不在 manifest。所以计数必须用 manifest 的 ready 论文（含兜底文件），不能只数 catalog。聚类输入的「同名去重」只作用于带 title 的兜底文件（paper-vector.ts:111-125），被丢的重复件不在 `catalogInputPapers` → 需要把 `droppedPaperKeys` 也记入已计入集合，否则永远算新。
- 上限继承：首扫最多看最近 1000 篇（`PERSONAL_LIBRARY_MAX_SELECTED_CATALOG_PAPERS`）。超过的老论文首扫看不到，会让提醒永远亮。计数与重扫都按「同一排序、同一 1000 上限」圈定范围。
- 共享路径：插件有自己的 `generatePersonalLibraryDirections`（main.ts:1123），CLI/工作台走 core `LibraryWorkflow.propose`（library-workflow.ts:212）。核心逻辑放 core 一份，两边各自接线。CLI 无 `library status` 的库内容读取（它只做 connection，library-cmd.ts:60-64）；已有的落点是 `library index` 输出（:86）、`library directions`（:90）和工作台概览里「分析之后新增的论文」（workbench/web/library-review.ts:171,203，按 `catalogInputPapers` 差集算，重扫后会把整库显示成新增，必须改）。
- 插件完成点：`indexPersonalLibraryFullText` 在 `refreshLibraryIndexTrace()` 之后（main.ts:~1703）；两处调用者（commands.ts:517、settings/tab.ts:900）各自发完成 Notice。计数放在方法内部（try/catch，失败不影响索引结果）。
- 弹窗里 `renderIncrementalNotice`（modal:966）是静态文字，被实时提醒取代。`needsCoverageUpdate`（:1270）的「Update suggestions」按钮是**整库**重验（方向文字被改→需重新判覆盖），不能被新增扫描取代；两个动作要分开命名。弹窗在 plugin 单测里有真实 DOM 测试（personal-library-interest-profile-modal.test.ts），不是不可测。
- 休眠代码面：core `library/incremental/{placement,recluster,diff-suggestions,suggestions-store,placeable-direction}.ts`、`src/index.ts:32-36` 导出、`prompts/personal-library-direction-diff.system.md`、tests `incremental-placement` / `direction-diff-suggestions`；插件 `INCREMENTAL_BUFFER_TRIGGER`(main.ts:272)、`IncrementalSuggestionsStore` 构造(~2376,2537)、`librarySuggestions`(365,1044)、`getIncrementalSuggestions`(1294)、snapshot 的 suggestions 字段(209,993)、modal 里 `incrementalSuggestionKey/PaperCount` 与 `DirectionDiffSuggestion` import、commands.ts:417 重复命令。磁盘上已有的 suggestions 文件不管（无害孤儿）。

## Assumptions

1. 「已计入」= 某次扫描的输入集（含未分组、含同名去重被丢者）。研究者后来丢弃候选不改变计入状态。
2. 库内论文被移除不计数（ADR 0015 Consequences）；重扫写入时把继承集合与当前范围取交集，再并本次输入，使文档有界（新增常量 `PERSONAL_LIBRARY_MAX_ACCOUNTED_PAPERS=10000`）。
3. 无提案→不提醒；首扫就是 ADR 0014 初跑流程。旧提案（无新字段）按其 `catalogInputPapers` 视作已计入，升级后不会突然全库都「新」。
4. 重扫新提案的 `catalogInputPapers` = 本次新论文（代表论文/复审仍只能取自它，proposal-review.ts:166,200 不用改）；整库「已分析」数改读已计入集合。
5. 「Generate again」保留为整库重跑（也写已计入集合）；「更新方向建议」是新按钮/新入口，ADR 0015 §4 的「随时手动启动」理解为此。

## Chunks

### Chunk 1 — 已计入集合与本地计数（core，纯函数 + 文档形状）
- change: 提案加可选 `accountedPaperKeys`（有序去重、规范 key、必须包含 `catalogInputPapers` 全部 key、<=上限；不升 schemaVersion）；decoder/return 登记；`accountedPaperKeysOf(proposal)`（字段缺省→输入集）；新文件 `library/personal-library-change.ts`：`listUnaccountedLibraryPapers({manifest, proposal})`（ready、同序同 1000 上限、兜底文件同名去重、返回按近期排序的 key），常量 `LIBRARY_CHANGE_NUDGE_COUNT=3` 与 `shouldNudgeLibraryChange(count)`；`index.ts` 导出。
- kind: behavior（持久格式的加法）
- first signal: 新建 `packages/core/tests/personal-library-change.test.ts`：「无提案返回 null」「旧提案无字段时输入集含未分组者都算已计入」「新增 2 篇不提醒、3 篇提醒」「manifest 里被去重丢掉的兜底文件不计」「failed 状态不计」；`personal-library-proposal-contract.test.ts` 新用例「accountedPaperKeys 缺输入 key 被拒绝」「v6 旧文档不带字段仍解码且 round-trip 不变」「updateCandidate/removeCandidate 后字段保留」。命令：`npm test -- tests/personal-library-change.test.ts tests/personal-library-proposal-contract.test.ts`（当前红：模块不存在）。
- checks: 同上 + `tests/personal-library-proposal-store.test.ts tests/personal-library-proposal-review.test.ts`；`npm run typecheck`；`npm run check:boundaries`。
- [x] accepted on observed evidence

### Chunk 2 — 只扫新增的提案者（core，假 LLM）
- change: `proposeNewPapersPersonalLibraryDirections({catalog, knowledgeBase, previous, existingTopics, llm, …})`（同文件或兄弟文件，复用 `organizeBoundedEvidence` 与同一套 strict decode）：取 Chunk 1 的未计入 key→每篇一个证据组（不聚类、不读向量）→模型→`coveredGroups`/新方向/新主题；写 `accountedPaperKeys = (previous 已计入 ∩ 当前范围) ∪ 本次输入`；无 previous、无现有主题抛 `no-baseline`，无新论文抛 `nothing-new`（宿主回退整库/提示）。`proposeClustered…` 也写 `accountedPaperKeys`（含 `mergedDuplicates` 丢弃者，并接受 `previous`）。生成合同指纹加 scope 标记；组织提示加一句「组可能只含一篇论文」，提示版本 v4。
- kind: behavior
- first signal: 新建 `packages/core/tests/library-new-papers-proposal.test.ts`：「发给模型的消息只含未计入论文与现有方向文字，不含老论文」「全部 covered=合法的空增量结果」「新主题/追加到现有主题均通过 strict decode」「重扫后同一 manifest 的未计入数=0，老论文未重新变新」「5 篇新论文单组也不触发 no-evidence」。命令：`npm test -- tests/library-new-papers-proposal.test.ts`。
- checks: `tests/clustered-direction-proposer.test.ts tests/personal-library-topic-organization.test.ts tests/personal-library-thin-evidence.test.ts`；typecheck；boundaries。
- [x] accepted on observed evidence

### Chunk 3 — LibraryWorkflow 与 CLI（对等）
- change: `LibraryWorkflow.propose({scope:"full"|"new-papers"})`（后者读 previous、调 Chunk 2，其余校验同现有）；`review()` 快照加 `newPaperKeys`；`library propose --new`；`library index` 的 JSON 输出加 `newPapersSinceSuggestions`，>=3 时 stderr 一行提示；`library directions` 视图带该数；工作台「分析之后新增的论文」改读已计入集合，并给该区块一个「更新方向建议」按钮（走 `propose` scope=new-papers，确认文案写明只发送新增论文）。`status` 不动（只读 connection，不开库存储）。
- kind: behavior
- first signal: `packages/core/tests/library-workflow.test.ts`「propose new-papers 只发送新增并保存带继承集合的提案」；`apps/cli/tests/cli-library.test.ts`「library index 输出未计入数，>=3 时 stderr 提示；library propose --new 不重复发老论文」；`apps/cli/tests/workbench-library-review-ui.test.ts`「重扫后新增列表只含真新增」。命令：`npm test -- tests/library-workflow.test.ts`；`npm run test --workspace arxiv-daily -- tests/cli-library.test.ts tests/workbench-library-review-ui.test.ts`（命令形式沿用 P2，已核实 workspace 名为 `arxiv-daily`）。
- checks: `apps/cli/tests/workbench-library.test.ts workbench-library-ui.test.ts`；typecheck。注意与 P2 Chunk 6 同改 apps/cli（runtime.ts vs library-cmd.ts），P2 Chunk 6 验收后再动。
- [x] accepted on observed evidence

### Chunk 4 — 插件：完成点计数与提醒
- change: `PersonalLibraryProfileSnapshot` 加 `newPaperKeys`；`refreshLibraryIndexTrace` 末尾与每次提案保存/重载后重算（读 manifest + 已载入提案，try/catch）；`indexPersonalLibraryFullText` 成功后若 >=3 弹一条 Notice（文案由纯函数 `describeLibraryChangeNudge(count)` 给出，放 `library/index-completion.ts` 旁；带「更新方向建议」按钮，点开复审弹窗）；`generatePersonalLibraryDirections` 加 `scope`，`new-papers` 走 Chunk 2，缺基线回退整库并在返回里说明。无提案/计数失败/<3 均不提示，且绝不使索引失败。
- kind: behavior
- first signal: 新建 `plugin/tests/library-change-nudge.test.ts`：纯函数文案（单数/复数、<3 返回 null）；`plugin/tests/fulltext-index-lifecycle.test.ts` 新用例「索引完成后有提案且新增 3 篇→发一条提示；2 篇/无提案→不发；计数抛错→索引仍成功」；`plugin/tests/library-generation-progress.test.ts` 新用例「scope=new-papers 只把新增发给模型」。命令：`npm run test --workspace obsidian-arxiv-daily -- tests/library-change-nudge.test.ts tests/fulltext-index-lifecycle.test.ts`。
- checks: `tests/commands.test.ts tests/settings-declarative-tab.test.ts tests/library-build-index-scan-first.test.ts`；`npm run typecheck`；与 P2 Chunk 5 同改 main.ts，其验收后再动。
- [x] accepted on observed evidence

### Chunk 5 — 插件：复审弹窗
- change: 用实时提醒取代静态 `renderIncrementalNotice`：库有 N 篇新论文时在提案页顶部显示一句话+主按钮「更新方向建议（N 篇新论文）」→ `generate(scope:"new-papers")`；沿用 `generate()` 里「有未审候选/草稿则确认」（modal:1069）；`Generate again` 留在「More options」；`needsCoverageUpdate` 的按钮改名「重新核对覆盖」避免和新按钮同名；概览「已分析 N 篇…有 M 篇不在分析中」改读已计入集合；重扫提案的概览标明「本次新增 K 篇」。
- kind: behavior
- first signal: `plugin/tests/personal-library-interest-profile-modal.test.ts` 新用例：「新增 3 篇时出现更新按钮并以 scope=new-papers 调 controller.generate」「新增 <3 篇不显示按钮」「有未审候选时先出确认」「覆盖已变更时按钮叫重新核对覆盖且走整库」「概览不把已计入论文说成未分析」。命令：`npm run test --workspace obsidian-arxiv-daily -- tests/personal-library-interest-profile-modal.test.ts`。
- checks: `tests/proposal-acceptance-ui.test.ts tests/library-modal.test.ts`；typecheck；`npm run build --workspace obsidian-arxiv-daily`。
- [x] accepted on observed evidence

### Chunk 6 — 删除休眠增量代码与修过期文档
- change: 删 core `library/incremental/*`、`index.ts:32-36` 导出、`personal-library-direction-diff.system.md`、两个测试文件；删插件孤儿项（上文清单）、重复命令 `review-incremental-suggestions`、modal 里 `incrementalSuggestion*`/`DirectionDiffSuggestion`；`unclassifiedBufferPool*`/`bufferPoolHeading` 的 ADR 0007 术语命名保持不动（仅注释更新，避免大面积测试改名）。文档：technical-report 约 50（增量更新段，删）、86（Proposed/Confirmed、"buffer pool" 描述改为当前主题/方向模型）、88-90（代表论文/novelty 输入段，与 P2 已改内容对一次，不覆盖 P2）、272 行实为 papers.json 条目，grep 复核后确有过期再改；ADR 0004 §3 只加一条注记（同 §4 的 2026-10-15 注记格式：代表论文基准已由 ADR 0012/0015 取代）；CONTEXT.md 复核「Representative set」条目，若无矛盾不动。
- kind: removal + docs
- first signal: 在 `plugin/tests/commands.test.ts` 新增「不注册 review-incremental-suggestions」（当前红）；删除后 `grep -rn "IncrementalSuggestions\|DirectionDiffSuggestion\|INCREMENTAL_BUFFER_TRIGGER\|library/incremental\|direction-diff" packages plugin apps` 无输出。命令：`npm run test --workspace obsidian-arxiv-daily -- tests/commands.test.ts`。
- checks: `npm run typecheck`、`npm run check:boundaries`、core 全量 `npm test`、`npm run test:workspaces`；`git diff --check`；`rg 代表论文 docs/technical-report.md` 人工核。
- [x] accepted on observed evidence

## Phase verification

- `npm run typecheck`、`npm run check:boundaries`、`npm test`、`npm run test:workspaces`、`npm run build --workspace obsidian-arxiv-daily` 全绿。
- 真实 Obsidian 检查（goal 要求）：用真实库先生成并接受一次提案；往库里加 >=3 篇新 PDF；跑「索引」→ 看到一条提示（含按钮是否可点）；打开复审弹窗→ 主按钮显示 N；点更新→ 只出现新增论文对应的 covered/新方向/新主题；接受后再索引不再提示；加 2 篇不提示。记录 Notice 带按钮在真实 Obsidian 的表现，不行则降为纯文字+命令指引，并记 journal。
- 真实模型对一次真实新增（>=5 篇）读结果：单篇成组是否导致方向过窄（thin evidence 标记是否合理）；只记录不加阈值。
- CLI：`library index` 与 `library propose --new` 在临时库上人工跑一遍。

## Abort / reshape triggers

- 真实模型对「每篇单独成组」大量产出 1 篇方向、质量明显不如聚类 → 停，回用户：选项是批量聚类（需自定新增集合专用阈值，与 ADR 0014「不拍脑袋定阈值」冲突）或接受低质量。
- 要求把 schemaVersion 升到 7 才能表达 → 停（会作废所有已发布提案，用户未授权）。
- 超过 1000 篇的库导致提醒常亮且无法用同序同上限圈定 → 回用户。
- Chunk 4/5 需要动 P2 尚未验收的 main.ts/modal 区域导致冲突 → 等 P2 验收后再做。
- 计数需要读向量或触发模型/嵌入模型加载（索引完成后卡顿）→ 停（违反 ADR 0007 教训）。


## Owner 注（激活时）

- 计划起草于 P2 之前；P2 已改动 plugin/main.ts、apps/cli/src/runtime.ts、modal 与 Dashboard，执行各 chunk 前按当前代码复核行号。
- 证据组「每篇新论文单独成组」：组织合同允许多组归入同一提议方向，由模型决定新论文是否同属一条方向；单篇方向的薄证据标记在真实评测中观察，不加阈值。
- 真实模型评测沿用 P2 的做法（隔离配置、只读库、不发邮件、预算明示）。
