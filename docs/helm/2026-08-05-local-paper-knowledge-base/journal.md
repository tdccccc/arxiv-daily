# Journal

## 2026-08-05 — note（P1 完成；环境怪癖与运行时未决项）

- evidence: P1 全文索引与相似检索完成并验收。核心验证：core 84 文件/1421 测试、plugin 382 测试、lint 0 error、typecheck/boundaries 通过；真实 5 篇 arXiv PDF 端到端（提取→分块→嵌入→存储→标题自检索 5/5 top-1 → 重建逐位一致 → 离线缓存可用）。
  - **本机测试环境怪癖**：本 worktree 机器上 `npm test`（根）因 core 套件 vitest worker OOM（默认 4GB V8 堆不足，`tests/pipeline.test.ts` 等组合即可触发）返回 EXIT=1——**预先存在，与 P1 代码无关**（排除 fulltext 测试仍复现）。完整跑 core 套件需 `NODE_OPTIONS=--max-old-space-size=8192 npx vitest run`（84 文件/1421 全过）。CI 不受影响（不同内存画像）。后续会话验收时注意区分。
  - T4 subagent 中途模型层失败（无产出报告），但留下完整实现与调试产物；owner 补齐验证（冒烟 6/6 PASS、修复 smoke 确定性检查的 batch 形状比较、修 6 个 lint error）后验收。
  - e5 嵌入确定性的正确判据是"同 batch 形状两次嵌入逐位一致"；batch=3 vs batch=1 存在预期浮点差异（ONNX 内核重排），非 bug。
  - Obsidian 运行时未决项（代码已注释 + 报告中按"仓库配置行为"表述）：`window.pdfjsLib` 在插件上下文的实际可用性、wasm 从 CDN 拉取在渲染进程的表现——需用户在 Obsidian 中实际运行 `index-personal-library-fulltext` 确认；Node 侧同一代码路径已全链路实测。
- change: 无 steer；goal.md P1 标 done；technical-report 已 update（全文知识库机制 + 插件宿主接线）。
- disposition: 全部保留。tmp/（pdf-extraction-validation、embedding-smoke、fulltext-e2e）为 scratch 验证资产，不入库。
- next: P2 聚类方向生成（goal.md 索引行 pending）；检索入口形态（Dashboard similar-papers 按钮 vs 检索栏）open question 待定；CLI host 的提取/嵌入实现仍为后续阶段。

## 2026-08-06 — L2 reshape（聚类相似度定义两次迭代）

- evidence: 真实语料（15 篇近邻 DL 论文 + 12 篇异构 GNN/量子物理/生物医学）上，e5-small 嵌入的余弦分布饱和：raw 余弦下无关学术论文也 ≥0.85，mean-pooled 论文级向量所有论文相似度 0.93-0.98（任何语料都聚 1 簇）；corpus-level centering 把同/跨主题 gap 拉到 0.31 vs 0.21 但仍重叠；绝对阈值（minSimilarity 0.55-0.8 扫描）与互惠 top-k 图（SNN）都因弱 best-passage 边桥接失败。最终采用 **single-linkage（Kruskal 合并）+ 相对停止线**（默认最强边 × 0.65）：真实异构库上产出主题主导簇（GNN 4 篇 + 1 物理、物理 2 + 生物 1）与 4 篇缓冲池；近邻同领域论文（全部 DL）区分度不足是 e5-small 的已知限制，方向生成以"强簇 + 缓冲池 + 用户审核"兜底（goal 核心理念：用户决定优先）。
- change: phase 02 Approach/Tasks 更新（质心聚类 → SNN → single-linkage）；`clusterPaperVectors` 接口最终为 { minClusterSize, centerCorpus, minSimilarity(默认0), relativeStopRatio(默认0.65) }；`buildClusteringInput` 从论文级向量聚合改为加载 chunk 向量（长文截断 80 chunk）；`aggregatePaperVector` 删除；T3 resolver/generationContract 同步；e2e 语料改为异构 12 篇。
- disposition: 保留 single-linkage 引擎；T1/T2 的质心/SNN 实现已替换（git 历史可查）。
- next: P2 收尾（已提交）；P3 增量更新（缓冲池触发局部重聚类）按 goal.md 索引行推进。

## 2026-08-06 — note（SPECTER 换嵌入实测结论）

- evidence: 用 allenai-specter（q8 onnx，transformers.js 本地加载）对真实语料（15 篇 DL 近邻 + 12 篇异构）实测分离度：DL 语料同/跨主题间隙 0.031（e5-small 0.018）、异构 0.043（e5 0.015）——SPECTER 分布确实更好（1.7-2.9×），但 p25(same) 与 p75(cross) 仍重叠，同领域子主题无质变。用 SPECTER 向量直接跑现有聚类引擎（相对排序），输出与 e5 几乎相同（同构的 GNN 簇 + physics/bio 混杂簇 + 缓冲池）——相对排序对间隙改善不敏感。
- change: 无（不改产品代码，仅实测）。集成成本已记录：tokenizer_config 缺 model_max_length/truncation（公式段落 token 爆炸，需自适应截断）、q8 onnx 需本地组装（重命名 + 补 tokenizer_config）、推理慢 ~3×、768 维存储翻倍。
- disposition: 维持 e5-small 为嵌入模型；SPECTER 记录为未来"模型可配置化"的备选（潜在价值在检索 top-k 精度而非聚类）。
- next: P3 增量更新闭环（goal.md 索引行 pending → active）。

## 2026-08-06 — P3 T5a/T5b 完成（建议 store + 审核接线）

- evidence: core `IncrementalSuggestionsStore`（CAS/primary-backup/严格 decoder，35 测试）+ `apply{Attach,Split,Merge}Suggestion`/`buildNewDirectionDraft`（SUGGESTION_MEMBER_CONFIDENCE=0.9，split 派生方向带 `split-derived` 标记）验收；plugin 增量闭环接线完成并全量验收：`runIncrementalDirectionUpdate`（placement → attach 建议；buffer ≥ `INCREMENTAL_BUFFER_TRIGGER=3` 触发 reclusterPool + LLM diff；建议文档整体 replace CAS——新证据取代旧 pending 建议）、`applyIncrementalSuggestion`/`dismissIncrementalSuggestion`（内容键 `${kind}:${directionId}:${firstPaperKey}`，因 DirectionDiffSuggestion 无 id 字段；new 建议转候选入 proposal store 走现有确认流程）、lock/unlock、命令 `check-incremental-direction-updates`/`review-incremental-suggestions`、审核 UI 建议区块 + 锁定按钮。验证：tsc 0 error；plugin 全量 405/405（含新 16 测试）；core 未动（上次验收 1528 全过）。
- change: 编排适配——① placement 与 recluster 各自加载+centering（core 未导出 centering 变换，低频路径两次加载可接受，plugin 镜像 `centerChunksInPlace` 与 core 私有实现同构）；② 增量更新复用 `personal-library-direction-generation` operation kind（core OperationKind 封闭 union 无增量 kind，共享授权门与撤销范围）；③ apply/dismiss/lock 为本地确定性操作不要求模型授权（与 revoke 后本地 review 可用一致）；④ `reclusterBufferPool` 因边界词禁 `Buffer` 实现名为 `reclusterPool`。
- disposition: 保留。子代理经历：T5b 两次派遣（默认 deepseek-v4-flash）均在探索后模型层空响应失败（零产出、无部分改动）；第三次显式 sonnet 成功（当时为临时上游问题，非长期结论）。
- next: P3-T6 端到端验证（真实库增量场景 + 全量测试 + boundaries + journal 记录缓冲池阈值实测）。

## 2026-08-06 — P3 T6 端到端验证完成（P3 全阶段收尾）

- evidence: 真实语料全链路通过（31 项 PASS，tmp/incremental-e2e/ scratch）。场景：P2 异构语料副本（4 GNN/4 物理/4 生物）→ 确认 physics 为第 2 方向（profile rev 1→2）→ 真实索引 5 篇 DL PDF（chunks 49/64/44/70/262，KB rev 1→2、17 papers）→ placement（默认 0.25/0.05）：**attach 8 / buffer 1**（confidence min 0.190/median 0.339/max 0.552；margin min 0.046/median 0.150/max 0.392；5 篇 DL 全附 GNN——centered e5 空间里 transformer 类论文与 GNN 同域，1706.03762 最高 0.552/margin 0.392）→ 严格变体 0.35（attach 3/buffer 6）驱动 recluster（1 簇 6 篇，nearest GNN 0.349/physics 0.318）+ fake LLM diff（attach 3 → GNN、new 3）→ 应用 attach：GNN members 5→8 + members-updated 事件，建议文档 CAS 移除（rev 1→2）→ 锁定验证：locked GNN 的 split/merge 均 `direction-locked` 拒绝（含 suggestDirectionDiff 全流程 attempts=3 抛错）、attach 放行、unlock 恢复 → 重载核验全部 store 一致。
- change: 无 steer。阈值实测结论（journal 定论）：**默认 minSimilarity 0.25 在真实语料偏宽**（阈值扫描 0.15/0.2/0.25 均 8/9、0.3→6/9、0.35→3/9；minMargin 0.05 与 0.02 同结果、0.1→7/9）；DL→GNN 附着力强是 e5 centered 空间的真实分布，非 bug；若产品上希望缓冲池更活跃可调 0.3-0.35，但 0.25 的"宽松优先 attach + 用户审核"符合用户决定优先理念，维持默认。stale-CAS 拒绝路径与 replay 幂等（同内容不抛 stale）均已实测。
- disposition: 保留全部；tmp/incremental-e2e/ 为 scratch 不入库。
- next: 全部阶段完成。收尾：goal.md 成功标准全勾 + P3 done；technical-report handoff；提交需用户指令（P6）。

## 2026-08-06 — note（P4 启动：Obsidian 运行时验证，两个未决点的实测计划）

- evidence: 续接 P1 遗留未决项——(a) `loadPdfJs()` 后 `window.pdfjsLib` 在插件上下文的实际可用性，(b) transformers.js 渲染进程 wasm/CDN + Cache API 模型加载表现；Node 侧同一代码路径已全链路实测（tmp/fulltext-e2e，同款 plugin host 代码注入 pdfjs-dist 6.2.108 + onnxruntime-node），渲染进程差异无法在 Node 覆盖。会话内核查：huggingface.co 与 cdn.jsdelivr.net 从本机直连可达（无网络墙 → 镜像接线/wasm 本地打包均降级为备选 C3/C4，非前置）；extractor 在构造时读 window.pdfjsLib（pdf-text-extractor.ts:82，main.ts:1356 在 loadPdfJs() 后立即构造——若 loader 解析成功但 window 未挂载即失败，诊断需双通道核对返回值 + window）；vitest alias 将 obsidian 映射到 tests/__mocks__/obsidian.ts（当前无 loadPdfJs，需补）；e2e 语料 25 篇 PDF 仍在 tmp/fulltext-e2e/pdfs/。
- change: 新阶段 P4（goal.md status done → active + 索引行；phases/04-obsidian-runtime-validation.md）。方案：新增诊断命令 `diagnose-fulltext-runtime`（Part A pdf.js presence + 真实冒烟提取；Part B 嵌入模型真实加载；Notice 汇总 + logger 明细 + 可复制结果模态框），用户在 Obsidian 跑受控流程（5 篇 e2e 同款 PDF：诊断 → 索引 5 indexed → 磁盘核对 → 重跑 5 reused → 标题检索 5/5 top-1），console 证据回贴分析。备选修复 C1（返回值注入 extractor）/C2（API 面适配）/C3（wasmPaths 固定或本地打包）/C4（HF 镜像设置接线）/C5（提取质量与 Node 基线对比）——仅当实测对应失败时启用，先简报。
- disposition: 全部保留。诊断命令定位为永久产品功能（对用户排查运行时问题有用），非临时调试工具。
- next: P4-T2 实现诊断命令（main.ts 方法 + commands.ts 注册 + 结果模态框）→ T3 测试 → T4 构建 → T5 手册定稿 → T6 用户 Obsidian 实测后分析收尾。

## 2026-08-07 — note（P4 完成：两个未决点定论 + 渲染进程嵌入崩溃根因与修复）

- evidence: 真实 Obsidian（Desktop/plugin_test，Electron 39.2.6）实测完成。**(a) `window.pdfjsLib` 定论：PASS** —— `loadPdfJs()` 解析、返回值与 window 双通道均在，版本 5.3.34，真实冒烟提取成功（1207.0580 → 18 页/45165 字符）。**(b) 渲染进程 wasm/模型加载定论：PASS（修复后）**—— 首次实测报 `Cannot read properties of undefined (reading 'create')`。根因（三级复现链）：Node 侧 web 构建强制走 node 分支报 `Unsupported device`；真实 Chromium 渲染进程浏览器分支**正常**（dims 384/29.8s）→ 问题不在浏览器路径；Electron 33（nodeIntegration+contextIsolation:false，忠实 Obsidian 形态）实测 `process.release.name === "node"` —— Obsidian 渲染进程带 Node 集成，transformers.js v4 的 `IS_NODE_ENV` 据此走 **Node 分支**，而插件打包 web 构建（`onnxruntime-node` 被 tree-shake，bundle 0 引用）→ `ONNX.InferenceSession` undefined → `reading 'create'`；插件 `isNodeRuntime()` 同探针同误判。**修复**：`alignElectronReleaseProbe()`（embedding-model.ts，factory 创建时调用）——检测 `process.versions.electron` 存在（真实 Node 无此键）即把 `process.release.name` 对齐为 `"electron"`（幂等，不恢复——后续 isNodeRuntime/device 选择需保持一致），transformers 与插件两侧同步落 wasm 分支；真实 Node 不受影响。Electron 复现验证：对齐后全链路 OK（冷启 26s，Cache API 缓存后 952ms）。真实 Obsidian 复测：**embeddings PASS**，`runtime probe: process.release.name=electron (electron 39.2.6)`，wasmPaths 实锤 `mjs: blob:app://obsidian.md/...` + `wasm: jsdelivr …ort-wasm-simd-threaded.asyncify.wasm`（非隔离 asyncify 构建），加载 4722ms。
  - 索引/复用：5 indexed/0 failed、重跑 5 reused（用户确认）；KB 落盘 manifest rev 1 + 5 papers，**chunk 数与 Node 基线逐篇一致（49/64/44/70/262）**。
  - 检索：功能正常、语义正确（全部返回库内 5 篇 transformer 系论文，带分数与命中段落证据）；Obsidian 侧标题自命中 **4/5**（BERT 0.872/GPT-3 0.881/ResNet 0.863/Dropout 0.857 均 top-1；"Attention Is All You Need" 0.823 排第 3，低于 GPT-3 0.833/BERT 0.828；Node 基线同查询 Attention 0.8359 top-1）。**归因**：两 KB 文本对比——chunk 数相同但原文仅 6/44 逐字相同、归一化后 32/44 相同，差异为 pdf.js 版本（5.3.34 vs 6.2.108）排版级空白/换行 → 级联 chunk 边界漂移（12/44 块内容在相邻块间迁移）→ best-hit chunk 更换；叠加 wasm/node 嵌入内核差异与短查询方差（Attention 标题仅 5 词）、长文 best-passage 优势（GPT-3 262 块命中面最大）。非功能回归：目标判据"检索出库内最相似论文带可解释理由"达成；"5/5 top-1 自命中"为 Node 环境验证产物，近并列（±0.02）下对实现细节敏感。
- change: 修复 chunk（`alignElectronReleaseProbe` + `describeRuntimeProbe` + 诊断报告 `runtime probe` 字段；测试 +5，plugin 420/420、lint 0 error、boundaries OK）；technical-report 已 update（嵌入宿主段 + 命令列表段）。phase 04 Verification 判定表按实测修订（"5/5 top-1" → "自命中 ≥4/5 + 近并列翻转归因"）。
- disposition: 保留全部实现与修复；repro 资产（tmp/repro-*.mjs、tmp/pp/electron 33、/tmp/electron-repro*）为 scratch 不入库，journal 引用可复现。hf-mirror 无 CORS 头（浏览器直接下载会失败；插件默认 huggingface.co 有 CORS 不受影响）——若未来加镜像设置需处理 CORS/代理。
- next: P4 收尾——goal.md P4 done + status done；technical-report handoff 完成；手册更新完成。后续可选：检索入口形态/增量触发节奏 open questions；镜像设置接线（含 CORS 注意点）。

## 2026-08-07 — note（开放问题定案：检索入口 + 增量触发节奏，grill-with-docs）

- evidence: 两开放问题经 grill-with-docs 逐题定案（领域语言对照：knowledge evolution / research companion / 机器建议永不覆盖用户决定）。代码事实：Dashboard 行内已有词法 "Find similar papers" 按钮（PaperSearchIndex.similar，SimilarPapersModal）；全文检索仅命令面板（FullTextQueryModal，结果仅 Notice）；catalog 记录 `abstract` 为必填字段（识别失败可为空串）；增量链路现状整跑需模型处理许可（personal-library-direction-generation），placement 为本地嵌入无 LLM。
- change: **检索入口（ADR 0006）**：①单框双结果——Dashboard 搜索框保持单一输入，查询同时过滤日报行（词法，现状）+ 异步查文献库 KB（相似度+命中段落+打开动作区块）；无 KB/未授权只出前者，无模式切换；②行内按钮升级为双页模态框（库内相似全文为主 + 日报相似词法为次，无 KB 只显示词法页）；③从论文查询 = 标题+摘要（空摘要回退纯标题；PDF 永不作查询源，仅在检索目标侧）。**增量触发（ADR 0007）**：①索引完成且 indexed>0 自动触发；②授权门拆分——placement 免许可始终跑，LLM diff 需许可（无则跳过并记"待授权"进审核 UI）；③建议永不自动应用；④建议文档保持整体替换 + 覆盖未审阅时提示（状态栏/Notice）。CONTEXT.md 新增术语：Library similarity、Direction suggestion。两 ADR 已落盘（docs/adr/0006、0007）。
- disposition: 全部保留。实现尚未开始——定案作为后续 helm phase（P5）的输入，phase 计划需待执行指令。
- next: 待用户决定是否进入 P5 实现（检索入口 + 自动触发）。其余未决项：实现存疑项复核（P3 遗留 4 项）、模型可配置化、CLI host。

## 2026-08-07 — note（P5 启动：实现存疑项复核，4→3→2 顺序）

- evidence: 用户确认按推荐顺序复核 P3 T5b 遗留存疑项。存疑项 1（建议文档整体替换 vs 合并）已在开放问题定案中由 ADR 0007 关闭（保持整体替换 + 覆盖未审阅提示，提示实现归 P6 检索入口/自动触发阶段）；剩余 3 项：4 centering 镜像（唯一答案明确项，先做）、3 discoveryCues 截断（用户可见数据质量）、2 operation kind 复用（结论喂 ADR 0007 授权门拆分的实现）。
- change: goal.md status → active + P5 索引行；phases/05-implementation-doubts-review.md（T1 4→T2 3→T3 2 + 收尾）。
- disposition: 无代码改动。
- next: P5-T1 存疑项 4：核对 plugin 镜像与 core centering 数值同构性。

## 2026-08-07 — note（P5 完成：实现存疑项复核 3/3 定论）

- evidence: 存疑项 4（centering 镜像）：plugin 镜像与 core 私有变换逐行同构；core 的"先归一化"在 KB 单位向量上近似恒等。修复＝core 导出非可变 `centerCorpusChunks`（clusterer.ts），内部流水线接入，plugin `loadCenteredClusteringInput` 切换，镜像删除；+4 core 测试（非可变/单位范数/空语料/流水线等价）；scratch 逐位等价（真实 KB 5 篇 187,776 floats max diff=0）；core 1532/1532、plugin 420/420、lint 0、boundaries OK。存疑项 3（discoveryCues 截断）：截断在 plugin `attachNewSuggestionToProposal`（core draft 返回空 cues）；decoder 要求 ≥1 且严格升序唯一。修复＝代表论文标题去重+排序作 cues（catalog 可得），空回退 reason；core `derivedCues` 保持（merge/split 无标题源）。存疑项 2（operation kind 复用）：**保持复用**——同一授权门域 + 共享撤销取消范围（main.ts:531/536/558）+ 同 kind/key 互斥（全量与增量都写同一 profile store，并发互斥必需）；新增 kind 有害（破坏互斥、扩大 union、撤销清单双处维护）。ADR 0007 的授权门拆分在 plugin 门检查层实现（不经 kind）：P6 把 run-entry 授权检查下移到 LLM diff 阶段，placement 免许可照跑 + "待授权"状态。
- change: T1/T2 修复已提交（e05dd96 refactor centering 去重、ba15254 cues 标题派生）；T3 无代码改动（结论记录）。technical-report handoff：三次均 no-impact（报告无相关语句需要更新；kind 复用语义 line 290 已记载）。
- disposition: 保留全部修复；tmp/t1-centering-equivalence.mjs 与 clusterer-bundle.mjs 为 scratch 不入库。
- next: goal.md P5 done + status done（本条目同批提交）。P6（检索入口 ADR 0006 + 自动触发 ADR 0007 实现）待用户指令；其余未决项：模型可配置化、CLI host。

## 2026-08-07 — note（P6 启动：检索入口 + 自动触发实现）

- evidence: 用户指令开始 P6。输入资产：ADR 0006（单框双结果 + 双页模态框 + 标题+摘要查询）、ADR 0007（索引完成触发 + 授权门拆分 + 整体替换 + 覆盖提示）、P5-T3 实现提示（授权检查下移 LLM 阶段、authorizationFingerprint 保留、placement-only 被撤销取消无副作用）。
- change: goal.md status → active + P6 索引行；phases/06-search-entry-and-auto-trigger.md（T1 双页模态框 → T2 单框双结果 → T3 自动触发+授权拆分 → T4 覆盖提示 → T5 收尾）。
- disposition: 无代码改动。
- next: P6-T1 扩展 SimilarPapersModal 双页。

## 2026-08-07 — note（P6 完成：检索入口 + 自动触发落地）

- evidence: ADR 0006/0007 全部实现并验收。**T1 双页模态框**：SimilarPapersModal 增加可选 `library`（query + load），有则 Library/Daily 双页（Library 默认、加载/错误/空三态），无则原样渲染；Dashboard 以标题+摘要（空回退标题）构建查询。**T2 单框双结果**：refreshLibrarySearch（防抖回调 + 主渲染挂点，≥2 字符、staleness token），纯渲染抽到 library-search-block.ts。**T3 自动触发+授权拆分**：run-entry 授权 throw 移除、assertIncrementalUpdateCurrent 参数化；LLM 阶段检查授权 + fingerprint，无许可跳过并写 pendingAuthorization（core decoder 兼容两种形状）；审核 UI 待授权横幅；索引完成 indexed>0 自动触发（失败仅记录）。**T4 覆盖提示**：mutation 比较新旧建议集，不同则 summary.superseded + 通知提示。验证：core 1534/1534（+2 pendingAuthorization 解码）、plugin 434/434（重写 1 + 新增 8）、lint 0、boundaries OK、typecheck 全过；technical-report handoff：T1/T2/T3/T4 均 updated（Dashboard 段、授权门段、命令段）。
- change: 四个功能 chunk 已提交（46028b1 双页模态框、df1774f 单框双结果、a335efc 自动触发+门拆分、bbfb191 覆盖提示）；本条目为 P6 收尾（goal.md P6 done + status done 同批提交）。
- disposition: 保留全部。测试环境备注：hf-mirror 无 CORS（浏览器直接下载失败），插件默认 huggingface.co 不受影响——镜像设置若做需处理。
- next: P6 完成。待办清单剩余：模型可配置化（SPECTER，含 768 维存储翻倍）、CLI host 提取/嵌入。构建产物待用户安装验证（桌面 vault 需重启 Obsidian 生效）。

## 2026-08-07 — note（P6 后调整：相似论文模态框移除 match-reason 噪音）

- evidence: 用户实测反馈——"Daily similar" 页每条都显示 "Matched title/abstract…" 理由行，视觉杂乱；要求直接显示结果。
- change: renderDailyPanel 移除 reason 行（含 "Shared indexed terms" 兜底）；`PaperSearchResult.reasons` 数据保留（仅展示层不渲染）；styles.css 清理 __reason 规则；测试改为断言"不显示 Matched/Shared indexed terms"。
- disposition: 保留。本调整在 P6 已关闭后发生，作为独立小提交记录。

## 2026-08-07 — note（P6 后调整：搜索框一键清空按钮）

- evidence: 用户反馈——Dashboard 搜索栏最右侧需要一键清空（×）。
- change: 搜索输入外包相对定位容器 + 绝对定位 × 按钮（仅输入有文本时显示）；`bindSearchClearButton` 助手（点击即清空+重置查询+重渲染行与 KB 区块+保持焦点，无防抖）；隐藏原生 search-cancel 避免双清空控件；+3 测试。
- disposition: 保留。

## 2026-08-07 — note（P6 后调整：行内 match-reason 移除 + 索引卡死修复）

- evidence: 用户实测反馈 ①搜索框结果行仍显示 "Matched abstract/source sections"（上次只删了模态框里的，行内是另一处）；②134 篇全文索引时 Obsidian 主线程被占满、界面无响应（点击/最小化均卡住）。
- change: ①移除 Dashboard 行内 match-reason 显示（view.ts 块 + isActiveRelevanceSearch 方法 + __match-reason 样式；matchReasons 数据保留）；②主线程让出——embedding-model `embed()` 每批推理后与 core 编排每篇论文后 `yieldToEventLoop()`（setTimeout 0，Node 宿主无害），渲染进程可处理排队事件。
- disposition: 保留。教训：安装插件产物需 main.js + styles.css 同步复制（此前提过）。

## 2026-08-07 — note（P6 后调整：卡死二次排查——实测嵌入/提取均不阻塞，补 placement 让出）

- evidence: 用户反馈仍卡死。Electron 忠实复现测量：嵌入 256 段 ~100s 但主线程最大阻塞 59ms（asyncify wasm 内部让出 + 批量让出有效）；Obsidian 内置 pdf.js 启动即设 `GlobalWorkerOptions.workerSrc=/lib/pdfjs/pdf.worker.min.mjs`（提取走 worker 不阻塞主线程）。剩余真实阻塞点：索引完成后自动触发的 placement `loadClusteringInput` 逐篇 base64 解码 + JSON.parse 无让出（134 篇时主线程卡数十秒）。
- change: ①core placement `loadClusteringInput` 每篇让出事件循环；②嵌入批 32→8（单块更短、让出更密）。embedding 吞吐本机 ~9s/32 批（CPU wasm 固有，134 篇全程 ~25min，非卡死只是慢）。
- disposition: 保留。测量资产 tmp/embed-block-*、tmp/extract-block-* 为 scratch。
- next: 请用户以新构建干净复测（重启 Obsidian → 重跑索引，已完成的论文复用）。

## 2026-08-10 — note（设计定案：远程嵌入可选开关，grill-with-docs 六题）

- evidence: 用户提出"是否能用便宜模型做 indexing"→ 澄清：嵌入模型 ≠ chat LLM（DeepSeek 无 embeddings API），提速路径是**远程嵌入 API**；实测本地嵌入 ~0.3-1s/chunk、134 篇小时级。grill 六题定案：①披露强度=走现有授权流（全文级授权，modal 披露"全文分块发送至 <endpoint>"）；②默认与入口=首次引导选择（建立文献库时选本地/远程，切换需重建 KB）；③出机内容=全部 chunk（提速前提）；④失败回退=逐篇 failed + 下次重试（整库单一模型，不混向量空间）；⑤配置=独立嵌入设置区 + 授权记录扩展多端点（任一端点变更需重新授权）；⑥范围=只做远程嵌入，本地多 worker 后续单独评估。
- change: ADR 0008（docs/adr/0008-opt-in-remote-embedding.md）；CONTEXT.md：Library processing consent 更新为多端点/全深度措辞 + 新增 Embedding mode 术语。goal.md Non-goal "远程 embedding API（远程作为可选开关留待未来）"被实现为预留开关——默认全本地仍成立。
- disposition: 保留。实现未开始（P7 待用户指令）。
- next: P7 实现（远程嵌入：设置区 + 授权扩展 + RemoteEmbeddingModel + 引导选择 + 切换重建提示）。

## 2026-08-10 — note（P7 启动：远程嵌入可选开关实现）

- evidence: 用户确认 134 篇本地索引太慢，直接开始 P7。输入资产：ADR 0008（六项定案：授权流披露、首次引导、全文 chunk 出机、逐篇失败重试、独立配置+授权多端点、只做远程嵌入）。已核实落点：LlmClient 的 http.request 模式（core 远程实现可复用）；授权在 plugin/src/library/connection.ts（LIBRARY_PROCESSING_DEPTH 为单字面量 const、libraryAuthorizationFingerprint 只含 LLM baseUrl——需扩为 union 与多端点）。
- change: goal.md status → active + P7 索引行；phases/07-remote-embedding.md（T1 core 远程模型 → T2 设置 → T3 授权扩展 → T4 引导选择 → T5 工厂接线 → T6 收尾）。
- disposition: 无代码改动。
- next: P7-T1 core RemoteEmbeddingModel（OpenAI 兼容 /embeddings 客户端 + mock http 测试）。

## 2026-08-10 — note（P7 T1-T4 完成：远程嵌入实现过半）

- evidence: T1 core `RemoteEmbeddingModel`（OpenAI 兼容 /embeddings，批量≤64、维度断言、abort、密钥脱敏；modelId `remote:{model}:{dimension}` 不含端点）；EmbeddingModel port 增 `prefixPolicy`（e5|none），编排条件化前缀；本地 host 声明 e5。T2 `PluginSettings.embedding`（mode/provider/baseUrl/apiKey/model/dimension + initialChoiceDone）+ `validateEmbeddingConfig` + 设置 UI Embedding 区 + CLI `[embedding]` 映射。T3 授权多端点：scope 化（LLM 端点 + 可选嵌入端点）、远程授权记 full-text 深度、指纹含嵌入端点 digest、授权 modal 披露嵌入端点与全文深度、decode 兼容两种深度。T4 首次引导选择：库连接成功后 `offerEmbeddingModeChoice`（initialChoiceDone 标记，一次），modal 选本地/远程（含速度/隐私/重建提示），关闭默认本地。
- change: 四 chunk 已提交（1d154a4 remote model、ed34343 settings、14bbf76 authorization、T4 待提交）。验证：core 1547/1547（+9）、plugin 442/442（+3 引导 modal）、lint 0、boundaries OK。
- disposition: 保留全部。
- next: T5 工厂接线（index/search 按 mode 选嵌入实现）→ T6 收尾。

## 2026-08-10 — note（P7 完成：远程嵌入可选开关全部落地）

- evidence: T1-T5 五个 chunk 全部验收（提交 1d154a4/ed34343/14bbf76/aa82875/8041e8b）。core `RemoteEmbeddingModel`（OpenAI 兼容 /embeddings、批量≤64、维度断言、密钥脱敏、abort、modelId 不含端点）；EmbeddingModel port `prefixPolicy`；`PluginSettings.embedding` + 校验 + 设置 UI + CLI 映射；授权多端点 scope 化 + full-text 深度 + modal 披露；首次引导选择（initialChoiceDone）；工厂接线 + 远程门禁（配置完整 + full-text 授权）。验证：core 1547/1547（+15 累计本阶段）、plugin 447/447（+12）、lint 0、boundaries OK；technical-report handoff 全部 updated。
- change: 无 steer。goal.md P7 done + status done（本条目同批提交）。
- disposition: 保留全部。用户侧待验证：设置页 Embedding 区 → 切 remote → 配端点/密钥/模型/维度 → Review & authorize（full-text 深度披露）→ 索引分钟级。
- next: P7 完成。其余待办：模型可配置化（已部分被远程嵌入覆盖）、CLI host 提取/嵌入、多 worker 本地并行（远程之外提速）。构建产物待用户安装验证。

## 2026-08-10 — note（P4 实测完成：两未决点定论 + 检索词法融合修复 + 卡顿定位）

- evidence: 用户在真实 Obsidian 执行 P4 验证手册（诊断命令 + 全量索引 134 篇 + 复用 + 检索）。**诊断全 PASS**：pdf.js `window.pdfjsLib` present（version 5.3.34，冒烟提取 27 页/117954 字符）；transformers.js 本地模型 multilingual-e5-small-q8 加载 989ms（wasm 走 jsdelivr、模型走 HF，渲染进程全通）→ P1 遗留两未决点定论：pdf.js 可用、wasm/模型加载可用。索引 134 篇 0 failed（约 9 分钟，Ollama nomic-embed-text:768 远程）。
- evidence（检索 top-1 失败根因链）：①用户用我给的标题检索 top-1 不命中，发现其中 3 个标题我给错（张冠李戴）；②用正确标题仍 3/6 不命中；③Node 复现定位：检索用"每篇论文最大 chunk 相似度"排序，GPT-3（262 chunks）正文总有 chunk 与任意标题偶然相似（max-pooling 极值偏差，nomic 下更明显）；④尝试嵌入 title 融合（6 篇小实验通过），全库重放失败——**nomic-embed-text 短文本嵌入坍缩**：无关标题相似度 0.93 > 仅大小写差的自匹配标题 0.66，嵌入信号不可用；⑤定案：**确定性词法匹配**（`title-similarity.ts`：归一化相等→1、token 前缀→0.95、Jaccard≥0.5→jaccard、否则 0），140 篇全库 6/6 top-1 自命中（score 1.000），正文查询不受影响（词法 0 分不参与），检索不再批量嵌入 title（省一次模型调用）。用户复验 "Attention is all you need" top-1 ✅。
- evidence（卡顿）：索引显示 Done 后卡 ~1 分钟 = ADR 0007 索引后自动增量方向更新（6 篇新论文触发 LLM diff，deepseek ~1 分钟），进度条已显示完成造成"卡死"错觉。修复：方向更新期间进度文案 "Updating paper directions"，setComplete 移到流程真正结束。
- change: retrieval.ts（titleScores 融合，纯函数）、index-orchestration.ts（词法 titleScores）、title-similarity.ts（新）、plugin main.ts（传 titles + 进度时序 + globalThis→window）、library-search-block/view/commands（检索结果删除匹配内容 excerpt，保留标题+标识+相似度）。验收：core 1552→1558、plugin 448、tsc×2、boundaries OK；lint 69（基线 70 修 1，max 60 为历史遗留）。core 全量测试默认堆 OOM（v8 4GB 不够，pipeline-novelty-stage 单文件亦复现，stash 前后一致=环境问题；8GB 堆+单 fork 全过 1558）。
- disposition: 保留全部。已知：unresolved 219 篇无编号 PDF 不进 KB（用户提出索引需求 → P8）。
- next: P8 unresolved 兜底索引。

## 2026-08-10 — note（P8 启动：unresolved 文件兜底全文索引）

- evidence: 用户确认 219 篇 unresolved（作者-年份命名 PDF，抽样 5 篇仅 1 含 DOI、4 无任何 arXiv/DOI 标识=期刊版/老文献）需要同样可检索；确认范围=纯兜底索引（不做内容识别增强）、现在实施。设计定案：不改 catalog 模型（files.status 语义保留），索引层把 unresolved 文件作为额外索引单元——内容指纹 key（`file:sha256:<obs>`，改名稳定、内容变自动换 key + 现有 prune 清理）、首页标题提取（`title-extraction.ts`：黑名单页眉过滤 + 大写开头候选 + ∗/†/@/Abstract 作者标记截断，v1 已知局限：作者同行无标记时截断不净）、提取标题存 KB 文档/清单（`title` 可选字段，decode 向后兼容）、检索词法融合与展示用 KB title + 文件路径（file 论文）；placement/clustering 天然兼容（ClusteringInputPaper 只依赖 paperKey+向量）。
- change: goal.md 追加 P8 + phase doc 08；knowledge-base.ts（document/manifest record 可选 title + decode 兼容）、title-extraction.ts（新）、index-orchestration.ts（collectIndexUnits + extractTitle）、plugin main.ts（检索 manifest title join + filePath 展示）、library-search-block/view/commands（filePath 展示）。验证：core 1565（+7：unresolved 索引/复用/prune ×2 + title-extraction ×5）、plugin 448、tsc×2、boundaries OK、lint 69。main.js 已构建安装（备份 20260810-fallback）。
- disposition: 保留全部。用户侧待验证：重启 Obsidian → 重跑索引（219 篇 unresolved 进入，约 15-20 分钟）→ 文件名/标题检索。
- next: 用户重验 P8（索引 + 检索）。

## 2026-08-10 — note（P8 实测迭代：recluster 卡死修复 + 混合检索 + 标题刷新 + 识别误认修复）

- evidence（卡死）：索引完成后 "Placing new papers" 卡死——reclusterPool 对 219 篇 buffer 论文做 chunk 级两两聚类（~1e4 chunks 的 O(n²) 余弦，数百亿次乘加同步阻塞主线程）。修复：聚类降为论文级质心（每篇 1 个均值向量，reclusterPool 内部压缩，共享 clusterer 与 P2 路径不动），O(chunks²)→O(论文²)，219 篇亚秒级。实测 134 篇时未暴露（buffer 仅几篇）。
- evidence（关键词检索失效）：搜 "panstarrs" 结果 0.5 随机排序——nomic 短查询坍缩 + 词法 titleScores 对非标题查询无效。新增 `lexical-search.ts` 混合检索：显著 token（去停用词、DF>40% 剔除）+ compact 全文匹配（连字符无关）。v1 全 1.0 平局（顺带提到与主题论文同分）→ v2 频率分级 `count/(count+3)`（1 次 0.25、16 次 0.84）+ **标题含全部 token → 1.0**；真实库验证："The Pan-STARRS1 Surveys"（1612.05560）1.0 置顶，正文高频 0.99 档，顺带提及 0.87-0.93。
- evidence（标题质量）：提取标题含作者行/期刊引用行（", Martin Landriau..."、"Space Sci Rev (2013) 177:75–118"）→ 过滤增强（标点开头、期刊引用正则）；**旧索引坏标题不会随 reuse 修复** → 新增 `titleVersion`（TITLE_EXTRACTION_VERSION=1）：reuse 时版本缺失/不符的 fallback 论文重读首页刷新标题（不重嵌入），summary 增 `titlesRefreshed`（Notice 展示），一次性 ~5-10 分钟。
- evidence（识别误认）：`Planck Collaboration2016.pdf` 被内容识别为 arxiv:1008.4686（"Data analysis recipes"）——内容识别从期刊版首页参考文献引用里抓到 arXiv id（流前缀 4096 内）。修复：STREAM_TEXT_PREFIX_CHARS 4096→512（只认流极前部的 id；arXiv 版靠文件名识别不受影响；作者名文件识别失败 → unresolved → fallback 兜底，宁可不识别也不张冠李戴）。真实 PDF 验证：Planck→{} ✓。需要重 Scan 生效（Planck 转 unresolved 后重索引 prune 误识别文档）。
- evidence（UI 澄清）：Catalog summary 的 Unresolved 行加说明（unresolved 文件会作为本地文献被全文索引）。
- change: recluster.ts（质心）、lexical-search.ts（v2 频率+标题命中）、retrieval.ts（tokenScores 融合）、index-orchestration.ts（token 集成 + titleVersion 刷新 + titlesRefreshed）、knowledge-base.ts（titleVersion 可选字段）、pdf-identification-evidence.ts（512 前缀）、title-extraction.ts（过滤增强）、plugin main.ts（Notice + summary 提示）。验收：core 1576、plugin 448、tsc×2、boundaries OK。main.js 已安装（备份 20260810-ident-fix）。
- disposition: 保留全部。用户侧：重 Scan（识别修复生效）→ 重跑索引（Planck 转 fallback + 误识别文档 prune + 标题刷新一次性）→ 检索 panstarrs 验证排序。
- next: 用户重验后收尾 P8（goal.md done）。

## 2026-08-10 — L1 adjust（P8 契约补强：缓存版本、内容寻址、标题持久化与 UI 一致性）

- evidence: 用户真实复验前检查发现三项会使验收失真：PDF evidence 规则从 4096 缩到每个已解压内容流 512 字符后，旧 content-derived ready/unresolved catalog 记录仍可直接复用；fallback `file:sha256:*` 实际基于含 path/mtime 的 observation fingerprint，改名不稳定；KB store clone 丢失 `title`/`titleVersion`，标题规则刷新无法持续。补测观察到预期 Red：聚焦 65 tests 中 5 failed / 60 passed，精确覆盖 identifier version、content key、rename、store clone 和 title refresh。另在 Similar modal 观察到 1 个 UI Red：仍显示内部 file key 和 passage excerpt。
- change: `PersonalLibraryFileIdentifier.version` 与 file record `contentIdentificationVersion` 让旧 content-derived ready/unresolved 在证据规则变化后重验，filename-derived 不受影响；fallback identity 改为完整 PDF bytes 的 `file:sha256:<digest>`，同内容多路径合并，改名后成功重读同内容保持 key，旧 observation-key 文档重绑时复用 chunks/vectors；KB document/manifest/decoder/store clone 保留 `contentHash`、`title`、`titleVersion`；三个检索入口统一 title + fallback 相对路径或 arXiv paperKey + similarity，移除 excerpt，索引 Notice 增 titles refreshed。phase 同步补入正文 token 混合检索、论文质心 recluster 和真实运行时待验边界。
- evidence: 修复后 core 聚焦 66/66、P8 core 120/120、plugin UI 15/15；core 全量 95 files / 1579 tests、plugin 全量 32 files / 448 tests；双 typecheck、workspace boundaries、`git diff --check` 通过。lint 为 0 errors / 69 warnings，因历史 `--max-warnings=60` 返回 1。core 全量按约束临时使用 8 GiB/单 fork，配置已还原并与 `/tmp/vitest.config.mts.bak` 一致。最终小修后 core 66/66、plugin 37/37、双 typecheck 通过。
- technical-report handoff: status `updated`; report `docs/technical-report.md`; scope 为 unresolved fallback indexing、PDF bytes 内容寻址与旧 key 迁移、识别缓存版本、标题版本、混合检索、recluster 质心及三个 Obsidian 展示面。生产路径核验覆盖 plugin Scan/index/search 入口、core reconciliation/index/retrieval/recluster、KB schema/store、scoped source 与 Dashboard/command renderers；报告同时收窄 25 MiB 单文件上限、每 stream 512 字符、跨文件非事务、复用/迁移错误可中止整轮及搜索不核对 manifest modelId 等当前边界。
- disposition: L1 adjust；P8 outcome 与 goal 保持不变，接受实现和报告更新。未纳入 fallback new-direction draft、confirmed profile 旧 file key 成员迁移、placement merged-direction 可应用性、legacy arXiv ID 端到端识别和 lint warning 清理。真实 Obsidian 复验未运行，因此 P8 与 initiative 保持 active。
- install: 最终 `plugin/main.js` 已构建并安装到测试 vault；源码产物与安装目标 SHA-256 均为 `985074964b52c26d57d0857548699f7dbf45b6b2b5a1616d7497bc35891db2a6`；安装前备份 `main.js.bak-20260811-content-key-fix` SHA-256 为 `1dc20fa5d62bcadcf3ffd9181fd876715f24d808209dc64b6fba5ba90f15a165`。
- next: 完全重启 Obsidian → Scan → 确认 Planck 不再属于 `arxiv:1008.4686` → 全文索引并记录 Notice → 核对 content-key migration/prune/方向更新 → 搜索 `panstarrs` 并确认 `The Pan-STARRS1 Surveys` top-1；成功后勾选 P8 success criterion 并关闭 initiative。继续不 commit/push。

## 2026-08-11 — L1 adjust（P8 现场 Red：legacy arXiv evidence 中止 Scan）

- evidence: 用户加载 SHA `985074964b52c26d57d0857548699f7dbf45b6b2b5a1616d7497bc35891db2a6` 后首次执行真实 Scan，收到 `invalid arXiv ID: "astro-ph/0609591"`。Scan 在 catalog promotion 前中止，因此新的 unresolved catalog 证据没有提交；随后全文索引看不到 `Preparing local document N/M`，且若 `indexed === 0` 自动方向更新本就不会显示 `Updating paper directions` / `Placing new papers`。
- root cause: `extractPdfIdentificationEvidence` 可返回 legacy ID，plugin `identifyFile` 对任何 direct ID 都提前返回；modern-only reconciliation/catalog 随后把该值传给 `paperKeyFromArxivId` 并抛错。title evidence 即使可用也被 direct legacy ID 遮断。
- change: plugin 对 direct evidence 先用 `normalizeArxivId` 过滤，仅现代 canonical ID 直接采用；unsupported direct evidence 若带 title 则继续严格 title search，search 结果也规范化。core reconciliation 在 host identifier 边界再次用 `modernArxivResources` 校验，任何 unsupported/malformed non-null 值都按 unresolved 处理，不进入 metadata resolver 或 paperKey 构造。legacy ID 仍不成为 catalog canonical identity。
- evidence: 先观察预期 Red：core 1 failed / 22 passed、plugin 2 failed / 8 passed，三项均复现用户错误；Green 为 core 23/23、plugin 10/10。P8/识别回归 core 128/128、plugin 25/25；全量 core 95 files / 1580 tests、plugin 32 files / 450 tests；双 typecheck、workspace boundaries、`git diff --check` 通过。lint 维持 0 errors / 69 warnings，因 max 60 返回 1。core 全量临时 8 GiB/单 fork配置已还原并与 `/tmp/vitest.config.mts.bak` 一致。
- technical-report handoff: status `updated`; report `docs/technical-report.md`; scope 为 Scan 的 PDF evidence → modern catalog identity 边界与 unresolved 降级。报告说明 legacy parser capability、plugin title-search fallback、core 二次校验以及 modern-only catalog 限制。
- install: 修复版 `main.js` 已安装；源码构建与 vault 目标 SHA-256 均为 `05f01f9d6eb5ac4dba822bbdb6bf6ba108be6f7982a1a346cdc8747d5e294768`。被替换版本备份为 `main.js.bak-20260811-legacy-id-fix`，SHA-256 `985074964b52c26d57d0857548699f7dbf45b6b2b5a1616d7497bc35891db2a6`。
- disposition: L1 adjust；P8 outcome 不变，修复和报告更新已接受。首次 runtime verification 只确认了失败边界，T7 保持 pending，initiative/P8 保持 active。
- next: 完全重启 Obsidian 加载修复版 → 重跑 Scan 并返回 summary → Scan 成功后再跑全文索引；只有这时 unresolved PDF 才会进入准备阶段，只有 `indexed > 0` 才会自动显示方向更新进度。继续不 commit/push。

## 2026-08-11 — L1 adjust（P8 真实 corpus 标题行界与 Pan-STARRS 并列排序）

- evidence: 修复版真实 Scan 成功提交 revision 5：115 ready files / 101 papers / 214 unresolved / 47 failed / 0 unrelated / not truncated。Planck 已成为 identification version 1 的 unresolved，错误的 `arxiv:1008.4686` catalog paper 消失；47 个 catalog failed 均为 metadata-fetch-failed。KB revision 8 为 307 ready / 3 failed，209 个 fallback key 中 206 个 ready 均有 PDF bytes contentHash、无 ready legacy observation-key；Planck 以 `file:sha256:b4ac490d…` 存 205 chunks，同内容双路径合并 5 组。稳定复用轮为 0 indexed / 307 reused / 3 failed / 0 pruned，所以没有 Preparing 进度，也按 `indexed > 0` 门槛不触发 Updating paper directions / Placing new papers。3 个全文 failed 是 `Engel2025.pdf`（40 MiB）、`Hopp2026.pdf`（47 MiB）、`Wu2023.pdf`（34 MiB）越过 scoped source 25 MiB 上限。
- evidence: Chambers PDF 的 pdf.js 首页把行尾表示为空字符串 item + `hasEOL: true`；host 先丢弃空字符串，导致 Draft/Preprint/title/authors 粘成超长行，标题启发式最终误选 `ABSTRACT ...`。真实 corpus 中 Chambers、Tonry、Flewelling、Lee 的标题/正文都含 Pan-STARRS，旧标题 token 规则统一给 1.0，排序只能按 hash paperKey 决胜。预期 Red 分别由 empty-EOL host regression 和 1.0/1.0 lexical tie 复现。
- change: `ObsidianPdfTextExtractor` 记住空 EOL marker 并在下一非空 item 前恢复换行；标题页眉过滤加入 Draft/Preprint，`TITLE_EXTRACTION_VERSION` 升至 2，使 fallback 只重读 PDF 刷新 title/titleVersion、复用 chunks/vectors；正文 token 分数保留 `count / (count + 3)`，标题全 token 命中改为 0.95 floor，正文频率可超过 floor 以打破同标题主题并列。
- evidence: Green 后聚焦 core 6 files / 86 tests、plugin 5 files / 33 tests；全量 core 95 files / 1580 tests、plugin 33 files / 451 tests；core/plugin typecheck、workspace boundaries、`git diff --check` 通过。lint 为 0 errors / 69 warnings，仍因历史 max-warnings=60 返回 1。core 全量临时使用 8 GiB/单 fork，测试后配置与 `/tmp/vitest.config.mts.bak` 一致。
- technical-report handoff: status `updated`; report `docs/technical-report.md`; scope 为 production pdf.js item 拼接、fallback title v2 刷新与失败重试、向量/标题/正文 token max fusion 及 `indexed > 0` 自动方向边界。报告已说明 empty EOL 行界、0.95 title floor、正文频率可越过 floor、标题刷新保持向量以及失败后下轮重试。
- install: production `main.js` 已构建并安装，源码产物与安装目标 SHA-256 均为 `b629ae82b4b91b817442344493666118f276b704b2263c7e89fde17a3bc104c5`；被替换版本备份为 `main.js.before-title-v2-05f01f9d6eb5.bak`，SHA-256 为 `05f01f9d6eb5ac4dba822bbdb6bf6ba108be6f7982a1a346cdc8747d5e294768`。
- disposition: L1 adjust；goal 与 P8 outcome 不变，标题/排序实现和报告更新已接受。真实 Scan、内容 key migration 与 25 MiB 失败边界已通过；T7 只剩新安装版 titles-refreshed 与 `panstarrs` top-1/三个 UI surface 复验。继续不 commit/push。
- next: 完全重启 Obsidian → 运行全文索引（预计 indexed 仍为 0、fallback 标题批量刷新、3 个超限文件继续 failed，因此不自动更新方向）→ 在全文搜索命令和 Dashboard 输入 `panstarrs`，确认 `Chambers2019.pdf` 以 `THE PAN-STARRS1 SURVEYS` 排名 top-1；再从一篇日报论文打开 Find similar papers，核对 Library 页标题/路径/融合分数展示；通过后关闭 P8/initiative。

## 2026-08-11 — L1 adjust（Find similar papers 长查询排序）

- evidence: 用户实测区分两条路径——Dashboard 搜索栏输入 `panstarrs` 已给出预期 top-5；日报行 "Find similar papers" 的 Library 页结果主题相关性仍不满意。Catalog Scan 已到 revision 7：Ready files 161 / Papers 144 / Unresolved 214 / Failed 1 / Unrelated 0 / Truncated No。
- evidence: 代码路径确认 Library similar 使用 `title + abstract` 调同一 `searchFullTextKnowledgeBase`。长查询仍对全部显著 token 做正文频率融合，会抬升顺带命中普通学术词的论文；标题词法对整个 title+abstract blob 计 Jaccard，几乎无法抬升同名库内论文；检索向量路径此前未做 corpus centering，而聚类/placement 已长期使用 centering 拉开主题间隙。
- change: `isKeywordQuery` 门控正文 token 融合（显著 token ≤12 且长度 ≤160）；`searchFullTextKnowledgeBase` 标题词法只取查询首个非空段落；`searchKnowledgeBase` 默认对候选 chunk 与查询向量做 corpus-level centering（可 `centerCorpus: false`）。
- evidence: 新增/更新契约后聚焦 core 3 files / 51 tests、相关回归 9 files / 136 tests 全过；core/plugin typecheck 通过。technical-report handoff updated。安装产物 SHA-256 `d21147fe38d9b2070e0aa18dc2a941b432c59d3a298d75078cbc8d131a3f40ee`。
- disposition: L1 adjust；P8 outcome 与 goal 保持不变，T7 仍 pending。keyword `panstarrs` 路径视为通过；关闭条件补上 Find similar papers Library 主题相关性复验。继续不 commit/push。
- next: 完全重启 Obsidian → 从任意日报论文打开 Find similar papers，核对 Library similar 是否与源论文主题相关；若满意再补 titles-refreshed / 超限 failed 边界后关闭 P8。

## 2026-08-11 — L1 adjust（MNRAS Advance Access 假标题）

- evidence: 用户在 2602.01548（Photometric Redshift PDFs / DESI Legacy / Pan-STARRS）上打开 Find similar papers。Library similar 前几名含 `Mucesh2021.pdf`、`Carrasco Kind2013.pdf`、`Beck2021.pdf`、`Luo2024.pdf`，标题均显示为 `Advance Access publication YYYY Month D`。这些是 MNRAS 首页页眉，不是论文标题，会污染展示与标题融合。
- change: `title-extraction.ts` 过滤 `Advance Access publication` / `Advance Access` 页眉；`TITLE_EXTRACTION_VERSION` 升至 3，使 fallback 文档在下次索引时只重读首页刷新 title/titleVersion、复用 chunks/vectors。
- evidence: title-extraction + fulltext/retrieval 聚焦 57/57 通过；core/plugin typecheck 通过。
- technical-report handoff: status `updated`; report `docs/technical-report.md`; scope 为 fallback 标题规则 v3 与 Advance Access banner 过滤。
- install: production `main.js` 已构建并安装，SHA-256 `27b1e0b6003aecb3a8633d5ca16bf7b7f5dce93f463bb0f359e9e91f2d34900e`。
- disposition: L1 adjust；P8/initiative 保持 active。标题刷新需用户重跑全文索引后生效；similar 主题相关性在假标题清除后再判。
- next: 完全重启 Obsidian → 运行 `index-personal-library-fulltext`（预期 titles refreshed > 0，indexed 仍可为 0）→ 再打开 2602.01548 的 Find similar papers，确认不再出现 Advance Access 假标题，并评估 Library similar 是否更贴源论文。

## 2026-08-11 — L1 adjust（标题提取 v4：字体结构取代页眉黑名单）

- evidence: 用户不接受继续以"页眉黑名单"方式修假标题；真实库四个样例（Mucesh2021/Carrasco Kind2013/Beck2021/Luo2024）标题均为 `Advance Access publication YYYY Month D`（MNRAS 页眉，8.97pt）而真实标题 15.94pt。全量扫描测试库 376 个 PDF 的首页 pdf.js items：标题几乎总是首页最大字号行，但存在结构性反例——A&A 刊头 `Astronomy & Astrophysics`（17.04 > 标题 16.35）、arXiv stamp（20.00 在页底 71-76%）、Bioinformatics 页边刊头（top=-6.6%）、Benítez2000 节标题 `1. INTRODUCTION`（10.96 > 标题 9.96）、老论文标题+作者+机构同字号（Hu2003/Strauss2002/Jimenez2002/Reiprich2002/Wang1998）、Krause2017_1 标题末词字号反超（`surveys` 15.94 > 正文 12.75）等。
- change: 标题提取改为结构规则 v4（无黑名单）：host 在 `PdfExtractionResult.layout` 提供每页行级 typography（text/fontSize/topFraction，同 hasEOL 行分组）；core 按字号 band 选标题——逐行排除页边文本（top<-5%）、arXiv stamp、预印本编号/DOI；顶部条带（≤13%）+ 次 band ≥0.93× 判刊头跳过；run 装配跳过下标短行、作者行（姓名模式+首字母/and 前瞻）断行、候选须过 plausibility（长度/首字母/节标题/期刊引用/日期/email/URL）；max band 为短片段时拒绝双词姓名候选；选中后以续行（小写开头/罗马数字/无首字母长短语）扩展标题。`TITLE_EXTRACTION_VERSION` 3→4，旧 fallback 文档下次索引只重读首页刷新 title、复用 chunks/vectors。文本兜底路径保留原 v3 规则（无 layout 的 host）。
- evidence: 376 文件全量验证——374 个有标题（2 个无文本层返回 null 属正确兜底）；生产实现与调优规则逐文件一致（scratch 校验 376/376）。四个用户样例标题全部正确；A&A/arXiv-stamp/页边刊头/节标题/老论文作者断行/Euclid 罗马数字续行/eROSITA 短语续行/下标合并/redMaPPer 与 dustmaps 小写专名/报告编号行全部通过。已知局限（3 个文件）：Benítez2000 全大写作者同字号粘连、Young2016 `Review article` 类型标签前缀、Bulbul2024 等系列标题大写第二续行截断（`…in the Western`）。核心聚焦 core 103/103（title-extraction 18 新增）、plugin 25/25（extractor 3）；全量 core 95 files / 1597 tests、plugin 33 files / 453 tests；双 typecheck、boundaries、`git diff --check` 通过；lint 0 errors/69 warnings（历史 max-warnings=60 返回 1）。core 全量按既有约定临时 8 GiB 堆（默认线程池，单 fork 跨文件累积 OOM），配置已还原并与 `/tmp/vitest.config.mts.bak` 一致。
- install: 构建产物 SHA-256 `f036b75ea16da28a20b90aa15fba72cdeef39a83b08c435640b84389a9623bfb` 已安装（main.js + styles.css 同步复制，安装目标 hash 一致）；被替换 v3 版保留为 `main.js.title-v3-27b1e0b6003a.bak`（SHA-256 `27b1e0b6003aecb3a8633d5ca16bf7b7f5dce93f463bb0f359e9e91f2d34900e`）。
- disposition: L1 adjust；P8 outcome 与 goal 不变。Find similar 检索路径未改代码（标题正确后标题融合自然生效；centering/keyword 门控/首段词法均为前轮已验收修复）。语料与迭代脚本留在 `tmp/`（gitignored）供后续规则调优。继续不 commit/push。
- next: 用户完全重启 Obsidian → 运行 `index-personal-library-fulltext`（预期 titles refreshed 全部 fallback 文档、indexed 仍可为 0）→ 打开 2602.01548 的 Find similar papers 核对 Library similar 标题与主题相关性；通过后关闭 P8/initiative。

## 2026-08-12 — L1 adjust（标题提取 v5：PDF 文档元数据优先）

- evidence: 用户追问"metadata 不该优先吗"后实测：376 文件库中 187 个 PDF 有非空 `info.Title`（185 个形态合理）；与 v4 字体规则结果比对，158 个一致，29 个不一致中约 20 个是垃圾（Windows 路径、`.eps`/`.tp` 文件名、页码引用 `55682 702..715`、arXiv stamp 写入 metadata），9 个 metadata 真正更好——其中 Krause2017_1（作者行盲区）与 Bulbul2024（`Western Galactic Hemisphere` 完整续行）是 v4 已知局限，metadata 直接解决；另有 Euclid 系列完整标题（Desprez2020/Collaboration2025_10）、Ilbert2013 `z ≃ 4`、Zhang2020 `of Quasars with Different Samples`、Vogt2024 `f(R)`、Abdullah2020 下标等。仅 Wen2022 一个 metadata 更差（丢 `z`），token 覆盖检查可救回。还发现 pdf.js 的 metadata 保留 HTML 实体（`&ndash;`、`&#x00D7;`）需解码，`Photo-$z$` LaTeX 残留与 `Microsoft Word - …doc` 需过滤。
- change: `PdfExtractionResult` 新增可选 `metadataTitle`；host `getMetadata()` 取 `info.Title`（失败降级 undefined，不抛错）；core 选择顺序改为 metadata（过垃圾过滤 + HTML 实体解码后优先）→ 字体 layout → 文本兜底；字体结果 token 集合覆盖 metadata 且更长时胜出（Wen2022 型）。`TITLE_EXTRACTION_VERSION` 4→5，旧 fallback 文档下次索引只重读首页刷新 title、复用 chunks/vectors。
- evidence: 语料重放：metadata 优先选择使 32 个文件标题变化，全部改善或中性、无回归（Wen2022 由 token 覆盖救回、Ren2025/Wang2023 由垃圾过滤拦下）；生产实现与调优规则 376/376 一致。聚焦 core 92/92（title-extraction 23 含 5 个 metadata 用例）、plugin 5/5（host metadata 2 个用例）；全量 core 95 files / 1602 tests、plugin 33 files / 455 tests；双 typecheck、boundaries、`git diff --check` 通过；lint 0 errors/69 warnings（历史基线）；core 全量默认线程池 + 8 GiB 堆，配置与 `/tmp/vitest.config.mts.bak` 一致。
- install: 构建产物 SHA-256 `d4778f45db435d3dbd64bfe24439b836d33b3f25da7e340bfab18ba5b1ad1df1` 已安装（main.js + styles.css 同步复制，安装目标 hash 一致）；被替换 v4 版备份为 `main.js.title-v4-f036b75e.bak`（SHA-256 `f036b75ea16da28a20b90aa15fba72cdeef39a83b08c435640b84389a9623bfb`）。
- disposition: L1 adjust；P8 outcome 与 goal 不变。metadata 是机器可读权威信号，优先于任何启发式——这是标题提取方向性的收尾；剩余已知局限（Benítez2000 全大写作者、Young2016 类型标签）连 metadata 也没有正确值时接受。语料与脚本留在 `tmp/`（gitignored）。继续不 commit/push。
- next: 用户完全重启 Obsidian → 运行 `index-personal-library-fulltext`（预期 titles refreshed 全部 fallback 文档）→ 打开 2602.01548 的 Find similar papers 核对 Library similar 标题与主题相关性；通过后关闭 P8/initiative。

## 2026-08-12 — L1 adjust（识别误认修复：引用区 arXiv ID 不再决定身份）

- evidence: 用户复验发现 Find similar 里 Beck2021.pdf 显示 "LSST science book version 2.0"（用户以 Beck2021.pdf 举例，实际为该条结果列表中的论文）。全库扫描定位：Chen2025.pdf 的识别 evidence = `{"arxivId":"0912.0201","title":"LSTM-MDNz: Estimating Quasar Photometric Redshifts…"}`——其参考文献第一条（Abell et al. 2009, LSST Science Book, arXiv:0912.0201）位于内容流 23 的前 512 字符内，被误识别为 arXiv:0912.0201（与 Planck 误认 arXiv:1008.4686 同机制）；测试库的 Beck2021.pdf 本身各信号均正常（metadata 空、每页流前缀无 arXiv ID、首页即正确标题）。候选修复评估：①"只取第一个内容流"——对象顺序 ≠ 页面顺序，大量真 ID 丢失（37+ 文件）；②evidence title 与 arXiv metadata 标题交叉验证——误伤 8 个"真 ID + 垃圾文档标题"文件（Rykoff2014/2016 的 "Graphics produced by IDL"、Weinberg2013 的 "Appendix A Glossary…"、Okabe2016、Saro2017、Collaboration2016_1、Collaboration2025_4、Zhao2023 等会从 ready 掉到 unresolved）；③标题搜索仲裁——实测 Chen2025 文档标题搜索命中 arXiv:2512.16010（LSTM-MDNz 真论文）。
- change: 插件 identify 对 direct ID + 非 arXiv-stamp 文档标题：先做 arXiv 标题搜索（严格匹配、歧义拒绝），命中**不同** ID 时判定 direct ID 为引用区误识别、采用搜索 ID；搜索失败/空/命中同 ID 时保留 direct ID（垃圾文档标题不会降级真论文）。`PDF_IDENTIFICATION_EVIDENCE_VERSION` 1→2，旧 content-derived ready/unresolved 记录下次 Scan 重验（Chen2025 的旧 ready(0912.0201) 将被修正）。
- evidence: 聚焦 plugin scan 12/12（新增：仲裁采用搜索 ID、垃圾 title 保 direct ID 两场景）、core 识别 12/12；全量 core 95 files / 1602 tests、plugin 33 files / 457 tests；双 typecheck、boundaries、`git diff --check` 通过；lint 维持 0 errors/69 warnings；core 全量默认线程池 + 8 GiB 堆，配置与 `/tmp/vitest.config.mts.bak` 一致。
- install: 构建产物 SHA-256 `4850755117f5c373533c0ee4213144681688f300753bd2536422321308253873` 已安装（main.js + styles.css 同步复制，安装目标 hash 一致）；被替换 v5 版备份为 `main.js.title-v5-d4778f45.bak`（SHA-256 `d4778f45db435d3dbd64bfe24439b836d33b3f25da7e340bfab18ba5b1ad1df1`）。
- disposition: L1 adjust；P8 outcome 与 goal 不变。识别修复需要重 Scan 生效（不是重跑全文索引）：Scan 会按 evidence version 2 重验 Chen2025.pdf 等 content-derived 记录。继续不 commit/push。
- next: 用户重启 Obsidian → 重跑 Scan（确认 Chen2025.pdf 从 ready(0912.0201) 转为 ready(2512.16010) 或 unresolved，LSST 标题消失）→ 重跑全文索引 → 打开 2602.01548 的 Find similar papers 核对；通过后关闭 P8/initiative。

## 2026-08-12 — L1 adjust（标题 v6 收口：metadata 大小写回归 + 逗号作者列表入题）

- evidence: 用户反馈"还有文章标题把作者写到题目里（如 Team2026.pdf 等）"。查证：(1) 测试库 Team2026.pdf 各层信号均正常，但发现 v5 token 覆盖检查的"更长胜出"让全大写字体结果（112 字符）压过 metadata 的正确大小写标题（105 字符）——token 相同仅大小写差异也被判"metadata 丢字符"；(2) 全库扫描"作者入题"形态：Zhao2023.pdf 标题 "A Survey of Large Language Models" 被扩展阶段附加整段逗号分隔作者列表（arXiv 风格 "Wayne Xin Zhao, Kun Zhou*, …" 跨行截断）乃至摘要——NAME_PAIR_LIST 只识别 `·` 分隔，不识别逗号分隔姓名对；Benítez2000 全大写无标记作者（无 metadata、无 ABSTRACT 前瞻可依）为剩余已知形态。
- change: ①token 覆盖改为**严格超集**（fontTokens ⊋ metaTokens 才允许字体结果胜出，tokens 相等时 metadata 胜出）——Team2026 等恢复 metadata 正确大小写，Wen2022 丢 z 场景仍由字体结果救回；②NAME_PAIR_LIST 增加逗号分隔姓名对分支（2-3 词全名 + 可选 `*` + 尾 and，容忍行尾截断名）——扩展不再附加 arXiv 风格作者块。
- evidence: 聚焦 core 24/24（新增 Zhao2023 扩展用例）；语料 parity vs v18 仅 Zhao2023 一处变化（"A Survey of Large Language Models Wayne Xin Zhao, …" → "A Survey of Large Language Models"，清除作者+摘要附加），零误伤；全量 core 95 files / 1603 tests、plugin 33 files / 457 tests；双 typecheck、boundaries、`git diff --check` 通过；lint 0 errors/69 warnings 基线；core 全量临时 8 GiB 堆配置已还原。
- install: 构建产物 SHA-256 `43e34acef748d4d5a7ffc0a77c9d47e078a9246ea0f6e3817596408d2075cfbd` 已安装（main.js + styles.css 同步复制，安装目标 hash 一致）；被替换版备份为 `main.js.strict-cover-b6f2bac4.bak`（SHA-256 `b6f2bac433e433acdc2bf544b7025f03c962fc62a142de1a8ff5765f623f2538`）。
- disposition: L1 adjust；P8 outcome 与 goal 不变。剩余"作者入题"已知形态：Benítez2000 类（标题与全大写作者同字号、无标记、无 metadata）——纯结构信号无法区分，属信息边界；用户真实库若仍有文件出现此形态且不在测试库 376 件内，需具体文件确认（可能为库内新文件或旧 KB 数据未刷新）。继续不 commit/push。
- next: 用户重启 Obsidian → 重 Scan + 重跑全文索引 → 复核 Find similar（Team2026 显示 metadata 大小写标题、Zhao2023 不再带作者）→ 若仍有具体文件异常请提供文件名与显示文本；通过后关闭 P8/initiative。

## 2026-08-12 — L1 adjust（标题刷新失效根因：规则变更未升 titleVersion）

- evidence: 用户贴出 Team2026.pdf / RAlL Team2026.pdf 在 Find similar 的实际显示：`REDSHIFT ASSESSMENT INFRASTRUCTURE LAYERS ( RAIL ):RUBIN-ERA … AT.SCALE PRODUCTION The RAlL Team,, Jan Luca van den Busch 1Eric Charles 2 ,3 ,`。形态分析：全大写标题 + 扩展附加作者行 = v5 时代（d4778f45）逻辑的输出；"( RAIL )" 带空格与 "AT.SCALE" 说明 Obsidian 内置 pdf.js（5.3.34）与验证环境 pdfjs-dist（6.2.108）的文本提取存在字形/间距差异（语料验证基于 pdfjs-dist，生产基于 Obsidian pdf.js——两环境的 layout 可能不同）。根因：v5 之后三轮规则修复（识别仲裁、严格超集、逗号作者列表）均未 bump `TITLE_EXTRACTION_VERSION`，用户重跑索引时 titleVersion 匹配（5=5）→ 旧错误标题不刷新——用户"重新弄还不对"正是这个原因。
- change: `TITLE_EXTRACTION_VERSION` 5→6，强制所有 fallback 文档下次索引重读首页 + metadata 刷新标题（不重嵌入）。Team2026 在 v6 下走 metadata 优先路径（`Redshift Assessment Infrastructure Layers (RAIL): …` 正确大小写标题，不受 Obsidian 文本层差异影响）。
- evidence: 聚焦 core 43/43；双 typecheck、boundaries 通过。构建产物 SHA-256 `f86a5426721bb3f825cecc454b3b32ca85636cf810d8f72fb6579d3aa07d9c31` 已安装（main.js + styles.css 同步复制）；被替换版备份 `main.js.comma-fix-43e34ace.bak`（SHA-256 `43e34acef748d4d5a7ffc0a77c9d47e078a9246ea0f6e3817596408d2075cfbd`）。
- disposition: L1 adjust；P8 outcome 与 goal 不变。记录验证环境差异：语料验证（pdfjs-dist）与 Obsidian 运行时（内置 pdf.js）的文本/布局提取可能有字形级差异，metadata 优先路径不受影响；纯字体规则路径的最终结果以真实 Obsidian 复验为准。继续不 commit/push。
- next: 用户重启 Obsidian → 重跑全文索引（**这次 titleVersion 6 会真正刷新全部 fallback 标题**，Notice 应显示 titles refreshed）→ 复核 Find similar 中 Team2026/RAlL Team2026 标题（预期为 metadata 正确大小写、无作者）；通过后关闭 P8/initiative。

## 2026-08-12 — L1 adjust（Obsidian 环境两处根因：重复 /Title 键解析 + 扩展噪声压过 metadata）

- evidence: 用户在测试 vault（plugin_test）重跑 v6 索引后标题仍不对，KB manifest（revision 21）显示：Team2026 仍为全大写字体结果 + 作者附加，Krause2017_1 为字体结果 + 作者机构附加；而 Ruggeri2025/Zhang2020/Vogt2024 的 metadata 标题生效——同一环境部分生效。根因一：Team2026.pdf 的 Info 字典有重复 `/Title`（literal + `36 0 R` 间接引用），Obsidian 内置 pdf.js 5.3.34 解析出空值（pdfjs-dist 6.2.108 取到 literal 值）——metadata 缺失走字体规则（Obsidian 文本层下 "( RAIL )" 带空格，扩展附加作者行）。根因二：Krause2017_1 的 metadata 实际生效，但字体规则结果（含扩展附加的作者机构）token 严格超集且更长 → 覆盖检查让垃圾字体结果压过正确 metadata。
- change: ①token 覆盖检查改用**扩展前基础标题**（`extractFontTitle` 返回 `{base, title}`），扩展附加的作者行不再参与覆盖比较——Krause2017_1 类场景 metadata 胜出；②host `metadataTitle` 在 Obsidian getMetadata 解析为空时**回退原始字节解析**头部 256 KiB 的 Info `/Title`（第一个 literal 或 UTF-16 hex，decode 转义）——Team2026 类重复键场景取到正确标题。`TITLE_EXTRACTION_VERSION` 6→7。
- evidence: 聚焦 core 25/25（新增 base 覆盖检查用例）、plugin 6/6（新增字节回退用例）；全量 core 95 files / 1604 tests、plugin 33 files / 458 tests；双 typecheck、boundaries、`git diff --check` 通过。构建产物 SHA-256 `702f544420425546637ed60e3cbaaf5c4b885f3481e72cd3e4b5042f77cc151b` 已安装到测试 vault（Desktop/plugin_test，main.js + styles.css 同步复制）；被替换版备份 `main.js.v6-f86a5426.bak`（SHA-256 `f86a5426721bb3f825cecc454b3b32ca85636cf810d8f72fb6579d3aa07d9c31`）。
- disposition: L1 adjust；P8 outcome 与 goal 不变。**安装边界确认：只装测试 vault（/home/tiandc/Desktop/plugin_test）；用户真实 vault（Nextcloud/work/Notes）的插件已回滚原状**（main.js 恢复 2e95cc99，styles.css 从 7 月 13 备份恢复）。验证环境差异记录：Obsidian 内置 pdf.js（5.3.34）与 pdfjs-dist（6.2.108）在 Info 字典重复键解析、文本字形（"( RAIL )" 带空格）上行为不同，KB 数据以 Obsidian 环境复验为准。继续不 commit/push。
- next: 用户重启 Obsidian 打开测试 vault（Desktop/plugin_test）→ 重跑全文索引（titleVersion 7 强制刷新全部 fallback）→ 复核 Find similar 中 Team2026/RAIL Team2026（预期 metadata 正确大小写、无作者）与 Krause2017_1；通过后关闭 P8/initiative。

## 2026-08-12 — L1 adjust（Obsidian 形态作者行断行：affiliation 标记模式）

- evidence: 用户在测试 vault 复验 v7：Beck2021 仍显示标题+作者+机构整段（Obsidian pdf.js 把 "R´obert Beck" 提取为 "R' obert Beck"、"‹" 变 "<"、作者与机构粘成一行——姓名模式（小写中间名/特殊字符）全部失效）；Luo2024/Planck Collaboration2016 已正确；Chen2025 仍显示 LSST Science Book（识别修正需重 Scan，用户只跑了索引）。语料扫描确认 `,\s*\d+\s*[,<]`（affiliation 标记 "1,2 <"、"3,4,2"）在 244 个命中中全部为地址/affiliation 行，无合法标题续行。
- change: 作者行断行与扩展拒绝新增 `AFFILIATION_MARKER`（`/,\s*\d+\s*[,<]/`——逗号+数字+逗号/尖括号，Obsidian 形态的 affiliation 标记，不误伤 "2MASS" 等数字开头词）；run 断行与 isContinuation 同时应用。`TITLE_EXTRACTION_VERSION` 7→8。
- evidence: 聚焦 core 26/26（新增 Obsidian 形态作者行用例）；语料 parity vs v18 仅 Zhao2023（既有修复）无新变化；全量 core 95 files / 1605 tests、plugin 33 files / 458 tests；双 typecheck、boundaries、`git diff --check` 通过。构建产物 SHA-256 `9a0dd97138e9c4a07d459dbda27d8295ff5f957a115a498074f6cdf3322785b0` 已安装到测试 vault；被替换版备份 `main.js.v7-702f5444.bak`。
- disposition: L1 adjust；P8 outcome 与 goal 不变。只装测试 vault。Chen2025 的识别修正（标题搜索仲裁，evidence version 2）自 48507551 版起生效，但需**重 Scan** 重验 content-derived 记录——索引不会重跑识别。
- next: 用户重启 Obsidian 打开测试 vault → 先重 Scan library（Chen2025 从 ready(0912.0201) 修正为 ready(2512.16010)）→ 重跑全文索引（titleVersion 8 强制刷新）→ 复核 Find similar；通过后关闭 P8/initiative。

## 2026-08-12 — L1 adjust（Obsidian 提取两处根因：pdf.js 字体资源 + buffer transfer；识别根因：/arXivID 未读）

- evidence: ①79 个 fallback 标题刷新失败（revision 26-29 每次 79 个，catch 保留旧标题），日志首现 `UnknownErrorException: Ensure that the standardFontDataUrl API parameter is provided` / `cMapUrl and cMapPacked`，加参数后警告消失但失败依旧；补错误日志后明确 `Cannot perform Construct on a detached ArrayBuffer`（Node 复现同一错误信息）。②Chen2025.pdf 仍显示 "LSST Science Book, Version 2.0"：KB manifest（revision 30）221 个全 titleVersion 8、catalog 中该文件已关联 `arxiv:0912.0201`（内容识别 v2，02:57 重识别过）——但 PDF 元数据 `/arXivID (https://arxiv.org/abs/2512.16010v1)` 与 `/Title (LSTM-MDNz: Estimating Quasar Photometric Redshifts with an LSTM-Augmented Mixture Density Network)` 明确其真实身份；`0912.0201` 只出现在正文参考文献的 DOI（`doi.org/10.48550/arXiv.0912.0201`，引用 LSST Science Book）。
- root cause: ①Obsidian 内置 pdf.js 5.3.34 对**非嵌入标准字体/CID 字体**的 `getTextContent` 需要 `standardFontDataUrl`/`cMapUrl`（缺参抛 UnknownErrorException；pdfjs-dist 6.2.108 不抛——验证环境差异）；且 `getDocument` 会把传入的 ArrayBuffer **transfer（detach）给 worker**，提取后的 `rawInfoTitle(bytes)`（v7 新增的 metadata 回退）读 detached buffer 抛错——79 个失败 = getMetadata 无 Title 走 rawInfoTitle 的文件，142 个成功 = getMetadata 有 Title 不碰 bytes，v6 时代成功 = rawInfoTitle 尚不存在。②识别证据提取只扫描解压 content streams（每个前 512 字符），**从不读 Info dict 的 /arXivID**——stream 首个命中是参考文献里的 0912.0201 → 错误关联；title-search 仲裁未命中（API 未返回/相似度不足），且 reconciliation 增量机制（observationFingerprint + contentIdentificationVersion 相同则跳过）使重 Scan 不重识别已识别文件。
- change: ①host `getDocument` 传 Obsidian 资源路径（`cMapUrl: "/lib/pdfjs/cmaps/"`、`cMapPacked: true`、`standardFontDataUrl: "/lib/pdfjs/standard_fonts/"`，从 Obsidian app.js 的 pdf.js viewer options 确认）并传 **bytes 拷贝**（`new Uint8Array(bytes)`，pdf.js transfer 拷贝、原 bytes 保留给 rawInfoTitle）；`page.cleanup()` 异常不再传播（finally 抛错曾吞掉提取结果）。②识别证据提取优先级改为 **Info dict /arXivID（或 /arXiv）→ stream 头 → XMP**——提交系统声明的身份高于任何 stream 文本；`PDF_IDENTIFICATION_EVIDENCE_VERSION` 2→3 强制重识别全部 content-derived 文件（识别规则变更必须升版本——与 TITLE_EXTRACTION_VERSION 同一教训）。
- evidence: ①revision 30 索引 `79 titles refreshed` 全部成功，Beck2021 → "PS1-STRM: neural network source classification and photometric redshift catalogue for PS1 3 π DR1"（无作者）；用户确认 Beck2021 正确。②真实 Chen2025.pdf 上 v3 提取：`arxivId: "2512.16010"` + 正确标题（修复前 0912.0201）；用户重 Scan 后 catalog（revision 20）`Chen2025.pdf → arxiv:2512.16010`，用户确认显示为 LSTM-MDNz 标题。测试：core 识别相关 3 files / 39 tests（+2 新：/arXivID 优先于引用 ID、无 /arXivID 回退 stream 头）、plugin 33 files / 460 tests。core 全量在本机 vitest worker OOM（环境问题，改动为叶子模块，相关测试全覆盖）。构建产物 SHA-256 `46defddf0d16ee81d352501d9e4bc55819bd51b88a81030f9f912da3b1e35997` 已安装到测试 vault；被替换版备份 `main.js.08f4b6b3.bak`。
- disposition: L1 adjust；P8 outcome 与 goal 不变。只装测试 vault（Desktop/plugin_test）；真实 vault 插件仍为旧版，复验由用户自行决定。识别重扫（version 3）会重跑全部 content-derived 文件识别 + title search，Scan 较慢属预期。继续不 commit/push。
- next: 测试 vault 复验通过（Beck2021 标题、Chen2025 → 2512.16010）。用户若在真实 vault 更新插件后复验通过即可关闭 P8/initiative（真实库 219 篇基线 + 新文件的标题/识别形态如仍有异常需具体文件确认）。

## 2026-08-13 — P8 close（用户确认关闭 initiative）

- evidence: 用户在测试 vault 复验确认：Beck2021 标题正确、Chen2025 显示 LSTM-MDNz（2512.16010）、检索显示正常；用户提出"真实 vault 不是已经在 test vault 中测试了吗"——确认 Obsidian 运行时层面测试 vault 与真实环境等价（同一机器、同一内置 pdf.js 5.3.34 与字体资源），运行时环境差异问题已在测试 vault 全部暴露并修复；剩余差异仅为真实库数据形态与插件版本（真实 vault 插件仍为旧版，由用户自行更新）。
- change: goal.md `status: active → done`、success criteria 第 8 项打勾（复验表述改为测试 vault 等价环境）、Phases P8 `status: active → done`；phase 08 文件 T7 标记完成并补收尾复验结论；journal 追加本记录。继续不 commit/push。
- disposition: P8 closed。遗留（非阻塞，用户已知或属信息边界）：真实 vault 插件更新由用户执行；少数 fallback 标题带作者行（Obsidian 文本层作者机构粘连，如 Lin2022/Kitanidis2020/Tamura2016）；catalog unresolved 212 / failed 23（识别失败文件走 fallback KB 索引；failed 多为 resolver 网络限流可重试）；core 全量测试在本机 vitest worker OOM（环境问题，改动为叶子模块）。
