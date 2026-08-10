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
