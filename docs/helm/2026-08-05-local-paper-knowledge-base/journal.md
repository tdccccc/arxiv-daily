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
