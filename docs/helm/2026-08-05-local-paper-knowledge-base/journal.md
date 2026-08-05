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
