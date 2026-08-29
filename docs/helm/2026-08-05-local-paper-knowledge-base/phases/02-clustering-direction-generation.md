# P2 — 聚类方向生成（clustering-direction-generation）

<!-- Filename 02-clustering-direction-generation.md ↔ P2 -->
goal_ref: ../goal.md
updated: 2026-08-05

## Outcome

方向生成从"一次性 LLM 分批提取"改为**聚类驱动**：以 P1 知识库的论文级向量为输入，core 聚类引擎（简化质心两遍聚类 + 离群阈值，HDBSCAN 的等价物）发现候选簇与离群缓冲池；LLM 仅为每簇生成方向草案（名称/描述/关键词/代表论文），草案携带成员置信度；方向存储升级为版本化实体（成员置信度 + 时间线，schema v3 带迁移）；审核界面可查看簇成员、置信度与缓冲池。P3 的增量触发不在本阶段。

## Assumptions

- core 依赖白名单仅 pako（check-boundaries 强制），因此聚类算法在 core 内自实现简化等价方案（质心两遍聚类 + 离群阈值），不引入 ml-hdbscan；goal open question 已认可该等价。
- 聚类输入 = P1 已索引论文的聚合向量（chunk 向量均值 + L2 归一化）；聚类入口要求知识库非空，否则提示先运行全文索引。
- 每簇一次有界 LLM 提取调用（复用现有 extraction prompt 与 validation/重试），synthesis 去重阶段保留；旧 LLM grouping 步骤被聚类替代并从入口移除。
- 缓冲池不新增持久化 schema：由 proposal 的 `catalogInputPapers` 减去各候选 `clusterMembers` 派生，审核界面即时计算。
- schema v2 → v3 迁移沿用 profile store 的 `migrateOnLoad` 与语义 revision 机制；迁移生成初始 `created` 时间线事件，不丢既有方向。

## Approach

core 新增 `packages/core/src/library/clustering/`：`clusterPaperVectors`（**SNN 图聚类**：corpus-level centering → 论文间 max-chunk 相似度 → mutual top-k 邻居图 → 连通分量成簇 → 孤立点入缓冲池；对 e5-small 饱和余弦分布鲁棒，绝对阈值在真实语料上实测不可靠——L2 reshape 2026-08-06）、`buildClusteringInput`（加载 ready 论文的 chunk 向量）；方向提案入口走聚类 → 每簇 extraction → synthesis。`personal-library-interest-profile.ts` schema v3：candidate 加 `clusterMembers`（paperKey + confidence），confirmed direction 加 `clusterMembers` 与 `timeline` 事件；确认/编辑/合并/删除流程写入时间线。plugin 审核 modal 展示簇成员与置信度、缓冲池视图；确认候选时把 `clusterMembers` 固化进 confirmed direction。

## Tasks

- [x] 聚类引擎（core 纯函数，L2 reshape 后为 SNN 图聚类）：corpus-centering + max-chunk 相似度 + mutual top-k 连通分量 + 缓冲池；确定性、参数化（neighborCount/minClusterSize）；单测（合成向量：同主题簇、离群、确定性、顺序无关）。
- [x] 聚类输入构建（core）：`buildClusteringInput` 加载 ready 论文 chunk 向量（替代被 reshape 移除的论文级向量聚合）；单测。
- [x] 聚类→方向草案（core）：提案入口改为 聚类 → 每簇一次有界 extraction（复用 prompt/validation/重试）→ synthesis；candidate 携带 `clusterMembers`；单测（fake LLM）。
- [x] 方向存储升级 v3：candidate + confirmed direction 增加 `clusterMembers` 与 `timeline` 事件类型；v2→v3 迁移（migrateOnLoad）；确认/编辑/合并/删除写时间线；单测。
- [x] 审核界面接入：Review modal 展示候选簇成员与置信度、缓冲池（未归类论文）视图；确认候选固化 `clusterMembers`；plugin 测试。
- [x] 端到端验证：真实异构小库（GNN/量子物理/生物医学 12 篇）索引 → 聚类 → 草案 → 确认 → profile v3 含置信度与时间线（E2E OK：2 主题簇 + 4 缓冲池 + 成员置信度 + created 事件）；全量测试 + boundaries。

## Verification

- `npm run typecheck && npm test && npm run check:boundaries`（core 套件用 `NODE_OPTIONS=--max-old-space-size=8192`）通过。
- 聚类引擎单测：合成向量下同主题簇、缓冲池、确定性、参数边界。
- 草案单测：每簇恰好 1 次 LLM 调用、`clusterMembers` 置信度与聚类一致、候选 bounds 校验。
- 迁移单测：v2 profile 加载 → v3 字段默认值 + 初始时间线事件，语义 revision 不变。
- 端到端：多主题真实库聚类后，草案方向可区分主题；确认后 confirmed direction 携带成员置信度与 `created` 事件；审核 modal 可见缓冲池。
- 离线约束保持：聚类与审核全本地；LLM 只接触标题/摘要/代表论文（沿用既有 evidence depth 边界）。

## Abort / reshape triggers

- 真实库上聚类质量不可用（簇内相似度普遍低于阈值、簇数失控或主题混杂）→ L2：评估 core 白名单豁免引入 ml-hdbscan，或改用 OPTICS 式密度聚类自实现。
- LLM 草案质量劣于旧 grouping 路径（可观察：方向名称/代表论文明显不相关）→ L1/L2：调整每簇提示词或候选上限。
- schema v3 迁移在真实 profile 上失败或丢失数据 → L2：回退 v2 兼容路径并 journal。
