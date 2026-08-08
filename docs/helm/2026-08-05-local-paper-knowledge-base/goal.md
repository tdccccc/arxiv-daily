# 本地论文知识库（local paper knowledge base）

status: done
updated: 2026-08-07
owner: current-session

## Intent

把个人文献库从"论文级索引"升级为"全文级本地知识库"：本地提取全文、分块向量化（multilingual-e5-small，CPU，离线），支撑相似论文检索、聚类式研究方向生成，以及随文献库增长的方向增量更新。方向成为版本化、可审核、可锁定的实体，用户决定永远优先于机器建议。

## Success criteria

- [x] 文献库中每篇已识别论文可本地提取全文并增量向量化（multilingual-e5-small 本地 CPU），离线可用；已嵌入内容按文件哈希复用，不重复计算。
- [x] 给定一篇论文或一段描述，可检索出库内最相似的论文，带相似度与可解释理由（标题/摘要/命中段落）。
- [x] 研究方向由聚类生成（替代一次性 LLM 分批提取）：方向 = 版本化实体，含名称、描述、关键词、代表论文、成员置信度、时间线。
- [x] 新论文入库后增量更新：就近归入现有方向或进候选缓冲池；缓冲池达到阈值触发局部重聚类与 LLM 建议（归入/新建/分裂/合并），全部进审核队列。
- [x] 用户确认、编辑、锁定方向的优先级最高：锁定的方向不参与自动合并/分裂/改名，但新论文仍可归入；机器建议永不覆盖用户决定。
- [x] 现有隐私与 consent 边界保持：本地全文索引与向量化不经模型授权；任何全文内容进入 LLM 必须新增 processingDepth 授权；路径/PDF bytes/凭据不进入任何 LLM 输入。
- [x] 与 P6 成果兼容：识别 v2（PDF 证据 + 标题搜索）、画像 store 的 CAS/版本机制、日报/新颖性管线不受影响；完整测试套件通过。

## Non-goals

- 云端向量服务或远程 embedding API（默认全本地；远程作为可选开关留待未来）。
- 自动全量重聚类作为常规路径（仅低频兜底，且差异需用户审核）。
- 自主 Agent 循环、自动确认方向。
- 引用图/共引分析（论文-论文关系图谱，可后续 initiative）。
- 全文级 LLM 问答（全文只做本地索引与检索；LLM 默认只接触标题+摘要）。

## Constraints

- Core 保持 host-neutral：PDF 全文提取、嵌入推理、wasm 加载全部在 host（Obsidian/CLI）侧实现，Core 定义端口与编排。
- 向量与全文块**不写入 papers.json**：独立知识库存储（旁路 catalog），可删除重建；路径按 scope/identification fingerprint 分片。
- 并发控制学画像 store 的 expectedRevision CAS，不学 catalog 的 replace。
- 嵌入模型 multilingual-e5-small（q8）按需下载（支持 HF 镜像）或随插件分发；模型文件不进入 vault。
- 全文索引是本地操作（可独立于模型授权）；全文进 LLM 严格绑定新增授权。
- 检索预算有界：日报路径延迟敏感，检索预计算、运行时只查 top-k。
- 不提交/推送代码需显式用户指令（沿用 P6 约束）。

## Phases

1. P1 — 全文索引与相似检索：PDF 全文提取（复用 Obsidian 内置 pdf.js）、分块、本地嵌入、暴力余弦检索、增量更新、存储与重建 —— status: done
2. P2 — 聚类方向生成：聚类（HDBSCAN 或等价）发现候选簇 + 离群缓冲池；LLM 为每簇生成方向草案；方向存储升级为版本化实体；审核界面 —— status: done
3. P3 — 方向增量更新：缓冲池触发局部重聚类、LLM diff 建议（归入/新建/分裂/合并）、审核队列、方向锁定与漂移检测、低频全量重建兜底 —— status: done
4. P4 — Obsidian 运行时验证与修复：真实 Obsidian 中验证 `window.pdfjsLib` 可用性与渲染进程 wasm/模型加载表现（P1 遗留未决项），发现问题则修复，结论记录在案 —— status: done
5. P5 — 实现存疑项复核（P3 T5b 遗留）：centering 镜像去重、new 候选 discoveryCues 截断、增量更新 operation kind 复用（存疑项 1 已由 ADR 0007 关闭）—— 每项代码核对定论，可修复项修复并验收 —— status: done

## Open questions

- 检索入口形态：Dashboard 内"similar papers"按钮（从某篇论文出发）vs 独立检索栏（从任意描述出发）vs 两者。
- 聚类算法实现：已定案——single-linkage（Kruskal 合并）+ 相对停止线（最强边 × 0.65），真实异构语料验证（P2 journal）；HDBSCAN 无需再评估。
- 增量触发节奏：每日批量 vs 扫描完成即触发 vs 手动（当前：手动命令 `check-incremental-direction-updates`）。
