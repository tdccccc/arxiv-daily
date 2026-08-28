# PDF 解析与混合检索基础（PDF parsing and hybrid retrieval foundation）

status: active
updated: 2026-08-23
owner: codex-main-session

## Intent

在不引入问答或 Agent 的前提下，把个人文献库升级为可演进的科研 PDF 解析与混合检索基础：结构化、可定位的解析和分块，关键词与语义互补的检索，以及可直接回到 PDF 原文的证据结果。现有 TypeScript 领域模型、目录权限、catalog、研究方向和阅读候选继续作为权威业务层。

## Success criteria

- [ ] Core 提供 host-neutral 的结构化文档解析契约；PDF.js 保持零配置 fast path，后续高质量解析器可通过同一端口接入。
- [ ] 全文 chunk 感知文档结构并携带稳定来源定位、解析器和分块器版本；现有知识库可安全迁移或重建。
- [ ] 文献搜索同时使用词法/BM25 与 dense vector 召回，以 RRF 融合，并通过固定评测集验证其优于或不劣于现有检索基线。
- [ ] 派生索引具有有界内存、增量更新、失败隔离、版本兼容检测和可恢复切换；不再要求查询时加载并重算全库向量。
- [ ] 搜索结果展示论文、章节、PDF 原文片段和页码；用户可打开本地 PDF 到对应页面，坐标高亮不可用时可靠降级到页码。
- [ ] 可选高质量 PDF 解析 sidecar 完成能力探测、隐私边界、失败降级和真实复杂论文对照验证；sidecar 不获得任意目录扫描权限。
- [ ] 现有 catalog、方向聚类、增量方向建议、阅读候选、日报和 consent 行为保持兼容，相关测试与边界检查通过。

## Non-goals

- 全文问答、答案生成和 claim-level 引用验证。
- 自主 Agent、知识图谱、引文图或共引分析。
- 多用户、云端托管文献库或默认部署独立向量数据库。
- 用 Zotero 或外部 RAG 平台替换现有权威领域数据。
- 在第一阶段引入 Docling、Python sidecar、新索引后端或改变检索排序。

## Constraints

- Core 保持 host-neutral，不依赖 Node、Obsidian、Python、数据库驱动或宿主运行时对象。
- `ScopedLibrarySource` 的 root-bound、只读、无符号链接权限边界必须保留；远程全文处理继续要求独立 consent。
- catalog、用户确认方向和阅读候选是权威数据；解析、chunk、FTS 和向量索引均为可删除重建的派生投影。
- 每个行为变更分块采用 Red–Green–Refactor，并在进入下一阶段前取得定向测试、相关回归、类型和边界检查证据。
- 迁移使用版本化、可回滚路径；不得原地删除唯一可用的旧知识库。
- 本目标不修改并行 active Helm 的状态；与 `2026-08-13-discovery-loop-and-library-insight` 共享文件时保持改动隔离。
- 不执行生产部署，不向外发送用户 PDF，不新增远程处理默认行为。

## Phases

1. P1 — 结构化解析契约与 PDF.js 兼容适配就绪，现有索引和检索产物保持不变 — status: done
2. P2 — 结构感知 EvidenceChunk、稳定 locator 与版本化迁移路径就绪 — status: done
3. P3 — 检索评测基线建立，BM25 与 dense 召回经 RRF 形成可验证的混合排序 — status: done
4. P4 — range/exclusive filesystem primitive 路径无法跨当前宿主兑现一致权限边界 — status: superseded
4. P4b — 固定上限 block 与事务化 generation 替代查询期全库 JSON/base64 装载 — status: done
5. P5 — 搜索 UI 展示章节、原文片段和页码，并可打开 PDF 到证据位置 — status: done
6. P6 — 可选高质量 PDF 解析 sidecar 接入并完成复杂论文、隐私与降级验证 — status: done
7. P7 — 旧库迁移、规模性能、跨平台运行与全量兼容性验收完成 — status: done

## Open questions

- P3 计划时用真实语料基线决定中文 FTS tokenizer、BM25 字段权重和 RRF 参数，不在目标层提前锁死。
- P6 计划时以解析评测决定首个 sidecar 使用 Docling，还是 Docling 加 GROBID enrichment。
