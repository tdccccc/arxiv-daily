# P4 — 有界、事务化的本地派生索引

goal_ref: ../goal.md
updated: 2026-08-18

## Outcome

纯 TypeScript、host-neutral 的不可变分代索引成为全文搜索默认后端：BM25 查询读取预建倒排表，dense 查询按固定大小二进制块执行与 P3 等价的 centered cosine 精确扫描；generation 原子切换且可从上一代恢复，已提交索引上的查询不再逐篇加载 JSON、解码 base64 或构造全库向量副本。

## Assumptions

- Node `>=20.11.0`、Obsidian 最低兼容版本和当前无原生资产的发布方式不能可靠承载 `node:sqlite`、SQLite vector extension 或 LanceDB；纯 TypeScript 文件格式可跨现有宿主落地。
- P4 保留旧 per-paper JSON 知识库作为迁移和重建源；它不是新查询后端，也不会在本阶段被原地删除或改写为唯一新格式。
- dense 暂用 exact scan；固定块 range read、预计算语料均值和有界 top-k 足以消除当前内存问题，ANN 是否必要留给 P7 的真实规模证据。
- P3 的 tokenizer、BM25、centered cosine、论文级 RRF、score 语义和固定评测集是兼容契约；本阶段只替换数据布局和读取方式。
- 派生索引失败不得回滚已提交的知识库更新；旧 committed generation 继续服务，下一轮可重建或增量追平。
- P5 UI、P6 parser sidecar、问答、native dependency 和默认远程处理均不在本阶段修改。

## Approach

先补充可选的二进制 range read 与原子写宿主能力，再定义带 checksum、版本和 current/backup 指针的不可变 generation。构建器从已提交知识库流式生成 chunk metadata/text、little-endian vectors 和 BM25 postings；reader pin 单一 generation，dense 分块扫描并 late-materialize evidence，lexical 只读取命中词项。知识库 manifest 成功 CAS 后更新派生索引，首次 promotion 前允许 legacy 查询，成功后只在 committed generations 间恢复。

## Test strategy

- change kind: optimization plus local persistence and query behavior change
- strategy: correctness + performance baseline；每个新行为块先写失败契约，观察预期 Red，再做最小 Green；P3 scorer/评测作为结果等价基线
- Red / baseline signal: adapter、codec/recovery、dense、lexical、promotion/dual-read 测试分别先因能力或 API 不存在失败；现有 P3 dense/BM25/RRF 与评测保持 Green
- Green / regression checks: 每块运行新增定向测试及相邻 store/retrieval regression；阶段末在受限 heap 下断言读取字节、block、candidate、paper load 上限，再运行完整 core/node-runtime/plugin、typecheck、boundaries、build 和 submission checks

## Tasks

- [ ] 扩展并验收可选二进制 range read、原子 binary replace 与宿主 capability fail-closed，Node 和 Obsidian 实现不得退化为整文件读取。
- [ ] 定义并验收版本化索引 codec、content-addressed objects、generation current/backup、checksum/offset 验证和崩溃恢复。
- [ ] 实现并验收固定块 exact centered dense reader：预计算 corpus mean、结果与 P3 等价、工作集与 top-k 有界、查询不调用 legacy `loadPaper`。
- [ ] 实现并验收预建 BM25 倒排索引：保持 Unicode/CJK/title alias 排名语义，只读取命中词项并延迟物化 evidence。
- [ ] 实现并验收单 writer 增量 generation、删除、source revision guard、失败隔离、首次迁移、dual-read 与 backup rollback，并接入索引/搜索编排。
- [ ] 完成 Core/Node/Obsidian host composition、固定评测、受限 heap 合成规模、跨平台路径语义和全量回归验收。

## Verification

- 待执行：每个任务记录 observed Red、Green、相邻回归及 technical-report handoff 结果。
- 阶段硬门：已提交 generation 查询的 `legacyPaperLoads` 为 0；单次 binary read 和 peak working bytes 受固定上限约束；lexical 不扫描无关 chunk 文本；dense 不创建 corpus-sized vector array。
- 阶段质量门：P3 固定 corpus 的 dense、BM25、hybrid 排名和 Recall@k、MRR、nDCG 不回归。

## Abort / reshape triggers

- 如果现有宿主无法在不提高产品最低版本或引入原生模块的情况下提供真实 range read 与原子切换，停止实现并 L2 重划为显式本地服务边界。
- 如果 generation promotion 可能暴露半写对象、混读两代，或 future schema 会被覆盖/降级，停止接线并先修复 fail-closed 与恢复协议。
- 如果新 reader 为兼容而整文件读取所有 vectors/chunks、重建全库 centered 副本或查询期 tokenize 全库，停止并重划 block/shard 格式。
- 如果增量方案导致 generation descriptor 或活跃 segment 数无上限，停止并加入硬上限与 streaming compaction 后再接生产。
- 如果固定评测排名或 score/source/locator 语义变化，停止优化并以 P3 合同修复等价性。
- 如果必须触碰 P5 UI、P6 parser、并行 Helm 状态或远程 consent 默认值才能完成，停止并隔离范围。
