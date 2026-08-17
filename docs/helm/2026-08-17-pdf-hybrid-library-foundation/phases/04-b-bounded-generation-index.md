# P4b — 有界 block 的事务化分代索引

goal_ref: ../goal.md
updated: 2026-08-18

## Outcome

纯 TypeScript、host-neutral 的不可变分代索引成为全文搜索默认后端：BM25 查询读取预建倒排 block，dense 查询逐个加载具有固定字节上限的 vector block 并执行与 P3 等价的 centered cosine 精确扫描；小型 text pointer 在所有 generation objects 校验后切换并可恢复，已提交索引查询不再逐篇加载 JSON、解码 base64 或构造全库向量副本。

## Assumptions

- `StorageAdapter.readBinary/writeBinary` 的单文件整读写可跨当前 Node 与 Obsidian 宿主使用；只要每个 object 有严格字节上限，查询峰值内存就不随语料总量增长。
- 每个 generation 使用唯一目录且 object 名单只由该 generation descriptor 引用；writer 在 pointer promotion 前完整写入并回读校验，reader 不枚举或读取未提交目录。
- P4b 不要求通用 binary replace、hard-link publish、range I/O、native filesystem API 或 adversarial descriptor anchoring；小型 `CURRENT`/backup 使用现有 text atomic/recovery 能力。
- 旧 per-paper JSON 知识库继续作为迁移和重建源，不被原地删除，也不作为已提交 generation 的查询 fallback。
- dense 暂用 exact scan；固定 block、预计算语料均值和有界 top-k 消除当前内存问题，ANN 必要性留给 P7 的规模证据。
- P3 tokenizer、BM25、centered cosine、论文级 RRF、score 语义和固定评测是兼容契约；P5 UI、P6 parser sidecar和远程 consent 不在本阶段修改。

## Approach

先定义带 magic、版本、长度、checksum 和硬上限的 metadata/text/vector/postings block codec，再实现 generation descriptor 与 current/backup 恢复。构建器从 committed KB 流式生成多个完整小 block，每写一个即回读校验并释放工作集；最后原子提交小型 text pointer。Reader pin 单一 generation，dense 逐 block 扫描并 late-materialize evidence，lexical 通过有界词典路由到命中 postings block。索引编排在 KB manifest CAS 后生成新 generation，失败时保留上一 committed generation。

## Test strategy

- change kind: optimization plus local persistence and query behavior change
- strategy: correctness + performance baseline；每个行为块先写失败契约并观察 Red，P3 scorer/评测作为结果等价基线
- Red / baseline signal: codec/size cap、generation recovery、dense、lexical、promotion/dual-read 测试分别先因 API 或行为不存在失败；现有 P3 dense/BM25/RRF 与评测保持 Green
- Green / regression checks: 每块运行新增定向测试与相邻 store/retrieval 回归；阶段末在受限 heap 下断言单 object、simultaneous blocks、candidate、paper load 上限，再运行完整 core/plugin、typecheck、boundaries 和 production build

## Tasks

- [x] 定义并验收固定上限 binary block codec、严格 schema/checksum/offset 解码、generation descriptor 路径与未来版本 fail-closed。
- [ ] 实现并验收 current/backup generation store、完整 object closure 校验、唯一目录 promotion、崩溃 seam 与上一代恢复。
- [ ] 实现并验收逐 block exact centered dense reader：预计算 corpus mean、结果与 P3 等价、工作集/top-k 有界、查询不调用 legacy `loadPaper`。
- [ ] 实现并验收预建 BM25 倒排 block：保持 Unicode/CJK/title alias 排名语义，只读取命中词项路由到的 block 并延迟物化 evidence。
- [ ] 实现并验收单 writer generation rebuild/incremental reuse、删除、source revision guard、失败隔离、首次迁移、dual-read 与索引/搜索编排接线。
- [ ] 完成固定评测、受限 heap 合成规模、Node/Obsidian composition、跨平台路径语义和全量回归验收。

## Verification

- P4b.1 observed Red：初始测试因 `generation-index-format` API 缺失失败；随后 descriptor closure、空语料统计、跨 block chunk/paper 顺序和非相邻 paperKey 重复分别产生真实失败后修复。
- P4b.1 Green：格式测试 17/17、相邻全文回归 8 文件 110/110、8 GiB heap 下完整 Core、全仓 typecheck、`check:boundaries` 与 `git diff --check` 通过；独立终审无高/中问题。
- P4b.1 technical-report handoff：`no-impact`；新格式尚无生产 builder/store/reader 调用，现有报告继续准确描述当前 JSON/base64 查询路径。
- 阶段硬门：已提交 generation 查询的 `legacyPaperLoads` 为 0；每个 binary object 与同时驻留 block 数有固定上限；lexical 不扫描无关 chunk text；dense 不创建 corpus-sized vector array。
- 阶段质量门：P3 固定 corpus 的 dense、BM25、hybrid 排名和 Recall@k、MRR、nDCG 不回归。

## Abort / reshape triggers

- 如果任何单个 object、descriptor 或查询同时驻留 block 数随 corpus 无界增长，停止并重划 shard/merge 格式。
- 如果 pointer 可在 object closure 完整校验前提交，查询会混读两代，或 future schema 会被覆盖/降级，停止接线并先修复恢复协议。
- 如果新 reader 整体读取所有 vectors/chunks、重建全库 centered 副本或查询期 tokenize 全库，停止并修复 block 路由。
- 如果 generation 目录或活跃 segment 数无清理/合并上限，停止并加入可恢复 GC 与 compaction gate。
- 如果固定评测排名或 score/source/locator 语义变化，停止优化并以 P3 合同修复等价性。
- 如果必须扩展任意目录 filesystem 权限、触碰 P5/P6、并行 Helm 状态或远程默认值，停止并隔离范围。
