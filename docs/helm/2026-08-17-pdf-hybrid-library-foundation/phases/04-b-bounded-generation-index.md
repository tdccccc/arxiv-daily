# P4b — 有界 block 的事务化分代索引

goal_ref: ../goal.md
updated: 2026-08-22

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
- [x] 实现并验收 current/backup generation store、完整 object closure 校验、唯一目录 promotion、崩溃 seam 与上一代恢复。
- [x] 实现并验收逐 block exact centered dense reader：预计算 corpus mean、结果与 P3 等价、工作集/top-k 有界、查询不调用 legacy `loadPaper`。
- [x] 实现并验收预建 BM25 倒排 block：postings 以 chunk order 保存权威 occurrence stream，并在同一对象保存经 exact-permutation 校验的 term catalog；dictionary 以 posting range 保存权威 route stream和 query permutation，descriptor bucket mask只路由命中页。分别持久化基础与单 Han 查询长度、compact 原值与 gram 候选；reader 按 P3 query term 顺序累加，在论文跨 block 完整结束后进入有界 top-k。promotion 通过 evidence↔postings、postings↔dictionary 的 ordered zipper 与 exact EOF 线性证明完整性，最多同时驻留两个固定上限对象。
- [x] 实现并验收 production generation builder：从 committed manifest snapshot 按 paperKey code-unit 顺序逐篇校验并读取 ready documents，经 bounded storage spool 生成 paired vector/evidence、metadata、postings、dictionary 与 descriptor，再以 one-shot iterator replay 到 transactional store；单篇 source 不一致使本代失败，所有完成、拒绝和失败路径关闭 iterator 并释放 spool。
- [x] 实现并验收首次迁移与生产编排：legacy manifest commit 后同步 generation；同 revision/derivation 整代复用，source revision 变化时流式重建，失败保留旧 current。搜索 pin 单一 current 并使用 generation dense/BM25/RRF；仅从未存在 current 的迁移窗口允许 legacy fallback，valid/corrupt/incompatible current 不静默降级，已提交 generation 查询的 legacy `loadPaper` 为 0。
- [x] 实现并验收宿主静默期 maintenance：opened handle 显式释放和 process-local active tracking；宿主停止 admission 并等待操作 settle 后，保守修复可证明残留的 promotion claim，枚举并仅清理非 current/backup、无 claim、无 active reader 的已知 generation；不按时间偷取、不做未授权跨进程在线 GC。
- [x] 完成固定评测、受限 heap 合成规模、Node/Obsidian composition、跨平台路径语义和全量回归验收。

## Verification

- P4b.1 observed Red：初始测试因 `generation-index-format` API 缺失失败；随后 descriptor closure、空语料统计、跨 block chunk/paper 顺序和非相邻 paperKey 重复分别产生真实失败后修复。
- P4b.1 Green：格式测试 17/17、相邻全文回归 8 文件 110/110、8 GiB heap 下完整 Core、全仓 typecheck、`check:boundaries` 与 `git diff --check` 通过；独立终审无高/中问题。
- P4b.1 technical-report handoff：`no-impact`；新格式尚无生产 builder/store/reader 调用，现有报告继续准确描述当前 JSON/base64 查询路径。
- P4b.2 observed Red：store API 缺失后，首次失败复活、跨 adapter pointer lost update、恢复回滚、claim/commit uncertainty、cleanup ownership、只读 capability、typed read error 等事务 seam 均先产生真实失败。
- P4b.2 Green：store 38/38、store + format + legacy KB store 86/86、相邻全文 74/74、8 GiB heap 下完整 Core、全仓 typecheck、`check:boundaries` 和 `git diff --check` 通过；多轮独立事务/安全终审无高/中问题。
- P4b.2 technical-report handoff：`no-impact`；store 仍仅由 Core tests 调用，插件/CLI 尚未选择该生产路径。
- P4b.3 observed Red：generation dense API 缺失，随后 centered/fusion、late cancel、跨 block evidence key、真实 top-k 上限和合法 `#` paperKey 哨兵分别产生行为失败后修复。vector ordinal schema 与 reader 同批实现，未独立观察 schema Red；补偿验证覆盖旧/未来 schema、little-endian bytes、长度/跳号 mutation 及 vector/evidence paired mismatch。
- P4b.3 Green：相邻 140/140、完整 Core 107 文件 1,898 项、全仓 typecheck、`check:boundaries` 与 `git diff --check` 通过；raw 固定指标保持 P3 基线，默认 centered generation 与 legacy 完整排名相同且明确不同于 raw；多轮终审无高/中问题。
- P4b.3 technical-report handoff：`no-impact`；无 production builder、KB 遍历或插件/CLI 调用，当前报告无需把 declared reader 写成 active behavior。
- P4b.4 observed Red：第一版 lexical candidate 改写了 P3 Han 长度 oracle、提前裁剪跨 window 论文、遗漏正文 compact alias且无法反向证明 postings 完整；第二版正确性 closure 因逐 term 重扫造成二次复杂度和多对象驻留。线性 schema v4 随后分别由 occurrence/catalog API 缺失、跨对象 closure 缺失、reader unavailable、真实 I/O/peak-hit stats、RRF evidence 丢失和评测非有限输入产生失败后修复。
- P4b.4 Green：schema v4 以 chunk-order postings、exact-permutation term/query catalogs 与两个 ordered zipper 线性验证 lexical closure；真实 store promotion/open/search、schema-v2 dense只读兼容、路由碰撞、mixed Han、compact alias、跨 block、selected/unselected corruption与取消边界通过。定向 119/119，8 GiB heap 下完整 Core 108 files / 1,919 tests，全仓 typecheck、`check:boundaries`、`git diff --check` 通过；BM25 固定指标保持 $2/3$，generation hybrid保持 1；多轮终审无 P4b.4 高/中问题。
- P4b.4 technical-report handoff：`no-impact`；generation builder、迁移和插件/CLI search orchestration尚未接线，当前生产仍使用 legacy per-paper JSON/base64与 P3 reader，现有报告准确。
- P4b.5 builder observed Red：首版 append 对候选整块反复 codec/sort，表现为二次工作；exact replay 与 promotion 前拒绝不消费 iterator 会遗留整代 spool；清理缺少显式 endpoint、失败后不能可靠重试，spool 边界异常还可泄漏原生 `TypeError`。后续测试分别捕获真实 codec invocation、4 MiB cap 自动分块、iterator 关闭时序、同步 throw/立即 reject cleanup 重试与 typed spool failure。
- P4b.5 builder Green：builder 深拷贝 committed manifest snapshot，按 UTF-16 code-unit paperKey 顺序逐篇绑定并派生固定上限 objects，以增量 byte budget 在 flush 时编码；one-shot replay 在正常、exact reuse、前置拒绝、部分消费与错误路径释放 spool，并保持主错误优先。真实 store promotion/open/closure 与 generation dense/BM25/RRF 对 P3 oracle 通过；完整 Core 109 files / 1,949 tests、全仓 typecheck、`check:boundaries` 与 `git diff --check` 通过；两轮独立终审无遗留高/中问题。
- P4b.5 builder technical-report handoff：`no-impact`；builder 仍无 executable production caller，现有报告继续准确描述 legacy load-all 搜索路径。
- P4b.6 observed Red：首个生产 cutover candidate 暴露 singleton fallback claim 无法表达并发 active set、marker 建立与 lease 写入之间的 admission race、manifest/CURRENT source revision 在 commit 窗口推进，以及 search 同时读取多个 manifest snapshot/title 来源的问题；新增 seam 测试先复现这些失败和 committed-then-thrown cleanup 边界。
- P4b.6 Green：active-set lease 使用独立路径、exact readback、双扫描和 fail-closed malformed-entry 检查；cutover marker 在 pointer promotion 前后复核，首次失败可安全回滚，已切换状态不会静默回到 legacy。插件在 manifest CAS 前后持有 lease、所有成功/失败/取消路径释放 lease，capable host 在 CAS 后同步 generation；同 revision/derivation 整代 exact reuse，source 推进时最多三次 post-commit convergence，重试 token 不复用。generation search 使用单一 committed manifest snapshot，catalog title 只作排名覆盖，已提交 generation 的 legacy `loadPaper` 为 0。
- P4b.6 verification：Core 111 files / 1,994 tests、Plugin 38 files / 601 tests、Node runtime 3 files / 45 tests；workspace typecheck、`check:boundaries`、`check:product-units`、生产 build 与 `git diff --check` 全部通过。`npm run lint` 当前仍受仓库既有 65 条 Obsidian/style warnings（上限 60）阻止，未新增 error；升级权限执行 `smoke:build` 后只发现既有 plugin bundle `canvas` forbidden-text 基线，未由本 chunk 引入，留给 P4b.8/P7 的全仓门禁收敛。
- P4b.6 technical-report handoff：`no-impact`；production generation/search 接线仍只改变全文索引的派生查询路径，不改变 catalog、方向、日报或 consent 领域契约。
- P4b.7 observed Red：原有 opened generation handle 不带生命周期，host 无法等待 pinned reader；promotion claim 清理失败后既不能证明可修复，也不能安全启动 orphan cleanup。新增 maintenance seam 先复现 opened handle 未关闭时 admission 必须停止、旧 generation reader 必须阻塞 maintenance、未证明 claim 必须保留所有 candidate，以及两 pointer 丢失后必须 fail-closed。
- P4b.7 Green：opened handle 有显式且幂等的 `close()`，所有 production synchronization/search/preflight 路径在 `finally` 中释放；runtime-local admission gate 会等待 active operation 和 reader settle。插件仅在 scheduler idle、operations 独占且 scheduler 已停止的宿主静默期调用 maintenance。maintenance 只删除非 current/backup、无 staging/promotion claim、无 active reader 的已知 generation；仅在 candidate 已是完整可验证的 CURRENT 且 claim/pointer 双次读回一致时修复 promotion claim，任何不确定性均保留并禁止收集。
- P4b.7 verification：定向 Core 96 项和 Plugin lifecycle 11 项通过；8 GiB heap 下完整 Core 111 files / 2,000 tests、Plugin 38 files / 602 tests、Node runtime 3 files / 45 tests 通过；workspace typecheck、`check:boundaries`、`check:product-units`、production build 与 `git diff --check` 通过。默认 Node heap 的完整 Core 运行在无断言失败后达到 heap 上限；8 GiB 是本阶段既定的受限 heap 验收配置，最终 Green 以该运行取得。
- P4b.7 technical-report handoff：`no-impact`；宿主授权的 generation maintenance 只维护可删除的全文派生投影，不改变 catalog、方向、日报或 consent 领域契约。
- P4b final acceptance：固定 graded corpus 继续锁定 dense/BM25/RRF 的 Recall@k、MRR、nDCG 和 P3 等价；120-paper BM25 fixture 锁定 candidate/hit/block 上限，builder 的 7 个大 chunk fixture 迫使 4 MiB object 切分并以真实 store closure 证明覆盖。Node composition 持久化 spool、promotion 和 generation search，plugin lifecycle 保持 manifest snapshot 与 cutover fail-closed；Node adapter 将反斜杠规范为 vault-relative `/` 路径。8 GiB heap 下完整 Core 111 files / 2,000 tests、Node runtime 3 files / 45 tests、CLI 7 files / 71 tests、Plugin 38 files / 602 tests、typecheck、boundary、product-unit 与 production build 通过。根 `npm test` 的 Core 子任务在默认 Node heap 触顶，使用本阶段既定 8 GiB 配置通过；lint 仍为无新增 error 的既有 65 warnings/60 cap 基线，升级权限的 smoke build 仅复现既有 plugin bundle `canvas` forbidden-text 基线。
- P4b final technical-report handoff：`no-impact`；P4b 只替换全文派生索引的持久化和查询实现，未改变 catalog、方向、日报或 consent 的领域关系。
- 阶段硬门：已提交 generation 查询的 `legacyPaperLoads` 为 0；每个 binary object 与同时驻留 block 数有固定上限；lexical 不扫描无关 chunk text；dense 不创建 corpus-sized vector array。
- 阶段质量门：P3 固定 corpus 的 dense、BM25、hybrid 排名和 Recall@k、MRR、nDCG 不回归。

## Abort / reshape triggers

- 如果任何单个 object、descriptor 或查询同时驻留 block 数随 corpus 无界增长，停止并重划 shard/merge 格式。
- 如果 pointer 可在 object closure 完整校验前提交，查询会混读两代，或 future schema 会被覆盖/降级，停止接线并先修复恢复协议。
- 如果新 reader 整体读取所有 vectors/chunks、重建全库 centered 副本或查询期 tokenize 全库，停止并修复 block 路由。
- 如果 generation 目录或活跃 segment 数无清理/合并上限，停止并加入可恢复 GC 与 compaction gate。
- 如果固定评测排名或 score/source/locator 语义变化，停止优化并以 P3 合同修复等价性。
- 如果必须扩展任意目录 filesystem 权限、触碰 P5/P6、并行 Helm 状态或远程默认值，停止并隔离范围。
