# P2 — index-titles-and-abstracts

goal_ref: ../goal.md
created: 2026-09-02T22:47:24+08:00
updated: 2026-09-02T22:47:24+08:00
revision: 1

## Outcome

库索引每篇论文只产出一到两块（标题 + 摘要），不再对 PDF 全文分块。千篇量级的库在 CPU 上分钟级重建完，索引体积从百 MB 级降到 MB 级。旧索引被**重建而非迁移**（ADR 0013 §3）。筛选管线与设置页在本阶段一行不改。

## Assumptions

- **摘要已经在手边，不需要新数据源。** `PersonalLibraryPaperRecord`（`packages/core/src/library/personal-library-catalog.ts:47`）已带 `title` 与 `abstract`，且 `evidenceDepth` 就是 `"metadata-and-abstract"`；`catalog` 本来就是 `indexPersonalLibraryFullText` 的入参（`index-orchestration.ts:109`）。所以本阶段是**改索引的输入来源**，不是新建一条元数据链路。

- **不是每篇论文都有摘要。** 未被识别的文件走 fallback 单元（`file:sha256:…`），没有任何 catalog 元数据，今天靠 `extractTitleFromFirstPage` 从首页取标题。这类论文的占比在 Chunk 1 之前**未知**，而它决定本阶段真正省下多少——fallback 仍然必须读 PDF。

- **ADR 0013 只数了 embedding 调用，但重建时间的大头可能是 PDF 解析。** 识别出的论文一旦改从 catalog 取文本，就**整个不必读 PDF 字节**（`buildPaperDocument` 的 `readBinary` + `parseIndexDocument` 全省）。这既是最大的机会，也是 Chunk 1 必须先量清楚的东西：若 fallback 占多数，这份收益不成立。

- **`textHash` 目前不参与复用判定**（已核实：`index-orchestration.ts:213–218` 的 `exactReady` 只看 `observationFingerprints` + `modelId` + `derivation`；`identificationFingerprint` 只覆盖扩展名与识别策略版本，不含任何逐篇元数据）。改用 catalog 文本之后这是一个**真实的陈旧洞**：arXiv 元数据重解析让摘要变了、而文件一个字节没动，索引会静默留着旧向量。P1 之前不存在这个洞，因为索引文本来自文件本身。

- **聚类会因此变化，这是 ADR 0013 认下的代价，不是本阶段要调的东西。** 聚类直接吃 chunk 向量（`clustering/paper-vector.ts`，2026-08-06 的 L2 已移除 paper-level mean pooling），每篇只剩一到两块之后 `maxChunkCosine` 事实上退化为「摘要对摘要的余弦」。方向是否变粗是**要测的量，不是要假设的事**（ADR 0013 Consequences），且阈值实测被 goal 排在 P4/P5。**P2 只负责确认聚类仍能跑出方向，不负责重新调参。**

- **`chunkFullText` / `chunkParsedDocument` 保留，不删。** ADR 0013 §2 把「把结论段也纳入索引」列为最可能的下一增量，删掉分块器等于给那次回头路凭空加成本。本阶段改的是**喂给分块器什么**，不是分块器本身。

## Approach

先量当前的真实成本与 fallback 占比（Chunk 1），把「分钟级」变成有对照的判据而不是口号。然后把 `buildPaperDocument` 的文本来源从「读 PDF → 解析 → 全文分块」改成「catalog 的 title + abstract」，识别出的论文彻底不碰 PDF 字节（Chunk 2）；fallback 论文没有摘要，单独给一条有上界的首页路径（Chunk 3）。再用一次 derivation 版本抬升强制全库重建，并顺手把复用判定里的陈旧洞堵上（Chunk 4）。最后在测试库上复量，并确认聚类没有塌掉（Chunk 5）。

## Chunks

### Chunk 1 — 量出改造前的基线与 fallback 占比

- change kind: non-behavioral（测量，不改源码）
- strategy: correctness + performance baseline
- Red / baseline signal: 无红可取——这是基线本身。在测试库上重建一次索引，记录：**总块数、索引体积、墙钟耗时、embedding 调用数**，以及**其中 PDF 解析与 embedding 各占多少时间**；另记 **catalog 识别出的论文数 vs fallback 单元数**。ADR 0013 已有的对照是 207 篇 → 23,423 块 → 140MB（约 113 块/篇）。
- Green check: 数字落进 journal.md，且与 ADR 0013 的记录可比对（若差异大，先解释再往下走）
- regression checks: 不适用（不改代码）
- exception: 这是测量任务，没有可取的红。补偿验证是**数字必须来自真实的测试库运行，不是估算**；解析/embedding 的耗时拆分若拿不到，明写为未测而不是猜。
- **这一 chunk 是门，不是仪式**：fallback 占比与「解析 vs embedding」的耗时拆分直接决定 Chunk 3 的分量，以及本阶段的 abort trigger 是否已经触发。
- [ ] implementation and tests accepted

### Chunk 2 — 识别出的论文只索引 catalog 的标题与摘要

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增测试——喂一篇 catalog 里有 `title` + `abstract`、PDF 有几十页正文的论文，断言产出的 document **恰有一到两块**，其文本就是标题与摘要，**且 `source.readBinary` 一次也没被调用**。红在当前会读 PDF 并产出几十块。
- Green check: `npm run test --workspace @arxiv-daily/core -- fulltext-index`
- regression checks: `fulltext-chunking.test.ts` 仍绿（分块器本身没动）；core 全量；`npm run typecheck`
- **「没读 PDF 字节」这条断言比「块数变少」更重要**：块数少可以靠截断伪造，而不读文件才是省下解析时间的那件事，也是这次改造真正的机制。
- 一到两块 vs 恰好一块：标题与摘要**分开成块**还是拼成一块，影响 `maxChunkCosine` 的行为（分开则标题能单独匹配）。取红时按「一到两块」写，实现时定死一种并在此处记下理由。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 3 — fallback 论文的有界文本路径

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增测试——喂一个没有 catalog 记录的 fallback 单元（`file:sha256:…`），断言产出的块数有**明确上界**（与 Chunk 2 同量级），而不是按全文分块。红在 fallback 仍走全文路径。
- Green check: `npm run test --workspace @arxiv-daily/core -- fulltext-index`
- regression checks: 既有的 fallback / 标题提取测试全绿（`TITLE_EXTRACTION_VERSION` 相关断言）；core 全量
- **fallback 仍然要读 PDF，这是没有出路的**：它没有摘要，标题今天就靠 `extractTitleFromFirstPage` 从首页取。本 chunk 的目标不是让它别读文件，而是让它**别把全文都嵌进去**。
- 首页取到的是版式文本而非结构化摘要，质量必然比 catalog 摘要差。**不为此新造摘要识别启发式**——那是另一个题目；取有上界的首页文本即可，质量差异记进 open questions。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 4 — 强制全库重建，并堵上陈旧元数据的复用洞

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 两条断言。(a) **重建**：一份按旧 derivation 建的 ready 记录，在本次改造后**不得被判为 `reused`**，必须重新索引；红在 `sameDerivation` 仍然相等。(b) **陈旧洞**：catalog 的 `abstract` 变了而文件的 `observationFingerprint` 一字未动时，该论文**必须重新嵌入**；红在当前 `exactReady` 不看文本，判为 `reused` 并留着旧向量。
- Green check: `npm run test --workspace @arxiv-daily/core -- fulltext-index`
- regression checks: 增量路径 `packages/core/src/library/incremental/` 相关测试；core 全量；CLI 71/71
- (a) 由抬升 `CHUNK_DERIVATION_VERSIONS`（`evidence-chunk.ts:33`）达成——这正是 ADR 0013 §3「重建而非迁移」的执行点，也顺带保证**不会出现半旧半新的混合索引**。混合是这里最坏的失败模式：一部分论文 113 块、一部分 2 块，`maxChunkCosine` 会系统性偏向全文那部分，而且**不报错**。
- (b) 需要让**索引文本的哈希**进入复用判定。`textHash` 字段已存在且已持久化（`knowledge-base.ts:48`），改为覆盖 title+abstract 并参与 `exactReady` 即可，**不需要动 manifest schema**。若发现必须改 schema，见 abort trigger。
- 识别出的论文不再读 PDF 之后，`derivation.parser` 对它们已无意义。实现时必须给这条路径一个**说得出口的 provenance**，不要把 PDF 解析器的 id 写在一条根本没解析过 PDF 的记录上。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 5 — 在测试库上复量，并确认聚类没有塌

- change kind: non-behavioral（验收测量）
- strategy: correctness + performance baseline
- Red / baseline signal: 对照 Chunk 1 的基线。判据：**块数降到每篇一到两块量级、体积降到 MB 级、重建耗时进入分钟级**。同时在同一份索引上跑一次聚类，确认**仍能产出方向**且不退化为「一个大簇」或「每篇一簇」。
- Green check: 测试库重建 + 一次聚类，数字与方向数落进 journal.md
- regression checks: core 全量、plugin 全量、CLI 71/71、`npm run typecheck`、`npm run lint`、`npm run check:boundaries`
- **「仍能产出方向」是活性判据，不是质量判据。** 方向是否变粗（ADR 0013 Consequences 明写为「要测的量」）需要与旧索引的方向做人可读的对比，**判断权在用户**，不由测试代劳。本 chunk 只负责把两边的方向都摆出来。
- 桌面验收：索引是后台过程，无渲染几何可量。**判定为不需要桌面场景**——但若改动触及设置页的索引进度显示，该判定作废（见 abort trigger）。
- exception: 这是测量与验收，没有可取的红。补偿验证是**必须与 Chunk 1 的基线同库同法比对**，不同库或不同参数的数字不作数。
- [ ] implementation and tests accepted

## Phase verification

- 测试库上的前后对照：块数、体积、耗时、embedding 调用数（Chunk 1 vs Chunk 5，同库同法）。
- 全量回归：core、plugin、CLI、typecheck 四包、lint 0 error（20 warning 为既有基线）、check:boundaries。
- **筛选管线未动的硬判据**：`packages/core/tests/settings-rollback.test.ts` 五条仍绿——P1 用它钉死 `topicLines` 逐字不变，P2 不该碰到它，碰到了就说明越界了。
- 旧索引确实被重建而非复用：全库重建后不存在 derivation 为旧版的 ready 记录。

## Abort / reshape triggers

- **若 Chunk 1 量出 fallback 论文占多数，且解析时间是大头**，则「改索引文本来源」买不到分钟级——停下，L2 reshape，重新想（例如先补识别率，或对 fallback 也只取首页而不解析全篇）。
- 若堵陈旧洞需要改 manifest schema 或存储格式，停下——那是独立的一个阶段，不在本阶段顺手做。
- **若换成摘要级之后聚类塌成一个大簇或每篇一簇**，停下。那是对 ADR 0013 §1 的反证，按 L2/L3 分类，**不要在 P2 里调阈值把它掩过去**——阈值实测是 P4/P5 的活，goal 的 Constraints 明写两个阈值不得拍脑袋定。
- 若本阶段发现某个 `description` 或筛选消费点必须改，停下移交 P3。P2 不碰筛选契约。
- 若改动触及设置页的索引进度 UI，则「不需要桌面验收」的判定作废，按上一轮的教训补场景——且**任何桌面变异检验必须先 `npm run build --workspace obsidian-arxiv-daily`**，否则跑的是陈旧产物（journal 2026-09-02 记过这次翻车）。

## Open questions（P2 不顺手定掉）

- **ADR 0008 的全文授权深度。** 本阶段之后离开机器的只剩标题与摘要，「全文深度」很可能已无对象（ADR 0013 Consequences 明写）。P2 **只记录这个事实**，退不退休那条 ADR 是 goal 的非目标，需要单独回头看 ADR 0008。
- **fallback 论文的文本质量。** 首页版式文本不等于摘要。占比若小可以忍；占比若大，值得单独做一次摘要识别——不在本阶段。
- **聚类的 per-paper chunk 上界（`MAX_CLUSTERING_CHUNKS_PER_PAPER = 80`）与 `recluster.ts` 的 centroid 理由**，在每篇只剩一到两块之后都成了死条款。清理它们是收益为零的改动，等 P4 真的动聚类时一并处理。
- **`2026-08-13-discovery-loop-and-library-insight` 的 P4「检索规模加固」**：动机在本阶段落地后基本消失，是否收束由用户定，本 goal 不动它。
