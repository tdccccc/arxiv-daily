# P2 — index-titles-and-abstracts

goal_ref: ../goal.md
created: 2026-09-02T22:47:24+08:00
updated: 2026-09-03T11:48:00+08:00
revision: 9

## Outcome

库索引每篇论文只产出一到两块（标题 + 摘要），不再对 PDF 全文分块。千篇量级的库在 CPU 上分钟级重建完，索引体积从百 MB 级降到 MB 级。旧索引被**重建而非迁移**（ADR 0013 §3）。筛选管线与设置页在本阶段一行不改。

## Assumptions

> **revision 2（2026-09-02）**：Chunk 1 的实测推翻了 revision 1 的核心前提。原计划假定「摘要来自 catalog」，识别不出的论文只是边缘情况；实测证明在真实语料上恰恰相反。下面是修正后的前提，L1 调整的记录见 journal。

- **索引文本一律以 PDF 内容为准，不看 arXiv 号也不看文件名**（2026-09-02 用户决定）。这条把 revision 1 的「catalog 优先 / fallback 兜底」双路径压成**一条路径**，并顺带消掉了原 Chunk 4 的陈旧洞：文本既然派生自文件本身，现有的 `observationFingerprint` 就已经是正确的变更信号，不需要让 `textHash` 参与复用判定。

- **驱动这条决定的实测**（冻结语料 `~/Desktop/plugin_test/baseline_library_207`，212 个文件，跑真实 core 代码）：能找到 arXiv id 的只有 **3 篇（1.4%）**；有 DOI 但无 arXiv id 的 96 篇（45.3%）；**什么元数据都没有的 113 篇（53.3%）**。这个库是天文期刊 PDF，不是 arXiv 预印本集。**任何依赖识别率的方案都有一半以上的库够不着**，所以 PDF 是唯一能覆盖全库的摘要来源。

- **「分钟级」的可行性取决于只解析首页，不解析全文。** 支持证据：只读元数据区的轻量识别提取器跑完 212 篇用 **26 秒（约 123ms/篇）**，外推千篇约 2 分钟。若摘要提取退化成解析整篇 PDF，这个预算立刻不成立——见 abort trigger。

- **摘要区域要靠启发式找，质量不如真摘要。** 期刊 PDF 的版式各异，「Abstract」标题、双栏、扫描件都会让提取失手。**失手的兜底必须是「取首页前 N 字符」而不是「退回全文分块」**，否则一篇论文就能把块数拉回三位数。

- **`chunkFullText` / `chunkParsedDocument` 保留，不删。** ADR 0013 §2 把「把结论段也纳入索引」列为最可能的下一增量，删掉分块器等于给那次回头路凭空加成本。本阶段改的是**喂给分块器什么**，不是分块器本身。

- **聚类会因此变化，这是 ADR 0013 认下的代价，不是本阶段要调的东西。** 聚类直接吃 chunk 向量（`clustering/paper-vector.ts`，2026-08-06 的 L2 已移除 paper-level mean pooling），每篇只剩一到两块之后 `maxChunkCosine` 事实上退化为「摘要对摘要的余弦」。方向是否变粗是**要测的量，不是要假设的事**（ADR 0013 Consequences），且阈值实测被 goal 排在 P4/P5。**P2 只负责确认聚类仍能跑出方向，不负责重新调参。**

## Approach

基线已经量到（Chunk 1）。接着写一个纯函数：从首页文本里定位标题与摘要区域，产出一到两块，找不到就取首页前 N 字符兜底（Chunk 2）。再把索引的文本来源从「解析全文 → 全文分块」换成它，并把解析范围收到首页（Chunk 3）。然后抬 derivation 版本强制全库重建（Chunk 4）。最后在冻结语料上复量，并确认聚类没塌（Chunk 5）。

## Chunks

### Chunk 1 — 量出改造前的基线与识别天花板

- change kind: non-behavioral（测量，不改源码）
- strategy: correctness + performance baseline
- Red / baseline signal: 无红可取——这是基线本身。
- Green check: 数字落进 journal.md，且与 ADR 0013 可比对
- regression checks: 不适用
- exception: 测量任务无红可取。补偿验证是数字必须来自真实语料的真实代码运行。
- **已完成，结果如下**：
  - **冻结语料**：`~/Desktop/plugin_test/baseline_library_207`，212 个文件，由磁盘上那份索引的 manifest `filePaths` 复原、以硬链接固定（磁盘零增长，link count 2）。Chunk 5 用同一份做前后对照。
  - **ADR 0013 基线完整复原**（无需重跑全文侧）：**207 篇 / 23,423 块 / 140MB / 768 维**，`remote:nomic-embed-text`；均值 **113.2** 块/篇，中位数 79，**最大 1002**；status ready 204 / failed 3。
  - **识别天花板**：arXiv id 命中 3（1.4%）、/Title 内再抽 0、有 DOI 无 arXiv id 96（45.3%）、无任何元数据 113（53.3%）。**这 207 篇当年 100% 落在 fallback 键上，且用的就是今天这版识别**（两代索引的 identification fingerprint 同为 `73049d71…`）。
  - **轻量提取吞吐**：212 篇 26 秒，约 123ms/篇 → 千篇约 2 分钟。这是「分钟级」的可行性依据。
  - **顺带发现一个真缺陷（本阶段不修）**：`pdf-text-utils.ts` 的 `LEGACY_ARXIV_ID_IN_TEXT_RE` 少了 `(?:v\d+)?`，而新式正则有——于是 `astro-ph/0003380v2` 这类带版本号的旧式 id 提取不出（尾部断言撞上 `v`）。按用户决定「以 PDF 内容为准，不纠结 arXiv 号」，本阶段不动它，记在 journal 里。
- [x] implementation and tests accepted

### Chunk 2 — 从首页文本定位标题与摘要（纯函数）

- change kind: behavior change（新增纯函数）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增 `packages/core/tests/` 测试——喂入首页文本，断言取出标题与摘要正文，且**摘要不含 "Abstract" 这个词本身**、不含作者与单位行；喂入找不到摘要标记的首页，断言**兜底为首页前 N 字符而不是空**、也不是全文。红在函数不存在。
- Green check: `npm run test --workspace @arxiv-daily/core -- abstract-extraction`
- regression checks: `fulltext-chunking.test.ts` 仍绿（没动分块器）；`npm run typecheck`
- **纯函数、无 I/O、确定性**，与 `chunking.ts` 同一风格——可以拿真实首页文本当夹具反复取红，不必碰 PDF 字节。
- **兜底必须有上界**，这是本 chunk 最要紧的一条断言：任何输入都不得产出超过约定字符数的文本，否则版式一怪就把块数拉回三位数。
- 用冻结语料里的真实首页做几条夹具（含双栏、含无 Abstract 标题的期刊版式），不要只测理想输入。
- exception: 无
- **已完成**。`packages/core/src/library/fulltext/abstract-extraction.ts`，三条路径逐页尝试：`marker`（`Abstract` / `ABSTRACT` / 字间距 `A B S T R A C T`，独占行与行内起头都认）、`dated`（REVTeX 不印 Abstract 一词，正文跟在 `(Dated: …)` 后）、`leading-text`（有界兜底）。无文本层判 `none` 而**不兜底**——把 bibcode 水印当摘要嵌进去比没有向量更糟。
- **真实语料实测（212 篇）**：marker 177（83.5%）、dated 2（0.9%）、leading-text 26（12.3%）、none 7（3.3%）。**84.4% 走的是真摘要路径，兜底率 12.3% 远低于 abort trigger 的「多数」阈值，不触发。** 摘要长度中位数 1594 字符、p90 2397；总索引字符 **337,786**，对比改造前约 4800 万字符。
- **变异检验发现了一个测试盲点**：禁掉终止规则后三条主路径断言变红，但「摘要在第 2 页」那条**照样绿**——它的 `not.toMatch(/Bocquet/)` 查的是上一页的内容，管不住终止。已补一条 `not.toMatch(/Introduction/)`。这与 P1 那条「红必须红在被测行为上」是同一类错误的另一面：**绿也必须绿在被测行为上**。
- **一处刻意不做的调优**：兜底样本里 `Beck2022`、`DES Collaboration2022` 取到的是期刊卷页页眉而非标题，加几条正则就能修好。**没有修**——那等于拿 8 个样本调参，正是 goal 的 Constraints 里「阈值不得拍脑袋定」警告的同类错误，且兜底文本质量我没有可测判据，只有主观观感。记入 open questions。
- [x] implementation and tests accepted

### Chunk 3 — 解析加页数上界（port + 宿主）

- change kind: behavior change（跨包契约变更）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 断言 `extractPdfText(bytes, { maxPages: 2 })` 只返回两页，且**宿主只对这两页调用 `getPage`**（后者是真判据——只截断返回值等于没省解析）。红在选项不存在。
- Green check: `npm run test --workspace obsidian-arxiv-daily -- pdf-text-extractor`
- regression checks: 不带 `maxPages` 时行为逐字不变（既有 extractor 测试全绿）；core / plugin 全量；typecheck
- **改动面**：core 的 `PdfExtractionOptions` 加可选 `maxPages`，plugin 的 `pdf-text-extractor.ts:229` 逐页循环改上界。用户 2026-09-02 定：做。
- **「只截断不少解析」是本 chunk 唯一会骗人的失败模式**，所以取红判据落在 `getPage` 调用次数上，不落在返回页数上。
- exception: 无
- **已完成**。`PdfExtractionOptions` 与 `ParseDocumentOptions` 各加可选 `maxPages`，`pdf-text-extractor.ts` 的逐页循环收上界。契约明写**解析器可以忽略它**，调用方不得假定结果已被限长——sidecar 那条路径就不吃这个选项。不传 `maxPages` 时行为逐字不变。
- 三条新断言，核心那条数的是 `getPage` 调用次数：解析全篇再 `slice` 能满足「返回两页」却一点不省，正是本 chunk 要防的假绿。
- [x] implementation and tests accepted

### Chunk 4 — 索引改走 extractor + 摘要提取，产出一到两块

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 一篇几十页的论文，索引产出**恰有一到两块**（标题 + 摘要），且**只解析前两页**。红在当前解析全文并产出几十块。
- Green check: `npm run test --workspace @arxiv-daily/core -- fulltext-index`
- regression checks: core 全量；plugin 全量；CLI 71/71；typecheck
- **索引路径不再走结构化 parser**（用户 2026-09-02 定）。`headings` / `locator` 都是为全文分块服务的，摘要提取只要纯文本；Docling 因此在**索引路径上**失去对象——与 ADR 0008 全文授权深度那条情形同类。`parseIndexDocument` 的 parser / parserSelector 分支在索引路径上退场，`derivation.parser` 收敛为 extractor 的 provenance。
- **必须给 extractor 补上 provenance，否则换引擎不再触发重建。** `DocumentParser` 声明 `provenance {id, version}`，它进 chunk 指纹也进复用判定，所以换解析器或升版本会整批重建；**而 `PdfTextExtractor` 没有这个字段**，走 extractor 分支时 derivation 硬编码为 `LEGACY_PARSER_PROVENANCE`（`index-orchestration.ts:666`）。把索引挪到 extractor 上就等于丢掉这层保护——**这是「混合索引」那个最坏失败模式的第二个入口**：不再是一部分 113 块一部分 2 块，而是一部分文本来自这个引擎、一部分来自那个引擎，同样静默，同样让 `maxChunkCosine` 失去可比性。取红判据：换一个 provenance 不同的 extractor 实现后，既有 ready 记录**不得被判为 `reused`**。
- **标题提取要对所有论文生效，不再只对 fallback**：catalog 有标题的只有 1.4%，维持现状会让 98.6% 的论文没有标题进索引。`extractTitleFromFirstPage` 本就是针对这个库调优的（26 条测试在），改的是调用条件不是算法。
- catalog 有摘要时是否优先用它？**不用**——用户定的是以 PDF 内容为准，且双来源会让同一个库里的文本失去可比性，聚类相似度跟着不可比。
- exception: 无
- **进行中。第一部分（extractor provenance）已完成并验证**：`PdfTextExtractor` 现在要求声明 provenance，与 parser 层对称；extractor 分支记录它，复用判定读它；Obsidian 的 extractor 报告它实际委托的那个 pdf.js parser 的 provenance。两条新测试——换 provenance 必须重建、不换必须复用且不重新解析。**变异检验把 provenance 从复用判定里拿掉后 7 条红**，确认「记录」与「判定」两处都被守着。core 2062 / plugin 721 / CLI 71 全绿。
- **第二部分（索引改走 extractor + 摘要提取）未做**，它必然要动既有测试：现有断言里有「记录每篇实际选中的 parser」「parser derivation 变则重建」「sidecar 失败回退到 fallback parser 并记录后者」这几条，索引不再走 parser 之后它们全部失去对象。**这不是可以顺手改绿的事**——要逐条判定哪些是被本阶段正当取代、哪些是不该丢的保护（例如 sidecar 回退的正确性在别处仍然成立）。

#### 三条 parser 断言的逐条判定（2026-09-03）

判据不是「索引还走不走 parser」，而是**每条断言守的那个失败模式，在别处有没有home**。逐条查过接管方后：**三条都可删，无一需要搬走**。

- `fulltext-index-orchestration.test.ts:307` `indexes ParsedDocument input with parser derivation and structured headings` — **删**。三个断言各有接管方：`derivation.parser` 记录引擎身份 → 已由 extractor 侧的 `:431`（换 provenance 必重建）+ `:472`（不换必复用）接管；`headings === ["Methods"]` → `fulltext-structured-chunking.test.ts:93-97` 覆盖更全（三级标题继承链）；`locator === {pageStart:2,pageEnd:2,blockStart:1,blockEnd:1}` → `fulltext-structured-chunking.test.ts:103-107` **逐字同一个断言**。此测唯一独有的是「索引路径能把结构化 parser 接通」——而这正是 ADR 0013 要退掉的东西，不是要保住的。

- `:344` `persists the actual parser selected for each indexed document` — **删**。这条是三条里唯一真的要想的，因为它守着一个**每篇不同**的失败模式（selector 逐篇选引擎，索引必须记下**实际选中**的那个而不是首选那个）。查证：
  - 「sidecar 失败 → 回退 → 报告 fallback 身份」→ `sidecar-document-parser-client.test.ts:132-167`，且第 164 行直接断言 `selected.parser.provenance === fallback.provenance`。**这就是 handoff 担心的那条保护，它在 selector 自己的层高上已经钉住了**，比在索引里验更贴切。
  - 「结构化文档带 headings / 纯页文档 headings 为空」→ `fulltext-structured-chunking.test.ts:21`（page-only 分支 headings 必空、pageEnd 必 undefined）+ `:93`。
  - 残余的「preferred vs actual」不对称确实只有此测在钉：`index-orchestration.ts:139` 的 `expectedDerivation` 取的是 `parserSelector.preferredParser.provenance`，而 `parseIndexDocument:651` 记的是**实际选中**的 parser。selector 退出索引路径后这条不对称**随之消失**（extractor 是单个对象、单个 provenance，没有选择动作就没有不对称）。**删此测时必须同时删掉 `:139` 的 `preferredParser` 分支**，否则留下一个没有测试守的歪逻辑。

- `:403` `re-indexes unchanged v2 content when parser derivation changes` — **删**。`:431` `re-indexes unchanged content when extractor provenance changes` 是它逐项的孪生（indexed=1 / reused=0 / 解析调用 1 次 / embedding +1 / 存储的 provenance 版本已更新），只是把 parser 端口换成 extractor 端口。反方向由 `:472` 守。

**判定期间发现的一处计划低估**（见下方 open questions）：上面第 88 条写的是「Docling 在**索引路径上**失去对象」，但 `buildFullTextDocumentParser`（`plugin/main.ts:1603`）**只有一个调用方**——`plugin/main.ts:1695`，就在索引里。索引路径即全部路径，所以失去对象的不只是索引侧的 Docling，而是整条 sidecar 装配：`SidecarFallbackDocumentParserSelector`、`probeLoopbackSidecarParser`，以及**三个用户可见设置行** `pdfParserSidecar.{enabled,capabilitiesUrl,parseUrl}`（`declarative-rows.ts:684-727`）。这超出本 chunk 的授权范围，**待用户定**，不在 Chunk 4 里顺手做。

#### 第二部分落地（2026-09-03）

- `buildPaperDocument` 改为：extractor 取前两页 → `extractTitleFromFirstPage` + `extractAbstractFromPages` → 拼成一段文本 → 走既有 `chunkParsedDocument`。`parseIndexDocument` 换成 `extractIndexPages`，parser / parserSelector 从 `IndexPersonalLibraryFullTextInput` 移除，`extractor` 变成必填。`:139` 的 `preferredParser` 分支按判定连带删除。
- **`maxPages` 传给 extractor 而不是切结果**，且与 `extractAbstractFromPages` 共用导出的 `MAX_LEADING_PAGES`——两个边界若各写各的，「摘要在第 2 页」那 0.9% 会静默退化成兜底且无人报告。
- 标题提取改为对所有论文生效，`titleVersion` 恒定写入；`:222` 的复用条件相应去掉 `!unit.fallback ||`，陈旧标题版本对全库都使复用失效。
- **一处实现中发现的真问题**：分块器默认 `minChunkChars: 16` 会把短标题整条滤掉。走全文时这是丢页眉噪声，走标题+摘要时它把「有标题、无可用摘要」的论文（实测 3.3% 的 `none` 路径）压成**零块 ready 记录**——检索永远匹配不到，且不报错。索引路径改传 `minChunkChars: 0`：这段文本是刻意拼出来的，不存在噪声。补测试钉住。
- **变异检验三条全部取红且位置正确**：去掉 `maxPages` → 只有「只开前两页」那条红（其余不动，说明该判据独立）；恢复 `minChunkChars` 默认 → 短标题那条红；标题退回只给 fallback → 6 条红。
- validation: core **2062**、plugin **720**、CLI **71/71**、typecheck 四包、lint **0 error / 20 warning**（与基线一致，中途多出的 4 条 unused import 已清）、check:boundaries OK。
- boundary: sidecar 代码与三个设置项**一个未删**，只是失去调用方；`pdfParserSidecar.enabled` 为真时索引记一条 info 说明它不参与索引。筛选管线未动。

- [x] implementation and tests accepted

### Chunk 5 — 抬 derivation 版本，强制全库重建

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 一份按旧 derivation 建的 ready 记录，在本次改造后**不得被判为 `reused`**，必须重新索引。红在 `sameDerivation` 仍然相等。
- Green check: `npm run test --workspace @arxiv-daily/core -- fulltext-index`
- regression checks: 增量路径 `packages/core/src/library/incremental/` 相关测试；core 全量；CLI 71/71
- 由抬升 `CHUNK_DERIVATION_VERSIONS`（`evidence-chunk.ts:33`）达成——ADR 0013 §3「重建而非迁移」的执行点。**最坏失败模式是混合索引**：一部分论文 113 块、一部分 2 块，`maxChunkCosine` 会系统性偏向全文那部分，**而且不报错**。
- Chunk 4 让 `derivation.parser` 变了，可能已经隐含触发重建。**仍要显式抬版本**：靠副作用达成的重建，下次有人改回 provenance 就会静默失效。
- **revision 1 里的「陈旧摘要复用洞」在新方向下不存在**：文本派生自文件内容，文件不变则文本不变，现有 `observationFingerprints` 已是正确信号。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 6 — 在冻结语料上复量，并确认聚类没塌

- change kind: non-behavioral（验收测量）
- strategy: correctness + performance baseline
- Red / baseline signal: 对照 Chunk 1。判据：块数降到每篇一到两块、体积降到 MB 级、重建耗时进入分钟级。同一份索引上跑一次聚类，确认**仍能产出方向**且不退化为「一个大簇」或「每篇一簇」。
- Green check: 冻结语料重建 + 一次聚类，数字与方向数落进 journal.md
- regression checks: core / plugin / CLI 全量、typecheck、lint、check:boundaries
- **必须用同一份冻结语料**（212 个文件，硬链接固定），否则数字不可比。
- **要用真解析器复核 Chunk 2 的路径分布**：那组百分比是拿 `pdftotext` 当代理量的，产品走 pdfjs，文本会有出入。
- **「仍能产出方向」是活性判据，不是质量判据。** 方向是否变粗需要与旧索引的方向做人可读的对比，**判断权在用户**。
- 桌面验收：索引是后台过程，无渲染几何可量，判定为不需要——除非改动触及设置页的索引进度显示。
- exception: 测量与验收无红可取。补偿验证是必须与 Chunk 1 同库同法比对。
- [ ] implementation and tests accepted

## Phase verification

- 冻结语料上的前后对照：块数、体积、耗时（Chunk 1 vs Chunk 5，同库同法）。基线为 207 篇 / 23,423 块 / 140MB。
- 全量回归：core、plugin、CLI、typecheck 四包、lint 0 error（20 warning 为既有基线）、check:boundaries。
- **筛选管线未动的硬判据**：`packages/core/tests/settings-rollback.test.ts` 五条仍绿——P1 用它钉死 `topicLines` 逐字不变，P2 碰到了就说明越界了。
- 全库重建后不存在 derivation 为旧版的 ready 记录。

## Abort / reshape triggers

- **若摘要提取为了准确率而不得不解析整篇 PDF**，「分钟级」立刻不成立（123ms/篇 的实测预算不容全篇解析）——停下重新分类，不要一边解析全文一边宣称瘦身成功。
- **若首页启发式的失手率高到让多数论文走兜底**，则索引到的是「首页前 N 字符」而非摘要。体积收益照拿，但「索引覆盖摘要」这条成功标准名不副实——停下告诉用户，由用户决定是否降低这条标准或引入 DOI/Crossref。
- **若换成摘要级之后聚类塌成一个大簇或每篇一簇**，停下。那是对 ADR 0013 §1 的反证，按 L2/L3 分类，**不要在 P2 里调阈值把它掩过去**——阈值实测是 P4/P5 的活。
- 若本阶段发现某个 `description` 或筛选消费点必须改，停下移交 P3。P2 不碰筛选契约。
- 若改动触及设置页的索引进度 UI，则「不需要桌面验收」的判定作废——且**任何桌面变异检验必须先 `npm run build --workspace obsidian-arxiv-daily`**（journal 2026-09-02 记过这次翻车）。

## Open questions（P2 不顺手定掉）

- **ADR 0008 的全文授权深度。** 本阶段之后离开机器的只剩标题与摘要级文本，「全文深度」很可能已无对象（ADR 0013 Consequences 明写）。P2 只记录事实，退不退休那条 ADR 是 goal 的非目标。
- **DOI/Crossref 作为摘要来源。** 实测 45.3% 的论文有 DOI，是拿到真摘要的现成路。本阶段不做——它新增外部依赖与授权面，值得单独决定。
- **旧式 arXiv id 带版本号提取不出**（`LEGACY_ARXIV_ID_IN_TEXT_RE` 少 `(?:v\d+)?`）。真缺陷，按用户方向本阶段不修。
- **兜底文本的质量。** 26 篇走 `leading-text` 的里面，有几篇开头是期刊卷页页眉而非标题。可修，但需要先有可测的质量判据，不能按样本调正则。
- **聚类的 `MAX_CLUSTERING_CHUNKS_PER_PAPER = 80` 与 `recluster.ts` 的 centroid 理由**，在每篇只剩一到两块之后都成了死条款。等 P4 真的动聚类时一并处理。
- **`2026-08-13-discovery-loop-and-library-insight` 的 P4「检索规模加固」**：动机在本阶段落地后基本消失，是否收束由用户定。
- **整条 PDF parser sidecar 装配的去留（2026-09-03 判定三条 parser 断言时发现）。** 计划第 88 条说的是「Docling 在索引路径上失去对象」，实际上 `buildFullTextDocumentParser`（`plugin/main.ts:1603`）只有 `plugin/main.ts:1695` 一个调用方，就在索引里——**索引路径即全部路径**。Chunk 4 第二部分落地后，`SidecarFallbackDocumentParserSelector`、`probeLoopbackSidecarParser`、`ObsidianPdfDocumentParser` 的 parser 身份（它作为 extractor 的委托对象仍在）、以及三个用户可见设置行 `pdfParserSidecar.{enabled,capabilitiesUrl,parseUrl}` 全部失去生产调用方。**Chunk 4 只让它们失去调用方，不删任何一个**——删设置项是用户可见的收窄，须用户单独决定。相关测试（`sidecar-document-parser-client.test.ts` 等）在代码还在时照常保留。
