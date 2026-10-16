# P2 — library-novelty-in-digest

goal_ref: ../goal.md
created: 2026-10-15T10:55:00+08:00
updated: 2026-10-15T13:10:00+08:00
revision: 2

## Outcome

插件与 CLI 的日报，对每篇入选论文（上限 20）用其标题+摘要在库索引里检索最相似的至多 5 篇库论文，再让模型判定「相对这些论文新在哪」或「无可比先验」，前者写入日报 Markdown（标记+可见行）并在 Dashboard 论文行显示。只在库处理授权有效时运行；未授权/无库/无索引/模型判无可比/任何出错 => 该篇无说明，日报照常完成。无相似度下限。

## Assumptions（均已读码核实）

- `pipeline/` 现在不 import `../library`（仅 `settings/accept-proposed-topics.ts` 引用 library）。`check:boundaries` 只管包层级（core 不得用 Node 内建/第三方），不禁止 core 内部目录互引；"流水线不依赖库模块"是 goal 约束，用端口落实，并加一条 grep 式测试守住（见 Chunk 3）。
- 库论文全是 arXiv：`PersonalLibraryPaperRecord.source: "arxiv"`，含 title/abstract/authors/published/categories（personal-library-catalog.ts:45-57）；`NoveltyRepresentativePaper` 与标记都要求规范 `arxiv:` paperKey（personalized-novelty.ts:1221）。非 arXiv 的兜底索引文件（搜索结果带 `filePath`）在 catalog 里没有摘要，不能当对比基准。
- 授权指纹绑定 root + `llmBaseUrl`（+远程 embedding 端点）（library-connection.ts:17-23,371），两个宿主都用日报同一个 `settings.llm.baseUrl` 算指纹（plugin/main.ts:608；apps/cli/src/library-connection-cmd.ts:74）。所以 `status.kind==="authorized"` 精确等于「用户同意把库的标题/摘要发给日报正在用的这个模型端点」，闸门是诚实的。
- 检索 `searchFullTextKnowledgeBase`（index-orchestration.ts:843 起）：`mode:"lexical"` 不调 embed，embedding 模型对象是惰性的（transformers-embedding-model.ts:40-，首次 embed 才加载），所以本地 lexical 不会触发 130MB 模型加载/卡顿。
- 日报重跑会重走本阶段；检查点以「计划指纹」（含对比论文内容）判重，库没变=命中，库变了=重新生成。checkpoint 在 `checkpointStores.filter` 里（NoveltyCheckpointPort，pipeline.ts:85-92），两个宿主都已传。
- 测试命令：core `npm test -- tests/<f>.test.ts`（根目录，转发到 core 工作区，run-root-tests.mjs）；插件 `npm run test --workspace obsidian-arxiv-daily -- tests/<f>.test.ts`；CLI `npm run test --workspace arxiv-daily -- tests/<f>.test.ts`。

## Approach（回答关键设计问题）

1. **端口**（类型放 `personalized-novelty.ts`，实现放 core `library/`，宿主只接线）：
   `LibraryPriorWorkPort { findSimilar(query:{paperKey,title,abstract}, limit, signal?): Promise<NoveltyRepresentativePaper[]> }`，经 `PipelineDeps.libraryPriorWork?` 注入；缺省/null=未授权或无库，阶段整体跳过且不报进度。每次 `findSimilar` 内部先调宿主传入的 `assertAuthorized()`（撤销/改端点后中途失效），失败抛 `LibraryPriorWorkUnavailableError`，阶段收到后对剩余论文不再检索（避免 20 条重复警告）。端口负责：检索 limit+1、排除论文自身 paperKey（已在库里的论文不与自己比）、catalog 联表、丢弃不满足 DTO 界限的库论文（标题/摘要空或超界，不截断）、截到 5。
2. **输入改形（最小）**：`PersonalNoveltyMatchInput` 从 direction→representatives 改为 `{ paperBases: [{paperKey, basisPaperKeys}] }`（删 `directionRepresentatives`/`directionIds`、`preparePersonalizedNoveltyRepresentatives` 及其宿主快照类型，无人使用）。`input.representatives` 取各论文对比集合并集（≤100，上界 `PERSONAL_NOVELTY_MAX_REPRESENTATIVES` 保持）。新增 `PERSONAL_NOVELTY_MAX_RETRIEVED_BASIS = 5` 用于**计划**阶段；读回标记仍用旧上界 40，免得老日报里的标记失效。不采用「每篇造一个伪 direction」的偷懒法（会污染 P3 删除休眠代码后的语义）。
   **「无可比先验」**：prompt 重写（基准是「按文本相似检索出的库论文，可能并不相关」，不可比必须明说）；模型可输出恰好 `{"noComparablePrior":true}`；`decodePersonalNovelty` 增加这个唯一例外形状，返回 `{ok:true, kind:"no-comparable"}`；结果 `no-novelty/reason:"no-comparable-prior"`，加入 stage 与检查点记录的可持久原因（`decodeNoveltyCheckpointRecords` 允许 call 条目以它结束，这样重跑不重问）；`PROMPT/RESULT_CONTRACT_VERSION` 升到 2（旧检查点自然 miss，按日期清理）。缺失的处理：无标记、无 `noveltyByReport` 条目；`parseDailyReportPersonalNovelty` 对 0 个标记返回 valid 空集（personal-novelty-marker.ts），`reconcileDailyReportOccurrenceNovelty` 以空集清除——现有路径已正确，不改。
3. **流水线位置**：在 `visiblePapers`（排除 ignored、全局排序、`slice(dailyLimit)`）之后、fetch-content 之前（pipeline.ts:261-266 处的占位注释换成调用），保证 ≤20 篇、≤20×(≤3 次校验重试) 次调用；结果写回 `visiblePapers[i].personalNovelty`（用 `attachPersonalNoveltyBasis` 带标题，`...p` 展开自然流到 summarizer.ts:112 → 渲染）。进度 `setStage("personal-novelty", i, n)`（类型与状态栏 status-bar.ts:9,21 已有，标签 "novelty" 不改）。仅 `sourceAdapter.sourceId==="arxiv"` 运行。任何非取消异常 => `logger.warn` + 空结果；取消照常上抛。检查点：沿用 `runPersonalNoveltyStage` 的逻辑（命中复用；任一篇 transport/output-limit 降级则不保存，重跑重试；无法保存=整体降级无说明）。注意检查点 JSON 内含对比论文摘要（本地库数据，随提交后 `removeAll` 删除），写进 ADR 0015 的 Consequences 一句。邮件摘要不取新颖性（digest 由 slots 构建，Chunk 3 加断言）。
4. **检索模式与授权**：宿主按自己的 embedding 模式选——`local` => `lexical`（零 embedding、零卡顿；召回质量未测，列入验证）；`remote` => `hybrid`（用既有远程端点，授权状态本来就含该端点；配置不全 => 端口不可用 => 无说明，不静默降级到别的基准）。授权判定=`getLibraryConnectionStatus().kind==="authorized"`（插件，live 值）/ `inspectCliLibraryConnection(config).status.kind==="authorized"`（CLI）。**需要用户确认的一点**：现有授权文案只写「用于后续方向生成步骤」（plugin/src/library/modal.ts:69）；日报新颖性是新用途。建议：不改指纹（旧授权继续有效，ADR 0015 §2 已写明按现有授权门控），但更新插件弹窗与 CLI disclosure 文案，说明日报会把检索到的库论文标题/摘要发给同一模型端点。若用户要求「旧授权失效、重新同意」，则需改指纹版本，属 UI 迁移，不在本 phase 范围，需回到用户。
5. **CLI 对等**：CLI 有库连接/授权/索引/搜索（library-cmd.ts:103-121，`createCliLibraryContext`），日报在 `buildCliRuntime`（runtime.ts:145）构造 `ArxivPipeline`。端口用 core 构造器 + 直接构造 `FullTextKnowledgeBaseFileStore/FullTextGenerationIndexStore/PersonalLibraryCatalogStore`（不要调 `createCliLibraryContext`：它会打开库目录、建 workflow、拉 LLM，日报用不上）。已知限制：CLI 配置在进程启动时读入，常驻 `schedule` 进程内授权变更要重启才生效（现状如此，不扩范围）。
6. **Dashboard**：不改行为。行已经显示类型/对比论文 paperKey/证据深度/解释（view.ts:157-168，model.ts:434-451；工作台 papers.ts 已投影）。标题不进索引（marker 极简载荷是既有契约），这里仍显示 `arxiv:` key。只修 view.ts:150-156 注释里「direction 的代表论文」的过时措辞。是否给 Dashboard 查标题=另议，不做。
7. **进度/状态栏**：无新类型；仅验证调用顺序与计数。

## Chunks

### Chunk 1 — 输入改为逐篇对比集（行为不变的重构）
- change: 按 Approach 2 改 `personalized-novelty.ts` 的 match 输入、规划、`partitionValidPapers`、`singlePaperMatches`、检查点指纹/解码；新增 5 篇上界常量；删无人使用的宿主快照类型；合同版本暂不动。
- kind: refactor
- first signal: 先把 `packages/core/tests/personalized-novelty.test.ts`（35 处 direction 引用）与 `daily-filter-checkpoint-store.test.ts`（7 处）的夹具改成 `paperBases` 形状，`npm test -- tests/personalized-novelty.test.ts` 在旧实现上红。
- checks: `npm test -- tests/personalized-novelty.test.ts tests/daily-filter-checkpoint-store.test.ts tests/personal-novelty-marker.test.ts`（如存在）；`npm run typecheck`。
- [x] accepted on observed evidence

### Chunk 2 — 「库中无可比论文」结果
- change: 重写 `prompts/personal-novelty.system.md`；解码器加 `noComparablePrior` 唯一例外形状；新原因 `no-comparable-prior` 贯穿 outcome/检查点记录/`isValidNoveltyPlan`/解码；合同版本升 2。
- kind: behavior
- first signal: `packages/core/tests/personalized-novelty.test.ts` 新用例：模型返回 `{"noComparablePrior":true}` => `status:"no-novelty",reason:"no-comparable-prior"` 且只调 1 次；额外键/`false`/与完整字段混用 => 走校验重试；检查点保存后再 lookup 命中且不再调模型。
- checks: 同上命令；`grep -rn "representative" packages/core/src/prompts/personal-novelty.system.md` 无残留；marker/paper-index 测试（`tests/paper-index.test.ts`、`tests/daily-summary-assembler.test.ts`）仍绿。
- [x] accepted on observed evidence

### Chunk 3 — 流水线阶段与端口（用假端口）
- change: `PipelineDeps.libraryPriorWork?`；新文件 `pipeline/library-novelty-stage.ts` 做「逐篇检索→组 input/bases→`runPersonalNoveltyStage`→`attachPersonalNoveltyBasis`」，在 pipeline.ts:261-266 处调用；进度、降级、取消语义见 Approach 3。
- kind: behavior
- first signal: 新建 `packages/core/tests/pipeline-library-novelty.test.ts`（复用 `pipeline-daily-cap.test.ts` 的 harness）：假端口返回 2 篇库论文、LLM 对新颖性 prompt 返回合法 JSON => 日报 Markdown 含 `arxiv-daily-personal-novelty` 标记与可见行，`paperIndex` 的 `noveltyByReport` 有值；`npm test -- tests/pipeline-library-novelty.test.ts` 红。同文件覆盖：端口缺省=0 次新颖性调用且日报与现在逐字相同；`findSimilar` 抛错/返回空/模型 `noComparablePrior` => 该篇无标记、日报仍 completed；LLM transport 失败 => 无标记且检查点不保存、重跑重试；`visiblePapers` 超 cap 时调用数=cap；邮件 digest 不含新颖性文字；`setStage("personal-novelty",i,n)` 出现在 filter 与 fetch-content 之间；取消仍返回 `cancelled`。另加一条测试读 `pipeline/*.ts` 源码断言无 `../library` import。
- checks: `npm test -- tests/pipeline-library-novelty.test.ts tests/pipeline-daily-cap.test.ts tests/pipeline.test.ts tests/pipeline/pipeline-error-handling.test.ts`；`npm run check:boundaries`；`npm run typecheck`。
- [x] accepted on observed evidence

### Chunk 4 — core 的库检索端口实现
- change: 新文件 `packages/core/src/library/prior-work.ts` 导出 `createLibraryPriorWork({ catalog:{load}, search(query,limit,mode,signal), assertAuthorized, mode, logger })`：自身排除、catalog 联表（只取有 title+abstract 的 arXiv 论文）、DTO 界限过滤、取前 5；并从 `index.ts` 导出。`search` 由宿主闭包包 `searchFullTextKnowledgeBase`（含 store/generationStore/embedding/titles）。
- kind: behavior
- first signal: 新建 `packages/core/tests/library-prior-work.test.ts`（检索夹具参考 `lexical-search.test.ts`）：自身论文被排除；超过 5 篇取前 5；无摘要的兜底文件被丢；未授权时抛 `LibraryPriorWorkUnavailableError` 且不调 search；索引为空返回 []。
- checks: `npm test -- tests/library-prior-work.test.ts tests/lexical-search.test.ts`；typecheck；boundaries。
- [x] accepted on observed evidence

### Chunk 5 — 插件接线、授权闸门与文案
- change: `plugin/main.ts` `buildPipeline()`（≈2769）按快照构造端口：库已连接且 `authorized` 才传；`mode = settings.embedding.mode==="remote" ? "hybrid" : "lexical"`；`search` 复用 `buildFullTextKnowledgeBaseStore/GenerationIndexStore/buildEmbeddingModel`（`assertRemoteEmbeddingReady` 语义，main.ts:1488）；更新 `plugin/src/library/modal.ts:69` 授权文案；修 view.ts 过时注释。
- kind: behavior
- first signal: 插件单测可测部分——抽出的接线小函数（如 `plugin/src/library/prior-work-wiring.ts`，输入=连接状态/embedding 模式/设置，输出=端口或 null+mode）新建 `plugin/tests/library-prior-work-wiring.test.ts`：未连接/未授权/授权失效=>null；local=>lexical；remote 配置不全=>端口抛不可用。`npm run test --workspace obsidian-arxiv-daily -- tests/library-prior-work-wiring.test.ts` 红。`main.ts` 本身不可单测，**交付证据=真实 Obsidian 手动检查**：授权后手动跑一天日报，看 Markdown 与 Dashboard 行；撤销授权后再跑，无说明、日报正常（做不到则明确记录未做）。
- checks: 同上 + `plugin/tests/dashboard-view.test.ts`、`library-modal.test.ts`；`npm run typecheck`；`npm run build --workspace obsidian-arxiv-daily`。
- [x] accepted on observed evidence

### Chunk 6 — CLI 接线与对等
- change: `apps/cli/src/runtime.ts` 在 `new ArxivPipeline` 前按 Approach 5 构造端口（`config.libraryConnection` 存在且授权有效才传）；CLI 授权 disclosure 文案同步；`library-connection-cmd.ts` 的 disclosure 说明日报用途。
- kind: behavior
- first signal: `apps/cli/tests/cli-runtime.test.ts` 新增（参考 `cli-personalized-workflow.test.ts` 的库夹具）：已授权+已建索引的临时库 => `runtime.pipeline.runForDate` 的日报含新颖性标记；未授权同配置 => 无标记且日报同样完成。`npm run test --workspace arxiv-daily -- tests/cli-runtime.test.ts` 红。
- checks: `tests/cli-library.test.ts tests/cli-library-config.test.ts tests/workbench-papers.test.ts`（工作台已投影 novelty）；typecheck。
- [x] accepted on observed evidence

## Phase verification

- `npm run typecheck`、`npm run check:boundaries`、`npm test`（core 全量）、`npm run test:workspaces`、`npm run build --workspace obsidian-arxiv-daily` 全绿。
- 真实库跑一次（CLI 或 Obsidian）：记录 20 篇里出现说明的比例，并人工读 ≥5 条判断「对比的是否真相关」（ADR 0015 明说质量取决于模型拒答，必须实测；不据此加相似度下限，只记录）。local/lexical 与 remote/hybrid 若都能测则各一次。
- 同一天重跑：库未变时第二次不再调用新颖性模型（检查点命中）。
- 插件 Dashboard/弹窗改动经真实 Obsidian 检查，或在 journal 明确记「未做」。

## Abort / reshape triggers

- Chunk 1 的测试改写超出预期（>1 个会话或要动持久格式之外的消费者）=> 退回「每篇伪 direction」适配层并在 journal 记账，P3 删除休眠代码时一并还债。
- 真实库上 lexical 检索给出的基准多数明显不相关、模型也不拒答 => 不加阈值；回到用户，选项是 local 也用 hybrid（要接受 wasm 模型加载与卡顿风险，ADR 0007 教训）或接受低质量。
- 授权闸门语义：用户要求旧授权失效重新同意 => 超出 P2（需 UI 迁移），回到用户。
- 发现 catalog 与索引常不同步导致大面积 DTO 过滤（基准经常 <1 篇）=> 先查 index/catalog 一致性，不在本阶段修。
- 任一改动要求 pipeline 直接 import library，或 core 引入 Node 全局/第三方 => 停，重设端口。

## 可能触及 goal 的点（供决定，均不强制改 goal）

1. 对比基准只能是库中有 catalog 摘要的 arXiv 论文；非 arXiv 兜底 PDF 不参与（goal 文字未区分，建议在 ADR 0015 Consequences 补一句）。
2. Dashboard 只显示 `arxiv:` paperKey，不显示标题（索引载荷契约如此）；goal 只要求「在 Dashboard 论文行显示」，满足。
3. 授权文案需更新（上文 4）；这是 goal「授权有效时运行」的诚实性要求，不改 goal 但要用户知情。
4. local 模式默认 lexical，ADR 0015 §2 的「follows embedding mode」可读作此意，建议在 ADR 里点明。

## Chunks 追加（L1，2026-10-15 真实运行之后）

### Chunk 7 — 词法查询受 BM25 词数上限约束
- change: 新颖性检索的查询在发给检索前裁到索引允许的不同词数以内（优先标题词，再按摘要顺序补足），不再因长摘要整体失败；端口适度多取以抵消被过滤的命中。
- kind: bug fix
- first signal: 用真实 BM25 生成索引（非假端口）+ 真实长度摘要（>64 个不同词）的测试，当前抛 `query has more than 64 unique terms`。
- checks: 新测试 Green；`library-prior-work`、`pipeline-library-novelty`、`lexical-search` 回归；typecheck。
- [x] accepted on observed evidence

### Chunk 8 — 非 arXiv 库论文可作对比基准，日报与 Dashboard 显示标题
- change: 按 `../research.md` §1.2–1.3 分两步：8a 让知识库中有标题摘要的 `file:sha256:` 文档成为合法对比基准（端口从知识库记录取标题摘要并排除与日报论文同题者；DTO、标记、索引、Markdown 行接受该形态，file: 无链接）；8b 标记 v2 携带至多 5 个显示标题（v1 继续可读），索引与 Dashboard、CLI 工作台显示标题、缺失时回退 key。两步一起交付。旧版插件读新标记时当天不显示新颖性（功能未发布，owner 接受）。
- kind: behavior
- first signal: 见 research.md §1.3 列出的失败用例与命令（8a：personalized-novelty / library-prior-work 新用例；8b：daily-summary-assembler 标记往返、v1 兼容与尺寸上界）。
- checks: research.md §1.3 所列回归；typecheck；boundaries；Dashboard 真实 Obsidian 检查或记未做。
- [x] accepted on observed evidence

### Chunk 9 — 失败日志与跳过原因如实
- change: 系统性检索失败只警告一次并停止逐篇重试；摘要缺失等输入问题记录真实原因，而不是 not comparable input。
- kind: behavior
- first signal: `pipeline-library-novelty.test.ts` 新用例：20 篇同类检索失败只产生 1 条警告；缺摘要的论文原因准确。
- checks: 同文件 Green；typecheck。
- [x] accepted on observed evidence

### Chunk 10 — 本地模式检索方式按实测确定
- change: 在 test_library 上对同一批日报论文比较 lexical（截词）与本地 hybrid 的前 5 相关度与耗时（不调业务模型），据此决定本地嵌入模式下用哪种；结果记入 journal，必要时更新 ADR 0015 §2 的说明。
- kind: non-behavioral（实测）+ 可能的一行接线改动
- first signal: n/a（实测数据）
- checks: 实测表格；若改接线，插件与 CLI 接线测试同步更新。
- [x] accepted on observed evidence

## Phase verification 修订（L1）
- 「同一天重跑不再调用新颖性模型」改为：检查点复用由单测证明（日报提交后检查点按设计清理，完整重跑本就会重新生成）。
- 修复后重新安装测试库，并用真实模型重跑一次质量评估：统计出说明的比例，逐条判断对比论文相关性与说明是否忠于摘要。

### Chunk 11 — 词法检索单次耗时
- change: 实测词法检索每次约 3.8 秒，约 70% 耗在纯 JS `sha256Hex`（research.md §2.4），20 篇日报约 76 秒宿主 CPU，Dashboard「找相似论文」同样受影响。在不改变检索结果的前提下降低单次耗时（如缓存或改用更快的同值计算）。
- kind: optimization
- first signal: 正确性基线（generation-bm25-index、lexical-search、library-prior-work-real-index 测试 Green）+ 可重复的单次检索耗时测量（同一库、同一查询集）。
- checks: 基线保持 Green；前后同一测量对比。
- [x] accepted on observed evidence

## Chunks 追加（L1，真实质量评估 2 之后）

### Chunk 12 — 提示收紧与说明语言
- change: 提示要求对比论文研究同一具体问题或场景；只是同领域、宽泛综述或大型目录类论文不算可比，此时应拒答；说明使用日报摘要语言（与日报其余文字一致）；提示合同版本递增。
- kind: behavior
- first signal: personalized-novelty 测试：提示含新约束；中文日报设置下请求携带目标语言、说明语言字段生效（当前 Red）。
- checks: personalized-novelty、pipeline-library-novelty、daily-summary-assembler 回归；typecheck。
- [x] accepted on observed evidence

### Chunk 13 — 新颖性阶段在流水线内的耗时
- change: 真实运行中该阶段 121.7 秒，而模型合计约 17 秒、独立检索约 1 秒/次；剖析流水线内检索的额外开销（如每次查找重读清单/目录、重开索引）并消除，结果不变。
- kind: optimization
- first signal: 可重复测量（同库、同 20 篇查询，经端口而非裸检索）+ 端口相关测试基线 Green。
- checks: 前后同一测量；library-prior-work、library-prior-work-real-index、插件与 CLI 接线测试保持 Green。
- outcome（L1）: 前提不成立，无代码改动。真实端口逐篇检索中位约 0.54 秒（20 篇约 11 秒），清单/目录加载与多取候选均可忽略；122 秒中约 108 秒是 20 次串行模型调用（流式+思考，约 5.4 秒/次，先前的 17 秒只计到响应头）。项目既有约定为每日 LLM 调用串行（2026-07-22-sequential-daily-summaries），故不引入并发。
- [x] accepted on observed evidence

### Chunk 14 — 逐篇结果日志与确定性
- change: 每篇记录一行结果（写出 / 无可比 / 跳过原因 / 失败），便于核对；若 LLM 客户端支持，新颖性调用使用确定性采样设置。
- kind: behavior
- first signal: pipeline-library-novelty 新用例：noComparablePrior 与写出各有一行结果日志（当前 Red）。
- checks: 同文件 Green；typecheck。
- [x] accepted on observed evidence

### Chunk 15 — 提示校准（收得过紧）
- change: Chunk 12 的提示在 20 篇复评中 20/20 拒答（误导 2→0，但有用 5→0，至少 1 条明确误拒）。在同一 20 篇（调参集）上迭代至多 3 个提示变体，再在同日未入选论文中取 20 篇作留出集验证；目标：误导为 0 或 1，有用条数恢复，留出集无明显误导。同一具体问题换数据或方法算可比；新论文明确使用、交叉匹配或检验的目录/数据发布算可比；综述与仅同领域/同关键词仍不算。提示合同版本递增，测试同步。
- kind: behavior（提示校准，以真实模型评测为证据）
- first signal: 调参集基线（Chunk 12 提示）：写出 0 / 有用 0 / 误导 0。
- checks: 调参集与留出集评测表；personalized-novelty、pipeline-library-novelty 测试；typecheck；acceptance 工具测试（fixture 按提示首句路由）。真实模型调用预算 ≤100 次。
- [x] accepted on observed evidence
