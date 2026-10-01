# P4 — first-scan-proposes-topics

goal_ref: ../goal.md
created: 2026-09-04T21:42:57+08:00
updated: 2026-09-04T22:10:00+08:00
revision: 2

## Outcome

首次扫库产出一整套**主题（带建议名）+ 每个主题下一列一行方向**的提议；复审页展示这套结构、可编辑可勾选；接受后直接写进 `settings.topics`，全程不必手写一个主题名。画像文档驱动的那条旧路径在本阶段物理删除，日报的方向命中此后只有 P3 那一个来源。

## Assumptions

- **失去主语的那批代码在本阶段物理删除**（2026-09-04 用户决定）。范围见同日 journal：profile schema、`personal-library-interest-profile-review.ts`、profile store 那一半、`evaluatePersonalLibraryInterestEligibility`、`buildPersonalizedDailyDiscoverySnapshot`、`personalized-paper-filter.ts`、`personal-library` 日报分节与 `discoveryProvenance` 的**生产端**。约 3000 行源码 + 约 4000 行测试。**P2 对 sidecar「留着不接线」的处理不构成先例**——那次不删是因为有用户可见设置，这里没有。
- **退休范围的大部分不是选择。** 接受动作一旦改道写 `settings.topics`，`interest-profile.json` 就没有写入者，其下游（资格判定 → 日报快照 → 第二分类器 → provenance → novelty，以及 incremental 的三个 apply）同时失去输入。本阶段不是在「选择退休谁」，是在处理改道的必然后果。
- **personal novelty 休眠并记账，不删也不改基准**（2026-09-04 用户决定）。`personalized-novelty.ts`（1388 行 + 1200 行测试）在输入断供后该 stage 永不触发。删它等于替用户决定 goal Non-goals 第 4 条明令不在本 goal 内决定的事。**这是本 goal 结束时必须交还给用户的显式欠债**，写进 Open questions 而不是留白。
- **方向的一行文本由提取提示词直接产出**（2026-09-04 用户决定）。不做「name + description + cues 拼成一行」的确定性折叠——拼出来的一行会直接变成用户设置页里的正式文本，且提议预览与最终文本对不上。cues 仍然产出，但**只供复审页帮用户判断，不进 settings**。
- **细簇的篇数上限抬起，代码单位上限当真守卫**（2026-09-04 用户决定）。见下面「两处已知缺陷」的第二条。
- **incremental 建议界面在本阶段之后到 P5 之前是暗的。** 它的建议挂靠对象是 profile 里的已确认方向，profile 空了就没有可挂靠的东西。`incremental/apply.ts` 的三个 `apply*Suggestion` 都返回 `PersonalLibraryInterestProfile`，写入目标要换成 topics，属 P5（ADR 0014 §2/§3）。**这段暗期是预期而非缺陷**，但复审页要说出来，不能让用户以为功能坏了。
- **删除没有迁移负担。** `docs/releases/` 里 0.3.0–0.4.6 共 13 份发布说明**零次**提到 personal library；`apps/cli` 对 personalized / interest-profile / libraryProfile 的引用数为 0。goal Constraints 里「插件与 CLI 两侧行为一致」在此不构成约束——CLI 从来没有这条路径。
- **README 那句话现在是假的，删除会让它重新成真。** README:87 写着 “The current preview does not change daily filtering, reports, paper notes, or email delivery”，但 `buildPipeline()`（`plugin/main.ts:3083`）每次日报都装配快照，没有任何 feature flag。删除第二分类器不是撤回承诺。
- **复审页必须桌面验收。** goal Constraints 点名复审页与 Dashboard 从未进过桌面验收，**测试绿不构成交付证据**。本阶段正面重写复审页，这条约束必然生效。
- **`normalizeTopic` 是所有主题的唯一入口，这一点不动。** P1 钉死的不变量。本阶段在它**之上**加一层唯一 tag 派生，不改它的语义。

## 两处已知缺陷（写计划时读代码翻出，本阶段一并处理）

- **聚类提议器的生成契约在说谎。** `createPersonalLibraryClusteredDirectionGenerationContract`（`personal-library-direction-proposer.ts:733`）记 `synthesisPrompt: "none"`、`strategy: "...-no-synthesis"`、`synthesis: "none-cluster-boundaries-are-theme-boundaries"`，模块头注释也写着 “skip the cross-cluster synthesis stage”。**但代码实际跑了综合阶段**（同文件 :946 的 `callValidatedStage("synthesis", ...)`，以及 `phase:"synthesis"` 的进度上报）。契约的自述职责是「让生成可复现、参数漂移可从提案里检出」，而整整一个 LLM 阶段连同其提示词版本对该指纹完全不可见——改综合提示词不会让任何旧提案失效。形状上是当初真的跳过综合、后来按 ADR 0009 §2 加了回来，契约与头注释没跟上。**Chunk 1 修**。
- **提议器在冻结语料上会当场抛错，而这从未被观察到。** `renderClusteredExtractionMessage`（同文件 :1047）对超过 `PERSONAL_LIBRARY_DIRECTION_MAX_PAPERS_PER_BATCH = 20` 篇的簇硬抛 `evidence-too-large`；P2 实测冻结语料按产品方式去重后**最大簇 21 篇**。P2 的测量是临时脚本直接调 `clusterPaperVectors`，提议器本身从未在这个库上跑过（P2 journal 原话：方向合成从未在此库上跑过），所以这条撞线一直没露面。**粗聚类只会让簇更大，本阶段第一次真跑必然撞上。Chunk 2 修**：真正的约束是消息体积（`MAX_BATCH_CODE_UNITS = 60_000`），20 篇只是它的代理；21 篇的标题+摘要远不到 60k。抬篇数上限、保留体积上限作真守卫，**不引入新阈值**——超体积仍然抛。

## Approach

**删除排在最前面。** revision 1 把删除放在最后，那是错的：提案文档变两级会立刻打断它的消费方（复审逻辑、复审页、store），而那些正是要删的东西——先改后删等于为将死的代码做一次移植。所以顺序是：先物理删除画像文档那条路径，把回归的重心一次性钉在「磁盘上的旧日报解析逐字不变」上；此后每个 chunk 面对的代码面都更小。

删完再做提议侧：提案文档改成两级结构、提取契约改成直接产出一行方向、顺手让生成契约说真话（一次成形，避免二次抬 schema 版本号）；再把聚类改成粗/细两级并解掉篇数上限那条假约束；然后给每个粗簇一个建议名，提议这才是完整的一套主题。此时停下来在冻结语料上实测粗聚类比例，把结果交给用户判——**这是 goal Constraints 明令要实测依据的两个阈值之一**。最后做写入侧（唯一 tag 派生与设置写入守卫）与复审页重写。

## Chunks

### Chunk 1 — 物理删除画像文档那条路径

- change kind: deletion
- strategy: Red 在「删掉之后什么都不该坏」上
- Red / baseline signal: **本 chunk 的红是回归红。** 删除范围见 Assumptions 第一条。核心断言是 **(a) 磁盘上已有的旧日报解析行为逐字不变**——`paper-index` 的 `discoveryProvenanceByReport`、`dashboard/history-sync`、`discovery-provenance-marker` 的**解析端**必须继续读得回 `personal-library` 分节与 v1 provenance 标记；(b) 日报**不再产出**这两样；(c) 一次完整日报运行在没有画像文档的情况下与 P3 交付的行为逐字相同。
- Green check: 四包全量
- regression checks: `npm run test --workspace @arxiv-daily/core`；`npm run test --workspace obsidian-arxiv-daily`；`npm run test --workspace arxiv-daily`；`npm run typecheck`；`npm run lint`；`npm run check:boundaries`
- **为什么排在最前**：Chunk 2 把提案文档改成两级，会立刻打断它的消费方（复审逻辑、复审页、store），而那些正是本 chunk 要删的。先改后删等于为将死的代码做一次移植。
- **停产与解析必须分开对待。** 生产端删干净，解析端一行不动——用户磁盘上已经有按旧格式写的日报，Dashboard 还要读它们。这是删除动作的第一条 abort trigger。
- **`personalized-novelty.ts` 不在删除范围内**（用户决定），但它的输入在本 chunk 断供，该 stage 此后永不触发。要断言：novelty 相关的既有测试仍全绿（它们直接构造输入，不经过 profile），且日报里不再出现 novelty 段落。
- **`incremental/` 的打分半边不删**（placement / recluster / diff-suggestions / suggestions-store 约 1400 行，P5 只换写入目标），`apply.ts` 的三个 `apply*Suggestion` 随 profile 类型一起删，P5 重建。
- **复审页在本 chunk 只做减法**：删掉「逐条确认方向」那一半与 incremental 建议区，保留提案展示。整体重写留到 Chunk 7，那时才需要桌面验收。
- 删完之后 README:87 那句「preview 不改变日报筛选」重新成真，**不需要改 README**。
- exception: 删除类 chunk，变异检验的形式是「把删掉的调用点恢复一处，对应的『不再产出』断言必须红」
- [ ] implementation and tests accepted

### Chunk 2 — 提案文档变两级，提取直接产出一行方向，生成契约说真话

- change kind: behavior change（提案 schema + 提取提示词契约 + 生成契约）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `packages/core/tests/personal-library-direction-proposer.test.ts` 与 `clustered-direction-proposer.test.ts` 新增——(a) 提案根从 `candidates: [...]` 变成 `topics: [{suggestedName, directions: [...]}]`，`PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION` 3→4，旧形状被 `decodePersonalLibraryDirectionProposal` 严格拒绝；(b) 提取阶段的候选 DTO 从 `{name, description, discoveryCues, representativePaperKeys}` 变成 `{text, discoveryCues, representativePaperKeys}`，`text` 是一行；(c) **生成契约里 `synthesisPrompt` 记的是真实的综合提示词版本常量**，退回成 `"none"` 时对应断言红。
- Green check: `npm run test --workspace @arxiv-daily/core -- direction-proposer`
- regression checks: core 全量；`npm run typecheck`
- **一行的长度上限沿用既有的 `PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH`（1000），只作 DTO 硬上限防膨胀，「一行」由提示词表达。** 不新造一个「一行有多长」的常数——goal Constraints 已经把两个阈值列为必须实测，再添一个拍脑袋的数是同一个错误的第三次。
- **`origin: "library"` 在本阶段第一次有生产者。** P1 把它加进 `DirectionOrigin` 枚举时没有任何写入方（`grep 'origin: "library"'` 至今零命中），所以「库来源的方向」这条路是本阶段接通的，不是既有行为的延续。
- **顺带删掉未聚类的那个提议器。** `proposePersonalLibraryDirections` 与整个 grouping 阶段（约 400 行 + `personal-library-direction-grouping.system.md` + 551 行测试）**没有任何生产调用方**——插件走的是 `proposeClusteredPersonalLibraryDirections`（`plugin/main.ts:959`），前者只被自己的测试引用。它与聚类提议器共用本 chunk 要改的提取契约，留着就得把死代码移植到新契约上。适用的是 2026-09-04 已定的同一条原则：从未发布 + 无调用方 → 删。
- **L1：与 Chunk 3、Chunk 4 合并为一次提交。** 三者是同一次提议契约变更的三面——两级 schema 没有两级聚类填不满，没有建议名也构不成一个主题。分开提交会留下两次自知的红。各自的变异检验仍分开做。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 3 — 粗/细两级聚类；篇数上限让位给体积上限

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: (a) 提议器对同一批向量跑两遍 `clusterPaperVectors`——粗（低 `relativeStopRatio`）产出主题，对每个粗簇的成员向量再跑细（当前 0.65）产出该主题的方向；断言细簇是粗簇的真子集、且粗簇成员被细簇与细层离群完整覆盖，不丢论文。(b) 一个 21 篇的簇不再抛 `evidence-too-large`，而是正常进入一次提取；一个体积超 `MAX_BATCH_CODE_UNITS` 的簇**仍然抛**。红在今天 `renderClusteredExtractionMessage` 见到 21 篇就抛。
- Green check: `npm run test --workspace @arxiv-daily/core -- clustered-direction-proposer`
- regression checks: core 全量；`npm run check:boundaries`
- **不需要新机器。** ADR 0014 Context 末段已经指明：同一批向量跑两遍，靠 `relativeStopRatio` 区分（`packages/core/src/library/clustering/clusterer.ts:82`，proposer 现用 0.65 见 `personal-library-direction-proposer.ts:714` 的 `resolvePersonalLibraryClusteringOptions`）。低 ratio → 继续合并更弱的边 → 更少更粗的簇。
- **粗聚类比例在本 chunk 内不给默认值，作为必填参数由调用方传入。** goal Constraints：没有实测依据之前它只能算调参旋钮。Chunk 4 才决定默认值。
- **`PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS = 512`（schema 上限）与 20 篇（提取上限）本来就互相矛盾**——schema 允许 512 成员的簇，提取见到 21 就抛。本 chunk 消掉这处矛盾，两个上限的关系要在注释里说清楚。
- 生成契约里的 `clustering` 字段要同时记粗与细两组参数——它们共同决定了每个候选的主题范围，参数漂移必须能从提案里检出。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 4 — 主题的建议名：每个粗簇一次命名调用

- change kind: behavior change（新增提示词契约）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 每个粗簇跑一次命名调用，输入是该主题下已产出的方向一行文本，输出一个短名（沿用 `PERSONAL_LIBRARY_MAX_NAME_LENGTH = 120`）。断言——(a) 提案里每个主题带 `suggestedName`；(b) 命名调用的输入**只有本主题的方向文本**，不含论文摘要、路径、指纹；(c) 命名失败时**退化为一个可辨认的占位名而不是让整份提议失败**（沿用「一次提取已经花掉了，碎片化的提议好过没有提议」的既有取舍，见 `personal-library-direction-proposer.ts:935` 的注释）。
- Green check: `npm run test --workspace @arxiv-daily/core -- clustered-direction-proposer`
- regression checks: core 全量
- ADR 0014 §1 原文是 “A proposed topic carries a suggested name”——**建议**，所以机器给、用户可改。goal 的成功标准是「全程不必手写一个主题名」，不是「不能改名」。
- 新提示词的版本常量要进生成契约（与 Chunk 1 修的那处同理）。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 5 — 粗聚类比例实测：在冻结语料上扫一遍，交给用户判

- change kind: measurement（不改源码）
- strategy: 测量，无 Red-Green
- 做法沿用 P2 的先例：临时脚本在冻结语料（207 篇去重后）上按若干个粗 ratio 各跑一遍，**每个 ratio 产出一份「提议出的主题名 + 各主题下的方向一行 + 各主题包含哪些论文标题」**，写到 `~/Desktop/plugin_test/` 下，脚本跑完即删。
- **判据是「提议出的主题是否说得出名字」，不是「离群率低不低」**（2026-09-03 用户决定：测试库里混着的无关 PDF 落进离群是可接受的现实，不清库、不为此调聚类参数）。P2 实测在 0.65 下 130 篇（61%）落进离群，这个量级在粗层会显著下降，但不构成判据。
- **判断权在用户。** 本 chunk 交付的是材料，不是结论。用户判完之后粗 ratio 才有默认值；判不出来就继续作为调参旋钮交付，并在 goal Open questions 里如实标明。
- **必须绕开 store 但不能绕开标识**（P2 栽过一次）：临时脚本可以不走 store，但论文标识必须用产品的内容寻址键，否则簇数虚报。
- **提议器从未在这个库上跑过**，本 chunk 是它的第一次真实运行——预期会顺带暴露 Chunk 3 那类「上限撞线」之外的问题。
- Green check: 无（测量）
- regression checks: 无源码改动
- exception: 非行为 chunk，无变异检验
- [ ] measurement done and judged by user

### Chunk 6 — 接受即写进 `settings.topics`：唯一 tag 的派生与守卫

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增一个「接受提议 → 主题列表」的纯函数，断言——(a) 每个被接受的主题产出 `{name, tag, detail, directions}`，方向的 `origin` 全为 `"library"`；(b) **tag 由建议名派生并保证唯一**，与已有主题重名时按确定性规则加后缀；(c) 接受结果喂给 `validateFilterConfig` **必须无抱怨**。红在今天 `normalizeTopic`（`packages/core/src/settings/topics.ts:68`）既不从 name 派生 tag、也不管唯一性，而重复 tag 在 `packages/core/src/settings/validation.ts:159` 是判错的——**照 ADR 0014 §1 直接写入会产出一份自己校验不过的设置**。
- Green check: `npm run test --workspace @arxiv-daily/core -- topics`；`-- validation`
- regression checks: core 全量；plugin 全量；`npm run typecheck`
- **这是 ADR 0014 Consequences 最后一条「凡是守设置写入的机制都要覆盖它」的具体落点。** 接受不再写一份自己的文档，而是写产品设置；设置侧所有既有守卫（规范化、校验、影子字段不变量）都要在这条新路径上成立。
- **派生与唯一化写在 `normalizeTopic` 之上，不改它。** P1 钉死「所有产生主题的路径都必须经过 `normalizeTopic`」，本 chunk 是新增一个经过它的调用方，不是给它加职责。变异检验：绕开 `normalizeTopic` 直接构造主题时，影子字段那条不变量的断言必须红。
- **接受粒度**：整套结构可见，逐主题、逐方向可勾选与编辑，一次写入。ADR 0014 §1「accepts a structure, not a pile of unattached lines」与 ADR 0009 §3「证据单薄的候选被标记且在整组接受时默认不选中」两条同时成立。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 7 — 复审页重写为「接受一整套主题」

- change kind: behavior change（用户界面）
- strategy: strict Red-Green-Refactor + **桌面验收**
- Red / baseline signal: `plugin/tests/personal-library-interest-profile-modal.test.ts` 重写——(a) 渲染的是两级结构：主题名（可改）下挂一列方向一行（可改、可勾选）；(b) 接受按钮调用 Chunk 5 的写入路径，而不是任何 profile 确认；(c) **离群篇数可见**：「有 N 篇论文未能归入任何主题」，但不为它们提议主题；(d) incremental 建议区在本阶段显式说明暂不可用，不是空白。
- Green check: `npm run test --workspace obsidian-arxiv-daily -- interest-profile-modal`
- regression checks: plugin 全量；`npm run test:desktop`
- **桌面验收是本 chunk 的交付条件，不是可选项。** goal Constraints 点名复审页从未进过桌面验收、单测里 Dashboard 视图也不实例化，**测试绿不构成交付证据**。要由用户在真实 Obsidian 里看过：提议渲染得出来、名字改得动、勾选生效、接受之后设置页里真的多出主题。
- **离群可见不静默**是本 goal 的一贯原则（P2 对近重复合并的处理同理）。P2 实测 0.65 下 61% 落进离群，粗层会低很多，但用户有权知道有多少论文没被这套提议覆盖。
- exception: 无
- [ ] implementation and tests accepted
- [ ] desktop acceptance by user

## Phase verification

- 四包全量测试、`npm run typecheck`、`npm run lint`（基线 0 error / 20 warning）、`npm run check:boundaries` 全过。P3 的基线是 core 2101 / plugin 719 / CLI 73；本阶段删除约 4000 行测试，**core 与 plugin 的用例数会显著下降，这是预期**——要在 journal 里记下删了多少、新增多少，不能只报一个净值让人以为没动。
- **真实端点跑一次完整提议**：冻结语料 + 真 Ollama embedding + 真 LLM，走完粗聚类 → 细聚类 → 提取 → 综合 → 命名 → 提案落盘 → 复审页渲染。**提议器从未在这个库上跑过**，单测里的 LLM 是桩，桩通过不构成真模型能通过的证据（P3 的同一条教训）。
- **桌面验收**：复审页（Chunk 6）。goal Constraints 点名，不可用测试绿替代。
- **旧日报解析回归**：拿 P3 之前产出的真实日报（含 `personal-library` 分节与 v1 provenance 标记）过一遍 `paper-index` 与 `history-sync`，结果逐字不变。
- 每个行为 chunk 至少一次变异检验：拿掉被测那一行，对应断言必须红在**它自己**身上（本 goal 前三个阶段共栽过四次「绿是别的原因造成的」）。
- **一次端到端日报不在本阶段**，那是 P6。

## Abort / reshape triggers

- 若粗聚类在任何 ratio 下都产不出「说得出名字」的主题——退化为一个大簇或每篇一簇（P2 实测旧全文索引就是前者，96% 挤在一个簇）→ **停下**。粗/细两级这条路可能不成立，属 L2：改用别的方式产出主题层（例如让模型直接从细簇的方向列表里归组）。
- 若删除旧路径牵动的兼容改动超出「读旧日报」这一层，比如要动 `paper-index` 的 schema 或 `history-sync` 的迁移 → **停下拆阶段**，不要在 P4 里顺手改派生索引（P3 有同形的 trigger）。
- 若接受写入必须改 `normalizeTopic` 的语义、而不只是在它之上加一层 → **停下确认**。P1 钉死它是所有主题的唯一入口，改它动摇的是 P1 的验收标准。
- 若复审页桌面验收发现「接受一整套结构」这个交互本身不成立（比如提议出的主题太多太杂，用户根本无法逐条判断）→ **L2**，回到 ADR 0014 §1 重新想提议的呈现粒度。
- 若实测发现粗聚类比例存在但极其敏感（相邻 ratio 产出完全不同的主题集）→ 不要挑一个好看的交付。如实记录敏感性，作为调参旋钮交付，并在 goal Open questions 里标明。

## Open questions（P4 不顺手定掉）

- **personal novelty 的新对比基准**——本阶段让它休眠，欠债显式记在此处。ADR 0012 line 43 明说 novelty 未必要走，基准可以改成检索索引里最相似的库内论文；goal Non-goals 第 4 条禁止在本 goal 内顺手定。**本 goal 结束时必须交还给用户。**
- 提议新主题的相似度下限（ADR 0014 §3）——P5 的活，同样需要实测依据。
- 详情选取（`detail-selector.ts:155`）是否也该看到全部方向而不是影子 `description`（P1/P3 两次推迟）。
- Dashboard 是否展示主题方向的命中来源（P3 只保证机器标记里有）。
- ADR 0014 §3 的理由段落假设了「主题名会被用来排除其他领域的论文」，而 2026-09-03 用户定「主题名仍只作标签」——**P5 若要引用那段理由，需注意该前提当前不成立**。
- `MAX_CLUSTERING_CHUNKS_PER_PAPER = 80` 与 `recluster.ts` 的 centroid 理由，在每篇只剩一到两块之后成死条款（P2 留下，本阶段动聚类但不动这两处）。
- `pdf-text-utils.ts` 的 `LEGACY_ARXIV_ID_IN_TEXT_RE` 少 `(?:v\d+)?`（P2 发现的真缺陷，按用户方向不修）。
- ADR 0008 的全文授权深度是否退休（ADR 0013 标记，需单独看那条 ADR）。
